# scheduler/main.py - FIXED VERSION

import asyncio
import json
import aiohttp
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import redis.asyncio as redis

app = FastAPI()

class InferenceJob(BaseModel):
    id: str
    model_id: str
    input_prompt: str
    user_pubkey: Optional[str] = None

class WorkerSpec(BaseModel):
    gpu: str
    vram_mb: int
    p2p_url: str

class Worker:
    def __init__(self, ws: WebSocket, pubkey: str, specs: dict):
        self.ws = ws
        self.pubkey = pubkey
        self.specs = specs
        self.busy_layers = set()

r = redis.from_url("redis://localhost:6379", decode_responses=True)

class MoEScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, Worker] = {}
        # ✅ CHANGED: Store full job state including topology and progress
        self.active_jobs: Dict[str, dict] = {}  # job_id -> {"job": Job, "topology": [...], "current_layer_idx": int}

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        import uuid
        wid = f"Worker_{uuid.uuid4().hex[:8]}"
        self.workers[wid] = Worker(ws, wid, specs)
        print(f"✅ Registered: {wid} | VRAM: {specs['vram_mb']}MB")
        return wid

    async def unregister_worker(self, wid: str):
        if wid in self.workers:
            del self.workers[wid]
            print(f"❌ Worker {wid} disconnected")

    async def process_queue(self):
        while True:
            try:
                job_data = await r.blpop("job_queue", timeout=5)
                if job_data:
                    job_json = json.loads(job_data[1])
                    job = InferenceJob(**job_json)

                    # Set initial status
                    await r.setex(f"result:{job.id}", 3600, json.dumps({
                        "status": "processing",
                        "progress": 0
                    }))

                    # Try to dispatch
                    try:
                        await self.dispatch_job(job)
                    except Exception as e:
                        print(f"⚠️ Planning failed ({e}). Re-queueing...")
                        await r.rpush("job_queue", job_data[1])
                        await asyncio.sleep(5)
            except Exception as e:
                print(f"Queue processing error: {e}")
                await asyncio.sleep(1)

    async def _fetch_model_info(self, model_id: str):
        """Fetch model metadata from registry"""
        sanitized = model_id.replace("/", "_")
        url = f"{self.registry_url}/models/info?model_id={model_id}"
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        raise ValueError(f"Model info not found for {model_id}")

    def plan_topology(self, model_id: str, model_info: dict):
        """Create execution topology (simplified for demo)"""
        if not self.workers:
            return None

        # For demo: assume model_info has 'layers' list with type info
        layers = model_info.get('layers', [])
        worker_list = list(self.workers.values())

        if len(worker_list) < 2:
            return None

        topology = []

        # Assign layers to workers in round-robin
        for i, layer in enumerate(layers):
            worker = worker_list[i % len(worker_list)]

            node = {
                "worker_id": worker.pubkey,
                "layer_idx": layer['index'],
                "type": layer['type'],
                "endpoint": f"{worker.specs['p2p_url']}/tensor_in",
                "role": layer.get('role'),
                "vram_estimate": layer.get('size_mb', 500)
            }

            if layer['type'] == 'moe':
                # Build expert map
                expert_map = {}
                num_experts = layer.get('num_experts', 8)
                for exp_idx in range(num_experts):
                    exp_worker = worker_list[exp_idx % len(worker_list)]
                    expert_map[str(exp_idx)] = exp_worker.specs['p2p_url']
                node['expert_map'] = expert_map

            topology.append(node)

        # Validate resources
        total_vram_needed = sum(n['vram_estimate'] for n in topology)
        total_vram_available = sum(w.specs['vram_mb'] for w in self.workers.values())

        if total_vram_available < total_vram_needed:
            print(f"⚠️ Insufficient VRAM: need {total_vram_needed}MB, have {total_vram_available}MB")
            return None

        return topology

    async def dispatch_job(self, job: InferenceJob):
        """
        ✅ FIXED: Dispatch only the FIRST layer initially.
        Subsequent layers are triggered by LAYER_COMPLETE messages.
        """
        model_info = await self._fetch_model_info(job.model_id)
        topology = self.plan_topology(job.model_id, model_info)

        if not topology:
            raise ValueError("Failed to create topology - insufficient resources")

        # Store job state
        self.active_jobs[job.id] = {
            "job": job,
            "topology": topology,
            "current_layer_idx": 0
        }

        print(f"🚀 Starting job {job.id} with {len(topology)} layers")

        # ✅ CRITICAL FIX: Only dispatch the FIRST layer
        await self._dispatch_layer(job.id, 0)

    async def _dispatch_layer(self, job_id: str, layer_idx: int):
        """
        ✅ NEW METHOD: Dispatch a single layer in the pipeline.
        This is called:
        1. Initially for layer 0 (embed)
        2. Subsequently when we receive LAYER_COMPLETE messages
        """
        job_state = self.active_jobs.get(job_id)
        if not job_state:
            print(f"⚠️ Job {job_id} not found in active jobs")
            return

        job = job_state["job"]
        topology = job_state["topology"]

        if layer_idx >= len(topology):
            print(f"⚠️ Layer index {layer_idx} exceeds topology length {len(topology)}")
            return

        node = topology[layer_idx]
        worker_id = node['worker_id']

        if worker_id not in self.workers:
            print(f"⚠️ Worker {worker_id} not connected")
            return

        w = self.workers[worker_id]

        # ✅ CRITICAL FIX: Always set next_hop to the next layer's endpoint
        if layer_idx + 1 < len(topology):
            next_hop = topology[layer_idx + 1]['endpoint']
        else:
            next_hop = None  # Last layer

        # Update progress
        progress = int((layer_idx / len(topology)) * 100)
        await r.setex(f"result:{job_id}", 3600, json.dumps({
            "status": "processing",
            "progress": progress,
            "current_layer": layer_idx
        }))

        print(f"📤 Dispatching layer {layer_idx}/{len(topology)-1} for job {job_id[:8]}")
        print(f"   Worker: {worker_id}, Type: {node['type']}, next_hop: {next_hop}")

        # Build and send payload
        payload = {
            "job_id": job_id,
            "model_id": job.model_id,
            "layer_idx": node['layer_idx'],
            "next_hop": next_hop  # ✅ CRITICAL: Always include this
        }

        if node['type'] == 'dense':
            role = node.get('role')
            payload.update({
                "type": "EXECUTE_DENSE",
                "input": job.input_prompt if role == 'embed' else None,
                "is_first": role == 'embed',
                "is_last": role == 'decode'
            })
            await w.ws.send_json(payload)

        elif node['type'] == 'moe':
            payload.update({
                "type": "EXECUTE_ROUTER",
                "expert_map": node['expert_map']
            })
            await w.ws.send_json(payload)

            # ✅ Send expert instructions (they'll wait for router to dispatch tensors)
            tasks = {}
            for exp_idx, url in node['expert_map'].items():
                exp_w = next((wk for wk in self.workers.values()
                             if wk.specs.get('p2p_url') == url), None)
                if exp_w:
                    if exp_w.pubkey not in tasks:
                        tasks[exp_w.pubkey] = []
                    tasks[exp_w.pubkey].append(int(exp_idx))

            for wid, indices in tasks.items():
                t_w = self.workers.get(wid)
                if t_w:
                    for idx in indices:
                        await t_w.ws.send_json({
                            "type": "EXECUTE_EXPERT",
                            "job_id": job_id,
                            "model_id": job.model_id,
                            "layer_idx": node['layer_idx'],
                            "expert_idx": idx,
                            "return_url": node['endpoint']
                        })

    async def handle_layer_complete(self, job_id: str, completed_layer_idx: int):
        """
        ✅ NEW METHOD: Handle notification that a layer has completed.
        Trigger dispatch of the next layer.
        """
        job_state = self.active_jobs.get(job_id)
        if not job_state:
            return

        topology = job_state["topology"]
        next_layer_idx = completed_layer_idx + 1

        print(f"✅ Layer {completed_layer_idx} complete for job {job_id[:8]}")

        if next_layer_idx < len(topology):
            # Update current layer index
            job_state["current_layer_idx"] = next_layer_idx

            # Dispatch next layer
            print(f"   Triggering layer {next_layer_idx}")
            await self._dispatch_layer(job_id, next_layer_idx)
        else:
            print(f"🎉 All layers complete for job {job_id[:8]}, waiting for final result")

    async def handle_result(self, wid: str, data: dict):
        """Handle final result from decode layer"""
        job_id = data.get('job_id')
        if data.get('status') == 'completed':
            print(f"🎉 Job {job_id} Finished")
            await r.setex(f"result:{job_id}", 3600, json.dumps(data))

            # Clean up job state
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

# Initialize scheduler
from shared.config import settings
scheduler = MoEScheduler(
    settings.REGISTRY_URL if hasattr(settings, 'REGISTRY_URL') else "http://localhost:8002"
)

@app.on_event("startup")
async def startup_event():
    print("🚀 [Scheduler] Intelligent Dispatcher Active")
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        msg = await ws.receive_json()
        if msg.get('type') == "REGISTER":
            wid = await scheduler.register_worker(ws, msg['specs'])

            # Message loop
            while True:
                data = await ws.receive_json()

                if data['type'] == "RESULT":
                    await scheduler.handle_result(wid, data)

                elif data['type'] == "LAYER_COMPLETE":
                    # ✅ NEW: Handle layer completion notification
                    job_id = data['job_id']
                    layer_idx = data['layer_idx']
                    await scheduler.handle_layer_complete(job_id, layer_idx)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if wid:
            await scheduler.unregister_worker(wid)
