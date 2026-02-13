import redis.asyncio as redis
import asyncio
import json
import time
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, WebSocket
from shared.config import settings

r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
app = FastAPI()

@dataclass
class WorkerState:
    pubkey: str
    ws: WebSocket
    specs: dict
    vram_total_mb: int
    actual_free_mb: int
    vram_used_mb: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    cached_layers: Set[str] = field(default_factory=set)

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    owner: str
    topology: List[dict]
    max_tokens: int = 50

class MoEScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        vram = specs.get('vram_mb', 24000)
        # Init actual_free to total until first heartbeat
        self.workers[wid] = WorkerState(pubkey=wid, ws=ws, specs=specs, vram_total_mb=vram, actual_free_mb=vram)
        print(f"✅ Registered: {wid} | VRAM: {vram}MB")
        return wid

    async def unregister_worker(self, wid: str):
        if wid in self.workers:
            del self.workers[wid]
            print(f"❌ Worker {wid} disconnected")

    async def handle_heartbeat(self, wid: str, data: dict):
        if wid in self.workers:
            self.workers[wid].actual_free_mb = data.get('vram_free_mb', 0)
            self.workers[wid].last_heartbeat = time.time()

    def _find_best_worker(self, size_mb: int, current_job_allocations: Dict[str, int],
                          previous_worker_id: str = None, cache_key: str = None) -> Optional[WorkerState]:
        candidates = list(self.workers.values())
        if not candidates: return None

        valid = []
        for w in candidates:
            # Check if cached
            is_cached = cache_key in w.cached_layers

            # Ledger Logic:
            # Available = (Reported Free) - (Allocated in CURRENT planning session)
            pending_load = current_job_allocations.get(w.pubkey, 0)

            # Cost is 0 if already cached (assuming cache persists)
            cost = 0 if is_cached else size_mb

            # --- FIX START: Use Internal Ledger + Safety Margin ---
            # Don't trust actual_free_mb (laggy race condition). Use internal vram_used_mb.
            # 90% Cap to allow for PyTorch overhead/activations
            safe_limit = w.vram_total_mb * 0.90
            projected_usage = w.vram_used_mb + pending_load

            if (projected_usage + cost) <= safe_limit:
                valid.append((w, cost))
            # --- FIX END ---

        if not valid: return None

        def score(item):
            w, cost = item
            s = 0
            if cost == 0: s += 10000 # Cached

            # USE ACTUAL AVAILABLE after pending allocations
            pending = current_job_allocations.get(w.pubkey, 0)

            # --- FIX START: Score based on internal ledger ---
            safe_limit = w.vram_total_mb * 0.90
            projected_usage = w.vram_used_mb + pending
            available = (safe_limit - projected_usage) / 1024
            # --- FIX END ---

            # Locality as multiplier not flat bonus
            if previous_worker_id and w.pubkey == previous_worker_id:
                available *= 1.2

            s += available
            return s

        chosen_worker, cost = max(valid, key=score)

        # Update ledger
        current_job_allocations[chosen_worker.pubkey] = current_job_allocations.get(chosen_worker.pubkey, 0) + cost

        return chosen_worker

    def _plan_job(self, model_info) -> Optional[List[dict]]:
        layers = model_info.get('layer_metadata', [])
        if not self.workers: return None

        topology = []
        prev_worker_id = None

        # Ephemeral ledger for this specific plan
        current_job_allocations = {}

        for layer in layers:
            layer_idx = layer['layer_idx']
            l_type = layer.get('type', 'dense')

            if l_type == 'dense':
                size = layer.get('size_mb', 0)
                cache_key = f"{model_info['model_id']}:{layer_idx}:main"

                target_worker = self._find_best_worker(size, current_job_allocations, prev_worker_id, cache_key)
                if not target_worker:
                    print(f"⚠️ Cluster Full! Cannot place layer {layer_idx}")
                    return None

                if cache_key not in target_worker.cached_layers:
                    target_worker.vram_used_mb += size
                    target_worker.cached_layers.add(cache_key)

                node = {
                    "layer_idx": layer_idx,
                    "type": l_type,
                    "worker_id": target_worker.pubkey,
                    "endpoint": target_worker.specs.get('p2p_url')
                }
                prev_worker_id = target_worker.pubkey
                topology.append(node)

            elif l_type == 'moe':
                # 1. Place Router + Shared
                shared_size = layer.get('shared_size_mb', 0)
                router_size = layer.get('router_size_mb', 0)
                combined_size = shared_size + router_size

                shared_key = f"{model_info['model_id']}:{layer_idx}:shared"

                router_worker = self._find_best_worker(combined_size, current_job_allocations, prev_worker_id, shared_key)
                if not router_worker:
                    print(f"⚠️ Network at capacity! Cannot place MoE shared+router for layer {layer_idx} (needs {combined_size:.1f}MB)")
                    return None

                if shared_key not in router_worker.cached_layers:
                     router_worker.vram_used_mb += combined_size
                     router_worker.cached_layers.add(shared_key)

                node = {
                    "layer_idx": layer_idx, "type": "moe",
                    "worker_id": router_worker.pubkey, "endpoint": router_worker.specs.get('p2p_url'),
                    "expert_map": {}
                }
                prev_worker_id = router_worker.pubkey

                # 2. Place Experts
                num_experts = layer.get('num_experts', 0)
                expert_size = layer.get('expert_size_mb', 0)

                for exp_idx in range(num_experts):
                    exp_key = f"{model_info['model_id']}:{layer_idx}:expert:{exp_idx}"
                    exp_worker = self._find_best_worker(expert_size, current_job_allocations, node['worker_id'], exp_key)

                    if not exp_worker:
                        print(f"⚠️ Network at capacity! Cannot place expert {exp_idx} for layer {layer_idx} (needs {expert_size:.1f}MB)")
                        return None

                    if exp_key not in exp_worker.cached_layers:
                        exp_worker.vram_used_mb += expert_size
                        exp_worker.cached_layers.add(exp_key)

                    node['expert_map'][str(exp_idx)] = exp_worker.specs.get('p2p_url')

                topology.append(node)

        if not topology:
            return None

        # Bookend the topology with explicit embed and decode nodes.
        first_worker = topology[0]
        last_worker = topology[-1]

        embed_node = {
            "layer_idx": -1,
            "type": "dense",
            "role": "embed",
            "worker_id": first_worker['worker_id'],
            "endpoint": first_worker['endpoint']
        }

        decode_node = {
            "layer_idx": -1,
            "type": "dense",
            "role": "decode",
            "worker_id": last_worker['worker_id'],
            "endpoint": last_worker['endpoint']
        }

        topology = [embed_node] + topology + [decode_node]
        return topology

    async def _check_model_status(self, model_id: str) -> str:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{self.registry_url}/models/status", params={"model_id": model_id}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status", "idle")
        except: pass
        return "error"

    async def process_queue(self):
        print("🚀 [Scheduler] Intelligent Dispatcher Active")
        while True:
            try:
                item = await r.blpop("job_queue", timeout=2)
                if not item: continue

                raw_job = json.loads(item[1])
                job_id = raw_job['id']
                model_id = raw_job['model_id']

                # 1. Check Status
                status = await self._check_model_status(model_id)

                if status == 'ready':
                    # Proceed to plan
                    pass
                elif status == 'processing':
                    print(f"⏳ JIT: Model {model_id} is processing. Re-queueing...")
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue
                elif status == 'failed':
                    print(f"❌ JIT: Model {model_id} failed to shard. Dropping job.")
                    continue
                else: # idle or unknown
                    print(f"⬇️ JIT: Triggering download for {model_id}")
                    try:
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(f"{self.registry_url}/models/shard", json={"model_id": model_id})
                    except: pass
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue

                # 2. Plan
                model_info = None
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.get(f"{self.registry_url}/models/info", params={"model_id": model_id}) as resp:
                            if resp.status == 200: model_info = await resp.json()
                except: pass

                if not model_info:
                    print("⚠️ No model info. Re-queueing.")
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(2)
                    continue

                plan = self._plan_job(model_info)

                if not plan:
                    print("⚠️ Planning failed (Insufficient Resources). Re-queueing...")
                    await asyncio.sleep(5)
                    await r.rpush("job_queue", item[1])
                    continue

                job = JobState(raw_job['id'], model_id, raw_job['input_prompt'], raw_job['owner'], plan, max_tokens=raw_job.get('tokens', 50))
                self.active_jobs[job_id] = job
                await self._dispatch(job)

            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _dispatch(self, job: JobState):
        print(f"📦 Dispatching Job {job.id} (Smart Route)")

        # Calculate the feedback loop endpoint (where the first worker lives)
        first_node_endpoint = job.topology[0]['endpoint']

        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w:
                print(f"⚠️ Worker {node['worker_id']} not found!")
                continue

            is_last_node = (i == len(job.topology) - 1)
            next_hop = job.topology[i + 1]['endpoint'] + "/tensor_in" if not is_last_node else None

            # NEW: Include target layer index for next hop
            next_layer_idx = job.topology[i + 1]['layer_idx'] if not is_last_node else None

            role = node.get('role')

            print(f"  Layer {i}: type={node['type']}, worker={w.pubkey[:8]}, next_hop={next_hop}")

            payload = {
                "job_id": job.id,
                "model_id": job.model_id,
                "layer_idx": node['layer_idx'],
                "next_hop": next_hop,
                "next_layer_idx": next_layer_idx  # NEW: Add target layer index
            }

            if node['type'] == 'dense':
                payload.update({
                    "type": "EXECUTE_DENSE",
                    "input": job.input_prompt if role == 'embed' else None,
                    "is_first": role == 'embed',
                    "is_last": role == 'decode',
                    "max_tokens": job.max_tokens, # Pass max tokens for generation control
                    "first_node_endpoint": first_node_endpoint if role == 'decode' else None # Pass loop target
                })
                await w.ws.send_json(payload)

            elif node['type'] == 'moe':
                payload.update({
                    "type": "EXECUTE_ROUTER",
                    "expert_map": node['expert_map']
                })
                await w.ws.send_json(payload)

                # Group experts by worker so we batch the sends
                tasks = {}
                for exp_idx, url in node['expert_map'].items():
                    exp_w = next((wk for wk in self.workers.values() if wk.specs.get('p2p_url') == url), None)
                    if exp_w:
                        if exp_w.pubkey not in tasks: tasks[exp_w.pubkey] = []
                        tasks[exp_w.pubkey].append(int(exp_idx))

                for wid, indices in tasks.items():
                    t_w = self.workers.get(wid)
                    for idx in indices:
                        await t_w.ws.send_json({
                            "type": "EXECUTE_EXPERT",
                            "job_id": job.id,
                            "model_id": job.model_id,
                            "layer_idx": node['layer_idx'],
                            "expert_idx": idx,
                            "return_url": node['endpoint']
                        })


    async def handle_result(self, wid: str, data: dict):
        job_id = data.get('job_id')
        if data.get('status') == 'completed':
            print(f"🎉 Job {job_id} Finished")
            await r.setex(f"result:{job_id}", 3600, json.dumps(data))
            if job_id in self.active_jobs: del self.active_jobs[job_id]

scheduler = MoEScheduler(settings.REGISTRY_URL if hasattr(settings, 'REGISTRY_URL') else "http://localhost:8002")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        msg = await ws.receive_json()
        if msg.get('type') == "REGISTER":
            wid = await scheduler.register_worker(ws, msg['specs'])
            while True:
                data = await ws.receive_json()
                if data['type'] == "RESULT":
                    await scheduler.handle_result(wid, data)
                elif data['type'] == "HEARTBEAT":
                    await scheduler.handle_heartbeat(wid, data)
    except:
        if wid: await scheduler.unregister_worker(wid)
