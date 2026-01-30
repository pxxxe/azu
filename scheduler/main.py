import asyncio
import json
import redis.asyncio as redis
import aiohttp
import uuid
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from shared.config import settings
from shared.economics import calculate_worker_share
from shared.solana_lib import sign_payout

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

# --- DATA STRUCTURES ---

@dataclass
class WorkerState:
    pubkey: str
    ws: WebSocket
    specs: dict  # {gpu: str, vram_gb: float, p2p_url: str}
    status: str = "IDLE"  # IDLE, BUSY, OFFLINE
    current_model: Optional[str] = None
    loaded_layers: List[int] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    est_tokens: int
    owner: str
    cost: int
    created_at: float
    status: str = "PENDING"
    topology: List[dict] = field(default_factory=list) # The execution plan
    results_buffer: Dict[str, any] = field(default_factory=dict)

class ProductionScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}
        self.lock = asyncio.Lock()

    # --- WORKER MANAGEMENT ---

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        async with self.lock:
            self.workers[wid] = WorkerState(pubkey=wid, ws=ws, specs=specs)
        print(f"✅ [Scheduler] Worker Registered: {wid[:8]} | GPU: {specs.get('gpu')} | VRAM: {specs.get('vram_gb')}GB | URL: {specs.get('p2p_url')}")
        return wid

    async def unregister_worker(self, wid: str):
        async with self.lock:
            if wid in self.workers:
                del self.workers[wid]
        print(f"🔌 [Scheduler] Worker Disconnected: {wid[:8]}")
        # Logic to fail jobs assigned to this worker could go here

    async def update_heartbeat(self, wid: str):
        if wid in self.workers:
            self.workers[wid].last_heartbeat = time.time()

    # --- RESOURCE PLANNING ---

    async def _get_model_structure(self, model_id: str) -> Optional[dict]:
        """Ask Registry for layer count and size estimates."""
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{self.registry_url}/models/info", params={"model_id": model_id}) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"⚠️ [Scheduler] Registry lookup failed: {e}")
        return None

    def _plan_execution(self, model_info: dict) -> Optional[List[dict]]:
        """
        The 'Brain'. Determines which workers run which layers.
        Algorithm: Greedy Best-Fit based on VRAM availability.
        """
        total_layers = model_info['num_layers']
        # Estimate VRAM per layer (Safety margin: 1.2x)
        # Using the total_size_mb from registry is safer than guessing.
        total_size_gb = (model_info['total_size_mb'] / 1024) * 1.2
        gb_per_layer = total_size_gb / total_layers

        # Static overhead for context window (KV Cache) - approximate 2GB buffer
        kv_buffer = 2.0

        # Filter IDLE workers
        available_workers = [
            w for w in self.workers.values()
            if w.status == "IDLE" and w.ws.client_state.name == "CONNECTED"
        ]

        # Sort by VRAM descending (pack large GPUs first)
        available_workers.sort(key=lambda w: w.specs['vram_gb'], reverse=True)

        plan = []
        assigned_layers = 0

        for w in available_workers:
            if assigned_layers >= total_layers:
                break

            # Calculate capacity
            # If worker already has this model loaded, we treat them as highly desirable (affinity)
            # But for simplicity, we recalculate capacity.

            usable_vram = w.specs['vram_gb'] - kv_buffer
            if usable_vram <= 0: continue

            can_fit_layers = int(usable_vram / gb_per_layer)
            if can_fit_layers <= 0: continue

            # Cap at remaining layers
            layers_to_take = min(can_fit_layers, total_layers - assigned_layers)

            # Create Layer Range
            start = assigned_layers
            end = assigned_layers + layers_to_take - 1 # Inclusive

            # Use the P2P URL advertised by the worker
            endpoint = w.specs.get('p2p_url')
            if not endpoint:
                # Fallback for old workers (shouldn't happen with new code)
                endpoint = f"http://{w.specs.get('public_ip', 'localhost')}:{w.specs.get('p2p_port', 8003)}"

            plan.append({
                "worker_id": w.pubkey,
                "layers": list(range(start, end + 1)),
                "endpoint": endpoint
            })

            assigned_layers += layers_to_take

        if assigned_layers < total_layers:
            return None # Impossible to fit model on current grid

        return plan

    # --- JOB EXECUTION ---

    async def process_queue(self):
        """Main Loop: Pops from Redis, Plans, Dispatches."""
        print("🚀 [Scheduler] Dispatcher Active")
        while True:
            try:
                # 1. Pop Job
                item = await r.blpop("job_queue", timeout=2)
                if not item:
                    continue

                job_data = json.loads(item[1])
                job = JobState(
                    id=job_data['id'],
                    model_id=job_data['model'],
                    input_prompt=job_data['input'],
                    est_tokens=job_data['tokens'],
                    owner=job_data['owner'],
                    cost=job_data['cost'],
                    created_at=time.time()
                )

                print(f"📋 [Scheduler] Processing Job {job.id} ({job.model_id})")

                # 2. Get Model Specs
                model_info = await self._get_model_structure(job.model_id)
                if not model_info:
                    print(f"❌ [Scheduler] Model {job.model_id} unknown or not sharded.")
                    await self._fail_job(job.id, "Model not found in registry")
                    continue

                # 3. Create Plan (Retry loop)
                plan = None
                for attempt in range(5):
                    plan = self._plan_execution(model_info)
                    if plan:
                        break
                    print(f"⏳ [Scheduler] Resources busy. Retrying job {job.id} in 2s...")
                    await asyncio.sleep(2)

                if not plan:
                    print(f"❌ [Scheduler] Insufficient cluster resources for {job.model_id}")
                    # Re-queue at head or fail? For now, fail to avoid blocking.
                    await self._fail_job(job.id, "Insufficient Cluster Resources")
                    continue

                # 4. Dispatch
                job.topology = plan
                self.active_jobs[job.id] = job

                await self._dispatch_job(job)

            except Exception as e:
                print(f"💥 [Scheduler] Critical Dispatch Error: {e}")
                import traceback
                traceback.print_exc()

    async def _dispatch_job(self, job: JobState):
        """Sends commands to all workers in the topology."""

        # Link the chain
        # Plan list is [WorkerA, WorkerB, WorkerC]
        # WorkerA sends to WorkerB
        # WorkerC sends to Scheduler (via WS)

        for i, node in enumerate(job.topology):
            worker_id = node['worker_id']
            worker = self.workers.get(worker_id)
            if not worker:
                await self._fail_job(job.id, "Worker vanished during dispatch")
                return

            worker.status = "BUSY"

            # Determine Next Hop
            next_hop = None
            if i < len(job.topology) - 1:
                next_node = job.topology[i+1]
                next_hop = f"{next_node['endpoint']}/tensor_in" # The P2P URL

            payload = {
                "type": "EXECUTE",
                "job_id": job.id,
                "model_id": job.model_id,
                "layers": node['layers'],
                "input": job.input_prompt if i == 0 else None, # Only first node gets text
                "next_hop": next_hop,
                "is_first": (i == 0),
                "is_last": (i == len(job.topology) - 1)
            }

            await worker.ws.send_json(payload)
            print(f"📤 [Scheduler] Sent instruction to {worker_id[:8]} (Layers {node['layers'][0]}-{node['layers'][-1]})")

    # --- RESULT HANDLING ---

    async def handle_worker_result(self, wid: str, data: dict):
        """Received payload from the LAST worker in the chain via WS."""
        job_id = data.get('job_id')
        status = data.get('status')

        worker = self.workers.get(wid)
        if worker: worker.status = "IDLE"

        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]

        if status == "completed":
            print(f"🎉 [Scheduler] Job {job_id} Finished! Output len: {len(data.get('output', ''))}")

            # Save to Redis for API pickup
            await r.setex(f"result:{job_id}", 3600, json.dumps({
                "job_id": job_id,
                "status": "completed",
                "output": data.get('output'),
                "model": job.model_id
            }))

            # Process Payment
            await self._settle_payments(job)

            del self.active_jobs[job_id]

            # Free up all workers in topology
            for node in job.topology:
                w = self.workers.get(node['worker_id'])
                if w: w.status = "IDLE"

        elif status == "failed":
            error = data.get('error', 'Unknown worker error')
            await self._fail_job(job_id, error)

    async def _fail_job(self, job_id: str, reason: str):
        print(f"❌ [Scheduler] Job {job_id} Failed: {reason}")
        await r.setex(f"result:{job_id}", 3600, json.dumps({
            "job_id": job_id,
            "status": "failed",
            "error": reason
        }))

        # Cleanup
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            for node in job.topology:
                w = self.workers.get(node['worker_id'])
                if w: w.status = "IDLE"
            del self.active_jobs[job_id]

    async def _settle_payments(self, job: JobState):
        """Calculate shares and credit workers."""
        if not job.topology: return

        # Simple equal split for now, or based on layer count
        total_layers = sum(len(n['layers']) for n in job.topology)
        share_pool = calculate_worker_share(job.cost)

        for node in job.topology:
            w_share = int(share_pool * (len(node['layers']) / total_layers))
            wid = node['worker_id']

            # Redis increment
            curr = await r.incrby(f"worker_bal:{wid}", w_share)

            # Check Payout Threshold (e.g. 0.1 SOL)
            # Threshold logic handles actual on-chain tx
            if curr >= 100_000_000: # 0.1 SOL
                sig = await sign_payout(wid, curr)
                if sig:
                    await r.set(f"worker_bal:{wid}", 0)
                    # Notify worker
                    w = self.workers.get(wid)
                    if w:
                        await w.ws.send_json({"type": "PAYMENT", "amount": curr, "sig": sig})

# --- FASTAPI APP ---

scheduler = ProductionScheduler(registry_url=settings.REGISTRY_URL if hasattr(settings, 'REGISTRY_URL') else "http://localhost:8002")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        # Handshake
        reg_msg = await ws.receive_json()
        if reg_msg.get('type') != "REGISTER":
            await ws.close(code=1008)
            return

        wid = await scheduler.register_worker(ws, reg_msg['specs'])

        # Main Loop
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get('type')

            if msg_type == "HEARTBEAT":
                await scheduler.update_heartbeat(wid)
            elif msg_type == "RESULT":
                await scheduler.handle_worker_result(wid, msg)

    except WebSocketDisconnect:
        if wid: await scheduler.unregister_worker(wid)
    except Exception as e:
        print(f"WS Error: {e}")
        if wid: await scheduler.unregister_worker(wid)

@app.get("/workers")
async def list_workers():
    return [
        {
            "id": w.pubkey,
            "status": w.status,
            "gpu": w.specs.get("gpu"),
            "vram": w.specs.get("vram_gb"),
            "url": w.specs.get("p2p_url")
        }
        for w in scheduler.workers.values()
    ]
