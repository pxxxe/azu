import asyncio
import json
import redis.asyncio as redis
import aiohttp
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from shared.config import settings
from shared.economics import calculate_worker_share, calculate_job_cost
from shared.solana_lib import sign_payout

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

@dataclass
class WorkerState:
    pubkey: str
    ws: WebSocket
    specs: dict  # {gpu, vram_gb, p2p_url}
    status: str = "IDLE"
    last_heartbeat: float = field(default_factory=time.time)

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    owner: str
    est_tokens: int
    created_at: float
    topology: List[dict] = field(default_factory=list)

class ProductionScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}
        self.lock = asyncio.Lock()

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        async with self.lock:
            self.workers[wid] = WorkerState(pubkey=wid, ws=ws, specs=specs)

        # Log registration details for debugging
        url = specs.get('p2p_url', 'UNKNOWN')
        gpu = specs.get('gpu', 'UNKNOWN')
        vram = specs.get('vram_gb', 0)
        print(f"✅ [Scheduler] Worker Registered: {wid[:8]} | GPU: {gpu} | VRAM: {vram}GB | URL: {url}")
        return wid

    async def unregister_worker(self, wid: str):
        async with self.lock:
            if wid in self.workers:
                del self.workers[wid]
        print(f"🔌 [Scheduler] Worker Disconnected: {wid[:8]}")

        # Fail any active jobs assigned to this worker
        to_fail = []
        for jid, job in self.active_jobs.items():
            for node in job.topology:
                if node['worker_id'] == wid:
                    to_fail.append(jid)
                    break

        for jid in to_fail:
            await self._fail_job(jid, f"Worker {wid[:8]} disconnected during execution")

    async def update_heartbeat(self, wid: str):
        if wid in self.workers:
            self.workers[wid].last_heartbeat = time.time()

    async def _get_model_structure(self, model_id: str) -> Optional[dict]:
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
        Greedy allocation logic.
        """
        total_layers = model_info['num_layers']

        # Estimate GB/layer with 1.2x safety margin
        model_size_gb = (model_info['total_size_mb'] / 1024) * 1.2
        gb_per_layer = model_size_gb / total_layers

        # KV Cache Buffer (2GB)
        kv_buffer = 2.0

        # Get Idle Workers
        available_workers = [
            w for w in self.workers.values()
            if w.status == "IDLE" and w.ws.client_state.name == "CONNECTED"
        ]

        # Sort by VRAM Descending
        available_workers.sort(key=lambda w: w.specs.get('vram_gb', 0), reverse=True)

        plan = []
        assigned_layers = 0

        for w in available_workers:
            if assigned_layers >= total_layers:
                break

            # Determine P2P Endpoint
            endpoint = w.specs.get('p2p_url')
            if not endpoint:
                # Fallback construction (risky behind NAT, but better than nothing)
                ip = w.specs.get('public_ip', '127.0.0.1')
                port = w.specs.get('p2p_port', 8003)
                endpoint = f"http://{ip}:{port}"

            # Capacity Check
            usable_vram = w.specs.get('vram_gb', 0) - kv_buffer
            if usable_vram <= 0: continue

            can_fit_count = int(usable_vram / gb_per_layer)
            if can_fit_count <= 0: continue

            # Allocate
            layers_to_take = min(can_fit_count, total_layers - assigned_layers)

            start = assigned_layers
            end = assigned_layers + layers_to_take - 1 # Inclusive

            plan.append({
                "worker_id": w.pubkey,
                "layers": list(range(start, end + 1)),
                "endpoint": endpoint
            })

            assigned_layers += layers_to_take

        if assigned_layers < total_layers:
            return None # Cannot fit model

        return plan

    async def process_queue(self):
        print("🚀 [Scheduler] Dispatcher Active")
        while True:
            try:
                # Blocking pop
                item = await r.blpop("job_queue", timeout=2)
                if not item:
                    continue

                raw_job = json.loads(item[1])
                job = JobState(
                    id=raw_job['id'],
                    model_id=raw_job['model'],
                    input_prompt=raw_job['input'],
                    owner=raw_job['owner'],
                    est_tokens=raw_job.get('tokens', 100),
                    created_at=time.time()
                )

                print(f"📋 [Scheduler] Processing Job {job.id}")

                # 1. Get Info
                model_info = await self._get_model_structure(job.model_id)
                if not model_info:
                    await self._fail_job(job.id, f"Model {job.model_id} not found/sharded in Registry")
                    continue

                # 2. Plan (with retry)
                plan = None
                for _ in range(5):
                    plan = self._plan_execution(model_info)
                    if plan: break
                    await asyncio.sleep(2)

                if not plan:
                    await self._fail_job(job.id, "Insufficient Cluster Resources (VRAM)")
                    continue

                # 3. Dispatch
                job.topology = plan
                self.active_jobs[job.id] = job
                await self._dispatch_job(job)

            except Exception as e:
                print(f"💥 [Scheduler] Critical Dispatch Error: {e}")
                import traceback
                traceback.print_exc()

    async def _dispatch_job(self, job: JobState):
        """Send EXECUTE commands to the chain."""

        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w:
                await self._fail_job(job.id, "Worker disconnected during dispatch")
                return

            w.status = "BUSY"

            next_hop = None
            if i < len(job.topology) - 1:
                # Next node's P2P endpoint
                next_node = job.topology[i+1]
                next_hop = f"{next_node['endpoint']}/tensor_in"

            payload = {
                "type": "EXECUTE",
                "job_id": job.id,
                "model_id": job.model_id,
                "layers": node['layers'],
                "input": job.input_prompt if i == 0 else None,
                "next_hop": next_hop,
                "is_first": (i == 0),
                "is_last": (i == len(job.topology) - 1)
            }

            try:
                await w.ws.send_json(payload)
                print(f"   📤 Sent layers {node['layers'][0]}-{node['layers'][-1]} to {w.pubkey[:8]}")
            except:
                await self._fail_job(job.id, f"Failed to send command to {w.pubkey[:8]}")
                return

    async def handle_worker_result(self, wid: str, data: dict):
        """Result comes from the LAST worker in the chain via WS."""
        job_id = data.get('job_id')
        status = data.get('status')

        # Free the worker who sent this
        w = self.workers.get(wid)
        if w: w.status = "IDLE"

        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]

        if status == "completed":
            output_text = data.get('output', '')
            print(f"🎉 [Scheduler] Job {job_id} Completed. Length: {len(output_text)}")

            # --- POST-RUN BILLING ---
            # 1. Calculate Real Cost
            in_tokens = len(job.input_prompt.split()) # Approximation
            out_tokens = len(output_text) // 4        # Approximation
            total_layers = sum(len(n['layers']) for n in job.topology)

            cost = calculate_job_cost(total_layers, in_tokens, out_tokens)

            # 2. Deduct User Balance
            new_balance = await r.decrby(f"balance:{job.owner}", cost)

            # 3. Pay Workers
            await self._settle_payments(job, cost)

            # 4. Store Result
            await r.setex(f"result:{job_id}", 3600, json.dumps({
                "job_id": job_id,
                "status": "completed",
                "output": output_text,
                "cost": cost,
                "final_balance": new_balance,
                "model": job.model_id
            }))

            # 5. Cleanup
            del self.active_jobs[job_id]
            # Free all other workers in topology
            for node in job.topology:
                topo_w = self.workers.get(node['worker_id'])
                if topo_w: topo_w.status = "IDLE"

        elif status == "failed":
            error = data.get('error', 'Unknown Error')
            await self._fail_job(job_id, error)

    async def _fail_job(self, job_id: str, reason: str):
        print(f"❌ [Scheduler] Job {job_id} Failed: {reason}")
        await r.setex(f"result:{job_id}", 3600, json.dumps({
            "job_id": job_id,
            "status": "failed",
            "error": reason
        }))

        if job_id in self.active_jobs:
            # Free everyone involved
            job = self.active_jobs[job_id]
            for node in job.topology:
                w = self.workers.get(node['worker_id'])
                if w: w.status = "IDLE"
            del self.active_jobs[job_id]

    async def _settle_payments(self, job: JobState, total_cost: int):
        """Distribute funds to workers involved."""
        if not job.topology: return

        share_pool = calculate_worker_share(total_cost)
        total_layers = sum(len(n['layers']) for n in job.topology)

        for node in job.topology:
            # Pro-rata based on layers processed
            node_layers = len(node['layers'])
            w_share = int(share_pool * (node_layers / total_layers))
            wid = node['worker_id']

            # Increment Ledger
            curr = await r.incrby(f"worker_bal:{wid}", w_share)

            # Check Payout Threshold (0.1 SOL = 100,000,000 Lamports)
            if curr >= 100_000_000:
                print(f"   💰 Paying out {curr} to {wid[:8]}...")
                sig = await sign_payout(wid, curr)
                if sig:
                    await r.set(f"worker_bal:{wid}", 0)
                    # Notify
                    w = self.workers.get(wid)
                    if w:
                        await w.ws.send_json({"type": "PAYMENT", "amount": curr, "sig": sig})

# --- APP INIT ---
registry_url = settings.REGISTRY_URL if hasattr(settings, 'REGISTRY_URL') else "http://localhost:8002"
scheduler = ProductionScheduler(registry_url)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        # Handshake
        msg = await ws.receive_json()
        if msg.get('type') != "REGISTER":
            await ws.close(code=1008)
            return

        wid = await scheduler.register_worker(ws, msg['specs'])

        # Loop
        while True:
            data = await ws.receive_json()
            if data['type'] == "HEARTBEAT":
                await scheduler.update_heartbeat(wid)
            elif data['type'] == "RESULT":
                await scheduler.handle_worker_result(wid, data)

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
            "gpu": w.specs.get('gpu'),
            "vram": w.specs.get('vram_gb'),
            "url": w.specs.get('p2p_url')
        }
        for w in scheduler.workers.values()
    ]
