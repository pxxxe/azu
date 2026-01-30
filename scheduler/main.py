"""
ENHANCED SCHEDULER - Multi-Worker Coordination

Features:
- Queries registry for worker capabilities
- Splits jobs across multiple workers
- Coordinates layer execution
- Handles tensor passing between workers
"""

import asyncio
import json
import redis.asyncio as redis
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Optional
from dataclasses import dataclass
from shared.config import settings
from shared.economics import calculate_worker_share
from shared.solana_lib import sign_payout
import os


app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

@dataclass
class WorkerInfo:
    pubkey: str
    ws: WebSocket
    specs: dict
    status: str  # IDLE, BUSY, OFFLINE, LOADING, READY
    assigned_layers: Optional[List[int]] = None
    loaded_model: Optional[str] = None  # NEW: Which model is loaded
    ready: bool = False  # NEW: Whether layers are loaded

@dataclass
class JobExecution:
    job_id: str
    model_id: str
    assigned_workers: List[str]  # worker pubkeys
    layer_splits: Dict[str, List[int]]  # worker_pubkey -> [layer_indices]
    results: Dict[str, any]  # worker_pubkey -> result
    status: str  # PENDING, IN_PROGRESS, COMPLETED, FAILED

class EnhancedScheduler:
    def __init__(self, registry_url: str = "http://localhost:8002"):
        self.workers: Dict[str, WorkerInfo] = {}
        self.jobs: Dict[str, JobExecution] = {}
        self.registry_url = registry_url
        self.ready_workers: List[str] = []  # NEW: Track ready workers
        self.target_model: Optional[str] = None  # NEW: Model we're preparing for

    async def register(self, ws: WebSocket, specs: dict) -> str:
        """Register worker with scheduler"""
        wid = specs['pubkey']

        self.workers[wid] = WorkerInfo(
            pubkey=wid,
            ws=ws,
            specs=specs,
            status="IDLE"
        )

        print(f"✅ Worker {wid[:8]}.. registered: {specs.get('gpu', 'unknown')} "
              f"({specs.get('vram_gb', 0)}GB VRAM)")

        # Also register with registry
        await self._register_worker_with_registry(wid, specs)

        return wid

    async def _register_worker_with_registry(self, worker_id: str, specs: dict):
        """Inform registry about worker capabilities"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/workers/register",
                    json={
                        "worker_id": worker_id,
                        "vram_gb": specs.get('vram_gb', 0),
                        "gpu": specs.get('gpu', 'unknown'),
                        "models": specs.get('models', [])
                    }
                ) as resp:
                    if resp.status == 200:
                        print(f"  ✅ Worker registered with registry")
        except Exception as e:
            print(f"  ⚠️ Registry registration failed: {e}")

    async def disconnect(self, wid: str):
        """Handle worker disconnect"""
        if wid in self.workers:
            print(f"Worker {wid[:8]}.. disconnected")
            self.workers[wid].status = "OFFLINE"

            # Notify registry
            try:
                async with aiohttp.ClientSession() as session:
                    await session.delete(f"{self.registry_url}/workers/{wid}")
            except:
                pass

    async def query_available_workers(self, model_id: str, required_layers: int) -> List[Dict]:
        """Query registry for workers that can handle this model"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/workers/query",
                    json={
                        "model_id": model_id,
                        "required_layers": required_layers
                    }
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('workers', [])
                    else:
                        print(f"Registry query failed: {resp.status}")
                        return []
        except Exception as e:
            print(f"Registry query error: {e}")
            return []

    def split_layers_across_workers(self, total_layers: int, workers: List[Dict]) -> Dict[str, List[int]]:
        """Intelligently split layers across available workers"""
        if not workers:
            return {}

        # Sort workers by VRAM capacity
        sorted_workers = sorted(workers, key=lambda w: w.get('vram_gb', 0), reverse=True)

        # Calculate layer distribution based on VRAM
        total_vram = sum(w.get('vram_gb', 0) for w in sorted_workers)

        splits = {}
        current_layer = 0

        for worker in sorted_workers:
            vram_ratio = worker.get('vram_gb', 0) / total_vram
            num_layers = int(total_layers * vram_ratio)

            # Ensure at least 1 layer per worker
            if num_layers == 0:
                num_layers = 1

            # Don't exceed remaining layers
            num_layers = min(num_layers, total_layers - current_layer)

            if num_layers > 0:
                layer_range = list(range(current_layer, current_layer + num_layers))
                splits[worker['worker_id']] = layer_range
                current_layer += num_layers

            if current_layer >= total_layers:
                break

        # Assign remaining layers to last worker
        if current_layer < total_layers:
            last_worker = sorted_workers[-1]['worker_id']
            if last_worker in splits:
                splits[last_worker].extend(range(current_layer, total_layers))

        return splits

    async def dispatch_loop(self):
      """Main job dispatcher loop"""
      print("🚀 Scheduler Dispatch Loop Started")

      while True:
        try:
            task = await r.blpop("job_queue", timeout=1)
            if not task:
              continue

            job = json.loads(task[1])
            job_id = job['id']
            model_id = job['model']

            print(f"\n📋 Processing Job {job_id}")

            # Check if workers have this model loaded
            ready_for_model = [
              w for w in self.workers.values()
              if w.ready and w.loaded_model == model_id
            ]

            if len(ready_for_model) < 2:
              print(f"  ⚠️ Workers not ready for {model_id}, requeueing...")
              await r.rpush("job_queue", json.dumps(job))
              await asyncio.sleep(2)
              continue

            print(f"  ✅ {len(ready_for_model)} workers ready")

            # Get layer splits from worker assignments
            layer_splits = {}
            for w in ready_for_model:
              layer_splits[w.pubkey] = w.assigned_layers

            print(f"  📊 Layer distribution:")
            for wid, layers in layer_splits.items():
                print(f"    {wid[:8]}...: layers {layers[0]}-{layers[-1]} ({len(layers)} total)")

            # Create job execution tracker
            self.jobs[job_id] = JobExecution(
                job_id=job_id,
                model_id=job['model'],
                assigned_workers=list(layer_splits.keys()),
                layer_splits=layer_splits,
                results={},
                status="IN_PROGRESS"
            )

            # Dispatch to workers
            await self._dispatch_to_workers(job, layer_splits)

        except Exception as e:
            print(f"❌ Dispatch error: {e}")
            import traceback
            traceback.print_exc()

    async def preload_model_to_workers(self, model_id: str):
      """
      Called after model is sharded.
      Assigns layer ranges to workers and tells them to preload.
      """
      # Get model info
      model_info = await self._get_model_info(model_id)
      if not model_info:
          print(f"❌ Model {model_id} not found in registry")
          return False

      total_layers = model_info['num_layers']

      # Get available workers
      available = [w for w in self.workers.values() if w.status == "IDLE"]
      if len(available) < 2:
          print(f"⚠️ Need at least 2 workers, only {len(available)} available")
          return False

      # Split layers across workers
      workers_to_use = available[:2]
      layers_per_worker = total_layers // 2

      print(f"📋 Preloading {model_id} to {len(workers_to_use)} workers...")

      for i, worker in enumerate(workers_to_use):
          layer_start = i * layers_per_worker
          layer_end = (i + 1) * layers_per_worker - 1 if i == 0 else total_layers - 1
          layers = list(range(layer_start, layer_end + 1))

          worker.status = "LOADING"
          worker.assigned_layers = layers
          worker.loaded_model = model_id

          # Tell worker to preload
          await worker.ws.send_json({
              "type": "PRELOAD",
              "model_id": model_id,
              "layers": layers,
              "is_first": i == 0,
              "is_last": i == len(workers_to_use) - 1
          })

          print(f"  → {worker.pubkey[:8]}...: layers {layer_start}-{layer_end}")

      self.target_model = model_id
      return True

    async def handle_ready(self, worker_id: str, data: dict):
      """Worker announces it has loaded layers"""
      if worker_id not in self.workers:
          return

      worker = self.workers[worker_id]
      worker.status = "READY"
      worker.ready = True

      if worker_id not in self.ready_workers:
          self.ready_workers.append(worker_id)

      print(f"🚀 Worker {worker_id[:8]}... ready with {worker.loaded_model}")

    async def _get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information from registry"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.registry_url}/models/{model_id}/info") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except:
            pass
        return None

    async def _dispatch_to_workers(self, job: dict, layer_splits: Dict[str, List[int]]):
        """Dispatch job to assigned workers"""
        job_id = job['id']

        # Send to first worker (they'll coordinate)
        worker_order = list(layer_splits.keys())

        for i, wid in enumerate(worker_order):
            if wid not in self.workers:
                continue

            worker = self.workers[wid]

            try:
                await worker.ws.send_json({
                    "type": "EXECUTE",
                    "job": {
                        **job,
                        "assigned_layers": layer_splits[wid],
                        "is_first": i == 0,
                        "is_last": i == len(worker_order) - 1,
                        "next_worker": worker_order[i+1] if i < len(worker_order)-1 else None
                    }
                })

                worker.status = "BUSY"
                worker.assigned_layers = layer_splits[wid]

                print(f"  ✅ Dispatched to worker {wid[:8]}...")

            except Exception as e:
                print(f"  ❌ Failed to dispatch to {wid[:8]}: {e}")
                await self._fail_job(job_id, f"Worker dispatch failed: {e}")

    async def handle_result(self, wid: str, result_data: dict):
        """Handle result from worker"""
        job_id = result_data['job_id']

        if wid not in self.workers:
            return

        worker = self.workers[wid]
        worker.status = "IDLE"
        worker.assigned_layers = None

        print(f"  ✅ Received result from worker {wid[:8]}...")

        if job_id not in self.jobs:
            print(f"  ⚠️ Job {job_id} not found in tracker")
            return

        job_exec = self.jobs[job_id]
        job_exec.results[wid] = result_data

        # Check if all workers completed
        if len(job_exec.results) == len(job_exec.assigned_workers):
            print(f"  ✅ All workers completed for job {job_id}")

            # Combine results and store
            final_output = self._combine_worker_results(job_exec)

            await r.setex(
                f"result:{job_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    "job_id": job_id,
                    "status": "completed",
                    "output": final_output,
                    "workers": len(job_exec.assigned_workers)
                })
            )

            # Pay workers
            await self._pay_workers(job_exec, result_data.get('cost', 0))

            job_exec.status = "COMPLETED"
            print(f"  🎉 Job {job_id} completed successfully!")

    def _combine_worker_results(self, job_exec: JobExecution) -> str:
        """Combine results from multiple workers"""
        # Last worker has the final output
        worker_order = sorted(
            job_exec.layer_splits.keys(),
            key=lambda w: job_exec.layer_splits[w][-1]
        )

        last_worker = worker_order[-1]

        if last_worker in job_exec.results:
            return job_exec.results[last_worker].get('output', '')

        return "Error: No output from workers"

    async def _pay_workers(self, job_exec: JobExecution, total_cost: int):
        """Distribute payment to workers"""
        worker_share = calculate_worker_share(total_cost)
        per_worker = worker_share // len(job_exec.assigned_workers)

        for wid in job_exec.assigned_workers:
            # Credit worker balance
            unpaid = await r.incrby(f"worker_bal:{wid}", per_worker)

            print(f"  💰 Credited {wid[:8]}... with {per_worker} lamports (total: {unpaid})")

            # Check payout threshold
            from shared.economics import MIN_PAYOUT_THRESHOLD
            if unpaid >= MIN_PAYOUT_THRESHOLD:
                print(f"  💸 Triggering payout for {wid[:8]}...")
                sig = await sign_payout(wid, unpaid)

                if sig:
                    await r.set(f"worker_bal:{wid}", 0)

                    # Notify worker
                    if wid in self.workers and self.workers[wid].ws:
                        try:
                            await self.workers[wid].ws.send_json({
                                "type": "PAYMENT",
                                "amount": unpaid,
                                "sig": sig
                            })
                            print(f"    ✅ Payment sent: {sig[:16]}...")
                        except:
                            pass

    async def _fail_job(self, job_id: str, reason: str):
        """Mark job as failed"""
        await r.setex(
            f"result:{job_id}",
            3600,
            json.dumps({
                "job_id": job_id,
                "status": "failed",
                "error": reason
            })
        )

        if job_id in self.jobs:
            self.jobs[job_id].status = "FAILED"

# Initialize scheduler
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
scheduler = EnhancedScheduler(registry_url="http://localhost:8002")

@app.on_event("startup")
async def start():
    asyncio.create_task(scheduler.dispatch_loop())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None

    try:
        data = await ws.receive_json()

        if data['type'] == 'REGISTER':
            wid = await scheduler.register(ws, data['specs'])

        # Message loop
        while True:
            msg = await ws.receive_json()

            if msg['type'] == 'READY':  # NEW
                await scheduler.handle_ready(wid, msg)

            elif msg['type'] == 'RESULT':
                await scheduler.handle_result(wid, msg)

            elif msg['type'] == 'HEARTBEAT':
                await ws.send_json({"type": "ACK"})

    except WebSocketDisconnect:
        if wid:
            await scheduler.disconnect(wid)

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job execution status"""
    if job_id in scheduler.jobs:
        job_exec = scheduler.jobs[job_id]

        return {
            "job_id": job_id,
            "status": job_exec.status,
            "assigned_workers": [
                {
                    "worker_id": wid,
                    "layer_range": f"{layers[0]}-{layers[-1]}"
                }
                for wid, layers in job_exec.layer_splits.items()
            ],
            "completed_workers": len(job_exec.results)
        }

    # Check Redis for completed job
    result = await r.get(f"result:{job_id}")
    if result:
        return json.loads(result)

    return {"job_id": job_id, "status": "not_found"}

@app.get("/workers")
async def list_workers():
    """List connected workers"""
    return [
      {
        "pubkey": w.pubkey,
        "status": w.status,
        "gpu": w.specs.get('gpu', 'unknown'),
        "vram_gb": w.specs.get('vram_gb', 0),
        "assigned_layers": w.assigned_layers
      }
      for w in scheduler.workers.values()
    ]

@app.get("/workers/ready")
async def get_ready_status():
  """Check how many workers are ready"""
  return {
    "total": len(scheduler.workers),
    "ready": len(scheduler.ready_workers),
    "target_model": scheduler.target_model,
    "workers": [
      {
        "id": w.pubkey[:8] + "...",
        "status": w.status,
        "ready": w.ready,
        "model": w.loaded_model,
        "layers": f"{w.assigned_layers[0]}-{w.assigned_layers[-1]}" if w.assigned_layers else None
      }
      for w in scheduler.workers.values()
    ]
  }

@app.post("/preload/{model_id}")
async def trigger_preload(model_id: str):
  """Trigger workers to preload model"""
  success = await scheduler.preload_model_to_workers(model_id)
  if success:
      return {"status": "preloading"}
  else:
      raise HTTPException(status_code=400, detail="Not enough workers or model not found")
