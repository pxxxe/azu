import redis.asyncio as redis
import asyncio
import json
import time
import sys
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
    # Tracks the size of layers we have ASSIGNED to this worker
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
            # We record this for observability, but we DO NOT use it for planning
            # because it lags behind the downloads.
            self.workers[wid].actual_free_mb = data.get('vram_free_mb', 0)
            self.workers[wid].last_heartbeat = time.time()

    def _find_best_worker(self, size_mb: int, current_job_allocations: Dict[str, int],
                          previous_worker_id: str = None, cache_key: str = None) -> Optional[WorkerState]:
        candidates = list(self.workers.values())
        if not candidates: return None

        valid = []
        # print(f"--- Planning allocation for {size_mb:.0f}MB layer ---")

        for w in candidates:
            # 1. Determine Safety Cap (70% STRICT LIMIT)
            # 24GB -> ~16.8GB Usable for weights.
            # 7.2GB reserved for PyTorch Context (~1GB), Activations (~1-2GB), Fragmentation, and Buffers.
            safe_limit = w.vram_total_mb * 0.70

            # 2. Check Capacity using INTERNAL LEDGER
            pending_load = current_job_allocations.get(w.pubkey, 0)

            is_cached = cache_key in w.cached_layers
            allocation_cost = 0 if is_cached else size_mb

            projected_usage = w.vram_used_mb + pending_load + allocation_cost

            if projected_usage <= safe_limit:
                 valid.append((w, allocation_cost))
            else:
                 # Debug rejection
                 # print(f"   ❌ Rejected {w.pubkey[:8]}: Proj {projected_usage:.0f}MB > Limit {safe_limit:.0f}MB")
                 pass

        if not valid:
            print(f"   ⚠️ NO WORKERS FIT {size_mb:.0f}MB")
            return None

        def score(item):
            w, cost = item
            s = 0
            if cost == 0: s += 10000

            pending = current_job_allocations.get(w.pubkey, 0)
            projected_usage = w.vram_used_mb + pending
            safe_limit = w.vram_total_mb * 0.70
            available_room = safe_limit - projected_usage

            s += (available_room / 1024)

            if previous_worker_id and w.pubkey == previous_worker_id:
                s += 5

            return s

        chosen_worker, cost = max(valid, key=score)

        # Update ephemeral ledger
        new_load = current_job_allocations.get(chosen_worker.pubkey, 0) + cost
        current_job_allocations[chosen_worker.pubkey] = new_load

        print(f"   ✅ Selected {chosen_worker.pubkey[:8]}. New Job Load: {new_load:.0f}MB. Total Proj: {chosen_worker.vram_used_mb + new_load:.0f}MB")
        sys.stdout.flush()

        return chosen_worker

    def _plan_job(self, model_info) -> Optional[List[dict]]:
        layers = model_info.get('layer_metadata', [])
        if not self.workers: return None

        topology = []
        prev_worker_id = None
        current_job_allocations = {}

        print(f"\n🧠 Planning Job for {model_info['model_id']} ({len(layers)} layers)...")

        for layer in layers:
            layer_idx = layer['layer_idx']
            l_type = layer.get('type', 'dense')

            if l_type == 'dense':
                size = layer.get('size_mb', 0)
                cache_key = f"{model_info['model_id']}:{layer_idx}:main"

                print(f"Layer {layer_idx} (Dense): Need {size:.0f}MB")
                target_worker = self._find_best_worker(size, current_job_allocations, prev_worker_id, cache_key)
                if not target_worker: return None

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

                print(f"Layer {layer_idx} (Router): Need {combined_size:.0f}MB")
                router_worker = self._find_best_worker(combined_size, current_job_allocations, prev_worker_id, shared_key)
                if not router_worker: return None

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
                    # print(f"Layer {layer_idx} (Expert {exp_idx}): Need {expert_size:.0f}MB")
                    exp_worker = self._find_best_worker(expert_size, current_job_allocations, node['worker_id'], exp_key)

                    if not exp_worker:
                        print(f"⚠️ Cluster Full! Cannot place expert {exp_idx}")
                        return None

                    node['expert_map'][str(exp_idx)] = exp_worker.specs.get('p2p_url')

                topology.append(node)

        # COMMIT: The plan is valid. Now update the persistent WorkerState.
        print("💾 Committing Plan to Persistent Ledger...")
        for node in topology:
            w = self.workers.get(node['worker_id'])
            if not w: continue

            if node['type'] == 'dense':
                 l_meta = next(l for l in layers if l['layer_idx'] == node['layer_idx'])
                 size = l_meta.get('size_mb', 0)
                 cache_key = f"{model_info['model_id']}:{node['layer_idx']}:main"

                 if cache_key not in w.cached_layers:
                     w.vram_used_mb += size
                     w.cached_layers.add(cache_key)

            elif node['type'] == 'moe':
                 l_meta = next(l for l in layers if l['layer_idx'] == node['layer_idx'])

                 # Router/Shared
                 combined = l_meta.get('shared_size_mb', 0) + l_meta.get('router_size_mb', 0)
                 shared_key = f"{model_info['model_id']}:{node['layer_idx']}:shared"
                 if shared_key not in w.cached_layers:
                     w.vram_used_mb += combined
                     w.cached_layers.add(shared_key)

                 # Experts
                 exp_size = l_meta.get('expert_size_mb', 0)
                 for exp_idx, url in node['expert_map'].items():
                     w_exp = next((wk for wk in self.workers.values() if wk.specs.get('p2p_url') == url), None)
                     if w_exp:
                         exp_key = f"{model_info['model_id']}:{node['layer_idx']}:expert:{exp_idx}"
                         if exp_key not in w_exp.cached_layers:
                             w_exp.vram_used_mb += exp_size
                             w_exp.cached_layers.add(exp_key)

        # Log final states
        for w in self.workers.values():
            print(f"   🏁 Worker {w.pubkey[:8]} Final Usage: {w.vram_used_mb:.0f}MB / {w.vram_total_mb:.0f}MB")

        # Bookend the topology
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

                status = await self._check_model_status(model_id)

                if status == 'ready':
                    pass
                elif status == 'processing':
                    print(f"⏳ JIT: Model {model_id} is processing. Re-queueing...")
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue
                elif status == 'failed':
                    print(f"❌ JIT: Model {model_id} failed to shard. Dropping job.")
                    continue
                else:
                    print(f"⬇️ JIT: Triggering download for {model_id}")
                    try:
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(f"{self.registry_url}/models/shard", json={"model_id": model_id})
                    except: pass
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue

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

        first_node_endpoint = job.topology[0]['endpoint']

        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w:
                print(f"⚠️ Worker {node['worker_id']} not found!")
                continue

            is_last_node = (i == len(job.topology) - 1)
            next_hop = job.topology[i + 1]['endpoint'] + "/tensor_in" if not is_last_node else None
            next_layer_idx = job.topology[i + 1]['layer_idx'] if not is_last_node else None
            role = node.get('role')

            print(f"  Layer {i}: type={node['type']}, worker={w.pubkey[:8]}, next_hop={next_hop}")

            payload = {
                "job_id": job.id,
                "model_id": job.model_id,
                "layer_idx": node['layer_idx'],
                "next_hop": next_hop,
                "next_layer_idx": next_layer_idx
            }

            if node['type'] == 'dense':
                payload.update({
                    "type": "EXECUTE_DENSE",
                    "input": job.input_prompt if role == 'embed' else None,
                    "is_first": role == 'embed',
                    "is_last": role == 'decode',
                    "max_tokens": job.max_tokens,
                    "first_node_endpoint": first_node_endpoint if role == 'decode' else None
                })
                await w.ws.send_json(payload)

            elif node['type'] == 'moe':
                payload.update({
                    "type": "EXECUTE_ROUTER",
                    "expert_map": node['expert_map']
                })
                await w.ws.send_json(payload)

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
