import asyncio
import json
import redis.asyncio as redis
import aiohttp
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from shared.config import settings

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

@dataclass
class WorkerState:
    pubkey: str
    ws: WebSocket
    specs: dict
    vram_total_mb: int
    vram_used_mb: int = 0
    cached_layers: Set[str] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def vram_free_mb(self):
        return self.vram_total_mb - self.vram_used_mb

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    owner: str
    topology: List[dict] = field(default_factory=list)

class MoEScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        vram = specs.get('vram_mb', 24000)
        self.workers[wid] = WorkerState(pubkey=wid, ws=ws, specs=specs, vram_total_mb=vram)
        print(f"✅ Registered: {wid[:8]} | VRAM: {vram}MB")
        return wid

    async def unregister_worker(self, wid: str):
        if wid in self.workers:
            del self.workers[wid]
            print(f"❌ Worker {wid[:8]} disconnected")

    def _find_best_worker(self, size_mb: int, previous_worker_id: str = None, cache_key: str = None) -> Optional[WorkerState]:
        candidates = list(self.workers.values())
        if not candidates: return None

        valid = []
        for w in candidates:
            has_cache = cache_key in w.cached_layers
            if has_cache or w.vram_free_mb >= size_mb:
                valid.append(w)

        if not valid: return None

        def score(w):
            s = 0
            if cache_key and cache_key in w.cached_layers: s += 10000
            if previous_worker_id and w.pubkey == previous_worker_id: s += 5000
            s += (w.vram_free_mb / 1024)
            return s

        return max(valid, key=score)

    def _plan_job(self, model_info) -> Optional[List[dict]]:
        layers = model_info.get('layer_metadata', [])
        if not self.workers: return None

        topology = []
        prev_worker_id = None

        for layer in layers:
            layer_idx = layer['layer_idx']
            l_type = layer.get('type', 'dense')
            main_size = layer.get('size_mb', 0) if l_type == 'dense' else 50
            cache_key = f"{model_info['model_id']}:{layer_idx}:main"

            target_worker = self._find_best_worker(main_size, prev_worker_id, cache_key)
            if not target_worker:
                print(f"⚠️ Cluster Full! Cannot place layer {layer_idx}")
                return None

            if cache_key not in target_worker.cached_layers:
                target_worker.vram_used_mb += main_size
                target_worker.cached_layers.add(cache_key)

            node = {
                "layer_idx": layer_idx,
                "type": l_type,
                "worker_id": target_worker.pubkey,
                "endpoint": target_worker.specs.get('p2p_url')
            }
            prev_worker_id = target_worker.pubkey

            if l_type == 'moe':
                node['expert_map'] = {}
                num_experts = layer.get('num_experts', 0)
                expert_size = layer.get('total_size_mb', 0) / max(1, num_experts)

                for exp_idx in range(num_experts):
                    exp_key = f"{model_info['model_id']}:{layer_idx}:expert:{exp_idx}"
                    exp_worker = self._find_best_worker(expert_size, node['worker_id'], exp_key)

                    if not exp_worker: return None

                    if exp_key not in exp_worker.cached_layers:
                        exp_worker.vram_used_mb += expert_size
                        exp_worker.cached_layers.add(exp_key)

                    node['expert_map'][str(exp_idx)] = exp_worker.specs.get('p2p_url')

            topology.append(node)

        if not topology:
            return None

        # Bookend the topology with explicit embed and decode nodes.
        # layer_idx=-1 tells the worker to run embeddings (is_first) or lm_head (is_last)
        # without running any transformer layer. This is what was broken: the old code
        # assumed topology[0] was dense and set is_first: i==0, but for Mixtral (and any
        # pure-MoE model) layer 0 is MoE so embeddings were never dispatched.
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
                model_id = raw_job['model']

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

                job = JobState(raw_job['id'], model_id, raw_job['input'], raw_job['owner'], plan)
                self.active_jobs[job_id] = job
                await self._dispatch(job)

            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _dispatch(self, job: JobState):
        print(f"📦 Dispatching Job {job.id} (Smart Route)")
        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w: continue

            is_last_node = (i == len(job.topology) - 1)
            next_hop = job.topology[i + 1]['endpoint'] + "/tensor_in" if not is_last_node else None
            role = node.get('role')  # 'embed', 'decode', or None

            payload = {
                "job_id": job.id,
                "model_id": job.model_id,
                "layer_idx": node['layer_idx'],
                "next_hop": next_hop
            }

            if node['type'] == 'dense':
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
    except:
        if wid: await scheduler.unregister_worker(wid)
