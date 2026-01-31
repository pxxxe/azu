import asyncio
import json
import redis.asyncio as redis
import aiohttp
import time
import base64
import io
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Tuple
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
    specs: dict
    status: str = "IDLE"
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    owner: str
    est_tokens: int
    created_at: float
    topology: List[dict] = field(default_factory=list)
    is_moe: bool = False
    model_structure: dict = None
    moe_state: dict = field(default_factory=dict)  # Stores routing state per layer

class MoEScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}
        self.lock = asyncio.Lock()

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        capabilities = specs.get('capabilities', ['dense'])

        async with self.lock:
            self.workers[wid] = WorkerState(
                pubkey=wid,
                ws=ws,
                specs=specs,
                capabilities=capabilities
            )

        url = specs.get('p2p_url', 'UNKNOWN')
        gpu = specs.get('gpu', 'UNKNOWN')
        vram = specs.get('vram_gb', 0)
        caps = ', '.join(capabilities)
        print(f"✅ [Scheduler] Worker Registered: {wid[:8]} | GPU: {gpu} | VRAM: {vram}GB | Caps: {caps} | P2P: {url}")
        return wid

    async def unregister_worker(self, wid: str):
        async with self.lock:
            if wid in self.workers:
                del self.workers[wid]
        print(f"🔌 [Scheduler] Worker Disconnected: {wid[:8]}")

        to_fail = []
        for jid, job in self.active_jobs.items():
            for node in job.topology:
                if node.get('worker_id') == wid or node.get('router_worker_id') == wid:
                    to_fail.append(jid)
                    break
                if node.get('type') == 'moe':
                    expert_assignment = node.get('expert_assignment', {})
                    if wid in expert_assignment:
                        to_fail.append(jid)
                        break

        for jid in to_fail:
            await self._fail_job(jid, f"Worker {wid[:8]} disconnected")

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

    def _plan_moe_execution(self, model_info: dict) -> Optional[List[dict]]:
        """Plan execution for MoE model."""
        layer_metadata = model_info.get('layer_metadata', [])

        moe_workers = [
            w for w in self.workers.values()
            if w.status == "IDLE"
            and w.ws.client_state.name == "CONNECTED"
            and ("moe_router" in w.capabilities or "moe_expert" in w.capabilities)
        ]

        dense_workers = [
            w for w in self.workers.values()
            if w.status == "IDLE"
            and w.ws.client_state.name == "CONNECTED"
            and "dense" in w.capabilities
        ]

        if not moe_workers and not dense_workers:
            return None

        plan = []

        for layer_meta in layer_metadata:
            layer_idx = layer_meta['layer_idx']
            layer_type = layer_meta.get('type', 'dense')

            if layer_type == 'dense':
                if dense_workers:
                    worker = dense_workers[0]
                    plan.append({
                        "type": "dense",
                        "layer_idx": layer_idx,
                        "worker_id": worker.pubkey,
                        "endpoint": worker.specs.get('p2p_url')
                    })

            elif layer_type == 'moe':
                num_experts = layer_meta.get('num_experts', 8)
                experts_per_token = layer_meta.get('experts_per_token', 2)

                if moe_workers:
                    router_worker = moe_workers[0]

                    expert_assignment = {}
                    for expert_idx in range(num_experts):
                        worker_idx = expert_idx % len(moe_workers)
                        worker = moe_workers[worker_idx]

                        if worker.pubkey not in expert_assignment:
                            expert_assignment[worker.pubkey] = []
                        expert_assignment[worker.pubkey].append(expert_idx)

                    plan.append({
                        "type": "moe",
                        "layer_idx": layer_idx,
                        "router_worker_id": router_worker.pubkey,
                        "router_endpoint": router_worker.specs.get('p2p_url'),
                        "expert_assignment": expert_assignment,
                        "num_experts": num_experts,
                        "experts_per_token": experts_per_token
                    })

        return plan if plan else None

    def _plan_dense_execution(self, model_info: dict) -> Optional[List[dict]]:
        """Plan execution for dense model."""
        total_layers = model_info['num_layers']
        model_size_gb = (model_info['total_size_mb'] / 1024) * 1.2
        gb_per_layer = model_size_gb / total_layers
        kv_buffer = 2.0

        available_workers = [
            w for w in self.workers.values()
            if w.status == "IDLE" and w.ws.client_state.name == "CONNECTED"
        ]

        available_workers.sort(key=lambda w: w.specs.get('vram_gb', 0), reverse=True)

        plan = []
        assigned_layers = 0

        for w in available_workers:
            if assigned_layers >= total_layers:
                break

            endpoint = w.specs.get('p2p_url')
            usable_vram = w.specs.get('vram_gb', 0) - kv_buffer
            if usable_vram <= 0:
                continue

            can_fit_count = int(usable_vram / gb_per_layer)
            if can_fit_count <= 0:
                continue

            layers_to_take = min(can_fit_count, total_layers - assigned_layers)
            start = assigned_layers
            end = assigned_layers + layers_to_take - 1

            plan.append({
                "type": "dense_range",
                "worker_id": w.pubkey,
                "layers": list(range(start, end + 1)),
                "endpoint": endpoint
            })

            assigned_layers += layers_to_take

        if assigned_layers < total_layers:
            return None

        return plan

    async def process_queue(self):
        print("🚀 [MoE Scheduler] Dispatcher Active")
        while True:
            try:
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

                model_info = await self._get_model_structure(job.model_id)
                if not model_info:
                    await self._fail_job(job.id, f"Model {job.model_id} not found")
                    continue

                job.model_structure = model_info
                job.is_moe = model_info.get('is_moe', False)

                plan = None
                for _ in range(5):
                    if job.is_moe:
                        print(f"   🎯 Planning MoE execution...")
                        plan = self._plan_moe_execution(model_info)
                    else:
                        print(f"   📦 Planning dense execution...")
                        plan = self._plan_dense_execution(model_info)

                    if plan:
                        break
                    await asyncio.sleep(2)

                if not plan:
                    await self._fail_job(job.id, "Insufficient resources")
                    continue

                job.topology = plan
                self.active_jobs[job.id] = job

                if job.is_moe:
                    await self._dispatch_moe_job(job)
                else:
                    await self._dispatch_dense_job(job)

            except Exception as e:
                print(f"💥 [Scheduler] Critical Error: {e}")
                import traceback
                traceback.print_exc()

    async def _dispatch_dense_job(self, job: JobState):
        """Dispatch job for dense model."""
        print(f"   📦 Dispatching dense job {job.id}")

        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w:
                await self._fail_job(job.id, "Worker disconnected")
                return

            w.status = "BUSY"

            next_hop = None
            if i < len(job.topology) - 1:
                next_node = job.topology[i+1]
                next_hop = f"{next_node['endpoint']}/tensor_in"

            payload = {
                "type": "EXECUTE_DENSE",
                "job_id": job.id,
                "model_id": job.model_id,
                "layer_idx": node['layers'][0],
                "input": job.input_prompt if i == 0 else None,
                "next_hop": next_hop,
                "is_first": (i == 0),
                "is_last": (i == len(job.topology) - 1)
            }

            try:
                await w.ws.send_json(payload)
                print(f"   📤 Sent to {w.pubkey[:8]}")
            except:
                await self._fail_job(job.id, f"Failed to send to {w.pubkey[:8]}")
                return

    async def _dispatch_moe_job(self, job: JobState):
        """Dispatch job for MoE model with FULL expert routing."""
        print(f"   🎯 Dispatching MoE job {job.id}")

        first_node = job.topology[0]

        if first_node['type'] == 'dense':
            first_worker_id = first_node['worker_id']
        elif first_node['type'] == 'moe':
            first_worker_id = first_node['router_worker_id']
        else:
            await self._fail_job(job.id, f"Unknown first node type: {first_node['type']}")
            return

        first_worker = self.workers.get(first_worker_id)
        if not first_worker:
            await self._fail_job(job.id, "No worker available for first layer")
            return

        for node in job.topology:
            if node['type'] == 'dense':
                w = self.workers.get(node['worker_id'])
                if w:
                    w.status = "BUSY"
            elif node['type'] == 'moe':
                router_w = self.workers.get(node['router_worker_id'])
                if router_w:
                    router_w.status = "BUSY"

                for worker_id in node['expert_assignment'].keys():
                    expert_w = self.workers.get(worker_id)
                    if expert_w:
                        expert_w.status = "BUSY"

        payload = {
            "type": "EXECUTE_DENSE",
            "job_id": job.id,
            "model_id": job.model_id,
            "layer_idx": -1,
            "input": job.input_prompt,
            "is_first": True,
            "is_last": False,
            "next_hop": None
        }

        try:
            await first_worker.ws.send_json(payload)
            print(f"   📤 Sent embeddings request to {first_worker.pubkey[:8]}")

            await self._orchestrate_moe_layers(job)

        except Exception as e:
            await self._fail_job(job.id, f"Failed to dispatch: {str(e)}")

    async def _orchestrate_moe_layers(self, job: JobState):
        """Orchestrate execution of MoE layers after embeddings."""
        for node_idx, node in enumerate(job.topology):
            if node['type'] == 'moe':
                await self._execute_moe_layer(job, node, node_idx)
            elif node['type'] == 'dense' and node_idx > 0:
                await self._execute_dense_node(job, node, node_idx)

    async def _execute_moe_layer(self, job: JobState, node: dict, node_idx: int):
        """Execute single MoE layer with routing."""
        layer_idx = node['layer_idx']
        router_worker_id = node['router_worker_id']
        router_worker = self.workers.get(router_worker_id)

        if not router_worker:
            await self._fail_job(job.id, f"Router worker {router_worker_id[:8]} not found")
            return

        payload = {
            "type": "EXECUTE_MOE_ROUTER",
            "job_id": job.id,
            "model_id": job.model_id,
            "layer_idx": layer_idx,
            "top_k": node.get('experts_per_token', 2),
            "has_input": (node_idx > 0)
        }

        await router_worker.ws.send_json(payload)
        print(f"   🎯 Sent router request for layer {layer_idx}")

    async def _execute_dense_node(self, job: JobState, node: dict, node_idx: int):
        """Execute dense layer node."""
        worker = self.workers.get(node['worker_id'])
        if not worker:
            await self._fail_job(job.id, f"Dense worker not found")
            return

        next_hop = None
        if node_idx < len(job.topology) - 1:
            next_node = job.topology[node_idx + 1]
            next_hop = f"{next_node.get('endpoint')}/tensor_in"

        payload = {
            "type": "EXECUTE_DENSE",
            "job_id": job.id,
            "model_id": job.model_id,
            "layer_idx": node['layer_idx'],
            "next_hop": next_hop,
            "is_first": False,
            "is_last": (node_idx == len(job.topology) - 1)
        }

        await worker.ws.send_json(payload)

    async def handle_router_result(self, wid: str, data: dict):
        """Handle routing results from MoE router."""
        job_id = data.get('job_id')
        layer_idx = data.get('layer_idx')
        selected_experts = data.get('selected_experts')
        routing_weights = data.get('routing_weights')
        hidden_states_b64 = data.get('hidden_states')

        if job_id not in self.active_jobs:
            print(f"⚠️  Received router result for unknown job {job_id}")
            return

        job = self.active_jobs[job_id]
        print(f"   🎯 Router result for layer {layer_idx} | Job {job_id[:8]}")

        moe_node = None
        current_node_idx = None
        for i, node in enumerate(job.topology):
            if node.get('type') == 'moe' and node.get('layer_idx') == layer_idx:
                moe_node = node
                current_node_idx = i
                break

        if not moe_node:
            await self._fail_job(job_id, f"MoE node not found for layer {layer_idx}")
            return

        expert_assignment = moe_node['expert_assignment']

        job.moe_state[layer_idx] = {
            'selected_experts': selected_experts,
            'routing_weights': routing_weights,
            'hidden_states': hidden_states_b64,
            'pending_experts': set(),
            'expert_results': {},
            'current_node_idx': current_node_idx
        }

        for worker_id, expert_indices in expert_assignment.items():
            worker = self.workers.get(worker_id)
            if not worker:
                await self._fail_job(job_id, f"Expert worker {worker_id[:8]} not found")
                return

            worker_endpoint = worker.specs.get('p2p_url')

            payload = {
                "type": "EXECUTE_MOE_EXPERT",
                "job_id": job_id,
                "model_id": job.model_id,
                "layer_idx": layer_idx,
                "expert_indices": expert_indices,
                "has_input": False
            }

            try:
                await worker.ws.send_json(payload)
                job.moe_state[layer_idx]['pending_experts'].add(worker_id)
                print(f"   📤 Dispatched experts {expert_indices} to {worker_id[:8]}")

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{worker_endpoint}/tensor_in", json={
                        "tensor": hidden_states_b64
                    }) as resp:
                        if resp.status != 200:
                            print(f"⚠️  Failed to send tensor to {worker_id[:8]}")

            except Exception as e:
                await self._fail_job(job_id, f"Failed to dispatch to expert worker: {str(e)}")
                return

    async def handle_expert_result(self, wid: str, data: dict):
        """Handle expert execution results."""
        job_id = data.get('job_id')
        layer_idx = data.get('layer_idx')
        expert_indices = data.get('expert_indices')
        outputs = data.get('outputs')

        if job_id not in self.active_jobs:
            print(f"⚠️  Received expert result for unknown job {job_id}")
            return

        job = self.active_jobs[job_id]

        if layer_idx not in job.moe_state:
            await self._fail_job(job_id, f"No MoE state for layer {layer_idx}")
            return

        layer_state = job.moe_state[layer_idx]

        for i, expert_idx in enumerate(expert_indices):
            layer_state['expert_results'][expert_idx] = outputs[i]

        layer_state['pending_experts'].discard(wid)
        print(f"   🔬 Expert results from {wid[:8]} | Pending: {len(layer_state['pending_experts'])}")

        if len(layer_state['pending_experts']) == 0:
            await self._aggregate_and_forward_moe(job_id, layer_idx)

    async def _aggregate_and_forward_moe(self, job_id: str, layer_idx: int):
        """Aggregate expert outputs and forward to next layer."""
        job = self.active_jobs[job_id]
        layer_state = job.moe_state[layer_idx]

        selected_experts = layer_state['selected_experts']
        routing_weights = layer_state['routing_weights']
        expert_results = layer_state['expert_results']
        current_node_idx = layer_state['current_node_idx']

        print(f"   ⚖️  Aggregating {len(expert_results)} expert outputs for layer {layer_idx}")

        expert_tensors = {}
        for expert_idx, b64_output in expert_results.items():
            tensor_bytes = base64.b64decode(b64_output)
            buff = io.BytesIO(tensor_bytes)
            expert_tensors[expert_idx] = torch.load(buff)

        batch_size = len(selected_experts)
        seq_len = len(selected_experts[0]) if batch_size > 0 else 0
        top_k = len(selected_experts[0][0]) if seq_len > 0 else 0

        if not expert_tensors:
            await self._fail_job(job_id, "No expert results to aggregate")
            return

        first_expert_output = list(expert_tensors.values())[0]
        hidden_dim = first_expert_output.shape[-1]

        aggregated = torch.zeros(batch_size, seq_len, hidden_dim)

        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(top_k):
                    expert_idx = selected_experts[b][s][k]
                    weight = routing_weights[b][s][k]

                    if expert_idx in expert_tensors:
                        aggregated[b, s] += weight * expert_tensors[expert_idx][b, s]

        buff = io.BytesIO()
        torch.save(aggregated, buff)
        aggregated_b64 = base64.b64encode(buff.getvalue()).decode('utf-8')

        if current_node_idx == len(job.topology) - 1:
            worker = list(self.workers.values())[0]

            payload = {
                "type": "EXECUTE_LM_HEAD",
                "job_id": job_id,
                "model_id": job.model_id,
                "hidden_states": aggregated_b64
            }

            await worker.ws.send_json(payload)
            print(f"   🏁 Sent to LM head for decoding")

        else:
            next_node = job.topology[current_node_idx + 1]
            next_worker_id = next_node.get('worker_id') or next_node.get('router_worker_id')
            next_worker = self.workers.get(next_worker_id)

            if not next_worker:
                await self._fail_job(job_id, f"Next worker {next_worker_id[:8]} not found")
                return

            if next_node['type'] == 'dense':
                next_hop = None
                if current_node_idx + 1 < len(job.topology) - 1:
                    next_next_node = job.topology[current_node_idx + 2]
                    next_hop = f"{next_next_node.get('endpoint')}/tensor_in"

                next_endpoint = next_node.get('endpoint')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{next_endpoint}/tensor_in", json={
                        "tensor": aggregated_b64
                    }) as resp:
                        if resp.status != 200:
                            await self._fail_job(job_id, "Failed to forward to dense layer")
                            return

                payload = {
                    "type": "EXECUTE_DENSE",
                    "job_id": job_id,
                    "model_id": job.model_id,
                    "layer_idx": next_node['layer_idx'],
                    "next_hop": next_hop,
                    "is_first": False,
                    "is_last": (current_node_idx + 1 == len(job.topology) - 1)
                }

                await next_worker.ws.send_json(payload)
                print(f"   ➡️  Forwarded to dense layer {next_node['layer_idx']}")

            elif next_node['type'] == 'moe':
                next_endpoint = next_node.get('router_endpoint')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{next_endpoint}/tensor_in", json={
                        "tensor": aggregated_b64
                    }) as resp:
                        if resp.status != 200:
                            await self._fail_job(job_id, "Failed to forward to MoE router")
                            return

                payload = {
                    "type": "EXECUTE_MOE_ROUTER",
                    "job_id": job_id,
                    "model_id": job.model_id,
                    "layer_idx": next_node['layer_idx'],
                    "top_k": next_node.get('experts_per_token', 2),
                    "has_input": False
                }

                await next_worker.ws.send_json(payload)
                print(f"   ➡️  Forwarded to MoE router layer {next_node['layer_idx']}")

    async def handle_worker_result(self, wid: str, data: dict):
        """Handle results from workers."""
        job_id = data.get('job_id')
        status = data.get('status')

        w = self.workers.get(wid)
        if w:
            w.status = "IDLE"

        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]

        if status == "completed":
            output_text = data.get('output', '')
            print(f"🎉 [Scheduler] Job {job_id} Completed")

            in_tokens = len(job.input_prompt.split())
            out_tokens = len(output_text.split())

            total_work = 0
            for node in job.topology:
                if node.get('type') in ['dense', 'dense_range']:
                    total_work += len(node.get('layers', [1]))
                elif node.get('type') == 'moe':
                    total_work += node.get('num_experts', 8)

            cost = calculate_job_cost(total_work, in_tokens, out_tokens)
            new_balance = await r.decrby(f"balance:{job.owner}", cost)

            await self._settle_payments(job, cost)

            await r.setex(f"result:{job_id}", 3600, json.dumps({
                "job_id": job_id,
                "status": "completed",
                "output": output_text,
                "cost": cost,
                "final_balance": new_balance,
                "model": job.model_id
            }))

            del self.active_jobs[job_id]

            for node in job.topology:
                for key in ['worker_id', 'router_worker_id']:
                    node_wid = node.get(key)
                    if node_wid and node_wid in self.workers:
                        self.workers[node_wid].status = "IDLE"

                if node.get('type') == 'moe':
                    for expert_wid in node.get('expert_assignment', {}).keys():
                        if expert_wid in self.workers:
                            self.workers[expert_wid].status = "IDLE"

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
            job = self.active_jobs[job_id]
            for node in job.topology:
                for key in ['worker_id', 'router_worker_id']:
                    w_id = node.get(key)
                    if w_id and w_id in self.workers:
                        self.workers[w_id].status = "IDLE"

                if node.get('type') == 'moe':
                    for expert_wid in node.get('expert_assignment', {}).keys():
                        if expert_wid in self.workers:
                            self.workers[expert_wid].status = "IDLE"

            del self.active_jobs[job_id]

    async def _settle_payments(self, job: JobState, total_cost: int):
        """Distribute funds to workers."""
        if not job.topology:
            return

        share_pool = calculate_worker_share(total_cost)

        worker_work_units = {}
        total_work = 0

        for node in job.topology:
            if node.get('type') in ['dense', 'dense_range']:
                work_units = len(node.get('layers', [1]))
                wid = node.get('worker_id')
                if wid:
                    worker_work_units[wid] = worker_work_units.get(wid, 0) + work_units
                    total_work += work_units

            elif node.get('type') == 'moe':
                router_wid = node.get('router_worker_id')
                if router_wid:
                    worker_work_units[router_wid] = worker_work_units.get(router_wid, 0) + 1
                    total_work += 1

                num_experts = node.get('num_experts', 8)
                expert_assignment = node.get('expert_assignment', {})
                for expert_wid, expert_indices in expert_assignment.items():
                    work_units = len(expert_indices)
                    worker_work_units[expert_wid] = worker_work_units.get(expert_wid, 0) + work_units
                    total_work += work_units

        if total_work == 0:
            return

        for wid, work_units in worker_work_units.items():
            w_share = int(share_pool * (work_units / total_work))

            curr = await r.incrby(f"worker_bal:{wid}", w_share)

            if curr >= 100_000_000:
                print(f"   💰 Paying out {curr} lamports to {wid[:8]}...")
                sig = await sign_payout(wid, curr)
                if sig:
                    await r.set(f"worker_bal:{wid}", 0)
                    w = self.workers.get(wid)
                    if w:
                        try:
                            await w.ws.send_json({"type": "PAYMENT", "amount": curr, "sig": sig})
                        except:
                            pass

registry_url = settings.REGISTRY_URL if hasattr(settings, 'REGISTRY_URL') else "http://localhost:8002"
scheduler = MoEScheduler(registry_url)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        msg = await ws.receive_json()
        if msg.get('type') != "REGISTER":
            await ws.close(code=1008)
            return

        wid = await scheduler.register_worker(ws, msg['specs'])

        while True:
            data = await ws.receive_json()
            if data['type'] == "HEARTBEAT":
                await scheduler.update_heartbeat(wid)
            elif data['type'] == "RESULT":
                await scheduler.handle_worker_result(wid, data)
            elif data['type'] == "ROUTER_RESULT":
                await scheduler.handle_router_result(wid, data)
            elif data['type'] == "EXPERT_RESULT":
                await scheduler.handle_expert_result(wid, data)

    except WebSocketDisconnect:
        if wid:
            await scheduler.unregister_worker(wid)
    except Exception as e:
        print(f"WS Error: {e}")
        if wid:
            await scheduler.unregister_worker(wid)

@app.get("/workers")
async def list_workers():
    return [
        {
            "id": w.pubkey,
            "status": w.status,
            "gpu": w.specs.get('gpu'),
            "vram": w.specs.get('vram_gb'),
            "capabilities": w.capabilities,
            "p2p_url": w.specs.get('p2p_url')
        }
        for w in scheduler.workers.values()
    ]

@app.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id": j.id,
            "model": j.model_id,
            "is_moe": j.is_moe,
            "topology_nodes": len(j.topology)
        }
        for j in scheduler.active_jobs.values()
    ]
