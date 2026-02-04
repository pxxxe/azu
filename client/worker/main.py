import asyncio
import torch
import json
import sys
import os
import traceback
from typing import Dict, Optional
import websockets
from aiohttp import web, ClientSession
from transformers import AutoTokenizer
from dataclasses import dataclass, field

# === CRITICAL: IMPORT THE ROBUST LOADER ===
from layer_loader import LayerLoader

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
P2P_PUBLIC_URL = os.getenv("P2P_PUBLIC_URL")
P2P_URL_TEMPLATE = os.getenv("P2P_URL_TEMPLATE")
P2P_PORT = 8003
P2P_TIMEOUT = 300

@dataclass
class JobContext:
    """Per-job state machine"""
    layer_input_queues: Dict[int, asyncio.Queue] = field(default_factory=dict)
    expert_queues: Dict[tuple, asyncio.Queue] = field(default_factory=dict)
    pending_expert_requests: Dict[tuple, asyncio.Future] = field(default_factory=dict)
    original_shape: Optional[tuple] = None

    def get_layer_input_queue(self, layer_idx: int) -> asyncio.Queue:
        if layer_idx not in self.layer_input_queues:
            self.layer_input_queues[layer_idx] = asyncio.Queue()
        return self.layer_input_queues[layer_idx]

    def get_expert_queue(self, layer_idx: int, expert_idx: int) -> asyncio.Queue:
        key = (layer_idx, expert_idx)
        if key not in self.expert_queues:
            self.expert_queues[key] = asyncio.Queue()
        return self.expert_queues[key]

class MoEWorker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 8000

        # === USE IMPORTED LOADER ===
        self.loader = LayerLoader(REGISTRY_URL, "layer_cache")

        self.current_model_id = None
        self.embeddings = None
        self.dense_layers: Dict[int, torch.nn.Module] = {}
        self.moe_routers: Dict[int, torch.nn.Module] = {}
        self.moe_experts: Dict[tuple, torch.nn.Module] = {}
        self.lm_head = None

        self.active_jobs: Dict[str, JobContext] = {}
        self.p2p_app = None

        print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | VRAM: {self.vram_total_mb} MB")
        sys.stdout.flush()

    def get_p2p_url(self):
        if P2P_PUBLIC_URL:
            return P2P_PUBLIC_URL
        elif P2P_URL_TEMPLATE:
            pod_id = os.getenv("RUNPOD_POD_ID", "unknown")
            return P2P_URL_TEMPLATE.replace("{RUNPOD_POD_ID}", pod_id)
        return f"http://localhost:{P2P_PORT}"

    async def start_p2p_server(self):
        self.p2p_app = web.Application(client_max_size=1024**3)
        self.p2p_app.router.add_post("/tensor_in", self.handle_p2p_tensor)
        runner = web.AppRunner(self.p2p_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', P2P_PORT)
        asyncio.create_task(site.start())
        print(f"👂 [P2P] Server listening on :{P2P_PORT} (Max Payload: 1GB)")
        sys.stdout.flush()

    async def handle_p2p_tensor(self, request):
        try:
            data = await request.json()
            job_id = data['job_id']
            tensor_data = data['tensor']
            msg_type = data.get('type')

            ctx = await self._get_context(job_id, create=True)
            tensor = self._decode_tensor(tensor_data)

            if msg_type == 'input':
                target_layer_idx = data.get('target_layer_idx')
                if target_layer_idx is not None:
                    await ctx.get_layer_input_queue(target_layer_idx).put(tensor)

                expert_idx = data.get('expert_idx')
                if expert_idx is not None:
                    layer_idx = data['layer_idx']
                    await ctx.get_expert_queue(layer_idx, expert_idx).put(tensor)

            elif msg_type == 'expert_result':
                layer_idx = data['layer_idx']
                expert_idx = data['expert_idx']
                key = (layer_idx, expert_idx)
                if key in ctx.pending_expert_requests:
                    future = ctx.pending_expert_requests[key]
                    if not future.done():
                        future.set_result(tensor)

            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ P2P Error: {e}")
            return web.Response(status=500, text=str(e))

    def _encode_tensor(self, tensor: torch.Tensor) -> dict:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data": tensor.cpu().numpy().tobytes().hex()
        }

    def _decode_tensor(self, data: dict) -> torch.Tensor:
        import numpy as np
        arr = np.frombuffer(bytes.fromhex(data['data']), dtype=data['dtype'])
        arr = arr.reshape(data['shape'])
        return torch.from_numpy(arr.copy())

    async def _send_p2p(self, url: str, payload: dict):
        # We create a new session here or could reuse one from self.loader if made public
        # Using a new one for now to avoid complexity, P2P traffic is lower frequency than shards
        async with ClientSession() as session:
            try:
                async with session.post(url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        print(f"   ⚠️ P2P send failed to {url}: {resp.status}")
            except Exception as e:
                print(f"   ❌ P2P send error to {url}: {e}")

    async def _get_context(self, job_id: str, create: bool = False) -> Optional[JobContext]:
        if job_id not in self.active_jobs and create:
            self.active_jobs[job_id] = JobContext()
        return self.active_jobs.get(job_id)

    async def _ensure_model(self, model_id: str):
        if self.current_model_id != model_id:
            print(f"🧹 New model {model_id} requested. Clearing VRAM...")
            sys.stdout.flush()
            self.embeddings = None
            self.dense_layers.clear()
            self.moe_routers.clear()
            self.moe_experts.clear()
            self.lm_head = None
            torch.cuda.empty_cache()
            self.current_model_id = model_id

    async def process_dense(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')
        next_layer_idx = msg.get('next_layer_idx')

        print(f"🔵 [DENSE] Processing job {job_id[:8]}, layer_idx={layer_idx}")
        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        hidden_states = None
        if msg.get('is_first'):
            print(f"   🔤 Tokenizing prompt...")
            prompt = msg['input']
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            if not self.embeddings:
                print(f"⚡ Loading embeddings...")
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)

            with torch.no_grad():
                hidden_states = self.embeddings(inputs['input_ids'])
            ctx.original_shape = hidden_states.shape
        else:
            queue = ctx.get_layer_input_queue(layer_idx)
            print(f"   ⏳ Waiting for input tensor on layer {layer_idx}...")
            try:
                hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"❌ [Job {job_id[:8]}] Timeout waiting for input")
                return

        # Process Layer
        layer_out = hidden_states
        if layer_idx != -1:
            if layer_idx not in self.dense_layers:
                print(f"📦 Loading dense layer {layer_idx}...")
                self.dense_layers[layer_idx] = await self.loader.load_dense_layer(model_id, layer_idx, self.device)

            with torch.no_grad():
                out = self.dense_layers[layer_idx](hidden_states.half())
                layer_out = out[0] if isinstance(out, tuple) else out

        # Decode or Forward
        if msg.get('is_last'):
            if not self.lm_head:
                print(f"🔚 Loading LM Head...")
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            with torch.no_grad():
                logits = self.lm_head(layer_out[:, -1, :])
                token_id = torch.argmax(logits, dim=-1).item()

            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            text = tokenizer.decode([token_id])
            print(f"   🎉 GENERATED TOKEN: '{text}'")

            await ws.send(json.dumps({"type": "RESULT", "job_id": job_id, "status": "completed", "output": text}))
            del self.active_jobs[job_id]
            return

        if next_hop:
            print(f"   ➡️ Forwarding to {next_hop}")
            await self._send_p2p(next_hop, {
                "job_id": job_id,
                "type": "input",
                "target_layer_idx": next_layer_idx,
                "tensor": self._encode_tensor(layer_out)
            })

    async def process_moe_router(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')
        expert_map = msg.get('expert_map', {})
        next_layer_idx = msg.get('next_layer_idx')

        print(f"🟢 [ROUTER] Processing job {job_id[:8]}, layer_idx={layer_idx}")
        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        queue = ctx.get_layer_input_queue(layer_idx)
        try:
            print(f"   ⏳ Waiting for input tensor...")
            hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"❌ [Job {job_id[:8]}] Router input timeout")
            return

        if layer_idx not in self.moe_routers:
            print(f"📦 Loading router {layer_idx}...")
            self.moe_routers[layer_idx] = await self.loader.load_moe_router(model_id, layer_idx, self.device)

        with torch.no_grad():
            logits = self.moe_routers[layer_idx](hidden_states.half())
            routing_weights, selected_indices = torch.topk(logits, k=2, dim=-1)
            routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

        top_indices = selected_indices.cpu()
        required_experts = set(top_indices.flatten().tolist())
        local_pending: Dict[int, asyncio.Future] = {}
        send_tasks = []

        for expert_idx in required_experts:
            target_url = expert_map.get(str(expert_idx))
            if not target_url: continue

            mask = (top_indices == expert_idx)
            rows, cols, _ = torch.where(mask)
            if rows.numel() == 0: continue

            sliced = hidden_states[rows, cols, :]
            future = asyncio.Future()
            local_pending[expert_idx] = future
            ctx.pending_expert_requests[(layer_idx, expert_idx)] = future

            send_tasks.append(asyncio.create_task(self._send_p2p(f"{target_url}/tensor_in", {
                "job_id": job_id,
                "type": "input",
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "tensor": self._encode_tensor(sliced)
            })))

        if send_tasks:
            await asyncio.gather(*send_tasks)

        pending = list(local_pending.values())
        if pending:
            print(f"   ⏳ Waiting for {len(pending)} expert results...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending), timeout=P2P_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"❌ [Job {job_id[:8]}] Expert results timeout")
                return

        # Merge
        batch, seq, hidden = hidden_states.shape
        final_output = torch.zeros((batch, seq, hidden), dtype=torch.float16, device=self.device)
        top_weights_dev = routing_weights.to(self.device)
        top_indices_dev = selected_indices.to(self.device)

        with torch.no_grad():
            for expert_idx, future in local_pending.items():
                if not future.done(): continue
                res = future.result().to(self.device)
                mask = (top_indices_dev == expert_idx)
                rows, cols, k_idx = torch.where(mask)
                w = top_weights_dev[rows, cols, k_idx].unsqueeze(-1)
                final_output.index_put_((rows, cols), res * w, accumulate=True)

        for expert_idx in local_pending:
            ctx.pending_expert_requests.pop((layer_idx, expert_idx), None)

        if next_hop:
            print(f"   ➡️ Forwarding to {next_hop}")
            await self._send_p2p(next_hop, {
                "job_id": job_id,
                "type": "input",
                "target_layer_idx": next_layer_idx,
                "tensor": self._encode_tensor(final_output)
            })

    async def process_moe_expert(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        print(f"🟡 [EXPERT] Processing expert {expert_idx} (Layer {layer_idx})")
        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        queue = ctx.get_expert_queue(layer_idx, expert_idx)
        try:
            hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"⏭️ [Job {job_id[:8]}] Expert {expert_idx} not used (timeout)")
            return

        cache_key = (layer_idx, expert_idx)
        if cache_key not in self.moe_experts:
            print(f"📦 Loading expert {expert_idx}...")
            self.moe_experts[cache_key] = await self.loader.load_moe_expert(model_id, layer_idx, expert_idx, self.device)

        with torch.no_grad():
            output = self.moe_experts[cache_key](hidden_states.half())

        await self._send_p2p(f"{return_url}/tensor_in", {
            "job_id": job_id,
            "type": "expert_result",
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "tensor": self._encode_tensor(output)
        })

    async def _safe_task_wrapper(self, coro, task_name):
        try:
            await coro
        except Exception as e:
            print(f"❌ TASK ERROR in {task_name}: {e}")
            traceback.print_exc()
            sys.stdout.flush()

    async def run(self):
        await self.start_p2p_server()
        while True:
            try:
                print(f"🔌 Connecting to {SCHEDULER_URL}...")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    p2p_url = self.get_p2p_url()
                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": {
                            "pubkey": "Worker_" + os.urandom(4).hex(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                            "vram_mb": self.vram_total_mb,
                            "p2p_url": p2p_url,
                            "capabilities": ["dense", "moe_router", "moe_expert"]
                        }
                    }))
                    print(f"✅ Connected & Registered")

                    async for raw in ws:
                        msg = json.loads(raw)
                        msg_type = msg['type']
                        job_id = msg.get('job_id', 'unknown')[:8]

                        if msg_type == 'EXECUTE_DENSE':
                            asyncio.create_task(self._safe_task_wrapper(
                                self.process_dense(msg, ws), f"EXECUTE_DENSE-{job_id}"))
                        elif msg_type == 'EXECUTE_ROUTER':
                            asyncio.create_task(self._safe_task_wrapper(
                                self.process_moe_router(msg, ws), f"EXECUTE_ROUTER-{job_id}"))
                        elif msg_type == 'EXECUTE_EXPERT':
                            asyncio.create_task(self._safe_task_wrapper(
                                self.process_moe_expert(msg, ws), f"EXECUTE_EXPERT-{job_id}"))
                        else:
                            print(f"⚠️ Unknown message type: {msg_type}")
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(MoEWorker().run())
