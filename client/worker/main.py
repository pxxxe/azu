import asyncio
import torch
import json
import sys
import os
import traceback
import aiohttp
import urllib.request
import gc
from typing import Dict, Optional, Tuple, Set
import websockets
from aiohttp import web, ClientSession, TCPConnector, ClientTimeout
from transformers import AutoTokenizer, AutoConfig
from dataclasses import dataclass, field

# === ROBUST LOADER IMPORT ===
from layer_loader import LayerLoader

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
P2P_PUBLIC_URL = os.getenv("P2P_PUBLIC_URL")
P2P_URL_TEMPLATE = os.getenv("P2P_URL_TEMPLATE")
P2P_PORT = 8003
P2P_TIMEOUT = 300

class JobContext:
    def __init__(self, job_id):
        self.job_id = job_id
        self.layer_input_queues: Dict[int, asyncio.Queue] = {}
        self.expert_input_queues: Dict[Tuple[int, int], asyncio.Queue] = {}
        self.pending_expert_requests: Dict[Tuple[int, int], asyncio.Future] = {}

    def get_layer_input_queue(self, layer_idx: int) -> asyncio.Queue:
        if layer_idx not in self.layer_input_queues:
            self.layer_input_queues[layer_idx] = asyncio.Queue()
        return self.layer_input_queues[layer_idx]

    def get_expert_queue(self, layer_idx: int, expert_idx: int) -> asyncio.Queue:
        key = (layer_idx, expert_idx)
        if key not in self.expert_input_queues:
            self.expert_input_queues[key] = asyncio.Queue()
        return self.expert_input_queues[key]

class MoEWorker:
    def __init__(self):
        # --- HARDWARE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb} MB")
        else:
            self.vram_total_mb = 32000
            print("⚠️ No GPU detected, using simulated 32GB RAM")

        # --- LOADER ---
        self.loader = LayerLoader(REGISTRY_URL, "layer_cache")

        # --- LOCKS (ROBUSTNESS) ---
        self._model_lock = asyncio.Lock()   # Prevents model switching race conditions
        self._context_lock = asyncio.Lock() # Prevents job context race conditions

        # --- STATE ---
        self.active_jobs: Dict[str, JobContext] = {}
        self.current_model_id = None

        # Model Components
        self.embeddings = None
        self.lm_head = None
        self.dense_layers = {}
        self.moe_routers = {}
        self.moe_experts = {}

        # --- NETWORKING (SOCKET FIX) ---
        self.p2p_session = None # Shared session to prevent socket exhaustion
        self.p2p_app = None

        sys.stdout.flush()

    def get_p2p_url(self):
        if P2P_PUBLIC_URL: return P2P_PUBLIC_URL.strip("/")
        if P2P_URL_TEMPLATE:
            try:
                pod_id = os.getenv("RUNPOD_POD_ID", "unknown")
                return P2P_URL_TEMPLATE.replace("{RUNPOD_POD_ID}", pod_id).strip("/")
            except: pass
        try:
            # Fallback to IP detection
            ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            return f"http://{ip}:{P2P_PORT}"
        except:
            return f"http://127.0.0.1:{P2P_PORT}"

    # --- P2P SERVER ---
    async def start_p2p_server(self):
        # MAX SIZE 1GB IS CRITICAL FOR MoE TENSORS
        self.p2p_app = web.Application(client_max_size=1024**3)
        self.p2p_app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        runner = web.AppRunner(self.p2p_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', P2P_PORT)
        await site.start()
        print(f"👂 [P2P] Server listening on :{P2P_PORT} (Max Payload: 1GB)")
        sys.stdout.flush()

    # --- TENSOR UTILS ---
    def _decode_tensor(self, data: dict) -> torch.Tensor:
        import numpy as np
        # Uses numpy buffer to avoid base64 overhead, but supports hex encoding
        arr = np.frombuffer(bytes.fromhex(data['data']), dtype=data['dtype'])
        arr = arr.reshape(data['shape'])
        # Create tensor and ensure it's on the correct device
        return torch.from_numpy(arr.copy()).to(self.device)

    def _encode_tensor(self, tensor: torch.Tensor) -> dict:
        # Move to CPU and convert to numpy
        np_tensor = tensor.detach().cpu().numpy()
        return {
            "shape": list(np_tensor.shape),
            "dtype": str(np_tensor.dtype), # FIX: Returns 'float16', not 'torch.float16'
            "data": np_tensor.tobytes().hex()
        }

    # --- SHARED SESSION (THE FIX) ---
    async def _get_p2p_session(self):
        """Lazy-load a shared session for P2P traffic to prevent socket exhaustion."""
        if self.p2p_session is None or self.p2p_session.closed:
            # High limits, tuned timeouts
            timeout = ClientTimeout(total=60, sock_read=30, sock_connect=10)
            connector = TCPConnector(limit=0, ttl_dns_cache=300)
            self.p2p_session = ClientSession(connector=connector, timeout=timeout)
        return self.p2p_session

    # --- INGRESS LOGIC ---
    async def _get_context(self, job_id, create=False):
        async with self._context_lock:
            if job_id not in self.active_jobs:
                if create:
                    self.active_jobs[job_id] = JobContext(job_id)
                else:
                    return None
            return self.active_jobs[job_id]

    async def handle_tensor_ingress(self, request):
        try:
            data = await request.json()
            await self.process_ingress_data(data)
            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ [P2P] Error handling ingress: {e}")
            traceback.print_exc()
            return web.Response(status=500, text=str(e))

    async def process_ingress_data(self, data):
        job_id = data['job_id']
        msg_type = data.get('type', 'input')
        tensor = self._decode_tensor(data['tensor'])

        ctx = await self._get_context(job_id, create=True)

        if msg_type == 'input':
            expert_idx = data.get('expert_idx')
            layer_idx = data.get('layer_idx')
            target_layer_idx = data.get('target_layer_idx')

            if expert_idx is not None and layer_idx is not None:
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                await queue.put(tensor)
            elif target_layer_idx is not None:
                queue = ctx.get_layer_input_queue(target_layer_idx)
                await queue.put(tensor)

        elif msg_type == 'expert_result':
            expert_idx = data.get('expert_idx')
            layer_idx = data.get('layer_idx')
            if expert_idx is not None and layer_idx is not None:
                key = (layer_idx, expert_idx)
                if key in ctx.pending_expert_requests:
                    future = ctx.pending_expert_requests[key]
                    if not future.done():
                        future.set_result(tensor)

    # --- EXECUTION LOGIC ---
    async def _send_p2p(self, url, payload):
        """Send tensor to another worker. Includes LOOPBACK OPTIMIZATION."""
        # 1. Loopback Check
        my_p2p = self.get_p2p_url().rstrip("/")
        target_base = url.replace("/tensor_in", "").rstrip("/")

        if my_p2p == target_base:
            # Short-circuit: Inject directly into local handler
            try:
                await self.process_ingress_data(payload)
                return
            except Exception as e:
                print(f"❌ Local P2P Error: {e}")
                traceback.print_exc()
                return

        # 2. Network Transfer (Using Shared Session)
        session = await self._get_p2p_session()
        for attempt in range(3):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return
                    else:
                        print(f"   ⚠️ P2P Handshake Rejected {resp.status} from {url}")
            except Exception as e:
                print(f"   ⚠️ Connection Failed (attempt {attempt+1}) to {url}: {e}")
                await asyncio.sleep(0.2)
        print(f"❌ Failed to send to {url}")

    async def _ensure_model(self, model_id):
        if self.current_model_id == model_id:
            return

        async with self._model_lock:
            if self.current_model_id == model_id: return

            print(f"🧹 New model {model_id} requested. Clearing VRAM...")
            sys.stdout.flush()

            self.embeddings = None
            self.lm_head = None
            self.dense_layers.clear()
            self.moe_routers.clear()
            self.moe_experts.clear()
            self.active_jobs.clear()
            gc.collect()
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

        # --- JIT Embeddings ---
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
        else:
            queue = ctx.get_layer_input_queue(layer_idx)
            print(f"   ⏳ Waiting for input tensor...")
            try:
                hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"❌ [Job {job_id[:8]}] Timeout waiting for input")
                return

        # --- JIT Dense Layer ---
        layer_out = hidden_states
        if layer_idx != -1:
            if layer_idx not in self.dense_layers:
                print(f"📦 Loading dense layer {layer_idx}...")
                self.dense_layers[layer_idx] = await self.loader.load_dense_layer(model_id, layer_idx, self.device)

            with torch.no_grad():
                out = self.dense_layers[layer_idx](hidden_states.half())
                layer_out = out[0] if isinstance(out, tuple) else out

        # --- JIT Head & Decode ---
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

        # --- JIT Router ---
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

        # --- JIT Expert ---
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
            finally:
                if self.p2p_session and not self.p2p_session.closed:
                    await self.p2p_session.close()

if __name__ == "__main__":
    asyncio.run(MoEWorker().run())
