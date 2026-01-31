import torch
import asyncio
import websockets
import aiohttp
import json
import base64
import io
import os
import urllib.request
import traceback
import time
from aiohttp import web
from typing import Dict, Set, List, Tuple
from transformers import AutoTokenizer, AutoConfig
from layer_loader import LayerLoader

# CONFIG
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

P2P_TIMEOUT = 120

class JobContext:
    def __init__(self, job_id, input_shape=None):
        self.job_id = job_id
        self.input_queue = asyncio.Queue()
        self.pending_expert_requests: Dict[int, asyncio.Future] = {}
        self.routing_weights: torch.Tensor = None
        self.selected_indices: torch.Tensor = None
        self.original_shape = input_shape

class MoEWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)

        # State
        self.active_jobs: Dict[str, JobContext] = {}
        self.loaded_model_id = None
        self.model_config = None

        # Hardware Specs
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb} MB")
        else:
            self.vram_total_mb = 32000 # Fallback for CPU dev
            print("⚠️ No GPU detected, using simulated 32GB RAM")

        # Cache
        self.dense_layers = {}
        self.moe_routers = {}
        self.moe_experts = {}
        self.embeddings = None
        self.lm_head = None

    def get_p2p_url(self):
        """
        Determines the public P2P URL for this worker.
        Priority:
        1. Explicit Env Var (P2P_PUBLIC_URL)
        2. Template Env Var (P2P_URL_TEMPLATE) -> useful for dynamic platform IDs
        3. Auto-detect Public IP
        """
        # 1. Explicit
        if os.getenv("P2P_PUBLIC_URL"):
            return os.getenv("P2P_PUBLIC_URL").strip("/")

        # 2. Template (e.g. "https://{RUNPOD_POD_ID}-8003.proxy.runpod.net")
        template = os.getenv("P2P_URL_TEMPLATE")
        if template:
            try:
                # Format using current environment variables
                return template.format(**os.environ).strip("/")
            except KeyError as e:
                print(f"⚠️ P2P Template missing env var: {e}")
            except Exception as e:
                print(f"⚠️ P2P Template error: {e}")

        # 3. Auto-IP
        try:
            ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            return f"http://{ip}:8003"
        except:
            return "http://127.0.0.1:8003"

    # --- P2P SERVER ---
    async def start_p2p_server(self):
        app = web.Application()
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8003)
        await site.start()
        print("👂 [P2P] Server listening on :8003")

    def _decode_tensor(self, b64_str):
        buff = io.BytesIO(base64.b64decode(b64_str))
        return torch.load(buff, map_location=self.device)

    def _encode_tensor(self, tensor):
        buff = io.BytesIO()
        torch.save(tensor, buff)
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    async def _get_context(self, job_id, create=False):
        if job_id not in self.active_jobs:
            if create:
                self.active_jobs[job_id] = JobContext(job_id)
            else:
                return None
        return self.active_jobs[job_id]

    async def handle_tensor_ingress(self, request):
        try:
            data = await request.json()
            job_id = data['job_id']
            msg_type = data.get('type', 'input')
            tensor = self._decode_tensor(data['tensor'])

            ctx = await self._get_context(job_id, create=True)

            if msg_type == 'input':
                await ctx.input_queue.put(tensor)

            elif msg_type == 'expert_result':
                expert_idx = data.get('expert_idx')
                if expert_idx is not None and expert_idx in ctx.pending_expert_requests:
                    future = ctx.pending_expert_requests[expert_idx]
                    if not future.done():
                        future.set_result(tensor)

            return web.Response(text="OK")
        except Exception as e:
            traceback.print_exc()
            return web.Response(status=500, text=str(e))

    # --- EXECUTION LOGIC ---
    async def _send_p2p(self, url, payload):
        for i in range(3):
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(url, json=payload, timeout=30) as resp:
                        if resp.status == 200: return
            except:
                await asyncio.sleep(0.5)
        print(f"❌ Failed to send to {url}")

    async def _ensure_model(self, model_id):
        if self.loaded_model_id != model_id:
            print(f"🧹 New model {model_id} requested. Clearing VRAM...")
            self.dense_layers.clear()
            self.moe_routers.clear()
            self.moe_experts.clear()
            self.embeddings = None
            self.lm_head = None
            torch.cuda.empty_cache()
            self.loaded_model_id = model_id

            # Load config
            sanitized = model_id.replace("/", "_")
            try:
                self.model_config = AutoConfig.from_pretrained(
                    self.loader.cache_dir / f"{sanitized}_config.json", trust_remote_code=True
                )
            except:
                self.model_config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)

    async def process_dense(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        # 1. Input
        hidden_states = None
        if msg.get('is_first'):
            print(f"⚡ [Job {job_id}] Embedding...")
            if not self.embeddings:
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)
            prompt = msg['input']
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden_states = self.embeddings(inputs.input_ids)
        else:
            try:
                hidden_states = await asyncio.wait_for(ctx.input_queue.get(), timeout=P2P_TIMEOUT)
            except asyncio.TimeoutError:
                return

        # 2. Compute
        if layer_idx != -1:
            if layer_idx not in self.dense_layers:
                self.dense_layers[layer_idx] = await self.loader.load_dense_layer(model_id, layer_idx, self.device)
            with torch.no_grad():
                hidden_states = hidden_states.half()
                hidden_states = self.dense_layers[layer_idx](hidden_states)
                if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]

        # 3. Output/Forward
        if msg.get('is_last'):
            print(f"🏁 [Job {job_id}] Decoding...")
            if not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)
            with torch.no_grad():
                logits = self.lm_head(hidden_states.half())
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                output_text = tokenizer.decode(next_token, skip_special_tokens=True)
            await ws.send(json.dumps({
                "type": "RESULT", "job_id": job_id, "status": "completed", "output": output_text
            }))
        elif next_hop:
            await self._send_p2p(next_hop, {
                "job_id": job_id, "type": "input", "tensor": self._encode_tensor(hidden_states)
            })

        if job_id in self.active_jobs: del self.active_jobs[job_id]

    async def process_moe_router(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_map = msg['expert_map']
        next_hop = msg.get('next_hop')

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        try:
            hidden_states = await asyncio.wait_for(ctx.input_queue.get(), timeout=P2P_TIMEOUT)
        except asyncio.TimeoutError:
            return

        hidden_states = hidden_states.half()

        # Routing
        if layer_idx not in self.moe_routers:
            self.moe_routers[layer_idx] = await self.loader.load_moe_router(model_id, layer_idx, self.device)

        top_k = getattr(self.model_config, "num_experts_per_tok", 2)

        with torch.no_grad():
            router = self.moe_routers[layer_idx]
            logits = router(hidden_states)
            weights = torch.softmax(logits, dim=-1)
            top_weights, top_indices = torch.topk(weights, k=top_k, dim=-1)

            ctx.routing_weights = top_weights.cpu()
            ctx.selected_indices = top_indices.cpu()
            flat_indices = top_indices.view(-1)

        # Scatter
        required_experts = torch.unique(flat_indices).tolist()
        send_tasks = []

        for expert_idx in required_experts:
            target_url = expert_map.get(str(expert_idx))
            if not target_url: continue

            # Create Mask & Slice
            mask = (top_indices == expert_idx)
            rows, cols, _ = torch.where(mask)
            sliced = hidden_states[rows, cols, :]

            ctx.pending_expert_requests[expert_idx] = asyncio.Future()
            send_tasks.append(asyncio.create_task(self._send_p2p(f"{target_url}/tensor_in", {
                "job_id": job_id, "type": "input", "tensor": self._encode_tensor(sliced)
            })))

        if send_tasks: await asyncio.gather(*send_tasks)

        # Gather
        pending = list(ctx.pending_expert_requests.values())
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout=P2P_TIMEOUT)

        # Aggregate
        batch, seq, hidden = hidden_states.shape
        final_output = torch.zeros((batch, seq, hidden), dtype=torch.float16, device=self.device)
        top_weights = ctx.routing_weights.to(self.device)
        top_indices = ctx.selected_indices.to(self.device)

        with torch.no_grad():
            for expert_idx, future in ctx.pending_expert_requests.items():
                res = future.result().to(self.device)
                mask = (top_indices == expert_idx)
                rows, cols, k_idx = torch.where(mask)

                # Weighted Add
                w = top_weights[rows, cols, k_idx].unsqueeze(-1)
                final_output.index_put_((rows, cols), res * w, accumulate=True)

        if next_hop:
            await self._send_p2p(next_hop, {
                "job_id": job_id, "type": "input", "tensor": self._encode_tensor(final_output)
            })

        if job_id in self.active_jobs: del self.active_jobs[job_id]

    async def process_moe_expert(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        try:
            hidden_states = await asyncio.wait_for(ctx.input_queue.get(), timeout=P2P_TIMEOUT)
        except asyncio.TimeoutError:
            return

        cache_key = (layer_idx, expert_idx)
        if cache_key not in self.moe_experts:
            self.moe_experts[cache_key] = await self.loader.load_moe_expert(
                model_id, layer_idx, expert_idx, self.device
            )

        with torch.no_grad():
            output = self.moe_experts[cache_key](hidden_states.half())

        await self._send_p2p(f"{return_url}/tensor_in", {
            "job_id": job_id, "type": "expert_result", "expert_idx": expert_idx, "tensor": self._encode_tensor(output)
        })

        if job_id in self.active_jobs: del self.active_jobs[job_id]

    async def run(self):
        await self.start_p2p_server()
        while True:
            try:
                print(f"🔌 Connecting to {SCHEDULER_URL}...")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    print(f"✅ Connected. Reporting {self.vram_total_mb}MB VRAM")
                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": {
                            "pubkey": "Worker_" + os.urandom(4).hex(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                            "vram_mb": self.vram_total_mb,
                            "p2p_url": self.get_p2p_url(),
                            "capabilities": ["dense", "moe_router", "moe_expert"]
                        }
                    }))
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg['type'] == 'EXECUTE_DENSE': asyncio.create_task(self.process_dense(msg, ws))
                        elif msg['type'] == 'EXECUTE_ROUTER': asyncio.create_task(self.process_moe_router(msg, ws))
                        elif msg['type'] == 'EXECUTE_EXPERT': asyncio.create_task(self.process_moe_expert(msg, ws))
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(MoEWorker().run())
