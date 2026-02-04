import asyncio
import torch
import json
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, Optional
import websockets
from aiohttp import web, ClientSession
from transformers import AutoTokenizer
from dataclasses import dataclass, field

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
P2P_PUBLIC_URL = os.getenv("P2P_PUBLIC_URL")
P2P_URL_TEMPLATE = os.getenv("P2P_URL_TEMPLATE")
P2P_PORT = 8003
P2P_TIMEOUT = 300  # 5 minutes

@dataclass
class JobContext:
    """Per-job state machine"""
    layer_input_queues: Dict[int, asyncio.Queue] = field(default_factory=dict)
    expert_queues: Dict[tuple, asyncio.Queue] = field(default_factory=dict)  # (layer_idx, expert_idx)
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

class LayerLoader:
    """Handles downloading layer files from registry"""
    def __init__(self, registry_url: str, cache_dir: str):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.download_locks: Dict[str, asyncio.Lock] = {}  # NEW: Per-file download lock

    async def _download(self, url: str, path: Path):
        """Download with lock to prevent duplicate concurrent downloads"""
        # Use the destination path as the lock key
        lock_key = str(path)
        if lock_key not in self.download_locks:
            self.download_locks[lock_key] = asyncio.Lock()

        async with self.download_locks[lock_key]:
            # Check again inside lock - another task may have downloaded it
            if path.exists():
                print(f"   ✓ Using cached {path.name}")
                return

            print(f"   📥 Downloading {url}")
            sys.stdout.flush()

            async with ClientSession() as session:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}")

                        # Download in chunks to avoid memory issues
                        temp_path = path.with_suffix('.tmp')
                        with open(temp_path, 'wb') as f:
                            async for chunk in resp.content.iter_chunked(1024 * 1024 * 256):  # 256MB chunks
                                f.write(chunk)

                        # Atomic rename
                        temp_path.rename(path)
                        print(f"   ✅ Downloaded {path.name}")
                        sys.stdout.flush()

                except Exception as e:
                    if path.with_suffix('.tmp').exists():
                        path.with_suffix('.tmp').unlink()
                    print(f"   ❌ Error downloading {url}: {e}")
                    sys.stdout.flush()
                    raise e

    async def load_embeddings(self, model_id: str, device):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_embeddings.pt"

        if not path.exists():
            await self._download(f"{self.registry_url}/layers/{sanitized}/embeddings.pt", path)

        return torch.load(path, map_location=device)

    async def load_dense_layer(self, model_id: str, layer_idx: int, device):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_layer_{layer_idx}_dense.pt"

        if not path.exists():
            await self._download(f"{self.registry_url}/layers/{sanitized}/layer_{layer_idx}_dense.pt", path)

        return torch.load(path, map_location=device)

    async def load_lm_head(self, model_id: str, device):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_lm_head.pt"

        if not path.exists():
            await self._download(f"{self.registry_url}/layers/{sanitized}/lm_head.pt", path)

        return torch.load(path, map_location=device)

    async def load_moe_router(self, model_id: str, layer_idx: int, device):
        sanitized = model_id.replace("/", "_")

        # Download config first
        config_path = self.cache_dir / f"{sanitized}_config.json"
        if not config_path.exists():
            print(f"   📥 Fetching model config from registry...")
            sys.stdout.flush()
            await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        else:
            print(f"   ✓ Using cached {sanitized}_config.json")

        # Load router weights
        filename = f"layer_{layer_idx}_router.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"

        print(f"   📋 Loading config from {config_path}...")
        sys.stdout.flush()

        if not path.exists():
            await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        return torch.load(path, map_location=device)

    async def load_moe_expert(self, model_id: str, layer_idx: int, expert_idx: int, device):
        sanitized = model_id.replace("/", "_")

        # Download config first
        config_path = self.cache_dir / f"{sanitized}_config.json"
        if not config_path.exists():
            print(f"   📥 Fetching model config from registry...")
            sys.stdout.flush()
            await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        else:
            print(f"   ✓ Using cached {sanitized}_config.json")

        # Load expert weights
        filename = f"layer_{layer_idx}_expert_{expert_idx}.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"

        print(f"   📋 Loading config from {config_path}...")
        sys.stdout.flush()

        if not path.exists():
            await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        return torch.load(path, map_location=device)

class MoEWorker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 8000

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
        """HTTP server for P2P tensor transfer"""
        self.p2p_app = web.Application(client_max_size=1024**3)
        self.p2p_app.router.add_post("/tensor_in", self.handle_p2p_tensor)

        runner = web.AppRunner(self.p2p_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', P2P_PORT)
        asyncio.create_task(site.start())
        print(f"👂 [P2P] Server listening on :{P2P_PORT} (Max Payload: 1GB)")
        sys.stdout.flush()

    async def handle_p2p_tensor(self, request):
        """Receive tensor from another worker"""
        data = await request.json()
        job_id = data['job_id']
        tensor_data = data['tensor']
        msg_type = data.get('type')

        ctx = await self._get_context(job_id, create=True)
        tensor = self._decode_tensor(tensor_data)

        if msg_type == 'input':
            # Input for a layer or expert
            target_layer_idx = data.get('target_layer_idx')
            if target_layer_idx is not None:
                queue = ctx.get_layer_input_queue(target_layer_idx)
                await queue.put(tensor)

            expert_idx = data.get('expert_idx')
            if expert_idx is not None:
                layer_idx = data['layer_idx']
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                await queue.put(tensor)

        elif msg_type == 'expert_result':
            # Result from an expert
            layer_idx = data['layer_idx']
            expert_idx = data['expert_idx']
            key = (layer_idx, expert_idx)

            if key in ctx.pending_expert_requests:
                future = ctx.pending_expert_requests[key]
                if not future.done():
                    future.set_result(tensor)

        return web.Response(text="OK")

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
        """Send tensor to another worker"""
        async with ClientSession() as session:
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"   ⚠️ P2P send failed: {resp.status}")
            except Exception as e:
                print(f"   ❌ P2P send error: {e}")

    async def _get_context(self, job_id: str, create: bool = False) -> Optional[JobContext]:
        if job_id not in self.active_jobs and create:
            self.active_jobs[job_id] = JobContext()
        return self.active_jobs.get(job_id)

    async def _ensure_model(self, model_id: str):
        """Ensure we're working with the correct model"""
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

        print(f"🔵 [DENSE] Processing job {job_id[:8]}, layer_idx={layer_idx}, is_first={msg.get('is_first')}, is_last={msg.get('is_last')}, next_hop={next_hop}")
        sys.stdout.flush()

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        # ---------------------------------------------------------
        # CRITICAL FIX: ONLY LOAD AFTER INPUT IS READY
        # ---------------------------------------------------------

        # 1. Get input FIRST
        hidden_states = None
        if msg.get('is_first'):
            print(f"   🔤 Tokenizing prompt...")
            sys.stdout.flush()
            prompt = msg['input']
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            # NOW load embeddings
            if not self.embeddings:
                print(f"⚡ [Job {job_id[:8]}] Loading embeddings...")
                sys.stdout.flush()
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)

            print(f"   ⚙️ Running embedding layer...")
            with torch.no_grad():
                hidden_states = self.embeddings(inputs['input_ids'])

            ctx.original_shape = hidden_states.shape
            print(f"   ✅ Embeddings complete. Shape: {hidden_states.shape}")

        else:
            # Wait for P2P input FIRST
            queue = ctx.get_layer_input_queue(layer_idx)
            print(f"   ⏳ Waiting for input tensor on layer {layer_idx} input queue (timeout={P2P_TIMEOUT}s)...")
            sys.stdout.flush()
            try:
                hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                print(f"   ✅ Received input tensor. Shape: {hidden_states.shape}")
            except asyncio.TimeoutError:
                print(f"❌ [Job {job_id[:8]}] No input tensor received (timeout)")
                sys.stdout.flush()
                return

        # 2. NOW load layer weights (after we have input)
        if layer_idx != -1 and layer_idx not in self.dense_layers:
            print(f"📦 [Job {job_id[:8]}] Loading dense layer {layer_idx}...")
            sys.stdout.flush()
            self.dense_layers[layer_idx] = await self.loader.load_dense_layer(
                model_id, layer_idx, self.device
            )

        # 3. Process Layer
        if layer_idx != -1:
            print(f"   ⚙️ Running dense layer {layer_idx}...")
            sys.stdout.flush()
            with torch.no_grad():
                layer_out = self.dense_layers[layer_idx](hidden_states.half())
                if isinstance(layer_out, tuple):
                    layer_out = layer_out[0]
            print(f"   ✅ Layer {layer_idx} complete.")
        else:
            layer_out = hidden_states

        # 4. Decode (if last)
        if msg.get('is_last'):
            # NOW load LM head
            if not self.lm_head:
                print(f"🔚 [Job {job_id[:8]}] Loading LM Head...")
                sys.stdout.flush()
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            print(f"   🔚 Final layer — decoding token...")
            sys.stdout.flush()
            with torch.no_grad():
                logits = self.lm_head(layer_out[:, -1, :])
                token_id = torch.argmax(logits, dim=-1).item()

            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            text = tokenizer.decode([token_id])

            print(f"   🎉 GENERATED TOKEN: '{text}'")
            sys.stdout.flush()

            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "completed",
                "output": text
            }))

            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            return

        # 5. Forward to next layer
        if next_hop:
            print(f"   ➡️ Forwarding to next hop: {next_hop}")
            sys.stdout.flush()
            await self._send_p2p(next_hop, {
                "job_id": job_id,
                "type": "input",
                "target_layer_idx": next_layer_idx,
                "tensor": self._encode_tensor(layer_out)
            })
            print(f"   ✅ Forwarded to next hop")
        else:
            print(f"   ⚠️ No next_hop - job may be incomplete")
            sys.stdout.flush()

    async def process_moe_router(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')
        expert_map = msg.get('expert_map', {})
        next_layer_idx = msg.get('next_layer_idx')

        print(f"🟢 [ROUTER] Processing job {job_id[:8]}, layer_idx={layer_idx}, next_hop={next_hop}")
        print(f"   Expert map: {expert_map}")
        sys.stdout.flush()

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        # ---------------------------------------------------------
        # CRITICAL FIX: GET INPUT FIRST, THEN LOAD ROUTER
        # ---------------------------------------------------------

        # 1. Wait for input FIRST
        queue = ctx.get_layer_input_queue(layer_idx)
        print(f"   ⏳ Waiting for input tensor on layer {layer_idx} input queue (timeout={P2P_TIMEOUT}s)...")
        sys.stdout.flush()
        try:
            hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
            print(f"   ✅ Received input tensor. Shape: {hidden_states.shape}")
        except asyncio.TimeoutError:
            print(f"❌ [Job {job_id[:8]}] Router timed out waiting for input")
            sys.stdout.flush()
            return

        # 2. NOW load router (after we have input)
        if layer_idx not in self.moe_routers:
            print(f"📦 [Job {job_id[:8]}] Loading router {layer_idx}...")
            sys.stdout.flush()
            self.moe_routers[layer_idx] = await self.loader.load_moe_router(
                model_id, layer_idx, self.device
            )

        print(f"   ⚙️ Running router...")
        sys.stdout.flush()

        with torch.no_grad():
            routing_logits = self.moe_routers[layer_idx](hidden_states.half())
            routing_weights, selected_indices = torch.topk(routing_logits, k=2, dim=-1)
            routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

        top_indices = selected_indices.cpu()
        required_experts = set(top_indices.flatten().tolist())

        print(f"   ✅ Routing complete. Selected experts: {sorted(required_experts)}")
        sys.stdout.flush()

        # Create per-router futures dict
        local_pending: Dict[int, asyncio.Future] = {}
        send_tasks = []

        for expert_idx in required_experts:
            target_url = expert_map.get(str(expert_idx))
            if not target_url:
                print(f"   ⚠️ No URL for expert {expert_idx}")
                continue

            mask = (top_indices == expert_idx)
            rows, cols, _ = torch.where(mask)
            sliced = hidden_states[rows, cols, :]

            print(f"   📤 Dispatching to expert {expert_idx} at {target_url} (slice shape: {sliced.shape})")
            sys.stdout.flush()

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
            print(f"   ✅ All expert sends complete")

        pending = list(local_pending.values())
        if pending:
            print(f"   ⏳ Waiting for {len(pending)} expert results (timeout={P2P_TIMEOUT}s)...")
            sys.stdout.flush()
            try:
                await asyncio.wait_for(asyncio.gather(*pending), timeout=P2P_TIMEOUT)
                print(f"   ✅ All expert results received")
            except asyncio.TimeoutError:
                print(f"❌ [Job {job_id[:8]}] Timed out waiting for expert results")
                return

        # Merge
        batch, seq, hidden = hidden_states.shape
        final_output = torch.zeros((batch, seq, hidden), dtype=torch.float16, device=self.device)
        top_weights_dev = routing_weights.to(self.device)
        top_indices_dev = selected_indices.to(self.device)

        print(f"   🔧 Merging expert outputs...")
        with torch.no_grad():
            for expert_idx, future in local_pending.items():
                res = future.result().to(self.device)
                mask = (top_indices_dev == expert_idx)
                rows, cols, k_idx = torch.where(mask)
                w = top_weights_dev[rows, cols, k_idx].unsqueeze(-1)
                final_output.index_put_((rows, cols), res * w, accumulate=True)

        # Cleanup
        for expert_idx in local_pending:
            ctx.pending_expert_requests.pop((layer_idx, expert_idx), None)

        if next_hop:
            print(f"   ➡️ Forwarding to next hop: {next_hop}")
            await self._send_p2p(next_hop, {
                "job_id": job_id,
                "type": "input",
                "target_layer_idx": next_layer_idx,
                "tensor": self._encode_tensor(final_output)
            })
            print(f"   ✅ Forwarded to next hop")

    async def process_moe_expert(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        print(f"🟡 [EXPERT] Processing expert {expert_idx} for job {job_id[:8]}, layer_idx={layer_idx}")
        print(f"   Return URL: {return_url}")
        sys.stdout.flush()

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        # ---------------------------------------------------------
        # CRITICAL FIX: WAIT FOR INPUT FIRST, THEN LOAD EXPERT
        # ---------------------------------------------------------

        # 1. Wait for input FIRST
        expert_queue = ctx.get_expert_queue(layer_idx, expert_idx)
        print(f"   ⏳ Waiting for input tensor on layer {layer_idx} expert queue {expert_idx} (timeout={P2P_TIMEOUT}s)...")
        sys.stdout.flush()

        try:
            hidden_states = await asyncio.wait_for(expert_queue.get(), timeout=P2P_TIMEOUT)
            print(f"   ✅ Received input tensor. Shape: {hidden_states.shape}")
        except asyncio.TimeoutError:
            print(f"⏭️ [Job {job_id[:8]}] Expert {expert_idx} layer {layer_idx} not selected by router — exiting cleanly")
            return

        # 2. NOW load expert (after we have input)
        cache_key = (layer_idx, expert_idx)
        if cache_key not in self.moe_experts:
            print(f"📦 [Job {job_id[:8]}] Loading expert {expert_idx} (Layer {layer_idx})...")
            sys.stdout.flush()
            self.moe_experts[cache_key] = await self.loader.load_moe_expert(
                model_id, layer_idx, expert_idx, self.device
            )

        print(f"   ⚙️ Running expert {expert_idx}...")
        with torch.no_grad():
            output = self.moe_experts[cache_key](hidden_states.half())

        print(f"   ✅ Expert {expert_idx} complete.")
        print(f"   ⬅️ Sending result back to router at {return_url}")
        sys.stdout.flush()

        await self._send_p2p(f"{return_url}/tensor_in", {
            "job_id": job_id,
            "type": "expert_result",
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "tensor": self._encode_tensor(output)
        })
        print(f"   ✅ Result sent back to router")

    async def _safe_task_wrapper(self, coro, task_name):
        """Wrapper that ensures exceptions in tasks are logged"""
        try:
            await coro
        except Exception as e:
            print(f"❌ TASK ERROR in {task_name}: {e}")
            traceback.print_exc()
            sys.stdout.flush()

    async def run(self):
        await self.start_p2p_server()
        import gc
        while True:
            try:
                print(f"🔌 Connecting to {SCHEDULER_URL}...")
                sys.stdout.flush()
                async with websockets.connect(SCHEDULER_URL) as ws:
                    p2p_url = self.get_p2p_url()
                    print(f"✅ Connected. Reporting {self.vram_total_mb}MB VRAM, P2P URL: {p2p_url}")
                    sys.stdout.flush()
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
                    print(f"📡 Registration sent")
                    sys.stdout.flush()

                    async for raw in ws:
                        msg = json.loads(raw)
                        msg_type = msg['type']
                        job_id = msg.get('job_id', 'unknown')[:8]

                        print(f"\n{'='*60}")
                        print(f"📬 Received message: {msg_type} for job {job_id}")
                        print(f"{'='*60}")
                        sys.stdout.flush()

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
                            sys.stdout.flush()
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                await asyncio.sleep(5)

if __name__ == "__main__":
    import aiohttp
    asyncio.run(MoEWorker().run())
