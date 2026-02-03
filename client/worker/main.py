import torch
import asyncio
import websockets
import aiohttp
import json
import base64
import io
import os
import sys
import urllib.request
import traceback
import time
from aiohttp import web
from typing import Dict, Set, List, Tuple
from transformers import AutoTokenizer, AutoConfig
from layer_loader import LayerLoader
import gc

# CONFIG
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

P2P_TIMEOUT = None

class JobContext:
    def __init__(self, job_id, input_shape=None):
        self.job_id = job_id
        # FIX: Add per-layer input queues for routers
        self.layer_input_queues: Dict[int, asyncio.Queue] = {}
        self.expert_input_queues: Dict[Tuple[int, int], asyncio.Queue] = {}
        self.pending_expert_requests: Dict[Tuple[int, int], asyncio.Future] = {}
        self.routing_weights: torch.Tensor = None
        self.selected_indices: torch.Tensor = None
        self.original_shape = input_shape

    def get_layer_input_queue(self, layer_idx: int) -> asyncio.Queue:
        """Get or create the input queue for a specific layer's router."""
        if layer_idx not in self.layer_input_queues:
            self.layer_input_queues[layer_idx] = asyncio.Queue()
        return self.layer_input_queues[layer_idx]

    def get_expert_queue(self, layer_idx: int, expert_idx: int) -> asyncio.Queue:
        """Get or create the input queue for a specific (layer, expert) pair."""
        key = (layer_idx, expert_idx)
        if key not in self.expert_input_queues:
            self.expert_input_queues[key] = asyncio.Queue()
        return self.expert_input_queues[key]

class MoEWorker:
    def __init__(self):
        self._model_lock = asyncio.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)

        # State
        self.active_jobs: Dict[str, JobContext] = {}
        self.loaded_model_id = None
        self.model_config = None
        self.current_model = None
        self.p2p_session = None

        # Hardware Specs
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb} MB")
        else:
            self.vram_total_mb = 32000 # Fallback for CPU dev
            print("⚠️ No GPU detected, using simulated 32GB RAM")

        sys.stdout.flush()

        # Cache
        self.dense_layers = {}
        self.moe_routers = {}
        self.moe_experts = {}
        self.embeddings = None
        self.lm_head = None
        self._context_lock = asyncio.Lock()


    def get_p2p_url(self):
        if os.getenv("P2P_PUBLIC_URL"):
            return os.getenv("P2P_PUBLIC_URL").strip("/")

        template = os.getenv("P2P_URL_TEMPLATE")
        if template:
            try:
                return template.format(**os.environ).strip("/")
            except: pass

        try:
            ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            return f"http://{ip}:8003"
        except:
            return "http://127.0.0.1:8003"

    # --- P2P SERVER ---
    async def start_p2p_server(self):
        # FIX: Set client_max_size to None (Unlimited) or 1GB.
        # Default is 1MB, which rejects MoE tensor payloads causing the "stall".
        app = web.Application(client_max_size=1024**3)
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8003)
        await site.start()
        print("👂 [P2P] Server listening on :8003 (Max Payload: 1GB)")
        sys.stdout.flush()

    def _decode_tensor(self, b64_str):
        buff = io.BytesIO(base64.b64decode(b64_str))
        return torch.load(buff, map_location=self.device)

    def _encode_tensor(self, tensor):
        buff = io.BytesIO()
        torch.save(tensor, buff)
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    async def _get_context(self, job_id, create=False):
        async with self._context_lock:
            if job_id not in self.active_jobs:
                if create:
                    self.active_jobs[job_id] = JobContext(job_id)
                else:
                    return None
            return self.active_jobs[job_id]

    async def process_ingress_data(self, data):
        """Internal handler for incoming tensor data (bypass logic)"""
        job_id = data['job_id']
        msg_type = data.get('type', 'input')

        print(f"📨 [P2P] Received {msg_type} for job {job_id[:8]}...")
        sys.stdout.flush()

        tensor = self._decode_tensor(data['tensor'])
        print(f"   ✓ Decoded tensor shape: {tensor.shape}")
        sys.stdout.flush()

        ctx = await self._get_context(job_id, create=True)

        if msg_type == 'input':
            expert_idx = data.get('expert_idx')
            layer_idx = data.get('layer_idx')
            target_layer_idx = data.get('target_layer_idx')  # NEW: For router-to-router forwarding

            if expert_idx is not None and layer_idx is not None:
                # Expert input
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                print(f"   ➡️ Enqueueing input tensor for layer {layer_idx} expert {expert_idx} of job {job_id[:8]}")
                sys.stdout.flush()
                await queue.put(tensor)
                print(f"   ✓ Layer {layer_idx} expert {expert_idx} input enqueued (queue size: {queue.qsize()})")
                sys.stdout.flush()
            elif target_layer_idx is not None:
                # NEW: Router-to-router input with explicit target layer
                queue = ctx.get_layer_input_queue(target_layer_idx)
                print(f"   ➡️ Enqueueing input tensor for router layer {target_layer_idx} of job {job_id[:8]}")
                sys.stdout.flush()
                await queue.put(tensor)
                print(f"   ✓ Router layer {target_layer_idx} input enqueued (queue size: {queue.qsize()})")
                sys.stdout.flush()
            else:
                print(f"   ⚠️ Input message missing target info (expert_idx, layer_idx, or target_layer_idx)")
                sys.stdout.flush()

        elif msg_type == 'expert_result':
            expert_idx = data.get('expert_idx')
            layer_idx = data.get('layer_idx')
            print(f"   ⬅️ Received expert {expert_idx} result for layer {layer_idx} job {job_id[:8]}")
            sys.stdout.flush()

            if expert_idx is not None and layer_idx is not None:
                key = (layer_idx, expert_idx)
                if key in ctx.pending_expert_requests:
                    future = ctx.pending_expert_requests[key]
                    if not future.done():
                        future.set_result(tensor)
                        print(f"   ✓ Layer {layer_idx} expert {expert_idx} future resolved")
                        sys.stdout.flush()

    async def handle_tensor_ingress(self, request):
        try:
            data = await request.json()
            await self.process_ingress_data(data)
            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ [P2P] Error handling ingress: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            return web.Response(status=500, text=str(e))

    # --- EXECUTION LOGIC ---
    async def _send_p2p(self, url, payload):
        """Send tensor to another worker. Includes LOOPBACK OPTIMIZATION."""
        job_id = payload.get('job_id', 'unknown')
        msg_type = payload.get('type', 'unknown')

        print(f"📤 [P2P] Sending {msg_type} for job {job_id[:8] if len(job_id) > 8 else job_id} to {url}")
        sys.stdout.flush()

        # 1. Check for Loopback (Self-Transfer)
        my_p2p = self.get_p2p_url().rstrip("/")
        target_base = url.replace("/tensor_in", "").rstrip("/")

        if my_p2p == target_base:
            # Short-circuit: Inject directly into local handler
            print(f"⚡ [P2P] Loopback detected. Short-circuiting to local handler")
            sys.stdout.flush()
            try:
                await self.process_ingress_data(payload)
                print(f"   ✓ Loopback delivery successful")
                sys.stdout.flush()
                return
            except Exception as e:
                print(f"❌ Local P2P Error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                return

        # 2. Actual Network Transfer
        if self.p2p_session is None or self.p2p_session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            self.p2p_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        print(f"   🌐 Sending over network to {url}")
        sys.stdout.flush()
        for attempt in range(3):
            try:
                # Use the persistent session
                async with self.p2p_session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        # Success - handshake completed and data accepted
                        return
                    else:
                        text = await resp.text()
                        print(f"   ⚠️ P2P Handshake Rejected {resp.status}: {text[:100]}")
                        sys.stdout.flush()
            except Exception as e:
                print(f"   ⚠️ Connection Failed (attempt {attempt + 1}): {e}")
                sys.stdout.flush()
                await asyncio.sleep(0.2)

        print(f"❌ Failed to send to {url} after 3 attempts")
        sys.stdout.flush()

    async def _ensure_model(self, model_id):
        if self.current_model == model_id:
            return  # fast path, no lock needed

        async with self._model_lock:
            # Double-check after acquiring lock
            if self.current_model == model_id:
                return

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

            print(f"📥 Fetching model config from registry...")
            sys.stdout.flush()
            sanitized = model_id.replace("/", "_")
            config_path = self.loader.cache_dir / f"{sanitized}_config.json"

            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as sess:
                    config_url = f"{REGISTRY_URL}/layers/{sanitized}/config.json"
                    async with sess.get(config_url) as cfg_resp:
                        if cfg_resp.status != 200:
                            raise Exception(f"Config not found: {cfg_resp.status}")
                        cfg_data = await cfg_resp.read()
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(config_path, 'wb') as f:
                            f.write(cfg_data)

                self.model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
                print(f"✅ Config loaded from registry ({self.model_config.architectures[0]})")
                sys.stdout.flush()

            except Exception as e:
                print(f"❌ Failed to load config from registry: {e}")
                sys.stdout.flush()
                raise

            self.current_model = model_id

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
        # ARROGANT FIX: PRE-LOAD EVERYTHING BEFORE WAITING
        # ---------------------------------------------------------

        # 1. Pre-load Embeddings if needed
        if msg.get('is_first') and not self.embeddings:
            print(f"⚡ [Job {job_id[:8]}] Pre-loading embeddings...")
            sys.stdout.flush()
            self.embeddings = await self.loader.load_embeddings(model_id, self.device)

        # 2. Pre-load Dense Layer if needed
        if layer_idx != -1 and layer_idx not in self.dense_layers:
            print(f"📦 [Job {job_id[:8]}] Pre-loading dense layer {layer_idx}...")
            sys.stdout.flush()
            self.dense_layers[layer_idx] = await self.loader.load_dense_layer(
                model_id, layer_idx, self.device
            )

        # 3. Pre-load LM Head if needed
        if msg.get('is_last') and not self.lm_head:
            print(f"🔚 [Job {job_id[:8]}] Pre-loading LM Head...")
            sys.stdout.flush()
            self.lm_head = await self.loader.load_lm_head(model_id, self.device)

        # ---------------------------------------------------------
        # END PRE-LOAD - NOW WE WAIT
        # ---------------------------------------------------------

        # 4. Input Processing
        hidden_states = None
        if msg.get('is_first'):
            print(f"   🔤 Tokenizing prompt...")
            sys.stdout.flush()
            prompt = msg['input']
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            print(f"   ⚙️ Running embedding layer...")
            with torch.no_grad():
                hidden_states = self.embeddings(inputs['input_ids'])

            ctx.original_shape = hidden_states.shape
            print(f"   ✅ Embeddings complete. Shape: {hidden_states.shape}")

        else:
            # Wait for P2P input
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

        # 5. Process Layer
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

        # 6. Decode (if last)
        if msg.get('is_last'):
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

        # 7. Forward to next layer
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
        # ARROGANT FIX: PRE-LOAD ROUTER
        # ---------------------------------------------------------
        if layer_idx not in self.moe_routers:
            print(f"📦 [Job {job_id[:8]}] Pre-loading router {layer_idx}...")
            sys.stdout.flush()
            self.moe_routers[layer_idx] = await self.loader.load_moe_router(
                model_id, layer_idx, self.device
            )

        # ---------------------------------------------------------
        # NOW WAIT
        # ---------------------------------------------------------
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
        # ARROGANT FIX: PRE-LOAD EXPERT
        # ---------------------------------------------------------
        cache_key = (layer_idx, expert_idx)
        if cache_key not in self.moe_experts:
            print(f"📦 [Job {job_id[:8]}] Pre-loading expert {expert_idx} (Layer {layer_idx})...")
            sys.stdout.flush()
            self.moe_experts[cache_key] = await self.loader.load_moe_expert(
                model_id, layer_idx, expert_idx, self.device
            )

        # ---------------------------------------------------------
        # NOW WAIT
        # ---------------------------------------------------------
        expert_queue = ctx.get_expert_queue(layer_idx, expert_idx)
        print(f"   ⏳ Waiting for input tensor on layer {layer_idx} expert queue {expert_idx} (timeout={P2P_TIMEOUT}s)...")
        sys.stdout.flush()

        try:
            hidden_states = await asyncio.wait_for(expert_queue.get(), timeout=P2P_TIMEOUT)
            print(f"   ✅ Received input tensor. Shape: {hidden_states.shape}")
        except asyncio.TimeoutError:
            print(f"⏭️ [Job {job_id[:8]}] Expert {expert_idx} layer {layer_idx} not selected by router — exiting cleanly")
            return

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
    asyncio.run(MoEWorker().run())
