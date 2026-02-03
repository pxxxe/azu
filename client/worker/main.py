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
import gc
from aiohttp import web
from typing import Dict, Set, List, Tuple
from transformers import AutoTokenizer, AutoConfig
from layer_loader import LayerLoader

# CONFIG
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")
P2P_TIMEOUT = None

# DYNAMIC VRAM SETTINGS
MAX_VRAM_USAGE_PERCENT = 0.95
EVICT_THRESHOLD_PERCENT = 0.85

class JobContext:
    def __init__(self, job_id, model_id, input_shape=None):
        self.job_id = job_id
        self.model_id = model_id
        self.required_layers = set()
        self.layer_input_queues: Dict[int, asyncio.Queue] = {}
        self.expert_input_queues: Dict[Tuple[int, int], asyncio.Queue] = {}
        self.pending_expert_requests: Dict[Tuple[int, int], asyncio.Future] = {}
        self.routing_weights: torch.Tensor = None
        self.selected_indices: torch.Tensor = None
        self.original_shape = input_shape

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
        self._model_lock = asyncio.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)
        self.active_jobs: Dict[str, JobContext] = {}
        self.loaded_model_id = None
        self.model_config = None
        self.current_model = None
        self.p2p_session = None

        # Hardware Specs & Dynamic Limits
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            self.max_vram_usage_mb = int(self.vram_total_mb * MAX_VRAM_USAGE_PERCENT)
            self.evict_threshold_mb = int(self.vram_total_mb * EVICT_THRESHOLD_PERCENT)
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb / 1024:.2f} GB")
            print(f"   ⚙️ Limits set to: Max {self.max_vram_usage_mb / 1024:.2f} GB (95%), Evict at {self.evict_threshold_mb / 1024:.2f} GB (85%)")
        else:
            self.vram_total_mb = 32000
            self.max_vram_usage_mb = int(self.vram_total_mb * MAX_VRAM_USAGE_PERCENT)
            self.evict_threshold_mb = int(self.vram_total_mb * EVICT_THRESHOLD_PERCENT)
            print(f"⚠️ No GPU detected, using simulated RAM limits: {self.vram_total_mb / 1024:.2f} GB")
        sys.stdout.flush()

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

    async def start_p2p_server(self):
        # FIX: Set client_max_size to None (Unlimited)
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

    def _log_vram_status(self, msg=""):
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"   🔋 VRAM Status: {msg} | Allocated: {alloc_mb:.0f}MB ({alloc_mb/1024:.2f}GB) / Reserved: {reserved_mb:.0f}MB ({reserved_mb/1024:.2f}GB)")
            sys.stdout.flush()

    async def _evict_layers(self, current_job_id):
        if not torch.cuda.is_available():
            return
        allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
        if allocated_mb < self.evict_threshold_mb:
            return

        print(f"   🧹 Approaching VRAM limit ({allocated_mb/1024:.2f}GB / {self.max_vram_usage_mb/1024:.2f}GB). Starting eviction...")
        sys.stdout.flush()
        ctx = self.active_jobs.get(current_job_id)
        required_keys = ctx.required_layers if ctx else set()
        keys_to_evict = []
        keys_to_evict.extend([k for k in self.loader.loaded_cache if k not in required_keys])

        freed_count = 0
        for key in keys_to_evict:
            if key in self.loader.loaded_cache:
                self.loader.unload_layer(key)
                freed_count += 1

        if freed_count > 0:
            torch.cuda.empty_cache()
            gc.collect()
            new_alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"   ♻️ Evicted {freed_count} items. Current VRAM: {new_alloc_mb/1024:.2f}GB")
            sys.stdout.flush()

    async def _ensure_layer_loaded(self, job_ctx, loader_func, *args, **kwargs):
        await self._evict_layers(job_ctx.job_id)
        layer = await loader_func(*args, **kwargs)

        loaded_key = None
        for k, v in self.loader.loaded_cache.items():
            if v is layer:
                loaded_key = k
                break

        if loaded_key:
            job_ctx.required_layers.add(loaded_key)

        self._log_vram_status(f"After loading {loaded_key}")
        return layer

    async def _send_p2p(self, url, payload):
        job_id = payload.get('job_id', 'unknown')
        msg_type = payload.get('type', 'unknown')

        print(f"📤 [P2P] Sending {msg_type} for job {job_id[:8] if len(job_id) > 8 else job_id} to {url}")
        sys.stdout.flush()

        my_p2p = self.get_p2p_url().rstrip("/")
        target_base = url.replace("/tensor_in", "").rstrip("/")

        if my_p2p == target_base:
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

        if self.p2p_session is None or self.p2p_session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            self.p2p_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        print(f"   🌐 Sending over network to {url}")
        sys.stdout.flush()
        for attempt in range(3):
            try:
                async with self.p2p_session.post(url, json=payload) as resp:
                    if resp.status == 200:
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

    # --- JOB EXECUTION HANDLERS ---

    async def handle_dense_task(self, job_ctx, layer_idx, prompt, next_hop_url):
        try:
            print(f"🔵 [DENSE] Processing job {job_ctx.job_id}, layer_idx={layer_idx}")
            sys.stdout.flush()

            # Lazy Load
            if layer_idx == -1: # Embeddings
                 embeddings = await self._ensure_layer_loaded(job_ctx, self.loader.load_embeddings, job_ctx.model_id)
                 tokenizer = AutoTokenizer.from_pretrained(job_ctx.model_id)
                 inputs = tokenizer(prompt, return_tensors="pt")
                 tensor = inputs["input_ids"].to(self.device)
                 result = embeddings(tensor)

            else: # Actual Dense Layer or LM Head
                 layer = await self._ensure_layer_loaded(job_ctx, self.loader.load_dense_layer, job_ctx.model_id, layer_idx)
                 queue = job_ctx.get_layer_input_queue(layer_idx)
                 tensor = await queue.get()

                 with torch.no_grad():
                     result = layer(tensor)

            # Forward result
            if next_hop_url:
                # Increment layer index for next stage
                payload = {
                    "job_id": job_ctx.job_id,
                    "type": "input",
                    "layer_idx": layer_idx + 1,
                    "tensor": self._encode_tensor(result)
                }
                await self._send_p2p(next_hop_url, payload)
            else:
                print(f"✅ Dense Layer {layer_idx} complete. No next hop.")

        except Exception as e:
            print(f"❌ Error in Dense Task: {e}")
            traceback.print_exc()

    async def handle_router_task(self, job_ctx, layer_idx, next_hop_url, expert_map):
        """
        PROPER MOE ROUTER LOGIC:
        1. Compute routing weights.
        2. Dispatch masked tensors to assigned experts.
        3. Await expert results.
        4. Aggregate weighted sum.
        5. Forward to next layer.
        """
        try:
            print(f"🟢 [ROUTER] Processing job {job_ctx.job_id}, layer_idx={layer_idx}")
            print(f"    Expert Map: {expert_map}")
            sys.stdout.flush()

            # Lazy Load Router
            router = await self._ensure_layer_loaded(job_ctx, self.loader.load_moe_router, job_ctx.model_id, layer_idx)

            # Wait for input tensor
            queue = job_ctx.get_layer_input_queue(layer_idx)
            tensor = await queue.get()

            # Compute Routing
            with torch.no_grad():
                router_output = router(tensor)
                # Top-1 routing for simplicity and standard MoE behavior
                # weights: (B, S, E), indices: (B, S, 1)
                routing_weights = torch.softmax(router_output, dim=-1)
                topk_weights, topk_indices = torch.topk(routing_weights, k=1, dim=-1)
                topk_indices = topk_indices.squeeze(-1) # Shape: (B, S)

            # Dispatch to Experts
            dispatched_experts = set(topk_indices.flatten().tolist())
            pending_futures = []

            print(f"   🏃 Dispatching to {len(dispatched_experts)} experts based on routing...")
            sys.stdout.flush()

            for expert_id_str, target_url in expert_map.items():
                expert_id = int(expert_id_str)

                # Optimization: Don't dispatch to experts not selected in the batch
                if expert_id not in dispatched_experts:
                    continue

                # Create Future to wait for result
                key = (layer_idx, expert_id)
                loop = asyncio.get_event_loop()
                future = loop.create_future()
                job_ctx.pending_expert_requests[key] = future
                pending_futures.append(future)

                # Prepare Input Slice (Masked)
                # We send the masked input. Zeros go to the expert (waste of compute but safe structure)
                mask = (topk_indices == expert_id).unsqueeze(-1).float()
                input_slice = tensor * mask

                payload = {
                    "job_id": job_ctx.job_id,
                    "type": "input",
                    "layer_idx": layer_idx,
                    "expert_idx": expert_id,
                    "tensor": self._encode_tensor(input_slice)
                }
                await self._send_p2p(target_url, payload)

            # Wait for all dispatched experts to finish
            if pending_futures:
                print(f"   ⏳ Waiting for {len(pending_futures)} expert responses...")
                sys.stdout.flush()
                await asyncio.gather(*pending_futures, return_exceptions=True)
                print(f"   ✓ All experts finished.")
                sys.stdout.flush()

            # Aggregate Results (Weighted Sum)
            # Initialize output with zeros
            final_output = torch.zeros_like(tensor)

            for expert_id_str, target_url in expert_map.items():
                expert_id = int(expert_id_str)
                key = (layer_idx, expert_id)

                if key in job_ctx.pending_expert_requests:
                    expert_output = job_ctx.pending_expert_requests[key].result()

                    # Weight the output
                    # Re-create mask for weighting
                    mask = (topk_indices == expert_id).unsqueeze(-1).float()
                    weight = topk_weights * mask

                    final_output = final_output + (expert_output * weight)

            # Cleanup Pending Requests
            for key in list(job_ctx.pending_expert_requests.keys()):
                del job_ctx.pending_expert_requests[key]

            # Forward aggregated tensor to next layer
            if next_hop_url:
                 payload = {
                    "job_id": job_ctx.job_id,
                    "type": "input",
                    "layer_idx": layer_idx + 1, # Move to next logical layer index
                    "tensor": self._encode_tensor(final_output)
                 }
                 await self._send_p2p(next_hop_url, payload)
                 print(f"   ➡️ Forwarded aggregated tensor to next layer")

        except Exception as e:
            print(f"❌ Error in Router Task: {e}")
            traceback.print_exc()

    async def handle_expert_task(self, job_ctx, layer_idx, expert_idx, return_url):
        try:
            print(f"🟡 [EXPERT] Processing expert {expert_idx} for job {job_ctx.job_id}, layer_idx={layer_idx}")
            sys.stdout.flush()

            # Lazy Load Expert
            expert = await self._ensure_layer_loaded(job_ctx, self.loader.load_moe_expert, job_ctx.model_id, layer_idx, expert_idx)

            # Wait for input tensor (from Router)
            queue = job_ctx.get_expert_queue(layer_idx, expert_idx)
            tensor = await queue.get()

            # Run Expert
            with torch.no_grad():
                result = expert(tensor)

            # Send result back
            # Note: This goes back to the Router's P2P endpoint or to the URL provided by Scheduler
            # The Router will handle aggregation.

            payload = {
                "job_id": job_ctx.job_id,
                "type": "expert_result",
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "tensor": self._encode_tensor(result)
            }
            await self._send_p2p(return_url, payload)
            print(f"   ✅ Expert {expert_idx} finished.")

        except Exception as e:
            print(f"❌ Error in Expert Task: {e}")
            traceback.print_exc()

    async def _get_context(self, job_id, create=False):
        async with self._context_lock:
            if job_id not in self.active_jobs:
                if create:
                    self.active_jobs[job_id] = JobContext(job_id, "")
                else:
                    return None
            return self.active_jobs[job_id]

    async def process_ingress_data(self, data):
        """Internal handler for incoming tensor data (bypass logic)"""
        job_id = data['job_id']
        msg_type = data.get('type', 'input')

        print(f"📨 [P2P] Received {msg_type} for job {job_id[:8] if len(job_id) > 8 else job_id}...")
        sys.stdout.flush()

        tensor = self._decode_tensor(data['tensor'])
        print(f"   ✓ Decoded tensor shape: {tensor.shape}")
        sys.stdout.flush()

        ctx = await self._get_context(job_id, create=True)

        if msg_type == 'input':
            expert_idx = data.get('expert_idx')
            layer_idx = data.get('layer_idx')
            target_layer_idx = data.get('target_layer_idx')

            if expert_idx is not None and layer_idx is not None:
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                print(f"   ➡️ Enqueueing input tensor for layer {layer_idx} expert {expert_idx} of job {job_id[:8]}")
                sys.stdout.flush()
                await queue.put(tensor)
                print(f"   ✓ Layer {layer_idx} expert {expert_idx} input enqueued (queue size: {queue.qsize()})")
                sys.stdout.flush()
            elif target_layer_idx is not None:
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

    async def handle_ws_message(self, message):
        data = json.loads(message)
        msg_type = data.get('type')
        job_id = data.get('job_id')

        if msg_type == 'EXECUTE_DENSE':
            asyncio.create_task(self.handle_dense_task(
                await self._get_context(job_id, create=True),
                data.get('layer_idx'),
                data.get('prompt'),
                data.get('next_hop')
            ))

        elif msg_type == 'EXECUTE_ROUTER':
            asyncio.create_task(self.handle_router_task(
                await self._get_context(job_id, create=True),
                data.get('layer_idx'),
                data.get('next_hop'),
                data.get('expert_map')
            ))

        elif msg_type == 'EXECUTE_EXPERT':
             asyncio.create_task(self.handle_expert_task(
                await self._get_context(job_id, create=True),
                data.get('layer_idx'),
                data.get('expert_idx'),
                data.get('return_url')
            ))

    async def run(self):
        print(f"🔌 Connecting to {SCHEDULER_URL}...")
        sys.stdout.flush()
        async with websockets.connect(SCHEDULER_URL) as ws:
            await self.start_p2p_server()

            reg_payload = {
                "type": "REGISTER",
                "vram_total_mb": self.vram_total_mb,
                "p2p_url": self.get_p2p_url()
            }
            await ws.send(json.dumps(reg_payload))
            print("📡 Registration sent")
            sys.stdout.flush()

            async for message in ws:
                await self.handle_ws_message(message)

if __name__ == "__main__":
    worker = MoEWorker()
    asyncio.run(worker.run())
