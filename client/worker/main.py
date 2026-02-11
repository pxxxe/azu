import asyncio
import torch
import json
import sys
import os
import traceback
import aiohttp
import urllib.request
import gc
import numpy as np
from typing import Dict, Optional, Tuple, Set, List
import websockets
from aiohttp import web, ClientSession, TCPConnector, ClientTimeout
from transformers import AutoTokenizer, AutoConfig
from dataclasses import dataclass, field

# --- NEW: Import Rotary Embeddings for v5 Compatibility ---
try:
    from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
except ImportError:
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as MixtralRotaryEmbedding
    except ImportError:
        print("⚠️ Could not import RotaryEmbedding. v5 Compat mode might fail.")
        MixtralRotaryEmbedding = None

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

        self.kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.generated_ids: List[int] = []
        self.token_queue: asyncio.Queue = asyncio.Queue()
        self.done = False

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
        self.loader = LayerLoader(REGISTRY_URL)
        self.device = self.loader.device
        self.dtype = self.loader.dtype

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb} MB | Dtype: {self.dtype}")
        else:
            self.vram_total_mb = 32000
            print("⚠️ No GPU detected, using simulated 32GB RAM")

        # --- LOCKS (ROBUSTNESS) ---
        self._model_lock = asyncio.Lock()   # Prevents model switching race conditions
        self._context_lock = asyncio.Lock() # Prevents job context race conditions

        # --- STATE ---
        self.active_jobs: Dict[str, JobContext] = {}
        self.current_model_id = None

        # Model Components
        self.config = None
        self.embeddings = None
        self.rotary_emb = None # NEW: Global RoPE cache
        self.lm_head = None
        self.tokenizer = None  # FIX: Add tokenizer caching
        self.final_norm = None
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
        # NEW: Handle token loopback
        self.p2p_app.router.add_post('/token_in', self.handle_token_ingress)
        runner = web.AppRunner(self.p2p_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', P2P_PORT)
        await site.start()
        print(f"👂 [P2P] Server listening on :{P2P_PORT} (Binary/High-Perf)")
        sys.stdout.flush()

    async def _get_p2p_session(self):
        """Lazy-load a shared session for P2P traffic to prevent socket exhaustion."""
        if self.p2p_session is None or self.p2p_session.closed:
            # High limits, tuned timeouts
            timeout = ClientTimeout(total=60, sock_read=30, sock_connect=10)
            connector = TCPConnector(limit=0, ttl_dns_cache=300)
            self.p2p_session = ClientSession(connector=connector, timeout=timeout)
        return self.p2p_session

    async def _get_context(self, job_id, create=False):
        async with self._context_lock:
            if job_id not in self.active_jobs:
                if create:
                    self.active_jobs[job_id] = JobContext(job_id)
                else:
                    return None
            return self.active_jobs[job_id]

    # --- NEW: Token Ingress Handler (Autoregression Loop) ---
    async def handle_token_ingress(self, request):
        try:
            data = await request.json()
            job_id = data.get("job_id")
            token_id = data.get("token_id")

            ctx = await self._get_context(job_id, create=False)
            if ctx:
                await ctx.token_queue.put(token_id)
                return web.Response(text="OK")
            return web.Response(status=404, text="Job context not found")
        except Exception as e:
            print(f"❌ [P2P] Error handling token ingress: {e}")
            return web.Response(status=500, text=str(e))

    # --- NEW: Binary Ingress Handler ---
    async def handle_tensor_ingress(self, request):
        try:
            # 1. Read Metadata from Headers
            headers = request.headers
            job_id = headers.get("x-job-id")
            if not job_id: return web.Response(status=400, text="Missing job_id")

            msg_type = headers.get("x-msg-type", "input")
            dtype_str = headers.get("x-dtype", "float16")
            shape = json.loads(headers.get("x-shape", "[]"))

            # 2. Read RAW Binary Body
            data = await request.read()

            # 3. Reconstruct Tensor (Zero-Copy-ish)
            dtype = getattr(np, dtype_str)
            # copy() is needed to make the array writable/contiguous for torch
            arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()

            tensor = torch.from_numpy(arr).to(self.device).to(self.dtype)

            # 4. Route
            ctx = await self._get_context(job_id, create=True)

            if msg_type == 'input':
                # Parse optional routing headers
                expert_idx = headers.get("x-expert-idx")
                layer_idx = headers.get("x-layer-idx")
                target_layer_idx = headers.get("x-target-layer-idx")

                if expert_idx is not None and layer_idx is not None:
                    queue = ctx.get_expert_queue(int(layer_idx), int(expert_idx))
                    await queue.put(tensor)
                elif target_layer_idx is not None:
                    queue = ctx.get_layer_input_queue(int(target_layer_idx))
                    await queue.put(tensor)

            elif msg_type == 'expert_result':
                expert_idx = int(headers.get("x-expert-idx"))
                layer_idx = int(headers.get("x-layer-idx"))
                key = (layer_idx, expert_idx)
                if key in ctx.pending_expert_requests:
                    future = ctx.pending_expert_requests[key]
                    if not future.done():
                        future.set_result(tensor)

            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ [P2P] Error handling ingress: {e}")
            traceback.print_exc()
            return web.Response(status=500, text=str(e))

    # --- NEW: Binary Egress (Sender) ---
    async def _send_p2p(self, url, payload_meta, tensor: torch.Tensor):
        """
        Sends tensor as raw binary body with metadata in headers.
        Includes LOOPBACK OPTIMIZATION from original code.
        """
        # 1. Loopback Check
        my_p2p = self.get_p2p_url().rstrip("/")
        target_base = url.replace("/tensor_in", "").rstrip("/")

        # Prepare Metadata
        # --- FIX: Handle BFloat16 -> Numpy conversion ---
        # Numpy crashes on BFloat16. We must detach().cpu().float() before numpy().
        np_tensor = tensor.detach().cpu().float().contiguous().numpy()
        dtype_str = str(np_tensor.dtype)
        shape_json = json.dumps(list(np_tensor.shape))

        headers = {
            "x-job-id": payload_meta["job_id"],
            "x-msg-type": payload_meta.get("type", "input"),
            "x-dtype": dtype_str,
            "x-shape": shape_json
        }

        # Add specific routing fields to headers
        if "expert_idx" in payload_meta: headers["x-expert-idx"] = str(payload_meta["expert_idx"])
        if "layer_idx" in payload_meta: headers["x-layer-idx"] = str(payload_meta["layer_idx"])
        if "target_layer_idx" in payload_meta:
            if payload_meta["target_layer_idx"] is not None:
                headers["x-target-layer-idx"] = str(payload_meta["target_layer_idx"])

        if my_p2p == target_base:
            # Short-circuit: Mock a request object for local processing
            # We skip serialization for loopback to save even more time
            try:
                # Direct injection into context queues (Skipping HTTP layer entirely)
                ctx = await self._get_context(payload_meta['job_id'], create=True)
                msg_type = payload_meta.get('type', 'input')

                # --- FIX: Ensure tensor matches model dtype even in loopback ---
                tensor = tensor.to(self.dtype)

                if msg_type == 'input':
                    e_idx = payload_meta.get("expert_idx")
                    l_idx = payload_meta.get("layer_idx")
                    t_idx = payload_meta.get("target_layer_idx")

                    if e_idx is not None and l_idx is not None:
                        await ctx.get_expert_queue(l_idx, e_idx).put(tensor)
                    elif t_idx is not None:
                        await ctx.get_layer_input_queue(t_idx).put(tensor)

                elif msg_type == 'expert_result':
                     e_idx = payload_meta.get("expert_idx")
                     l_idx = payload_meta.get("layer_idx")
                     key = (l_idx, e_idx)
                     if key in ctx.pending_expert_requests:
                        future = ctx.pending_expert_requests[key]
                        if not future.done(): future.set_result(tensor)
                return
            except Exception as e:
                print(f"❌ Local P2P Error: {e}")
                return

        # 2. Network Transfer (Binary)
        session = await self._get_p2p_session()
        data_bytes = np_tensor.tobytes()

        for attempt in range(3):
            try:
                # Content-Type application/octet-stream is standard for binary
                headers["Content-Type"] = "application/octet-stream"
                async with session.post(url, data=data_bytes, headers=headers) as resp:
                    if resp.status == 200: return
                    else: print(f"   ⚠️ P2P Handshake Rejected {resp.status} from {url}")
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

            self.config = None
            self.embeddings = None
            self.rotary_emb = None
            self.lm_head = None
            self.final_norm = None
            self.tokenizer = None  # FIX: Clear tokenizer cache
            self.dense_layers.clear()
            self.moe_routers.clear()
            self.moe_experts.clear()
            self.active_jobs.clear()
            gc.collect()
            torch.cuda.empty_cache()

            # --- V5 COMPAT: Initialize Global Rotary Embeddings ---
            try:
                print(f"   ⚙️ Initializing Rotary Embeddings for {model_id}...")
                config_path, config_url = self.loader._get_paths(model_id, "config.json")
                if not config_path.exists():
                      await self.loader._download(config_url, config_path)

                self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

                if MixtralRotaryEmbedding:
                    # TRANSFORMERS 5.0+ FIX: Pass config object instead of individual params
                    # The new API signature is: MixtralRotaryEmbedding(config=config, device=device)

                    self.rotary_emb = MixtralRotaryEmbedding(
                        config=self.config,
                        device=self.device
                    ).to(self.dtype)

                    print(f"   ✅ RoPE Initialized (head_dim={self.config.hidden_size // self.config.num_attention_heads}, "
                          f"base={getattr(self.config, 'rope_theta', 10000.0)})")

            except Exception as e:
                print(f"   ⚠️ Failed to init RoPE: {e}")
                traceback.print_exc()
                # Set to None so code can continue without RoPE (will fail later, but more gracefully)
                self.rotary_emb = None

            self.current_model_id = model_id

    async def _load_tokenizer(self, model_id):
        """Load tokenizer from registry only (never HuggingFace)"""
        if not self.tokenizer:
            print(f"📖 Loading Tokenizer for {model_id}...")
            sanitized = model_id.replace("/", "_")
            config_path = self.loader.cache_dir / sanitized

            # All possible tokenizer files
            tokenizer_files = {
                "config.json": True,  # Required
                "tokenizer_config.json": False,
                "tokenizer.model": False,  # SentencePiece (Llama/Mistral/Mixtral)
                "vocab.json": False,
                "merges.txt": False,
                "tokenizer.json": False,
                "special_tokens_map.json": False,
                "added_tokens.json": False,
                "generation_config.json": False,
            }

            downloaded_files = []
            failed_required = []

            for filename, is_required in tokenizer_files.items():
                file_path, file_url = self.loader._get_paths(model_id, filename)
                try:
                    await self.loader._download(file_url, file_path, quiet=(filename != "config.json"))
                    if file_path.exists():
                        size = file_path.stat().st_size
                        if size == 0:
                            if is_required:
                                raise RuntimeError(f"{filename} is empty")
                        else:
                            downloaded_files.append(filename)
                            if filename.endswith('.model'):
                                print(f"      ✅ {filename} ({size/1024:.1f} KB)")
                except Exception as e:
                    if is_required:
                        failed_required.append(filename)

            if failed_required:
                raise RuntimeError(f"Missing required files: {failed_required}")

            print(f"      📦 Downloaded {len(downloaded_files)} files")

            try:
                # CRITICAL: token=None prevents HuggingFace fallback
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(config_path),
                    token=None,
                    trust_remote_code=True,
                    local_files_only=True
                )

                # Test tokenizer works
                test_ids = self.tokenizer.encode("Hello, world!")
                test_decoded = self.tokenizer.decode(test_ids)
                if '<unk>' in test_decoded or len(test_ids) == 1:
                    raise RuntimeError(
                        f"Tokenizer broken. Encoded: {test_ids}, Decoded: '{test_decoded}'. "
                        f"tokenizer.model likely missing."
                    )

                print(f"      ✅ Tokenizer OK (vocab_size={len(self.tokenizer)})")

            except Exception as e:
                print(f"\n❌ TOKENIZER LOAD FAILED: {e}")
                print(f"Downloaded: {downloaded_files}")
                print(f"Registry must serve ALL tokenizer files from {self.loader.registry_url}\n")
                raise

    def _prepare_inputs(self, hidden_states, past_kv):
        """
        V5 COMPAT: Pre-compute position embeddings and mask.
        Returns: (position_embeddings, attention_mask)
        """
        seq_len = hidden_states.shape[1]
        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[2]

        # 1. Position IDs
        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).view(-1, seq_len)

        # 2. Rotary Embeddings (Cos, Sin)
        # transformers v5/v4.36+ expects (cos, sin) tuple
        position_embeddings = None
        if self.rotary_emb:
            # We call the RoPE module. It returns cos, sin
            # Note: The signature of forward might vary, usually it takes (x, seq_len) or (x, position_ids)
            # MixtralRotaryEmbedding.forward(x, position_ids) -> cos, sin
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 3. Attention Mask
        # For inference (ragged/single batch), usually 4D: (batch, 1, q_len, kv_len)
        # We need to mask out future tokens if seq_len > 1 (prefill)
        # If seq_len == 1 (decode), it's all ones.
        total_len = past_len + seq_len

        # Create causal mask
        # (1, 1, seq_len, total_len)
        mask = torch.full(
            (1, 1, seq_len, total_len),
            0, # 0 means unmasked in some versions, but standard is min_dtype for masked
            dtype=self.dtype,
            device=self.device
        )

        # If prefill (seq_len > 1), we need causality (triangular)
        if seq_len > 1:
             # Standard causal mask: -inf above diagonal
             causal_mask = torch.triu(
                 torch.full((seq_len, total_len), float("-inf"), device=self.device),
                 diagonal=1
             )
             mask = mask + causal_mask.unsqueeze(0).unsqueeze(0)

        # NOTE: Transformers often expects 0 for "attend", -inf for "mask"
        # Since we initialized with 0, we are good.

        return position_embeddings, mask, position_ids

    async def process_dense(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg.get('layer_idx', -1)
        next_hop = msg.get('next_hop')
        next_layer_idx = msg.get('next_layer_idx')
        is_first = msg.get('is_first', False)
        is_last = msg.get('is_last', False)
        max_tokens = msg.get('max_tokens', 50)
        first_node_endpoint = msg.get('first_node_endpoint')

        print(f"🔵 [DENSE] Processing job {job_id[:8]}, layer_idx={layer_idx}")

        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        while not ctx.done:
            try:
                # --- JIT Embedding ---
                if is_first:
                    if not self.embeddings:
                        print(f"📦 Loading embeddings...")
                        self.embeddings = await self.loader.load_embeddings(model_id)
                        await self._load_tokenizer(model_id)

                    input_tensor = None

                    if not ctx.generated_ids and msg.get('input'):
                        print(f"   📝 Encoding Prompt...")
                        ctx.kv_cache.clear()
                        input_tensor = self.tokenizer.encode(msg['input'], return_tensors='pt').to(self.device)
                        msg['input'] = None

                    # 2. Feedback Token (Loop)
                    else:
                        try:
                            # Wait for token from the last worker
                            token_id = await asyncio.wait_for(ctx.token_queue.get(), timeout=P2P_TIMEOUT)
                            input_tensor = torch.tensor([[token_id]], device=self.device)
                        except asyncio.TimeoutError:
                            if ctx.done: break # Normal exit
                            print(f"❌ [Job {job_id[:8]}] Timeout waiting for loopback token")
                            break

                    hidden_states = self.embeddings(input_tensor)

                else:
                    # Middle Layer Input
                    queue = ctx.get_layer_input_queue(layer_idx)
                    try:
                        # print(f"   ⏳ Waiting for input tensor...")
                        hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Timeout waiting for input")
                        break

                hidden_states = hidden_states.to(self.dtype)

                # --- JIT Dense Layer & KV Cache ---
                layer_out = hidden_states
                if layer_idx != -1:
                    if layer_idx not in self.dense_layers:
                        print(f"📦 Loading dense layer {layer_idx}...")
                        self.dense_layers[layer_idx] = await self.loader.load_dense_layer(model_id, layer_idx)

                    # Get KV Cache for this layer
                    past_kv = ctx.kv_cache.get(layer_idx, None)

                    # --- V5 FIX: Prepare Positional Args ---
                    pos_emb, attn_mask, pos_ids = self._prepare_inputs(hidden_states, past_kv)

                    with torch.no_grad():
                        out = self.dense_layers[layer_idx](
                            hidden_states,
                            past_key_values=past_kv,
                            use_cache=True,
                            # Explicitly pass V5 required args
                            position_embeddings=pos_emb,
                            attention_mask=attn_mask,
                            position_ids=pos_ids
                        )

                        if isinstance(out, tuple):
                            layer_out = out[0]
                            # Update KV Cache
                            if len(out) > 1:
                                ctx.kv_cache[layer_idx] = out[1]
                        else:
                            layer_out = out

                # --- JIT Head & Decode ---
                if is_last:
                    if not self.lm_head:
                        print(f"🔚 Loading LM Head...")
                        self.lm_head = await self.loader.load_lm_head(model_id)
                        self.final_norm = await self.loader.load_final_norm(model_id)
                        await self._load_tokenizer(model_id)

                    with torch.no_grad():
                        if self.final_norm:
                            latents = self.final_norm(layer_out[:, -1, :])
                        else:
                            latents = layer_out[:, -1, :]
                        logits = self.lm_head(latents)
                        token_id = torch.argmax(logits, dim=-1).item()

                    # Record Generation
                    ctx.generated_ids.append(token_id)
                    gen_text = self.tokenizer.decode([token_id])
                    print(f"   ✨ Gen: {gen_text}")

                    # Check Stop Conditions
                    stop = False
                    reason = ""
                    if len(ctx.generated_ids) >= max_tokens:
                        stop = True
                        reason = "max_tokens"
                    elif token_id == self.tokenizer.eos_token_id:
                        stop = True
                        reason = "EOS"

                    if stop:
                        full_text = self.tokenizer.decode(ctx.generated_ids)
                        print(f"   🎉 GENERATION COMPLETE ({reason}): {len(ctx.generated_ids)} tokens")
                        await ws.send(json.dumps({
                            "type": "RESULT",
                            "job_id": job_id,
                            "status": "completed",
                            "output": full_text
                        }))
                        ctx.done = True
                        del self.active_jobs[job_id]
                        return
                    else:
                        if first_node_endpoint:
                            session = await self._get_p2p_session()
                            try:
                                target = f"{first_node_endpoint}/token_in"
                                async with session.post(target, json={
                                    "job_id": job_id,
                                    "token_id": token_id
                                }) as resp:
                                    if resp.status != 200:
                                        print(f"   ⚠️ Loopback failed: {resp.status}")
                            except Exception as e:
                                print(f"   ❌ Loopback error: {e}")

                # --- Forward to Next Worker ---
                elif next_hop:
                    # print(f"   ➡️ Forwarding to {next_hop}")
                    await self._send_p2p(next_hop, {
                        "job_id": job_id,
                        "type": "input",
                        "target_layer_idx": next_layer_idx
                    }, layer_out)

            except Exception as e:
                print(f"❌ Error in dense loop: {e}")
                traceback.print_exc()
                break

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

        while not ctx.done:
            try:
                queue = ctx.get_layer_input_queue(layer_idx)
                try:
                    # print(f"   ⏳ Waiting for input tensor...")
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                except asyncio.TimeoutError:
                    if ctx.done: break
                    print(f"❌ [Job {job_id[:8]}] Router input timeout")
                    break

                hidden_states = hidden_states.to(self.dtype)

                # =========================================================
                # STEP 1: Execute Shared Attention & Norms
                # =========================================================
                shared_layer = await self.loader.load_moe_shared(model_id, layer_idx)
                past_kv = ctx.kv_cache.get(layer_idx, None)

                # --- V5 FIX: Prepare Positional Args ---
                pos_emb, attn_mask, pos_ids = self._prepare_inputs(hidden_states, past_kv)

                # A. Input Residual & Norm
                residual = hidden_states
                if hasattr(shared_layer, 'input_layernorm'):
                    hidden_states = shared_layer.input_layernorm(hidden_states)

                # B. Self Attention (V5 SAFE CALL)
                if hasattr(shared_layer, 'self_attn'):
                    # Explicitly pass the required args for V5
                    attn_out, new_kv = shared_layer.self_attn(
                        hidden_states,
                        position_embeddings=pos_emb,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        past_key_values=past_kv,
                        use_cache=True
                    )
                    hidden_states = attn_out
                    ctx.kv_cache[layer_idx] = new_kv

                # C. First Residual Connection
                hidden_states = residual + hidden_states

                # D. Save state for Post-MoE Residual
                post_attn_residual = hidden_states

                # E. Post-Attention Norm (Pre-MoE Norm)
                if hasattr(shared_layer, 'post_attention_layernorm'):
                    hidden_states = shared_layer.post_attention_layernorm(hidden_states)

                # =========================================================
                # STEP 2: Router logic
                # =========================================================

                # --- JIT Router ---
                if layer_idx not in self.moe_routers:
                    print(f"📦 Loading router {layer_idx}...")
                    self.moe_routers[layer_idx] = await self.loader.load_moe_router(model_id, layer_idx)

                with torch.no_grad():
                    logits = self.moe_routers[layer_idx](hidden_states)
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

                    # Dispatch to experts in PARALLEL
                    send_tasks.append(asyncio.create_task(self._send_p2p(f"{target_url}/tensor_in", {
                        "job_id": job_id,
                        "type": "input",
                        "layer_idx": layer_idx,
                        "expert_idx": expert_idx
                    }, sliced))) # Send tensor as arg

                if send_tasks:
                    await asyncio.gather(*send_tasks)

                pending = list(local_pending.values())
                if pending:
                    # print(f"   ⏳ Waiting for {len(pending)} expert results...")
                    try:
                        await asyncio.wait_for(asyncio.gather(*pending), timeout=P2P_TIMEOUT)
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Expert results timeout")
                        break

                # =========================================================
                # STEP 3: Merge & Final Residual
                # =========================================================
                batch, seq, hidden = hidden_states.shape
                moe_output = torch.zeros((batch, seq, hidden), dtype=self.dtype, device=self.device)
                top_weights_dev = routing_weights.to(self.device)
                top_indices_dev = selected_indices.to(self.device)

                with torch.no_grad():
                    for expert_idx, future in local_pending.items():
                        if not future.done(): continue
                        res = future.result().to(self.device).to(self.dtype)
                        mask = (top_indices_dev == expert_idx)
                        rows, cols, k_idx = torch.where(mask)
                        w = top_weights_dev[rows, cols, k_idx].unsqueeze(-1)
                        moe_output.index_put_((rows, cols), res * w, accumulate=True)

                # Final Residual: (Attn Output) + (MoE Output)
                final_output = post_attn_residual + moe_output

                for expert_idx in local_pending:
                    ctx.pending_expert_requests.pop((layer_idx, expert_idx), None)

                if next_hop:
                    # print(f"   ➡️ Forwarding to {next_hop}")
                    await self._send_p2p(next_hop, {
                        "job_id": job_id,
                        "type": "input",
                        "target_layer_idx": next_layer_idx
                    }, final_output) # Send tensor as arg
            except Exception as e:
                print(f"❌ Error in router loop: {e}")
                traceback.print_exc()
                break

    async def process_moe_expert(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        print(f"🟡 [EXPERT] Processing expert {expert_idx} (Layer {layer_idx})")
        await self._ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)

        while not ctx.done:
            try:
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                try:
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                except asyncio.TimeoutError:
                    if ctx.done: break
                    print(f"⏭️ [Job {job_id[:8]}] Expert {expert_idx} not used (timeout)")
                    break

                # --- FIX: Precision Guard ---
                hidden_states = hidden_states.to(self.dtype)

                # --- JIT Expert ---
                cache_key = (layer_idx, expert_idx)
                if cache_key not in self.moe_experts:
                    print(f"📦 Loading expert {expert_idx}...")
                    self.moe_experts[cache_key] = await self.loader.load_moe_expert(model_id, layer_idx, expert_idx)

                with torch.no_grad():
                    output = self.moe_experts[cache_key](hidden_states)

                await self._send_p2p(f"{return_url}/tensor_in", {
                    "job_id": job_id,
                    "type": "expert_result",
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx
                }, output) # Send tensor as arg
            except Exception as e:
                 print(f"❌ Error in expert loop: {e}")
                 break

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
