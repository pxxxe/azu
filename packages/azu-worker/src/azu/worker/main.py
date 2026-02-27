"""
Main entry point for the worker node.
Coordinates P2P networking, job processing, and scheduler communication.

Two modes controlled by WORKER_MODE env var:

  persistent  (default)
    Maintains a long-lived WebSocket to the scheduler.
    Scheduler pushes JOB_START / EXECUTE_* messages over the socket.
    Results sent back over the same socket as RESULT messages.
    Heartbeat runs every second.

  serverless
    No persistent connection.  On startup the worker:
      1. Starts the P2P HTTP server on :8003 (unchanged)
      2. Resolves its public P2P URL
      3. Calls POST /workers on the scheduler to upsert its endpoint record
      4. Polls scheduler via GET /worker/poll/{worker_id} for control messages
         (instead of scheduler POSTing to worker's proxy URL, which fails with 403)
      5. When a job completes, POSTs result to POST /worker/result
      6. Sends periodic heartbeats to POST /worker/heartbeat
      7. Idle watchdog terminates worker after IDLE_TIMEOUT seconds without a job

The /control endpoint is still added to the existing P2PServer aiohttp app so
there is no second HTTP server or second port.  Tensor transfer (P2P)
is completely untouched.
"""

import asyncio
import json
import os
import sys
import traceback
import urllib.request
from typing import Dict, Optional

import torch
import aiohttp
import websockets
from transformers import DynamicCache
import time

from azu.worker.config import (
    SCHEDULER_URL,
    SCHEDULER_HTTP_URL,
    WORKER_MODE,
    REGISTRY_URL,
    P2P_PORT,
    DEFAULT_CPU_VRAM_MB,
    PAYMENT_PROVIDER,
    WAKE_URL,
    IDLE_TIMEOUT,
)
from azu.worker.layer_loader import LayerLoader
from azu.worker.model_manager import ModelManager
from azu.worker.job_context import JobContext
from azu.worker.p2p_server import P2PServer

# Import wallet module for payments
try:
    from azu.worker.wallet import get_worker_wallet
    HAS_WALLET = True
except ImportError:
    HAS_WALLET = False


class MoEWorker:
    """
    Main worker class that coordinates all worker operations.
    Handles P2P networking, job processing, and scheduler communication.
    """

    def __init__(self):
        # Initialize wallet for payments FIRST
        self.wallet = None
        self.payment_address = None
        if HAS_WALLET:
            try:
                self.wallet = get_worker_wallet(provider_type=PAYMENT_PROVIDER)
                self.payment_address = self.wallet.address
                print(f"💰 Worker payment address: {self.payment_address}")
            except Exception as e:
                print(f"⚠️ Failed to initialize wallet: {e}")

        # Initialize components
        self.loader = LayerLoader(REGISTRY_URL)
        self.model_manager = ModelManager(self.loader)
        self.device = self.loader.device
        self.dtype = self.loader.dtype

        # Detect GPU
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.vram_total_mb = int(props.total_memory / (1024**2))
            print(f"🎮 GPU Detected: {props.name} | VRAM: {self.vram_total_mb} MB | Dtype: {self.dtype}")
        else:
            self.vram_total_mb = DEFAULT_CPU_VRAM_MB
            print("⚠️ No GPU detected, using simulated 32GB RAM")

        # Stable worker ID — generated once per process.  Persists for the
        # lifetime of the container so the scheduler can correlate /worker/ready
        # callbacks with the registered worker_id.
        self._worker_id = "Worker_" + os.urandom(4).hex()

        # Lock for thread-safe operations
        self._context_lock = asyncio.Lock()

        # Active jobs
        self.active_jobs: Dict[str, JobContext] = {}

        # P2P session
        self.p2p_session: Optional[aiohttp.ClientSession] = None

        # P2P server
        self.p2p_server = P2PServer(
            get_p2p_url_fn=self.get_p2p_url,
            get_context_fn=self._get_context,
            get_p2p_session_fn=self._get_p2p_session,
            device=self.device,
            dtype=self.dtype
        )

        # ── serverless idle tracking (new) ───────────────────────────────────────
        self._last_job_time = time.time() if WORKER_MODE == "serverless" else None

        sys.stdout.flush()

    def get_p2p_url(self) -> Optional[str]:
        """Get this worker's P2P URL."""
        from azu.worker.config import P2P_PUBLIC_URL, P2P_URL_TEMPLATE

        if P2P_PUBLIC_URL:
            return P2P_PUBLIC_URL.strip("/")

        if P2P_URL_TEMPLATE:
            try:
                return P2P_URL_TEMPLATE.format_map(os.environ).strip("/")
            except KeyError as e:
                print(f"⚠️ P2P_URL_TEMPLATE references missing env var: {e}. Falling back.")

        if os.environ.get("P2P_EXPOSE_PORT", "").lower() in ("1", "true", "yes"):
            try:
                ip = urllib.request.urlopen('https://api.ipify.org', timeout=3).read().decode('utf8')
                return f"http://{ip}:{P2P_PORT}"
            except:
                pass

        return None

    async def _get_p2p_session(self) -> aiohttp.ClientSession:
        """Get or create P2P session."""
        if self.p2p_session is None or self.p2p_session.closed:
            timeout = aiohttp.ClientTimeout(total=60, sock_read=30, sock_connect=10)
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            self.p2p_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.p2p_session

    async def _get_context(self, job_id: str, create: bool = False) -> Optional[JobContext]:
        """Get job context."""
        async with self._context_lock:
            if job_id not in self.active_jobs:
                if create:
                    self.active_jobs[job_id] = JobContext(job_id)
                else:
                    return None
            return self.active_jobs[job_id]

    async def _safe_task_wrapper(self, coro, task_name: str):
        """Wrapper for safe async task execution."""
        try:
            await coro
        except Exception as e:
            print(f"❌ TASK ERROR in {task_name}: {e}")
            traceback.print_exc()
            sys.stdout.flush()

    # =========================================================================
    # Message dispatcher — shared by both persistent and serverless paths.
    # Receives a parsed dict and the ws handle (None for serverless).
    # =========================================================================

    async def _dispatch_message(self, msg: dict, ws):
        """Route an incoming control message to the correct handler."""
        msg_type = msg.get('type')
        job_id   = msg.get('job_id', 'unknown')[:8]

        if msg_type == 'JOB_START':
            job_id_full = msg.get('job_id')
            topology    = msg.get('topology', [])
            model_id    = msg.get('model_id')
            auth_token  = msg.get('auth_token')

            print(f"🔗 [Job {job_id}] Received JOB_START, initiating mesh handshake...")

            my_p2p_url = self.get_p2p_url()
            if my_p2p_url:
                my_p2p_url = my_p2p_url.rstrip("/")
                try:
                    session = await self._get_p2p_session()
                    async with session.post(
                        f"{my_p2p_url}/control/job_start",
                        json={
                            "job_id": job_id_full,
                            "model_id": model_id,
                            "topology": topology,
                            "auth_token": auth_token,
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            print(f"🔗 [Job {job_id}] Mesh handshake initiated successfully")
                        else:
                            print(f"⚠️ [Job {job_id}] Mesh handshake initiation failed: {resp.status}")
                except Exception as e:
                    print(f"⚠️ [Job {job_id}] Failed to trigger mesh handshake: {e}")

            # Serverless: report ready so the scheduler unblocks Phase 2 dispatch.
            # The P2P server is already running (started before registration), so
            # p2p_url is known immediately — no cold-start delay.
            if WORKER_MODE == "serverless" and my_p2p_url:
                asyncio.create_task(self._report_ready(job_id_full, my_p2p_url))

        elif msg_type == 'EXECUTE_DENSE':
            asyncio.create_task(self._safe_task_wrapper(
                self._process_dense(msg, ws), f"EXECUTE_DENSE-{job_id}"))
        elif msg_type == 'EXECUTE_ROUTER':
            asyncio.create_task(self._safe_task_wrapper(
                self._process_moe_router(msg, ws), f"EXECUTE_ROUTER-{job_id}"))
        elif msg_type == 'EXECUTE_EXPERT':
            asyncio.create_task(self._safe_task_wrapper(
                self._process_moe_expert(msg, ws), f"EXECUTE_EXPERT-{job_id}"))
        else:
            print(f"⚠️ Unknown message type: {msg_type}")

    # =========================================================================
    # Persistent mode — long-lived WebSocket
    # =========================================================================

    async def heartbeat(self, ws):
        """Send heartbeat to scheduler with VRAM status."""
        while True:
            try:
                if torch.cuda.is_available():
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    free_mb = int(free_bytes / (1024**2))
                else:
                    free_mb = self.vram_total_mb

                await ws.send(json.dumps({
                    "type": "HEARTBEAT",
                    "vram_free_mb": free_mb
                }))
                await asyncio.sleep(1.0)
            except Exception:
                break

    async def _run_persistent(self):
        """Persistent mode: maintain long-lived WebSocket to scheduler."""
        while True:
            try:
                print(f"🔌 Connecting to {SCHEDULER_URL}...")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    p2p_url   = self.get_p2p_url()
                    relay_mode = p2p_url is None

                    if relay_mode:
                        print("📡 No public P2P URL — operating in relay mode (tensors via Scheduler WebSocket)")
                    else:
                        print(f"🌐 P2P URL: {p2p_url}")

                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": {
                            "pubkey":          self._worker_id,
                            "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                            "vram_mb":         self.vram_total_mb,
                            "p2p_url":         p2p_url,
                            "relay_mode":      relay_mode,
                            "capabilities":    ["dense", "moe_router", "moe_expert"],
                            "payment_address": self.payment_address,
                        }
                    }))
                    print(f"✅ Connected & Registered")

                    heartbeat_task = asyncio.create_task(self.heartbeat(ws))

                    async for raw in ws:
                        msg = json.loads(raw)
                        await self._dispatch_message(msg, ws)

                    heartbeat_task.cancel()

            except Exception as e:
                print(f"❌ Connection Error: {e}")
                await asyncio.sleep(5)
            finally:
                if self.p2p_session and not self.p2p_session.closed:
                    await self.p2p_session.close()

    # =========================================================================
    # Serverless mode — HTTP control endpoint + scheduler HTTP callbacks
    # =========================================================================

    def _mount_control_endpoint(self):
        """
        Add POST /control to the existing P2PServer aiohttp app.

        This is the inbound channel for serverless workers.  The scheduler
        calls this endpoint instead of pushing over a WebSocket.  The payload
        shape is identical to WS messages: JOB_START, EXECUTE_DENSE, etc.

        Must be called BEFORE p2p_server.start() so the route is registered
        before the aiohttp runner is set up.
        """
        from aiohttp import web

        async def handle_control(request: web.Request) -> web.Response:
            try:
                msg = await request.json()
                await self._dispatch_message(msg, ws=None)
                return web.Response(text="OK")
            except Exception as e:
                print(f"❌ [Control] Error handling message: {e}")
                traceback.print_exc()
                return web.Response(status=500, text=str(e))

        # p2p_app is created inside P2PServer.start(); we pre-create it here
        # so we can add our route before the runner initialises.
        if self.p2p_server.p2p_app is None:
            self.p2p_server.p2p_app = web.Application(
                client_max_size=self.p2p_server.__class__.__dict__.get(
                    '_P2P_MAX_SIZE', 1024**3
                )
            )

        self.p2p_server.p2p_app.router.add_post('/control', handle_control)

    async def _register_with_scheduler(self, p2p_url: Optional[str]):
        """
        Call POST /workers on the scheduler to upsert this worker's endpoint.

        For serverless workers the endpoint_url is the /control route on the
        worker's own P2P server, since that's where the scheduler will POST
        control messages.
        """
        endpoint_url = f"{p2p_url}/control" if p2p_url else ""

        payload = {
            "worker_id":       self._worker_id,
            "worker_type":     "serverless",
            "endpoint_url":    endpoint_url,
            "vram_mb":         self.vram_total_mb,
            "capabilities":    ["dense", "moe_router", "moe_expert"],
            "platform":        os.getenv("PLATFORM", "self_hosted"),
            "payment_address": self.payment_address,
            "p2p_url":         p2p_url,
            "wake_url":        WAKE_URL,  # NEW: pass wake URL so scheduler can wake us if scaled to 0
        }

        session = await self._get_p2p_session()
        url = f"{SCHEDULER_HTTP_URL}/workers"
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status in (200, 201):
                    print(f"✅ [Serverless] Registered with scheduler at {url}")
                else:
                    body = await resp.text()
                    print(f"⚠️ [Serverless] Registration returned HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            print(f"⚠️ [Serverless] Failed to register with scheduler: {e}")

    async def _send_result_http(self, job_id: str, status: str,
                                 output: Optional[str] = None,
                                 tokens_generated: Optional[int] = None):
        """
        POST a completed job result to the scheduler's /worker/result endpoint.

        Called instead of ws.send_json({"type": "RESULT", ...}) in serverless mode.
        """
        payload = {
            "job_id":           job_id,
            "worker_id":        self._worker_id,
            "status":           status,
            "output":           output,
            "tokens_generated": tokens_generated,
        }
        session = await self._get_p2p_session()
        url = f"{SCHEDULER_HTTP_URL}/worker/result"
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"⚠️ [Serverless] Result POST returned HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            print(f"⚠️ [Serverless] Failed to POST result: {e}")

    async def _report_ready(self, job_id: str, p2p_url: str):
        """
        Call POST /worker/ready on the scheduler once the P2P server is up
        and the worker knows its public URL.  Unblocks the scheduler's Phase 2
        dispatch gate for this job.
        """
        payload = {
            "job_id":    job_id,
            "worker_id": self._worker_id,
            "p2p_url":   p2p_url,
        }
        session = await self._get_p2p_session()
        url = f"{SCHEDULER_HTTP_URL}/worker/ready"
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    print(f"🟢 [Serverless] Reported ready for job {job_id[:8]} at {p2p_url}")
                else:
                    body = await resp.text()
                    print(f"⚠️ [Serverless] /worker/ready returned HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            print(f"⚠️ [Serverless] Failed to report ready: {e}")

    async def _serverless_heartbeat(self):
        """
        Periodic heartbeat for serverless workers via HTTP POST.

        The scheduler doesn't require this for serverless workers (there's no
        live session to keep alive) but it lets the scheduler update its VRAM
        accounting for planning purposes.
        """
        session = await self._get_p2p_session()
        url = f"{SCHEDULER_HTTP_URL}/worker/heartbeat"
        while True:
            try:
                await asyncio.sleep(10.0)
                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_mb = int(free_bytes / (1024**2))
                else:
                    free_mb = self.vram_total_mb

                async with session.post(url, json={
                    "worker_id":   self._worker_id,
                    "vram_free_mb": free_mb,
                }, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    pass  # fire-and-forget; don't crash on failure
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # heartbeat failure is non-fatal

    async def _poll_scheduler(self):
        """
        Poll the scheduler for control messages via long-poll.

        This replaces the push model (scheduler POSTing to worker proxy URL)
        which fails on RunPod LB serverless because proxy URLs return 403.

        The worker calls GET /worker/poll/{worker_id} and holds the request
        for ~29 seconds. The scheduler either returns a queued message (200)
        or times out (204), at which point we immediately re-poll.
        """
        import time
        session = await self._get_p2p_session()
        poll_url = f"{SCHEDULER_HTTP_URL}/worker/poll/{self._worker_id}"

        while True:
            try:
                # Update last poll time for idle watchdog
                self._last_job_time = time.time()

                async with session.get(
                    poll_url,
                    timeout=aiohttp.ClientTimeout(total=35),  # slightly longer than scheduler's 29s
                ) as resp:
                    if resp.status == 200:
                        msg = await resp.json()
                        msg_type = msg.get("type", "unknown")
                        print(f"📥 [Poll] Received: {msg_type}")
                        await self._dispatch_message(msg, ws=None)
                    elif resp.status == 204:
                        # No message — normal, just re-poll
                        pass
                    else:
                        body = await resp.text()
                        print(f"⚠️ [Poll] Unexpected status {resp.status}: {body[:200]}")

            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                # Scheduler timed out — normal, re-poll
                pass
            except Exception as e:
                print(f"⚠️ [Poll] Error: {e}")
                await asyncio.sleep(1)  # brief backoff on error

    async def _idle_watchdog(self):
        """
        Monitor idle time and self-terminate when IDLE_TIMEOUT is exceeded.

        This enables scale-to-zero for cost savings.  After IDLE_TIMEOUT seconds
        without receiving a job (meaning no poll requests are being processed),
        the worker exits cleanly.
        """
        import time
        while True:
            await asyncio.sleep(30)
            if WORKER_MODE != "serverless":
                continue

            idle_time = time.time() - self._last_job_time
            if idle_time > IDLE_TIMEOUT:
                print(f"😴 [Idle] No activity for {idle_time:.0f}s > {IDLE_TIMEOUT}s — terminating")
                # Clean shutdown
                os._exit(0)

    async def _run_serverless(self):
        """
        Serverless mode: no WebSocket.

        On startup:
          1. Mounts POST /control on the P2P app (inbound channel for the scheduler)
          2. Starts the combined P2P + control HTTP server on :8003
          3. Registers with the scheduler (advertises worker_id + endpoint_url + wake_url)
          4. Starts background polling loop (long-poll for control messages)
          5. Starts idle watchdog (self-terminates after idle timeout)
          6. Sends periodic heartbeats
          7. Idles — the scheduler drives execution via poll responses

        NOTE: The scheduler NO LONGER pushes to the worker's proxy URL.
        Instead, the worker polls GET /worker/poll/{worker_id} and the scheduler
        returns queued messages. This avoids the 403 error that occurs when the
        scheduler tries to POST to {podId}-{port}.proxy.runpod.net for serverless.
        """
        # Mount /control on the P2P app BEFORE start() so the route is
        # registered on the same web.Application (not discarded by start()).
        self._mount_control_endpoint()

        # Start the P2P + control HTTP server
        await self.p2p_server.start()

        p2p_url = self.get_p2p_url()
        if not p2p_url:
            print("❌ [Serverless] No P2P URL available. "
                  "Set P2P_PUBLIC_URL or P2P_URL_TEMPLATE.")
            return

        print(f"🌐 [Serverless] P2P/Control URL: {p2p_url}")

        # Register endpoint with scheduler (includes wake_url for scale-to-zero)
        await self._register_with_scheduler(p2p_url)

        # Start background tasks
        asyncio.create_task(self._serverless_heartbeat())
        asyncio.create_task(self._poll_scheduler())
        asyncio.create_task(self._idle_watchdog())

        print(f"✅ [Serverless] Worker {self._worker_id} ready. "
              f"Polling scheduler at {SCHEDULER_HTTP_URL}/worker/poll/{self._worker_id} ...")
        print(f"   Idle timeout: {IDLE_TIMEOUT}s (scale-to-zero enabled)")

        # Keep process alive — aiohttp runner owns the event loop from here
        while True:
            await asyncio.sleep(3600)

    # =========================================================================
    # Job processors
    # =========================================================================

    async def _process_dense(self, msg: Dict, ws):
        """Process dense layer job."""
        from azu.worker.layer_processor import DenseLayerProcessor

        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg.get('layer_idx', -1)
        next_hop = msg.get('next_hop')
        next_layer_idx = msg.get('next_layer_idx')
        is_first = msg.get('is_first', False)
        is_last = msg.get('is_last', False)
        max_tokens = msg.get('max_tokens', 50)
        first_node_endpoint = msg.get('first_node_endpoint')

        # Capture prompt into local scope immediately — before any await.
        # Multiple EXECUTE_DENSE tasks for the same job share the same msg dict;
        # reading msg['input'] after an await risks it already being cleared.
        initial_prompt = msg.get('input') if is_first else None

        print(f"🔵 [DENSE] Processing job {job_id[:8]}, layer_idx={layer_idx}")

        await self.model_manager.ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)
        self.model_manager._print_vram_stats(f"Dense Start {layer_idx}", ctx)

        while not ctx.done:
            try:
                # ==== First Node: Embedding ====
                if is_first:
                    if not self.model_manager.embeddings:
                        print(f"📦 Loading embeddings...")
                        self.model_manager.embeddings = await self.loader.load_embeddings(model_id)
                        await self.model_manager.load_tokenizer(model_id)
                        self.model_manager._print_vram_stats("Loaded Emb", ctx)

                    input_tensor = None

                    # ------------------------------------------------------------------
                    # RACE-CONDITION FIX
                    #
                    # Old code: if not ctx.generated_ids and msg.get('input'):
                    # Multiple EXECUTE_DENSE tasks all pass this guard simultaneously
                    # because there is no `await` between the check and the mutation,
                    # so asyncio never yields between them — all N tasks encode the
                    # prompt N times, producing the log spam seen in production.
                    #
                    # Fix: atomic check-and-set under _context_lock.  Only the first
                    # coroutine to acquire the lock sets prompt_encoded=True and wins
                    # the encode path.  All others fall through to the token-queue path
                    # and exit cleanly once ctx.done is set.
                    # ------------------------------------------------------------------
                    should_encode = False
                    if initial_prompt and not ctx.generated_ids:
                        async with self._context_lock:
                            if not getattr(ctx, 'prompt_encoded', False):
                                ctx.prompt_encoded = True
                                should_encode = True

                    if should_encode:
                        print(f"   📝 Encoding Prompt...")
                        ctx.kv_cache = DynamicCache()
                        input_tensor = self.model_manager.tokenizer.encode(
                            initial_prompt, return_tensors='pt'
                        ).to(self.device)

                        ctx.prompt_token_count = input_tensor.shape[1]
                        print(f"   📊 Prompt tokens: {ctx.prompt_token_count}")

                    elif ctx.generated_ids:
                        try:
                            token_id = await asyncio.wait_for(ctx.token_queue.get(), timeout=300)
                            if token_id == -1:
                                print(f"   🏁 [Job {job_id[:8]}] EOS signal received")
                                ctx.done = True
                                break
                            input_tensor = torch.tensor([[token_id]], device=self.device)
                        except asyncio.TimeoutError:
                            if ctx.done:
                                break
                            print(f"❌ [Job {job_id[:8]}] Token queue timeout")
                            break
                    else:
                        # Duplicate task: prompt not yet encoded by winner, no tokens yet.
                        # Yield and retry — winner will set prompt_encoded imminently.
                        await asyncio.sleep(0.01)
                        continue

                    with torch.no_grad():
                        hidden_states = self.model_manager.embeddings(input_tensor)

                    if next_hop:
                        await self.p2p_server.send_tensor(next_hop, {
                            "job_id": job_id,
                            "type": "input",
                            "target_layer_idx": next_layer_idx
                        }, hidden_states)

                # ==== Last Node: Decode ====
                elif is_last:
                    if not self.model_manager.lm_head:
                        print(f"📦 Loading lm_head...")
                        self.model_manager.lm_head = await self.loader.load_lm_head(model_id)
                        self.model_manager.final_norm = await self.loader.load_final_norm(model_id)
                        if not self.model_manager.tokenizer:
                            await self.model_manager.load_tokenizer(model_id)
                        self.model_manager._print_vram_stats("Loaded LM Head", ctx)

                    queue = ctx.get_layer_input_queue(layer_idx)
                    try:
                        hidden_states = await asyncio.wait_for(queue.get(), timeout=300)
                    except asyncio.TimeoutError:
                        if ctx.done:
                            break
                        print(f"❌ [Job {job_id[:8]}] Last node input timeout")
                        break

                    hidden_states = hidden_states.to(self.dtype)

                    with torch.no_grad():
                        if self.model_manager.final_norm:
                            hidden_states = self.model_manager.final_norm(hidden_states)
                        logits = self.model_manager.lm_head(hidden_states)

                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    token_id = next_token.item()

                    if not hasattr(ctx, 'generated_ids'):
                        ctx.generated_ids = []
                    ctx.generated_ids.append(token_id)

                    eos_token_id = self.model_manager.tokenizer.eos_token_id if self.model_manager.tokenizer else None
                    is_eos = (token_id == eos_token_id) if eos_token_id is not None else False

                    if is_eos or len(ctx.generated_ids) >= max_tokens:
                        output_text = self.model_manager.tokenizer.decode(
                            ctx.generated_ids, skip_special_tokens=True
                        ) if self.model_manager.tokenizer else str(ctx.generated_ids)

                        print(f"✅ [Job {job_id[:8]}] Generation complete: '{output_text[:50]}...'")

                        result_payload = {
                            "type":             "RESULT",
                            "job_id":           job_id,
                            "status":           "completed",
                            "output":           output_text,
                            "tokens_generated": len(ctx.generated_ids),
                        }

                        if WORKER_MODE == "serverless":
                            await self._send_result_http(
                                job_id           = job_id,
                                status           = "completed",
                                output           = output_text,
                                tokens_generated = len(ctx.generated_ids),
                            )
                        else:
                            if ws:
                                await ws.send(json.dumps(result_payload))

                        ctx.done = True

                        # Signal first node to stop
                        if first_node_endpoint:
                            try:
                                session = await self._get_p2p_session()
                                _loopback_headers = {}
                                ctx_ref = await self._get_context(job_id, create=False)
                                if ctx_ref and ctx_ref.auth_token:
                                    _loopback_headers["x-auth-token"] = ctx_ref.auth_token
                                async with session.post(
                                    f"{first_node_endpoint}/token_in",
                                    json={"job_id": job_id, "token_id": -1},
                                    headers=_loopback_headers
                                ) as resp:
                                    if resp.status != 200:
                                        print(f"   ⚠️ Loopback failed: {resp.status}")
                            except Exception as e:
                                print(f"   ❌ Loopback error: {e}")
                        break
                    else:
                        # Send token back to first node
                        if first_node_endpoint:
                            try:
                                session = await self._get_p2p_session()
                                _loopback_headers = {}
                                ctx_ref = await self._get_context(job_id, create=False)
                                if ctx_ref and ctx_ref.auth_token:
                                    _loopback_headers["x-auth-token"] = ctx_ref.auth_token
                                async with session.post(
                                    f"{first_node_endpoint}/token_in",
                                    json={"job_id": job_id, "token_id": token_id},
                                    headers=_loopback_headers
                                ) as resp:
                                    if resp.status != 200:
                                        print(f"   ⚠️ Loopback failed: {resp.status}")
                            except Exception as e:
                                print(f"   ❌ Loopback error: {e}")

                # ==== Middle Node: Dense Layer ====
                else:
                    queue = ctx.get_layer_input_queue(layer_idx)
                    try:
                        hidden_states = await asyncio.wait_for(queue.get(), timeout=300)
                    except asyncio.TimeoutError:
                        if ctx.done:
                            break
                        print(f"❌ [Job {job_id[:8]}] Middle node input timeout at layer {layer_idx}")
                        break

                    hidden_states = hidden_states.to(self.dtype)

                    dense_layer = await self.loader.load_dense_layer(model_id, layer_idx)
                    self.model_manager._print_vram_stats(f"Loaded Dense {layer_idx}", ctx)

                    pos_emb, attn_mask, pos_ids = self.model_manager.prepare_inputs(
                        hidden_states, ctx.kv_cache, layer_idx
                    )

                    with torch.no_grad():
                        layer_out = dense_layer(
                            hidden_states,
                            position_embeddings=pos_emb,
                            attention_mask=attn_mask,
                            position_ids=pos_ids,
                            past_key_values=ctx.kv_cache,
                            use_cache=True
                        )

                    if isinstance(layer_out, tuple):
                        hidden_states = layer_out[0]
                        if len(layer_out) > 1 and layer_out[1] is not None:
                            ctx.kv_cache = layer_out[1]
                    else:
                        hidden_states = layer_out

                    if next_hop:
                        await self.p2p_server.send_tensor(next_hop, {
                            "job_id": job_id,
                            "type": "input",
                            "target_layer_idx": next_layer_idx
                        }, hidden_states)

            except Exception as e:
                print(f"❌ Error in dense loop: {e}")
                traceback.print_exc()
                break

    async def _process_moe_router(self, msg: Dict, ws):
        """Process MoE router job."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')
        expert_map = msg.get('expert_map', {})
        next_layer_idx = msg.get('next_layer_idx')

        print(f"🟢 [ROUTER] Processing job {job_id[:8]}, layer_idx={layer_idx}")

        await self.model_manager.ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)
        self.model_manager._print_vram_stats(f"Router Start {layer_idx}", ctx)

        while not ctx.done:
            try:
                queue = ctx.get_layer_input_queue(layer_idx)
                try:
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    if ctx.done:
                        break
                    print(f"❌ [Job {job_id[:8]}] Router input timeout")
                    break

                hidden_states = hidden_states.to(self.dtype)

                # Step 1: Shared Attention & Norms
                shared_layer = await self.loader.load_moe_shared(model_id, layer_idx)
                self.model_manager._print_vram_stats(f"Loaded Shared {layer_idx}", ctx)

                pos_emb, attn_mask, pos_ids = self.model_manager.prepare_inputs(
                    hidden_states, ctx.kv_cache, layer_idx
                )

                residual = hidden_states
                if hasattr(shared_layer, 'input_layernorm'):
                    hidden_states = shared_layer.input_layernorm(hidden_states)

                if hasattr(shared_layer, 'self_attn'):
                    attn_out, new_kv = shared_layer.self_attn(
                        hidden_states,
                        position_embeddings=pos_emb,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        past_key_values=ctx.kv_cache,
                        use_cache=True
                    )
                    hidden_states = attn_out

                hidden_states = residual + hidden_states
                post_attn_residual = hidden_states

                if hasattr(shared_layer, 'post_attention_layernorm'):
                    hidden_states = shared_layer.post_attention_layernorm(hidden_states)

                # Step 2: Router
                if layer_idx not in self.model_manager.moe_routers:
                    print(f"📦 Loading router {layer_idx}...")
                    self.model_manager.moe_routers[layer_idx] = await self.loader.load_moe_router(
                        model_id, layer_idx
                    )
                    self.model_manager._print_vram_stats(f"Loaded Router {layer_idx}", ctx)

                with torch.no_grad():
                    logits = self.model_manager.moe_routers[layer_idx](hidden_states)
                    routing_weights, selected_indices = torch.topk(logits, k=2, dim=-1)
                    routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

                top_indices = selected_indices.cpu()
                required_experts = set(top_indices.flatten().tolist())
                local_pending: Dict[int, asyncio.Future] = {}
                send_tasks = []

                for expert_idx in required_experts:
                    target_url = expert_map.get(str(expert_idx))
                    if not target_url:
                        continue

                    mask = (top_indices == expert_idx)
                    rows, cols, _ = torch.where(mask)
                    sliced = hidden_states[rows, cols, :]

                    future = asyncio.Future()
                    local_pending[expert_idx] = future
                    ctx.pending_expert_requests[(layer_idx, expert_idx)] = future

                    send_tasks.append(asyncio.create_task(
                        self.p2p_server.send_tensor(f"{target_url}/tensor_in", {
                            "job_id": job_id,
                            "type": "input",
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx
                        }, sliced)
                    ))

                if send_tasks:
                    await asyncio.gather(*send_tasks)

                pending = list(local_pending.values())
                if pending:
                    try:
                        await asyncio.wait_for(asyncio.gather(*pending), timeout=300)
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Expert results timeout")
                        break

                self.model_manager._print_vram_stats(f"Router Inf {layer_idx}", ctx)

                # Step 3: Merge
                batch, seq, hidden = hidden_states.shape
                moe_output = torch.zeros(
                    (batch, seq, hidden),
                    dtype=self.dtype,
                    device=self.device
                )
                top_weights_dev = routing_weights.to(self.device)
                top_indices_dev = selected_indices.to(self.device)

                with torch.no_grad():
                    for expert_idx, future in local_pending.items():
                        if not future.done():
                            continue
                        res = future.result().to(self.device).to(self.dtype)
                        mask = (top_indices_dev == expert_idx)
                        rows, cols, k_idx = torch.where(mask)
                        w = top_weights_dev[rows, cols, k_idx].unsqueeze(-1)
                        moe_output.index_put_((rows, cols), res * w, accumulate=True)

                final_output = post_attn_residual + moe_output

                for expert_idx in local_pending:
                    ctx.pending_expert_requests.pop((layer_idx, expert_idx), None)

                if next_hop:
                    await self.p2p_server.send_tensor(next_hop, {
                        "job_id": job_id,
                        "type": "input",
                        "target_layer_idx": next_layer_idx
                    }, final_output)

            except Exception as e:
                print(f"❌ Error in router loop: {e}")
                traceback.print_exc()
                break

    async def _process_moe_expert(self, msg: Dict, ws):
        """Process MoE expert job."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        print(f"🟡 [EXPERT] Processing expert {expert_idx} (Layer {layer_idx})")

        await self.model_manager.ensure_model(model_id)
        ctx = await self._get_context(job_id, create=True)
        self.model_manager._print_vram_stats(f"Expert Start {layer_idx}:{expert_idx}", ctx)

        tokens_processed = 0

        while not ctx.done:
            try:
                queue = ctx.get_expert_queue(layer_idx, expert_idx)
                try:
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    if ctx.done:
                        print(f"✅ [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) finished - job done, processed {tokens_processed} tokens")
                        break
                    print(f"⏳ [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) timeout, continuing to wait...")
                    continue

                tokens_processed += 1

                hidden_states = hidden_states.to(self.dtype)

                cache_key = (layer_idx, expert_idx)
                if cache_key not in self.model_manager.moe_experts:
                    print(f"📦 Loading expert {expert_idx}...")
                    self.model_manager.moe_experts[cache_key] = await self.loader.load_moe_expert(
                        model_id, layer_idx, expert_idx
                    )
                    self.model_manager._print_vram_stats(f"Loaded Expert {layer_idx}:{expert_idx}", ctx)

                with torch.no_grad():
                    output = self.model_manager.moe_experts[cache_key](hidden_states)

                await self.p2p_server.send_tensor(f"{return_url}/tensor_in", {
                    "job_id": job_id,
                    "type": "expert_result",
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx
                }, output)

                self.model_manager._print_vram_stats(f"Expert Inf {layer_idx}:{expert_idx}", ctx)

            except Exception as e:
                print(f"❌ Error in expert loop: {e}")
                traceback.print_exc()
                break

        print(f"👋 [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) exiting, processed {tokens_processed} total tokens")

    # =========================================================================
    # Entry point
    # =========================================================================

    async def run(self):
        """Main run loop — dispatch to persistent or serverless mode."""
        if WORKER_MODE == "serverless":
            print(f"🚀 Starting in SERVERLESS mode (worker_id={self._worker_id})")
            await self._run_serverless()
        else:
            print(f"🚀 Starting in PERSISTENT mode (worker_id={self._worker_id})")
            # Persistent mode starts P2P server first, then connects WebSocket
            await self.p2p_server.start()
            await self._run_persistent()


def main():
    """Entry point."""
    asyncio.run(MoEWorker().run())


if __name__ == "__main__":
    main()
