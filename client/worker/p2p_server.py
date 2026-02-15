"""
P2P server module for worker-to-worker communication.
Handles tensor ingress/egress, mesh handshake, and peer discovery.
"""

import asyncio
import json
import sys
import traceback
import urllib.request
from typing import Dict, Optional, List, Callable, Any

import aiohttp
from aiohttp import web, ClientSession, TCPConnector, ClientTimeout

from client.worker.config import (
    P2P_PORT,
    P2P_MAX_SIZE,
    P2P_CONNECTION_RETRIES,
    P2P_HANDSHAKE_RETRIES,
    P2P_HANDSHAKE_TIMEOUT
)
from client.worker.p2p_protocol import P2PProtocol
from client.worker.job_context import JobContext


class P2PServer:
    """
    Manages P2P server for tensor transfer between workers.
    Handles binary tensor ingress/egress, mesh handshake, and peer discovery.
    """

    def __init__(
        self,
        get_p2p_url_fn: Callable[[], str],
        get_context_fn: Callable[[str, bool], Optional[JobContext]],
        get_p2p_session_fn: Callable[[], ClientSession],
        device: Any,
        dtype: Any
    ):
        """
        Initialize P2P server.

        Args:
            get_p2p_url_fn: Function to get this worker's P2P URL
            get_context_fn: Function to get job context by ID
            get_p2p_session_fn: Function to get shared P2P session
            device: Torch device for tensors
            dtype: Torch dtype for tensors
        """
        self.get_p2p_url = get_p2p_url_fn
        self._get_context = get_context_fn
        self._get_p2p_session = get_p2p_session_fn
        self.device = device
        self.dtype = dtype
        self.protocol = P2PProtocol()

        self.p2p_app: Optional[web.Application] = None
        self.p2p_session: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the P2P HTTP server."""
        # MAX SIZE 1GB IS CRITICAL FOR MoE TENSORS
        self.p2p_app = web.Application(client_max_size=P2P_MAX_SIZE)

        # Register routes
        self.p2p_app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        self.p2p_app.router.add_post('/token_in', self.handle_token_ingress)
        self.p2p_app.router.add_post('/control/job_start', self.handle_job_start)
        self.p2p_app.router.add_get('/p2p/ping', self.handle_ping)
        self.p2p_app.router.add_post('/control/peer_ready', self.handle_peer_ready)

        # Start server
        runner = web.AppRunner(self.p2p_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', P2P_PORT)
        await site.start()

        print(f"👂 [P2P] Server listening on :{P2P_PORT} (Binary/High-Perf)")
        sys.stdout.flush()

    # -------------------------------------------------------------------------
    # HTTP Handlers
    # -------------------------------------------------------------------------

    async def handle_job_start(self, request: web.Request) -> web.Response:
        """
        Receive job topology and initiate mesh handshake with all peers.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            topology = data.get("topology", [])
            my_p2p_url = self.get_p2p_url().rstrip("/")

            print(f"🔗 [Job {job_id[:8]}] Received job_start with {len(topology)} peers")

            # Create job context with topology
            ctx = await self._get_context(job_id, create=True)
            ctx.topology = topology
            ctx.status = "HANDSHAKING"

            # Remove self from topology (don't ping yourself)
            peer_urls = [url.rstrip("/") for url in topology if url.rstrip("/") != my_p2p_url]

            if peer_urls:
                print(f"🔗 [Job {job_id[:8]}] Starting mesh handshake with {len(peer_urls)} peers...")
                # Start handshake in background
                asyncio.create_task(self._perform_mesh_handshake(job_id, peer_urls))
            else:
                # No peers to wait for, mark as ready immediately
                ctx.status = "RUNNING"
                ctx.handshake_done.set()
                print(f"🔗 [Job {job_id[:8]}] No peers to handshake, running immediately")

            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ [P2P] Error in handle_job_start: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_ping(self, request: web.Request) -> web.Response:
        """Health check - verify job context exists."""
        job_id = request.query.get("job_id")
        if not job_id:
            return web.Response(status=400, text="Missing job_id")

        ctx = await self._get_context(job_id, create=False)
        if ctx is None:
            return web.Response(status=404, text="Job context not found")

        return web.json_response({
            "status": ctx.status,
            "job_id": job_id
        })

    async def handle_peer_ready(self, request: web.Request) -> web.Response:
        """Peer notifies us that they're ready for this job."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            peer_url = data.get("peer_url")

            ctx = await self._get_context(job_id, create=False)
            if ctx:
                ctx.mark_peer_ready(peer_url)
                print(f"🔗 [Job {job_id[:8]}] Peer {peer_url} ready ({len(ctx.peers_ready)}/{len(ctx.topology)})")

                # Check if all peers are ready
                if ctx.all_peers_ready():
                    ctx.status = "RUNNING"
                    ctx.handshake_done.set()
                    print(f"🔗 [Job {job_id[:8]}] ALL PEERS READY - Starting execution!")

            return web.Response(text="OK")
        except Exception as e:
            print(f"❌ [P2P] Error in handle_peer_ready: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_token_ingress(self, request: web.Request) -> web.Response:
        """Handle token loopback for autoregressive generation."""
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

    async def handle_tensor_ingress(self, request: web.Request) -> web.Response:
        """Handle incoming tensor data from peers."""
        try:
            # 1. Read Metadata from Headers
            headers = request.headers
            job_id = headers.get("x-job-id")
            if not job_id:
                return web.Response(status=400, text="Missing job_id")

            parsed = self.protocol.parse_headers(headers)
            msg_type = parsed["msg_type"]
            dtype_str = parsed["dtype"]
            shape = parsed["shape"]

            # 2. Read RAW Binary Body
            data = await request.read()

            # 3. Reconstruct Tensor
            tensor = self.protocol.bytes_to_tensor(
                data, dtype_str, shape, self.device, self.dtype
            )

            # 4. Route - Get or create context
            ctx = await self._get_context(job_id, create=True)

            # 4b. WAIT for handshake to complete before processing tensors
            if ctx.topology and ctx.status == "HANDSHAKING":
                print(f"⏳ [Job {job_id[:8]}] Waiting for mesh handshake before processing tensor...")
                await ctx.handshake_done.wait()
                print(f"✅ [Job {job_id[:8]}] Handshake complete, processing tensor")

            # Route based on message type
            if msg_type == 'input':
                expert_idx = parsed.get("expert_idx")
                layer_idx = parsed.get("layer_idx")
                target_layer_idx = parsed.get("target_layer_idx")

                if expert_idx is not None and layer_idx is not None:
                    queue = ctx.get_expert_queue(layer_idx, expert_idx)
                    await queue.put(tensor)
                elif target_layer_idx is not None:
                    queue = ctx.get_layer_input_queue(target_layer_idx)
                    await queue.put(tensor)

            elif msg_type == 'expert_result':
                expert_idx = parsed.get("expert_idx")
                layer_idx = parsed.get("layer_idx")
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

    # -------------------------------------------------------------------------
    # Mesh Handshake
    # -------------------------------------------------------------------------

    async def _perform_mesh_handshake(self, job_id: str, peer_urls: List[str]) -> None:
        """
        Ping all peers to verify connectivity before tensor transfer.
        """
        ctx = await self._get_context(job_id, create=False)
        if not ctx:
            print(f"❌ [Job {job_id[:8]}] Context disappeared during handshake!")
            return

        my_url = self.get_p2p_url().rstrip("/")
        session = await self._get_p2p_session()

        # Ping each peer with retry
        for peer_url in peer_urls:
            peer_url_clean = peer_url.rstrip("/")
            ping_url = f"{peer_url_clean}/p2p/ping?job_id={job_id}"

            for attempt in range(P2P_HANDSHAKE_RETRIES):
                try:
                    async with session.get(ping_url, timeout=aiohttp.ClientTimeout(total=P2P_HANDSHAKE_TIMEOUT)) as resp:
                        if resp.status == 200:
                            ctx.mark_peer_ready(peer_url_clean)
                            print(f"🔗 [Job {job_id[:8]}] ✓ Peer {peer_url_clean} reachable")
                            break
                        else:
                            await asyncio.sleep(0.5 * (attempt + 1))
                except Exception as e:
                    await asyncio.sleep(0.5 * (attempt + 1))
            else:
                print(f"⚠️ [Job {job_id[:8]}] ✗ Peer {peer_url_clean} unreachable after {P2P_HANDSHAKE_RETRIES} attempts")

            # Also notify peer that we're ready
            try:
                await session.post(
                    f"{peer_url_clean}/control/peer_ready",
                    json={"job_id": job_id, "peer_url": my_url},
                    timeout=aiohttp.ClientTimeout(total=P2P_HANDSHAKE_TIMEOUT)
                )
            except Exception as e:
                print(f"⚠️ [Job {job_id[:8]}] Failed to notify {peer_url_clean}: {e}")

        # Check if all peers are ready
        if ctx.all_peers_ready():
            ctx.status = "RUNNING"
            ctx.handshake_done.set()
            print(f"🔗 [Job {job_id[:8]}] ALL PEERS READY - Starting execution!")
        else:
            # Wait a bit more for peers to notify us
            print(f"🔗 [Job {job_id[:8]}] Waiting for peers to complete handshake...")
            await asyncio.sleep(2)
            if ctx.all_peers_ready():
                ctx.status = "RUNNING"
                ctx.handshake_done.set()
                print(f"🔗 [Job {job_id[:8]}] ALL PEERS READY (delayed) - Starting execution!")
            else:
                print(f"⚠️ [Job {job_id[:8]}] Handshake incomplete ({len(ctx.peers_ready)}/{len(ctx.topology)}), proceeding anyway")

    # -------------------------------------------------------------------------
    # Send Methods
    # -------------------------------------------------------------------------

    async def send_tensor(
        self,
        url: str,
        payload_meta: Dict[str, Any],
        tensor: torch.Tensor
    ) -> None:
        """
        Send tensor as raw binary body with metadata in headers.
        Includes loopback optimization.
        """
        # 1. Loopback Check
        my_p2p = self.get_p2p_url().rstrip("/")
        target_base = url.replace("/tensor_in", "").rstrip("/")

        # Convert tensor to bytes
        data_bytes, dtype_str, shape_json = self.protocol.tensor_to_bytes(tensor)

        # Prepare headers
        headers = {
            "x-job-id": payload_meta["job_id"],
            "x-msg-type": payload_meta.get("type", "input"),
            "x-dtype": dtype_str,
            "x-shape": shape_json
        }

        # Add routing fields
        if "expert_idx" in payload_meta:
            headers["x-expert-idx"] = str(payload_meta["expert_idx"])
        if "layer_idx" in payload_meta:
            headers["x-layer-idx"] = str(payload_meta["layer_idx"])
        if "target_layer_idx" in payload_meta and payload_meta["target_layer_idx"] is not None:
            headers["x-target-layer-idx"] = str(payload_meta["target_layer_idx"])

        # Loopback optimization - skip HTTP
        if my_p2p == target_base:
            try:
                ctx = await self._get_context(payload_meta['job_id'], create=True)
                msg_type = payload_meta.get('type', 'input')

                # Ensure tensor matches model dtype
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
                        if not future.done():
                            future.set_result(tensor)
                return
            except Exception as e:
                print(f"❌ Local P2P Error: {e}")
                return

        # Network Transfer (Binary)
        session = await self._get_p2p_session()

        for attempt in range(P2P_CONNECTION_RETRIES):
            try:
                headers["Content-Type"] = "application/octet-stream"
                async with session.post(url, data=data_bytes, headers=headers) as resp:
                    if resp.status == 200:
                        return
                    else:
                        print(f"   ⚠️ P2P Handshake Rejected {resp.status} from {url}")
            except Exception as e:
                print(f"   ⚠️ Connection Failed (attempt {attempt+1}) to {url}: {e}")
                await asyncio.sleep(0.2)

        print(f"❌ Failed to send to {url}")


# Import torch at module level for type hints
import torch
