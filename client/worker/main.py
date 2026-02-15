"""
Main entry point for the worker node.
Coordinates P2P networking, job processing, and scheduler communication.
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

from client.worker.config import (
    SCHEDULER_URL,
    P2P_PORT,
    DEFAULT_CPU_VRAM_MB,
)
from client.worker.layer_loader import LayerLoader
from client.worker.model_manager import ModelManager
from client.worker.job_context import JobContext
from client.worker.p2p_server import P2PServer


class MoEWorker:
    """
    Main worker class that coordinates all worker operations.
    Handles P2P networking, job processing, and scheduler communication.
    """

    def __init__(self):
        # Initialize components
        self.loader = LayerLoader(SCHEDULER_URL.replace("ws://", "http://").replace("/ws/worker", ""))
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

        sys.stdout.flush()

    def get_p2p_url(self) -> str:
        """Get this worker's P2P URL."""
        from client.worker.config import P2P_PUBLIC_URL, P2P_URL_TEMPLATE

        if P2P_PUBLIC_URL:
            return P2P_PUBLIC_URL.strip("/")

        if P2P_URL_TEMPLATE:
            try:
                pod_id = os.getenv("RUNPOD_POD_ID", "unknown")
                return P2P_URL_TEMPLATE.replace("{RUNPOD_POD_ID}", pod_id).strip("/")
            except:
                pass

        try:
            ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            return f"http://{ip}:{P2P_PORT}"
        except:
            return f"http://127.0.0.1:{P2P_PORT}"

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

    async def _process_dense(self, msg: Dict, ws):
        """Process dense layer job."""
        from client.worker.layer_processor import DenseLayerProcessor

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

                    if not ctx.generated_ids and msg.get('input'):
                        print(f"   📝 Encoding Prompt...")
                        ctx.kv_cache = DynamicCache()
                        input_tensor = self.model_manager.tokenizer.encode(
                            msg['input'], return_tensors='pt'
                        ).to(self.device)
                        msg['input'] = None
                    else:
                        # Feedback Token (Loop)
                        try:
                            token_id = await asyncio.wait_for(
                                ctx.token_queue.get(),
                                timeout=300
                            )
                            input_tensor = torch.tensor([[token_id]], device=self.device)
                        except asyncio.TimeoutError:
                            if ctx.done:
                                break
                            print(f"❌ [Job {job_id[:8]}] Timeout waiting for loopback token")
                            break

                    hidden_states = self.model_manager.embeddings(input_tensor)

                else:
                    # Middle Layer Input
                    queue = ctx.get_layer_input_queue(layer_idx)
                    try:
                        hidden_states = await asyncio.wait_for(queue.get(), timeout=300)
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Timeout waiting for input")
                        break

                hidden_states = hidden_states.to(self.dtype)

                # ==== Dense Layer Inference ====
                layer_out = hidden_states
                if layer_idx != -1:
                    if layer_idx not in self.model_manager.dense_layers:
                        print(f"📦 Loading dense layer {layer_idx}...")
                        self.model_manager.dense_layers[layer_idx] = await self.loader.load_dense_layer(
                            model_id, layer_idx
                        )
                        self.model_manager._print_vram_stats(f"Loaded Dense {layer_idx}", ctx)

                    pos_emb, attn_mask, pos_ids = self.model_manager.prepare_inputs(
                        hidden_states, ctx.kv_cache
                    )

                    with torch.no_grad():
                        out = self.model_manager.dense_layers[layer_idx](
                            hidden_states,
                            past_key_values=ctx.kv_cache,
                            use_cache=True,
                            position_embeddings=pos_emb,
                            attention_mask=attn_mask,
                            position_ids=pos_ids
                        )

                        if isinstance(out, tuple):
                            layer_out = out[0]
                        else:
                            layer_out = out

                    self.model_manager._print_vram_stats(f"Dense Inf {layer_idx}", ctx)

                # ==== Last Node: LM Head & Decode ====
                if is_last:
                    if not self.model_manager.lm_head:
                        print(f"🔚 Loading LM Head...")
                        self.model_manager.lm_head = await self.loader.load_lm_head(model_id)
                        self.model_manager.final_norm = await self.loader.load_final_norm(model_id)
                        await self.model_manager.load_tokenizer(model_id)
                        self.model_manager._print_vram_stats("Loaded Head", ctx)

                    with torch.no_grad():
                        if self.model_manager.final_norm:
                            latents = self.model_manager.final_norm(layer_out[:, -1, :])
                        else:
                            latents = layer_out[:, -1, :]
                        logits = self.model_manager.lm_head(latents)
                        token_id = torch.argmax(logits, dim=-1).item()

                    ctx.generated_ids.append(token_id)
                    gen_text = self.model_manager.tokenizer.decode([token_id])
                    print(f"   ✨ Gen: {gen_text}")

                    # Check Stop Conditions
                    stop = False
                    reason = ""
                    if len(ctx.generated_ids) >= max_tokens:
                        stop = True
                        reason = "max_tokens"
                    elif token_id == self.model_manager.tokenizer.eos_token_id:
                        stop = True
                        reason = "EOS"

                    if stop:
                        full_text = self.model_manager.tokenizer.decode(ctx.generated_ids)
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
                        # Loopback token
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

                # ==== Forward to Next Worker ====
                elif next_hop:
                    await self.p2p_server.send_tensor(next_hop, {
                        "job_id": job_id,
                        "type": "input",
                        "target_layer_idx": next_layer_idx
                    }, layer_out)

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
                    hidden_states, ctx.kv_cache
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

    async def run(self):
        """Main run loop - connect to scheduler and process jobs."""
        await self.p2p_server.start()

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

                    # Start heartbeat
                    heartbeat_task = asyncio.create_task(self.heartbeat(ws))

                    async for raw in ws:
                        msg = json.loads(raw)
                        msg_type = msg['type']
                        job_id = msg.get('job_id', 'unknown')[:8]

                        if msg_type == 'JOB_START':
                            job_id_full = msg.get('job_id')
                            topology = msg.get('topology', [])
                            model_id = msg.get('model_id')

                            print(f"🔗 [Job {job_id}] Received JOB_START, initiating mesh handshake...")

                            my_p2p_url = self.get_p2p_url().rstrip("/")
                            try:
                                session = await self._get_p2p_session()
                                async with session.post(
                                    f"{my_p2p_url}/control/job_start",
                                    json={
                                        "job_id": job_id_full,
                                        "model_id": model_id,
                                        "topology": topology
                                    },
                                    timeout=aiohttp.ClientTimeout(total=30)
                                ) as resp:
                                    if resp.status == 200:
                                        print(f"🔗 [Job {job_id}] Mesh handshake initiated successfully")
                                    else:
                                        print(f"⚠️ [Job {job_id}] Mesh handshake initiation failed: {resp.status}")
                            except Exception as e:
                                print(f"⚠️ [Job {job_id}] Failed to trigger mesh handshake: {e}")

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

                    # Cleanup on disconnect
                    heartbeat_task.cancel()
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                await asyncio.sleep(5)
            finally:
                if self.p2p_session and not self.p2p_session.closed:
                    await self.p2p_session.close()


def main():
    """Entry point."""
    asyncio.run(MoEWorker().run())


if __name__ == "__main__":
    main()
