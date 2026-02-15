"""
Layer processor module for handling dense, MoE router, and MoE expert processing.
Contains the main inference logic for each layer type.
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional

import torch

from config import P2P_TIMEOUT
from job_context import JobContext
from model_manager import ModelManager
from layer_loader import LayerLoader


class DenseLayerProcessor:
    """Processes dense transformer layers."""

    def __init__(self, model_manager: ModelManager, layer_loader: LayerLoader):
        self.model_manager = model_manager
        self.loader = layer_loader
        self.device = model_manager.device
        self.dtype = model_manager.dtype

    async def process(
        self,
        msg: Dict[str, Any],
        ws: Any,
        job_context: JobContext
    ) -> None:
        """
        Process a dense layer job.

        Args:
            msg: Job message containing model_id, layer_idx, next_hop, etc.
            ws: WebSocket for sending results back to scheduler
            job_context: Job context for state management
        """
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
        self.model_manager._print_vram_stats(f"Dense Start {layer_idx}", job_context)

        while not job_context.done:
            try:
                # --- JIT Embedding ---
                if is_first:
                    if not self.model_manager.embeddings:
                        print(f"📦 Loading embeddings...")
                        self.model_manager.embeddings = await self.loader.load_embeddings(model_id)
                        await self.model_manager.load_tokenizer(model_id)
                        self.model_manager._print_vram_stats("Loaded Emb", job_context)

                    input_tensor = None

                    if not job_context.generated_ids and msg.get('input'):
                        print(f"   📝 Encoding Prompt...")
                        job_context.kv_cache = DynamicCache()
                        input_tensor = self.model_manager.tokenizer.encode(
                            msg['input'], return_tensors='pt'
                        ).to(self.device)
                        msg['input'] = None
                    else:
                        # Feedback Token (Loop)
                        try:
                            token_id = await asyncio.wait_for(
                                job_context.token_queue.get(),
                                timeout=P2P_TIMEOUT
                            )
                            input_tensor = torch.tensor([[token_id]], device=self.device)
                        except asyncio.TimeoutError:
                            if job_context.done:
                                break
                            print(f"❌ [Job {job_id[:8]}] Timeout waiting for loopback token")
                            break

                    hidden_states = self.model_manager.embeddings(input_tensor)

                else:
                    # Middle Layer Input
                    queue = job_context.get_layer_input_queue(layer_idx)
                    try:
                        hidden_states = await asyncio.wait_for(
                            queue.get(),
                            timeout=P2P_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Timeout waiting for input")
                        break

                hidden_states = hidden_states.to(self.dtype)

                # --- JIT Dense Layer & KV Cache ---
                layer_out = hidden_states
                if layer_idx != -1:
                    if layer_idx not in self.model_manager.dense_layers:
                        print(f"📦 Loading dense layer {layer_idx}...")
                        self.model_manager.dense_layers[layer_idx] = await self.loader.load_dense_layer(
                            model_id, layer_idx
                        )
                        self.model_manager._print_vram_stats(f"Loaded Dense {layer_idx}", job_context)

                    # Prepare positional args
                    pos_emb, attn_mask, pos_ids = self.model_manager.prepare_inputs(
                        hidden_states, job_context.kv_cache
                    )

                    with torch.no_grad():
                        out = self.model_manager.dense_layers[layer_idx](
                            hidden_states,
                            past_key_values=job_context.kv_cache,
                            use_cache=True,
                            position_embeddings=pos_emb,
                            attention_mask=attn_mask,
                            position_ids=pos_ids
                        )

                        if isinstance(out, tuple):
                            layer_out = out[0]
                        else:
                            layer_out = out

                    self.model_manager._print_vram_stats(f"Dense Inf {layer_idx}", job_context)

                # --- JIT Head & Decode ---
                if is_last:
                    if not self.model_manager.lm_head:
                        print(f"🔚 Loading LM Head...")
                        self.model_manager.lm_head = await self.loader.load_lm_head(model_id)
                        self.model_manager.final_norm = await self.loader.load_final_norm(model_id)
                        await self.model_manager.load_tokenizer(model_id)
                        self.model_manager._print_vram_stats("Loaded Head", job_context)

                    with torch.no_grad():
                        if self.model_manager.final_norm:
                            latents = self.model_manager.final_norm(layer_out[:, -1, :])
                        else:
                            latents = layer_out[:, -1, :]
                        logits = self.model_manager.lm_head(latents)
                        token_id = torch.argmax(logits, dim=-1).item()

                    # Record Generation
                    job_context.generated_ids.append(token_id)
                    gen_text = self.model_manager.tokenizer.decode([token_id])
                    print(f"   ✨ Gen: {gen_text}")

                    # Check Stop Conditions
                    stop = False
                    reason = ""
                    if len(job_context.generated_ids) >= max_tokens:
                        stop = True
                        reason = "max_tokens"
                    elif token_id == self.model_manager.tokenizer.eos_token_id:
                        stop = True
                        reason = "EOS"

                    if stop:
                        full_text = self.model_manager.tokenizer.decode(job_context.generated_ids)
                        print(f"   🎉 GENERATION COMPLETE ({reason}): {len(job_context.generated_ids)} tokens")
                        await ws.send(json.dumps({
                            "type": "RESULT",
                            "job_id": job_id,
                            "status": "completed",
                            "output": full_text
                        }))
                        job_context.done = True
                        return
                    else:
                        # Send token back for loopback
                        if first_node_endpoint:
                            from p2p_server import P2PServer
                            # Access p2p_server through the parent - this will be set later
                            pass

                # --- Forward to Next Worker ---
                elif next_hop:
                    # This will be handled by the caller (MoEWorker)
                    yield {"next_hop": next_hop, "layer_out": layer_out}

            except Exception as e:
                print(f"❌ Error in dense loop: {e}")
                import traceback
                traceback.print_exc()
                break


class MoERouterProcessor:
    """Processes MoE router layers."""

    def __init__(self, model_manager: ModelManager, layer_loader: LayerLoader):
        self.model_manager = model_manager
        self.loader = layer_loader
        self.device = model_manager.device
        self.dtype = model_manager.dtype

    async def process(
        self,
        msg: Dict[str, Any],
        ws: Any,
        job_context: JobContext,
        send_p2p_fn: Any
    ) -> None:
        """
        Process an MoE router job.

        Args:
            msg: Job message containing model_id, layer_idx, expert_map, etc.
            ws: WebSocket for sending results
            job_context: Job context
            send_p2p_fn: Function to send tensors via P2P
        """
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        next_hop = msg.get('next_hop')
        expert_map = msg.get('expert_map', {})
        next_layer_idx = msg.get('next_layer_idx')

        print(f"🟢 [ROUTER] Processing job {job_id[:8]}, layer_idx={layer_idx}")

        await self.model_manager.ensure_model(model_id)
        self.model_manager._print_vram_stats(f"Router Start {layer_idx}", job_context)

        while not job_context.done:
            try:
                queue = job_context.get_layer_input_queue(layer_idx)
                try:
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                except asyncio.TimeoutError:
                    if job_context.done:
                        break
                    print(f"❌ [Job {job_id[:8]}] Router input timeout")
                    break

                hidden_states = hidden_states.to(self.dtype)

                # =========================================================
                # STEP 1: Execute Shared Attention & Norms
                # =========================================================
                shared_layer = await self.loader.load_moe_shared(model_id, layer_idx)
                self.model_manager._print_vram_stats(f"Loaded Shared {layer_idx}", job_context)

                # Prepare positional args
                pos_emb, attn_mask, pos_ids = self.model_manager.prepare_inputs(
                    hidden_states, job_context.kv_cache
                )

                # A. Input Residual & Norm
                residual = hidden_states
                if hasattr(shared_layer, 'input_layernorm'):
                    hidden_states = shared_layer.input_layernorm(hidden_states)

                # B. Self Attention
                if hasattr(shared_layer, 'self_attn'):
                    attn_out, new_kv = shared_layer.self_attn(
                        hidden_states,
                        position_embeddings=pos_emb,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        past_key_values=job_context.kv_cache,
                        use_cache=True
                    )
                    hidden_states = attn_out

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

                # JIT Router
                if layer_idx not in self.model_manager.moe_routers:
                    print(f"📦 Loading router {layer_idx}...")
                    self.model_manager.moe_routers[layer_idx] = await self.loader.load_moe_router(
                        model_id, layer_idx
                    )
                    self.model_manager._print_vram_stats(f"Loaded Router {layer_idx}", job_context)

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
                    job_context.pending_expert_requests[(layer_idx, expert_idx)] = future

                    # Dispatch to experts in PARALLEL
                    send_tasks.append(asyncio.create_task(
                        send_p2p_fn(f"{target_url}/tensor_in", {
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
                        await asyncio.wait_for(asyncio.gather(*pending), timeout=P2P_TIMEOUT)
                    except asyncio.TimeoutError:
                        print(f"❌ [Job {job_id[:8]}] Expert results timeout")
                        break

                self.model_manager._print_vram_stats(f"Router Inf {layer_idx}", job_context)

                # =========================================================
                # STEP 3: Merge & Final Residual
                # =========================================================
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

                # Final Residual: (Attn Output) + (MoE Output)
                final_output = post_attn_residual + moe_output

                for expert_idx in local_pending:
                    job_context.pending_expert_requests.pop((layer_idx, expert_idx), None)

                if next_hop:
                    await send_p2p_fn(next_hop, {
                        "job_id": job_id,
                        "type": "input",
                        "target_layer_idx": next_layer_idx
                    }, final_output)

            except Exception as e:
                print(f"❌ Error in router loop: {e}")
                import traceback
                traceback.print_exc()
                break


class MoEExpertProcessor:
    """Processes MoE expert layers."""

    def __init__(self, model_manager: ModelManager, layer_loader: LayerLoader):
        self.model_manager = model_manager
        self.loader = layer_loader
        self.device = model_manager.device
        self.dtype = model_manager.dtype

    async def process(
        self,
        msg: Dict[str, Any],
        ws: Any,
        job_context: JobContext,
        send_p2p_fn: Any
    ) -> None:
        """
        Process an MoE expert job.

        Args:
            msg: Job message containing model_id, layer_idx, expert_idx, return_url
            ws: WebSocket for sending results
            job_context: Job context
            send_p2p_fn: Function to send tensors via P2P
        """
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_idx = msg['expert_idx']
        return_url = msg['return_url']

        print(f"🟡 [EXPERT] Processing expert {expert_idx} (Layer {layer_idx})")

        await self.model_manager.ensure_model(model_id)
        self.model_manager._print_vram_stats(f"Expert Start {layer_idx}:{expert_idx}", job_context)

        tokens_processed = 0

        while not job_context.done:
            try:
                queue = job_context.get_expert_queue(layer_idx, expert_idx)
                try:
                    hidden_states = await asyncio.wait_for(queue.get(), timeout=P2P_TIMEOUT)
                except asyncio.TimeoutError:
                    if job_context.done:
                        print(f"✅ [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) finished - job done, processed {tokens_processed} tokens")
                        break
                    print(f"⏳ [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) timeout, continuing to wait...")
                    continue

                tokens_processed += 1

                # Precision Guard
                hidden_states = hidden_states.to(self.dtype)

                # JIT Expert
                cache_key = (layer_idx, expert_idx)
                if cache_key not in self.model_manager.moe_experts:
                    print(f"📦 Loading expert {expert_idx}...")
                    self.model_manager.moe_experts[cache_key] = await self.loader.load_moe_expert(
                        model_id, layer_idx, expert_idx
                    )
                    self.model_manager._print_vram_stats(f"Loaded Expert {layer_idx}:{expert_idx}", job_context)

                with torch.no_grad():
                    output = self.model_manager.moe_experts[cache_key](hidden_states)

                await send_p2p_fn(f"{return_url}/tensor_in", {
                    "job_id": job_id,
                    "type": "expert_result",
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx
                }, output)

                self.model_manager._print_vram_stats(f"Expert Inf {layer_idx}:{expert_idx}", job_context)

            except Exception as e:
                print(f"❌ Error in expert loop: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"👋 [Job {job_id[:8]}] Expert {expert_idx} (Layer {layer_idx}) exiting, processed {tokens_processed} total tokens")


# Import at module level for DynamicCache
from transformers import DynamicCache
