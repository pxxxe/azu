"""
Model manager module for handling model loading, switching, and component caching.
Manages VRAM, rotary embeddings, tokenizers, and model components.
"""

import asyncio
import gc
import sys
import traceback
from typing import Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoConfig

from client.worker.layer_loader import LayerLoader
from client.worker.config import LAYER_CACHE_DIR


# Try to import Rotary Embeddings for v5 Compatibility
try:
    from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
except ImportError:
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as MixtralRotaryEmbedding
    except ImportError:
        print("⚠️ Could not import RotaryEmbedding. v5 Compat mode might fail.")
        MixtralRotaryEmbedding = None


class ModelManager:
    """
    Manages model components, caching, VRAM, and tokenizer loading.
    Handles model switching and ensures proper memory management.
    """

    def __init__(self, loader: LayerLoader):
        """
        Initialize model manager.

        Args:
            loader: LayerLoader instance for loading model components
        """
        self.loader = loader
        self.device = loader.device
        self.dtype = loader.dtype

        # Model components
        self.config: Optional[AutoConfig] = None
        self.embeddings: Optional[torch.nn.Module] = None
        self.rotary_emb: Optional[Any] = None
        self.lm_head: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.final_norm: Optional[torch.nn.Module] = None

        # Layer caches
        self.dense_layers: Dict[int, torch.nn.Module] = {}
        self.moe_routers: Dict[int, torch.nn.Module] = {}
        self.moe_experts: Dict[tuple, torch.nn.Module] = {}

        # Current model ID
        self.current_model_id: Optional[str] = None

        # Lock for thread-safe model switching
        self._model_lock = asyncio.Lock()

    async def ensure_model(self, model_id: str) -> None:
        """
        Ensure the specified model is loaded.
        Switches models if necessary, clearing VRAM and loading new components.

        Args:
            model_id: HuggingFace model ID
        """
        if self.current_model_id == model_id:
            return

        async with self._model_lock:
            if self.current_model_id == model_id:
                return

            print(f"🧹 New model {model_id} requested. Clearing VRAM...")
            sys.stdout.flush()

            # Clear current model
            self._clear_model()

            # Log clear state
            self._print_vram_stats("Cleared")

            # Initialize Rotary Embeddings for v5 Compat
            await self._init_rotary_embeddings(model_id)

            # Update current model ID
            self.current_model_id = model_id

    def _clear_model(self) -> None:
        """Clear all model components and free VRAM."""
        self.config = None
        self.embeddings = None
        self.rotary_emb = None
        self.lm_head = None
        self.tokenizer = None
        self.final_norm = None
        self.dense_layers.clear()
        self.moe_routers.clear()
        self.moe_experts.clear()

        gc.collect()
        torch.cuda.empty_cache()

    async def _init_rotary_embeddings(self, model_id: str) -> None:
        """Initialize global rotary embeddings for the model."""
        try:
            print(f"   ⚙️ Initializing Rotary Embeddings for {model_id}...")
            config_path, config_url = self.loader._get_paths(model_id, "config.json")
            if not config_path.exists():
                await self.loader._download(config_url, config_path)

            self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

            if MixtralRotaryEmbedding:
                # TRANSFORMERS 5.0+ FIX: Pass config object instead of individual params
                self.rotary_emb = MixtralRotaryEmbedding(
                    config=self.config,
                    device=self.device
                ).to(self.dtype)

                print(f"   ✅ RoPE Initialized (head_dim={self.config.hidden_size // self.config.num_attention_heads}, "
                      f"base={getattr(self.config, 'rope_theta', 10000.0)})")

        except Exception as e:
            print(f"   ⚠️ Failed to init RoPE: {e}")
            traceback.print_exc()
            self.rotary_emb = None

        self._print_vram_stats("Init")

    async def load_tokenizer(self, model_id: str) -> AutoTokenizer:
        """
        Load tokenizer from registry.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Loaded tokenizer
        """
        if self.tokenizer:
            return self.tokenizer

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

        return self.tokenizer

    def prepare_inputs(self, hidden_states: torch.Tensor, past_kv: Any) -> tuple:
        """
        Prepare inputs for transformer layer: position embeddings, attention mask, position IDs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden]
            past_kv: Past key/value cache

        Returns:
            Tuple of (position_embeddings, attention_mask, position_ids)
        """
        seq_len = hidden_states.shape[1]
        past_len = 0

        if past_kv is not None:
            # Handle standard tuple kv
            if isinstance(past_kv, tuple) and len(past_kv) > 0:
                past_len = past_kv[0].shape[2]
            # Handle HuggingFace Cache object (best effort)
            elif hasattr(past_kv, 'get_seq_length'):
                past_len = past_kv.get_seq_length()

        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).view(-1, seq_len)

        position_embeddings = None
        if self.rotary_emb:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        total_len = past_len + seq_len

        # Create causal mask (1, 1, seq_len, total_len)
        mask = torch.full(
            (1, 1, seq_len, total_len),
            0,
            dtype=self.dtype,
            device=self.device
        )

        # Apply causal masking accounting for KV cache
        causal_mask = torch.triu(
            torch.full((seq_len, total_len), float("-inf"), device=self.device),
            diagonal=past_len + 1
        )
        mask = mask + causal_mask.unsqueeze(0).unsqueeze(0)

        return position_embeddings, mask, position_ids

    def _print_vram_stats(self, tag: str, ctx: Any = None) -> None:
        """Print VRAM statistics."""
        if not torch.cuda.is_available():
            return

        try:
            # Physical Memory
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            used_gb = total_gb - free_gb

            # PyTorch Allocator
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)

            # KV Cache Size
            kv_mb = 0.0
            if ctx and hasattr(ctx, 'kv_cache'):
                try:
                    if hasattr(ctx.kv_cache, 'key_cache') and hasattr(ctx.kv_cache, 'value_cache'):
                        for t_list in ctx.kv_cache.key_cache + ctx.kv_cache.value_cache:
                            if torch.is_tensor(t_list):
                                kv_mb += (t_list.element_size() * t_list.nelement()) / (1024**2)
                except:
                    pass

            print(f"   📊 [{tag}] Used: {used_gb:.2f}GB (Alloc: {allocated_gb:.2f}GB | Res: {reserved_gb:.2f}GB) | Free: {free_gb:.2f}GB | KV: {kv_mb:.1f}MB")
            sys.stdout.flush()
        except Exception as e:
            print(f"   ⚠️ VRAM Stats Error: {e}")
