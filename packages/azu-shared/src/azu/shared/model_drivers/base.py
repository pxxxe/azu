"""
ModelDriver — per-architecture adapter contract.

Every architecture-specific decision in the network delegates here.
The core (LayerStore, LayerLoader, ModelManager) stays model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Type, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class JobState:
    """
    Opaque per-job inference state managed entirely by the driver.

    Workers create one instance via driver.init_job_state() at the start of
    each job and pass it back into every driver call — they never inspect or
    mutate it directly.

    layer_caches  — maps layer_idx → the cache object returned by that layer's
                    last forward pass.  For standard transformers this is a
                    DynamicCache; for recurrent / linear-attention layers it is
                    whatever recurrent state tensor the layer returns.  Stored
                    per-layer so that mixed-architecture models (e.g. Qwen3.5)
                    never share state between incompatible layer types.

    seq_position  — number of tokens that have completed a full forward pass
                    through the pipeline so far.  Set explicitly on the embed
                    worker; used as a positional fallback for recurrent layers
                    whose state carries no sequence-length information.
    """
    layer_caches: Dict[int, Any] = field(default_factory=dict)
    seq_position: int = 0


class ModelDriver:
    """
    Abstract base for per-architecture model drivers.

    A driver encapsulates all knowledge needed to:
      - shard a model on the registry   (LayerStore)
      - load layers on workers          (LayerLoader)
      - run inference on workers        (ModelManager, layer_processor)

    Subclasses MUST declare `architectures` — the list of transformers
    architecture strings (config.architectures[0]) they handle.

    All methods have working default implementations so drivers only need
    to override what differs from the standard transformers layout.
    """

    # Declare which architecture strings this driver handles.
    # e.g. ["MixtralForCausalLM"]
    architectures: list[str] = []

    # ── Config normalization ──────────────────────────────────────────────────

    def normalize_config(self, config) -> None:
        """
        Patch missing top-level config attributes in place.

        Some VLM / multimodal architectures (e.g. Qwen3.5) store the LM
        config inside a nested sub-object (text_config, language_config,
        llm_config).  This method makes those attrs accessible at the top
        level so all downstream code can use config.vocab_size etc. uniformly.

        Mutates config in place; idempotent (safe to call multiple times).
        Default is a no-op — flat configs need no patching.
        """

    # ── Layer discovery (sharding + loading) ─────────────────────────────────

    @property
    def layer_module_paths(self) -> list[str]:
        """
        Ordered list of dot-attribute paths tried when locating the decoder
        layer list inside an nn.Module tree.  First match wins.
        """
        return [
            "model.layers",
            "transformer.h",
            "model.decoder.layers",
            "transformer.layers",
            "language_model.model.layers",
            "model.language_model.layers",
            "model.text_model.layers",
            "text_model.encoder.layers",
            "model.model.layers",
        ]

    def get_layer_classes(self, config) -> Optional[List[Type]]:
        """
        Return the per-index list of layer classes for this architecture,
        or None to let LayerLoader fall back to meta-device model inspection.

        Override this for architectures whose layer __init__ is incompatible
        with init_empty_weights / meta device (e.g. fla-based models like
        Qwen3.5 whose custom kernels silently leave the ModuleList empty).

        The returned list must have exactly config.num_hidden_layers entries.
        LayerLoader caches the result — this is called at most once per
        worker process per architecture.

        Default: None (LayerLoader uses meta-device model).
        """
        return None

    def get_layer_class_from_keys(self, weight_keys: List[str]) -> Optional[Type]:
        """
        Return the layer class for a single layer given its safetensors weight
        keys, or None to fall back to the standard class resolution path.

        This is the preferred override point for architectures (like Qwen3.5)
        where layer type is directly readable from the weight key names —
        e.g. presence of "linear_attn" vs "self_attn" in the keys.

        Called by LayerLoader after the layer file is downloaded, before
        instantiation. The keys come from safetensors metadata only (no
        tensor data is loaded).

        Default: None.
        """
        return None

    def reconcile_layer_weight_prefix(
        self, layer_prefix_base: str, weight_map: dict
    ) -> str:
        """
        The module-tree path to the layer list (e.g. "model.layers") may
        differ from the prefix used in the checkpoint weight_map
        (e.g. "model.language_model.layers" for VLM wrappers).

        Returns the prefix that weight_map keys actually use.
        Default: probe with common top-level wrappers.
        """
        probe = f"{layer_prefix_base}.0."
        if any(k.startswith(probe) for k in weight_map):
            return layer_prefix_base
        for wrapper in ("model.", "transformer.", "language_model."):
            candidate = f"{wrapper}{layer_prefix_base}.0."
            if any(k.startswith(candidate) for k in weight_map):
                return f"{wrapper}{layer_prefix_base}"
        return layer_prefix_base  # fallback; suffix-match will handle it

    # ── MoE ──────────────────────────────────────────────────────────────────

    def is_moe(self, config) -> bool:
        """Return True if this architecture uses Mixture-of-Experts."""
        return False

    def moe_router_path(self, layer: "torch.nn.Module") -> str:
        """
        Dot-attribute path from the decoder layer to the MoE block that
        contains both the router (gate) and the expert list.
        e.g. "mlp" for Mixtral, "" if the layer itself is the block.
        """
        return "mlp"

    def num_experts(self, config) -> int:
        """Return the number of experts for an MoE architecture."""
        for key in ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts"):
            val = getattr(config, key, None)
            if val and int(val) > 1:
                return int(val)
        return 8  # safe default for Mixtral-class models

    # ── Weight key resolution (sharding) ─────────────────────────────────────

    def resolve_weight_key(self, key: str, weight_map: dict) -> str:
        """
        Given a constructed weight key that may not exist verbatim in
        weight_map, return the actual key to use.

        Applies a sequence of aliasing strategies in order.  Returns key
        unchanged if nothing matches — the caller's suffix-match fallback
        will run as a last resort.

        Override to add architecture-specific aliases on top of these defaults.
        """
        if key in weight_map:
            return key

        # 1: mlp ↔ block_sparse_moe  (Mixtral checkpoint variants)
        if ".mlp." in key:
            alt = key.replace(".mlp.", ".block_sparse_moe.")
            if alt in weight_map:
                return alt
        elif ".block_sparse_moe." in key:
            alt = key.replace(".block_sparse_moe.", ".mlp.")
            if alt in weight_map:
                return alt

        # 2: lm_head with / without "model." prefix
        if key.startswith("lm_head."):
            alt = "model." + key
            if alt in weight_map:
                return alt
        elif key.startswith("model.lm_head."):
            alt = key[len("model."):]
            if alt in weight_map:
                return alt

        # 3: embed_tokens with / without "model." prefix
        if key.startswith("embed_tokens."):
            alt = "model." + key
            if alt in weight_map:
                return alt
        elif key.startswith("model.embed_tokens."):
            alt = key[len("model."):]
            if alt in weight_map:
                return alt

        # 4: final norm with / without "model." prefix
        if key.startswith("norm."):
            alt = "model." + key
            if alt in weight_map:
                return alt
        elif key.startswith("model.norm."):
            alt = key[len("model."):]
            if alt in weight_map:
                return alt

        # 5: strip common VLM wrapper prefixes
        for strip in (
            "model.language_model.",
            "language_model.",
            "model.text_model.",
            "text_model.",
            "model.model.",
        ):
            if key.startswith(strip):
                alt = key[len(strip):]
                if alt in weight_map:
                    return alt
                alt2 = "model." + alt
                if alt2 in weight_map:
                    return alt2

        return key  # not found; suffix-match fallback will run

    def resolve_lm_head(
        self,
        model: "torch.nn.Module",
        weight_map: dict,
        backbone_weight_prefix: str,
    ) -> Optional[Tuple[Any, str]]:
        """
        Return (lm_head_module, checkpoint_weight_prefix) or None to signal
        that the generic scan in LayerStore should run instead.

        checkpoint_weight_prefix is the key prefix in weight_map (e.g.
        "lm_head"), which may differ from the module's attribute path
        (e.g. "model.language_model.lm_head") for VLM-wrapped architectures.

        Default: None → let LayerStore's generic scan handle it.
        """
        return None

    def is_lm_head_tied(self, config) -> bool:
        """
        Return True if lm_head shares weights with embed_tokens and therefore
        no lm_head.safetensors file was sharded.

        Default: reads the standard transformers tie_word_embeddings field.
        Override only if a model signals this in a non-standard way.
        """
        return bool(getattr(config, "tie_word_embeddings", False))

    # ── Shared component attribute names ─────────────────────────────────────

    @property
    def embedding_module_attr(self) -> str:
        """Attribute name of embed_tokens on the backbone module."""
        return "embed_tokens"

    @property
    def final_norm_module_attr(self) -> str:
        """Attribute name of the final norm on the backbone module."""
        return "norm"

    # ── Inference — job state lifecycle ───────────────────────────────────────

    def init_job_state(self, config) -> JobState:
        """
        Create and return a fresh JobState for one inference job.

        Called once per job on the worker that handles the first layer.
        The returned object is stored on JobContext and passed back into
        every subsequent driver call for that job.

        Default: returns a JobState with empty per-layer caches and
        seq_position=0.  Override to pre-populate architecture-specific
        fields (e.g. pre-allocated recurrent state buffers).
        """
        return JobState()

    def get_layer_cache(self, state: JobState, layer_idx: int) -> Any:
        """
        Return the cached state for a specific layer from the job state.

        Workers call this to retrieve the correct past_key_value / recurrent
        state to pass into a layer's forward().  Because state is per-layer,
        hybrid architectures (e.g. Qwen3.5) never accidentally pass a
        DynamicCache to a recurrent layer or vice-versa.

        Default: simple dict lookup; returns None if the layer has never run.
        """
        return state.layer_caches.get(layer_idx)

    def update_state(self, state: JobState, layer_idx: int, layer_output: Any) -> None:
        """
        Extract the new cache from a layer's output and store it in state.

        Called by the worker immediately after every layer.forward() call.
        Replaces direct assignment to a shared ctx.kv_cache.

        Default: if layer_output is a tuple, stores element [1] (the new
        cache) under layer_idx.  Element [0] is the hidden states and is
        extracted separately via extract_hidden().  No-op if output is a
        plain tensor (layer returned no cache).
        """
        if isinstance(layer_output, tuple) and len(layer_output) > 1:
            new_cache = layer_output[1]
            if new_cache is not None:
                state.layer_caches[layer_idx] = new_cache

    def extract_hidden(self, layer_output: Any) -> "torch.Tensor":
        """
        Extract the hidden-states tensor from a layer's output.

        Layer forward() may return a plain tensor or a tuple
        (hidden_states, cache, ...).  This method normalises both cases
        so the worker never needs to know the output format.

        Default: returns element [0] for tuples, the value itself otherwise.
        """
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    def advance_position(self, state: JobState, n: int) -> None:
        """
        Advance the explicit sequence-position counter by n tokens.

        Called on the embed worker:
          • after encoding the prompt          (n = prompt_token_count)
          • before embedding each feedback token (n = 1)

        seq_position is used as a positional fallback in prepare_inputs()
        for recurrent / linear-attention layers whose cached state carries
        no sequence-length information (unlike DynamicCache).
        """
        state.seq_position += n

    # ── Inference — forward kwargs ────────────────────────────────────────────

    def init_rope(self, config, device: str, dtype) -> Optional["torch.nn.Module"]:
        """
        Initialize and return a RoPE module, or None if the architecture
        handles position encoding internally.

        Default: attempt MixtralRotaryEmbedding; return None on failure.
        Qwen3.5 and other architectures that bake RoPE into their layer
        forward() should override this to return None unconditionally.
        """
        try:
            try:
                from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
            except ImportError:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as MixtralRotaryEmbedding
            return MixtralRotaryEmbedding(config=config, device=device).to(dtype)
        except Exception:
            return None

    def build_forward_kwargs(
        self,
        layer_forward_params,   # frozenset[str] from ModelManager._get_layer_forward_params
        hidden_states: "torch.Tensor",
        position_embeddings,
        attention_mask: "torch.Tensor",
        position_ids: "torch.Tensor",
        past_kv,
        layer_idx: int,
    ) -> dict:
        """
        Build the kwargs dict for a single layer.forward() call.

        Only includes args that the layer's signature actually accepts,
        preventing TypeError on layers that don't take every standard arg
        (e.g. architectures with custom attention that ignores position_embeddings).

        Override to hard-code the exact kwargs for a known architecture
        instead of relying on signature introspection.

        Note: past_kv here is the per-layer cache already extracted from
        JobState by ModelManager.build_forward_kwargs — not the raw JobState.
        """
        has_var = "_has_var_keyword" in layer_forward_params

        def _include(name, value) -> Optional[Tuple[str, Any]]:
            if has_var or name in layer_forward_params:
                return (name, value)
            return None

        kwargs: dict = {"hidden_states": hidden_states}

        for pair in [
            _include("attention_mask", attention_mask),
            _include("position_ids", position_ids),
            _include("past_key_value", past_kv),
            _include("position_embeddings", position_embeddings),
            _include("use_cache", True),
        ]:
            if pair is not None:
                kwargs[pair[0]] = pair[1]

        return kwargs
