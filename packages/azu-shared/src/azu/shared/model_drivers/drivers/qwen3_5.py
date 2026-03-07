"""
Qwen35Driver — Qwen 3.5 family (VLM-wrapped, nested text_config).

This is the architecture that exposed the VLM wrapping problem.

Key differences from standard transformers layout:

  Config
  ------
  Qwen3_5Config does NOT expose vocab_size, hidden_size,
  max_position_embeddings, etc. at the top level.  These live inside
  config.text_config.  normalize_config() proxies them up.

  Module tree
  -----------
  The decoder layers live at model.language_model.layers (not model.layers).
  embed_tokens is at model.language_model.embed_tokens.
  lm_head is at model.language_model.lm_head in the module tree BUT its
  weight is stored as the flat key "lm_head.weight" in the checkpoint —
  not "model.language_model.lm_head.weight".

  Weight map (from Qwen/Qwen3.5-27B):
    "lm_head.weight"                                  ← flat
    "model.language_model.embed_tokens.weight"        ← nested
    "model.language_model.layers.N.mlp.*"             ← nested
    "mtp.layers.0.*"                                  ← MTP head, not a decoder layer

  Layer class resolution
  ----------------------
  Qwen3.5 is a hybrid model — layers alternate between GatedDeltaNet and
  standard attention.  The layer __init__ code (from fla) is incompatible
  with init_empty_weights / meta device: GatedDeltaNet silently fails to
  populate the ModuleList, so the generic meta-model path in LayerLoader
  always returns an empty list.

  get_layer_classes() resolves the full per-index class list directly from
  config.text_config.layer_types — "linear_attention" maps to
  Qwen3_5GatedDeltaNetDecoderLayer and "full_attention" maps to
  Qwen3_5AttentionDecoderLayer.  This requires zero model instantiation and
  is immune to fla/causal-conv1d availability.

  The modeling module is guaranteed to be in sys.modules as a side-effect of
  the AutoModel.from_config call that LayerLoader runs before get_layer_classes
  is invoked, even when fla falls back to its torch implementation.

  If (in some edge case) the classes cannot be found in sys.modules, the
  driver falls back to importing the module via AutoConfig's auto-class
  resolution so the worker can still serve requests without the fast kernels.

  RoPE
  ----
  Qwen3.5 layers compute RoPE internally; they do not accept an external
  position_embeddings tensor.  init_rope() returns None unconditionally
  so ModelManager skips external RoPE construction entirely.

  Forward kwargs
  --------------
  Qwen3.5 layers accept standard kwargs (hidden_states, attention_mask,
  position_ids, past_key_value, use_cache) but NOT position_embeddings.
  build_forward_kwargs() omits position_embeddings regardless of what
  layer introspection says to avoid silent failures on layers that accept
  **kwargs.
"""

from __future__ import annotations

import sys
from typing import Optional, List, Tuple, Any, Type

from ..base import ModelDriver
from ..registry import register

# Attrs to proxy from text_config → top-level config
_PROXY_ATTRS = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rms_norm_eps",
    "layer_norm_eps",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "initializer_range",
    "tie_word_embeddings",
    # MoE-related (Qwen3.5 is dense but include for completeness)
    "num_local_experts",
    "num_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    # Architecture must be propagated so downstream code sees it
    "architectures",
]

# Maps config.text_config.layer_types values → transformer class names.
# These names come from the trust_remote_code modeling module and are
# stable across all Qwen3.5 checkpoint sizes.
_LAYER_TYPE_TO_CLASS: dict[str, str] = {
    "linear_attention": "Qwen3_5GatedDeltaNetDecoderLayer",
    "full_attention":   "Qwen3_5AttentionDecoderLayer",
}


def _find_class_in_sys_modules(cls_name: str) -> Optional[type]:
    """
    Search sys.modules for a class with the given name.

    Returns the first match that is a type, or None if not found.
    This is safe to call even when fla/causal-conv1d are absent —
    the fallback torch implementation registers the same class names.
    """
    for mod in sys.modules.values():
        if mod is None:
            continue
        try:
            cls = getattr(mod, cls_name, None)
            if cls is not None and isinstance(cls, type):
                return cls
        except Exception:
            continue
    return None


@register
class Qwen35Driver(ModelDriver):

    architectures = [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3ForCausalLM",              # Qwen3 family (same layout)
        "Qwen3_5ForCausalLM",            # future flat variant if released
    ]

    # ── Config ───────────────────────────────────────────────────────────────

    def normalize_config(self, config) -> None:
        """
        Proxy language-model attrs from config.text_config to the top level.
        Idempotent — already-present attrs are never overwritten.
        """
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            return
        for attr in _PROXY_ATTRS:
            try:
                getattr(config, attr)   # raises AttributeError if missing
            except AttributeError:
                try:
                    setattr(config, attr, getattr(text_cfg, attr))
                except AttributeError:
                    pass  # not on text_config either — skip silently

    # ── Layer discovery ───────────────────────────────────────────────────────

    @property
    def layer_module_paths(self) -> list[str]:
        # Qwen3.5's language_model IS the backbone (not a ForCausalLM wrapper),
        # so layers sit directly at model.language_model.layers — matching both
        # the live Python module tree and the checkpoint weight-map keys.
        return [
            "model.language_model.layers",  # Qwen3.5 primary path
        ] + super().layer_module_paths

    def get_layer_classes(self, config) -> Optional[List[Type]]:
        """
        Build the per-index class list from config.text_config.layer_types.

        Qwen3.5 encodes the hybrid attention pattern directly in the config:
          "linear_attention" → Qwen3_5GatedDeltaNetDecoderLayer
          "full_attention"   → Qwen3_5AttentionDecoderLayer

        This avoids any model instantiation entirely and is immune to whether
        fla / causal-conv1d are installed — both the fast-kernel and the pure
        torch fallback register the same class names in sys.modules.

        LayerLoader guarantees that AutoModel.from_config has already run
        (even if it produced an empty layer list), so the trust_remote_code
        modeling module is in sys.modules by the time this is called.

        Returns None if layer_types is absent or a class cannot be resolved,
        signalling LayerLoader to raise its informative ValueError.
        """
        text_cfg = getattr(config, "text_config", config)
        layer_types: Optional[list] = getattr(text_cfg, "layer_types", None)
        if not layer_types:
            return None

        # Resolve each distinct class name once from sys.modules.
        cls_cache: dict[str, Optional[type]] = {}
        for lt in set(layer_types):
            cls_name = _LAYER_TYPE_TO_CLASS.get(lt)
            if cls_name is None:
                # Unknown layer type — we cannot build a complete list.
                return None
            cls_cache[lt] = _find_class_in_sys_modules(cls_name)

        # If any required class is missing from sys.modules, attempt a
        # forced import of the trust_remote_code modeling module so the
        # worker can still serve without fast kernels.
        missing = [lt for lt, cls in cls_cache.items() if cls is None]
        if missing:
            try:
                from transformers import AutoConfig as _AC
                # Importing through AutoConfig re-triggers trust_remote_code
                # registration without needing model instantiation.
                _AC.for_model(config.model_type)
            except Exception:
                pass
            # Retry after potential re-import.
            for lt in missing:
                cls_name = _LAYER_TYPE_TO_CLASS[lt]
                cls_cache[lt] = _find_class_in_sys_modules(cls_name)

        # Final check — if still missing, give up and let the caller raise.
        if any(v is None for v in cls_cache.values()):
            still_missing = [
                _LAYER_TYPE_TO_CLASS[lt]
                for lt, v in cls_cache.items()
                if v is None
            ]
            print(
                f"   ⚠️ [Qwen35Driver] get_layer_classes: could not find "
                f"{still_missing} in sys.modules — falling back"
            )
            return None

        # Build the ordered list, one entry per layer index.
        return [cls_cache[lt] for lt in layer_types]

    def get_layer_class_from_keys(self, weight_keys: List[str]) -> Optional[Type]:
        """
        Determine the layer class directly from the safetensors weight key names.

        Qwen3.5 layers are one of two types, distinguishable without any model
        instantiation by the presence of a discriminating submodule name in the
        weight keys:

          "linear_attn"  →  GatedDeltaNet layer  (fla hybrid attention)
          "self_attn"    →  standard attention layer

        The modeling module is already in sys.modules because AutoModel.from_config
        ran (even with empty layers) before this is called.
        """
        has_linear_attn = any("linear_attn" in k for k in weight_keys)
        has_self_attn = any("self_attn" in k for k in weight_keys)

        if not has_linear_attn and not has_self_attn:
            return None  # unrecognised — fall back

        target_cls_name = (
            "Qwen3_5GatedDeltaNetDecoderLayer" if has_linear_attn
            else "Qwen3_5AttentionDecoderLayer"
        )
        return _find_class_in_sys_modules(target_cls_name)

    # ── MoE ──────────────────────────────────────────────────────────────────

    def is_moe(self, config) -> bool:
        # Qwen3.5-27B is dense.
        return False

    # ── LM head ──────────────────────────────────────────────────────────────

    def resolve_lm_head(
        self,
        model,
        weight_map: dict,
        backbone_weight_prefix: str,
    ) -> Optional[Tuple[Any, str]]:
        """
        The lm_head weight is stored as the FLAT key "lm_head.weight" in the
        Qwen3.5 checkpoint, even though the module lives deep inside the
        VLM wrapper at model.language_model.lm_head.

        We find the module through the module tree but return the flat
        checkpoint prefix so _save_module constructs the correct key.
        """
        checkpoint_prefix = None
        for wk in weight_map:
            if wk.endswith("lm_head.weight"):
                checkpoint_prefix = wk[: -len(".weight")]
                break

        if checkpoint_prefix is None:
            return None  # not found; generic scan will try

        _vlm_paths = [
            "model.language_model.lm_head",
            "language_model.lm_head",
            "model.lm_head",
        ]
        for path in _vlm_paths:
            try:
                mod = model
                for part in path.split("."):
                    mod = getattr(mod, part)
                print(f"      🔧 [Qwen35] lm_head module at '{path}', "
                      f"checkpoint prefix '{checkpoint_prefix}'")
                return (mod, checkpoint_prefix)
            except AttributeError:
                continue

        return None  # let generic scan handle it

    # ── Inference ─────────────────────────────────────────────────────────────

    def init_rope(self, config, device: str, dtype) -> None:
        """
        Qwen3.5 layers compute RoPE internally.
        Returning None tells ModelManager to skip external RoPE construction.
        position_ids are still passed to each layer; they use them directly.
        """
        return None

    def build_forward_kwargs(
        self,
        layer_forward_params,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        past_kv,
        layer_idx: int,
    ) -> dict:
        """
        Qwen3.5 layers do NOT accept position_embeddings — they compute RoPE
        from position_ids internally.  Omit it unconditionally.
        """
        has_var = "_has_var_keyword" in layer_forward_params

        def _include(name, value):
            if has_var or name in layer_forward_params:
                return (name, value)
            return None

        kwargs: dict = {"hidden_states": hidden_states}

        for pair in [
            _include("attention_mask", attention_mask),
            _include("position_ids", position_ids),
            _include("past_key_value", past_kv),
            # position_embeddings intentionally omitted
            _include("use_cache", True),
        ]:
            if pair is not None:
                kwargs[pair[0]] = pair[1]

        return kwargs
