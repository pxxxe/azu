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

from typing import Optional, Tuple, Any

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
        # Qwen3_5ForConditionalGeneration wraps a ForCausalLM which in turn
        # wraps a Model, so the live Python module tree is:
        #   self.model                        (VLM outer)
        #     .language_model                 (Qwen3ForCausalLM)
        #       .model                        (Qwen3Model backbone)
        #         .layers                     (nn.ModuleList)
        #
        # HuggingFace serialises checkpoints stripping the inner .model.,
        # so weight_map keys use "model.language_model.layers.N.*" while
        # getattr traversal requires "model.language_model.model.layers".
        # Both paths are listed so _build_layer_class_cache (module tree
        # traversal) and reconcile_layer_weight_prefix (weight-map probe)
        # each hit their correct variant.
        return [
            "model.language_model.model.layers",  # live Python module path
            "model.language_model.layers",         # checkpoint weight-map path
        ] + super().layer_module_paths

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
        # Find the checkpoint key prefix (should be "lm_head")
        checkpoint_prefix = None
        for wk in weight_map:
            if wk.endswith("lm_head.weight"):
                checkpoint_prefix = wk[: -len(".weight")]
                break

        if checkpoint_prefix is None:
            return None  # not found; generic scan will try

        # Find the module — try VLM paths
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
