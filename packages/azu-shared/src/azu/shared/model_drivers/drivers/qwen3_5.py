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

  get_layer_classes() resolves classes without any model instantiation by
  reading _no_split_modules from the already-loaded trust_remote_code module
  and matching against the per-layer pattern encoded in config.text_config.

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
    "layer_types"
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
        Proxy all language-model attrs from config.text_config to the top level.
        Idempotent — already-present attrs are never overwritten.
        """
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            return
        for attr, val in vars(text_cfg).items():
            if attr.startswith("_"):
                continue
            try:
                getattr(config, attr)   # already present — skip
            except AttributeError:
                try:
                    setattr(config, attr, val)
                except AttributeError:
                    pass

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
            n = getattr(config, 'num_hidden_layers', None)
            print(f"   [Qwen35] get_layer_classes: n={n}, arch={getattr(config, 'architectures', None)}")
            if not n:
                return None

            model_cls = None
            arch_name = (getattr(config, 'architectures', None) or [None])[0]

            for mod_name in (
                'transformers.models.qwen3_5.modeling_qwen3_5',
                'transformers.models.qwen3_5_text.modeling_qwen3_5_text',
            ):
                try:
                    import importlib
                    mod = importlib.import_module(mod_name)
                    cls = getattr(mod, arch_name, None) if arch_name else None
                    print(f"   [Qwen35] importlib {mod_name} → {cls}")
                    if cls is not None and isinstance(cls, type):
                        model_cls = cls
                        break
                except (ImportError, AttributeError) as e:
                    print(f"   [Qwen35] importlib {mod_name} failed: {e}")
                    continue

            if model_cls is None and arch_name:
                print(f"   [Qwen35] falling back to sys.modules scan for {arch_name}")
                for k, mod in sys.modules.items():
                    if mod is None:
                        continue
                    try:
                        cls = getattr(mod, arch_name, None)
                    except Exception:
                        continue
                    if cls is not None and isinstance(cls, type):
                        print(f"   [Qwen35] found {arch_name} in sys.modules['{k}']")
                        model_cls = cls
                        break

            print(f"   [Qwen35] model_cls={model_cls}")
            if model_cls is None:
                return None

            no_split: List[str] = getattr(model_cls, '_no_split_modules', None) or []
            if not no_split:
                return None

            src_mod = sys.modules.get(model_cls.__module__)
            cls_by_name: dict = {}
            for name in no_split:
                cls = getattr(src_mod, name, None) if src_mod else None
                if cls is not None:
                    cls_by_name[name] = cls

            if not cls_by_name:
                return None

            # Filter out vision-encoder classes — they appear in _no_split_modules
            # due to the VLM wrapper but are not part of the language-model layer
            # sequence.  e.g. Qwen3_5VisionBlock is not one of the 24 LM layers.
            text_cls_by_name = {
                name: cls for name, cls in cls_by_name.items()
                if 'vision' not in name.lower()
            }
            if text_cls_by_name:
                cls_by_name = text_cls_by_name

            # ── Step 3: homogeneous model ─────────────────────────────────────────
            if len(cls_by_name) == 1:
                return [next(iter(cls_by_name.values()))] * n

            pattern = None
            for attr in ('attn_mode', 'attn_modes', 'layer_type', 'layer_types',
                         'layer_mode', 'layer_modes', 'model_type_list'):
                for cfg in (config, getattr(config, 'text_config', None)):
                    if cfg is None:
                        continue
                    val = getattr(cfg, attr, None)
                    if isinstance(val, (list, tuple)) and len(val) == n:
                        pattern = val
                        break
                if pattern is not None:
                    break

            print(f"   [Qwen35] pattern attr found: {pattern is not None}, sample={pattern[:4] if pattern else None}")
            if pattern is None:
                return None

            pattern_to_cls: dict = {}
            for pval in set(pattern):
                tokens = set(str(pval).lower().replace('_', ' ').split())
                best: Optional[Type] = None
                best_score = 0
                for cls_name, cls in cls_by_name.items():
                    cls_lower = cls_name.lower()
                    score = sum(1 for t in tokens if len(t) > 3 and t in cls_lower)
                    if score > best_score:
                        best_score = score
                        best = cls
                print(f"   [Qwen35] pattern '{pval}' → {best}")
                if best is None:
                    return None
                pattern_to_cls[pval] = best

            return [pattern_to_cls[p] for p in pattern]

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
