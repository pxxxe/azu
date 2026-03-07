"""
Qwen35Driver — Qwen 3.5 family (VLM-wrapped, nested text_config).

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

  Layer class resolution
  ----------------------
  Qwen3.5 is a hybrid model — layers alternate between GatedDeltaNet and
  standard attention.  The layer __init__ code (from fla) is incompatible
  with init_empty_weights / meta device: GatedDeltaNet silently fails to
  populate the ModuleList, so the generic meta-model path in LayerLoader
  always returns an empty list.

  get_layer_classes() resolves the full per-index class list by:
    1. Finding the trust_remote_code modeling module in sys.modules (any
       module that exports the main architecture class).
    2. Reading _no_split_modules from the model class to get the canonical
       decoder-layer class name(s).
    3. Mapping those class names to the per-index layer_types from
       config.text_config.layer_types.

  This approach is immune to fla version differences — it reads whatever
  class names the actual downloaded modeling code uses rather than
  hardcoding names that may change between releases.

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

import torch.nn as nn

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
    "num_local_experts",
    "num_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    "architectures",
]

# Name substrings (lowercased) used to classify a decoder-layer class as
# "linear_attention" vs "full_attention" when there are multiple candidates.
_LINEAR_HINTS = ("gateddelta", "linear", "mamba", "ssm", "recurrent")
_ATTN_HINTS   = ("attention", "attn")


def _find_modeling_module(arch: str):
    """
    Return the sys.modules module that defines the top-level architecture class.

    Trust_remote_code modules are imported under `transformers_modules.*`
    but the exact path varies across HF versions and cache layouts.
    Searching by the presence of the architecture class is reliable regardless.
    """
    for mod in sys.modules.values():
        if mod is None:
            continue
        try:
            candidate = getattr(mod, arch, None)
            if candidate is not None and isinstance(candidate, type):
                return mod
        except Exception:
            continue
    return None


def _decoder_layer_classes_from_module(modeling_mod, arch: str) -> List[Tuple[str, type]]:
    """
    Enumerate nn.Module subclasses in `modeling_mod` that look like decoder
    layers.  Uses _no_split_modules on the main model class as the primary
    signal; falls back to scanning class names.

    Returns a list of (name, cls) pairs, may be empty.
    """
    # Primary: _no_split_modules lists exactly the classes that should not be
    # split across devices — these are the decoder-layer classes.
    model_cls = getattr(modeling_mod, arch, None)
    if model_cls is not None and hasattr(model_cls, "_no_split_modules"):
        found = []
        for cls_name in model_cls._no_split_modules:
            cls = getattr(modeling_mod, cls_name, None)
            if not (cls is not None and isinstance(cls, type) and issubclass(cls, nn.Module)):
                # Class may be defined in a fla submodule not re-exported on modeling_mod.
                for m in sys.modules.values():
                    if m is None:
                        continue
                    try:
                        candidate = getattr(m, cls_name, None)
                        if candidate is not None and isinstance(candidate, type) and issubclass(candidate, nn.Module):
                            cls = candidate
                            break
                    except Exception:
                        continue
            if cls is not None and isinstance(cls, type) and issubclass(cls, nn.Module):
                found.append((cls_name, cls))
        if found:
            return found

    # No safe fallback — dir() scanning is too broad and picks up unrelated
    # transformers classes (e.g. BertGenerationDecoder) that happen to be
    # imported into the same namespace.
    return []


@register
class Qwen35Driver(ModelDriver):

    architectures = [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3ForCausalLM",
        "Qwen3_5ForCausalLM",
    ]

    # ── Config ───────────────────────────────────────────────────────────────

    def normalize_config(self, config) -> None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            return
        for attr in _PROXY_ATTRS:
            try:
                getattr(config, attr)
            except AttributeError:
                try:
                    setattr(config, attr, getattr(text_cfg, attr))
                except AttributeError:
                    pass

    # ── Layer discovery ───────────────────────────────────────────────────────

    @property
    def layer_module_paths(self) -> list[str]:
        return [
            "model.language_model.layers",
        ] + super().layer_module_paths

    def get_layer_classes(self, config) -> Optional[List[Type]]:
        """
        Build the per-index class list without any model instantiation.

        Resolution strategy
        -------------------
        1. Read layer_types from config.text_config — this tells us whether
           each position is "linear_attention" or "full_attention".

        2. Find the trust_remote_code modeling module in sys.modules by
           searching for the module that exports the main architecture class.
           LayerLoader calls AutoModel.from_config before us, which imports
           the module as a side-effect even when the layer list is empty.

        3. From that module, collect decoder-layer classes via
           _no_split_modules (preferred) or name-pattern scanning.

        4a. Single class found → use it for every layer (unified decoder).
        4b. Two classes found → map to layer_types via name heuristics
            (linear/gated-delta → linear_attention, attention → full_attention).
        4c. Neither → return None so LayerLoader raises a clear error.
        """
        text_cfg = getattr(config, "text_config", config)
        layer_types: Optional[list] = getattr(text_cfg, "layer_types", None)
        if not layer_types:
            return None

        num_layers = len(layer_types)
        arch = (getattr(config, "architectures", None) or [""])[0]

        # ── Find the modeling module ──────────────────────────────────────
        modeling_mod = _find_modeling_module(arch)
        if modeling_mod is None:
            print(f"   ⚠️ [Qwen35Driver] get_layer_classes: "
                  f"could not find modeling module for {arch} in sys.modules")
            return None

        # ── Collect candidate decoder-layer classes ───────────────────────
        candidates = _decoder_layer_classes_from_module(modeling_mod, arch)

        if not candidates:
            print(f"   ⚠️ [Qwen35Driver] get_layer_classes: "
                  f"no decoder-layer classes found in {getattr(modeling_mod, '__name__', repr(modeling_mod))!r}")
            return None

        # ── Case 1: single unified decoder layer ──────────────────────────
        if len(candidates) == 1:
            cls = candidates[0][1]
            print(f"   ✅ [Qwen35Driver] unified decoder layer: {candidates[0][0]}")
            return [cls] * num_layers

        # ── Case 2: multiple classes — map to linear_attention / full_attention
        # Match by name heuristics.
        linear_cls: Optional[type] = None
        attn_cls: Optional[type] = None

        for cls_name, cls in candidates:
            lower = cls_name.lower()
            if linear_cls is None and any(h in lower for h in _LINEAR_HINTS):
                linear_cls = cls
            elif attn_cls is None and any(h in lower for h in _ATTN_HINTS):
                attn_cls = cls

        if linear_cls is not None and attn_cls is not None:
            print(
                f"   ✅ [Qwen35Driver] hybrid mapping: "
                f"linear_attention → {linear_cls.__name__}, "
                f"full_attention → {attn_cls.__name__}"
            )
            return [
                linear_cls if lt == "linear_attention" else attn_cls
                for lt in layer_types
            ]

        # Heuristics inconclusive — assign positionally by distinct layer_types.
        distinct_types = list(dict.fromkeys(layer_types))  # ordered, deduplicated
        if len(distinct_types) <= len(candidates):
            type_to_cls = {lt: candidates[i][1] for i, lt in enumerate(distinct_types)}
            print(
                f"   ✅ [Qwen35Driver] positional mapping: "
                + ", ".join(f"{lt} → {candidates[i][0]}"
                            for i, lt in enumerate(distinct_types))
            )
            return [type_to_cls[lt] for lt in layer_types]

        # Last resort: use the first candidate for all layers.
        cls = candidates[0][1]
        print(f"   ⚠️ [Qwen35Driver] using {candidates[0][0]} for all layers (fallback)")
        return [cls] * num_layers

    def get_layer_class_from_keys(self, weight_keys: List[str]) -> Optional[Type]:
        """
        Determine the layer class from safetensors weight key names.

        "linear_attn" in keys → GatedDeltaNet layer
        "self_attn"   in keys → standard attention layer
        """
        has_linear_attn = any("linear_attn" in k for k in weight_keys)
        has_self_attn   = any("self_attn"   in k for k in weight_keys)

        if not has_linear_attn and not has_self_attn:
            return None

        arch = self.architectures[0]
        modeling_mod = _find_modeling_module(arch)
        if modeling_mod is None:
            return None

        candidates = _decoder_layer_classes_from_module(modeling_mod, arch)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][1]

        for cls_name, cls in candidates:
            lower = cls_name.lower()
            if has_linear_attn and any(h in lower for h in _LINEAR_HINTS):
                return cls
            if has_self_attn and any(h in lower for h in _ATTN_HINTS):
                return cls

        return candidates[0][1]

    # ── MoE ──────────────────────────────────────────────────────────────────

    def is_moe(self, config) -> bool:
        return False

    # ── LM head ──────────────────────────────────────────────────────────────

    def resolve_lm_head(
        self,
        model,
        weight_map: dict,
        backbone_weight_prefix: str,
    ) -> Optional[Tuple[Any, str]]:
        checkpoint_prefix = None
        for wk in weight_map:
            if wk.endswith("lm_head.weight"):
                checkpoint_prefix = wk[: -len(".weight")]
                break

        if checkpoint_prefix is None:
            return None

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

        return None

    # ── Inference ─────────────────────────────────────────────────────────────

    def init_rope(self, config, device: str, dtype) -> None:
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
