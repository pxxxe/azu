"""
Qwen25Driver — Qwen 2.5 family (dense, flat config).

Qwen2.5 follows standard transformers conventions closely enough that the
DefaultDriver would also work.  We register it explicitly so future
Qwen2.5-specific quirks have a home without touching other drivers.
"""

from __future__ import annotations

from ..base import ModelDriver
from ..registry import register


@register
class Qwen25Driver(ModelDriver):

    architectures = [
        "Qwen2ForCausalLM",
        "Qwen2_5ForCausalLM",
        "Qwen2MoeForCausalLM",   # Qwen2-MoE variant
    ]

    def is_moe(self, config) -> bool:
        # Qwen2-MoE uses num_experts
        val = getattr(config, "num_experts", None)
        return bool(val and int(val) > 1)

    def moe_router_path(self, layer) -> str:
        return "mlp"

    def num_experts(self, config) -> int:
        val = getattr(config, "num_experts", None)
        if val and int(val) > 1:
            return int(val)
        return super().num_experts(config)
