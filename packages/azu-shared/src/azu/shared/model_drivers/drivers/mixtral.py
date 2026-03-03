"""
MixtralDriver — Mixtral MoE family.

Covers MixtralForCausalLM.  Key differences from default:
  - MoE architecture with router at layer.mlp (aliased to block_sparse_moe
    in some checkpoint variants — handled by the base resolve_weight_key).
  - num_experts from config.num_local_experts.
"""

from __future__ import annotations

from ..base import ModelDriver
from ..registry import register


@register
class MixtralDriver(ModelDriver):

    architectures = ["MixtralForCausalLM"]

    def is_moe(self, config) -> bool:
        return True

    def moe_router_path(self, layer) -> str:
        # Mixtral stores the MoE block at layer.mlp.
        # Checkpoint keys may use block_sparse_moe — resolve_weight_key handles that alias.
        return "mlp"

    def num_experts(self, config) -> int:
        val = getattr(config, "num_local_experts", None)
        if val and int(val) > 1:
            return int(val)
        return super().num_experts(config)
