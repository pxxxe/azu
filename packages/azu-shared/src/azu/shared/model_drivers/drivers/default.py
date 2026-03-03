"""
DefaultDriver — fallback for any standard transformers causal LM.

Handles flat configs (no nesting), model.layers layout, dense-only,
standard RoPE.  Works out of the box for LLaMA, Mistral, Falcon, GPT-NeoX,
and any model that follows transformers conventions without VLM wrapping.

This driver is never directly registered (no architectures list) — it is
instantiated by the registry as a singleton fallback when no specific driver
matches.
"""

from __future__ import annotations

from ..base import ModelDriver


class DefaultDriver(ModelDriver):
    """
    Fallback driver for any standard transformers architecture.
    No overrides needed — all base class defaults apply.
    """

    # Not registered for specific architectures.
    # The registry returns this when no specific driver matches.
    architectures: list[str] = []
