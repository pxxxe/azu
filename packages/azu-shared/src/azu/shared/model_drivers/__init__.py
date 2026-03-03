"""
azu model drivers — per-architecture adapter layer.

The network core (LayerStore, LayerLoader, ModelManager) stays model-agnostic.
All architecture-specific knowledge lives in driver classes registered here.

Quick start
-----------
Adding support for a new model family:

    # packages/azu-shared/src/azu/shared/model_drivers/drivers/my_model.py

    from azu.shared.model_drivers.base import ModelDriver
    from azu.shared.model_drivers.registry import register

    @register
    class MyModelDriver(ModelDriver):
        architectures = ["MyModelForCausalLM"]

        def normalize_config(self, config) -> None:
            # proxy nested attrs if needed
            ...

        def is_moe(self, config) -> bool:
            return False

Then add the import to drivers/__init__.py so it registers on startup.
"""

from .base import ModelDriver
from .registry import get_driver, register, load_all_drivers, list_registered

# Auto-register all built-in drivers when the package is imported.
load_all_drivers()

__all__ = [
    "ModelDriver",
    "get_driver",
    "register",
    "load_all_drivers",
    "list_registered",
]
