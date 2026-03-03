"""
Driver registry.

Usage:
    from azu.shared.model_drivers.registry import register, get_driver

    @register
    class MyDriver(ModelDriver):
        architectures = ["MyModelForCausalLM"]
        ...

    driver = get_driver(config)   # returns correct driver for the config
"""

from __future__ import annotations

from typing import Dict, Optional, Type, TYPE_CHECKING

from .base import ModelDriver

if TYPE_CHECKING:
    pass

# architecture string → driver class
_DRIVER_CLASSES: Dict[str, Type[ModelDriver]] = {}

# architecture string → driver singleton instance
_DRIVER_INSTANCES: Dict[str, ModelDriver] = {}

# singleton default driver instance
_DEFAULT_INSTANCE: Optional[ModelDriver] = None


def register(driver_cls: Type[ModelDriver]) -> Type[ModelDriver]:
    """
    Class decorator that registers a driver for all architectures it declares.

        @register
        class MixtralDriver(ModelDriver):
            architectures = ["MixtralForCausalLM"]
    """
    for arch in driver_cls.architectures:
        _DRIVER_CLASSES[arch] = driver_cls
    return driver_cls


def get_driver(config) -> ModelDriver:
    """
    Return the driver instance for a given model config.

    Looks up config.architectures[0] in the registry.  Falls back to
    DefaultDriver if no specific driver is registered for that architecture.

    Drivers are singletons — instantiated once per architecture per process.
    """
    global _DEFAULT_INSTANCE

    archs = getattr(config, "architectures", None) or []
    for arch in archs:
        if arch in _DRIVER_CLASSES:
            if arch not in _DRIVER_INSTANCES:
                _DRIVER_INSTANCES[arch] = _DRIVER_CLASSES[arch]()
            return _DRIVER_INSTANCES[arch]

    # No registered driver — use DefaultDriver
    if _DEFAULT_INSTANCE is None:
        from .drivers.default import DefaultDriver
        _DEFAULT_INSTANCE = DefaultDriver()
    return _DEFAULT_INSTANCE


def load_all_drivers() -> None:
    """
    Import every built-in driver module so they self-register via @register.
    Call this once at process startup (done automatically by __init__.py).
    """
    from .drivers import default, mixtral, qwen2_5, qwen3_5  # noqa: F401


def list_registered() -> list[str]:
    """Return all currently registered architecture strings (for debugging)."""
    return sorted(_DRIVER_CLASSES.keys())
