"""
Worker package for decentralized inference network.
Contains modules for P2P communication, model management, and layer processing.
"""

from config import (
    HF_TOKEN,
    SCHEDULER_URL,
    REGISTRY_URL,
    P2P_PUBLIC_URL,
    P2P_URL_TEMPLATE,
    P2P_PORT,
    P2P_TIMEOUT,
    LAYER_CACHE_DIR,
    P2P_MAX_SIZE,
    P2P_CONNECTION_RETRIES,
    P2P_HANDSHAKE_RETRIES,
    P2P_HANDSHAKE_TIMEOUT,
    MAX_DOWNLOAD_WORKERS,
    MAX_DOWNLOAD_SEMAPHORE,
    DEFAULT_CPU_VRAM_MB,
)

from job_context import JobContext
from p2p_protocol import P2PProtocol
from p2p_server import P2PServer
from model_manager import ModelManager
from layer_processor import (
    DenseLayerProcessor,
    MoERouterProcessor,
    MoEExpertProcessor,
)
from layer_loader import LayerLoader

__all__ = [
    # Config
    "HF_TOKEN",
    "SCHEDULER_URL",
    "REGISTRY_URL",
    "P2P_PUBLIC_URL",
    "P2P_URL_TEMPLATE",
    "P2P_PORT",
    "P2P_TIMEOUT",
    "LAYER_CACHE_DIR",
    "P2P_MAX_SIZE",
    "P2P_CONNECTION_RETRIES",
    "P2P_HANDSHAKE_RETRIES",
    "P2P_HANDSHAKE_TIMEOUT",
    "MAX_DOWNLOAD_WORKERS",
    "MAX_DOWNLOAD_SEMAPHORE",
    "DEFAULT_CPU_VRAM_MB",
    # Core classes
    "JobContext",
    "P2PProtocol",
    "P2PServer",
    "ModelManager",
    "DenseLayerProcessor",
    "MoERouterProcessor",
    "MoEExpertProcessor",
    "LayerLoader",
]
