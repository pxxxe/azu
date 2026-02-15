"""
Configuration module for the worker.
Contains environment variables and constants.
"""

import os

# HuggingFace token for model downloads (optional)
HF_TOKEN = os.getenv("HF_TOKEN")

# Scheduler connection
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")

# Registry URL for layer downloads
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")

# P2P networking configuration
P2P_PUBLIC_URL = os.getenv("P2P_PUBLIC_URL")
P2P_URL_TEMPLATE = os.getenv("P2P_URL_TEMPLATE")
P2P_PORT = int(os.getenv("P2P_PORT", "8003"))
P2P_TIMEOUT = int(os.getenv("P2P_TIMEOUT", "300"))

# Layer cache directory
LAYER_CACHE_DIR = os.getenv("LAYER_CACHE_DIR", "/app/layer_cache")

# P2P server settings
P2P_MAX_SIZE = 1024**3  # 1GB for MoE tensors
P2P_CONNECTION_RETRIES = 3
P2P_HANDSHAKE_RETRIES = 5
P2P_HANDSHAKE_TIMEOUT = 2

# Concurrency settings
MAX_DOWNLOAD_WORKERS = 8
MAX_DOWNLOAD_SEMAPHORE = 10

# VRAM settings
DEFAULT_CPU_VRAM_MB = 32000
