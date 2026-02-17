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

# ============================================================================
# Payment Configuration
# ============================================================================

# Payment provider type ('hyperliquid' or 'solana')
PAYMENT_PROVIDER = os.getenv("PAYMENT_PROVIDER", "hyperliquid")

# Worker wallet private key
# If not provided, a new wallet will be generated
# Format: JSON array like "[0,0,0,0,...]" or hex "0x..."
WORKER_PRIVATE_KEY = os.getenv("WORKER_PRIVATE_KEY")

# API URL for balance checks and withdrawals
# If not set, will use API_URL or default to localhost:8000
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Minimum balance threshold for auto-withdrawal (in token units)
# Set to 0 to disable auto-withdrawal
AUTO_WITHDRAWAL_THRESHOLD = float(os.getenv("AUTO_WITHDRAWAL_THRESHOLD", "0.0"))

# Withdrawal destination address (for auto-withdrawal)
# If not set, auto-withdrawal is disabled
WITHDRAWAL_ADDRESS = os.getenv("WITHDRAWAL_ADDRESS")
