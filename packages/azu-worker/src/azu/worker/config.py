"""
Configuration module for the worker.
Contains environment variables and constants.
"""

import os
import time

# HuggingFace token for model downloads (optional)
HF_TOKEN = os.getenv("HF_TOKEN")

# Scheduler connection
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")

# HTTP base URL of the scheduler — used by serverless workers to call
# POST /worker/ready and POST /worker/result instead of the WebSocket.
# Derived automatically from SCHEDULER_URL if not set explicitly.
def _derive_scheduler_http_url() -> str:
    explicit = os.getenv("SCHEDULER_HTTP_URL")
    if explicit:
        return explicit.rstrip("/")
    ws = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
    base = ws.replace("wss://", "https://").replace("ws://", "http://")
    if "/ws" in base:
        base = base[:base.index("/ws")]
    return base.rstrip("/")

SCHEDULER_HTTP_URL = _derive_scheduler_http_url()

# Worker mode: "persistent" (default) keeps a long-lived WebSocket to the
# scheduler. "serverless" registers via HTTP, polls scheduler for control messages
# via GET /worker/poll/{worker_id}, and reports results via POST /worker/result.
WORKER_MODE = os.getenv("WORKER_MODE", "persistent").lower()

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
# Serverless Configuration (NEW)
# ============================================================================

# WAKE_URL: URL used by the scheduler to wake a dormant serverless worker.
# The worker reports this URL to the scheduler on registration so the scheduler
# can boot a cold worker when needed (scale-from-zero).
# Set this to whatever your platform uses to cold-start a new worker container.
# If not set, scale-from-zero wake is disabled (worker must be pre-running).
WAKE_URL = os.getenv("WAKE_URL")

# IDLE_TIMEOUT: Seconds of inactivity before the worker self-terminates.
# Enables scale-to-zero for cost savings.  The idle watchdog monitors poll activity
# and exits cleanly after IDLE_TIMEOUT seconds without receiving a job.
# Set to 0 to disable (worker runs forever).
# Default: 300 seconds (5 minutes)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))

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

# ============================================================================
# Auth Configuration (tensor transfer security)
# ============================================================================

# Auth provider type. Selects the BYOA token scheme used to authenticate
# inter-worker tensor transfers. Supported: 'hmac' (default)
AUTH_PROVIDER = os.getenv("AUTH_PROVIDER", "hmac")

# Shared secret used by the scheduler to generate per-job HMAC tokens.
# Workers do NOT need this — they verify by comparing the token received
# in JOB_START against the x-auth-token header on incoming P2P requests.
# When unset, authentication is disabled (dev / local mode).
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
