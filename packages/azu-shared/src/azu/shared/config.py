"""
Configuration Module

Central configuration for the azu decentralized inference network.
Loads environment variables and provides configuration values.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False

    @classmethod
    def from_env(cls) -> "RedisConfig":
        return cls(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            db=int(os.environ.get("REDIS_DB", 0)),
            password=os.environ.get("REDIS_PASSWORD"),
            ssl=os.environ.get("REDIS_SSL", "false").lower() == "true"
        )


@dataclass
class PaymentConfig:
    """Payment layer configuration."""
    provider: str = "hyperliquid"
    confirmations_required: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    # Payout settings
    payout_threshold: float = 0.001
    # Hyperliquid specific
    hyperliquid_rpc_url: str = "https://rpc.hyperliquid.xyz/evm"
    hyperliquid_address: Optional[str] = None
    # Solana specific
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    solana_address: Optional[str] = None
    # Internal ledger
    use_internal_ledger: bool = True

    @classmethod
    def from_env(cls) -> "PaymentConfig":
        return cls(
            provider=os.environ.get("PAYMENT_PROVIDER", "hyperliquid"),
            confirmations_required=int(os.environ.get("PAYMENT_CONFIRMATIONS", 1)),
            max_retries=int(os.environ.get("PAYMENT_MAX_RETRIES", 3)),
            retry_delay=float(os.environ.get("PAYMENT_RETRY_DELAY", 1.0)),
            payout_threshold=float(os.environ.get("PAYOUT_THRESHOLD", 0.001)),
            hyperliquid_rpc_url=os.environ.get(
                "HYPERLIQUID_RPC_URL",
                "https://api.hyperliquid.xyz"
            ),
            hyperliquid_address=os.environ.get("HYPERLIQUID_ADDRESS"),
            solana_rpc_url=os.environ.get(
                "SOLANA_RPC_URL",
                "https://api.mainnet-beta.solana.com"
            ),
            solana_address=os.environ.get("SOLANA_ADDRESS"),
            use_internal_ledger=os.environ.get(
                "USE_INTERNAL_LEDGER",
                "true"
            ).lower() == "true"
        )


@dataclass
class WorkerConfig:
    """Worker-related configuration."""
    scheduler_url: str = "ws://localhost:8001/ws/worker"
    registry_url: str = "http://localhost:8002"
    p2p_public_url: Optional[str] = None
    p2p_url_template: Optional[str] = None
    # Worker payment
    worker_address: Optional[str] = None

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        return cls(
            scheduler_url=os.environ.get(
                "SCHEDULER_URL",
                "ws://localhost:8001/ws/worker"
            ),
            registry_url=os.environ.get(
                "REGISTRY_URL",
                "http://localhost:8002"
            ),
            p2p_public_url=os.environ.get("P2P_PUBLIC_URL"),
            p2p_url_template=os.environ.get("P2P_URL_TEMPLATE"),
            worker_address=os.environ.get("WORKER_ADDRESS")
        )


@dataclass
class RegistryConfig:
    """Registry service configuration."""
    host: str = "0.0.0.0"
    port: int = 8002
    storage_path: str = "/app/models"
    hf_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        return cls(
            host=os.environ.get("REGISTRY_HOST", "0.0.0.0"),
            port=int(os.environ.get("REGISTRY_PORT", 8002)),
            storage_path=os.environ.get("MODEL_STORAGE_PATH", "/app/models"),
            hf_token=os.environ.get("HF_TOKEN")
        )


@dataclass
class SchedulerConfig:
    """Scheduler service configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    worker_timeout: int = 300
    job_timeout: int = 600
    max_retries: int = 3
    # Payment-related
    payout_threshold: float = 0.001
    payout_batch_size: int = 10

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        return cls(
            host=os.environ.get("SCHEDULER_HOST", "0.0.0.0"),
            port=int(os.environ.get("SCHEDULER_PORT", 8001)),
            worker_timeout=int(os.environ.get("WORKER_TIMEOUT", 300)),
            job_timeout=int(os.environ.get("JOB_TIMEOUT", 600)),
            max_retries=int(os.environ.get("MAX_RETRIES", 3)),
            payout_threshold=float(os.environ.get("PAYOUT_THRESHOLD", 0.001)),
            payout_batch_size=int(os.environ.get("PAYOUT_BATCH_SIZE", 10))
        )


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit: int = 100

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            host=os.environ.get("API_HOST", "0.0.0.0"),
            port=int(os.environ.get("API_PORT", 8000)),
            cors_origins=os.environ.get(
                "CORS_ORIGINS",
                "*"
            ).split(","),
            rate_limit=int(os.environ.get("RATE_LIMIT", 100))
        )


@dataclass
class Config:
    """Main configuration container."""
    redis: RedisConfig = field(default_factory=RedisConfig.from_env)
    payment: PaymentConfig = field(default_factory=PaymentConfig.from_env)
    worker: WorkerConfig = field(default_factory=WorkerConfig.from_env)
    registry: RegistryConfig = field(default_factory=RegistryConfig.from_env)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig.from_env)
    api: APIConfig = field(default_factory=APIConfig.from_env)

    # Legacy compatibility
    @property
    def REDIS_HOST(self) -> str:
        return self.redis.host

    @property
    def REDIS_PORT(self) -> int:
        return self.redis.port

    @property
    def SCHEDULER_URL(self) -> str:
        return self.worker.scheduler_url

    @property
    def REGISTRY_URL(self) -> str:
        return self.worker.registry_url

    @property
    def P2P_PUBLIC_URL(self) -> Optional[str]:
        return self.worker.p2p_public_url

    @property
    def HF_TOKEN(self) -> Optional[str]:
        return self.registry.hf_token


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global _config
    _config = Config()
    return _config
