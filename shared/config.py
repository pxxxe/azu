import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SOLANA_RPC_URL: str
    PLATFORM_WALLET_PUBKEY: str
    SCHEDULER_PRIVATE_KEY: str
    REDIS_HOST: str
    REDIS_PORT: int

    class Config:
        env_file = ".env"

settings = Settings()
