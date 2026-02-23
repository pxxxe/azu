"""
azu/shared/user_auth/api_key_provider.py

API-key-based UserAuthProvider.

Keys are stored in Redis as:
    user_auth:api_key:<key>  →  <wallet_address>

The incoming request carries:
    Authorization: Bearer <api_key>

The provider resolves the key to the linked wallet address, which becomes
the ledger identity. The end-user never needs to know their wallet address.

Keys are issued and revoked via the static helpers below, which are called
from the /auth/api-keys management endpoints in the core API.
"""

import secrets

import redis.asyncio as aioredis
from fastapi import HTTPException, Request

from .base import UserAuthProvider

_KEY_PREFIX = "user_auth:api_key:"


class APIKeyAuthProvider(UserAuthProvider):
    """
    Resolves opaque API keys to wallet/ledger identities via Redis.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    async def authenticate(self, request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header required: Bearer <api_key>",
            )
        api_key = auth.removeprefix("Bearer ").strip()
        if not api_key:
            raise HTTPException(status_code=401, detail="Empty Bearer token.")

        wallet = await self._redis.get(f"{_KEY_PREFIX}{api_key}")
        if not wallet:
            raise HTTPException(status_code=401, detail="Invalid or expired API key.")

        return wallet

    # ------------------------------------------------------------------
    # Key lifecycle helpers (called by user_auth_router, not at auth time)
    # ------------------------------------------------------------------

    @staticmethod
    async def create_key(redis_client: aioredis.Redis, wallet_address: str) -> str:
        """
        Issue a new cryptographically random API key linked to wallet_address.

        Returns the key — shown once. Store it securely; it is not retrievable
        from Redis because the raw value is never stored elsewhere.
        """
        key = secrets.token_urlsafe(32)
        await redis_client.set(f"{_KEY_PREFIX}{key}", wallet_address)
        return key

    @staticmethod
    async def revoke_key(redis_client: aioredis.Redis, api_key: str) -> bool:
        """
        Revoke an API key.

        Returns:
            True if the key existed and was deleted, False if it was not found.
        """
        deleted = await redis_client.delete(f"{_KEY_PREFIX}{api_key}")
        return bool(deleted)
