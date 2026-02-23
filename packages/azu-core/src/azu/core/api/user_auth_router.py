"""
azu/core/api/user_auth_router.py

API key management endpoints.

Only relevant when USER_AUTH_PROVIDER=api_key.  When using the default
wallet provider these routes are still mounted but are simply unused.

Endpoints:

    POST   /auth/api-keys        Issue a new API key for a wallet address.
    DELETE /auth/api-keys        Revoke an existing API key.

Note: in production you will want to gate POST /auth/api-keys behind an
admin secret or on-chain signature check. The route is intentionally left
open here so the operator can layer their own access control on top.
"""

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from azu.shared.config import get_config
from azu.shared.user_auth.api_key_provider import APIKeyAuthProvider

user_auth_router = APIRouter(prefix="/auth", tags=["User Auth"])

config = get_config()


async def _get_redis() -> aioredis.Redis:
    return aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True,
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateAPIKeyReq(BaseModel):
    """Request body for issuing a new API key."""
    wallet_address: str


class RevokeAPIKeyReq(BaseModel):
    """Request body for revoking an API key."""
    api_key: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@user_auth_router.post("/api-keys", summary="Issue a new API key")
async def create_api_key(req: CreateAPIKeyReq):
    """
    Issue a new API key linked to wallet_address.

    The key is returned once in plaintext — store it securely.
    """
    r = await _get_redis()
    key = await APIKeyAuthProvider.create_key(r, req.wallet_address)
    return {"api_key": key, "wallet_address": req.wallet_address}


@user_auth_router.delete("/api-keys", summary="Revoke an API key")
async def revoke_api_key(req: RevokeAPIKeyReq):
    """
    Revoke an existing API key. The key is immediately invalidated.
    """
    r = await _get_redis()
    existed = await APIKeyAuthProvider.revoke_key(r, req.api_key)
    if not existed:
        raise HTTPException(status_code=404, detail="API key not found.")
    return {"status": "revoked"}
