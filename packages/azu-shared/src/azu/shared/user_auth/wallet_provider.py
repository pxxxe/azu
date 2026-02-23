"""
azu/shared/user_auth/wallet_provider.py

Default UserAuthProvider.

Expects:  Authorization: Bearer <wallet_address>

The wallet address is returned directly as the ledger identity, preserving
the original API contract exactly. No Redis round-trip, no secret required.
"""

from fastapi import HTTPException, Request

from .base import UserAuthProvider


class WalletAuthProvider(UserAuthProvider):
    """
    Resolves a wallet address supplied as a Bearer token.

    This is the default provider and reproduces the original behaviour of
    _extract_pubkey() from openai_adapter.py verbatim.
    """

    async def authenticate(self, request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header required: Bearer <wallet_address>",
            )
        pubkey = auth.removeprefix("Bearer ").strip()
        if not pubkey:
            raise HTTPException(status_code=401, detail="Empty Bearer token.")
        return pubkey
