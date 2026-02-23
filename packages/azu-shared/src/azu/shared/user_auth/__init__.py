"""
azu/shared/user_auth

Bring-Your-Own-Auth layer for user-facing API requests.

This is separate from azu.shared.auth, which handles inter-worker
HMAC tokens for P2P tensor transfer.

Usage (in API route handlers):

    from azu.shared.user_auth import get_user_auth_provider

    user_pubkey = await get_user_auth_provider().authenticate(request)

Provider is selected at startup via USER_AUTH_PROVIDER env var:

    USER_AUTH_PROVIDER=wallet     (default) — Bearer <wallet_address>
    USER_AUTH_PROVIDER=api_key              — Bearer <api_key> via Redis
"""

from .api_key_provider import APIKeyAuthProvider
from .base import UserAuthProvider
from .factory import get_user_auth_provider, reset_user_auth_provider
from .wallet_provider import WalletAuthProvider

__all__ = [
    "UserAuthProvider",
    "WalletAuthProvider",
    "APIKeyAuthProvider",
    "get_user_auth_provider",
    "reset_user_auth_provider",
]
