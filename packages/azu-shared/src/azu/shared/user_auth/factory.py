"""
azu/shared/user_auth/factory.py

Factory for the user-facing auth provider.

Provider selection via USER_AUTH_PROVIDER environment variable:

    USER_AUTH_PROVIDER=wallet     (default) — Bearer <wallet_address>
    USER_AUTH_PROVIDER=api_key              — Bearer <api_key> resolved via Redis

Adding a custom provider:
    1. Subclass UserAuthProvider in a new module under azu/shared/user_auth/.
    2. Add a branch in get_user_auth_provider() below.
    3. Set USER_AUTH_PROVIDER=<your_value> in the environment.
"""

import os
from typing import Optional

from .base import UserAuthProvider
from .wallet_provider import WalletAuthProvider

_provider: Optional[UserAuthProvider] = None


def get_user_auth_provider(
    provider_type: Optional[str] = None,
) -> UserAuthProvider:
    """
    Return the configured UserAuthProvider singleton.

    Args:
        provider_type: Explicit override; falls back to USER_AUTH_PROVIDER
                       env var, then 'wallet'.

    Returns:
        A ready-to-use UserAuthProvider.

    Raises:
        ValueError: If the requested provider type is not recognised.
    """
    global _provider

    # Return the cached singleton unless an explicit override is given.
    if _provider is not None and provider_type is None:
        return _provider

    resolved = provider_type or os.getenv("USER_AUTH_PROVIDER", "wallet")

    if resolved == "wallet":
        _provider = WalletAuthProvider()
        return _provider

    if resolved == "api_key":
        import redis.asyncio as aioredis

        from azu.shared.config import get_config

        from .api_key_provider import APIKeyAuthProvider

        cfg = get_config()
        r = aioredis.Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            db=cfg.redis.db,
            decode_responses=True,
        )
        _provider = APIKeyAuthProvider(redis_client=r)
        return _provider

    raise ValueError(
        f"Unknown USER_AUTH_PROVIDER '{resolved}'. "
        "Supported values: 'wallet', 'api_key'. "
        "To add a new provider, subclass UserAuthProvider and register it "
        "in azu/shared/user_auth/factory.py."
    )


def reset_user_auth_provider() -> None:
    """
    Reset the singleton instance.

    Useful in tests that need to swap providers between cases.
    """
    global _provider
    _provider = None
