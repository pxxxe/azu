"""
shared/auth/factory.py

Factory for instantiating the configured auth provider.

Provider selection follows the same convention as the payment layer:
    AUTH_PROVIDER=hmac   (default)

Adding a new provider:
    1. Create shared/auth/my_provider.py implementing AuthProvider.
    2. Add a branch here.
    3. Set AUTH_PROVIDER=my_provider in the environment.
"""

import os
from typing import Optional

from .base import AuthProvider
from .hmac_provider import HMACAuthProvider


def get_auth_provider(
    provider_type: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> AuthProvider:
    """
    Return the configured AuthProvider instance.

    Args:
        provider_type: Override for AUTH_PROVIDER env var.
        secret_key:    Override for AUTH_SECRET_KEY env var.

    Returns:
        An AuthProvider ready for use.

    Raises:
        ValueError: If the requested provider type is unknown.
    """
    provider_type = provider_type or os.getenv("AUTH_PROVIDER", "hmac")
    secret_key = secret_key or os.getenv("AUTH_SECRET_KEY")

    if provider_type == "hmac":
        return HMACAuthProvider(secret_key=secret_key)

    raise ValueError(
        f"Unknown AUTH_PROVIDER '{provider_type}'. "
        "Supported values: 'hmac'. "
        "Add a new provider in shared/auth/factory.py."
    )


def is_auth_enabled() -> bool:
    """
    Return True if tensor-transfer authentication is active.

    Auth is enabled when AUTH_SECRET_KEY is set.  When disabled, all
    x-auth-token checks are skipped (useful for local dev / unit tests).
    """
    return bool(os.getenv("AUTH_SECRET_KEY"))
