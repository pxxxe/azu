"""
shared/auth

Bring-Your-Own-Auth abstraction for inter-worker tensor transfer security.

Usage (scheduler side — token generation):
    from shared.auth import get_auth_provider, is_auth_enabled
    if is_auth_enabled():
        provider = get_auth_provider()
        token = provider.generate_token(job_id)

Usage (worker side — token verification):
    from shared.auth import get_auth_provider, is_auth_enabled
    if is_auth_enabled():
        provider = get_auth_provider()
        ok = provider.verify_token(received_token, ctx.auth_token)
"""

from .base import AuthProvider
from .factory import get_auth_provider, is_auth_enabled
from .hmac_provider import HMACAuthProvider

__all__ = [
    "AuthProvider",
    "HMACAuthProvider",
    "get_auth_provider",
    "is_auth_enabled",
]
