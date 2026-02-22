"""
shared/auth/hmac_provider.py

HMAC-SHA256 auth provider (default).

Token = HMAC-SHA256(AUTH_SECRET_KEY, job_id).hexdigest()

The scheduler generates the token using the secret key and distributes it to
workers via JOB_START.  Workers verify incoming x-auth-token headers by
constant-time comparison against the stored token — no secret key required
on the worker side for this scheme.

If AUTH_SECRET_KEY is not set the provider can still be instantiated for the
worker-side verify path (secret_key=None).  generate_token() will raise in
that case.
"""

import hashlib
import hmac
from typing import Optional

from .base import AuthProvider


class HMACAuthProvider(AuthProvider):
    """
    HMAC-SHA256 based auth provider.

    Scheduler usage (secret_key required):
        provider = HMACAuthProvider(secret_key="my-secret")
        token    = provider.generate_token(job_id)

    Worker usage (secret_key not required — constant-time compare only):
        provider = HMACAuthProvider()
        ok       = provider.verify_token(received_token, ctx.auth_token)
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialise the HMAC provider.

        Args:
            secret_key: The shared HMAC secret.  Required for generate_token().
                        Not required for verify_token().
        """
        if secret_key:
            self._key: Optional[bytes] = (
                secret_key.encode("utf-8")
                if isinstance(secret_key, str)
                else secret_key
            )
        else:
            self._key = None

    def generate_token(self, job_id: str) -> str:
        """
        Produce HMAC-SHA256(secret_key, job_id) as a hex string.

        Args:
            job_id: The unique job identifier.

        Returns:
            64-character hex string token.

        Raises:
            RuntimeError: If AUTH_SECRET_KEY was not supplied at construction time.
        """
        if self._key is None:
            raise RuntimeError(
                "HMACAuthProvider cannot generate tokens without a secret key. "
                "Set AUTH_SECRET_KEY on the scheduler."
            )
        return hmac.new(
            self._key, job_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_token(self, received_token: str, expected_token: str) -> bool:
        """
        Constant-time comparison of received_token against expected_token.

        Does not require the secret key — the expected_token was already
        produced by the scheduler and stored in JobContext.

        Args:
            received_token: Token from the x-auth-token HTTP header.
            expected_token: Token from JobContext.auth_token.

        Returns:
            True if both tokens are non-empty and equal.
        """
        if not received_token or not expected_token:
            return False
        return hmac.compare_digest(received_token, expected_token)
