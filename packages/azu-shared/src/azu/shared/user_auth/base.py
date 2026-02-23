"""
azu/shared/user_auth/base.py

Abstract interface for the user-facing Bring-Your-Own-Auth provider.

This is distinct from azu.shared.auth, which handles inter-worker
HMAC tokens for P2P tensor transfer. This layer sits at the API
boundary and resolves an incoming HTTP request to a ledger identity
(wallet address or equivalent key) used for balance checks and job
attribution.

Implementing a custom provider:
    1. Subclass UserAuthProvider.
    2. Register it in factory.py.
    3. Set USER_AUTH_PROVIDER=<your_value> in the environment.
"""

from abc import ABC, abstractmethod

from fastapi import Request


class UserAuthProvider(ABC):
    """
    Pluggable auth provider for user-facing API requests.

    A single method contract: receive the raw FastAPI Request, return the
    ledger identity string (wallet address or any unique key the ledger
    understands), or raise HTTPException(401) on failure.
    """

    @abstractmethod
    async def authenticate(self, request: Request) -> str:
        """
        Authenticate the request and return the caller's ledger identity.

        Args:
            request: The incoming FastAPI request.

        Returns:
            A non-empty string that identifies the user in the ledger
            (e.g. a wallet address, or a stable opaque user ID).

        Raises:
            fastapi.HTTPException(401): If the credential is missing,
                malformed, expired, or otherwise invalid.
        """
