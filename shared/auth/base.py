"""
shared/auth/base.py

Abstract interface for the Bring-Your-Own-Auth provider.

Scheduler calls generate_token() to produce a per-job token and distributes
it to all workers via the JOB_START WebSocket message.

Workers call verify_token() on every incoming P2P request to confirm the
peer was authorised for this job by the scheduler.

Implementing a custom provider:
    1. Subclass AuthProvider.
    2. Register it in factory.py.
    3. Set AUTH_PROVIDER=<name> in the environment.
"""

from abc import ABC, abstractmethod


class AuthProvider(ABC):
    """
    Pluggable authentication provider for inter-worker tensor transfers.

    Scheduler side: generate_token(job_id) → opaque token string
    Worker side:    verify_token(received_token, expected_token) → bool
    """

    @abstractmethod
    def generate_token(self, job_id: str) -> str:
        """
        Generate an auth token for a job.

        Called by the scheduler exactly once per job.  The returned token
        is forwarded to every participating worker inside the JOB_START
        WebSocket message and stored in JobContext.auth_token.

        Args:
            job_id: The unique job identifier.

        Returns:
            An opaque, URL-safe string token.
        """

    @abstractmethod
    def verify_token(self, received_token: str, expected_token: str) -> bool:
        """
        Verify an auth token on the receiving worker side.

        For symmetric schemes (e.g. HMAC) expected_token is the value stored
        in JobContext (originally issued by the scheduler).  For asymmetric
        schemes (e.g. JWT) expected_token may be ignored and the token is
        self-validated.

        Args:
            received_token: The value from the incoming x-auth-token HTTP header.
            expected_token: The token stored in JobContext.auth_token.

        Returns:
            True if the token is valid, False otherwise.
        """
