"""
Worker Wallet Module

Manages the worker's cryptocurrency wallet for receiving payments.
Handles wallet generation, balance queries, and withdrawals.

The worker uses this module to:
- Generate or load a wallet address
- Register with the scheduler for payment
- Check accumulated earnings
- Withdraw funds to an external wallet
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from eth_account import Account


@dataclass
class WorkerWallet:
    """Worker wallet information."""
    address: str
    private_key: bytes
    provider_type: str


class WorkerWalletManager:
    """
    Manages the worker's wallet for receiving payments.

    The wallet can be configured via:
    1. WORKER_PRIVATE_KEY env var (existing wallet)
    2. Generate new wallet if not provided

    Workers register their payment address with the scheduler
    and receive credits directly to this address.
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        provider_type: str = "hyperliquid"
    ):
        """
        Initialize the worker wallet.

        Args:
            private_key: Optional private key (hex or JSON array)
            provider_type: Payment provider type
        """
        self.provider_type = provider_type

        if private_key:
            self._wallet = self._load_wallet(private_key)
        else:
            # Check environment for private key
            env_key = os.environ.get("WORKER_PRIVATE_KEY")
            if env_key:
                self._wallet = self._load_wallet(env_key)
            else:
                # Generate new wallet
                self._wallet = self._generate_wallet()
                print(f"💰 Generated new worker wallet: {self._wallet.address}")
                print(f"   Save this private key to persist the wallet:")
                print(f"   WORKER_PRIVATE_KEY={json.dumps(list(self._wallet.private_key))}")

    def _load_wallet(self, private_key: str) -> WorkerWallet:
        """Load wallet from private key."""
        if private_key.startswith('['):
            # JSON array format
            key_bytes = bytes(json.loads(private_key))
        elif private_key.startswith('0x'):
            # Hex format
            key_bytes = bytes.fromhex(private_key[2:])
        else:
            # Plain hex
            key_bytes = bytes.fromhex(private_key)

        account = Account.from_key(key_bytes)

        return WorkerWallet(
            address=account.address,
            private_key=key_bytes,
            provider_type=self.provider_type
        )

    def _generate_wallet(self) -> WorkerWallet:
        """Generate a new wallet."""
        account = Account.create()

        return WorkerWallet(
            address=account.address,
            private_key=account.key,
            provider_type=self.provider_type
        )

    @property
    def address(self) -> str:
        """Get the worker's payment address."""
        return self._wallet.address

    @property
    def private_key(self) -> bytes:
        """Get the worker's private key (for signing transactions)."""
        return self._wallet.private_key

    def get_private_key_hex(self) -> str:
        """Get private key as hex string."""
        return "0x" + self._wallet.private_key.hex()

    def get_private_key_json(self) -> str:
        """Get private key as JSON array."""
        return json.dumps(list(self._wallet.private_key))

    async def get_balance(self) -> dict:
        """
        Get the worker's current balance.

        Returns:
            Dict with available, locked, and total balance
        """
        from azu.shared.payments import get_payment_provider

        try:
            provider = get_payment_provider(
                provider_type=self.provider_type,
                private_key=self.get_private_key_hex()
            )
            balance = await provider.get_balance(self.address)

            return {
                "available": balance.available,
                "locked": balance.locked,
                "total": balance.total
            }
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {
                "available": 0.0,
                "locked": 0.0,
                "total": 0.0
            }

    async def withdraw(
        self,
        destination_address: str,
        amount: float
    ) -> dict:
        """
        Withdraw funds to an external address.

        Args:
            destination_address: Address to send funds to
            amount: Amount to withdraw

        Returns:
            Dict with tx_hash and status
        """
        from azu.shared.payments import get_payment_provider

        provider = get_payment_provider(
            provider_type=self.provider_type,
            private_key=self.get_private_key_hex()
        )

        payout_info = await provider.payout(
            recipient_address=destination_address,
            amount=amount,
            memo="Worker withdrawal"
        )

        return {
            "tx_hash": payout_info.tx_hash,
            "amount": payout_info.amount,
            "status": payout_info.status
        }

    async def get_ledger_balance(self) -> dict:
        """
        Get internal ledger balance (from scheduler).

        This is the preferred way to check earnings as the scheduler
        maintains the internal balance for workers.

        Returns:
            Dict with internal balance
        """
        import redis.asyncio as redis
        from azu.shared.config import get_config

        config = get_config()

        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=False
        )

        try:
            key = f"ledger:balance:{self.address}"
            data = await r.hgetall(key)

            if not data:
                return {
                    "available": 0.0,
                    "locked": 0.0,
                    "total": 0.0
                }

            return {
                "available": float(data.get(b"available", 0)),
                "locked": float(data.get(b"locked", 0)),
                "total": float(data.get(b"total", 0))
            }
        finally:
            await r.aclose()


# Singleton instance
_wallet_manager: Optional[WorkerWalletManager] = None


def get_worker_wallet(
    private_key: Optional[str] = None,
    provider_type: str = "hyperliquid"
) -> WorkerWalletManager:
    """
    Get the worker wallet manager instance.

    Args:
        private_key: Optional private key override
        provider_type: Payment provider type

    Returns:
        WorkerWalletManager instance
    """
    global _wallet_manager

    if _wallet_manager is None:
        _wallet_manager = WorkerWalletManager(
            private_key=private_key,
            provider_type=provider_type
        )

    return _wallet_manager
