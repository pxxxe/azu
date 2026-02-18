"""
Internal Ledger Module

Manages internal balances for users and workers without requiring
on-chain transactions for every operation. This is a gas-optimized
approach where small payments are accumulated internally and
settled on-chain periodically.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import redis.asyncio as redis


class TransactionType(Enum):
    """Types of ledger transactions."""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    JOB_PAYMENT = "job_payment"
    WORKER_CREDIT = "worker_credit"
    PLATFORM_FEE = "platform_fee"
    PAYOUT = "payout"


class TransactionStatus(Enum):
    """Status of a transaction."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LedgerTransaction:
    """A single ledger transaction."""
    id: str
    address: str
    transaction_type: TransactionType
    amount: float
    status: TransactionStatus
    job_id: Optional[str] = None
    tx_hash: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict = field(default_factory=dict)


@dataclass
class AccountBalance:
    """Balance information for an account."""
    address: str
    available: float
    locked: float
    total: float
    last_updated: int


class InternalLedger:
    """
    Internal ledger for tracking balances.

    This provides an in-memory/Redis-backed ledger for tracking
    user deposits and worker credits without requiring on-chain
    transactions for every operation.
    """

    def __init__(self, redis_client: redis.Redis, prefix: str = "ledger"):
        """
        Initialize the internal ledger.

        Args:
            redis_client: Redis client instance
            prefix: Key prefix for Redis storage
        """
        self.redis = redis_client
        self.prefix = prefix

    def _balance_key(self, address: str) -> str:
        """Get Redis key for account balance."""
        return f"{self.prefix}:balance:{address}"

    def _tx_key(self, tx_id: str) -> str:
        """Get Redis key for transaction."""
        return f"{self.prefix}:tx:{tx_id}"

    def _tx_list_key(self, address: str) -> str:
        """Get Redis key for address transaction list."""
        return f"{self.prefix}:txs:{address}"

    async def get_balance(self, address: str) -> AccountBalance:
        """
        Get the balance for an address.

        Args:
            address: The account address

        Returns:
            AccountBalance with available, locked, and total
        """
        key = self._balance_key(address)
        data = await self.redis.hgetall(key)

        if not data:
            return AccountBalance(
                address=address,
                available=0.0,
                locked=0.0,
                total=0.0,
                last_updated=int(time.time())
            )

        return AccountBalance(
            address=address,
            available=float(data.get("available", 0)),
            locked=float(data.get("locked", 0)),
            total=float(data.get("total", 0)),
            last_updated=int(data.get("last_updated", 0))
        )

    async def credit(
        self,
        address: str,
        amount: float,
        transaction_type: TransactionType,
        job_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> LedgerTransaction:
        """
        Credit an account (add funds).

        Args:
            address: The account to credit
            amount: Amount to add
            transaction_type: Type of transaction
            job_id: Optional job ID
            metadata: Additional metadata

        Returns:
            The created transaction
        """
        tx = LedgerTransaction(
            id=f"tx_{int(time.time() * 1000000)}",
            address=address,
            transaction_type=transaction_type,
            amount=amount,
            status=TransactionStatus.COMPLETED,
            job_id=job_id,
            metadata=metadata or {}
        )

        # Update balance atomically
        key = self._balance_key(address)
        await self.redis.hincrbyfloat(key, "available", amount)
        await self.redis.hincrbyfloat(key, "total", amount)
        await self.redis.hset(key, "last_updated", str(int(time.time())))

        # Store transaction
        await self._store_transaction(tx)

        return tx

    async def debit(
        self,
        address: str,
        amount: float,
        transaction_type: TransactionType,
        job_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> LedgerTransaction:
        """
        Debit an account (subtract funds).

        Args:
            address: The account to debit
            amount: Amount to subtract
            transaction_type: Type of transaction
            job_id: Optional job ID
            metadata: Additional metadata

        Returns:
            The created transaction

        Raises:
            ValueError: If insufficient balance
        """
        # Check balance first
        balance = await self.get_balance(address)
        if balance.available < amount:
            raise ValueError(
                f"Insufficient balance: {balance.available} < {amount}"
            )

        tx = LedgerTransaction(
            id=f"tx_{int(time.time() * 1000000)}",
            address=address,
            transaction_type=transaction_type,
            amount=-amount,
            status=TransactionStatus.COMPLETED,
            job_id=job_id,
            metadata=metadata or {}
        )

        # Update balance atomically
        key = self._balance_key(address)
        await self.redis.hincrbyfloat(key, "available", -amount)
        await self.redis.hincrbyfloat(key, "total", -amount)
        await self.redis.hset(key, "last_updated", str(int(time.time())))

        # Store transaction
        await self._store_transaction(tx)

        return tx

    async def lock_funds(
        self,
        address: str,
        amount: float,
        job_id: str
    ) -> bool:
        """
        Lock funds for a job (move from available to locked).

        Args:
            address: The account
            amount: Amount to lock
            job_id: The job ID

        Returns:
            True if successful

        Raises:
            ValueError: If insufficient available balance
        """
        balance = await self.get_balance(address)
        if balance.available < amount:
            raise ValueError(
                f"Insufficient available balance: {balance.available} < {amount}"
            )

        key = self._balance_key(address)
        pipe = self.redis.pipeline()
        pipe.hincrbyfloat(key, "available", -amount)
        pipe.hincrbyfloat(key, "locked", amount)
        pipe.hset(key, "last_updated", str(int(time.time())))
        await pipe.execute()

        # Record lock transaction
        tx = LedgerTransaction(
            id=f"tx_{int(time.time() * 1000000)}",
            address=address,
            transaction_type=TransactionType.JOB_PAYMENT,
            amount=-amount,
            status=TransactionStatus.COMPLETED,
            job_id=job_id,
            metadata={"action": "lock"}
        )
        await self._store_transaction(tx)

        return True

    async def unlock_funds(
        self,
        address: str,
        amount: float,
        job_id: str
    ) -> bool:
        """
        Unlock funds (move from locked back to available).

        Args:
            address: The account
            amount: Amount to unlock
            job_id: The job ID

        Returns:
            True if successful
        """
        balance = await self.get_balance(address)
        if balance.locked < amount:
            amount = balance.locked  # Unlock what's available

        if amount <= 0:
            return True

        key = self._balance_key(address)
        pipe = self.redis.pipeline()
        pipe.hincrbyfloat(key, "available", amount)
        pipe.hincrbyfloat(key, "locked", -amount)
        pipe.hset(key, "last_updated", str(int(time.time())))
        await pipe.execute()

        # Record unlock transaction
        tx = LedgerTransaction(
            id=f"tx_{int(time.time() * 1000000)}",
            address=address,
            transaction_type=TransactionType.JOB_PAYMENT,
            amount=amount,
            status=TransactionStatus.COMPLETED,
            job_id=job_id,
            metadata={"action": "unlock"}
        )
        await self._store_transaction(tx)

        return True

    async def settle_job_payment(
        self,
        user_address: str,
        worker_addresses: List[Tuple[str, float]],  # (address, amount)
        job_id: str,
        platform_fee: float
    ) -> Tuple[List[LedgerTransaction], List[LedgerTransaction]]:
        """
        Settle a completed job payment.

        Args:
            user_address: The user's address
            worker_addresses: List of (worker_address, amount) tuples
            job_id: The job ID
            platform_fee: The platform fee amount

        Returns:
            Tuple of (worker_transactions, platform_transactions)
        """
        worker_txs = []
        platform_txs = []

        # Locked amount should already be sufficient
        # First, reduce locked funds from user
        user_key = self._balance_key(user_address)
        total_cost = platform_fee + sum(amount for _, amount in worker_addresses)

        # Deduct from locked balance
        await self.redis.hincrbyfloat(user_key, "locked", -total_cost)
        await self.redis.hset(user_key, "last_updated", str(int(time.time())))

        # Credit workers
        for worker_addr, amount in worker_addresses:
            tx = await self.credit(
                worker_addr,
                amount,
                TransactionType.WORKER_CREDIT,
                job_id=job_id,
                metadata={"action": "job_settlement"}
            )
            worker_txs.append(tx)

        # Credit platform
        if platform_fee > 0:
            # Credit platform fee to a designated address or keep as revenue
            platform_address = "platform"  # Internal platform account
            tx = await self.credit(
                platform_address,
                platform_fee,
                TransactionType.PLATFORM_FEE,
                job_id=job_id,
                metadata={"action": "job_settlement"}
            )
            platform_txs.append(tx)

        return worker_txs, platform_txs

    async def _store_transaction(self, tx: LedgerTransaction) -> None:
        """Store a transaction in Redis."""
        # Store transaction data
        tx_key = self._tx_key(tx.id)
        tx_data = {
            "id": tx.id,
            "address": tx.address,
            "type": tx.transaction_type.value,
            "amount": str(tx.amount),
            "status": tx.status.value,
            "job_id": tx.job_id or "",
            "tx_hash": tx.tx_hash or "",
            "timestamp": str(tx.timestamp),
            "metadata": json.dumps(tx.metadata)
        }
        await self.redis.hset(tx_key, mapping={
            k: v for k, v in tx_data.items()
        })

        # Add to address transaction list
        tx_list_key = self._tx_list_key(tx.address)
        await self.redis.zadd(tx_list_key, {tx.id: tx.timestamp})

    async def get_transaction_history(
        self,
        address: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[LedgerTransaction]:
        """
        Get transaction history for an address.

        Args:
            address: The account address
            limit: Maximum number of transactions
            offset: Offset for pagination

        Returns:
            List of transactions
        """
        tx_list_key = self._tx_list_key(address)
        tx_ids = await self.redis.zrevrange(
            tx_list_key,
            offset,
            offset + limit - 1
        )

        transactions = []
        for tx_id in tx_ids:
            tx_key = self._tx_key(tx_id.decode() if isinstance(tx_id, bytes) else tx_id)
            data = await self.redis.hgetall(tx_key)

            if data:
                transactions.append(LedgerTransaction(
                    id=data[b"id"].decode(),
                    address=data[b"address"].decode(),
                    transaction_type=TransactionType(data[b"type"].decode()),
                    amount=float(data[b"amount"].decode()),
                    status=TransactionStatus(data[b"status"].decode()),
                    job_id=data[b"job_id"].decode() or None,
                    tx_hash=data[b"tx_hash"].decode() or None,
                    timestamp=int(data[b"timestamp"].decode()),
                    metadata=json.loads(data[b"metadata"].decode())
                ))

        return transactions


# Ledger instance management
_ledger_instance: Optional[InternalLedger] = None


async def get_ledger(redis_client: Optional[redis.Redis] = None) -> InternalLedger:
    """
    Get the internal ledger instance.

    Args:
        redis_client: Optional Redis client (creates new if not provided)

    Returns:
        InternalLedger instance
    """
    global _ledger_instance

    if _ledger_instance is None:
        if redis_client is None:
            # Create new Redis connection
            from shared.config import get_config
            config = get_config()
            redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
                decode_responses=False
            )
        _ledger_instance = InternalLedger(redis_client)

    return _ledger_instance
