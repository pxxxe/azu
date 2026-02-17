"""
Payment Provider Abstract Base Class

This module defines the interface that all payment providers must implement.
The abstraction allows the system to support multiple blockchain networks
(Hyperliquid, Solana, etc.) through a unified API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class PaymentProviderType(Enum):
    """Enum for supported payment providers."""
    HYPERLIQUID = "hyperliquid"
    SOLANA = "solana"


@dataclass
class DepositInfo:
    """Information about a verified deposit."""
    tx_hash: str
    sender_address: str
    amount: float
    confirmations: int
    timestamp: int


@dataclass
class PayoutInfo:
    """Information about a completed payout."""
    tx_hash: str
    recipient_address: str
    amount: float
    status: str


@dataclass
class BalanceInfo:
    """Balance information for an address."""
    available: float
    locked: float
    total: float


class PaymentProvider(ABC):
    """
    Abstract base class for payment providers.

    All payment implementations must inherit from this class and implement
    the required methods. This ensures a consistent interface across
    different blockchain networks.
    """

    @property
    @abstractmethod
    def provider_type(self) -> PaymentProviderType:
        """Returns the type of payment provider."""
        pass

    @property
    @abstractmethod
    def chain_symbol(self) -> str:
        """Returns the native token symbol (e.g., 'HYPE', 'SOL')."""
        pass

    @abstractmethod
    def get_deposit_address(self) -> str:
        """
        Returns the system wallet address where users should deposit funds.

        Returns:
            str: The deposit address string
        """
        pass

    @abstractmethod
    async def verify_deposit(
        self,
        tx_hash: str,
        expected_sender: Optional[str] = None,
        expected_amount: Optional[float] = None
    ) -> Tuple[bool, Optional[DepositInfo]]:
        """
        Verifies a deposit transaction on-chain.

        Args:
            tx_hash: The transaction hash to verify
            expected_sender: Optional expected sender address
            expected_amount: Optional expected amount

        Returns:
            Tuple of (is_valid, deposit_info)
            - is_valid: True if deposit is valid and confirmed
            - deposit_info: Deposit details if valid, None otherwise
        """
        pass

    @abstractmethod
    async def get_balance(self, address: str) -> BalanceInfo:
        """
        Gets the balance for an address.

        Args:
            address: The wallet address to check

        Returns:
            BalanceInfo: Contains available, locked, and total balance
        """
        pass

    @abstractmethod
    async def payout(
        self,
        recipient_address: str,
        amount: float,
        memo: Optional[str] = None
    ) -> PayoutInfo:
        """
        Executes a payout to a recipient.

        Args:
            recipient_address: The address to send funds to
            amount: The amount to send
            memo: Optional memo/note for the transaction

        Returns:
            PayoutInfo: Contains transaction hash and status
        """
        pass

    @abstractmethod
    async def get_transaction_status(self, tx_hash: str) -> Tuple[str, int]:
        """
        Gets the status of a transaction.

        Args:
            tx_hash: The transaction hash to check

        Returns:
            Tuple of (status, confirmations)
            - status: 'pending', 'confirmed', or 'failed'
            - confirmations: Number of block confirmations
        """
        pass

    @abstractmethod
    def is_address_valid(self, address: str) -> bool:
        """
        Validates an address format for this provider.

        Args:
            address: The address to validate

        Returns:
            True if address format is valid
        """
        pass

    def format_amount(self, amount: float) -> str:
        """
        Formats amount for display with appropriate decimals.

        Args:
            amount: The amount to format

        Returns:
            Formatted string with token symbol
        """
        return f"{amount:.8f} {self.chain_symbol}"

    def parse_amount(self, amount_str: str) -> float:
        """
        Parses amount string to float.

        Args:
            amount_str: The string to parse

        Returns:
            Float value of amount
        """
        # Remove any non-numeric characters except decimal point
        return float(amount_str.replace(self.chain_symbol, '').strip())
