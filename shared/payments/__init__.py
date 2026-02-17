"""
Payment Provider Module

A pluggable payment layer for azu that supports multiple blockchain networks.
Provides a unified interface for deposits, balances, and payouts.

Usage:
    from shared.payments import get_payment_provider, PaymentProviderType

    # Get the configured provider
    provider = get_payment_provider()

    # Get deposit address
    deposit_address = provider.get_deposit_address()

    # Verify a deposit
    is_valid, deposit_info = await provider.verify_deposit(tx_hash, amount)

    # Get balance
    balance = await provider.get_balance(address)

    # Execute a payout
    payout_info = await provider.payout(recipient, amount)
"""

from .base import (
    BalanceInfo,
    DepositInfo,
    PaymentProvider,
    PaymentProviderType,
    PayoutInfo,
)
from .exceptions import (
    DepositError,
    InsufficientBalanceError,
    InvalidAddressError,
    InvalidTransactionError,
    PaymentError,
    ProviderConfigurationError,
    ProviderNotFoundError,
    PayoutError,
    TransactionFailedError,
    TransactionPendingError,
)
from .factory import (
    get_payment_provider,
    get_provider,
    get_provider_type,
    reset_provider,
    set_provider,
    validate_provider_config,
)
from .hyperliquid import HyperliquidProvider, HyperliquidTestnetProvider
from .solana import SolanaProvider

__all__ = [
    # Base classes
    "PaymentProvider",
    "PaymentProviderType",
    "BalanceInfo",
    "DepositInfo",
    "PayoutInfo",
    # Exceptions
    "PaymentError",
    "DepositError",
    "PayoutError",
    "InsufficientBalanceError",
    "InvalidAddressError",
    "TransactionPendingError",
    "TransactionFailedError",
    "ProviderNotFoundError",
    "ProviderConfigurationError",
    "InvalidTransactionError",
    # Factory functions
    "get_payment_provider",
    "get_provider",
    "get_provider_type",
    "validate_provider_config",
    "set_provider",
    "reset_provider",
    # Implementations
    "HyperliquidProvider",
    "HyperliquidTestnetProvider",
    "SolanaProvider",
]

__version__ = "1.0.0"
