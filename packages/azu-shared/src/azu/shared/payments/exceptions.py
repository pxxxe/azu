"""
Payment Provider Exceptions

Custom exceptions for the payment layer to provide clear error handling.
"""


class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass


class DepositError(PaymentError):
    """Exception raised when a deposit verification fails."""
    pass


class PayoutError(PaymentError):
    """Exception raised when a payout fails."""
    pass


class InsufficientBalanceError(PaymentError):
    """Exception raised when there's insufficient balance for an operation."""
    pass


class InvalidAddressError(PaymentError):
    """Exception raised when an address is invalid."""
    pass


class TransactionPendingError(PaymentError):
    """Exception raised when a transaction is still pending."""
    pass


class TransactionFailedError(PaymentError):
    """Exception raised when a transaction fails."""
    pass


class ProviderNotFoundError(PaymentError):
    """Exception raised when a payment provider is not found."""
    pass


class ProviderConfigurationError(PaymentError):
    """Exception raised when provider configuration is invalid."""
    pass


class InvalidTransactionError(PaymentError):
    """Exception raised when a transaction is invalid."""
    pass
