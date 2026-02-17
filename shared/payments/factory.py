"""
Payment Provider Factory

Factory module for creating payment provider instances based on configuration.
Supports both Hyperliquid and Solana providers with easy swapping.
"""

import os
from typing import Optional

from .base import PaymentProvider, PaymentProviderType
from .exceptions import ProviderConfigurationError, ProviderNotFoundError
from .hyperliquid import HyperliquidProvider
from .solana import SolanaProvider


# Default configuration values
DEFAULT_CONFIRMATIONS = 1
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


def get_payment_provider(
    provider_type: Optional[str] = None,
    **config
) -> PaymentProvider:
    """
    Factory function to create a payment provider instance.

    Args:
        provider_type: Type of provider ('hyperliquid' or 'solana')
                      If None, reads from PAYMENT_PROVIDER env var
        **config: Additional configuration overrides

    Returns:
        PaymentProvider: An instance of the requested payment provider

    Raises:
        ProviderNotFoundError: If provider type is not supported
        ProviderConfigurationError: If required configuration is missing
    """
    # Determine provider type
    if provider_type is None:
        provider_type = os.environ.get("PAYMENT_PROVIDER", "hyperliquid").lower()

    # Get common configuration
    confirmations = config.get(
        "confirmations_required",
        int(os.environ.get("PAYMENT_CONFIRMATIONS", DEFAULT_CONFIRMATIONS))
    )
    max_retries = config.get(
        "max_retries",
        int(os.environ.get("PAYMENT_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    )
    retry_delay = config.get(
        "retry_delay",
        float(os.environ.get("PAYMENT_RETRY_DELAY", DEFAULT_RETRY_DELAY))
    )

    if provider_type == "hyperliquid":
        return _create_hyperliquid_provider(
            confirmations=confirmations,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **config
        )
    elif provider_type == "solana":
        return _create_solana_provider(
            confirmations=confirmations,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **config
        )
    else:
        raise ProviderNotFoundError(
            f"Unknown payment provider: {provider_type}. "
            f"Supported providers: hyperliquid, solana"
        )


def _create_hyperliquid_provider(
    confirmations: int,
    max_retries: int,
    retry_delay: float,
    **config
) -> HyperliquidProvider:
    """
    Create a Hyperliquid provider with configuration from env vars.

    Args:
        confirmations: Required confirmations for deposits
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        **config: Additional overrides

    Returns:
        HyperliquidProvider instance

    Raises:
        ProviderConfigurationError: If required config is missing
    """
    # Get configuration from environment or overrides
    rpc_url = config.get(
        "rpc_url",
        os.environ.get("HYPERLIQUID_RPC_URL", "https://api.hyperliquid.xyz")
    )
    address = config.get(
        "address",
        os.environ.get("HYPERLIQUID_ADDRESS")
    )
    private_key = config.get(
        "private_key",
        os.environ.get("SCHEDULER_PRIVATE_KEY")
    )

    # Validate required configuration
    if not address:
        raise ProviderConfigurationError(
            "HYPERLIQUID_ADDRESS is required for Hyperliquid provider"
        )
    if not private_key:
        raise ProviderConfigurationError(
            "SCHEDULER_PRIVATE_KEY is required for Hyperliquid provider"
        )

    return HyperliquidProvider(
        rpc_url=rpc_url,
        address=address,
        private_key=private_key,
        confirmations_required=confirmations,
        max_retries=max_retries,
        retry_delay=retry_delay
    )


def _create_solana_provider(
    confirmations: int,
    max_retries: int,
    retry_delay: float,
    **config
) -> SolanaProvider:
    """
    Create a Solana provider with configuration from env vars.

    Args:
        confirmations: Required confirmations for deposits
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        **config: Additional overrides

    Returns:
        SolanaProvider instance

    Raises:
        ProviderConfigurationError: If required config is missing
    """
    # Get configuration from environment or overrides
    rpc_url = config.get(
        "rpc_url",
        os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    )
    address = config.get(
        "address",
        os.environ.get("SOLANA_ADDRESS")
    )
    private_key = config.get(
        "private_key",
        os.environ.get("SOLANA_PRIVATE_KEY")
    )

    # Validate required configuration
    if not address:
        raise ProviderConfigurationError(
            "SOLANA_ADDRESS is required for Solana provider"
        )
    if not private_key:
        raise ProviderConfigurationError(
            "SOLANA_PRIVATE_KEY is required for Solana provider"
        )

    return SolanaProvider(
        rpc_url=rpc_url,
        address=address,
        private_key=private_key,
        confirmations_required=confirmations,
        max_retries=max_retries,
        retry_delay=retry_delay
    )


def get_provider_type() -> PaymentProviderType:
    """
    Get the current payment provider type from configuration.

    Returns:
        PaymentProviderType enum value
    """
    provider = os.environ.get("PAYMENT_PROVIDER", "hyperliquid").lower()

    if provider == "hyperliquid":
        return PaymentProviderType.HYPERLIQUID
    elif provider == "solana":
        return PaymentProviderType.SOLANA
    else:
        return PaymentProviderType.HYPERLIQUID  # Default to hyperliquid


def validate_provider_config(provider_type: Optional[str] = None) -> bool:
    """
    Validate that the provider is properly configured.

    Args:
        provider_type: Type to validate, or current provider if None

    Returns:
        True if configuration is valid

    Raises:
        ProviderConfigurationError: If configuration is invalid
    """
    try:
        get_payment_provider(provider_type)
        return True
    except (ProviderNotFoundError, ProviderConfigurationError) as e:
        raise ProviderConfigurationError(f"Provider configuration invalid: {e}")


# Singleton instance for convenience
_provider_instance: Optional[PaymentProvider] = None


def get_provider() -> PaymentProvider:
    """
    Get a singleton payment provider instance.

    This function caches the provider instance to avoid recreating
    it for each request. Use this in production code.

    Returns:
        PaymentProvider: The configured payment provider instance
    """
    global _provider_instance

    if _provider_instance is None:
        _provider_instance = get_payment_provider()

    return _provider_instance


def set_provider(provider: PaymentProvider) -> None:
    """
    Set a custom provider instance (useful for testing).

    Args:
        provider: The provider instance to use
    """
    global _provider_instance
    _provider_instance = provider


def reset_provider() -> None:
    """Reset the cached provider instance."""
    global _provider_instance
    _provider_instance = None
