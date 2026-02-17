"""
Economics Module

Handles pricing calculations and revenue splits for the azu network.
Defines how costs are calculated and how payments are distributed
between workers and the platform.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Revenue split constants
WORKER_SHARE = 0.80  # 80% goes to workers
PLATFORM_SHARE = 0.20  # 20% goes to platform

# Token-layer pricing (per 1M token-layers)
# Default: 2 lamports per token-layer
# This translates to:
# - 70B model (80 layers), 100 tokens = 8,000 token-layers = ~$0.001
LAMports_PER_TOKEN_LAYER = 2


@dataclass
class CostBreakdown:
    """Detailed breakdown of job costs."""
    total_cost: float
    worker_payment: float
    platform_fee: float
    token_layers: int
    estimated_tokens: int
    layer_count: int


@dataclass
class WorkerPayment:
    """Payment information for a single worker."""
    worker_address: str
    amount: float
    job_id: str
    layer_count: int


def calculate_token_layers(
    num_layers: int,
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1
) -> int:
    """
    Calculate the total token-layers for a job.

    Token-layers = tokens * layers
    For MoE models, expert layers are counted separately.

    Args:
        num_layers: Number of layers in the model
        num_tokens: Number of tokens in the prompt/response
        is_moe: Whether this is a Mixture of Experts model
        num_experts: Number of experts per MoE layer

    Returns:
        Total token-layers
    """
    if is_moe:
        # For MoE: count router + each expert as separate
        # Router layer + expert layers
        effective_layers = num_layers * num_experts
    else:
        effective_layers = num_layers

    return effective_layers * num_tokens


def calculate_cost_lamports(
    num_layers: int,
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1,
    lamports_per_token_layer: int = LAMports_PER_TOKEN_LAYER
) -> int:
    """
    Calculate job cost in lamports.

    Args:
        num_layers: Number of layers in the model
        num_tokens: Number of tokens
        is_moe: Whether this is a MoE model
        num_experts: Number of experts per MoE layer
        lamports_per_token_layer: Price per token-layer

    Returns:
        Cost in lamports
    """
    token_layers = calculate_token_layers(
        num_layers=num_layers,
        num_tokens=num_tokens,
        is_moe=is_moe,
        num_experts=num_experts
    )
    return token_layers * lamports_per_token_layer


def calculate_cost(
    num_layers: int,
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1,
    lamports_per_token_layer: int = LAMports_PER_TOKEN_LAYER
) -> Tuple[float, float]:
    """
    Calculate job cost split between worker and platform.

    Args:
        num_layers: Number of layers in the model
        num_tokens: Number of tokens
        is_moe: Whether this is a MoE model
        num_experts: Number of experts per MoE layer
        lamports_per_token_layer: Price per token-layer

    Returns:
        Tuple of (total_cost, worker_payment)
        - total_cost: Total cost in HYPE/SOL units
        - worker_payment: Worker's 80% share
    """
    lamports = calculate_cost_lamports(
        num_layers=num_layers,
        num_tokens=num_tokens,
        is_moe=is_moe,
        num_experts=num_experts,
        lamports_per_token_layer=lamports_per_token_layer
    )

    # Convert lamports to native units
    # Note: HYPE has 18 decimals, SOL has 9 decimals
    # We'll normalize to token units (assumes token has 18 decimals like HYPE)
    total_cost = lamports / 1e18
    worker_payment = total_cost * WORKER_SHARE

    return total_cost, worker_payment


def calculate_cost_breakdown(
    num_layers: int,
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1
) -> CostBreakdown:
    """
    Calculate detailed cost breakdown for a job.

    Args:
        num_layers: Number of layers in the model
        num_tokens: Number of tokens
        is_moe: Whether this is a MoE model
        num_experts: Number of experts per MoE layer

    Returns:
        CostBreakdown with all cost details
    """
    lamports_per_layer = int(os.environ.get(
        "LAMPORTS_PER_TOKEN_LAYER",
        str(LAMports_PER_TOKEN_LAYER)
    ))

    total_cost, worker_payment = calculate_cost(
        num_layers=num_layers,
        num_tokens=num_tokens,
        is_moe=is_moe,
        num_experts=num_experts,
        lamports_per_token_layer=lamports_per_layer
    )

    token_layers = calculate_token_layers(
        num_layers=num_layers,
        num_tokens=num_tokens,
        is_moe=is_moe,
        num_experts=num_experts
    )

    return CostBreakdown(
        total_cost=total_cost,
        worker_payment=worker_payment,
        platform_fee=total_cost * PLATFORM_SHARE,
        token_layers=token_layers,
        estimated_tokens=num_tokens,
        layer_count=num_layers
    )


def calculate_worker_payments(
    worker_layers: Dict[str, List[int]],
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1
) -> List[WorkerPayment]:
    """
    Calculate payments for each worker based on their assigned layers.

    Args:
        worker_layers: Dict mapping worker_address to list of layer indices
        num_tokens: Number of tokens processed
        is_moe: Whether this is a MoE model
        num_experts: Number of experts per MoE layer

    Returns:
        List of WorkerPayment for each worker
    """
    # Get per-layer cost
    lamports_per_layer = int(os.environ.get(
        "LAMPORTS_PER_TOKEN_LAYER",
        str(LAMports_PER_TOKEN_LAYER)
    ))

    # Calculate cost per layer (per token)
    cost_per_layer = lamports_per_layer / 1e18

    payments = []

    for worker_address, layer_indices in worker_layers.items():
        if not layer_indices:
            continue

        num_layers = len(layer_indices)

        if is_moe:
            # For MoE, count each expert as a separate layer
            effective_layers = num_layers * num_experts
        else:
            effective_layers = num_layers

        # Calculate worker payment
        token_layers = effective_layers * num_tokens
        total_cost = (token_layers * cost_per_layer) / 1e18
        worker_payment = total_cost * WORKER_SHARE

        payments.append(WorkerPayment(
            worker_address=worker_address,
            amount=worker_payment,
            job_id="",  # Will be filled by caller
            layer_count=num_layers
        ))

    return payments


def get_price_per_token() -> float:
    """
    Get the estimated price per token in USD (approximate).

    Returns:
        Price per token in USD
    """
    # This is a placeholder - in production, would fetch live prices
    # Assuming ~$0.001 per 1K token-layers for 80-layer model
    # = $0.001 / 80,000 = $0.0000000125 per token
    return 0.0000000125


def estimate_usd_cost(
    num_layers: int,
    num_tokens: int,
    is_moe: bool = False,
    num_experts: int = 1
) -> float:
    """
    Estimate cost in USD.

    Args:
        num_layers: Number of layers
        num_tokens: Number of tokens
        is_moe: Whether this is a MoE model
        num_experts: Number of experts per MoE layer

    Returns:
        Estimated cost in USD
    """
    total_cost, _ = calculate_cost(
        num_layers=num_layers,
        num_tokens=num_tokens,
        is_moe=is_moe,
        num_experts=num_experts
    )

    # Convert to approximate USD (placeholder)
    return total_cost * get_price_per_token() * 1e18  # Simplified


# Revenue tracking
@dataclass
class RevenueStats:
    """Statistics about revenue and payments."""
    total_revenue: float
    total_paid_to_workers: float
    total_platform_fees: float
    jobs_completed: int


def calculate_revenue_stats(
    total_revenue: float,
    jobs_completed: int
) -> RevenueStats:
    """
    Calculate revenue statistics.

    Args:
        total_revenue: Total revenue collected
        jobs_completed: Number of jobs completed

    Returns:
        RevenueStats with breakdown
    """
    return RevenueStats(
        total_revenue=total_revenue,
        total_paid_to_workers=total_revenue * WORKER_SHARE,
        total_platform_fees=total_revenue * PLATFORM_SHARE,
        jobs_completed=jobs_completed
    )
