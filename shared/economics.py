# PRICING CONSTANTS
LAMPORT_PER_SOL = 1_000_000_000
MIN_PAYOUT_THRESHOLD = 0.1 * LAMPORT_PER_SOL  # Worker gets paid after 0.1 SOL work

# Cost Calculation
# A "Token-Layer" is one token passing through one layer.
# Price: 2 Lamports per token-layer.
# Example: 70B Model (80 layers) * 100 Tokens * 2 = 16,000 Lamports ($0.002)
PRICE_PER_TOKEN_LAYER_LAMPORT = 2

def calculate_job_cost(num_layers: int, input_tokens: int, output_tokens: int) -> int:
    total_tokens = input_tokens + output_tokens
    return num_layers * total_tokens * PRICE_PER_TOKEN_LAYER_LAMPORT

def calculate_worker_share(job_cost: int) -> int:
    # Worker gets 80%, Platform keeps 20%
    return int(job_cost * 0.80)
