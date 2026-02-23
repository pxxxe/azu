"""
API Main Entry Point

FastAPI application for the azu decentralized inference network.
Provides endpoints for job submission and deposit handling.
"""

import os
import sys
import uuid
import json
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azu.shared.config import get_config
from azu.shared.payments import get_payment_provider
from azu.shared.ledger import get_ledger, TransactionType
from .openai_adapter import mount_openai_adapter
from .user_auth_router import user_auth_router

# ============================================================================
# Pydantic Models
# ============================================================================

class DepositReq(BaseModel):
    """Request for deposit verification."""
    tx_sig: str
    user_pubkey: str


class JobReq(BaseModel):
    """Request for job submission."""
    user_pubkey: str
    model_id: str
    prompt: str
    est_tokens: int = 100


# ============================================================================
# FastAPI App
# ============================================================================

config = get_config()
app = FastAPI(
    title="azu API",
    description="Decentralized inference network API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_redis():
    """Get Redis client."""
    return redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True
    )


# ============================================================================
# User-Facing Endpoints
# ============================================================================

@app.post("/deposit")
async def deposit(req: DepositReq):
    """
    Verify deposit and credit user's internal balance.

    This endpoint verifies a blockchain transaction and credits
    the user's internal ledger balance.
    """
    r = await get_redis()

    # Idempotency check
    if await r.get(f"tx:{req.tx_sig}"):
        return {"status": "processed"}

    # Verify deposit using payment provider
    try:
        provider = get_payment_provider()
        is_valid, deposit_info = await provider.verify_deposit(
            tx_hash=req.tx_sig,
            expected_sender=req.user_pubkey
        )

        if not is_valid or deposit_info is None:
            raise HTTPException(400, "Invalid Transaction or Amount 0")

        amount = deposit_info.amount

    except Exception as e:
        raise HTTPException(400, f"Deposit verification failed: {str(e)}")

    # Credit internal ledger
    ledger = await get_ledger(await get_redis())
    await ledger.credit(
        address=req.user_pubkey,
        amount=amount,
        transaction_type=TransactionType.DEPOSIT,
        metadata={"tx_hash": req.tx_sig}
    )

    await r.setex(f"tx:{req.tx_sig}", 86400, "1")

    # Get updated balance
    balance = await ledger.get_balance(req.user_pubkey)

    return {"status": "success", "new_balance": balance.available}


@app.post("/submit")
async def submit(req: JobReq):
    """
    Submit an inference job to the scheduler queue.

    The job will be processed by available workers and the result
    can be retrieved using the job_id.
    """
    r = await get_redis()

    # 1. Check Balance (Minimum Threshold)
    MIN_BALANCE = 0.001  # Token units
    ledger = await get_ledger(r)
    balance = await ledger.get_balance(req.user_pubkey)

    if balance.available < MIN_BALANCE:
        raise HTTPException(402, f"Insufficient funds. Balance: {balance.available or 0}")

    # 2. Estimate job cost and lock funds
    from azu.shared.economics import calculate_cost_breakdown

    # Get layer count from model_id or use default
    est_layers = 40  # Default for small models
    cost_breakdown = calculate_cost_breakdown(
        num_layers=est_layers,
        num_tokens=req.est_tokens,
        is_moe=False,
        num_experts=1
    )

    estimated_cost = cost_breakdown.total_cost

    # Lock funds for the job
    try:
        await ledger.lock_funds(
            address=req.user_pubkey,
            amount=estimated_cost,
            job_id=""  # Will be updated with actual job_id
        )
    except ValueError as e:
        raise HTTPException(402, f"Insufficient funds to lock: {str(e)}")

    # 3. Enqueue Job
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model_id": req.model_id,
        "input_prompt": req.prompt,
        "tokens": req.est_tokens,
        "owner": req.user_pubkey,
        "cost": estimated_cost
    }

    await r.rpush("job_queue", json.dumps(job))

    return {"job_id": job_id, "status": "queued", "estimated_cost": estimated_cost}


@app.get("/results/{job_id}")
async def get_result(job_id: str):
    """
    Get the result of a completed job.

    Returns the job result if completed, or status if still processing.
    """
    r = await get_redis()
    res = await r.get(f"result:{job_id}")
    if not res:
        return {"status": "processing", "job_id": job_id}

    return json.loads(res)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "azu API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# OpenAI Compatibility Layer
# ============================================================================

mount_openai_adapter(app)


# ============================================================================
# User Auth Management
# ============================================================================

app.include_router(user_auth_router)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True
    )
