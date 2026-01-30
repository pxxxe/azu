import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import json
from shared.config import settings
from shared.solana_lib import verify_deposit

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

class DepositReq(BaseModel):
    tx_sig: str
    user_pubkey: str

class JobReq(BaseModel):
    user_pubkey: str
    model_id: str
    prompt: str
    est_tokens: int = 100

@app.post("/deposit")
async def deposit(req: DepositReq):
    # Idempotency check
    if await r.get(f"tx:{req.tx_sig}"):
        return {"status": "processed"}

    amount = await verify_deposit(req.tx_sig)
    if amount == 0:
        raise HTTPException(400, "Invalid Transaction or Amount 0")

    new_bal = await r.incrby(f"balance:{req.user_pubkey}", amount)
    await r.setex(f"tx:{req.tx_sig}", 86400, "1")

    return {"status": "success", "new_balance": new_bal}

@app.post("/submit")
async def submit(req: JobReq):
    # 1. Check Balance (Minimum Threshold)
    # We don't deduct yet. Deduct post-run.
    MIN_BALANCE = 1000  # Lamports
    balance = await r.get(f"balance:{req.user_pubkey}")

    if not balance or int(balance) < MIN_BALANCE:
        raise HTTPException(402, f"Insufficient funds. Balance: {balance or 0}")

    # 2. Enqueue Job
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model": req.model_id,
        "input": req.prompt,
        "tokens": req.est_tokens,
        "owner": req.user_pubkey,
        "cost": 0 # Will be filled post-run
    }

    await r.rpush("job_queue", json.dumps(job))

    return {"job_id": job_id, "status": "queued"}

@app.get("/results/{job_id}")
async def get_result(job_id: str):
    res = await r.get(f"result:{job_id}")
    if not res:
        return {"status": "processing", "job_id": job_id}

    return json.loads(res)
