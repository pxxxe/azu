import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import json
from shared.config import settings
from shared.solana_lib import verify_deposit
from shared.economics import calculate_job_cost

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

class DepositReq(BaseModel):
    tx_sig: str
    user_pubkey: str

class JobReq(BaseModel):
    user_pubkey: str
    model_id: str
    prompt: str
    est_tokens: int = 100 # Default estimation

@app.post("/deposit")
async def deposit(req: DepositReq):
    # Idempotency
    if await r.get(f"tx:{req.tx_sig}"):
        return {"status": "processed"}

    amount = await verify_deposit(req.tx_sig)
    if amount == 0:
        raise HTTPException(400, "Invalid Transaction")

    # Credit Balance
    new_bal = await r.incrby(f"balance:{req.user_pubkey}", amount)
    await r.setex(f"tx:{req.tx_sig}", 86400, "1")

    return {"status": "success", "new_balance": new_bal}

@app.post("/submit")
async def submit(req: JobReq):
    # 1. Calc Cost (Assuming 80 layers for now)
    cost = calculate_job_cost(80, len(req.prompt.split()), req.est_tokens)

    # 2. Check Balance
    balance = await r.get(f"balance:{req.user_pubkey}")
    if not balance or int(balance) < cost:
        raise HTTPException(402, f"Insufficient funds. Cost: {cost}, Bal: {balance or 0}")

    # 3. Deduct
    await r.decrby(f"balance:{req.user_pubkey}", cost)

    # 4. Enqueue
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model": req.model_id,
        "input": req.prompt,
        "tokens": req.est_tokens,
        "owner": req.user_pubkey,
        "cost": cost
    }

    # Push to Redis Queue
    await r.rpush("job_queue", json.dumps(job))

    return {"job_id": job_id, "status": "queued", "cost": cost}

@app.get("/results/{job_id}")
async def get_result(job_id: str):
    """Poll for job result"""
    result = await r.get(f"result:{job_id}")

    if not result:
        # Check if job still in queue
        return {"status": "processing", "job_id": job_id}

    return json.loads(result)

# ALSO UPDATE handle_result in scheduler/main.py:
async def handle_result(self, wid, result_data):
    self.workers[wid]['status'] = "IDLE"

    # Store result in Redis
    job_id = result_data['job_id']
    await r.setex(
        f"result:{job_id}",
        3600,  # 1 hour TTL
        json.dumps(result_data)
    )
