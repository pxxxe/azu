import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
from shared.config import settings
from .layer_storage import LayerStore

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
store = LayerStore()

# ---------------------------------------------------------
# 1. FILE SERVER (Crucial)
# This allows the Worker's LayerLoader to download files.
# e.g., GET http://registry:8002/layers/Qwen_.../layer_0.pt
# ---------------------------------------------------------
app.mount("/layers", StaticFiles(directory="/data/layers"), name="layers")

# ---------------------------------------------------------
# 2. SHARDING ENDPOINT
# The Orchestrator calls this to make the Registry cut up the model.
# ---------------------------------------------------------
class ShardRequest(BaseModel):
    model_id: str

@app.post("/models/shard")
async def shard_model(req: ShardRequest):
    hf_token = settings.HF_TOKEN
    if not hf_token:
        raise HTTPException(500, "HF_TOKEN not set on Registry")

    try:
        # Calls the CPU-safe sharding logic in layer_storage.py
        num = store.shard_model(req.model_id, hf_token)
        return {"status": "success", "num_layers": num}
    except Exception as e:
        print(f"Sharding Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Used by Scheduler to know how many layers a model has"""
    sanitized = model_id.replace("/", "_")
    # Check if we have the structure file on disk
    path = f"/data/layers/{sanitized}/structure.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(404, "Model not sharded or found")

# ---------------------------------------------------------
# 3. WORKER DISCOVERY
# ---------------------------------------------------------
@app.post("/workers/register")
async def register_worker(data: dict):
    # Store worker metadata in Redis for 5 minutes
    await r.setex(f"worker_meta:{data['worker_id']}", 300, json.dumps(data))
    return {"status": "ok"}

@app.post("/workers/query")
async def query_workers(data: dict):
    # Simple lookup for now
    keys = await r.keys("worker_meta:*")
    workers = []
    for k in keys:
        w_data = await r.get(k)
        if w_data:
            workers.append(json.loads(w_data))
    return {"workers": workers}
