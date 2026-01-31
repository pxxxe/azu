import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import traceback
import asyncio
from shared.config import settings
from .layer_storage import LayerStore

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
store = LayerStore()

app.mount("/layers", StaticFiles(directory="/data/layers"), name="layers")

class ShardRequest(BaseModel):
    model_id: str

async def background_shard_task(model_id: str, hf_token: str):
    try:
        print(f"🔄 BACKGROUND: Starting shard for {model_id}")
        await r.set(f"shard_status:{model_id}", "processing")
        loop = asyncio.get_event_loop()
        num_layers = await loop.run_in_executor(None, store.shard_model, model_id, hf_token)
        await r.set(f"shard_status:{model_id}", "ready")
        print(f"✅ BACKGROUND: Finished sharding {model_id} ({num_layers} layers)")
    except Exception as e:
        err_msg = str(e)
        print(f"❌ BACKGROUND: Failed {model_id}: {err_msg}")
        traceback.print_exc()
        await r.set(f"shard_status:{model_id}", f"failed: {err_msg}")

@app.post("/models/shard")
async def shard_model(req: ShardRequest, background_tasks: BackgroundTasks):
    if store.has_model(req.model_id):
        return {"status": "ready", "message": "Model already exists"}
    current_status = await r.get(f"shard_status:{req.model_id}")
    if current_status == "processing":
        return {"status": "processing", "message": "Already processing"}
    hf_token = settings.HF_TOKEN
    if not hf_token:
        raise HTTPException(500, "HF_TOKEN not set on Registry")
    background_tasks.add_task(background_shard_task, req.model_id, hf_token)
    return {"status": "started", "message": "Background task started"}

@app.get("/models/status")
async def get_shard_status(model_id: str):
    if store.has_model(model_id): return {"status": "ready"}
    status = await r.get(f"shard_status:{model_id}")
    if not status: return {"status": "idle"}
    if status.startswith("failed"): return {"status": "failed", "error": status.split(": ", 1)[1]}
    return {"status": status}

@app.get("/models/info")
async def get_model_info(model_id: str):
    sanitized = model_id.replace("/", "_")
    path = f"/data/layers/{sanitized}/structure.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(404, "Model not sharded or found")

@app.post("/workers/register")
async def register_worker(data: dict):
    await r.setex(f"worker_meta:{data['worker_id']}", 300, json.dumps(data))
    return {"status": "ok"}

@app.post("/workers/query")
async def query_workers(data: dict):
    keys = await r.keys("worker_meta:*")
    workers = []
    for k in keys:
        w_data = await r.get(k)
        if w_data: workers.append(json.loads(w_data))
    return {"workers": workers}
