import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import traceback
import asyncio
import sys
from shared.config import settings
from .layer_storage import LayerStore

# Setup logging flush
sys.stdout.reconfigure(line_buffering=True)

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

# Initialize Store safely with error printing
try:
    print("🚀 Initializing LayerStore...")
    store = LayerStore()
    os.makedirs("/data/layers", exist_ok=True)
    app.mount("/layers", StaticFiles(directory="/data/layers"), name="layers")
    print("✅ LayerStore Initialized")
except Exception as e:
    print(f"🔥 FATAL: Failed to initialize LayerStore or Mounts: {e}")
    traceback.print_exc()
    # Don't exit, let the app start so we can see the logs, but requests will fail

class ShardRequest(BaseModel):
    model_id: str

async def background_shard_task(model_id: str, hf_token: str):
    try:
        print(f"🔄 BACKGROUND: Starting shard for {model_id}")
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
    try:
        # 1. Check if physically exists
        if store.has_model(req.model_id):
            return {"status": "ready", "message": "Model already exists"}

        # 2. Check lock
        current_status = await r.get(f"shard_status:{req.model_id}")
        if current_status == "processing":
            return {"status": "processing", "message": "Already processing"}

        hf_token = settings.HF_TOKEN
        if not hf_token:
            print("❌ Error: HF_TOKEN is missing")
            raise HTTPException(500, "HF_TOKEN not set on Registry")

        # 3. Lock & Spawn
        await r.set(f"shard_status:{req.model_id}", "processing")
        background_tasks.add_task(background_shard_task, req.model_id, hf_token)

        return {"status": "started", "message": "Background task started"}

    except Exception as e:
        # FORCE PRINT ERROR TO LOGS
        print(f"🔥 API ERROR in /models/shard: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(500, f"Internal Server Error: {str(e)}")

@app.get("/models/status")
async def get_shard_status(model_id: str):
    try:
        if store.has_model(model_id): return {"status": "ready"}
        status = await r.get(f"shard_status:{model_id}")
        if not status: return {"status": "idle"}
        if status.startswith("failed"): return {"status": "failed", "error": status.split(": ", 1)[1]}
        return {"status": status}
    except Exception as e:
        print(f"🔥 ERROR in /models/status: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/models/info")
async def get_model_info(model_id: str):
    sanitized = model_id.replace("/", "_")
    path = f"/data/layers/{sanitized}/structure.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(404, "Model not sharded or found")
