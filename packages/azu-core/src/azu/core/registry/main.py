import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import traceback
import asyncio
import sys
import shutil
from pathlib import Path
from shared import get_config
from .layer_storage import LayerStore

# Setup logging flush
sys.stdout.reconfigure(line_buffering=True)

config = get_config()

app = FastAPI()
r = redis.Redis(host=config.redis.host, port=config.redis.port, decode_responses=True)

# Initialize Store safely
try:
    print("🚀 Initializing LayerStore...")
    store = LayerStore()
    os.makedirs("/data/layers", exist_ok=True)
    app.mount("/layers", StaticFiles(directory="/data/layers"), name="layers")
    print("✅ LayerStore Initialized")
except Exception as e:
    print(f"🔥 FATAL: Failed to initialize LayerStore or Mounts: {e}")
    traceback.print_exc()

class ShardRequest(BaseModel):
    model_id: str
    force: bool = False

async def background_shard_task(model_id: str, hf_token: str):
    try:
        print(f"🔄 BACKGROUND: Starting shard for {model_id}")
        # Run the heavy blocking IO in a separate thread
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
    """
    Trigger the sharding process using the robust LayerStore.
    """
    # Check if already done
    if not req.force:
        status = await r.get(f"shard_status:{req.model_id}")
        if status == "ready" and store.has_model(req.model_id):
            return {"status": "ready", "msg": "Model already sharded"}

    await r.set(f"shard_status:{req.model_id}", "processing")

    # Offload to background task so API doesn't time out
    background_tasks.add_task(background_shard_task, req.model_id, config.registry.hf_token)

    return {"status": "processing", "job_id": req.model_id}

@app.get("/models/status")
async def model_status(model_id: str):
    """Check the sharding/availability status of a model."""
    try:
        # Check physical existence in LayerStore
        if store.has_model(model_id):
            return {"status": "ready", "model_id": model_id}

        # Check Redis status
        status = await r.get(f"shard_status:{model_id}")
        if status:
            return {"status": status, "model_id": model_id}

        return {"status": "not_found", "model_id": model_id}

    except Exception as e:
        print(f"❌ Status check error: {e}")
        raise HTTPException(500, str(e))

@app.get("/models/info")
async def model_info(model_id: str):
    """Get detailed information about a sharded model."""
    try:
        sanitized = model_id.replace("/", "_")
        structure_path = store.storage_path / sanitized / "structure.json"

        if not structure_path.exists():
            raise HTTPException(404, f"Model {model_id} not found or not sharded")

        with open(structure_path) as f:
            return json.load(f)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Info error: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a sharded model to free up space or force re-shard."""
    try:
        sanitized = model_id.replace("/", "_")
        model_dir = store.storage_path / sanitized

        if not model_dir.exists():
            raise HTTPException(404, f"Model {model_id} not found")

        # Delete directory
        shutil.rmtree(model_dir)

        # Clear Redis status
        await r.delete(f"shard_status:{model_id}")

        return {
            "status": "deleted",
            "model_id": model_id,
            "message": f"Deleted {model_dir}"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Delete error: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "storage_path": str(store.storage_path),
        "storage_exists": store.storage_path.exists()
    }


# Import LayerStore at the end to avoid circular imports
