import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
import traceback
import asyncio
import sys
import shutil
from pathlib import Path
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
    force: bool = False  # NEW: Force re-shard even if model exists

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
        # FORCE MODE: Delete existing files and re-shard
        if req.force:
            sanitized = req.model_id.replace("/", "_")
            model_dir = store.storage_path / sanitized
            if model_dir.exists():
                print(f"🗑️ FORCE MODE: Deleting existing model directory {model_dir}")
                shutil.rmtree(model_dir)
                print(f"✅ Deleted {model_dir}")

            # Clear Redis status
            await r.delete(f"shard_status:{req.model_id}")
            print(f"🔄 FORCE MODE: Re-sharding {req.model_id} from scratch")

        # 1. Check if physically exists (after potential deletion)
        if store.has_model(req.model_id):
            return {"status": "ready", "message": "Model already exists"}

        # 2. Check if currently sharding
        status = await r.get(f"shard_status:{req.model_id}")
        if status == "processing":
            return {"status": "processing", "message": "Sharding in progress"}

        # 3. Begin Sharding
        await r.set(f"shard_status:{req.model_id}", "processing")
        background_tasks.add_task(background_shard_task, req.model_id, settings.HF_TOKEN)

        return {"status": "processing", "message": "Sharding started"}

    except Exception as e:
        print(f"❌ Shard endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/models/status")
async def model_status(model_id: str):
    """Check the sharding/availability status of a model."""
    try:
        # Check physical existence
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

@app.get("/debug/files/{model_id}")
async def debug_files(model_id: str):
    """Debug endpoint to list all files for a model."""
    try:
        sanitized = model_id.replace("/", "_")
        model_dir = store.storage_path / sanitized

        if not model_dir.exists():
            return {"error": "Model directory not found", "path": str(model_dir)}

        files = []
        for f in model_dir.glob("*"):
            files.append({
                "name": f.name,
                "size": f.stat().st_size if f.is_file() else 0,
                "is_file": f.is_file(),
                "path": str(f)
            })

        return {
            "model_id": model_id,
            "directory": str(model_dir),
            "file_count": len(files),
            "files": sorted(files, key=lambda x: x["name"])
        }
    except Exception as e:
        return {"error": str(e)}

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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "storage_path": str(store.storage_path),
        "storage_exists": store.storage_path.exists()
    }
