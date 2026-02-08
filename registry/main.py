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
async def shard_model(model_id: str):
    model_dir = os.path.join(LAYER_DIR, model_id.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"🔽 Downloading {model_id} from HuggingFace...")

    cache_dir = snapshot_download(
        repo_id=model_id,
        token=os.getenv("HF_TOKEN"),
        ignore_patterns=["*.md", "*.txt"]
    )

    import shutil
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt"
    ]

    for fname in tokenizer_files:
        src = os.path.join(cache_dir, fname)
        dst = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"   ✓ Copied {fname}")

    config_path = os.path.join(cache_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    num_hidden_layers = cfg.get("num_hidden_layers")
    architectures = cfg.get("architectures", [])
    is_moe = any("Moe" in arch or "MoE" in arch for arch in architectures)

    logger.info(f"📊 Model: {num_hidden_layers} layers, MoE={is_moe}")

    layer_files = [f for f in os.listdir(cache_dir) if f.endswith('.safetensors') or f.endswith('.bin')]
    layer_files.sort()

    full_sd = {}
    for lf in layer_files:
        path = os.path.join(cache_dir, lf)
        if lf.endswith('.safetensors'):
            from safetensors.torch import load_file
            full_sd.update(load_file(path))
        else:
            full_sd.update(torch.load(path, map_location='cpu'))

    structure = {"layers": []}

    # Embedding layer
    embed_layer = {k: v for k, v in full_sd.items() if 'embed' in k.lower()}
    torch.save(embed_layer, os.path.join(model_dir, "layer_embed.pt"))
    structure["layers"].append({"type": "embed", "layer_idx": -1, "file": "layer_embed.pt"})
    logger.info("   ✅ Sharded embedding layer")

    # Hidden layers
    for i in range(num_hidden_layers):
        prefix = f"model.layers.{i}."
        layer_weights = {k: v for k, v in full_sd.items() if k.startswith(prefix)}

        if is_moe:
            # Extract router
            router_weights = {k: v for k, v in layer_weights.items() if 'gate' in k or 'router' in k or ('block_sparse_moe.gate' in k)}
            router_file = f"layer_{i}_router.pt"
            torch.save(router_weights, os.path.join(model_dir, router_file))

            # Extract experts
            num_experts = cfg.get("num_local_experts", 8)
            expert_files = []
            for e in range(num_experts):
                expert_weights = {k: v for k, v in layer_weights.items() if f".experts.{e}." in k}
                expert_file = f"layer_{i}_expert_{e}.pt"
                torch.save(expert_weights, os.path.join(model_dir, expert_file))
                expert_files.append(expert_file)

            structure["layers"].append({
                "type": "moe",
                "layer_idx": i,
                "router_file": router_file,
                "expert_files": expert_files
            })
            logger.info(f"   ✅ Sharded MoE layer {i} (router + {num_experts} experts)")
        else:
            # Dense layer
            dense_file = f"layer_{i}_dense.pt"
            torch.save(layer_weights, os.path.join(model_dir, dense_file))
            structure["layers"].append({
                "type": "dense",
                "layer_idx": i,
                "file": dense_file
            })
            logger.info(f"   ✅ Sharded dense layer {i}")

    # LM head
    lm_head = {k: v for k, v in full_sd.items() if 'lm_head' in k or 'output' in k}
    torch.save(lm_head, os.path.join(model_dir, "layer_lm_head.pt"))
    structure["layers"].append({"type": "lm_head", "layer_idx": num_hidden_layers, "file": "layer_lm_head.pt"})
    logger.info("   ✅ Sharded LM head")

    # Save structure
    with open(os.path.join(model_dir, "structure.json"), 'w') as f:
        json.dump(structure, f, indent=2)

    await r.set(f"shard_status:{model_id}", "completed")
    logger.info(f"✅ Sharding complete for {model_id}")

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
