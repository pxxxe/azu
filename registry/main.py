import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from shared.config import settings

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

# ========================
# MODELS
# ========================

class WorkerCapability(BaseModel):
    worker_id: str
    gpu_name: str
    vram_gb: float
    models: List[str]  # ["Qwen/Qwen2.5-0.5B", ...]
    layers: Dict[str, List[int]]  # {"model_id": [0,1,2,3,...]}

class LayerAssignmentRequest(BaseModel):
    model_id: str
    layer_start: int
    layer_end: int

class ModelSpec(BaseModel):
    model_id: str
    num_layers: int
    vram_per_layer_mb: float
    dtype: str = "float16"

# ========================
# MODEL SPECS DATABASE
# ========================

# Hardcoded for common models (in production, fetch from HF)
MODEL_SPECS = {
    "Qwen/Qwen2.5-0.5B": {"num_layers": 24, "vram_per_layer_mb": 80},
    "Qwen/Qwen2.5-1.5B": {"num_layers": 28, "vram_per_layer_mb": 200},
    "meta-llama/Llama-2-7b-hf": {"num_layers": 32, "vram_per_layer_mb": 800},
    "meta-llama/Llama-2-70b-hf": {"num_layers": 80, "vram_per_layer_mb": 3500},
}

@app.get("/models/specs")
async def list_model_specs():
    """Returns known model specifications"""
    return MODEL_SPECS

@app.post("/models/register")
async def register_model_spec(spec: ModelSpec):
    """Allows adding new model specs"""
    MODEL_SPECS[spec.model_id] = {
        "num_layers": spec.num_layers,
        "vram_per_layer_mb": spec.vram_per_layer_mb
    }
    return {"status": "registered", "model_id": spec.model_id}

# ========================
# WORKER REGISTRATION
# ========================

@app.post("/workers/register")
async def register_worker(cap: WorkerCapability):
    """Worker announces what it can handle"""

    # Store worker capabilities
    await r.set(
        f"worker:{cap.worker_id}",
        json.dumps({
            "gpu": cap.gpu_name,
            "vram": cap.vram_gb,
            "models": cap.models,
            "layers": cap.layers
        }),
        ex=300  # 5 min TTL, workers must heartbeat
    )

    # Index by model for fast lookup
    for model_id in cap.models:
        await r.sadd(f"workers_for_model:{model_id}", cap.worker_id)

    return {"status": "registered", "worker_id": cap.worker_id}

@app.post("/workers/heartbeat/{worker_id}")
async def heartbeat(worker_id: str):
    """Worker stays alive"""
    await r.expire(f"worker:{worker_id}", 300)
    return {"status": "alive"}

# ========================
# LAYER ASSIGNMENT
# ========================

@app.post("/assign/layers")
async def assign_layers(req: LayerAssignmentRequest):
    """
    Finds best worker(s) for executing layers [start:end] of a model.
    Returns: List of worker assignments
    """

    # Get workers that support this model
    worker_ids = await r.smembers(f"workers_for_model:{req.model_id}")

    if not worker_ids:
        raise HTTPException(404, f"No workers available for {req.model_id}")

    # Filter workers that have these layers
    candidates = []
    for wid in worker_ids:
        data = await r.get(f"worker:{wid}")
        if not data:
            continue

        worker = json.loads(data)
        if req.model_id in worker["layers"]:
            available_layers = set(worker["layers"][req.model_id])
            requested_layers = set(range(req.layer_start, req.layer_end))

            # Check if worker can handle ALL requested layers
            if requested_layers.issubset(available_layers):
                candidates.append({
                    "worker_id": wid,
                    "gpu": worker["gpu"],
                    "layers": list(requested_layers)
                })

    if not candidates:
        # Fallback: Split across multiple workers
        return await split_across_workers(req, worker_ids)

    # Return best candidate (for now, just pick first)
    return {"assignments": [candidates[0]]}

async def split_across_workers(req: LayerAssignmentRequest, worker_ids: set):
    """
    Advanced: Split layer range across multiple workers.
    For MVP, we'll just fail if no single worker can handle it.
    """
    raise HTTPException(
        503,
        f"No single worker can handle layers {req.layer_start}-{req.layer_end}. "
        "Multi-worker splitting not yet implemented."
    )

# ========================
# MODEL ANALYSIS
# ========================

@app.get("/models/{model_id}/analyze")
async def analyze_model(model_id: str):
    """
    Returns how many workers can handle this model and layer coverage.
    """
    worker_ids = await r.smembers(f"workers_for_model:{model_id}")

    if not worker_ids:
        return {"model_id": model_id, "workers": 0, "coverage": "0%"}

    # Calculate layer coverage
    if model_id not in MODEL_SPECS:
        return {"error": "Model spec unknown"}

    total_layers = MODEL_SPECS[model_id]["num_layers"]
    covered_layers = set()

    for wid in worker_ids:
        data = await r.get(f"worker:{wid}")
        if data:
            worker = json.loads(data)
            if model_id in worker["layers"]:
                covered_layers.update(worker["layers"][model_id])

    coverage = (len(covered_layers) / total_layers) * 100

    return {
        "model_id": model_id,
        "workers": len(worker_ids),
        "total_layers": total_layers,
        "covered_layers": len(covered_layers),
        "coverage": f"{coverage:.1f}%"
    }
