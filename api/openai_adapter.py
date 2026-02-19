"""
api/openai_adapter.py

OpenAI-compatible adapter for the azu decentralized inference API.

Mounts /v1/chat/completions onto the existing FastAPI app.
Works with the Vercel AI SDK, OpenAI clients, and anything else
that speaks the OpenAI chat completions wire format.

Auth: pass the user's wallet address (pubkey) as a Bearer token.
  Authorization: Bearer <wallet_pubkey>

Usage — add one line to api/main.py:

    from api.openai_adapter import mount_openai_adapter
    mount_openai_adapter(app)
"""

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from shared.config import get_config
from shared.ledger import TransactionType, get_ledger
from shared.economics import calculate_cost_breakdown

config = get_config()

# ---------------------------------------------------------------------------
# OpenAI wire-format models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    # Any extra fields are silently ignored — keeps compatibility wide.
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True,
    )


def _extract_pubkey(request: Request) -> str:
    """Pull wallet pubkey from Authorization: Bearer <pubkey>."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header required: Bearer <wallet_pubkey>",
        )
    pubkey = auth.removeprefix("Bearer ").strip()
    if not pubkey:
        raise HTTPException(status_code=401, detail="Empty Bearer token.")
    return pubkey


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """
    Collapse the message list into a single prompt string.

    Uses the ChatML-ish format that most azu-supported models understand.
    System message is prepended verbatim; then alternating Human/Assistant turns.
    """
    parts: list[str] = []
    for m in messages:
        if m.role == "system":
            parts.append(f"<|system|>\n{m.content}")
        elif m.role == "user":
            parts.append(f"<|user|>\n{m.content}")
        elif m.role == "assistant":
            parts.append(f"<|assistant|>\n{m.content}")
    parts.append("<|assistant|>")  # prime the model to respond
    return "\n".join(parts)


async def _submit_job(
    r: redis.Redis,
    user_pubkey: str,
    model_id: str,
    prompt: str,
    est_tokens: int,
) -> tuple[str, float]:
    """Validate balance, lock funds, push to job_queue. Returns (job_id, cost)."""
    MIN_BALANCE = 0.001
    ledger = await get_ledger(r)
    balance = await ledger.get_balance(user_pubkey)

    if balance.available < MIN_BALANCE:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance: {balance.available or 0}. "
                   "Deposit tokens at POST /deposit.",
        )

    cost_breakdown = calculate_cost_breakdown(
        num_layers=40,
        num_tokens=est_tokens,
        is_moe=False,
        num_experts=1,
    )
    estimated_cost = cost_breakdown.total_cost

    try:
        await ledger.lock_funds(
            address=user_pubkey,
            amount=estimated_cost,
            job_id="",
        )
    except ValueError as e:
        raise HTTPException(status_code=402, detail=f"Cannot lock funds: {e}")

    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model_id": model_id,
        "input_prompt": prompt,
        "tokens": est_tokens,
        "owner": user_pubkey,
        "cost": estimated_cost,
    }
    await r.rpush("job_queue", json.dumps(job))
    return job_id, estimated_cost


async def _poll_result(
    r: redis.Redis,
    job_id: str,
    timeout: float = 120.0,
    interval: float = 0.5,
) -> dict:
    """Block until result:{job_id} appears in Redis or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        raw = await r.get(f"result:{job_id}")
        if raw:
            return json.loads(raw)
        await asyncio.sleep(interval)
    raise HTTPException(
        status_code=504,
        detail=f"Job {job_id} timed out after {timeout}s.",
    )


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _build_completion(
    job_id: str,
    model_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Build a standard OpenAI chat completion response object."""
    return {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_completion(
    job_id: str,
    model_id: str,
    content: str,
) -> AsyncIterator[str]:
    """
    Yield SSE chunks in the OpenAI streaming format.

    The azu scheduler currently returns the full completion at once rather
    than streaming individual tokens. We therefore emit the content as a
    single delta chunk followed by the terminal [DONE] event.

    When the scheduler gains per-token streaming (WebSocket push or Redis
    pub/sub), replace the two `yield` calls below with an async loop that
    reads from the token stream and yields one chunk per token.
    """
    chunk_id = f"chatcmpl-{job_id}"
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # Role chunk
    yield _chunk({"role": "assistant"})

    # Content chunk  (split into words to give the UI a typing feel)
    words = content.split(" ")
    for i, word in enumerate(words):
        text = word if i == 0 else " " + word
        yield _chunk({"content": text})
        await asyncio.sleep(0)  # yield to event loop between chunks

    # Final chunk
    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    user_pubkey = _extract_pubkey(request)

    body = await request.json()
    req = ChatCompletionRequest(**body)

    prompt = _messages_to_prompt(req.messages)
    est_tokens = req.max_tokens or 256

    r = await _get_redis()
    job_id, _ = await _submit_job(r, user_pubkey, req.model, prompt, est_tokens)

    result = await _poll_result(r, job_id)

    # The scheduler stores results as {"output": "...", "tokens_generated": N, ...}
    content: str = result.get("output") or result.get("text") or ""
    completion_tokens: int = result.get("tokens_generated", len(content.split()))
    prompt_tokens: int = len(prompt.split())

    if req.stream:
        return StreamingResponse(
            _stream_completion(job_id, req.model, content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    return JSONResponse(
        _build_completion(job_id, req.model, content, prompt_tokens, completion_tokens)
    )


# ---------------------------------------------------------------------------
# Models list endpoint (AI SDK probes this on init)
# ---------------------------------------------------------------------------

async def list_models(request: Request) -> JSONResponse:
    """Return a minimal /v1/models response."""
    # Optionally proxy to the registry to get the real list.
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "azu/default",
                "object": "model",
                "created": 1700000000,
                "owned_by": "azu",
            }
        ],
    })


# ---------------------------------------------------------------------------
# Mount function — call this from api/main.py
# ---------------------------------------------------------------------------

def mount_openai_adapter(app: FastAPI) -> None:
    """
    Attach OpenAI-compatible routes to an existing FastAPI app.

    In api/main.py, add:

        from api.openai_adapter import mount_openai_adapter
        mount_openai_adapter(app)
    """
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions,
        methods=["POST"],
        tags=["OpenAI Compatibility"],
        summary="OpenAI-compatible chat completions",
    )
    app.add_api_route(
        "/v1/models",
        list_models,
        methods=["GET"],
        tags=["OpenAI Compatibility"],
        summary="List available models",
    )
