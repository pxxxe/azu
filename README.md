# azu

Decentralized inference network. Split large language models across independent worker nodes, run distributed pipeline inference, pay workers on-chain.

---

## Overview

Azu splits a model's layers across a cluster of workers. A user submits a prompt via an OpenAI-compatible API. The scheduler plans layer placement across available worker VRAM, dispatches execution in two phases, workers stream tensors peer-to-peer, and the result is returned. No single node holds the full model.

Workers earn HYPE for compute contributed. The platform takes 20%; 80% goes to the worker. Payouts settle on-chain via Hyperliquid (or Solana) when a worker's accumulated balance crosses the payout threshold.

---

## Architecture

```
user
 │  HTTP
 ▼
┌─────────────────────────────────────────┐  CORE
│  API :8000                              │
│   │ enqueue                             │
│  Redis :6379 ◄── dequeue ── Scheduler :8001
│                              │                │
│  Registry :8002 ─────────────┼── layer files  │
└─────────────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    │  WORKER NETWORK        │
                    │                        │
              Worker (persistent)    Worker (serverless)
              WS + :8003 P2P         /control + :8003 P2P
              layers 0–N             layers M–K
                    │                        │
                    └──── tensor (P2P) ───────┘
```

### Components

| name | port | package | description |
|---|---|---|---|
| API | :8000 | azu-core | Entry point. Verifies on-chain deposits, accepts job submissions via OpenAI-compatible `/v1/chat/completions` or raw `/submit`, writes to Redis queue, polls for and serves results. |
| Scheduler | :8001 | azu-core | Manages worker pool. Plans layer placement across available VRAM. Dispatches control messages via WebSocket (persistent) or HTTP long-poll (serverless). Threads sampling params through to workers. Credits workers on completion, triggers on-chain payouts at threshold. |
| Registry | :8002 | azu-core | Downloads models from HuggingFace, extracts per-layer safetensors, serves over HTTP. Workers verify SHA-256 checksums before loading. |
| Worker (persistent) | :8003 | azu-worker | Connects to Scheduler over WebSocket on startup. Reports GPU specs and VRAM. Pulls layer files from Registry on demand. Runs P2P HTTP server for direct tensor transfer. |
| Worker (serverless) | :8003 | azu-worker | No persistent connection. Registers with scheduler via `POST /workers`. Receives control messages via long-poll on `GET /worker/poll/{id}`. Reports results via `POST /worker/result`. Set `WORKER_MODE=serverless`. |

---

## Install

```bash
git clone https://github.com/pxxxe/azu
pip install -e packages/azu-shared
pip install -e packages/azu-core
pip install -e packages/azu-worker
```

---

## Run

```bash
# core (API + Scheduler + Registry)
azu-core

# persistent worker
azu-worker

# serverless worker
WORKER_MODE=serverless \
SCHEDULER_HTTP_URL=http://localhost:8001 \
azu-worker
```

---

## Docker

```bash
docker build -f Dockerfile.core   -t azu-core .
docker build -f Dockerfile.worker -t azu-worker .

docker run -d --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  azu-core

# persistent worker
docker run -d --gpus all \
  --env SCHEDULER_URL=ws://host.docker.internal:8001/ws/worker \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env AUTH_SECRET_KEY=<same_as_core> \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker

# serverless worker
docker run -d --gpus all \
  --env WORKER_MODE=serverless \
  --env SCHEDULER_HTTP_URL=http://host.docker.internal:8001 \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env AUTH_SECRET_KEY=<same_as_core> \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker
```

---

## HTTP API

### Inference — OpenAI-compatible

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <wallet_address>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

Streaming is supported via `"stream": true`. Response format matches the OpenAI wire protocol.

`temperature` and `top_p` are threaded through the full pipeline to the decode worker:
- `temperature = 0` → greedy decoding (deterministic, best for code/structured output)
- `temperature > 0` → nucleus (top-p) sampling (default `1.0 / 1.0`)

### Inference — raw prompt

```bash
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"user_pubkey":"0x...","model_id":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"hello","est_tokens":50}'

curl http://localhost:8000/results/<job_id>
```

### Model Sharding

```bash
# trigger shard (async — poll status)
curl -X POST http://localhost:8002/models/shard \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}'

curl "http://localhost:8002/models/status?model_id=Qwen/Qwen2.5-0.5B-Instruct"
```

### Worker Registry

```bash
# list all workers
curl http://localhost:8001/workers

# register a serverless worker (called automatically by the worker on startup)
curl -X POST http://localhost:8001/workers \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "Worker_abc123",
    "worker_type": "serverless",
    "endpoint_url": "https://pod-8003.proxy.runpod.net/control",
    "vram_mb": 24000,
    "payment_address": "0x..."
  }'

# deregister
curl -X DELETE http://localhost:8001/workers/Worker_abc123
```

### Payments

```bash
# verify an on-chain deposit and credit user's internal balance
curl -X POST http://localhost:8000/deposit \
  -H "Content-Type: application/json" \
  -d '{"tx_sig": "0x...", "user_pubkey": "0xYOUR_ADDRESS"}'
```

---

## Dispatch

The scheduler selects a transport per worker based on `worker_type` stored in the Redis-backed worker registry. Persistent and serverless workers can coexist in the same job topology.

| worker_type | registration | control messages | results |
|---|---|---|---|
| persistent | WebSocket REGISTER on connect | pushed over open WebSocket | RESULT message over WebSocket |
| serverless | POST /workers (HTTP) | long-poll GET /worker/poll/{id} | POST /worker/result on scheduler |

**Two-phase dispatch for serverless workers:**

1. Scheduler sends `JOB_START` to all workers in the topology.
2. Serverless workers cold-start, bind their P2P server, then call `POST /worker/ready` with their ephemeral P2P URL.
3. Scheduler waits up to 60s for all serverless workers to report ready, patches topology URLs, then sends `EXECUTE_*`.
4. Persistent workers proceed after the existing 2s handshake window.

Serverless workers use a long-poll model (`GET /worker/poll/{worker_id}`, held up to 29s) instead of the scheduler pushing to the worker's proxy URL. This avoids the 403 error on RunPod LB where `{podId}-{port}.proxy.runpod.net` blocks inbound connections.

---

## Prompt Formatting

The embed worker applies the model's chat template using `tokenizer.apply_chat_template()`. This means:

- Requests via `/v1/chat/completions` pass messages as a JSON array — the embed worker formats them correctly for whichever model is running (e.g. ChatML for Qwen, Llama-chat for Llama).
- Requests via `/submit` with a plain string prompt are automatically wrapped as a single user turn before the template is applied.
- The API layer has no knowledge of model-specific token formats. Chat template logic lives entirely on the worker.

---

## Security

**Inter-worker auth** — The Scheduler generates a per-job HMAC-SHA256 token keyed on `AUTH_SECRET_KEY` and sends it to all workers inside `JOB_START`. Workers attach it as `x-auth-token` on every outgoing P2P tensor request. Receiving workers verify before processing. Disabled when `AUTH_SECRET_KEY` is unset.

**Layer integrity** — Every safetensors file is verified against HuggingFace's published SHA-256 manifest before loading. Files failing verification are deleted and the load aborts. Only `.safetensors` files are accepted — `.pt` and `.bin` are rejected before any bytes are written to disk.

---

## Economics

Pricing unit: **token-layers** — one token passing through one layer.

Revenue split: 80% to the worker, 20% to the platform.

Workers accumulate earnings in a Redis-backed ledger and receive on-chain payouts when balance crosses `PAYOUT_THRESHOLD` (default: 0.001).

| unit | cost |
|---|---|
| 1 token-layer | 2 Lamports |
| 70B model (80 layers), 100 tokens | 16,000 Lamports (~$0.002) |

---

## Environment Variables

| variable | used by | description |
|---|---|---|
| `PAYMENT_PROVIDER` | core | `hyperliquid` or `solana` |
| `HYPERLIQUID_RPC_URL` | core | RPC endpoint |
| `HYPERLIQUID_ADDRESS` | core | Platform wallet (receives deposits) |
| `SCHEDULER_PRIVATE_KEY` | core | Signs worker payouts |
| `REDIS_HOST` | core | Redis hostname |
| `REDIS_PORT` | core | Redis port |
| `AUTH_SECRET_KEY` | core + workers | Shared secret for inter-worker HMAC auth. Disabled if unset. |
| `HF_TOKEN` | core + workers | HuggingFace token (required for gated models) |
| `SCHEDULER_URL` | workers | WebSocket URL of the Scheduler (`ws://host:8001/ws/worker`) |
| `SCHEDULER_HTTP_URL` | workers | HTTP base URL of the Scheduler. Auto-derived from `SCHEDULER_URL` if not set. Required for serverless mode. |
| `WORKER_MODE` | workers | `persistent` (default) or `serverless` |
| `REGISTRY_URL` | workers | HTTP URL of the Registry |
| `P2P_PUBLIC_URL` | workers | Externally reachable URL of worker P2P server |
| `P2P_URL_TEMPLATE` | workers | URL template with `{RUNPOD_POD_ID}` for cloud deployments |
| `IDLE_TIMEOUT` | workers | Seconds of inactivity before serverless worker self-terminates (scale-to-zero) |
| `WORKER_PRIVATE_KEY` | workers | Optional. Generated on first run if absent. |

---

## Supported Architectures

Llama, Mistral, Mixtral (MoE), Qwen2, Qwen2-MoE, GPT-2, GPT-Neo, GPT-J, OPT, BLOOM, Falcon, MPT, Phi, Phi-3, Gemma, Gemma2, Starcoder2, DeepSeek-V2, Qwen3.5.

Generic fallback for any architecture following the standard `XForCausalLM` → `XDecoderLayer` naming convention.

---

## License

MIT. See [LICENSE](LICENSE).
