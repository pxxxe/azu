# azu

Decentralized inference network.

Azu splits large language models across a network of independent workers and runs inference as a distributed pipeline. A user submits a prompt, the scheduler plans layer placement across available worker VRAM, dispatches execution, and the result comes back. No single node holds the full model.

Payments flow over Hyperliquid. Workers earn HYPE for compute contributed. The platform takes 20%; the rest goes to the worker. The API is OpenAI-compatible (`/v1/chat/completions`).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                            Core                              │
│                                                              │
│  ┌──────────┐   ┌──────────────────────┐   ┌─────────────┐  │
│  │   API    │   │      Scheduler       │   │  Registry   │  │
│  │  :8000   │   │       :8001          │   │    :8002    │  │
│  └────┬─────┘   └──────────┬───────────┘   └──────┬──────┘  │
│       │                    │                       │         │
│       │  job_queue         │  dispatch             │  HTTP   │
│       │  (Redis)           │  (WS or HTTP)         │  layers │
│       └────────────────────┤                       │         │
│                       Redis :6379                  │         │
└───────────────────────────────────────────────────-┼─────────┘
                             │                       │
          ┌──────────────────┼───────────────────────┤
          │      persistent  │  serverless           │
          │    ┌─────────────▼──────┐  ┌─────────────▼──────┐
          │    │  Worker (WS)       │  │  Worker (HTTP)      │
          │    │  :8003 (P2P)       │  │  :8003 (P2P+ctrl)  │
          │    └────────────────────┘  └────────────────────┘
          │              ↕ tensor (P2P HTTP, direct)
          └────────────────────────────────────────────────────
```

**API** — Entry point. Verifies on-chain deposits, accepts job submissions, writes to Redis queue, serves results.

**Scheduler** — Manages worker pool. Plans layer placement across available VRAM. Dispatches control messages to workers via WebSocket (persistent) or HTTP POST (serverless). Credits workers on completion and triggers on-chain payouts at threshold.

**Registry** — Downloads models from HuggingFace, extracts per-layer weight files as safetensors, serves them over HTTP. Workers verify SHA-256 checksums before loading.

**Workers (persistent)** — Connect to the Scheduler over WebSocket on startup. Report GPU specs, VRAM, and a public P2P URL. Pull layer files from Registry on demand, cache in VRAM. Run an HTTP server on `:8003` for direct P2P tensor transfer.

**Workers (serverless)** — No persistent connection. On startup, register their HTTP endpoint with the scheduler via `POST /workers`. Receive control messages (`JOB_START`, `EXECUTE_*`) on `POST /control` (served on the same `:8003` aiohttp app as the P2P server). Report job results via `POST /worker/result`. Set `WORKER_MODE=serverless`.

---

## Package Structure

```
packages/
├── azu-shared/        # Config, economics, ledger, auth, payments
├── azu-core/          # API, Scheduler, Registry
│   └── src/azu/core/scheduler/
│       ├── main.py            # Scheduler + HTTP worker registry API
│       ├── worker_registry.py # Redis-backed durable endpoint records
│       └── worker_driver.py   # Transport abstraction (WS / HTTP)
└── azu-worker/        # Worker node
    └── src/azu/worker/
        └── main.py    # Persistent + serverless run modes
```

Install for local development:

```bash
pip install -e packages/azu-shared
pip install -e packages/azu-core
pip install -e packages/azu-worker
```

CLI entry points:

```bash
azu-core         # starts API + Scheduler + Registry
azu-api          # API only     (:8000)
azu-scheduler    # Scheduler only (:8001)
azu-registry     # Registry only  (:8002)
azu-worker       # Worker node    (:8003)
```

---

## Environment Variables

| Variable | Used by | Description |
|---|---|---|
| `PAYMENT_PROVIDER` | core | `hyperliquid` or `solana` |
| `HYPERLIQUID_RPC_URL` | core | Hyperliquid RPC endpoint |
| `HYPERLIQUID_ADDRESS` | core | Platform wallet address (receives deposits) |
| `SCHEDULER_PRIVATE_KEY` | core | Private key for signing worker payouts |
| `REDIS_HOST` | core | Redis hostname |
| `REDIS_PORT` | core | Redis port |
| `AUTH_SECRET_KEY` | core, workers | Shared secret for inter-worker HMAC auth |
| `HF_TOKEN` | core, workers | HuggingFace token (required for gated models) |
| `SCHEDULER_URL` | workers | WebSocket URL of the Scheduler (`ws://host:8001/ws/worker`) |
| `SCHEDULER_HTTP_URL` | workers (serverless) | HTTP base URL of the Scheduler. Auto-derived from `SCHEDULER_URL` if not set. |
| `WORKER_MODE` | workers | `persistent` (default) or `serverless` |
| `REGISTRY_URL` | workers | HTTP URL of the Registry |
| `P2P_PUBLIC_URL` | workers | Externally reachable URL of worker's P2P server |
| `P2P_URL_TEMPLATE` | workers | URL template using `{RUNPOD_POD_ID}` for cloud deployments |

Minimal `.env`:

```env
PAYMENT_PROVIDER=hyperliquid
HYPERLIQUID_RPC_URL=https://rpc.hyperliquid-testnet.xyz/evm
HYPERLIQUID_ADDRESS=<platform_address>
SCHEDULER_PRIVATE_KEY=<hex_or_json_array>
REDIS_HOST=localhost
REDIS_PORT=6379
AUTH_SECRET_KEY=<random_32_byte_hex>
HF_TOKEN=hf_...
```

---

## Running with Docker

Build:

```bash
docker build -f Dockerfile.core -t azu-core .
docker build -f Dockerfile.worker -t azu-worker .
```

Run core:

```bash
docker run -d --name azu-core \
  --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  azu-core
```

Run a persistent worker:

```bash
docker run -d --name azu-worker-0 \
  --gpus all \
  --env SCHEDULER_URL=ws://host.docker.internal:8001/ws/worker \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env AUTH_SECRET_KEY=<same_key_as_core> \
  --env HF_TOKEN=hf_... \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker
```

Run a serverless worker:

```bash
docker run -d --name azu-worker-sl \
  --gpus all \
  --env WORKER_MODE=serverless \
  --env SCHEDULER_HTTP_URL=http://host.docker.internal:8001 \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env AUTH_SECRET_KEY=<same_key_as_core> \
  --env HF_TOKEN=hf_... \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker
```

---

## Usage

### Shard a model

```bash
curl -X POST http://localhost:8002/models/shard \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}'

curl "http://localhost:8002/models/status?model_id=Qwen/Qwen2.5-0.5B-Instruct"
```

### Submit a job

```bash
curl -X POST http://localhost:8000/deposit \
  -H "Content-Type: application/json" \
  -d '{"tx_sig": "<tx_signature>", "user_pubkey": "<your_address>"}'

curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"user_pubkey": "<your_address>", "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": "What is the capital of France?", "est_tokens": 50}'

curl http://localhost:8000/results/<job_id>
```

### OpenAI-compatible

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your_wallet_address>" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

### Worker registry API (scheduler)

```bash
# list all workers (persistent + serverless, live + durable)
curl http://localhost:8001/workers

# register a serverless worker endpoint (done automatically by the worker on startup)
curl -X POST http://localhost:8001/workers \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "Worker_abc123",
    "worker_type": "serverless",
    "endpoint_url": "https://pod-8003.proxy.runpod.net/control",
    "vram_mb": 24000,
    "payment_address": "0x..."
  }'

# deregister a worker
curl -X DELETE http://localhost:8001/workers/Worker_abc123
```

---

## Scheduler Dispatch

The scheduler selects a transport per worker based on `worker_type` stored in the registry:

| Worker type | Registration | Control messages | Results |
|---|---|---|---|
| `persistent` | WebSocket `REGISTER` message | Pushed over open WebSocket | `RESULT` message over WebSocket |
| `serverless` | `POST /workers` (HTTP) | `POST /control` on worker's P2P server | `POST /worker/result` on scheduler |

The planner (`_plan_job`) treats both types identically — VRAM accounting is the same. The only difference is transport. Persistent and serverless workers can coexist in the same job topology.

Two-phase dispatch for jobs containing serverless workers:
1. Scheduler POSTs `JOB_START` to all workers (WS or HTTP).
2. Serverless workers cold-start, bind their P2P server, then call `POST /worker/ready` on the scheduler with their ephemeral P2P URL.
3. Scheduler waits up to 60s for all serverless workers to report ready, patches topology URLs, then sends `EXECUTE_*` messages.

Persistent workers go straight to phase 2 (existing `asyncio.sleep(2)` handshake window).

---

## Security

**Inter-worker auth** — The Scheduler generates a per-job HMAC-SHA256 token (keyed on `AUTH_SECRET_KEY`) inside `JOB_START`. Workers attach it as `x-auth-token` on every outgoing P2P tensor request. Receiving workers verify the header before processing. Disabled when `AUTH_SECRET_KEY` is unset.

**Layer integrity** — Every safetensors file is verified against HuggingFace's published SHA-256 manifest before loading. Files failing verification are deleted and the load aborts. Only `.safetensors` files are accepted — `.pt` and `.bin` are rejected before any bytes are written to disk.

---

## Economics

Pricing unit: **token-layers** — one token passing through one layer.

| Unit | Cost |
|---|---|
| 1 token-layer | 2 Lamports |
| 70B model (80 layers), 100 tokens | 16,000 Lamports (~$0.002) |

Revenue split: 80% to the worker, 20% to the platform. Workers accumulate earnings in a Redis-backed ledger and receive on-chain payouts when balance crosses `PAYOUT_THRESHOLD` (default: 0.001).

---

## Payment Layer

Pluggable provider interface. Switch via `PAYMENT_PROVIDER`:

```bash
PAYMENT_PROVIDER=hyperliquid   # default
PAYMENT_PROVIDER=solana
```

To add a provider, subclass `azu.shared.payments.base.PaymentProvider` and register in `azu/shared/payments/factory.py`.

---

## Supported Model Architectures

Llama, Mistral, Mixtral (MoE), Qwen2, Qwen2-MoE, GPT-2, GPT-Neo, GPT-J, OPT, BLOOM, Falcon, MPT, Phi, Phi-3, Gemma, Gemma2, Starcoder2, DeepSeek-V2.

Generic fallback for any architecture following the standard `XForCausalLM` → `XDecoderLayer` naming convention.

---

## End-to-End Test

`infra_test.py` deploys the full stack to RunPod Secure Cloud and tears everything down on exit. Runs persistent worker inference (sections 0–5), then serverless worker inference in isolation (section 6) — persistent workers are terminated before section 6 so the scheduler is forced to route through the HTTP dispatch path.

```bash
export RUNPOD_API_KEY=<key>
export HF_TOKEN=<token>
export FUNDED_ACCOUNT_KEY=<hex_or_json>
export VOLUME_ID=<runpod_volume_id>

python infra_test.py
```

---

## License

MIT — see [LICENSE](LICENSE).
