# azu

**"Go forth and multiply."**

Decentralized inference network. Consumer GPU hardware, pooled.

Azu splits large language models across a network of independent workers and runs inference as a distributed pipeline. A user submits a prompt, the core routes it through sharded layers held by different machines, and the result comes back — no single node needs to hold the full model.

Payments flow over Hyperliquid. Workers earn HYPE for compute they contribute. The platform takes a 20% cut; the rest goes straight to the worker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Core                               │
│                                                         │
│   ┌──────────┐   ┌───────────┐   ┌──────────────┐     │
│   │   API    │   │ Scheduler │   │   Registry   │     │
│   │  :8000   │   │  :8001    │   │    :8002     │     │
│   └────┬─────┘   └─────┬─────┘   └──────┬───────┘     │
│        │               │                │              │
│        │  job_queue    │  WebSocket     │  HTTP        │
│        │  (Redis)      │  (register,   │  (layers/   │
│        └───────────────┤   dispatch,   │   info,     │
│                        │   results)    │   shard)    │
│                        │               │              │
│                   Redis :6379          │              │
└────────────────────────┼───────────────┼──────────────┘
                         │               │
          ┌──────────────┼───────────────┼──────┐
          │              ▼               ▼      │
          │   ┌────────────────┐  ┌───────────┐  │
          │   │   Worker 0      |  │  Worker 1 │  │
          │   │   :8003 (P2P)  |──│  :8003    │  │  ← direct tensor transfer
          │   └────────────────┘  └───────────┘  │
          │           …                          │
          └──────────────────────────────────────┘
                    Worker Network
```

**API** — Entry point for users. Accepts deposits (verified on-chain via Hyperliquid) and inference job submissions. Writes jobs to a Redis queue. Serves results once they arrive.

**Scheduler** — Maintains persistent WebSocket connections to every worker. Consumes the job queue, plans layer placement, dispatches execution instructions, credits workers on job completion, and triggers on-chain payouts when thresholds are met. Writes completed results back to Redis.

**Registry** — Owns model sharding and layer storage. Downloads models from Hugging Face, extracts per-layer weight files, and serves them as static files to workers. Tracks shard status in Redis so the Scheduler can do just-in-time sharding if a model hasn't been seen before.

**Workers** — Connect to the Scheduler over WebSocket on startup and report their GPU, VRAM, and a public P2P URL. They pull layer files from the Registry on demand and cache them in VRAM. They also run a small HTTP server on port 8003 for direct peer-to-peer tensor transfer.

---

## System Layout

```
azu/
├── api/                    # User-facing API (FastAPI)
│   ├── main.py
│   └── requirements.txt
├── scheduler/              # Job planner & dispatcher (FastAPI + WebSocket)
│   ├── main.py
│   └── requirements.txt
├── registry/              # Model store & sharding engine (FastAPI)
│   ├── main.py
│   ├── layer_storage.py
│   └── requirements.txt
├── client/
│   ├── worker/            # Worker node
│   │   ├── main.py
│   │   ├── layer_loader.py
│   │   ├── wallet.py
│   │   └── requirements.txt
│   └── user/              # CLI for interacting with the network
├── shared/                # Shared utilities
│   ├── config.py         # Configuration management
│   ├── economics.py       # Cost calculations & revenue split
│   ├── ledger.py         # Internal balance ledger (Redis-backed)
│   └── payments/         # Pluggable payment provider layer
│       ├── base.py       # Abstract PaymentProvider interface
│       ├── factory.py    # Provider factory
│       ├── hyperliquid.py # Hyperliquid implementation
│       └── solana.py     # Solana implementation
├── Dockerfile.core        # Builds the core (API + Scheduler + Registry + Redis)
├── Dockerfile.worker      # Builds a worker node (CUDA)
└── infra_test.py         # Full e2e test (deploys to RunPod, runs Mixtral)
```

---

## Environment Variables

Both the core and workers read from environment variables.

| Variable | Required by | Description |
|---|---|---|
| `PAYMENT_PROVIDER` | Core | Payment provider: `hyperliquid` or `solana` |
| `HYPERLIQUID_RPC_URL` | Core | Hyperliquid RPC endpoint (mainnet or testnet) |
| `HYPERLIQUID_ADDRESS` | Core | Platform wallet address (receives user deposits) |
| `SCHEDULER_PRIVATE_KEY` | Core | Private key for signing worker payouts |
| `SOLANA_RPC_URL` | Core | Solana RPC endpoint (fallback provider) |
| `REDIS_HOST` | Core | Redis hostname |
| `REDIS_PORT` | Core | Redis port |
| `HF_TOKEN` | Core, Workers | Hugging Face API token |
| `SCHEDULER_URL` | Workers | WebSocket URL of the Scheduler |
| `REGISTRY_URL` | Workers | HTTP URL of the Registry |
| `P2P_PUBLIC_URL` | Workers | Externally reachable URL of worker's P2P server |
| `P2P_URL_TEMPLATE` | Workers | URL template with `{RUNPOD_POD_ID}` for cloud environments |

Minimal `.env` for local testing:

```env
PAYMENT_PROVIDER=hyperliquid
HYPERLIQUID_RPC_URL=https://api.hyperliquid-testnet.xyz
HYPERLIQUID_ADDRESS=<your_platform_address>
SCHEDULER_PRIVATE_KEY=[<byte_array>]
REDIS_HOST=localhost
REDIS_PORT=6379
HF_TOKEN=hf_your_token_here
```

---

## Payment Layer

The payment system uses a pluggable provider architecture supporting both Hyperliquid and Solana.

### Flow

1. **User Deposit** — User sends tokens to platform wallet on-chain. API verifies transaction and credits internal Redis ledger.

2. **Job Submission** — User submits inference job. Scheduler checks internal ledger balance, deducts estimated cost, enqueues job.

3. **Job Completion** — Workers complete inference. Scheduler credits worker internal ledger (80% of job cost).

4. **On-Chain Payout** — When worker ledger balance exceeds `PAYOUT_THRESHOLD` (default: 0.001), Scheduler triggers on-chain transfer from platform wallet to worker wallet.

### Provider Configuration

```python
# Switch between providers
export PAYMENT_PROVIDER=hyperliquid  # Primary
export PAYMENT_PROVIDER=solana       # Fallback
```

The provider is selected at runtime via `PAYMENT_PROVIDER` environment variable. Both providers implement the same `PaymentProvider` interface.

---

## Setup & Local Testing

Prerequisites: Docker, Docker Compose (or equivalent), and a funded Hyperliquid wallet.

### 1. Build the images

```bash
docker build -f Dockerfile.core -t azu-core .
docker build -f Dockerfile.worker -t azu-worker .
```

### 2. Run the core

```bash
docker run -d --name azu-core \
  --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  azu-core
```

Verify with:

```bash
curl http://localhost:8002/health
```

### 3. Run a worker

```bash
docker run -d --name azu-worker-0 \
  --gpus all \
  --env SCHEDULER_URL=ws://host.docker.internal:8001/ws/worker \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env HF_TOKEN=hf_your_token_here \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker
```

Spin up as many workers as you have GPUs. Each one connects back to the Scheduler automatically.

### 4. Shard a model

```bash
curl -X POST http://localhost:8002/models/shard \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-0.5B"}'
```

Check status:

```bash
curl "http://localhost:8002/models/status?model_id=Qwen/Qwen2.5-0.5B"
```

### 5. Fund an account & submit a job

```bash
# 1. Deposit (on-chain transaction signature required)
curl -X POST http://localhost:8000/deposit \
  -H "Content-Type: application/json" \
  -d '{"tx_sig": "<hyperliquid_tx_signature>", "user_pubkey": "<your_address>"}'

# 2. Submit inference job
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"user_pubkey": "<your_address>", "model_id": "Qwen/Qwen2.5-0.5B", "prompt": "What is the capital of France?", "est_tokens": 50}'

# 3. Poll for result
curl http://localhost:8000/results/<job_id>
```

---

## End-to-End Cloud Test

`infra_test.py` deploys everything to RunPod Secure Cloud, runs Mixtral-8x7B through the network, and tears everything down afterwards.

Required environment variables:

```bash
export RUNPOD_API_KEY=<your_key>
export HF_TOKEN=<your_hf_token>
export HYPERLIQUID_RPC_URL=https://api.hyperliquid-testnet.xyz
export FUNDED_ACCOUNT_KEY=<base58_or_json>
export VOLUME_ID=<runpod_volume_id>
```

Run:

```bash
python infra_test.py
```

---

## Economics

Pricing based on **token-layers** — one token passing through one layer.

| Unit | Cost |
|---|---|
| 1 token-layer | 2 Lamports |
| 70B model (80 layers), 100 tokens | 16,000 Lamports (~$0.002) |

Revenue split: **80% to the worker**, 20% to the platform. Workers accumulate earnings in the internal ledger and receive on-chain payouts when their balance crosses the threshold (default: 0.001 tokens).

---

## Supported Architectures

The worker's layer loader supports: Llama, Mistral, Mixtral, Qwen2, Qwen2-MoE, GPT-2, GPT-Neo, GPT-J, OPT, BLOOM, Falcon, MPT, Phi, Phi-3, Gemma, Gemma2, Starcoder2, DeepSeek-V2.

Plus a generic fallback for any architecture following the standard `XForCausalLM` → `XDecoderLayer` naming pattern.
