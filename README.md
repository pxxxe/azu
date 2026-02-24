# azu

**Go forth and multiply.**

Decentralized inference network.

Azu splits large language models across a network of independent workers and runs inference as a distributed pipeline. A user submits a prompt, the core routes it through sharded layers held by different machines, and the result comes back — no single node needs to hold the full model.

Payments flow over Hyperliquid. Workers earn HYPE for compute they contribute. The platform takes a 20% cut; the rest goes straight to the worker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Core                             │
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
          │   │   Worker 0     │  │  Worker 1 │  │
          │   │   :8003 (P2P)  │──│  :8003    │  │  ← direct tensor transfer
          │   └────────────────┘  └───────────┘  │
          │           …                           │
          └───────────────────────────────────────┘
                      Worker Network
```

**API** — Entry point for users. Accepts deposits (verified on-chain via Hyperliquid) and inference job submissions. Writes jobs to a Redis queue. Serves results once they arrive. OpenAI-compatible (`/v1/chat/completions`).

**Scheduler** — Maintains persistent WebSocket connections to every worker. Consumes the job queue, plans layer placement across available VRAM, dispatches execution instructions, credits workers on job completion, and triggers on-chain payouts when thresholds are met.

**Registry** — Owns model sharding and layer storage. Downloads models from HuggingFace, extracts per-layer weight files as safetensors, and serves them as static files to workers. Tracks shard status in Redis for just-in-time sharding.

**Workers** — Connect to the Scheduler over WebSocket on startup and report GPU specs, VRAM, and a public P2P URL. Pull layer files from the Registry on demand and cache them in VRAM. Run an HTTP server on `:8003` for direct peer-to-peer tensor transfer. Layer files are verified against HuggingFace's published SHA256 checksums before loading — a rogue registry cannot serve malicious weights.

---

## Package Structure

Azu is structured as three installable namespace packages under `azu.*`.

```
packages/
├── azu-shared/        # Config, economics, ledger, auth, payments
├── azu-core/          # API, Scheduler, Registry
└── azu-worker/        # Worker node
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

| Variable | Required by | Description |
|---|---|---|
| `PAYMENT_PROVIDER` | Core | `hyperliquid` or `solana` |
| `HYPERLIQUID_RPC_URL` | Core | Hyperliquid RPC endpoint |
| `HYPERLIQUID_ADDRESS` | Core | Platform wallet address (receives deposits) |
| `SCHEDULER_PRIVATE_KEY` | Core | Private key for signing worker payouts |
| `REDIS_HOST` | Core | Redis hostname |
| `REDIS_PORT` | Core | Redis port |
| `AUTH_SECRET_KEY` | Core, Workers | Shared secret for interworker HMAC auth |
| `HF_TOKEN` | Core, Workers | HuggingFace token (required for gated models) |
| `SCHEDULER_URL` | Workers | WebSocket URL of the Scheduler |
| `REGISTRY_URL` | Workers | HTTP URL of the Registry |
| `P2P_PUBLIC_URL` | Workers | Externally reachable URL of worker's P2P server |
| `P2P_URL_TEMPLATE` | Workers | URL template with `{RUNPOD_POD_ID}` for cloud deployments |

Minimal `.env` for local development:

```env
PAYMENT_PROVIDER=hyperliquid
HYPERLIQUID_RPC_URL=https://rpc.hyperliquid-testnet.xyz/evm
HYPERLIQUID_ADDRESS=<your_platform_address>
SCHEDULER_PRIVATE_KEY=<hex_or_json_array>
REDIS_HOST=localhost
REDIS_PORT=6379
AUTH_SECRET_KEY=<random_32_byte_hex>
HF_TOKEN=hf_your_token_here
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

Run a worker:

```bash
docker run -d --name azu-worker-0 \
  --gpus all \
  --env SCHEDULER_URL=ws://host.docker.internal:8001/ws/worker \
  --env REGISTRY_URL=http://host.docker.internal:8002 \
  --env AUTH_SECRET_KEY=<same_key_as_core> \
  --env HF_TOKEN=hf_your_token_here \
  --env P2P_PUBLIC_URL=http://127.0.0.1:8003 \
  -p 8003:8003 \
  azu-worker
```

Spin up as many workers as you have GPUs. Each connects to the Scheduler automatically on startup.

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
# Deposit
curl -X POST http://localhost:8000/deposit \
  -H "Content-Type: application/json" \
  -d '{"tx_sig": "<tx_signature>", "user_pubkey": "<your_address>"}'

# Submit
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"user_pubkey": "<your_address>", "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": "What is the capital of France?", "est_tokens": 50}'

# Poll result
curl http://localhost:8000/results/<job_id>
```

### OpenAI-compatible

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your_wallet_address>" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

---

## Security

**Interworker auth** — The Scheduler generates a per-job HMAC-SHA256 token (keyed on `AUTH_SECRET_KEY`) and distributes it to all participating workers inside the `JOB_START` message. Workers verify the `x-auth-token` header on every incoming P2P request. Auth is disabled when `AUTH_SECRET_KEY` is not set (local dev).

**Layer integrity** — Every safetensors file downloaded from the registry is verified against HuggingFace's published SHA256 manifest before loading. Files that fail verification are deleted and the load is aborted. Only `.safetensors` files are accepted — `.pt` and `.bin` (pickle-based, RCE vector) are rejected before any bytes are written to disk.

---

## Economics

Pricing unit: **token-layers** — one token passing through one layer.

| Unit | Cost |
|---|---|
| 1 token-layer | 2 Lamports |
| 70B model (80 layers), 100 tokens | 16,000 Lamports (~$0.002) |

Revenue split: **80% to the worker**, 20% to the platform. Workers accumulate earnings in an internal Redis-backed ledger and receive on-chain payouts when their balance crosses `PAYOUT_THRESHOLD` (default: 0.001).

---

## Payment Layer

Pluggable provider interface. Switch at runtime via `PAYMENT_PROVIDER`:

```bash
PAYMENT_PROVIDER=hyperliquid   # default
PAYMENT_PROVIDER=solana        # fallback
```

To implement a custom provider, subclass `azu.shared.payments.base.PaymentProvider` and register it in `azu/shared/payments/factory.py`.

---

## Supported Model Architectures

Llama, Mistral, Mixtral (MoE), Qwen2, Qwen2-MoE, GPT-2, GPT-Neo, GPT-J, OPT, BLOOM, Falcon, MPT, Phi, Phi-3, Gemma, Gemma2, Starcoder2, DeepSeek-V2.

Generic fallback for any architecture following the standard `XForCausalLM` → `XDecoderLayer` naming convention.

---

## End-to-End Test

`infra_test.py` deploys the full stack to RunPod Secure Cloud, runs a model through the network, and tears everything down.

```bash
export RUNPOD_API_KEY=<key>
export HF_TOKEN=<token>
export HYPERLIQUID_RPC_URL=https://rpc.hyperliquid-testnet.xyz/evm
export FUNDED_ACCOUNT_KEY=<hex_or_json>
export VOLUME_ID=<runpod_volume_id>

python infra_test.py
```

---

## License

MIT — see [LICENSE](LICENSE).
