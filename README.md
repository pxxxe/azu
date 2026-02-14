# azu

> **"Go forth and multiply."**

**Beta** — Decentralised inference network. Consumer GPU hardware, pooled.

Azu splits large language models across a network of independent workers and runs inference as a distributed pipeline. A user submits a prompt, the core routes it through sharded layers held by different machines, and the result comes back — no single node needs to hold the full model.

Payments will flow over Hyperliquid. Workers earn HYPE for compute they contribute. The platform takes a 20% cut; the rest goes straight to the worker.

---

## How It Works

A job moves through three stages:

**1. Sharding** — When a model is first requested, the Registry downloads it from Hugging Face and splits it into individual layer files. Dense layers become single `.pt` files. MoE layers are split further: one file for the router/gate, one per expert. The resulting structure is stored on disk and served to workers on demand.

**2. Planning** — The Scheduler knows what's connected. When a job arrives it pulls the model's `structure.json` from the Registry, then walks through every layer and assigns it to a worker based on available VRAM, whether that worker already has the layer cached, and pipeline locality (keeping consecutive layers on the same machine where possible). For MoE layers each expert is placed independently — experts can land on different workers than the router. The output is a topology: an ordered list of nodes with routing instructions.

**3. Execution** — The topology is dispatched. The first worker in the chain tokenises the prompt and runs it through the embedding layer, then forwards the hidden-state tensor to the next worker via a direct HTTP P2P link (not back through the core). Each worker runs its assigned layer(s) and passes the tensor along. MoE router workers fan out tensor slices to the relevant expert workers, collect results, apply routing weights, and forward the merged output. The last worker runs the LM head and decodes the output token.

The pipeline supports full autoregressive generation — not single-token decoding. Workers iteratively process tokens through the complete layer stack, with the final layer's output fed back as input for the next generation step until the specified token limit is reached or an end-of-sequence token is generated.

### Sharding strategies

| Type | What gets split | File layout |
|---|---|---|
| **Dense** | Each transformer layer is one shard | `layer_0_dense.pt`, `layer_1_dense.pt`, … |
| **MoE** | Router + each expert are separate shards | `layer_N_router.pt`, `layer_N_expert_0.pt`, … |

Both types coexist in the same model. Mixtral, for example, is fully MoE. Qwen2-MoE mixes dense and MoE layers. The Registry detects this automatically from the model config.

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

**Scheduler** — Maintains persistent WebSocket connections to every worker. Consumes the job queue, plans layer placement, dispatches execution instructions, and writes completed results back to Redis.

**Registry** — Owns model sharding and layer storage. Downloads models from Hugging Face, extracts per-layer weight files, and serves them as static files to workers. Tracks shard status in Redis so the Scheduler can do just-in-time sharding if a model hasn't been seen before.

**Workers** — Connect to the Scheduler over WebSocket on startup and report their GPU, VRAM, and a public P2P URL. They pull layer files from the Registry on demand and cache them in VRAM. They also run a small HTTP server on port 8003 for direct peer-to-peer tensor transfer.

---

## Project Layout

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
│   ├── worker/             # Worker node
│   │   ├── main.py
│   │   ├── layer_loader.py
│   │   └── requirements.txt
│   └── user/               # CLI for interacting with the network
│       ├── main.py
│       └── requirements.txt
├── shared/                 # Shared config, Hyperliquid helpers, economics
│   ├── config.py
│   ├── hyperliquid_lib.py
│   └── economics.py
├── Dockerfile.core         # Builds the core (API + Scheduler + Registry + Redis)
├── Dockerfile.worker       # Builds a worker node (CUDA)
├── infra_test.py           # Full e2e test (deploys to RunPod, runs Mixtral)
└── e2etest.sh              # Shell wrapper for infra_test.py
```

---

## Environment Variables

Both the core and workers read from a `.env` file (or system environment). Create one at the project root before building.

| Variable | Required by | Description |
|---|---|---|
| `HYPERLIQUID_RPC_URL` | Core | Hyperliquid RPC endpoint (mainnet or testnet) |
| `HYPERLIQUID_ADDRESS` | Core | Address of the wallet that receives user deposits |
| `SCHEDULER_PRIVATE_KEY` | Core | JSON array of bytes — the keypair used to sign worker payouts |
| `REDIS_HOST` | Core | Redis hostname (`localhost` inside the core container) |
| `REDIS_PORT` | Core | Redis port (`6379`) |
| `HF_TOKEN` | Core, Workers | Hugging Face API token for downloading gated models |
| `SCHEDULER_URL` | Workers | WebSocket URL of the Scheduler (`ws://…:8001/ws/worker`) |
| `REGISTRY_URL` | Workers | HTTP URL of the Registry (`http://…:8002`) |
| `P2P_PUBLIC_URL` | Workers | The externally reachable URL of this worker's P2P server |
| `P2P_URL_TEMPLATE` | Workers | (Alternative to above) A URL template with `{RUNPOD_POD_ID}` etc. for cloud environments |

Minimal `.env` for local testing:

```env
HYPERLIQUID_RPC_URL=https://api.hyperliquid.xyz
HYPERLIQUID_ADDRESS=<your_platform_address>
SCHEDULER_PRIVATE_KEY=[<byte,array>]
REDIS_HOST=localhost
REDIS_PORT=6379
HF_TOKEN=hf_your_token_here
```

---

## Setup & Local Testing

Prerequisites: Docker, Docker Compose (or equivalent), and a funded Hyperliquid wallet.

### 1. Build the images

```bash
# Core — API, Scheduler, Registry, Redis all in one container
docker build -f Dockerfile.core -t azu-core .

# Worker — requires a CUDA-capable GPU
docker build -f Dockerfile.worker -t azu-worker .
```

The core Dockerfile installs CPU-only PyTorch intentionally — the core never runs inference. This keeps the image small. The worker Dockerfile pulls from the official `pytorch/pytorch` CUDA runtime image.

### 2. Run the core

```bash
docker run -d --name azu-core \
  --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  azu-core
```

Give it a few seconds to start Redis and spin up all three services. You can verify with:

```bash
curl http://localhost:8002/health
```

### 3. Run a worker

You need a machine with an NVIDIA GPU and the CUDA toolkit installed.

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

> **Note on P2P URLs:** Workers need to be reachable by *other* workers for tensor transfer. In a single-machine local test `127.0.0.1:8003` works. In a multi-machine or cloud setup, set `P2P_PUBLIC_URL` to the externally reachable address (or use `P2P_URL_TEMPLATE` for platform-specific URL patterns like RunPod proxies).

Spin up as many workers as you have GPUs. Each one connects back to the Scheduler automatically.

### 4. Shard a model

The Registry shards on demand when a job arrives, but you can also trigger it manually to pre-load:

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

Using the user CLI (requires a Hyperliquid wallet):

```bash
cd client/user
pip install -r requirements.txt

# Deposit HYPE (you need a real on-chain transaction signature)
python main.py deposit 0.01

# Run inference
python main.py prompt "What is the capital of France?" --model Qwen/Qwen2.5-0.5B
```

Or directly via curl:

```bash
# 1. Deposit (you need a real on-chain transaction signature)
curl -X POST http://localhost:8000/deposit \
  -H "Content-Type: application/json" \
  -d '{"tx_sig": "<hyperliquid_tx_signature>", "user_address": "<your_address>"}'

# 2. Submit
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"user_address": "<your_address>", "model_id": "Qwen/Qwen2.5-0.5B", "prompt": "What is the capital of France?", "est_tokens": 50}'

# 3. Poll for result (use the job_id from step 2)
curl http://localhost:8000/results/<job_id>
```

---

## End-to-End Cloud Test

`infra_test.py` is a full integration test that deploys everything to RunPod Secure Cloud, runs Mixtral-8x7B through the network, and tears everything down afterwards. It handles wallet setup, pod deployment with GPU fallback, health polling, and automatic cleanup.

To run it you need the following environment variables set:

```bash
export RUNPOD_API_KEY=<your_key>
export HF_TOKEN=<your_hf_token>
export HYPERLIQUID_RPC=<rpc_url>               # testnet recommended
export FUNDED_ACCOUNT_KEY=<base58_or_json>      # a wallet with enough HYPE to cover test transfers
export VOLUME_ID=<runpod_volume_id>              # persistent volume for HF model cache
```

Then:

```bash
pip install runpod requests solana solders aiohttp
python infra_test.py
```

The script deploys the core and two workers, shards Mixtral, deposits funds on behalf of a test user, submits a prompt, polls until completion, and prints the output. All pods are terminated in a `finally` block regardless of outcome.

---

## Economics

Pricing is based on **token-layers** — one token passing through one layer.

| Unit | Cost |
|---|---|
| 1 token-layer | 2 Lamports |
| 70B model (80 layers), 100 tokens | 16,000 Lamports (~$0.002) |

Revenue split: **80% to the worker**, 20% to the platform. Workers accumulate earnings and are paid out once their balance crosses a threshold (to be determined based on Hyperliquid gas costs).

---

## Supported Architectures

The worker's layer loader has explicit support for the following model families, plus a generic fallback that attempts dynamic import for any architecture following the standard `XForCausalLM` → `XDecoderLayer` naming pattern:

Llama, Mistral, Mixtral, Qwen2, Qwen2-MoE, GPT-2, GPT-Neo, GPT-J, OPT, BLOOM, Falcon, MPT, Phi, Phi-3, Gemma, Gemma2, Starcoder2, DeepSeek-V2.

---

## Core Scheduler Components — Implementation Status

The following components of the core Scheduler have been implemented and are functional:

- **Worker Registry**: Maintains a list of connected workers with their GPU capabilities, VRAM availability, and P2P endpoints.
- **Layer Planning**: Calculates optimal layer placement based on worker VRAM, cache state, and pipeline locality.
- **Job Queue Consumption**: Reads jobs from Redis and dispatches them to workers.
- **Topology Generation**: Creates execution plans mapping each layer to specific workers.
- **Result Aggregation**: Collects outputs from workers and assembles the final response.

The following components require implementation or completion:

### 1. Payment Processing (Hyperliquid Integration)

**Status: NOT IMPLEMENTED**

The economics module exists and the cost calculation is correct, but the actual HYPE transfer to workers on job completion is not triggered automatically. This is the highest-priority gap for production readiness.

Required work:

- Implement Hyperliquid wallet connection and signature verification
- Build deposit confirmation flow (verify on-chain transactions)
- Create worker payout mechanism (batch payments when balance threshold is reached)
- Wire up payment triggers on successful job completion

### 2. Fault Tolerance — Worker Disconnection Handling

**Status: NOT IMPLEMENTED**

If a worker drops mid-job the job fails. There is no checkpoint or retry at the layer level.

Required work:

- **Heartbeat Monitoring**: Implement periodic health checks between Scheduler and workers. Detect worker disconnection within a configurable timeout window (recommended: 5-10 seconds).
- **Failure Detection**: When a worker fails to acknowledge a tensor transfer or heartbeat, mark that worker as unavailable.
- **Layer Redistribution**: When a worker disconnects mid-pipeline:
  - Identify which layers were assigned to the failed worker
  - Find replacement workers with sufficient VRAM and the required layers cached (prefer workers with already-cached layers to minimise load time)
  - If no cached layer exists, dispatch the failed worker to retrieve the layer from the Registry
  - Rebuild the pipeline topology with the new worker(s)
  - Resume execution from the last successfully processed layer (requires workers to store intermediate hidden states or implement rollback)
- **Job Retry Logic**: For jobs that cannot be recovered (e.g., too many worker failures), implement automatic re-queue with appropriate backoff.

### 3. VRAM Accounting (Live GPU Memory Queries)

**Status: ESTIMATED ONLY**

The Scheduler tracks used VRAM based on layer file sizes, not live GPU memory queries. Workers can OOM under heavy load.

Required work:

- Implement periodic VRAM reporting from workers (query actual GPU memory usage via `torch.cuda.memory_allocated()`)
- Update Scheduler's worker state with real-time VRAM availability
- Add OOM prediction and preemption before job assignment

### 4. P2P Tensor Transfer Encryption

**Status: NOT IMPLEMENTED**

Tensors move over plain HTTP between workers. Not suitable for sensitive workloads yet.

Required work:

- Implement TLS for P2P connections between workers
- Add tensor integrity verification (hash checksums)

---

## Known Limitations (Beta)

- **Payment layer is not wired into job completion.** The economics module exists and the cost calculation is correct, but the actual HYPE transfer to workers on job completion is not triggered automatically in this build.
- **No fault tolerance.** If a worker drops mid-job the job fails. There is no checkpoint or retry at the layer level.
- **VRAM accounting is estimated.** The Scheduler tracks used VRAM based on layer file sizes, not live GPU memory queries. Workers can OOM under heavy load.
- **P2P tensor transfer is unencrypted.** Tensors move over plain HTTP between workers. Not suitable for sensitive workloads yet.
