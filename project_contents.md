# Project: azu.cx

## File: .dockerignore

```plaintext
.env
.env.local
.git
__pycache__
*.pem
*.key
venv/
registry_data/
```

---

## File: .env

```plaintext
# NETWORK CONFIG
SOLANA_RPC_URL=https://api.devnet.solana.com
# SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# WALLETS
# The public key users send money TO
PLATFORM_WALLET_PUBKEY=YourPlatformPublicKeyHere
# The private key used to pay workers (JSON Array format)
SCHEDULER_PRIVATE_KEY=[11,22,33,...]

# INFRA
REDIS_HOST=redis
REDIS_PORT=6379
```

---

## File: Dockerfile.core

```plaintext
FROM python:3.10-slim

# Install Redis & System deps
RUN apt-get update && apt-get install -y redis-server git curl

WORKDIR /app

COPY registry/requirements.txt reqs_reg.txt
COPY scheduler/requirements.txt reqs_sched.txt
COPY api/requirements.txt reqs_api.txt

# --- FIX IS HERE ---
# 1. Install CPU-only Torch first (avoids downloading 4GB of CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Then install the rest (pip will see torch is already installed and skip the huge download)
RUN pip install --no-cache-dir -r reqs_reg.txt -r reqs_sched.txt -r reqs_api.txt

COPY . .

# Startup Script
RUN echo "#!/bin/bash\n\
redis-server --daemonize yes\n\
uvicorn registry.main:app --host 0.0.0.0 --port 8002 &\n\
uvicorn scheduler.main:app --host 0.0.0.0 --port 8001 &\n\
uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
" > start.sh && chmod +x start.sh

CMD ["./start.sh"]
```

---

## File: Dockerfile.worker

```plaintext
# CHANGE FROM 2.1.0 TO 2.2.0
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY client/worker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy shared library
COPY shared/ ./shared/

# Copy worker code
COPY client/worker/ ./

# Create cache directory
RUN mkdir -p /app/layer_cache

CMD ["python", "main.py"]
```

---

## File: api/Dockerfile

```plaintext
FROM python:3.10-slim
WORKDIR /app

# Copy Requirements
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy Shared Lib
COPY shared/ ./shared/
# Copy App
COPY api/ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## File: api/main.py

```python
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import json
from shared.config import settings
from shared.solana_lib import verify_deposit
from shared.economics import calculate_job_cost

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

class DepositReq(BaseModel):
    tx_sig: str
    user_pubkey: str

class JobReq(BaseModel):
    user_pubkey: str
    model_id: str
    prompt: str
    est_tokens: int = 100 # Default estimation

@app.post("/deposit")
async def deposit(req: DepositReq):
    # Idempotency
    if await r.get(f"tx:{req.tx_sig}"):
        return {"status": "processed"}

    amount = await verify_deposit(req.tx_sig)
    if amount == 0:
        raise HTTPException(400, "Invalid Transaction")

    # Credit Balance
    new_bal = await r.incrby(f"balance:{req.user_pubkey}", amount)
    await r.setex(f"tx:{req.tx_sig}", 86400, "1")

    return {"status": "success", "new_balance": new_bal}

@app.post("/submit")
async def submit(req: JobReq):
    # 1. Calc Cost (Assuming 80 layers for now)
    cost = calculate_job_cost(80, len(req.prompt.split()), req.est_tokens)

    # 2. Check Balance
    balance = await r.get(f"balance:{req.user_pubkey}")
    if not balance or int(balance) < cost:
        raise HTTPException(402, f"Insufficient funds. Cost: {cost}, Bal: {balance or 0}")

    # 3. Deduct
    await r.decrby(f"balance:{req.user_pubkey}", cost)

    # 4. Enqueue
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model": req.model_id,
        "input": req.prompt,
        "tokens": req.est_tokens,
        "owner": req.user_pubkey,
        "cost": cost
    }

    # Push to Redis Queue
    await r.rpush("job_queue", json.dumps(job))

    return {"job_id": job_id, "status": "queued", "cost": cost}

@app.get("/results/{job_id}")
async def get_result(job_id: str):
    """Poll for job result"""
    result = await r.get(f"result:{job_id}")

    if not result:
        # Check if job still in queue
        return {"status": "processing", "job_id": job_id}

    return json.loads(result)

# ALSO UPDATE handle_result in scheduler/main.py:
async def handle_result(self, wid, result_data):
    self.workers[wid]['status'] = "IDLE"

    # Store result in Redis
    job_id = result_data['job_id']
    await r.setex(
        f"result:{job_id}",
        3600,  # 1 hour TTL
        json.dumps(result_data)
    )
```

---

## File: api/requirements.txt

```plaintext
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
redis>=5.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
solana>=0.30.2
solders>=0.21.0
```

---

## File: client/user/main.py

```python
import typer
import asyncio
import aiohttp
import json
import os
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction

app = typer.Typer()

# LOAD CONFIG
API_URL = "http://localhost:8000"
with open(".env") as f:
    env = dict(line.strip().split('=') for line in f if '=' in line)

PLATFORM_WALLET = Pubkey.from_string(env['PLATFORM_WALLET_PUBKEY'])
RPC_URL = env['SOLANA_RPC_URL']

def get_wallet():
    with open(os.path.expanduser("~/.config/solana/id.json")) as f:
        return Keypair.from_bytes(bytes(json.load(f)))

@app.command()
def deposit(amount_sol: float):
    async def _run():
        kp = get_wallet()
        client = AsyncClient(RPC_URL)

        print(f"💳 Sending {amount_sol} SOL...")
        lamports = int(amount_sol * 1_000_000_000)

        # Create transfer instruction using the function, not a class
        ix = transfer(TransferParams(
            from_pubkey=kp.pubkey(),
            to_pubkey=PLATFORM_WALLET,
            lamports=lamports
        ))

        # Get latest blockhash
        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash

        # Create and sign transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            kp.pubkey(),
            [kp],
            blockhash
        )

        # Send transaction
        sig = await client.send_transaction(tx)
        print(f"Tx Sent: {sig.value}")

        # Notify Backend
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{API_URL}/deposit", json={
                "tx_sig": str(sig.value),
                "user_pubkey": str(kp.pubkey())
            }) as resp:
                print(await resp.json())

    asyncio.run(_run())

@app.command()
def prompt(text: str, model: str = "Qwen/Qwen2.5-0.5B"):
    async def _run():
        kp = get_wallet()
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{API_URL}/submit", json={
                "user_pubkey": str(kp.pubkey()),
                "model_id": model,
                "prompt": text
            }) as resp:
                print(await resp.json())

    asyncio.run(_run())

if __name__ == "__main__":
    app()
```

---

## File: client/user/requirements.txt

```plaintext
typer>=0.9.0
aiohttp>=3.9.0
solana>=0.30.2
solders>=0.21.0
```

---

## File: client/worker/Dockerfile

```plaintext
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY client/worker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared/ ./shared/

# Copy worker code
COPY client/worker/ ./

# Create cache directory
RUN mkdir -p /app/layer_cache

CMD ["python", "main.py"]
```

---

## File: client/worker/layer_loader.py

```python
import torch
import aiohttp
import os
from pathlib import Path
from transformers import AutoConfig
# We need these imports to construct the empty shell on the GPU
# so we can pour the weights into it.
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class LayerLoader:
    def __init__(self, registry_url, cache_dir="./layer_cache"):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_layers = {} # RAM Cache

    async def _download(self, url: str, path: Path):
        """Helper to download a file from Registry to Worker Disk"""
        if path.exists(): return

        print(f"   ⬇️ Downloading {url}...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Download failed [{resp.status}]: {url}")
                data = await resp.read()

                # Atomic write
                temp = path.with_suffix('.tmp')
                with open(temp, 'wb') as f: f.write(data)
                os.rename(temp, path)

    def _get_layer_class(self, config):
        """Map architecture string to actual PyTorch Class"""
        arch = config.architectures[0]
        if "Llama" in arch: return LlamaDecoderLayer
        if "Qwen" in arch: return Qwen2DecoderLayer
        if "Mistral" in arch: return MistralDecoderLayer
        if "GPT2" in arch: return GPT2Block
        raise ValueError(f"Worker does not support architecture: {arch}")

    async def load_layers(self, model_id: str, layer_indices: list, device="cuda"):
        """
        1. Download specific shard (layer_x.pt) from Registry.
        2. Create empty Transformer Layer on GPU.
        3. Load weights into it.
        """
        # RAM Cache check
        key = f"{model_id}_{tuple(layer_indices)}"
        if key in self.loaded_layers: return self.loaded_layers[key]

        print(f"📦 Loading layers {layer_indices} for {model_id}...")

        # 1. Get Config (Tiny JSON file)
        sanitized = model_id.replace("/", "_")
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        LayerClass = self._get_layer_class(config)
        modules = []

        for idx in layer_indices:
            filename = f"layer_{idx}.pt"
            path = self.cache_dir / f"{sanitized}_{filename}"

            # 2. Download Weights
            await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

            # 3. Create Empty Shell on GPU
            layer = LayerClass(config, layer_idx=idx).to(device).half()

            # 4. Fill with Weights
            state_dict = torch.load(path, map_location=device)
            layer.load_state_dict(state_dict)
            layer.eval()
            modules.append(layer)

        self.loaded_layers[key] = modules
        return modules

    async def load_embeddings(self, model_id: str, device="cuda"):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_embeddings.pt"

        # Download
        await self._download(f"{self.registry_url}/layers/{sanitized}/embeddings.pt", path)

        # Load
        config_path = self.cache_dir / f"{sanitized}_config.json"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        emb = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
        emb.load_state_dict(torch.load(path, map_location=device))
        emb.eval()
        return emb

    async def load_lm_head(self, model_id: str, device="cuda"):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_lm_head.pt"

        # Download
        await self._download(f"{self.registry_url}/layers/{sanitized}/lm_head.pt", path)

        # Load
        config_path = self.cache_dir / f"{sanitized}_config.json"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device).half()
        head.load_state_dict(torch.load(path, map_location=device))
        head.eval()
        return head
```

---

## File: client/worker/main.py

```python
import torch
import asyncio
import websockets
import aiohttp
import json
import base64
import io
import os
import urllib.request
from aiohttp import web
from transformers import AutoTokenizer

from layer_loader import LayerLoader

# Load Env
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

class ProductionWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)
        self.input_queue = asyncio.Queue() # For P2P tensors

        # State
        self.active_layers = []
        self.embeddings = None
        self.lm_head = None

        print(f"🚀 Worker Initialized on {self.device}")

    def get_public_ip(self):
        """Determines external IP for P2P connection"""
        try:
            return urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        except:
            print("⚠️ Could not resolve Public IP, using localhost")
            return "127.0.0.1"

    # --- P2P SERVER ---
    async def start_p2p_server(self, port=8003):
        app = web.Application()
        # Accept POST requests with binary tensors
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        print(f"👂 P2P Tensor Listener active on port {port}")

    async def handle_tensor_ingress(self, request):
        """Receives tensor from previous worker"""
        try:
            # Read binary body directly (more efficient than JSON wrapper for large tensors)
            data = await request.read()

            # If wrapped in JSON/base64 (Scheduler might enforce this format)
            # Let's support the JSON format used in the test harness
            try:
                json_data = await request.json()
                if 'tensor' in json_data:
                    tensor_bytes = base64.b64decode(json_data['tensor'])
                    buffer = io.BytesIO(tensor_bytes)
                    tensor = torch.load(buffer, map_location=self.device)
                    await self.input_queue.put(tensor)
                    return web.Response(text="Accepted")
            except:
                pass

            # Fallback: Raw binary
            buffer = io.BytesIO(data)
            tensor = torch.load(buffer, map_location=self.device)
            await self.input_queue.put(tensor)
            return web.Response(text="Accepted")

        except Exception as e:
            print(f"❌ Error receiving tensor: {e}")
            return web.Response(status=500, text=str(e))

    # --- INFERENCE LOGIC ---
    async def process_job(self, job, ws):
        job_id = job['id']
        model_id = job['model']
        layers = job['assigned_layers']
        is_first = job.get('is_first', False)
        is_last = job.get('is_last', False)
        next_worker_url = job.get('next_worker')

        print(f"⚙️ Processing Job {job_id} | Layers {layers}")

        try:
            # 1. Load Resources (skip if already loaded from PRELOAD)
            if not self.active_layers:
                self.active_layers = await self.loader.load_layers(model_id, layers, self.device)

            if is_first and not self.embeddings:
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)
            if is_last and not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            # 2. Prepare Input
            hidden_states = None

            if is_first:
                # Tokenize (Using HF Tokenizer)
                prompt = job['input']
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

                # Embedding Lookup
                with torch.no_grad():
                    hidden_states = self.embeddings(input_ids)
            else:
                # Wait for tensor from P2P
                print(f"⏳ Waiting for tensor input...")
                hidden_states = await self.input_queue.get()
                print("   ✅ Input received")

            # 3. Execute Layers
            # Real Torch Compute
            with torch.no_grad():
                # Ensure half precision
                hidden_states = hidden_states.half()

                for i, layer in enumerate(self.active_layers):
                    # Transformer layers usually return tuple (hidden_states, ...)
                    out = layer(hidden_states)
                    if isinstance(out, tuple):
                        hidden_states = out[0]
                    else:
                        hidden_states = out

            # 4. Handle Output
            if is_last:
                # LM Head -> Logits -> Text
                with torch.no_grad():
                    logits = self.lm_head(hidden_states)
                    # Simple Greedy Decode for now
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)

                    # We need the tokenizer again to decode
                    # (In efficient systems, we'd cache this or send tokens back)
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                    output_text = tokenizer.decode(next_token)

                # Send Final Result to Scheduler
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "output": output_text,
                    "status": "completed"
                }))
                print(f"🏁 Job {job_id} Completed. Output: {output_text}")

            else:
                # Send to Next Worker
                if not next_worker_url:
                    raise Exception("Not last worker, but no next_worker_url provided")

                print(f"📤 Sending tensor to {next_worker_url}")

                # Serialize
                buffer = io.BytesIO()
                torch.save(hidden_states, buffer)
                tensor_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(next_worker_url, json={"tensor": tensor_b64}) as resp:
                        if resp.status != 200:
                            raise Exception(f"Next worker rejected tensor: {resp.status}")

                # Notify Scheduler we are done with our part
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "status": "partial_complete"
                }))

        except Exception as e:
            print(f"❌ Job Execution Failed: {e}")
            import traceback
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    # --- MAIN LOOP ---
    async def run(self):
        # 1. Start P2P
        p2p_port = 8003
        await self.start_p2p_server(p2p_port)

        # 2. Connect to Scheduler
        retry_wait = 2
        while True:
            try:
                print(f"🔌 Connecting to Scheduler: {SCHEDULER_URL}")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    print("✅ Connected")

                    # Register
                    specs = {
                        "pubkey": "Worker_" + os.urandom(4).hex(), # Mock Pubkey for now
                        "gpu": torch.cuda.get_device_name(0),
                        "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                        "public_ip": self.get_public_ip(),
                        "p2p_port": p2p_port
                    }

                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": specs
                    }))

                    # Msg Loop
                    async for msg in ws:
                        data = json.loads(msg)
                        msg_type = data.get('type')

                        if msg_type == 'PRELOAD':
                            # Scheduler tells us to preload layers
                            model_id = data['model_id']
                            layers = data['layers']

                            print(f"📦 Preloading {model_id} layers {layers[0]}-{layers[-1]}...")

                            # Download and load to GPU
                            self.active_layers = await self.loader.load_layers(model_id, layers, self.device)

                            # Load embeddings if first worker
                            if data.get('is_first'):
                                self.embeddings = await self.loader.load_embeddings(model_id, self.device)

                            # Load LM head if last worker
                            if data.get('is_last'):
                                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

                            # Broadcast ready
                            await ws.send(json.dumps({
                                "type": "READY",
                                "model_id": model_id,
                                "layers": layers
                            }))

                            print(f"✅ Ready with {len(self.active_layers)} layers")

                        elif msg_type == 'EXECUTE':
                            # Job execution - layers already loaded from PRELOAD
                            asyncio.create_task(self.process_job(data['job'], ws))

            except Exception as e:
                print(f"❌ Connection Lost: {e}")
                await asyncio.sleep(retry_wait)
                retry_wait = min(retry_wait * 2, 30)

if __name__ == "__main__":
    worker = ProductionWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
```

---

## File: client/worker/requirements.txt

```plaintext
torch>=2.2.0
transformers>=4.38.0
huggingface-hub>=0.19.0
aiohttp>=3.9.0
solana>=0.30.2
solders>=0.21.0
websockets>=12.0
accelerate>=0.27.0
```

---

## File: contentgen.sh

```bash
#!/bin/bash

generate_file_contents() {
    local base="$1"
    local output_file="$2"
    local files=()

    # Read all files into array, excluding certain directories and file types
    while IFS= read -r file; do
        # Skip node_modules, .git, dist, .turbo, and binary files
        if [[ "$file" != *"node_modules"* && \
              "$file" != *".git"* && \
              "$file" != *"dist"* && \
              "$file" != *".turbo"* && \
              "$file" != *".png" && \
              "$file" != *".jpg" && \
              "$file" != *".jpeg" && \
              "$file" != *".gif" && \
              "$file" != *".pdf" && \
              "$file" != *".zip" && \
              "$file" != *".tar" && \
              "$file" != *".sqlite" && \
              "$file" != *".log" && \
              "$file" != *".gz" && \
              "$file" != *".ico" && \
              "$file" != *".aws-sam"* && \
              "$file" != *"__pycache__"* && \
              "$file" != *"venv"* && \
              "$file" != *"package-lock.json"* && \
              "$file" != *"dist"* && \
              "$file" != *".lock"* && \
              "$file" != *"pnpm-lock.yaml"* && \
              "$file" != *".next"* && \
              "$file" != *".erb"* && \
              "$file" != *"docs"* && \
              "$file" != *"types"* && \
              "$file" != *".map"* && \
              "$file" != *"assets"* && \
              "$file" != *"avatars"* && \
              "$file" != *"files"* && \
              "$file" != *"public"* && \
              "$file" != *"components/ui"* && \
              "$file" != *".DS_Store"* && \
              -f "$file" ]]; then
            files+=("$file")
        fi
    done < <(find "$base" -type f | sort)

    # Create or truncate the output file
    : > "$output_file"

    # Get project directory name
    local dir_name=$(basename "$PWD")
    echo "# Project: $dir_name" >> "$output_file"
    echo "" >> "$output_file"

    # Process each file
    for file in "${files[@]}"; do
        # Get relative path
        local rel_path="${file#./}"

        # Detect file type for syntax highlighting
        local extension="${file##*.}"
        local lang=""

        # Map common extensions to markdown language tags
        case "$extension" in
            js|jsx)     lang="javascript" ;;
            ts|tsx)     lang="typescript" ;;
            py)         lang="python" ;;
            rb)         lang="ruby" ;;
            java)       lang="java" ;;
            cpp|cc|c)   lang="cpp" ;;
            h|hpp)      lang="cpp" ;;
            css)        lang="css" ;;
            html)       lang="html" ;;
            xml)        lang="xml" ;;
            md)         lang="markdown" ;;
            sh)         lang="bash" ;;
            yaml|yml)   lang="yaml" ;;
            json)       lang="json" ;;
            *)          lang="plaintext" ;;
        esac

        # Add file header and content
        echo "## File: $rel_path" >> "$output_file"
        echo "" >> "$output_file"
        echo '```'"$lang" >> "$output_file"
        cat "$file" >> "$output_file"
        echo '```' >> "$output_file"
        echo "" >> "$output_file"
        echo "---" >> "$output_file"
        echo "" >> "$output_file"
    done

    echo "Generated file contents have been saved to $output_file"
}

# Set output file name
output_file="project_contents.md"

# Generate contents starting from current directory
generate_file_contents "." "$output_file"
```

---

## File: e2etest.sh

```bash
#!/bin/bash

# Exit on error
set -e

# Configuration
export RUNPOD_API_KEY=rpa_ZKZLYZ29PVCVGNVCDH50GWIWU6T4QA5NBDRUJFH1axwo14
export HF_TOKEN=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

echo "=========================================="
echo "🚀 AZU.CX E2E Test Suite"
echo "=========================================="
echo ""
echo "⚠️  Make sure GitHub Actions has built and pushed your images!"
echo "   Check: https://github.com/YOUR_USERNAME/azu.cx/actions"
echo ""
echo "⚠️  Make sure infra_test.py has correct config:"
echo "    - CORE_IMG = 'pxxxe/azu-core:latest'"
echo "    - WORKER_IMG = 'pxxxe/azu-worker:latest'"
echo "    - Your Volume ID"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo ""
echo "🚀 Launching RunPod instances..."
echo "   (RunPod will pull images directly from Docker Hub)"
echo ""

python3 infra_test.py

echo ""
echo "=========================================="
echo "✅ E2E Test Complete!"
echo "=========================================="
```

---

## File: envscript.sh

```bash
export RUNPOD_API_KEY=rpa_ZKZLYZ29PVCVGNVCDH50GWIWU6T4QA5NBDRUJFH1axwo14
export HF_TOKEN=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
export DOCKERHUB_TOKEN=dckr_pat_Lo3j0T60HK03bRY-XXQcIBeK3qs
export FUNDED_ACCOUNT_KEY=3qW64tsiFGDTVeDCHmWnzJNCwYBUMK4GXgWWn7NWmQc2A9hotHJytdhiqh6XAwqUqZKEZnb3aJPsSSDDSjJ7F761
```

---

## File: infra_test.py

```python
import runpod
import time
import requests
import os
import sys
import json
import asyncio

# === SOLANA IMPORTS ===
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction

# ================= CONFIGURATION =================
API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

CORE_IMG = "pxxxe/azu-core:latest"     # <--- UPDATE USERNAME
WORKER_IMG = "pxxxe/azu-worker:latest" # <--- UPDATE USERNAME

RPC_URL = "https://devnet.helius-rpc.com/?api-key=1d7ca6e1-7700-42eb-b086-8183fda42d76"
DISTRIBUTION_AMOUNT = 0.1
# =================================================

runpod.api_key = API_KEY

def load_funded_account():
    """Load your funded devnet account keypair"""
    global FUNDED_ACCOUNT_KEY
    if os.getenv("FUNDED_ACCOUNT_KEY"):
        secret = os.getenv("FUNDED_ACCOUNT_KEY", "")
        try:
            return Keypair.from_base58_string(secret)
        except:
            return Keypair.from_bytes(bytes(json.loads(secret)))

    if os.path.exists("funded_account.json"):
        with open("funded_account.json") as f:
            return Keypair.from_bytes(bytes(json.load(f)))

    solana_config = os.path.expanduser("~/.config/solana/id.json")
    if os.path.exists(solana_config):
        with open(solana_config) as f:
            return Keypair.from_bytes(bytes(json.load(f)))

    print("❌ ERROR: No funded account found!")
    sys.exit(1)

def get_pod_service_url(pod_id, internal_port):
    """
    Determines the best URL to reach a specific port on the pod.
    1. Checks for a true Public IP (194.x, etc).
    2. If IP is private (100.x), returns the RunPod Proxy domain IMMEDIATELY.
    """
    print(f"   ⏳ Resolving connection for {pod_id} (port {internal_port})...")

    for i in range(12):
        try:
            pod = runpod.get_pod(pod_id)
            runtime = pod.get('runtime')

            if runtime:
                found_ip = runtime.get('publicIp') or runtime.get('public_ip')
                found_port = None

                for p in runtime.get('ports', []):
                    if p['privatePort'] == internal_port:
                        found_port = p['publicPort']
                        if not found_ip: found_ip = p.get('ip')
                        break

                # CASE A: Real Public IP
                if found_ip and found_port and not (found_ip.startswith("100.") or found_ip.startswith("10.") or found_ip.startswith("192.")):
                    url = f"{found_ip}:{found_port}"
                    print(f"      ✅ Found Public IP: {url}")
                    return url, False

                # CASE B: Private/CGNAT IP -> Use Proxy
                if found_ip and (found_ip.startswith("100.") or found_ip.startswith("10.")):
                    proxy_domain = f"{pod_id}-{internal_port}.proxy.runpod.net"
                    print(f"      ✅ Detected Private IP ({found_ip}). Using Proxy: {proxy_domain}")
                    return proxy_domain, True

            status = pod.get('lastStatus') or "Unknown"
            if i % 2 == 0: print(f"      [{i}/12] Pod Status: {status}...")

        except Exception as e:
            print(f"      ⚠️ API Polling Error: {e}")

        time.sleep(10)

    print("      ⚠️ API Timeout or No IP info. Defaulting to Proxy URL.")
    return f"{pod_id}-{internal_port}.proxy.runpod.net", True

def wait_for_http(url, name="Service", retries=30):
    """Waits for the HTTP service inside the pod to actually start"""
    print(f"   ⏳ Waiting for {name} to be healthy at {url}...")
    for i in range(retries):
        try:
            requests.get(url, timeout=5)
            print(f"      ✅ {name} is responding!")
            return True
        except Exception as e:
            if i % 5 == 0:
                print(f"      [{i}/{retries}] Service starting... ({str(e)[:50]})")
            time.sleep(5)
    print(f"      ⚠️ {name} did not start in time. Check Pod Logs.")
    return False

def distribute_sol_with_retry(funder_kp, recipient_pubkey, amount_sol, client):
    lamports = int(amount_sol * 1_000_000_000)
    try:
        ix = transfer(TransferParams(from_pubkey=funder_kp.pubkey(), to_pubkey=recipient_pubkey, lamports=lamports))
        blockhash = client.get_latest_blockhash().value.blockhash
        tx = Transaction.new_signed_with_payer([ix], funder_kp.pubkey(), [funder_kp], blockhash)
        sig = client.send_transaction(tx).value
        print(f"      📤 Transfer sent: {sig}")
    except Exception as e:
        print(f"      ❌ Transfer failed: {e}")
        return False

    print("      ⏳ Waiting for balance update...")
    for _ in range(20):
        time.sleep(2)
        bal = client.get_balance(recipient_pubkey).value / 1e9
        if bal > 0:
            print(f"      ✅ Recipient balance confirmed: {bal} SOL")
            return True
    return False

def setup_solana_accounts():
    print("\n💰 0. Setting up Solana Wallets...")
    funder_kp = load_funded_account()
    client = Client(RPC_URL)
    funder_balance = client.get_balance(funder_kp.pubkey()).value / 1e9
    print(f"   💵 Funder balance: {funder_balance} SOL")

    if funder_balance < 0.2:
        print(f"❌ Insufficient funds! Need 0.2 SOL, have {funder_balance} SOL")
        sys.exit(1)

    platform_kp = Keypair()
    scheduler_kp = Keypair()
    user_kp = Keypair()

    print(f"   🔑 Platform: {platform_kp.pubkey()}")
    print(f"   🔑 Scheduler: {scheduler_kp.pubkey()}")
    print(f"   🔑 User: {user_kp.pubkey()}")

    print(f"\n   💸 Distributing {DISTRIBUTION_AMOUNT} SOL to Scheduler...")
    if not distribute_sol_with_retry(funder_kp, scheduler_kp.pubkey(), DISTRIBUTION_AMOUNT, client):
        sys.exit(1)

    print(f"\n   💸 Distributing {DISTRIBUTION_AMOUNT} SOL to User...")
    if not distribute_sol_with_retry(funder_kp, user_kp.pubkey(), DISTRIBUTION_AMOUNT, client):
        sys.exit(1)

    scheduler_priv = json.dumps(list(bytes(scheduler_kp)))
    return {
        "platform_pub": str(platform_kp.pubkey()),
        "scheduler_priv": scheduler_priv,
        "user_kp": user_kp
    }

def run_lifecycle():
    core_id = None
    worker_ids = []

    try:
        # STEP 0: WALLETS
        wallets = setup_solana_accounts()

        # STEP 1: CORE
        print("\n🚀 1. Deploying Core...")
        core = runpod.create_pod(
            name="azu-core",
            image_name=CORE_IMG,
            gpu_type_id="NVIDIA GeForce RTX 4090",
            cloud_type="COMMUNITY",
            ports="8000/http,8001/http,8002/http",
            env={
                "HF_TOKEN": HF_TOKEN,
                "REDIS_HOST": "localhost",
                "REDIS_PORT": "6379",
                "SOLANA_RPC_URL": RPC_URL,
                "PLATFORM_WALLET_PUBKEY": wallets['platform_pub'],
                "SCHEDULER_PRIVATE_KEY": wallets['scheduler_priv']
            }
        )
        core_id = core['id']

        # RESOLVE CONNECTION (Proxy vs IP)
        api_host, api_is_proxy = get_pod_service_url(core_id, 8000)
        reg_host, reg_is_proxy = get_pod_service_url(core_id, 8002)
        sched_host, sched_is_proxy = get_pod_service_url(core_id, 8001)

        api_scheme = "https" if api_is_proxy else "http"
        ws_scheme = "wss" if sched_is_proxy else "ws"

        api_url = f"{api_scheme}://{api_host}"
        reg_url = f"{api_scheme}://{reg_host}"
        sched_url = f"{ws_scheme}://{sched_host}/ws/worker"

        print(f"   ✅ Core API: {api_url}")
        print(f"   ✅ Registry: {reg_url}")
        print(f"   ✅ Scheduler: {sched_url}")

        wait_for_http(f"{reg_url}/docs", "Registry")

        # STEP 2: SHARDING
        print("\n⚡ 2. Sharding Model...")
        model_id = "Qwen/Qwen2.5-0.5B"

        print("   📥 Downloading and sharding (2-3 minutes)...")
        try:
            shard_resp = requests.post(
                f"{reg_url}/models/shard",
                json={"model_id": model_id},
                timeout=300
            )

            if shard_resp.status_code != 200:
                raise Exception(f"Sharding failed: {shard_resp.text}")

            data = shard_resp.json()
            print(f"   ✅ Model sharded: {data['num_layers']} layers")

        except Exception as e:
            print(f"   ❌ Sharding error: {e}")
            raise

        # STEP 3: WORKERS
        print("\n🚀 3. Deploying 2 GPU Workers...")
        for i in range(2):
            w = runpod.create_pod(
                name=f"azu-worker-{i}",
                image_name=WORKER_IMG,
                gpu_type_id="NVIDIA GeForce RTX 4090",
                cloud_type="COMMUNITY",
                ports="8003/http",
                env={
                    "SCHEDULER_URL": sched_url,
                    "REGISTRY_URL": reg_url,
                    "HF_TOKEN": HF_TOKEN
                }
            )
            worker_ids.append(w['id'])
            print(f"   Worker {i} deployed: {w['id']}")

        # STEP 3.5: Wait for workers to connect
        print("\n⏳ Waiting for workers to connect to scheduler...")
        scheduler_base = f"{api_url.replace('8000', '8001')}"

        time.sleep(20)  # Give pods time to start

        # Check workers are connected
        for i in range(12):
            try:
                workers_resp = requests.get(f"{scheduler_base}/workers")
                if workers_resp.status_code == 200:
                    worker_count = len(workers_resp.json())
                    if worker_count >= 2:
                        print(f"   ✅ {worker_count} workers connected")
                        break
            except:
                pass
            time.sleep(5)

        # STEP 3.6: Trigger preload
        print("\n📦 Triggering workers to preload model...")
        try:
            preload_resp = requests.post(
                f"{scheduler_base}/preload/{model_id}",
                timeout=10
            )
            if preload_resp.status_code == 200:
                print("   ✅ Preload triggered")
            else:
                print(f"   ⚠️ Preload response: {preload_resp.text}")
        except Exception as e:
            print(f"   ⚠️ Preload trigger failed: {e}")

        # STEP 3.7: Poll for worker readiness (NO ARBITRARY WAIT)
        print("\n⏳ Waiting for workers to load layers...")
        ready_url = f"{scheduler_base}/workers/ready"

        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                ready_resp = requests.get(ready_url, timeout=5)
                if ready_resp.status_code == 200:
                    data = ready_resp.json()
                    total = data['total']
                    ready = data['ready']

                    # Show status
                    print(f"   [{int(time.time() - start_time)}s] Workers: {ready}/{total} ready")
                    for w in data['workers']:
                        status = w['status']
                        layers = w['layers'] or 'none'
                        print(f"     - {w['id']}: {status} (layers: {layers})")

                    # Check if enough workers ready
                    if ready >= 2:
                        print(f"   ✅ {ready} workers READY!")
                        break
            except Exception as e:
                print(f"   ⚠️ Poll error: {e}")

            time.sleep(10)
        else:
            # Timeout - workers didn't become ready
            raise Exception(f"Workers not ready after {max_wait}s")

        # STEP 4: USER DEPOSIT (continues as before)
        print("\n💳 4. Simulating User Deposit...")
        user_kp = wallets['user_kp']
        platform_pub = Pubkey.from_string(wallets['platform_pub'])
        client = Client(RPC_URL)

        # FIX: Lower amount to cover fees
        transfer_amount = 0.05
        lamports = int(transfer_amount * 1_000_000_000)

        ix = transfer(TransferParams(from_pubkey=user_kp.pubkey(), to_pubkey=platform_pub, lamports=lamports))
        blockhash = client.get_latest_blockhash().value.blockhash
        tx = Transaction.new_signed_with_payer([ix], user_kp.pubkey(), [user_kp], blockhash)

        print(f"   Sending on-chain deposit transaction ({transfer_amount} SOL)...")
        sig = client.send_transaction(tx).value
        print(f"   Tx Sent: {sig}")

        # --- CRITICAL FIX: WAIT FOR CONFIRMATION ---
        print("      ⏳ Waiting 20s for transaction confirmation...")
        time.sleep(20)
        # -------------------------------------------

        print("   Notifying API of deposit...")
        wait_for_http(f"{api_url}/docs", "API")

        res = requests.post(f"{api_url}/deposit", json={
            "tx_sig": str(sig),
            "user_pubkey": str(user_kp.pubkey())
        })
        print(f"   Deposit Result: {res.json()}")

        # STEP 5: INFERENCE
        print("\n🧪 5. Running Inference Job...")
        res = requests.post(f"{api_url}/submit", json={
            "user_pubkey": str(user_kp.pubkey()),
            "model_id": "Qwen/Qwen2.5-0.5B",
            "prompt": "Explain quantum physics in one sentence.",
            "est_tokens": 50
        })
        print(f"   Submission: {res.text}")

        if res.status_code != 200:
            raise Exception(f"Submission failed: {res.text}")

        job_id = res.json()['job_id']
        print("   Polling results...")
        for _ in range(30):
            res = requests.get(f"{api_url}/results/{job_id}")
            data = res.json()
            if data.get('status') == 'completed':
                print(f"\n🎉 SUCCESS: {data['output']}\n")
                return
            if data.get('status') == 'failed':
                print(f"\n❌ JOB FAILED: {data.get('error')}\n")
                return
            time.sleep(2)
        raise Exception("Test Timed Out")

    except Exception as e:
        print(f"\n❌ CRITICAL FAIL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Tearing Down...")
        if core_id: runpod.terminate_pod(core_id)
        for wid in worker_ids: runpod.terminate_pod(wid)

if __name__ == "__main__":
    run_lifecycle()
```

---

## File: registry/Dockerfile

```plaintext
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY registry/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy shared library
COPY shared/ ./shared/

# Copy registry code
COPY registry/ ./

# Create data directory
RUN mkdir -p /data/layers

# Expose port
EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

---

## File: registry/layer_storage.py

```python
# registry/layer_storage.py
import torch
from transformers import AutoModel, AutoConfig
from pathlib import Path
import json
import os

class LayerStore:
    def __init__(self, storage_path="/data/layers"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def _find_layers(self, model):
        """
        Find the transformer layers in the model.
        Returns the actual layer list object.
        """
        # Try common layer attribute names
        for attr_path in [
            'model.layers',           # Llama, Mistral, Qwen
            'transformer.h',          # GPT-2, GPT-Neo
            'encoder.layer',          # BERT
            'decoder.layers',         # T5
            'h',                      # Some GPT models
            'layers'                  # Generic
        ]:
            try:
                obj = model
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)

                # Verify it's a list/ModuleList of layers
                if hasattr(obj, '__len__') and len(obj) > 0:
                    return obj
            except AttributeError:
                continue

        raise ValueError(f"Could not find transformer layers in model. Available attributes: {dir(model)}")

    def shard_model(self, model_id: str, hf_token: str):
        """
        Download model and extract individual layers.
        Returns number of layers extracted.
        """
        print(f"🔪 Sharding {model_id}...")

        # Load full model temporarily (CPU only to save VRAM)
        model = AutoModel.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        config = AutoConfig.from_pretrained(model_id, token=hf_token)

        # Find layer structure
        layers = self._find_layers(model)
        num_layers = len(layers)

        # Create storage directory
        model_dir = self.storage_path / model_id.replace("/", "_")
        model_dir.mkdir(exist_ok=True, parents=True)

        print(f"Found {num_layers} layers, extracting...")

        # Save each layer separately
        for i, layer in enumerate(layers):
            layer_path = model_dir / f"layer_{i}.pt"

            # Save layer weights
            torch.save(layer.state_dict(), layer_path)

            # Calculate size
            size_mb = layer_path.stat().st_size / (1024**2)

            # Store metadata
            metadata = {
                "model_id": model_id,
                "layer_idx": i,
                "size_mb": size_mb,
                "dtype": "float16",
                "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown"
            }

            with open(model_dir / f"layer_{i}.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Layer {i}/{num_layers-1}: {size_mb:.1f}MB")

        # Save embeddings separately
        if hasattr(model, 'get_input_embeddings'):
            emb = model.get_input_embeddings()
            torch.save(emb.state_dict(), model_dir / "embeddings.pt")
            print(f"  Embeddings: {(model_dir / 'embeddings.pt').stat().st_size / (1024**2):.1f}MB")

        # Save LM head if exists
        if hasattr(model, 'lm_head'):
            torch.save(model.lm_head.state_dict(), model_dir / "lm_head.pt")
            print(f"  LM Head: {(model_dir / 'lm_head.pt').stat().st_size / (1024**2):.1f}MB")

        # Save config
        config.save_pretrained(model_dir)

        # Save layer structure info
        structure_info = {
            "model_id": model_id,
            "num_layers": num_layers,
            "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown",
            "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
            "total_size_mb": sum((model_dir / f"layer_{i}.pt").stat().st_size for i in range(num_layers)) / (1024**2)
        }

        with open(model_dir / "structure.json", "w") as f:
            json.dump(structure_info, f, indent=2)

        del model  # Free memory
        torch.cuda.empty_cache()

        print(f"✅ Sharded {model_id} into {num_layers} layers")
        return num_layers

    def get_layer_path(self, model_id: str, layer_idx: int):
        """Get filesystem path to a layer file"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        return model_dir / f"layer_{layer_idx}.pt"

    def get_layer_metadata(self, model_id: str, layer_idx: int):
        """Get metadata for a specific layer"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        metadata_path = model_dir / f"layer_{layer_idx}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def get_model_structure(self, model_id: str):
        """Get overall model structure info"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        structure_path = model_dir / "structure.json"

        if not structure_path.exists():
            return None

        with open(structure_path) as f:
            return json.load(f)

    def has_model(self, model_id: str):
        """Check if model is already sharded"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        return (model_dir / "structure.json").exists()
```

---

## File: registry/main.py

```python
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
from shared.config import settings
from .layer_storage import LayerStore

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
store = LayerStore()

# ---------------------------------------------------------
# 1. FILE SERVER (Crucial)
# This allows the Worker's LayerLoader to download files.
# e.g., GET http://registry:8002/layers/Qwen_.../layer_0.pt
# ---------------------------------------------------------
app.mount("/layers", StaticFiles(directory="/data/layers"), name="layers")

# ---------------------------------------------------------
# 2. SHARDING ENDPOINT
# The Orchestrator calls this to make the Registry cut up the model.
# ---------------------------------------------------------
class ShardRequest(BaseModel):
    model_id: str

@app.post("/models/shard")
async def shard_model(req: ShardRequest):
    hf_token = settings.HF_TOKEN
    if not hf_token:
        raise HTTPException(500, "HF_TOKEN not set on Registry")

    try:
        # Calls the CPU-safe sharding logic in layer_storage.py
        num = store.shard_model(req.model_id, hf_token)
        return {"status": "success", "num_layers": num}
    except Exception as e:
        print(f"Sharding Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Used by Scheduler to know how many layers a model has"""
    sanitized = model_id.replace("/", "_")
    # Check if we have the structure file on disk
    path = f"/data/layers/{sanitized}/structure.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(404, "Model not sharded or found")

# ---------------------------------------------------------
# 3. WORKER DISCOVERY
# ---------------------------------------------------------
@app.post("/workers/register")
async def register_worker(data: dict):
    # Store worker metadata in Redis for 5 minutes
    await r.setex(f"worker_meta:{data['worker_id']}", 300, json.dumps(data))
    return {"status": "ok"}

@app.post("/workers/query")
async def query_workers(data: dict):
    # Simple lookup for now
    keys = await r.keys("worker_meta:*")
    workers = []
    for k in keys:
        w_data = await r.get(k)
        if w_data:
            workers.append(json.loads(w_data))
    return {"workers": workers}
```

---

## File: registry/requirements.txt

```plaintext
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
redis>=5.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
transformers>=4.36.0
huggingface-hub>=0.19.0
accelerate>=0.25.0
```

---

## File: scheduler/Dockerfile

```plaintext
FROM python:3.10-slim
WORKDIR /app

# Copy Requirements
COPY scheduler/requirements.txt .
RUN pip install -r requirements.txt

# Copy Shared Lib
COPY shared/ ./shared/
# Copy App
COPY api/ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## File: scheduler/main.py

```python
"""
ENHANCED SCHEDULER - Multi-Worker Coordination

Features:
- Queries registry for worker capabilities
- Splits jobs across multiple workers
- Coordinates layer execution
- Handles tensor passing between workers
"""

import asyncio
import json
import redis.asyncio as redis
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
from dataclasses import dataclass
from shared.config import settings
from shared.economics import calculate_worker_share
from shared.solana_lib import sign_payout

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

@dataclass
class WorkerInfo:
    pubkey: str
    ws: WebSocket
    specs: dict
    status: str  # IDLE, BUSY, OFFLINE, LOADING, READY
    assigned_layers: Optional[List[int]] = None
    loaded_model: Optional[str] = None  # NEW: Which model is loaded
    ready: bool = False  # NEW: Whether layers are loaded

@dataclass
class JobExecution:
    job_id: str
    model_id: str
    assigned_workers: List[str]  # worker pubkeys
    layer_splits: Dict[str, List[int]]  # worker_pubkey -> [layer_indices]
    results: Dict[str, any]  # worker_pubkey -> result
    status: str  # PENDING, IN_PROGRESS, COMPLETED, FAILED

class EnhancedScheduler:
    def __init__(self, registry_url: str = "http://registry:8002"):
        self.workers: Dict[str, WorkerInfo] = {}
        self.jobs: Dict[str, JobExecution] = {}
        self.registry_url = registry_url
        self.ready_workers: List[str] = []  # NEW: Track ready workers
        self.target_model: Optional[str] = None  # NEW: Model we're preparing for

    async def register(self, ws: WebSocket, specs: dict) -> str:
        """Register worker with scheduler"""
        wid = specs['pubkey']

        self.workers[wid] = WorkerInfo(
            pubkey=wid,
            ws=ws,
            specs=specs,
            status="IDLE"
        )

        print(f"✅ Worker {wid[:8]}.. registered: {specs.get('gpu', 'unknown')} "
              f"({specs.get('vram_gb', 0)}GB VRAM)")

        # Also register with registry
        await self._register_worker_with_registry(wid, specs)

        return wid

    async def _register_worker_with_registry(self, worker_id: str, specs: dict):
        """Inform registry about worker capabilities"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/workers/register",
                    json={
                        "worker_id": worker_id,
                        "vram_gb": specs.get('vram_gb', 0),
                        "gpu": specs.get('gpu', 'unknown'),
                        "models": specs.get('models', [])
                    }
                ) as resp:
                    if resp.status == 200:
                        print(f"  ✅ Worker registered with registry")
        except Exception as e:
            print(f"  ⚠️ Registry registration failed: {e}")

    async def disconnect(self, wid: str):
        """Handle worker disconnect"""
        if wid in self.workers:
            print(f"Worker {wid[:8]}.. disconnected")
            self.workers[wid].status = "OFFLINE"

            # Notify registry
            try:
                async with aiohttp.ClientSession() as session:
                    await session.delete(f"{self.registry_url}/workers/{wid}")
            except:
                pass

    async def query_available_workers(self, model_id: str, required_layers: int) -> List[Dict]:
        """Query registry for workers that can handle this model"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/workers/query",
                    json={
                        "model_id": model_id,
                        "required_layers": required_layers
                    }
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('workers', [])
                    else:
                        print(f"Registry query failed: {resp.status}")
                        return []
        except Exception as e:
            print(f"Registry query error: {e}")
            return []

    def split_layers_across_workers(self, total_layers: int, workers: List[Dict]) -> Dict[str, List[int]]:
        """Intelligently split layers across available workers"""
        if not workers:
            return {}

        # Sort workers by VRAM capacity
        sorted_workers = sorted(workers, key=lambda w: w.get('vram_gb', 0), reverse=True)

        # Calculate layer distribution based on VRAM
        total_vram = sum(w.get('vram_gb', 0) for w in sorted_workers)

        splits = {}
        current_layer = 0

        for worker in sorted_workers:
            vram_ratio = worker.get('vram_gb', 0) / total_vram
            num_layers = int(total_layers * vram_ratio)

            # Ensure at least 1 layer per worker
            if num_layers == 0:
                num_layers = 1

            # Don't exceed remaining layers
            num_layers = min(num_layers, total_layers - current_layer)

            if num_layers > 0:
                layer_range = list(range(current_layer, current_layer + num_layers))
                splits[worker['worker_id']] = layer_range
                current_layer += num_layers

            if current_layer >= total_layers:
                break

        # Assign remaining layers to last worker
        if current_layer < total_layers:
            last_worker = sorted_workers[-1]['worker_id']
            if last_worker in splits:
                splits[last_worker].extend(range(current_layer, total_layers))

        return splits

    async def dispatch_loop(self):
        """Main job dispatcher loop"""
        print("🚀 Scheduler Dispatch Loop Started")

        while True:
            try:
                # Pop job from Redis queue (blocking)
                task = await r.blpop("job_queue", timeout=1)
                if not task:
                    continue

                job = json.loads(task[1])
                job_id = job['id']

                print(f"\n📋 Processing Job {job_id}")
                print(f"  Model: {job['model']}")
                print(f"  Prompt: {job['input'][:50]}...")

                # Get model info from registry
                model_info = await self._get_model_info(job['model'])
                if not model_info:
                    print(f"  ❌ Model {job['model']} not found in registry")
                    await self._fail_job(job_id, "Model not available")
                    continue

                total_layers = model_info.get('num_layers', 0)
                print(f"  Layers: {total_layers}")

                # Query available workers
                available = await self.query_available_workers(job['model'], total_layers)

                if not available:
                    print(f"  ⚠️ No workers available, requeueing...")
                    await r.rpush("job_queue", json.dumps(job))
                    await asyncio.sleep(2)
                    continue

                print(f"  ✅ Found {len(available)} workers")

                # Split layers
                layer_splits = self.split_layers_across_workers(total_layers, available)

                print(f"  📊 Layer distribution:")
                for wid, layers in layer_splits.items():
                    print(f"    {wid[:8]}...: layers {layers[0]}-{layers[-1]} ({len(layers)} total)")

                # Create job execution tracker
                self.jobs[job_id] = JobExecution(
                    job_id=job_id,
                    model_id=job['model'],
                    assigned_workers=list(layer_splits.keys()),
                    layer_splits=layer_splits,
                    results={},
                    status="IN_PROGRESS"
                )

                # Dispatch to workers
                await self._dispatch_to_workers(job, layer_splits)

            except Exception as e:
                print(f"❌ Dispatch error: {e}")
                import traceback
                traceback.print_exc()

    async def preload_model_to_workers(self, model_id: str):
      """
      Called after model is sharded.
      Assigns layer ranges to workers and tells them to preload.
      """
      # Get model info
      model_info = await self._get_model_info(model_id)
      if not model_info:
          print(f"❌ Model {model_id} not found in registry")
          return False

      total_layers = model_info['num_layers']

      # Get available workers
      available = [w for w in self.workers.values() if w.status == "IDLE"]
      if len(available) < 2:
          print(f"⚠️ Need at least 2 workers, only {len(available)} available")
          return False

      # Split layers across workers
      workers_to_use = available[:2]
      layers_per_worker = total_layers // 2

      print(f"📋 Preloading {model_id} to {len(workers_to_use)} workers...")

      for i, worker in enumerate(workers_to_use):
          layer_start = i * layers_per_worker
          layer_end = (i + 1) * layers_per_worker - 1 if i == 0 else total_layers - 1
          layers = list(range(layer_start, layer_end + 1))

          worker.status = "LOADING"
          worker.assigned_layers = layers
          worker.loaded_model = model_id

          # Tell worker to preload
          await worker.ws.send_json({
              "type": "PRELOAD",
              "model_id": model_id,
              "layers": layers,
              "is_first": i == 0,
              "is_last": i == len(workers_to_use) - 1
          })

          print(f"  → {worker.pubkey[:8]}...: layers {layer_start}-{layer_end}")

      self.target_model = model_id
      return True

    async def handle_ready(self, worker_id: str, data: dict):
      """Worker announces it has loaded layers"""
      if worker_id not in self.workers:
          return

      worker = self.workers[worker_id]
      worker.status = "READY"
      worker.ready = True

      if worker_id not in self.ready_workers:
          self.ready_workers.append(worker_id)

      print(f"🚀 Worker {worker_id[:8]}... ready with {worker.loaded_model}")

    async def _get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information from registry"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.registry_url}/models/{model_id}/info") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except:
            pass
        return None

    async def _dispatch_to_workers(self, job: dict, layer_splits: Dict[str, List[int]]):
        """Dispatch job to assigned workers"""
        job_id = job['id']

        # Send to first worker (they'll coordinate)
        worker_order = list(layer_splits.keys())

        for i, wid in enumerate(worker_order):
            if wid not in self.workers:
                continue

            worker = self.workers[wid]

            try:
                await worker.ws.send_json({
                    "type": "EXECUTE",
                    "job": {
                        **job,
                        "assigned_layers": layer_splits[wid],
                        "is_first": i == 0,
                        "is_last": i == len(worker_order) - 1,
                        "next_worker": worker_order[i+1] if i < len(worker_order)-1 else None
                    }
                })

                worker.status = "BUSY"
                worker.assigned_layers = layer_splits[wid]

                print(f"  ✅ Dispatched to worker {wid[:8]}...")

            except Exception as e:
                print(f"  ❌ Failed to dispatch to {wid[:8]}: {e}")
                await self._fail_job(job_id, f"Worker dispatch failed: {e}")

    async def handle_result(self, wid: str, result_data: dict):
        """Handle result from worker"""
        job_id = result_data['job_id']

        if wid not in self.workers:
            return

        worker = self.workers[wid]
        worker.status = "IDLE"
        worker.assigned_layers = None

        print(f"  ✅ Received result from worker {wid[:8]}...")

        if job_id not in self.jobs:
            print(f"  ⚠️ Job {job_id} not found in tracker")
            return

        job_exec = self.jobs[job_id]
        job_exec.results[wid] = result_data

        # Check if all workers completed
        if len(job_exec.results) == len(job_exec.assigned_workers):
            print(f"  ✅ All workers completed for job {job_id}")

            # Combine results and store
            final_output = self._combine_worker_results(job_exec)

            await r.setex(
                f"result:{job_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    "job_id": job_id,
                    "status": "completed",
                    "output": final_output,
                    "workers": len(job_exec.assigned_workers)
                })
            )

            # Pay workers
            await self._pay_workers(job_exec, result_data.get('cost', 0))

            job_exec.status = "COMPLETED"
            print(f"  🎉 Job {job_id} completed successfully!")

    def _combine_worker_results(self, job_exec: JobExecution) -> str:
        """Combine results from multiple workers"""
        # Last worker has the final output
        worker_order = sorted(
            job_exec.layer_splits.keys(),
            key=lambda w: job_exec.layer_splits[w][-1]
        )

        last_worker = worker_order[-1]

        if last_worker in job_exec.results:
            return job_exec.results[last_worker].get('output', '')

        return "Error: No output from workers"

    async def _pay_workers(self, job_exec: JobExecution, total_cost: int):
        """Distribute payment to workers"""
        worker_share = calculate_worker_share(total_cost)
        per_worker = worker_share // len(job_exec.assigned_workers)

        for wid in job_exec.assigned_workers:
            # Credit worker balance
            unpaid = await r.incrby(f"worker_bal:{wid}", per_worker)

            print(f"  💰 Credited {wid[:8]}... with {per_worker} lamports (total: {unpaid})")

            # Check payout threshold
            from shared.economics import MIN_PAYOUT_THRESHOLD
            if unpaid >= MIN_PAYOUT_THRESHOLD:
                print(f"  💸 Triggering payout for {wid[:8]}...")
                sig = await sign_payout(wid, unpaid)

                if sig:
                    await r.set(f"worker_bal:{wid}", 0)

                    # Notify worker
                    if wid in self.workers and self.workers[wid].ws:
                        try:
                            await self.workers[wid].ws.send_json({
                                "type": "PAYMENT",
                                "amount": unpaid,
                                "sig": sig
                            })
                            print(f"    ✅ Payment sent: {sig[:16]}...")
                        except:
                            pass

    async def _fail_job(self, job_id: str, reason: str):
        """Mark job as failed"""
        await r.setex(
            f"result:{job_id}",
            3600,
            json.dumps({
                "job_id": job_id,
                "status": "failed",
                "error": reason
            })
        )

        if job_id in self.jobs:
            self.jobs[job_id].status = "FAILED"

# Initialize scheduler
scheduler = EnhancedScheduler()

@app.on_event("startup")
async def start():
    asyncio.create_task(scheduler.dispatch_loop())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None

    try:
        data = await ws.receive_json()

        if data['type'] == 'REGISTER':
            wid = await scheduler.register(ws, data['specs'])

        # Message loop
        while True:
            msg = await ws.receive_json()

            if msg['type'] == 'READY':  # NEW
                await scheduler.handle_ready(wid, msg)

            elif msg['type'] == 'RESULT':
                await scheduler.handle_result(wid, msg)

            elif msg['type'] == 'HEARTBEAT':
                await ws.send_json({"type": "ACK"})

    except WebSocketDisconnect:
        if wid:
            await scheduler.disconnect(wid)

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job execution status"""
    if job_id in scheduler.jobs:
        job_exec = scheduler.jobs[job_id]

        return {
            "job_id": job_id,
            "status": job_exec.status,
            "assigned_workers": [
                {
                    "worker_id": wid,
                    "layer_range": f"{layers[0]}-{layers[-1]}"
                }
                for wid, layers in job_exec.layer_splits.items()
            ],
            "completed_workers": len(job_exec.results)
        }

    # Check Redis for completed job
    result = await r.get(f"result:{job_id}")
    if result:
        return json.loads(result)

    return {"job_id": job_id, "status": "not_found"}

@app.get("/workers")
async def list_workers():
    """List connected workers"""
    return [
      {
        "pubkey": w.pubkey,
        "status": w.status,
        "gpu": w.specs.get('gpu', 'unknown'),
        "vram_gb": w.specs.get('vram_gb', 0),
        "assigned_layers": w.assigned_layers
      }
      for w in scheduler.workers.values()
    ]

@app.get("/workers/ready")
async def get_ready_status():
  """Check how many workers are ready"""
  return {
    "total": len(scheduler.workers),
    "ready": len(scheduler.ready_workers),
    "target_model": scheduler.target_model,
    "workers": [
      {
        "id": w.pubkey[:8] + "...",
        "status": w.status,
        "ready": w.ready,
        "model": w.loaded_model,
        "layers": f"{w.assigned_layers[0]}-{w.assigned_layers[-1]}" if w.assigned_layers else None
      }
      for w in scheduler.workers.values()
    ]
  }
```

---

## File: scheduler/requirements.txt

```plaintext
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
redis>=5.0.0
websockets>=12.0
solana>=0.30.2
solders>=0.21.0
aiohttp>=3.9.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
```

---

## File: shared/config.py

```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SOLANA_RPC_URL: str
    PLATFORM_WALLET_PUBKEY: str
    SCHEDULER_PRIVATE_KEY: str
    REDIS_HOST: str
    REDIS_PORT: int
    HF_TOKEN: str

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## File: shared/economics.py

```python
# PRICING CONSTANTS
LAMPORT_PER_SOL = 1_000_000_000
MIN_PAYOUT_THRESHOLD = 0.1 * LAMPORT_PER_SOL  # Worker gets paid after 0.1 SOL work

# Cost Calculation
# A "Token-Layer" is one token passing through one layer.
# Price: 2 Lamports per token-layer.
# Example: 70B Model (80 layers) * 100 Tokens * 2 = 16,000 Lamports ($0.002)
PRICE_PER_TOKEN_LAYER_LAMPORT = 2

def calculate_job_cost(num_layers: int, input_tokens: int, output_tokens: int) -> int:
    total_tokens = input_tokens + output_tokens
    return num_layers * total_tokens * PRICE_PER_TOKEN_LAYER_LAMPORT

def calculate_worker_share(job_cost: int) -> int:
    # Worker gets 80%, Platform keeps 20%
    return int(job_cost * 0.80)
```

---

## File: shared/solana_lib.py

```python
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.keypair import Keypair
from solders.system_program import transfer, TransferParams
from solana.rpc.async_api import AsyncClient
from solders.transaction import Transaction
from .config import settings

client = AsyncClient(settings.SOLANA_RPC_URL)
platform_pubkey = Pubkey.from_string(settings.PLATFORM_WALLET_PUBKEY)

# Load Scheduler Key
try:
    import json
    _key_bytes = json.loads(settings.SCHEDULER_PRIVATE_KEY)
    scheduler_keypair = Keypair.from_bytes(bytes(_key_bytes))
except:
    pass # Handle gracefully if env not set in build context

async def verify_deposit(tx_sig: str) -> int:
    """
    Checks on-chain if PLATFORM_WALLET received SOL in this transaction.
    Returns: Amount in Lamports (0 if invalid).
    """
    try:
        resp = await client.get_transaction(
            Signature.from_string(tx_sig),
            max_supported_transaction_version=0
        )

        if not resp.value: return 0

        meta = resp.value.transaction.meta
        if meta.err: return 0

        # Map accounts to find Platform Wallet index
        account_keys = resp.value.transaction.transaction.message.account_keys
        try:
            # Look through all accounts to find ours
            # Note: In newer transaction versions, logic handles lookup tables,
            # this handles standard transfers.
            idx = -1
            for i, key in enumerate(account_keys):
                if str(key) == str(platform_pubkey):
                    idx = i
                    break

            if idx == -1: return 0

            # Calc diff
            pre = meta.pre_balances[idx]
            post = meta.post_balances[idx]
            return max(0, post - pre)

        except Exception as e:
            print(f"Parsing error: {e}")
            return 0

    except Exception as e:
        print(f"RPC Error: {e}")
        return 0

async def sign_payout(worker_pubkey_str: str, lamports: int) -> str:
    """Sends SOL from Scheduler to Worker"""
    try:
        dest = Pubkey.from_string(worker_pubkey_str)

        # Use transfer function instead of Transfer class
        ix = transfer(TransferParams(
            from_pubkey=scheduler_keypair.pubkey(),
            to_pubkey=dest,
            lamports=lamports
        ))

        # Get latest blockhash
        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash

        # Create and sign transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            scheduler_keypair.pubkey(),
            [scheduler_keypair],
            blockhash
        )

        # Send transaction
        resp = await client.send_transaction(tx)
        return str(resp.value)
    except Exception as e:
        print(f"Payout failed: {e}")
        return None
```

---

