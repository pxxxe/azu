# Project: azu.cx

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
```

---

## File: api/requirements.txt

```plaintext
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
from solana.transaction import Transaction
from solders.system_program import Transfer, TransferParams
from solders.message import Message

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

        ix = Transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=PLATFORM_WALLET, lamports=lamports))
        blockhash = await client.get_latest_blockhash()
        msg = Message([ix], kp.pubkey())
        tx = Transaction([kp], msg, blockhash.value.blockhash)

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

## File: client/worker/main.py

```python
import asyncio
import json
import torch
import aiohttp
import os
from transformers import AutoModel, AutoConfig
from huggingface_hub import login
from solders.keypair import Keypair

# CONFIG
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
HF_TOKEN = os.getenv("HF_TOKEN")

class GPUWorker:
    def __init__(self, hf_token):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        self.loaded_models = {}

        # Load Solana Wallet
        # Assumes ~/.config/solana/id.json exists
        try:
            with open(os.path.expanduser("~/.config/solana/id.json")) as f:
                self.keypair = Keypair.from_bytes(bytes(json.load(f)))
        except:
            print("No wallet found, generating temp one")
            self.keypair = Keypair()

        login(token=hf_token)

    async def ensure_model(self, model_id):
        """Lazy loads model to VRAM"""
        if model_id not in self.loaded_models:
            print(f"⬇️ Downloading {model_id}...")
            # Using device_map="auto" for automatic offloading
            model = AutoModel.from_pretrained(
                model_id,
                token=self.hf_token,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.loaded_models[model_id] = model
            print(f"✅ Loaded {model_id}")
        return self.loaded_models[model_id]

    def execute_layer(self, model, input_data, layer_idx=None):
        # SIMPLIFIED: Running full forward pass for demo
        # In prod: Slicing logic applies here
        with torch.no_grad():
            # Mock input tensor creation from input_data string
            # Real app: Deserialize tensor bytes
            inputs = torch.randn(1, 10, 4096).to(self.device).half()

            # Run
            if hasattr(model, 'layers'):
                # Run specific layer
                out = model.layers[layer_idx](inputs)[0]
            else:
                # Fallback for generic models
                out = model(inputs)[0]

            return out

    async def start(self):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(SCHEDULER_URL) as ws:

                # 1. Register
                print(f"🔐 Authenticating as {self.keypair.pubkey()}")
                await ws.send_json({
                    "type": "REGISTER",
                    "specs": {
                        "pubkey": str(self.keypair.pubkey()),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                    }
                })

                # 2. Work Loop
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)

                        if data['type'] == 'EXECUTE':
                            job = data['job']
                            print(f"⚡ Executing Job {job['id']} ({job['model']})")

                            model = await self.ensure_model(job['model'])

                            # Run Compute
                            # (Layer 0 assumed for demo)
                            result = self.execute_layer(model, job['input'], 0)

                            # Reply
                            await ws.send_json({
                                "type": "RESULT",
                                "job_id": job['id'],
                                "cost": job['cost'], # Echo cost for payment calculation
                                "status": "success"
                            })

                        elif data['type'] == 'PAYMENT':
                            print(f"💰 PAID! {data['amount']} Lamports. Tx: {data['sig']}")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Set HF_TOKEN env var")
        exit(1)
    worker = GPUWorker(HF_TOKEN)
    asyncio.run(worker.start())
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

## File: docker-compose.yml

```yaml
version: '3.8'

services:
  redis:
    image: redis:alpine
    ports: ["6379:6379"]

  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [redis]

  scheduler:
    build:
      context: .
      dockerfile: scheduler/Dockerfile
    ports: ["8001:8001"]
    env_file: .env
    depends_on: [redis]
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
import asyncio
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from shared.config import settings
from shared.economics import calculate_worker_share
from shared.solana_lib import sign_payout

app = FastAPI()
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

class Scheduler:
    def __init__(self):
        # { worker_id: {ws: WebSocket, specs: dict} }
        self.workers = {}

    async def register(self, ws, specs):
        wid = specs['pubkey']
        self.workers[wid] = {"ws": ws, "specs": specs, "status": "IDLE"}
        print(f"Worker {wid[:8]}.. registered with {specs['gpu']}")
        return wid

    async def disconnect(self, wid):
        if wid in self.workers:
            del self.workers[wid]

    async def dispatch_loop(self):
        print("Scheduler Dispatch Loop Started")
        while True:
            # Pop job from Redis (Blocking)
            task = await r.blpop("job_queue", timeout=1)
            if not task: continue

            job = json.loads(task[1])
            print(f"Processing Job {job['id']}")

            # Find Idle Worker
            # (V2: Match model/vram requirements)
            assigned_wid = None
            for wid, w_data in self.workers.items():
                if w_data["status"] == "IDLE":
                    assigned_wid = wid
                    break

            if assigned_wid:
                try:
                    await self.workers[assigned_wid]['ws'].send_json({
                        "type": "EXECUTE",
                        "job": job
                    })
                    self.workers[assigned_wid]['status'] = "BUSY"
                except:
                    # Retry logic would go here
                    await r.lpush("job_queue", json.dumps(job))
            else:
                # No workers, push back
                await r.rpush("job_queue", json.dumps(job))
                await asyncio.sleep(1)

    async def handle_result(self, wid, result_data):
        self.workers[wid]['status'] = "IDLE"

        # Calculate Payment
        job_cost = result_data.get('cost', 0)
        worker_pay = calculate_worker_share(job_cost)

        # Credit Worker in Redis
        unpaid = await r.incrby(f"worker_bal:{wid}", worker_pay)

        # Check Payout Threshold
        from shared.economics import MIN_PAYOUT_THRESHOLD
        if unpaid >= MIN_PAYOUT_THRESHOLD:
            print(f"Triggering Payout for {wid}")
            sig = await sign_payout(wid, unpaid)
            if sig:
                await r.set(f"worker_bal:{wid}", 0)
                await self.workers[wid]['ws'].send_json({
                    "type": "PAYMENT",
                    "amount": unpaid,
                    "sig": sig
                })

coord = Scheduler()

@app.on_event("startup")
async def start():
    asyncio.create_task(coord.dispatch_loop())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        # 1. Auth Handshake (Simplified)
        # In prod: Challenge/Response signature verification here
        data = await ws.receive_json()
        if data['type'] == 'REGISTER':
            wid = await coord.register(ws, data['specs'])

        # 2. Loop
        while True:
            msg = await ws.receive_json()
            if msg['type'] == 'RESULT':
                await coord.handle_result(wid, msg)

    except WebSocketDisconnect:
        await coord.disconnect(wid)
```

---

## File: scheduler/requirements.txt

```plaintext
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
from solders.system_program import Transfer, TransferParams
from solders.message import Message
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
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
        ix = Transfer(TransferParams(
            from_pubkey=scheduler_keypair.pubkey(),
            to_pubkey=dest,
            lamports=lamports
        ))

        blockhash = await client.get_latest_blockhash()
        msg = Message([ix], scheduler_keypair.pubkey())
        tx = Transaction([scheduler_keypair], msg, blockhash.value.blockhash)

        resp = await client.send_transaction(tx)
        return str(resp.value)
    except Exception as e:
        print(f"Payout failed: {e}")
        return None
```

---

