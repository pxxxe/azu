#!/usr/bin/env python3
"""
E2E Test for AZU.CX with FULL MoE Support & Auto-Teardown
Tests MoE and Dense model architecture on secure cloud GPUs

This replaces Solana with Hyperliquid for payments.
"""

import runpod
import requests
import time
import json
import sys
import os
import asyncio
import traceback
import secrets

# === HYPERLIQUID IMPORTS ===
from eth_account import Account
import aiohttp

# ==========================================
# CONFIGURATION
# ==========================================

CORE_IMG = 'pxxxe/azu-core:latest'
WORKER_IMG = 'pxxxe/azu-worker:latest'
# TEST_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

VOLUME_ID = os.getenv("VOLUME_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Hyperliquid configuration
HYPERLIQUID_RPC = os.getenv("HYPERLIQUID_RPC", "https://rpc.hyperliquid-testnet.xyz/evm")

# Prioritize High-RAM GPUs for Stability
GPU_TYPES_SECURE = [
    "NVIDIA GeForce RTX 4090",          # 24GB
    "NVIDIA H200 NVL",                  # 143GB
    "NVIDIA H100 NVL",                  # 94GB
    "NVIDIA RTX PRO 6000",              # 96GB
    "NVIDIA RTX 6000 Ada Generation",   # 48GB
    "NVIDIA A100 80GB PCIe",            # 80GB
    "NVIDIA A100-SXM4-80GB",            # 80GB
    "NVIDIA H100 80GB HBM3",            # 80GB (H100 SXM)
    "NVIDIA H100 PCIe",                 # 80GB
    "NVIDIA RTX A6000",                 # 48GB
    "NVIDIA L40S",                      # 48GB
    "NVIDIA L40",                       # 48GB
    "NVIDIA RTX A5000",                 # 24GB
    "NVIDIA GeForce RTX 3090",          # 24GB
    "NVIDIA L4",                        # 24GB
    "NVIDIA RTX PRO 4500",              # 32GB
    "NVIDIA RTX A4500",                 # 20GB
    "NVIDIA RTX 4000 Ada Generation",   # 20GB
    "NVIDIA RTX A4000",                 # 16GB
    "NVIDIA RTX 2000 Ada Generation",   # 16GB
]

# GPU tier codes for GraphQL API (used to create Load Balancer endpoints)
# These are the tier codes that the RunPod GraphQL API accepts
GPU_TIERS_GRAPHQL = [
    "ADA_24",      # L4, RTX 4000 Ada, RTX 4090
    "AMPERE_24",   # RTX 3090, A5000
    "ADA_48_PRO",  # L40, L40S, RTX 6000 Ada
    "AMPERE_48",   # A40, RTX A6000
    "AMPERE_80",   # A100
    "ADA_80_PRO",  # H100, H200
    "AMPERE_16",   # A4000, RTX 3080
]

# VRAM sizes in GB (approximate)
GPU_VRAM_MAP = {
    "NVIDIA H200 NVL": 143,
    "NVIDIA H100 NVL": 94,
    "NVIDIA RTX PRO 6000": 96,
    "NVIDIA RTX 6000 Ada Generation": 48,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 PCIe": 80,
    "NVIDIA RTX A6000": 48,
    "NVIDIA L40S": 48,
    "NVIDIA L40": 48,
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA RTX A5000": 24,
    "NVIDIA GeForce RTX 3090": 24,
    "NVIDIA L4": 24,
    "NVIDIA RTX PRO 4500": 32,
    "NVIDIA RTX A4500": 20,
    "NVIDIA RTX 4000 Ada Generation": 20,
    "NVIDIA RTX A4000": 16,
    "NVIDIA RTX 2000 Ada Generation": 16,
}

# TARGET_TOTAL_VRAM = 120  # GB needed for Mixtral test
TARGET_TOTAL_VRAM = 24  # GB needed for Mixtral test


runpod.api_key = RUNPOD_API_KEY
RUNPOD_REST_URL = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def log_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def load_funded_account():
    """Load your funded Hyperliquid account."""
    # Try to load from FUNDED_ACCOUNT_KEY (base58 or JSON array format)
    if os.getenv("FUNDED_ACCOUNT_KEY"):
        secret = os.getenv("FUNDED_ACCOUNT_KEY", "")
        try:
            # Try JSON array format first
            if secret.startswith('['):
                key_bytes = bytes(json.loads(secret))
                return Account.from_key(key_bytes)
            # Try hex format
            elif secret.startswith('0x'):
                return Account.from_key(secret)
            # Try base58 (Solana style - not applicable for Hyperliquid but keep for compatibility)
            else:
                return Account.from_key(secret)
        except Exception as e:
            print(f"   ⚠️ Failed to parse FUNDED_ACCOUNT_KEY: {e}")

    # Try to load from funded_account.json
    if os.path.exists("funded_account.json"):
        with open("funded_account.json") as f:
            data = json.load(f)
            if isinstance(data, list):
                return Account.from_key(bytes(data))
            elif isinstance(data, str):
                return Account.from_key(data)

    # Try to load from ~/.config/hyperliquid/id.json
    hl_config = os.path.expanduser("~/.config/hyperliquid/id.json")
    if os.path.exists(hl_config):
        with open(hl_config) as f:
            return Account.from_key(f.read().strip())

    print("❌ ERROR: No funded Hyperliquid account found!")
    print("   Make sure you have FUNDED_ACCOUNT_KEY set or ~/.config/hyperliquid/id.json")
    sys.exit(1)


def resolve_connection(pod_id, port, max_wait=120):
    """
    Robustly resolves connection URL.
    On Secure Cloud, direct IPs are rare/delayed. We prioritize the Proxy URL.
    """
    print(f"   ⏳ Resolving connection for {pod_id} (port {port})...")

    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net"

    for i in range(0, max_wait, 5):
        try:
            pod = runpod.get_pod(pod_id)

            # --- CRITICAL FIX: Safe Access ---
            runtime = pod.get('runtime')
            if not runtime:
                if i % 10 == 0: print(f"      [{i}s] Waiting for runtime info...")
                time.sleep(5)
                continue
            # ---------------------------------

            ports = runtime.get('ports', [])
            is_port_open = any(p.get('privatePort') == port for p in ports)

            if is_port_open:
                print(f"      ✅ Port open. Using Proxy: {proxy_url}")
                return proxy_url

        except Exception as e:
            print(f"      ⚠️ API Polling Error: {e}")

        time.sleep(5)

    print("      ⚠️ Timeout waiting for port check. Assuming Proxy is valid.")
    return proxy_url


def deploy_pod_with_retry(name, image, env_vars, gpu_type, max_retries=1):
    """Deploy pod with retries."""
    print(f"   🎯 Trying GPU type: {gpu_type}")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   🔄 Attempt {attempt}/{max_retries}...")

            response = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu_type,
                cloud_type="SECURE",
                env=env_vars,
                ports="8000/http,8001/http,8002/http,8003/http",
                network_volume_id=VOLUME_ID,
                volume_mount_path="/data"
            )

            if isinstance(response, dict) and response.get('id'):
                pod_id = response['id']
                print(f"   ✅ Pod created: {pod_id}")
                return pod_id, gpu_type

        except Exception as e:
            err_str = str(e).lower()
            if "unavailable" in err_str or "specifications" in err_str:
                 print(f"   ⚠️  {gpu_type} unavailable.")
            else:
                 print(f"   ⚠️  Error: {e}")

        if attempt < max_retries:
            time.sleep(1)

    return None, None


def deploy_with_fallback(name, image, env_vars):
    """Try deploying across multiple GPU types."""
    for gpu_type in GPU_TYPES_SECURE:
        pod_id, gpu = deploy_pod_with_retry(name, image, env_vars, gpu_type)
        if pod_id:
            return pod_id, gpu
    raise Exception(f"❌ Could not deploy {name} on any GPU type")



def runpod_rest(method: str, path: str, body: dict = None) -> dict:
    """
    Call the RunPod REST API.
    Auth: Authorization: Bearer KEY (unlike GraphQL which uses ?api_key=).
    Docs: https://docs.runpod.io/api-reference
    """
    resp = requests.request(
        method,
        f"{RUNPOD_REST_URL}{path}",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def runpod_graphql(query: str, variables: dict = None) -> dict:
    """
    Call the RunPod GraphQL API.
    Auth: api_key query param (unlike REST which uses Bearer header).
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = requests.post(
        f"{RUNPOD_GRAPHQL_URL}?api_key={RUNPOD_API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        raise Exception(f"GraphQL error: {data['errors']}")

    return data.get("data", {})


def create_serverless_endpoint(name: str, image: str, env_vars: dict):
    """
    Create a RunPod *Load Balancing* Serverless endpoint for the azu-worker.

    Why load balancing, not queue-based:
      The azu-worker runs its own aiohttp HTTP server on port 8003.  It does
      NOT implement the RunPod queue handler pattern (runpod.serverless.start).
      Queue-based endpoints expect that pattern and will kill workers that
      don't speak it.  Load balancing endpoints forward HTTP directly to the
      worker's server — matching azu's architecture exactly.

    IMPORTANT: The REST API does NOT support creating Load Balancer endpoints.
    The endpointType: "LOAD_BALANCER" field doesn't exist in the REST schema.
    We MUST use the GraphQL saveEndpoint mutation with type: "LB" instead.

    GraphQL also requires GPU tier codes (ADA_24, AMPERE_80, etc.) instead of
    full GPU names ("NVIDIA GeForce RTX 4090").

    How the azu scheduler finds the worker:
      On startup the worker resolves P2P_URL_TEMPLATE using RUNPOD_POD_ID
      (injected by RunPod into every container, including serverless workers),
      then POSTs that URL to the azu scheduler's POST /workers endpoint.
      The scheduler subsequently talks to the worker via long-poll (GET /worker/poll)
      instead of pushing to a proxy URL (which returns 403 for serverless containers).

    Health check:
      RunPod's load balancer polls GET /ping on PORT_HEALTH.  The azu-worker
      must expose this route (returning 200 when ready, 204 while starting).
      See packages/azu-worker/src/azu/worker/main.py.

    Returns:
        (endpoint_id, template_id, gpu_tier_code)
    """
    # 1. Create a serverless template (SDK handles auth correctly).
    print(f"   📋 Creating serverless template for {name}...")
    new_template = runpod.create_template(
        name=name,
        image_name=image,
        is_serverless=True,
        container_disk_in_gb=10,
        env=env_vars,          # dict {key: value}  -- SDK requires dict
    )
    template_id = new_template["id"]
    print(f"   ✅ Template created: {template_id}")

    # 2. Create a Load Balancing endpoint via GraphQL.
    #    REST API cannot create LB endpoints - it doesn't have the type field.
    #    GraphQL saveEndpoint with type: "LB" is the only way to create LB endpoints.
    #    GraphQL uses tier codes (ADA_24, AMPERE_80, etc.) not full GPU names.
    print(f"   🔎 Creating load balancing endpoint for {name} via GraphQL...")
    for gpu_tier in GPU_TIERS_GRAPHQL:
        try:
            # GraphQL mutation to create a Load Balancer endpoint
            mutation = """
            mutation CreateEndpoint($input: EndpointInput!) {
                saveEndpoint(input: $input) {
                    id
                    type
                    gpuIds
                }
            }
            """

            variables = {
                "input": {
                    "name": name,
                    "templateId": template_id,
                    "gpuIds": gpu_tier,
                    "type": "LB",
                    "gpuCount": 1,
                    "workersMin": 1,
                    "workersMax": 1,
                    "idleTimeout": 60,
                    "scalerType": "QUEUE_DELAY",
                    "scalerValue": 4
                }
            }

            result = runpod_graphql(mutation, variables)
            endpoint_data = result.get("saveEndpoint", {})
            endpoint_id = endpoint_data.get("id")
            ep_type = endpoint_data.get("type")

            if not endpoint_id:
                raise Exception(f"No endpoint ID returned: {endpoint_data}")

            if ep_type != "LB":
                raise Exception(f"Expected type=LB, got type={ep_type}")

            print(f"   ✅ Endpoint created: {endpoint_id} ({gpu_tier}, type={ep_type})")
            return endpoint_id, template_id, gpu_tier

        except Exception as e:
            print(f"   ⚠️  {gpu_tier}: {e}")

    raise Exception(f"❌ Could not create load balancing endpoint {name} on any GPU")


def delete_serverless_endpoint(endpoint_id: str, template_id: str, gpu_tier: str):
    """
    Delete a RunPod Serverless endpoint and its backing template.

    Scale to 0 first (RunPod requires workersMin=workersMax=0 before delete),
    then delete the endpoint, then delete the template.
    """
    # Scale to zero first
    try:
        runpod_rest("PATCH", f"/endpoints/{endpoint_id}", {
            "workersMin": 0,
            "workersMax": 0,
        })
        print(f"   ✅ Endpoint scaled to 0 workers")
    except Exception as e:
        print(f"   ⚠️ Could not scale to 0 (continuing): {e}")

    try:
        runpod_rest("DELETE", f"/endpoints/{endpoint_id}")
        print(f"   ✅ Endpoint deleted ({endpoint_id})")
    except Exception as e:
        print(f"   ⚠️ Failed to delete endpoint {endpoint_id}: {e}")

    try:
        runpod_rest("DELETE", f"/templates/{template_id}")
        print(f"   ✅ Template deleted ({template_id})")
    except Exception as e:
        print(f"   ⚠️ Failed to delete template {template_id}: {e}")


async def transfer_hyperliquid(from_account, to_address, amount_eth, rpc_url):
    """
    Transfer native token (ETH/HYPE) on Hyperliquid.

    Note: Hyperliquid is primarily a perpetuals exchange. For native token
    transfers, we use the standard Ethereum-style transfer via RPC.
    """
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    # Get nonce
    nonce = w3.eth.get_transaction_count(from_account.address)

    # Get gas price
    gas_price = w3.eth.gas_price

    # Build transaction
    tx = {
        'nonce': nonce,
        'gasPrice': gas_price,
        'gas': 21000,  # Standard gas for native transfer
        'to': to_address,
        'value': int(amount_eth * 1e18),  # Convert ETH to wei
        'chainId': 998,  # Hyperliquid chain ID
    }

    # Sign and send
    signed_tx = from_account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"      📤 Transfer sent: {tx_hash.hex()}")

    # Wait for confirmation
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    if receipt['status'] == 1:
        print(f"      ✅ Transfer confirmed!")
    else:
        print(f"      ❌ Transfer failed!")

    return tx_hash.hex()


async def get_hyperliquid_balance(address, rpc_url):
    """Get native balance from Hyperliquid/Ethereum chain."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    balance = w3.eth.get_balance(address)
    return balance / 1e18  # Convert to ETH/HYPE


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    log_section("🚀 AZU.CX - Mixtral MoE Test (Hyperliquid)")

    # TRACKING VARIABLES FOR TEARDOWN
    core_pod_id = None
    worker_ids = []
    worker_gpus = []
    sl_endpoint_id = None   # RunPod Serverless LB endpoint (section 6)
    sl_template_id = None   # backing template
    sl_gpu_tier = None     # GPU tier code used (needed for teardown)

    try:
        # ==========================================
        # 0. Setup Hyperliquid Wallets
        # ==========================================
        log_section("💰 0. Setting up Hyperliquid Wallets")

        # Load funded account
        funder = load_funded_account()

        # Generate new wallets for the test
        platform = Account.create()
        scheduler = Account.create()

        print(f"   👤 Funder: {funder.address}")
        print(f"   💵 Platform: {platform.address}")
        print(f"   📋 Scheduler: {scheduler.address}")

        # Generate shared secret for interworker auth
        interworker_secret = secrets.token_hex(32)
        print(f"   🔑 Auth secret: {interworker_secret[:8]}...")

        # Check funder balance (async)
        async def check_funder_balance():
            async with aiohttp.ClientSession() as session:
                from web3 import Web3
                w3 = Web3(Web3.HTTPProvider(HYPERLIQUID_RPC))
                try:
                    bal = w3.eth.get_balance(funder.address)
                    bal_eth = bal / 1e18
                    print(f"   💵 Funder balance: {bal_eth:.4f} ETH/HYPE")
                    if bal_eth < 0.01:
                        raise Exception("Insufficient funds - need ETH/HYPE on Hyperliquid testnet")
                except Exception as e:
                    print(f"   ⚠️ Could not check balance: {e}")
                    print(f"   ℹ️ Assuming testnet faucet has been used...")

        asyncio.run(check_funder_balance())

        # ==========================================
        # 1. Deploy Core
        # ==========================================
        log_section("🚀 1. Deploying Core")

        core_env = {
            'HF_TOKEN': HF_TOKEN,
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            # Payment configuration
            'PAYMENT_PROVIDER': 'hyperliquid',
            'HYPERLIQUID_RPC_URL': HYPERLIQUID_RPC,
            'HYPERLIQUID_ADDRESS': platform.address,
            'SCHEDULER_PRIVATE_KEY': '0x' + scheduler.key.hex(),
            # Legacy/fallback
            'PLATFORM_WALLET_PUBKEY': platform.address,
            'PUBLIC_KEY': 'null',
            'HF_HOME': '/data/hf_cache',
            # Interworker auth
            'AUTH_SECRET_KEY': interworker_secret,
        }

        core_pod_id, _ = deploy_with_fallback("azu-core-moe", CORE_IMG, core_env)

        # Resolve URLs
        api_url = resolve_connection(core_pod_id, 8000)
        reg_url = resolve_connection(core_pod_id, 8002)
        sched_url = resolve_connection(core_pod_id, 8001)  # HTTP URL

        # Convert HTTP->WS for scheduler
        ws_scheme = "wss" if "https" in sched_url else "ws"
        sched_ws_url = sched_url.replace("https://", "").replace("http://", "")
        sched_ws_full = f"{ws_scheme}://{sched_ws_url}/ws/worker"

        print(f"   ✅ API: {api_url}")
        print(f"   ✅ Registry: {reg_url}")
        print(f"   ✅ Scheduler: {sched_ws_full}")

        # Wait for Core Health
        print("   ⏳ Waiting for Registry health check...")
        for i in range(20):
            try:
                if requests.get(f"{reg_url}/docs", timeout=5).status_code == 200:
                    print("      ✅ Registry Healthy")
                    break
            except: time.sleep(3)
        else:
            raise Exception("Registry failed to become healthy")

        # ==========================================
        # 2. Shard Model (CHECK FIRST)
        # ==========================================
        log_section(f"⚡ 2. Checking Model Status")

        status_res = requests.get(f"{reg_url}/models/status", params={"model_id": TEST_MODEL}, timeout=10)
        if status_res.status_code == 200:
            status = status_res.json().get('status')
            print(f"   📊 Model status: {status}")

            if status == 'ready':
                print(f"   ✅ Model already sharded, skipping...")
            elif status == 'processing':
                print(f"   ⏳ Sharding already in progress, waiting...")
                # Wait for it to finish
                for i in range(180):
                    time.sleep(5)
                    check = requests.get(f"{reg_url}/models/status", params={"model_id": TEST_MODEL}, timeout=5)
                    if check.status_code == 200 and check.json().get('status') == 'ready':
                        print(f"   ✅ Sharding complete!")
                        break
            else:
                print(f"   🔪 Triggering shard...")
                shard_res = requests.post(f"{reg_url}/models/shard", json={"model_id": TEST_MODEL}, timeout=900)
                if shard_res.status_code != 200:
                    raise Exception(f"Sharding Failed: {shard_res.text}")
                print(f"   ✅ Sharding Complete: {shard_res.json()}")
        else:
            print(f"   ⚠️ Could not get status, will attempt to shard...")
            shard_res = requests.post(f"{reg_url}/models/shard", json={"model_id": TEST_MODEL}, timeout=900)

        # ==========================================
        # 3. Deploy Workers (DYNAMIC PROVISIONING)
        # ==========================================
        log_section(f"🚀 3. Deploying Workers (Target: {TARGET_TOTAL_VRAM}GB VRAM)")

        worker_env = {
            'SCHEDULER_URL': sched_ws_full,
            'REGISTRY_URL': reg_url,
            'HF_TOKEN': HF_TOKEN,
            'PUBLIC_KEY': 'null',
            # Worker payment config
            'PAYMENT_PROVIDER': 'hyperliquid',
            # DECOUPLED: We pass the RunPod-specific URL template here via environment
            'P2P_URL_TEMPLATE': 'https://{RUNPOD_POD_ID}-8003.proxy.runpod.net',
            'LAYER_CACHE_DIR': '/data/layers',
            # Interworker auth
            'AUTH_SECRET_KEY': interworker_secret,
        }

        total_vram = 0
        worker_count = 0

        while total_vram < TARGET_TOTAL_VRAM:
            wid, gpu_type = deploy_with_fallback(f"azu-worker-{worker_count}", WORKER_IMG, worker_env)
            worker_ids.append(wid)
            worker_gpus.append(gpu_type)

            gpu_vram = GPU_VRAM_MAP.get(gpu_type, 24)  # Default to 24GB if unknown
            total_vram += gpu_vram
            worker_count += 1

            print(f"   📊 Worker {worker_count}: {gpu_type} ({gpu_vram}GB)")
            print(f"   📊 Total VRAM: {total_vram}GB / {TARGET_TOTAL_VRAM}GB")

            if total_vram >= TARGET_TOTAL_VRAM:
                print(f"\n   ✅ Target VRAM reached! Deployed {worker_count} workers with {total_vram}GB total")
                break

        print("\n   ⏳ Waiting for workers to connect to Scheduler...")
        time.sleep(10)

        # ==========================================
        # 4. Deposit & Inference (raw HTTP — baseline)
        # ==========================================
        log_section("🧪 4. Running Inference")

        # Deposit (using Hyperliquid transfer)
        # funder IS the user — they send the tx and submit jobs under the same address
        async def do_deposit():
            print("   💳 Sending Deposit (Hyperliquid)...")

            amount = 0.01  # Small amount for testing

            try:
                tx_hash = await transfer_hyperliquid(
                    funder,
                    platform.address,
                    amount,
                    HYPERLIQUID_RPC
                )

                # Wait for confirmation
                await asyncio.sleep(10)

                # Notify API — sender and user are both funder
                requests.post(f"{api_url}/deposit", json={
                    "tx_sig": tx_hash,
                    "user_pubkey": funder.address  # FIX: was user.address, but funder sent the tx
                })
                print("   ✅ Deposit Registered")

            except Exception as e:
                print(f"   ⚠️ Deposit failed: {e}")
                print(f"   ℹ️ Simulating deposit for testing...")
                requests.post(f"{api_url}/deposit", json={
                    "tx_sig": "0x" + "00" * 32,
                    "user_pubkey": funder.address
                })
                print("   ✅ Simulated Deposit Registered")

        asyncio.run(do_deposit())

        # Submit Job
        print(f"   🧠 Submitting Prompt...")
        sub_res = requests.post(f"{api_url}/submit", json={
            "user_pubkey": funder.address,  # FIX: was user.address, must match deposit
            "model_id": TEST_MODEL,
            "prompt": "What is the capital of France?",
            "est_tokens": 50
        })

        if sub_res.status_code != 200:
            raise Exception(f"Submit failed: {sub_res.text}")

        job_id = sub_res.json()['job_id']
        print(f"   ✅ Job ID: {job_id}")

        # Poll
        for i in range(200):
            try:
                res = requests.get(f"{api_url}/results/{job_id}").json()
                status = res.get('status')
                if status == 'completed':
                    print(f"\n🎉 RESULT: {res['output']}\n")
                    break
                if status == 'failed':
                    print(f"\n❌ FAILED: {res.get('error')}\n")
                    break
                if i%5==0: print(f"      Status: {status}...")
            except: pass
            time.sleep(5)

        # ==========================================
        # 5. OpenAI Compat Test (ai_sdk)
        # ==========================================
        log_section("🤖 5. OpenAI Compat Test (ai_sdk)")

        try:
            from ai_sdk import openai as azu_openai, generate_text, stream_text

            # ai_sdk's openai() reads OPENAI_BASE_URL and OPENAI_API_KEY from
            # the environment via the underlying openai Python client.
            # The wallet address doubles as the API key.
            os.environ["OPENAI_BASE_URL"] = f"{api_url}/v1"
            os.environ["OPENAI_API_KEY"] = funder.address

            azu_model = azu_openai(TEST_MODEL)

            # --- Non-streaming ---
            print("   🧠 generate_text (non-streaming)...")
            res = generate_text(
                model=azu_model,
                prompt="What is the capital of France? Answer in one word.",
                max_tokens=32,
            )
            print(f"   ✅ Response : {res.text}")
            print(f"   📊 Usage   : {res.usage}")

            assert res.text, "generate_text returned empty response"

            # --- Streaming ---
            # stream_text.text_stream is an async iterator
            # print("\n   🌊 stream_text (streaming)...")

            # async def _run_stream():
            #     stream_res = stream_text(
            #         model=azu_model,
            #         prompt="Count from 1 to 5, one number per line.",
            #         max_tokens=64,
            #     )
            #     chunks = []
            #     async for chunk in stream_res.text_stream:
            #         chunks.append(chunk)
            #         print(chunk, end="", flush=True)
            #     print()
            #     return chunks

            # chunks = asyncio.run(_run_stream())
            # assert chunks, "stream_text yielded no chunks"
            # print("   ✅ Streaming OK")

        except ImportError:
            print("   ⚠️  ai_sdk not installed — skipping (uv add ai-sdk-python)")
        except Exception as e:
            print(f"   ❌ ai_sdk test failed: {e}")
            traceback.print_exc()

        # ==========================================
        # 6. Serverless Worker Test
        # ==========================================
        log_section("🤖 6. Serverless Worker Dispatch Test")

        # Terminate persistent workers for isolation — if they're still alive
        # the scheduler may plan the job on them and never exercise the
        # serverless HTTP dispatch path.
        print("   🔌 Terminating persistent workers for isolation...")
        for wid in list(worker_ids):
            try:
                runpod.terminate_pod(wid)
                print(f"   ✅ Persistent worker terminated ({wid})")
                worker_ids.remove(wid)
            except Exception as e:
                print(f"   ⚠️ Failed to terminate {wid}: {e}")

        # Give the scheduler time to detect the disconnects and drop sessions
        print("   ⏳ Waiting for scheduler to drop persistent sessions (15s)...")
        time.sleep(15)

        # Deploy one serverless worker — same image, different mode
        # NOTE: We now pass WAKE_URL - the RunPod LB endpoint URL that the
        # scheduler can use to wake this worker when it's scaled to 0.
        # The worker reports this URL to the scheduler on registration.
        sl_worker_env = {
            'WORKER_MODE':        'serverless',
            'SCHEDULER_HTTP_URL': sched_url,   # HTTP, NOT the ws:// URL
            'REGISTRY_URL':       reg_url,
            'HF_TOKEN':           HF_TOKEN,
            'PUBLIC_KEY':         'null',
            'PAYMENT_PROVIDER':   'hyperliquid',
            # Worker resolves its own public URL via this template
            'P2P_URL_TEMPLATE':   'https://{RUNPOD_POD_ID}-8003.proxy.runpod.net',
            'LAYER_CACHE_DIR':    '/data/layers',
            'AUTH_SECRET_KEY':    interworker_secret,
            # Load balancer health check — must match the aiohttp server port.
            # The azu-worker exposes GET /ping on port 8003 (see worker main.py).
            'PORT':              '8003',
            'PORT_HEALTH':       '8003',
            # WAKE_URL will be set after we get the endpoint ID
            # IDLE_TIMEOUT enables self-termination after idle (scale-to-zero)
            'IDLE_TIMEOUT':      '300',
        }

        # Deploy as a RunPod Load Balancing endpoint — NOT a queue-based endpoint
        # and NOT a pod.  The azu-worker runs its own aiohttp server on port 8003
        # and does not implement RunPod's queue handler pattern.
        # workersMin=1 keeps one worker alive so it can register with the azu
        # scheduler on startup and stay ready to receive HTTP dispatch.
        # RunPod injects RUNPOD_POD_ID into every serverless container, so
        # P2P_URL_TEMPLATE resolves identically to a pod deployment.
        print("   🚀 Creating RunPod Load Balancing endpoint for worker...")
        sl_endpoint_id, sl_template_id, sl_gpu_tier = create_serverless_endpoint(
            "azu-worker-sl", WORKER_IMG, sl_worker_env
        )
        print(f"   ✅ Endpoint ready: {sl_endpoint_id} ({sl_gpu_tier})")

        # Now set WAKE_URL to the RunPod LB endpoint URL format
        # This is what the scheduler uses to wake the worker when it's scaled to 0
        sl_worker_env['WAKE_URL'] = f'https://api.runpod.ai/v2/{sl_endpoint_id}/run'

        # Note: We can't update a running serverless container's env vars.
        # In a real deployment, you'd pass WAKE_URL at container creation time.
        # For this test, the worker will auto-derive from RUNPOD_ENDPOINT_ID
        # if that's injected by RunPod, or we accept that scale-to-zero won't work
        # in this specific test (but will work in production).

        # Wait for the worker to boot, start its P2P server, and call
        # POST /workers on the scheduler.  Poll GET /workers until we see
        # a serverless entry with session_active=true.
        print("   ⏳ Waiting for serverless worker to register with scheduler...")
        sl_registered = False
        sl_worker_id  = None
        for i in range(48):  # 4 minutes max (image pull + boot + register)
            try:
                res = requests.get(f"{sched_url}/workers", timeout=5)
                if res.status_code == 200:
                    workers = res.json().get('workers', [])
                    active_sl = [
                        w for w in workers
                        if w.get('type') == 'serverless' and w.get('session_active')
                    ]
                    if active_sl:
                        sl_worker_id  = active_sl[0]['worker_id']
                        sl_registered = True
                        print(f"   ✅ Serverless worker registered: {sl_worker_id[:20]}")
                        print(f"      p2p_url:  {active_sl[0].get('p2p_url')}")
                        print(f"      vram_mb:  {active_sl[0].get('vram_mb')}")
                        break
            except Exception as e:
                pass
            if i % 4 == 0:
                print(f"      [{i*5}s] Still waiting...")
            time.sleep(5)

        if not sl_registered:
            print("   ❌ Serverless worker never registered — skipping serverless inference test")
        else:
            # Submit a job.  Only the serverless worker is in the pool so the
            # scheduler MUST route through the HTTP dispatch path.
            print(f"\n   🧠 Submitting job (serverless-only pool)...")
            sub_res = requests.post(f"{api_url}/submit", json={
                "user_pubkey": funder.address,
                "model_id":    TEST_MODEL,
                "prompt":      "Name one planet in the solar system.",
                "est_tokens":  20
            })

            if sub_res.status_code != 200:
                print(f"   ❌ Submit failed: {sub_res.text}")
            else:
                sl_job_id = sub_res.json()['job_id']
                print(f"   ✅ Job submitted: {sl_job_id}")

                # Longer poll timeout — serverless workers have cold-start
                # overhead on first job (layer downloads from registry).
                sl_passed = False
                for i in range(240):  # 20 minutes
                    try:
                        res    = requests.get(f"{api_url}/results/{sl_job_id}", timeout=5).json()
                        status = res.get('status')
                        if status == 'completed':
                            print(f"\n   🎉 [SERVERLESS] RESULT: {res.get('output', '')}\n")
                            sl_passed = True
                            break
                        if status == 'failed':
                            print(f"\n   ❌ [SERVERLESS] Job failed: {res.get('error')}\n")
                            break
                        if i % 10 == 0:
                            print(f"      Status: {status}... ({i*5}s elapsed)")
                    except Exception as e:
                        pass
                    time.sleep(5)
                else:
                    print("   ❌ [SERVERLESS] Timed out waiting for result")

                if sl_passed:
                    print("   ✅ Serverless dispatch path: PASSED")
                else:
                    print("   ❌ Serverless dispatch path: FAILED")
    except KeyboardInterrupt:
        print("\n\n🛑 INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
    finally:
        # ==========================================
        # 6. TEARDOWN (ALWAYS RUNS)
        # ==========================================
        log_section("🧹 TEARDOWN & CLEANUP")
        print("   Terminating pods to prevent overcharges...")

        if core_pod_id:
            try:
                runpod.terminate_pod(core_pod_id)
                print(f"   ✅ Core terminated ({core_pod_id})")
            except: print(f"   ⚠️ Failed to term core ({core_pod_id})")

        for wid in worker_ids:
            try:
                runpod.terminate_pod(wid)
                print(f"   ✅ Worker terminated ({wid})")
            except: print(f"   ⚠️ Failed to term worker ({wid})")

        # Delete the serverless endpoint and template.
        # Must scale to 0 first (RunPod requirement).
        if sl_endpoint_id:
            delete_serverless_endpoint(sl_endpoint_id, sl_template_id, sl_gpu_tier)

        print("\n👋 Done.")


if __name__ == "__main__":
    main()
