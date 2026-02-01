#!/usr/bin/env python3
"""
E2E Test for AZU.CX with FULL MoE Support & Auto-Teardown
Tests Mixtral-8x7B-Instruct-v0.1 on secure cloud GPUs
"""

import runpod
import requests
import time
import json
import sys
import os
import asyncio
import traceback

# === SOLANA IMPORTS ===
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction

# ==========================================
# CONFIGURATION
# ==========================================

CORE_IMG = 'pxxxe/azu-core:latest'
WORKER_IMG = 'pxxxe/azu-worker:latest'

VOLUME_ID = "vkv0m5g4ef"
# VOLUME_ID = None

TEST_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

SOLANA_RPC = "https://devnet.helius-rpc.com/?api-key=1d7ca6e1-7700-42eb-b086-8183fda42d76"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Prioritize High-RAM GPUs for Stability
GPU_TYPES_SECURE = [
    "NVIDIA RTX A6000",             # 48GB VRAM, usually 128GB+ System RAM
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-80GB",
    "NVIDIA GeForce RTX 4090"       # Fallback (Consumer, lower RAM risk)
]

runpod.api_key = RUNPOD_API_KEY

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def log_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def load_funded_account():
    """Load your funded devnet account keypair."""
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
    print("   Make sure you have ~/.config/solana/id.json OR set FUNDED_ACCOUNT_KEY")
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

            # Check if ports are exposed yet
            ports = runtime.get('ports', [])
            is_port_open = any(p.get('privatePort') == port for p in ports)

            if is_port_open:
                # On Secure Cloud, we often rely on Proxy.
                # If we see it's running, return the proxy immediately.
                print(f"      ✅ Port open. Using Proxy: {proxy_url}")
                return proxy_url

        except Exception as e:
            print(f"      ⚠️ API Polling Error: {e}")

        time.sleep(5)

    print("      ⚠️ Timeout waiting for port check. Assuming Proxy is valid.")
    return proxy_url

def deploy_pod_with_retry(name, image, env_vars, gpu_type, max_retries=3):
    """Deploy pod with retries."""
    print(f"   🎯 Trying GPU type: {gpu_type}")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   🔄 Attempt {attempt}/{max_retries}...")

            req = {
                "name": name,
                "image_name": image,
                "gpu_type_id": gpu_type,
                "cloud_type": "SECURE",
                "env": env_vars,
                "ports": "8000/http,8001/http,8002/http,8003/http",
            }
            if VOLUME_ID:
                req["network_volume_id"] = VOLUME_ID
                req["volume_mount_path"] = "/data"

            response = runpod.create_pod(**req)

            if isinstance(response, dict) and response.get('id'):
                pod_id = response['id']
                print(f"   ✅ Pod created: {pod_id}")
                return pod_id

        except Exception as e:
            err_str = str(e).lower()
            if "unavailable" in err_str or "specifications" in err_str:
                 print(f"   ⚠️  {gpu_type} unavailable.")
            else:
                 print(f"   ⚠️  Error: {e}")

        if attempt < max_retries:
            time.sleep(3)

    return None

def deploy_with_fallback(name, image, env_vars):
    """Try deploying across multiple GPU types."""
    for gpu_type in GPU_TYPES_SECURE:
        pod_id = deploy_pod_with_retry(name, image, env_vars, gpu_type)
        if pod_id:
            return pod_id
    raise Exception(f"❌ Could not deploy {name} on any GPU type")

async def transfer_sol(client, from_kp, to_pubkey, amount_sol):
    lamports = int(amount_sol * 1_000_000_000)
    ix = transfer(TransferParams(from_pubkey=from_kp.pubkey(), to_pubkey=to_pubkey, lamports=lamports))
    blockhash = (await client.get_latest_blockhash()).value.blockhash
    tx = Transaction.new_signed_with_payer([ix], from_kp.pubkey(), [from_kp], blockhash)
    sig = await client.send_transaction(tx)
    print(f"      📤 Transfer sent: {sig.value}")

    # Simple wait
    await asyncio.sleep(5)
    return sig.value

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    log_section("🚀 AZU.CX - Mixtral MoE Test")

    # TRACKING VARIABLES FOR TEARDOWN
    core_pod_id = None
    worker_ids = []

    try:
        # ==========================================
        # 0. Setup Solana Wallets
        # ==========================================
        log_section("💰 0. Setting up Solana Wallets")
        funder = load_funded_account()
        platform = Keypair()
        scheduler = Keypair()
        user = Keypair()
        # Removed global client creation to avoid event loop issues

        async def setup_wallets():
            # Create client INSIDE the async loop context
            async with AsyncClient(SOLANA_RPC) as client:
                bal = (await client.get_balance(funder.pubkey())).value / 1e9
                print(f"   💵 Funder: {funder.pubkey()} ({bal} SOL)")
                if bal < 0.3: raise Exception("Insufficient funds")

                print("   💸 Funding Scheduler & User...")
                await transfer_sol(client, funder, scheduler.pubkey(), 0.1)
                await transfer_sol(client, funder, user.pubkey(), 0.1)

        asyncio.run(setup_wallets())

        # ==========================================
        # 1. Deploy Core
        # ==========================================
        log_section("🚀 1. Deploying Core")

        core_env = {
            'HF_TOKEN': HF_TOKEN,
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'SOLANA_RPC_URL': SOLANA_RPC,
            'PLATFORM_WALLET_PUBKEY': str(platform.pubkey()),
            'SCHEDULER_PRIVATE_KEY': str(list(bytes(scheduler))),
            'PUBLIC_KEY': 'null',
            'HF_HOME': '/data/hf_cache'
        }

        core_pod_id = deploy_with_fallback("azu-core-moe", CORE_IMG, core_env)

        # Resolve URLs
        api_url = resolve_connection(core_pod_id, 8000)
        reg_url = resolve_connection(core_pod_id, 8002)
        sched_url = resolve_connection(core_pod_id, 8001) # HTTP URL

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
        # 2. Shard Model
        # ==========================================
        log_section(f"⚡ 2. Sharding {TEST_MODEL}")
        print("   (This takes time for large models...)")

        shard_res = requests.post(f"{reg_url}/models/shard", json={"model_id": TEST_MODEL}, timeout=900)
        if shard_res.status_code != 200:
            raise Exception(f"Sharding Failed: {shard_res.text}")
        print(f"   ✅ Sharding Complete: {shard_res.json()}")

        # ==========================================
        # 3. Deploy Workers
        # ==========================================
        log_section("🚀 3. Deploying 2 Workers")

        worker_env = {
            'SCHEDULER_URL': sched_ws_full,
            'REGISTRY_URL': reg_url,
            'HF_TOKEN': HF_TOKEN,
            'PUBLIC_KEY': 'null',
            # DECOUPLED: We pass the RunPod-specific URL template here via environment
            'P2P_URL_TEMPLATE': 'https://{RUNPOD_POD_ID}-8003.proxy.runpod.net'
        }

        for i in range(2):
            wid = deploy_with_fallback(f"azu-worker-{i}", WORKER_IMG, worker_env)
            worker_ids.append(wid)

        print("\n   ⏳ Waiting for workers to connect to Scheduler...")
        time.sleep(60)

        # ==========================================
        # 4 & 5. Deposit & Inference
        # ==========================================
        log_section("🧪 4. Running Inference")

        # Deposit
        async def do_deposit():
            print("   💳 Sending Deposit...")
            # Create fresh client for this loop
            async with AsyncClient(SOLANA_RPC) as client:
                # Send SOL
                ix = transfer(TransferParams(from_pubkey=user.pubkey(), to_pubkey=platform.pubkey(), lamports=50_000_000))
                blockhash = (await client.get_latest_blockhash()).value.blockhash
                tx = Transaction.new_signed_with_payer([ix], user.pubkey(), [user], blockhash)
                sig = await client.send_transaction(tx)

                # Notify API (Sync request is fine here)
                # But we wait a bit for confusion
                await asyncio.sleep(20)

                requests.post(f"{api_url}/deposit", json={"tx_sig": str(sig.value), "user_pubkey": str(user.pubkey())})
                print("   ✅ Deposit Registered")

        asyncio.run(do_deposit())

        # Submit Job
        print(f"   🧠 Submitting Prompt...")
        sub_res = requests.post(f"{api_url}/submit", json={
            "user_pubkey": str(user.pubkey()),
            "model_id": TEST_MODEL,
            "prompt": "What is the capital of France?",
            "est_tokens": 50
        })

        if sub_res.status_code != 200: raise Exception(f"Submit failed: {sub_res.text}")
        job_id = sub_res.json()['job_id']
        print(f"   ✅ Job ID: {job_id}")

        # Poll
        for i in range(60):
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

        print("\n👋 Done.")

if __name__ == "__main__":
    main()
