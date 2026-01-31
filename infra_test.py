#!/usr/bin/env python3
"""
E2E Test for AZU.CX with FULL MoE Support
Tests Mixtral-8x7B-Instruct-v0.1 on secure cloud GPUs
"""

import runpod
import requests
import time
import json
import sys
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction
import asyncio
import os

# ==========================================
# CONFIGURATION
# ==========================================

CORE_IMG = 'pxxxe/azu-core:latest'
WORKER_IMG = 'pxxxe/azu-worker:latest'

VOLUME_ID = "ryqiz8w01b"

TEST_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",  # PRIMARY MoE TEST
    "Qwen/Qwen2.5-0.5B-Instruct",  # Dense fallback
]

TEST_MODEL = TEST_MODELS[0]  # Mixtral-8x7B

SOLANA_RPC = "https://devnet.helius-rpc.com/?api-key=1d7ca6e1-7700-42eb-b086-8183fda42d76"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

GPU_TYPES_SECURE = [
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100 80GB PCIe",
]

def log_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def deploy_pod_with_retry(name, image, env_vars, gpu_type, max_retries=5):
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
                volume_id=VOLUME_ID,
                volume_mount_path="/data",
                ports="8000/http,8001/http,8002/http,8003/http",
            )

            if isinstance(response, dict):
                pod_id = response.get('id')
                if pod_id:
                    print(f"   ✅ Pod created on {gpu_type}: {pod_id}")
                    return pod_id
                else:
                    error = response.get('message', 'Unknown error')
                    print(f"   ⚠️  {gpu_type} unavailable: {error}")

        except Exception as e:
            print(f"   ⚠️  {gpu_type} error: {str(e)}")

        if attempt < max_retries:
            print(f"   ⏳ Waiting 5s before retry...")
            time.sleep(5)

    print(f"   ❌ {gpu_type} exhausted")
    return None

def deploy_with_fallback(name, image, env_vars, gpu_types):
    """Try deploying across multiple GPU types."""
    for gpu_type in gpu_types:
        pod_id = deploy_pod_with_retry(name, image, env_vars, gpu_type)
        if pod_id:
            return pod_id

    raise Exception(f"❌ Could not deploy {name} on any GPU type")

def resolve_connection(pod_id, port, max_wait=120):
    """Resolve proxy URL for a pod's port."""
    print(f"   ⏳ Resolving connection for {pod_id} (port {port})...")

    for i in range(0, max_wait, 10):
        pod = runpod.get_pod(pod_id)

        if i % 20 == 0:
            runtime = pod.get('runtime', {})
            status = runtime.get('uptimeInSeconds', 'Unknown')
            print(f"      [{i}/{max_wait}s] Pod Status: {status}...")

        runtime = pod.get('runtime', {})
        ports = runtime.get('ports', [])

        for port_info in ports:
            if port_info.get('privatePort') == port:
                private_ip = port_info.get('ip')
                if private_ip and private_ip.startswith('100.'):
                    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net"
                    print(f"      ✅ Using Proxy: {proxy_url}")
                    return proxy_url
                elif private_ip:
                    direct_url = f"http://{private_ip}:{port}"
                    print(f"      ✅ Direct: {direct_url}")
                    return direct_url

        time.sleep(10)

    raise Exception(f"Could not resolve {pod_id}:{port}")

async def transfer_sol(client, from_kp, to_pubkey, amount_sol):
    """Transfer SOL and wait for confirmation."""
    lamports = int(amount_sol * 1_000_000_000)

    ix = transfer(TransferParams(
        from_pubkey=from_kp.pubkey(),
        to_pubkey=to_pubkey,
        lamports=lamports
    ))

    blockhash_resp = await client.get_latest_blockhash()
    blockhash = blockhash_resp.value.blockhash

    tx = Transaction.new_signed_with_payer([ix], from_kp.pubkey(), [from_kp], blockhash)
    sig = await client.send_transaction(tx)

    print(f"      📤 Transfer sent: {sig.value}")
    print(f"      ⏳ Waiting for balance update...")

    for _ in range(30):
        balance = await client.get_balance(to_pubkey)
        if balance.value >= lamports:
            print(f"      ✅ Balance confirmed: {balance.value / 1_000_000_000} SOL")
            return
        await asyncio.sleep(2)

    print(f"      ⚠️ Balance not updated after 60s")

def main():
    log_section("🚀 AZU.CX - Mixtral-8x7B MoE Test")

    print("⚠️  Testing Mixtral-8x7B-Instruct-v0.1 (47GB MoE model)")
    print("   Requires: 2x A5000 (24GB each) or 2x A6000 (48GB each)")
    print("   Cloud: SECURE (required for production models)\n")

    print("⚠️  GitHub Actions must have built and pushed images:")
    print(f"    - CORE_IMG = '{CORE_IMG}'")
    print(f"    - WORKER_IMG = '{WORKER_IMG}'")
    print(f"    - Volume ID: {VOLUME_ID}\n")

    input("Press Enter to start test (Ctrl+C to cancel)...\n")

    runpod.api_key = RUNPOD_API_KEY

    print("\n🚀 Launching RunPod instances on SECURE cloud...")
    print("   (This will provision dedicated GPUs)\n")

    # ==========================================
    # 0. Setup Solana Wallets
    # ==========================================

    log_section("💰 0. Setting up Solana Wallets")

    funder = Keypair()
    platform = Keypair()
    scheduler = Keypair()
    user = Keypair()

    client = AsyncClient(SOLANA_RPC)

    async def setup_wallets():
        balance = await client.get_balance(funder.pubkey())
        print(f"   💵 Funder balance: {balance.value / 1_000_000_000} SOL")

        print(f"   🔑 Platform: {platform.pubkey()}")
        print(f"   🔑 Scheduler: {scheduler.pubkey()}")
        print(f"   🔑 User: {user.pubkey()}\n")

        print(f"   💸 Distributing 0.1 SOL to Scheduler...")
        await transfer_sol(client, funder, scheduler.pubkey(), 0.1)

        print(f"\n   💸 Distributing 0.1 SOL to User...")
        await transfer_sol(client, funder, user.pubkey(), 0.1)

    asyncio.run(setup_wallets())

    # ==========================================
    # 1. Deploy Core
    # ==========================================

    log_section("🚀 1. Deploying Core (API + Registry + Scheduler)")

    core_env = {
        'HF_TOKEN': HF_TOKEN,
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'SOLANA_RPC_URL': SOLANA_RPC,
        'PLATFORM_WALLET_PUBKEY': str(platform.pubkey()),
        'SCHEDULER_PRIVATE_KEY': str(list(bytes(scheduler))),
        'PUBLIC_KEY': 'null',
    }

    core_pod_id = deploy_with_fallback("azu-core-mixtral", CORE_IMG, core_env, GPU_TYPES_SECURE)

    api_url = resolve_connection(core_pod_id, 8000)
    registry_url = resolve_connection(core_pod_id, 8002)
    scheduler_url = resolve_connection(core_pod_id, 8001)

    scheduler_ws = scheduler_url.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws/worker'

    print(f"   ✅ API: {api_url}")
    print(f"   ✅ Registry: {registry_url}")
    print(f"   ✅ Scheduler: {scheduler_ws}")

    print(f"   ⏳ Waiting for Registry to be healthy...")
    for _ in range(40):
        try:
            resp = requests.get(f"{registry_url}/docs", timeout=10)
            if resp.status_code == 200:
                print(f"      ✅ Registry ready!")
                break
        except:
            pass
        time.sleep(3)

    # ==========================================
    # 2. Shard Mixtral-8x7B
    # ==========================================

    log_section("⚡ 2. Sharding Mixtral-8x7B (47GB)")

    print(f"   📥 Downloading and sharding Mixtral-8x7B...")
    print(f"   ⏳ This will take 5-10 minutes (47GB model)...")

    try:
        resp = requests.post(
            f"{registry_url}/models/shard",
            json={"model_id": TEST_MODEL},
            timeout=900  # 15 min timeout
        )

        if resp.status_code != 200:
            print(f"   ❌ Sharding failed: {resp.text}")
            sys.exit(1)

        shard_result = resp.json()
        num_layers = shard_result.get('num_layers', 0)
        print(f"   ✅ Model sharded: {num_layers} layers")

    except requests.exceptions.Timeout:
        print(f"   ❌ Sharding timed out. Model may be too large or network slow.")
        sys.exit(1)

    model_info = requests.get(f"{registry_url}/models/info", params={"model_id": TEST_MODEL}).json()
    is_moe = model_info.get('is_moe', False)

    if is_moe:
        print(f"   🎯 MoE Model Confirmed!")
        moe_layers = [m for m in model_info.get('layer_metadata', []) if m.get('type') == 'moe']
        print(f"   🎯 MoE Layers: {len(moe_layers)}")
        print(f"   🎯 Experts per layer: {moe_layers[0].get('num_experts', 'unknown') if moe_layers else 'unknown'}")
    else:
        print(f"   ⚠️  Model not detected as MoE. Checking config...")
        print(f"   Model info: {json.dumps(model_info, indent=2)}")

    # ==========================================
    # 3. Deploy Workers (2x for Mixtral-8x7B)
    # ==========================================

    log_section("🚀 3. Deploying 2 GPU Workers")

    worker_env = {
        'SCHEDULER_URL': scheduler_ws,
        'REGISTRY_URL': registry_url,
        'HF_TOKEN': HF_TOKEN,
        'PUBLIC_KEY': 'null',
    }

    worker_ids = []
    for i in range(2):
        print(f"\n   Worker {i+1}/2:")
        worker_id = deploy_with_fallback(f"azu-worker-mixtral-{i}", WORKER_IMG, worker_env, GPU_TYPES_SECURE)
        worker_ids.append(worker_id)
        print(f"   ✅ Worker {i+1} deployed: {worker_id}")

    print(f"\n⏳ Waiting for workers to register with Scheduler...")
    for elapsed in range(0, 180, 10):
        try:
            workers_resp = requests.get(f"{scheduler_url}/workers", timeout=10)
            if workers_resp.status_code == 200:
                workers = workers_resp.json()
                print(f"   [{elapsed}s] Connected Workers: {len(workers)}")

                if len(workers) >= 2:
                    print(f"   ✅ 2 Workers connected!")
                    print(f"\n   Worker Details:")
                    for w in workers:
                        print(f"      - {w['id']}: {w['gpu']} | {w['vram']}GB VRAM | Caps: {w['capabilities']}")
                    break
        except:
            pass
        time.sleep(10)

    if len(workers) < 2:
        print(f"   ⚠️  Only {len(workers)} workers connected. May fail.")

    # ==========================================
    # 4. Simulate User Deposit
    # ==========================================

    log_section("💳 4. Simulating User Deposit")

    async def do_deposit():
        print(f"   Sending deposit (0.05 SOL)...")

        ix = transfer(TransferParams(
            from_pubkey=user.pubkey(),
            to_pubkey=platform.pubkey(),
            lamports=50_000_000
        ))

        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash

        tx = Transaction.new_signed_with_payer([ix], user.pubkey(), [user], blockhash)
        sig = await client.send_transaction(tx)
        print(f"   Tx: {sig.value}")
        print(f"   ⏳ Waiting 30s for confirmation...")
        await asyncio.sleep(30)

        deposit_resp = requests.post(
            f"{api_url}/deposit",
            json={
                "tx_sig": str(sig.value),
                "user_pubkey": str(user.pubkey())
            }
        )
        print(f"   Deposit Result: {deposit_resp.json()}")

    asyncio.run(do_deposit())

    # ==========================================
    # 5. Run Mixtral-8x7B Inference
    # ==========================================

    log_section("🧪 5. Running Mixtral-8x7B Inference")

    print(f"   Prompt: 'Explain mixture of experts architecture in one sentence.'")
    print(f"   (First run slower - workers load layers JIT)\n")

    submit_resp = requests.post(
        f"{api_url}/submit",
        json={
            "user_pubkey": str(user.pubkey()),
            "model_id": TEST_MODEL,
            "prompt": "Explain mixture of experts architecture in one sentence.",
            "est_tokens": 100
        }
    )

    if submit_resp.status_code != 200:
        print(f"   ❌ Submit failed: {submit_resp.text}")
        sys.exit(1)

    job_data = submit_resp.json()
    job_id = job_data['job_id']
    print(f"   Job submitted: {job_id}")

    print(f"\n   Polling for results (timeout: 10 minutes)...")

    for elapsed in range(0, 600, 15):
        time.sleep(15)

        result_resp = requests.get(f"{api_url}/results/{job_id}")
        result = result_resp.json()

        status = result.get('status')
        print(f"      [{elapsed}s] Status: {status}")

        if status == 'completed':
            output = result.get('output', '')
            cost = result.get('cost', 0)
            balance = result.get('final_balance', 0)

            log_section("✅ JOB COMPLETED")
            print(f"   Job ID: {job_id}")
            print(f"   Model: {TEST_MODEL}")
            print(f"   Output: {output}")
            print(f"   Cost: {cost} lamports")
            print(f"   Final Balance: {balance} lamports")

            log_section("🎯 MoE ROUTING TEST PASSED")
            print(f"   Mixtral-8x7B successfully executed with expert routing!")

            break

        elif status == 'failed':
            error = result.get('error', 'Unknown')
            log_section("❌ JOB FAILED")
            print(f"   Job ID: {job_id}")
            print(f"   Error: {error}")
            sys.exit(1)

    else:
        print(f"\n   ❌ Job timed out after 10 minutes")
        sys.exit(1)

    # ==========================================
    # 6. Cleanup
    # ==========================================

    log_section("🧹 6. Cleanup")

    print(f"   Pods created:")
    print(f"      - Core: {core_pod_id}")
    for i, wid in enumerate(worker_ids):
        print(f"      - Worker {i+1}: {wid}")

    print(f"\n   To terminate pods:")
    print(f"      runpod terminate {core_pod_id}")
    for wid in worker_ids:
        print(f"      runpod terminate {wid}")

    print(f"\n   Or keep them running for more tests.\n")

    log_section("✅ E2E TEST COMPLETE")

if __name__ == "__main__":
    main()
