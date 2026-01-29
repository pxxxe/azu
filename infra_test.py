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
NETWORK_VOLUME_ID = "vkv0m5g4ef" # <--- PASTE YOUR VOLUME ID
DATA_CENTER = "EU-RO-1"

CORE_IMG = "pxxxe/azu-core:latest"     # <--- UPDATE USERNAME
WORKER_IMG = "pxxxe/azu-worker:latest" # <--- UPDATE USERNAME

RPC_URL = "https://devnet.helius-rpc.com/?api-key=1d7ca6e1-7700-42eb-b086-8183fda42d76"
# =================================================

runpod.api_key = API_KEY

def get_pod_ip(pod_id):
    print(f"   ⏳ Waiting for IP on {pod_id}...")
    for _ in range(30):
        try:
            pod = runpod.get_pod(pod_id)
            if pod['runtime'] and pod['runtime']['public_ip']:
                return pod['runtime']['public_ip']
        except: pass
        time.sleep(4)
    raise Exception("Pod failed to get IP")

def setup_solana_accounts():
    """Generates wallets and funds them via Devnet Airdrop"""
    print("\n💰 0. Setting up Solana Wallets...")

    platform_kp = Keypair()
    scheduler_kp = Keypair()
    user_kp = Keypair()

    client = Client(RPC_URL)

    accounts = [("Scheduler", scheduler_kp), ("User", user_kp)]

    for name, kp in accounts:
        balance = 0
        retries = 3
        print(f"   💧 Requesting Airdrop for {name} ({kp.pubkey()})...")

        while retries > 0 and balance == 0:
            try:
                # 1 SOL
                client.request_airdrop(kp.pubkey(), 1_000_000_000)
                for _ in range(10):
                    time.sleep(2)
                    bal_resp = client.get_balance(kp.pubkey())
                    if bal_resp.value > 0:
                        balance = bal_resp.value
                        print(f"      ✅ {name} Funded: {balance / 1e9} SOL")
                        break
            except Exception as e:
                print(f"      ⚠️ Retry ({e})...")
                time.sleep(2)
            retries -= 1

        if balance == 0:
            print(f"❌ Failed to fund {name}. If devnet is down, use a private key from Phantom.")
            sys.exit(1)

    scheduler_priv = json.dumps(list(bytes(scheduler_kp)))

    return {
        "platform_pub": str(platform_kp.pubkey()),
        "scheduler_priv": scheduler_priv,
        "user_kp": user_kp,
        "scheduler_pub": str(scheduler_kp.pubkey())
    }

def run_lifecycle():
    core_id = None
    worker_ids = []

    try:
        # STEP 0: WALLETS
        wallets = setup_solana_accounts()
        print(f"   🔑 Platform Pubkey: {wallets['platform_pub']}")

        # STEP 1: CORE
        print("\n🚀 1. Deploying Core...")
        core = runpod.create_pod(
            name="azu-core",
            image_name=CORE_IMG,
            gpu_type_id="CPU-Only",
            cloud_type="COMMUNITY",
            data_center_id=DATA_CENTER,
            ports="8000/http,8001/http,8002/http",
            network_volume_id=NETWORK_VOLUME_ID,
            volume_mount_path="/data/layers",
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
        core_ip = get_pod_ip(core_id)
        print(f"   ✅ Core Online: {core_ip}")

        # STEP 2: SHARDING
        print("\n⚡ 2. Sharding Model...")
        requests.post(f"http://{core_ip}:8002/models/shard", json={"model_id": "Qwen/Qwen2.5-0.5B"})
        time.sleep(5)

        # STEP 3: WORKERS
        print("\n🚀 3. Deploying 2 GPU Workers...")
        for i in range(2):
            w = runpod.create_pod(
                name=f"azu-worker-{i}",
                image_name=WORKER_IMG,
                gpu_type_id="NVIDIA GeForce RTX 3090",
                data_center_id=DATA_CENTER,
                ports="8003/http",
                env={
                    "SCHEDULER_URL": f"ws://{core_ip}:8001/ws/worker",
                    "REGISTRY_URL": f"http://{core_ip}:8002",
                    "HF_TOKEN": HF_TOKEN
                }
            )
            worker_ids.append(w['id'])
            print(f"   Worker {i} deployed: {w['id']}")

        print("   ⏳ Waiting for Swarm Assembly (60s)...")
        time.sleep(60)

        # STEP 4: USER DEPOSIT
        print("\n💳 4. Simulating User Deposit...")

        user_kp = wallets['user_kp']
        platform_pub = Pubkey.from_string(wallets['platform_pub'])
        client = Client(RPC_URL)

        # --- MODERN TRANSACTION CONSTRUCTION ---
        # 1. Create Instruction using transfer function
        ix = transfer(TransferParams(
            from_pubkey=user_kp.pubkey(),
            to_pubkey=platform_pub,
            lamports=100_000_000
        ))

        # 2. Get Blockhash
        blockhash = client.get_latest_blockhash().value.blockhash

        # 3. Create and Sign Transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            user_kp.pubkey(),
            [user_kp],
            blockhash
        )

        # 5. Send
        print("   Sending on-chain deposit transaction...")
        sig = client.send_transaction(tx).value
        print(f"   Tx Sent: {sig}")
        time.sleep(10)

        # Notify API
        print("   Notifying API of deposit...")
        res = requests.post(f"http://{core_ip}:8000/deposit", json={
            "tx_sig": str(sig),
            "user_pubkey": str(user_kp.pubkey())
        })
        print(f"   Deposit Result: {res.json()}")

        # STEP 5: INFERENCE
        print("\n🧪 5. Running Inference Job...")
        res = requests.post(f"http://{core_ip}:8000/submit", json={
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
            res = requests.get(f"http://{core_ip}:8000/results/{job_id}")
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
