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

# Your funded devnet account (5 SOL)
# Put your private key JSON array here or in a file

# Amount to distribute to each wallet (in SOL)
DISTRIBUTION_AMOUNT = 0.1
# =================================================

runpod.api_key = API_KEY

def load_funded_account():
    """Load your funded devnet account keypair"""
    global FUNDED_ACCOUNT_KEY

    # Option 1: Load from environment variable
    if os.getenv("FUNDED_ACCOUNT_KEY"):
        secret = os.getenv("FUNDED_ACCOUNT_KEY", "")
        FUNDED_ACCOUNT_KEY = Keypair.from_base58_string(secret)
        print(f"   ✅ Loaded funded account from base58 env: {FUNDED_ACCOUNT_KEY.pubkey()}")
        return FUNDED_ACCOUNT_KEY

    # Option 2: Load from file (funded_account.json)
    if os.path.exists("funded_account.json"):
        with open("funded_account.json") as f:
            key_bytes = json.load(f)
            FUNDED_ACCOUNT_KEY = Keypair.from_bytes(bytes(key_bytes))
            print(f"   ✅ Loaded funded account from file: {FUNDED_ACCOUNT_KEY.pubkey()}")
            return FUNDED_ACCOUNT_KEY

    # Option 3: Load from Solana CLI default wallet
    solana_config = os.path.expanduser("~/.config/solana/id.json")
    if os.path.exists(solana_config):
        with open(solana_config) as f:
            key_bytes = json.load(f)
            FUNDED_ACCOUNT_KEY = Keypair.from_bytes(bytes(key_bytes))
            print(f"   ✅ Loaded funded account from Solana CLI: {FUNDED_ACCOUNT_KEY.pubkey()}")
            return FUNDED_ACCOUNT_KEY

    print("❌ ERROR: No funded account found!")
    print("   Please create funded_account.json with your private key JSON array")
    print("   OR set FUNDED_ACCOUNT_KEY environment variable")
    print("   OR place your key in ~/.config/solana/id.json")
    sys.exit(1)

def get_pod_ip(pod_id):
    print(f"   ⏳ Waiting for IP on {pod_id}...")

    # Increase wait time to ~10 minutes (60 loops * 10 seconds)
    # Network volumes and large images take time!
    for i in range(60):
        try:
            pod = runpod.get_pod(pod_id)

            # Check if runtime info exists
            if pod.get('runtime') and pod['runtime'].get('public_ip'):
                ip = pod['runtime']['public_ip']
                print(f"      ✅ Found IP: {ip}")
                return ip

            # If not, print what the pod is doing currently
            status = pod.get('desiredStatus', 'Unknown')
            last_status = pod.get('lastStatus', 'Unknown')
            print(f"      [{i}/60] Status: {last_status} (Target: {status})...")

        except Exception as e:
            print(f"      ⚠️ API Error: {e}")

        time.sleep(10)

    # If we fail, dump the whole pod object to debug
    print("❌ DUMPING POD DATA FOR DEBUGGING:")
    try:
        print(json.dumps(runpod.get_pod(pod_id), indent=2))
    except:
        pass

    raise Exception(f"Pod {pod_id} failed to get IP after 10 minutes")

def distribute_sol(funder_kp, recipient_pubkey, amount_sol, client):
    """Transfer SOL from funder to recipient"""
    try:
        lamports = int(amount_sol * 1_000_000_000)

        # Create transfer instruction
        ix = transfer(TransferParams(
            from_pubkey=funder_kp.pubkey(),
            to_pubkey=recipient_pubkey,
            lamports=lamports
        ))

        # Get latest blockhash
        blockhash = client.get_latest_blockhash().value.blockhash

        # Create and sign transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            funder_kp.pubkey(),
            [funder_kp],
            blockhash
        )

        # Send transaction
        sig = client.send_transaction(tx).value
        print(f"      📤 Transfer sent: {sig}")

        # Wait for confirmation
        time.sleep(2)

        # Verify balance
        balance = client.get_balance(recipient_pubkey).value
        print(f"      ✅ Recipient balance: {balance / 1e9} SOL")

        return True
    except Exception as e:
        print(f"      ❌ Transfer failed: {e}")
        return False

def setup_solana_accounts():
    """Generates wallets and funds them from your funded account"""
    print("\n💰 0. Setting up Solana Wallets...")

    # Load your funded account
    funder_kp = load_funded_account()

    # Check funder balance
    client = Client(RPC_URL)
    funder_balance = client.get_balance(funder_kp.pubkey()).value / 1e9
    print(f"   💵 Funder balance: {funder_balance} SOL")

    needed_sol = DISTRIBUTION_AMOUNT * 2  # scheduler + user
    if funder_balance < needed_sol:
        print(f"❌ Insufficient funds! Need {needed_sol} SOL, have {funder_balance} SOL")
        sys.exit(1)

    # Generate new keypairs
    platform_kp = Keypair()
    scheduler_kp = Keypair()
    user_kp = Keypair()

    print(f"   🔑 Platform: {platform_kp.pubkey()} (no funding needed)")
    print(f"   🔑 Scheduler: {scheduler_kp.pubkey()}")
    print(f"   🔑 User: {user_kp.pubkey()}")

    # Fund scheduler
    print(f"\n   💸 Distributing {DISTRIBUTION_AMOUNT} SOL to Scheduler...")
    if not distribute_sol(funder_kp, scheduler_kp.pubkey(), DISTRIBUTION_AMOUNT, client):
        sys.exit(1)

    # Fund user
    print(f"\n   💸 Distributing {DISTRIBUTION_AMOUNT} SOL to User...")
    if not distribute_sol(funder_kp, user_kp.pubkey(), DISTRIBUTION_AMOUNT, client):
        sys.exit(1)

    scheduler_priv = json.dumps(list(bytes(scheduler_kp)))

    # Check remaining funder balance
    remaining = client.get_balance(funder_kp.pubkey()).value / 1e9
    print(f"\n   💰 Remaining funder balance: {remaining} SOL")

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
            gpu_type_id="NVIDIA GeForce RTX 4090",
            cloud_type="SECURE",
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
                gpu_type_id="NVIDIA GeForce RTX 4090",
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
