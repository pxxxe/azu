import runpod
import time
import requests
import os
import sys
import json
import asyncio
import random

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

# ============================================
# RETRY LOGIC FOR GPU PROVISIONING
# ============================================

GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",      # 24GB, SM_89
    "NVIDIA GeForce RTX 4080",      # 16GB, SM_89
    "NVIDIA RTX A5000",             # 24GB, SM_86
    "NVIDIA GeForce RTX 3090",      # 24GB, SM_86
    "NVIDIA RTX A4500",             # 20GB, SM_86
    "NVIDIA GeForce RTX 4070 Ti",   # 12GB, SM_89
    "NVIDIA RTX A4000",             # 16GB, SM_86
]


def create_pod_with_retry(
    name,
    image_name,
    cloud_type,
    ports,
    env,
    max_attempts_per_gpu=5,
    wait_between_attempts=5
):
    """
    Create RunPod pod with retry logic across multiple GPU types.
    """
    total_attempts = 0

    for gpu_type in GPU_TYPES:
        print(f"\n   🎯 Trying GPU type: {gpu_type}")

        for attempt in range(1, max_attempts_per_gpu + 1):
            total_attempts += 1

            try:
                print(f"   🔄 Attempt {attempt}/{max_attempts_per_gpu} with {gpu_type}...")

                pod = runpod.create_pod(
                    name=name,
                    image_name=image_name,
                    gpu_type_id=gpu_type,
                    cloud_type=cloud_type,
                    ports=ports,
                    env=env
                )

                print(f"   ✅ Pod created successfully on {gpu_type}: {pod['id']}")
                return pod

            except runpod.error.QueryError as e:
                error_msg = str(e)
                is_gpu_error = (
                    "does not have the resources" in error_msg or
                    "failed to provision" in error_msg.lower() or
                    "no longer any instances available" in error_msg.lower() or
                    "requested specifications" in error_msg.lower()
                )

                if is_gpu_error:
                    if attempt < max_attempts_per_gpu:
                        print(f"   ⚠️  {gpu_type} unavailable. Waiting {wait_between_attempts}s...")
                        time.sleep(wait_between_attempts)
                    else:
                        print(f"   ❌ {gpu_type} exhausted.")
                        break
                else:
                    print(f"   ❌ Non-retryable error: {error_msg}")
                    raise

            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                raise

    raise Exception(f"Failed to create pod '{name}' after trying {len(GPU_TYPES)} GPU types.")

# ============================================

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
        core = create_pod_with_retry(
            name="azu-core",
            image_name=CORE_IMG,
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

        # Scheduler HTTP URL for querying worker status
        sched_api_url = f"{api_scheme}://{sched_host}"

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
                print(f"\n   ❌ SHARDING FAILED: {shard_resp.text}")
                raise Exception(f"Sharding failed")

            data = shard_resp.json()
            print(f"   ✅ Model sharded: {data['num_layers']} layers")

        except Exception as e:
            print(f"\n   ❌ Sharding error: {e}")
            raise

        # STEP 3: WORKERS
        print("\n🚀 3. Deploying 2 GPU Workers...")
        for i in range(2):
            w = create_pod_with_retry(
                name=f"azu-worker-{i}",
                image_name=WORKER_IMG,
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
        # NOTE: No explicit preload. Workers connect, Registry has model. Scheduler maps them JIT.
        print("\n⏳ Waiting for workers to register with Scheduler...")

        workers_ready = False
        start_wait = time.time()

        while time.time() - start_wait < 300: # 5 mins max
            try:
                resp = requests.get(f"{sched_api_url}/workers", timeout=5)
                if resp.status_code == 200:
                    workers = resp.json()
                    count = len(workers)
                    print(f"   [{int(time.time()-start_wait)}s] Connected Workers: {count}")

                    if count >= 2:
                        print(f"   ✅ {count} Workers connected and ready for jobs.")
                        workers_ready = True
                        break
            except Exception as e:
                print(f"   ⚠️ Polling error: {e}")

            time.sleep(5)

        if not workers_ready:
            raise Exception("Workers failed to connect in time.")

        # STEP 4: USER DEPOSIT
        print("\n💳 4. Simulating User Deposit...")
        user_kp = wallets['user_kp']
        platform_pub = Pubkey.from_string(wallets['platform_pub'])
        client = Client(RPC_URL)

        transfer_amount = 0.05
        lamports = int(transfer_amount * 1_000_000_000)

        ix = transfer(TransferParams(from_pubkey=user_kp.pubkey(), to_pubkey=platform_pub, lamports=lamports))
        blockhash = client.get_latest_blockhash().value.blockhash
        tx = Transaction.new_signed_with_payer([ix], user_kp.pubkey(), [user_kp], blockhash)

        print(f"   Sending on-chain deposit transaction ({transfer_amount} SOL)...")
        sig = client.send_transaction(tx).value
        print(f"   Tx Sent: {sig}")

        print("      ⏳ Waiting 20s for transaction confirmation...")
        time.sleep(20)

        res = requests.post(f"{api_url}/deposit", json={
            "tx_sig": str(sig),
            "user_pubkey": str(user_kp.pubkey())
        })
        print(f"   Deposit Result: {res.json()}")

        # STEP 5: INFERENCE
        print("\n🧪 5. Running Inference Job...")
        print("   (Note: First run will be slower as workers download layers JIT)")

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
        print(f"   Polling results for Job {job_id}...")

        # INCREASED TIMEOUT to 300s (5 minutes) for slow JIT downloads
        for i in range(150):
            res = requests.get(f"{api_url}/results/{job_id}")
            data = res.json()

            status = data.get('status')
            if status == 'completed':
                print(f"\n🎉 SUCCESS: {data['output']}\n")
                return
            if status == 'failed':
                print(f"\n❌ JOB FAILED: {data.get('error')}\n")
                return

            if i % 10 == 0:
                print(f"   [{i*2}s] Status: {status}...")
            time.sleep(2)

        raise Exception("Test Timed Out waiting for inference result")

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
