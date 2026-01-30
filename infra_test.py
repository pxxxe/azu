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

from urllib.parse import quote

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
    Tries each GPU type in GPU_TYPES list before giving up.
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

                # Check if it's a retryable GPU availability error
                is_gpu_error = (
                    "does not have the resources" in error_msg or
                    "failed to provision" in error_msg.lower() or
                    "no longer any instances available" in error_msg.lower() or
                    "requested specifications" in error_msg.lower()
                )

                if is_gpu_error:
                    if attempt < max_attempts_per_gpu:
                        print(f"   ⚠️  {gpu_type} unavailable: {error_msg[:100]}...")
                        print(f"   ⏳ Waiting {wait_between_attempts}s before retry...")
                        time.sleep(wait_between_attempts)
                    else:
                        print(f"   ❌ {gpu_type} exhausted after {max_attempts_per_gpu} attempts")
                        # Break to try next GPU type
                        break
                else:
                    # Non-retryable error, raise immediately
                    print(f"   ❌ Non-retryable error: {error_msg}")
                    raise

            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                raise

    # If we get here, all GPU types failed
    raise Exception(
        f"Failed to create pod '{name}' after trying {len(GPU_TYPES)} GPU types "
        f"({total_attempts} total attempts)"
    )

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

        # STEP 1: CORE (WITH RETRY)
        print("\n🚀 1. Deploying Core...")
        core = create_pod_with_retry(
            name="azu-core",
            image_name=CORE_IMG,
            # gpu_type_id="NVIDIA GeForce RTX 4080",
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

            print(f"\n   📊 Response Status: {shard_resp.status_code}")
            print(f"   📊 Response Headers: {dict(shard_resp.headers)}")

            if shard_resp.status_code != 200:
                print(f"\n   ❌ SHARDING FAILED ❌")
                print(f"   Status Code: {shard_resp.status_code}")
                print(f"   Response Text:\n{shard_resp.text}")

                # Try to parse JSON error
                try:
                    error_data = shard_resp.json()
                    if 'detail' in error_data:
                        detail = error_data['detail']
                        if isinstance(detail, dict):
                            print(f"\n   🔍 Error Details:")
                            print(f"      Error: {detail.get('error', 'Unknown')}")
                            print(f"      Model: {detail.get('model_id', 'Unknown')}")
                            if 'traceback' in detail:
                                print(f"\n   📋 Full Traceback:")
                                print(detail['traceback'])
                        else:
                            print(f"\n   Error: {detail}")
                except:
                    pass

                raise Exception(f"Sharding failed with status {shard_resp.status_code}: {shard_resp.text}")

            data = shard_resp.json()
            print(f"   ✅ Model sharded: {data['num_layers']} layers")

        except requests.exceptions.Timeout:
            print(f"\n   ❌ Request timed out after 300 seconds")
            print(f"   The registry might still be processing. Check pod logs:")
            print(f"   runpod ssh {core_id}")
            raise

        except Exception as e:
            print(f"\n   ❌ Sharding error: {e}")
            print(f"\n   🔍 Debugging steps:")
            print(f"   1. Check registry logs:")
            print(f"      runpod ssh {core_id}")
            print(f"      docker logs <container_id>")
            print(f"   2. Check HF_TOKEN is valid")
            print(f"   3. Check disk space on pod")
            print(f"   4. Try manually:")
            print(f"      curl -X POST {reg_url}/models/shard -H 'Content-Type: application/json' -d '{{'model_id': '{model_id}'}}'")
            raise

        # STEP 3: WORKERS (WITH RETRY)
        print("\n🚀 3. Deploying 2 GPU Workers...")
        for i in range(2):
            w = create_pod_with_retry(
                name=f"azu-worker-{i}",
                image_name=WORKER_IMG,
                # gpu_type_id="NVIDIA GeForce RTX 4080",
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
                f"{scheduler_base}/preload/{quote(model_id, safe='')}",
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
