import redis.asyncio as redis
import asyncio
import json
import time
import sys
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, WebSocket
from azu.shared import get_config
from azu.shared.auth import get_auth_provider, is_auth_enabled

# Initialize config
config = get_config()
r = redis.Redis(host=config.redis.host, port=config.redis.port, decode_responses=True)
app = FastAPI()

@dataclass
class WorkerState:
    pubkey: str
    ws: WebSocket
    specs: dict
    vram_total_mb: int
    actual_free_mb: int
    # Tracks the size of layers we have ASSIGNED to this worker
    vram_used_mb: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    cached_layers: Set[str] = field(default_factory=set)
    # Payment address for receiving earnings - ADDED FOR PAYMENTS
    payment_address: Optional[str] = None

@dataclass
class JobState:
    id: str
    model_id: str
    input_prompt: str
    owner: str
    topology: List[dict]
    max_tokens: int = 50
    auth_token: Optional[str] = None

class MoEScheduler:
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.workers: Dict[str, WorkerState] = {}
        self.active_jobs: Dict[str, JobState] = {}

    async def register_worker(self, ws: WebSocket, specs: dict) -> str:
        wid = specs['pubkey']
        vram = specs.get('vram_mb', 24000)
        # Init actual_free to total until first heartbeat
        payment_addr = specs.get('payment_address')  # EXTRACT PAYMENT ADDRESS
        self.workers[wid] = WorkerState(pubkey=wid, ws=ws, specs=specs, vram_total_mb=vram, actual_free_mb=vram, payment_address=payment_addr)
        print(f"✅ Registered: {wid} | VRAM: {vram}MB | Payment: {payment_addr[:20] if payment_addr else 'None'}...")
        return wid

    async def unregister_worker(self, wid: str):
        if wid in self.workers:
            del self.workers[wid]
            print(f"❌ Worker {wid} disconnected")

    async def handle_heartbeat(self, wid: str, data: dict):
        if wid in self.workers:
            # We record this for observability, but we DO NOT use it for planning
            # because it lags behind the downloads.
            self.workers[wid].actual_free_mb = data.get('vram_free_mb', 0)
            self.workers[wid].last_heartbeat = time.time()

    def _find_best_worker(self, size_mb: int, current_job_allocations: Dict[str, int],
                          previous_worker_id: str = None, cache_key: str = None) -> Optional[WorkerState]:
        # Only consider workers with a routable P2P URL — relay-mode workers cannot
        # participate in tensor routing and must be excluded from placement.
        candidates = [w for w in self.workers.values() if w.specs.get('p2p_url')]
        if not candidates: return None

        # --- SAFETY PARAMETERS ---
        # 1. Runtime Reserve: Fixed deduction for PyTorch Context, CUDA Kernels, OS
        RUNTIME_RESERVE_MB = 4096

        # 2. Weight Safety Factor: Safetensors are packed FP16. PyTorch runtime
        # needs extra for fragmentation, alignments, and gradients (even if disabled).
        # Log analysis shows Real Usage ~= 1.6 * Disk Size.
        WEIGHT_SAFETY_FACTOR = 1

        valid = []

        for w in candidates:
            usable_vram = max(0, w.vram_total_mb - RUNTIME_RESERVE_MB)

            # Use the Safety Factor on the WEIGHTS, not the limit
            # This allows us to accurately track "Effective Load"

            pending_load = current_job_allocations.get(w.pubkey, 0)

            # Existing load + Pending load
            current_effective_load = (w.vram_used_mb + pending_load) * WEIGHT_SAFETY_FACTOR

            # New allocation cost (also multiplied)
            allocation_cost_raw = 0 if (cache_key in w.cached_layers) else size_mb
            allocation_cost_effective = allocation_cost_raw * WEIGHT_SAFETY_FACTOR

            projected_usage = current_effective_load + allocation_cost_effective

            if projected_usage <= usable_vram:
                 valid.append((w, allocation_cost_effective))
            else:
                 # Debug rejection
                 # print(f"   ❌ Rejected {w.pubkey[:8]}: Proj {projected_usage:.0f} > Usable {usable_vram:.0f}")
                 pass

        if not valid:
            print(f"   ⚠️ NO WORKERS FIT {size_mb:.0f}MB (Effective: {size_mb * WEIGHT_SAFETY_FACTOR:.0f}MB)")
            return None

        def score(item):
            w, effective_cost = item
            s = 0
            if effective_cost == 0: s += 10000

            pending = current_job_allocations.get(w.pubkey, 0)

            # Re-calc for scoring
            usable = max(0, w.vram_total_mb - RUNTIME_RESERVE_MB)
            current_effective = (w.vram_used_mb + pending) * WEIGHT_SAFETY_FACTOR

            available_room = usable - current_effective
            s += (available_room / 1024)

            # --- REDUCED LOCALITY BONUS ---
            # Was 5. Reduced to 2 to prevent "Clumping" (filling one worker to 99% before moving).
            # We want to spread the load across the 8 workers to reduce OOM risk.
            if previous_worker_id and w.pubkey == previous_worker_id:
                s += 5

            return s

        chosen_worker, effective_cost = max(valid, key=score)

        # Update ephemeral ledger with RAW size (safety factor is applied at read-time)
        raw_cost = 0 if (cache_key in chosen_worker.cached_layers) else size_mb
        new_load = current_job_allocations.get(chosen_worker.pubkey, 0) + raw_cost
        current_job_allocations[chosen_worker.pubkey] = new_load

        # Print effective usage for debugging
        eff_total = (chosen_worker.vram_used_mb + new_load) * WEIGHT_SAFETY_FACTOR
        print(f"   ✅ Selected {chosen_worker.pubkey[:8]}. Eff Load: {eff_total:.0f}MB / {chosen_worker.vram_total_mb - RUNTIME_RESERVE_MB:.0f}MB")
        sys.stdout.flush()

        return chosen_worker

    def _plan_job(self, model_info) -> Optional[List[dict]]:
        layers = model_info.get('layer_metadata', [])
        if not self.workers: return None

        topology = []
        prev_worker_id = None
        current_job_allocations = {}

        print(f"\n🧠 Planning Job for {model_info['model_id']} ({len(layers)} layers)...")

        for layer in layers:
            layer_idx = layer['layer_idx']
            l_type = layer.get('type', 'dense')

            if l_type == 'dense':
                size = layer.get('size_mb', 0)
                cache_key = f"{model_info['model_id']}:{layer_idx}:main"

                # print(f"Layer {layer_idx} (Dense): Need {size:.0f}MB")
                target_worker = self._find_best_worker(size, current_job_allocations, prev_worker_id, cache_key)
                if not target_worker: return None

                node = {
                    "layer_idx": layer_idx,
                    "type": l_type,
                    "worker_id": target_worker.pubkey,
                    "endpoint": target_worker.specs.get('p2p_url')
                }
                prev_worker_id = target_worker.pubkey
                topology.append(node)

            elif l_type == 'moe':
                # 1. Place Router + Shared
                shared_size = layer.get('shared_size_mb', 0)
                router_size = layer.get('router_size_mb', 0)
                combined_size = shared_size + router_size
                shared_key = f"{model_info['model_id']}:{layer_idx}:shared"

                # print(f"Layer {layer_idx} (Router): Need {combined_size:.0f}MB")
                router_worker = self._find_best_worker(combined_size, current_job_allocations, prev_worker_id, shared_key)
                if not router_worker: return None

                node = {
                    "layer_idx": layer_idx, "type": "moe",
                    "worker_id": router_worker.pubkey, "endpoint": router_worker.specs.get('p2p_url'),
                    "expert_map": {}
                }
                prev_worker_id = router_worker.pubkey

                # 2. Place Experts
                num_experts = layer.get('num_experts', 0)
                expert_size = layer.get('expert_size_mb', 0)

                for exp_idx in range(num_experts):
                    exp_key = f"{model_info['model_id']}:{layer_idx}:expert:{exp_idx}"
                    # print(f"Layer {layer_idx} (Expert {exp_idx}): Need {expert_size:.0f}MB")
                    exp_worker = self._find_best_worker(expert_size, current_job_allocations, node['worker_id'], exp_key)

                    if not exp_worker:
                        print(f"⚠️ Cluster Full! Cannot place expert {exp_idx}")
                        return None

                    node['expert_map'][str(exp_idx)] = exp_worker.specs.get('p2p_url')

                topology.append(node)

        # COMMIT: The plan is valid. Now update the persistent WorkerState.
        print("💾 Committing Plan to Persistent Ledger...")
        for node in topology:
            w = self.workers.get(node['worker_id'])
            if not w: continue

            if node['type'] == 'dense':
                 l_meta = next(l for l in layers if l['layer_idx'] == node['layer_idx'])
                 size = l_meta.get('size_mb', 0)
                 cache_key = f"{model_info['model_id']}:{node['layer_idx']}:main"

                 if cache_key not in w.cached_layers:
                     w.vram_used_mb += size
                     w.cached_layers.add(cache_key)

            elif node['type'] == 'moe':
                 l_meta = next(l for l in layers if l['layer_idx'] == node['layer_idx'])

                 # Router/Shared
                 combined = l_meta.get('shared_size_mb', 0) + l_meta.get('router_size_mb', 0)
                 shared_key = f"{model_info['model_id']}:{node['layer_idx']}:shared"
                 if shared_key not in w.cached_layers:
                     w.vram_used_mb += combined
                     w.cached_layers.add(shared_key)

                 # Experts
                 exp_size = l_meta.get('expert_size_mb', 0)
                 shared_size = l_meta.get('shared_size_mb', 0)
                 router_size = l_meta.get('router_size_mb', 0)

                 for exp_idx, url in node['expert_map'].items():
                     w_exp = next((wk for wk in self.workers.values() if wk.specs.get('p2p_url') == url), None)
                     if w_exp:
                         exp_key = f"{model_info['model_id']}:{node['layer_idx']}:expert:{exp_idx}"
                         shared_key_exp = f"{model_info['model_id']}:{node['layer_idx']}:shared"

                         if exp_key not in w_exp.cached_layers:
                             # BUG FIX: If worker doesn't have the router/shared for this layer,
                             # it needs to load them when processing the expert
                             if shared_key_exp not in w_exp.cached_layers:
                                 w_exp.vram_used_mb += shared_size + router_size + exp_size
                                 w_exp.cached_layers.add(shared_key_exp)
                             else:
                                 w_exp.vram_used_mb += exp_size

                             w_exp.cached_layers.add(exp_key)

        # Log final states
        for w in self.workers.values():
            # Apply safety factor for display
            eff = w.vram_used_mb * 1.5
            print(f"   🏁 Worker {w.pubkey[:8]} Usage: {w.vram_used_mb:.0f}MB (Eff: {eff:.0f}MB) / {w.vram_total_mb:.0f}MB")

        # Bookend the topology
        first_worker = topology[0]
        last_worker = topology[-1]

        embed_node = {
            "layer_idx": -1,
            "type": "dense",
            "role": "embed",
            "worker_id": first_worker['worker_id'],
            "endpoint": first_worker['endpoint']
        }

        decode_node = {
            "layer_idx": -1,
            "type": "dense",
            "role": "decode",
            "worker_id": last_worker['worker_id'],
            "endpoint": last_worker['endpoint']
        }

        topology = [embed_node] + topology + [decode_node]
        return topology

    async def _check_model_status(self, model_id: str) -> str:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{self.registry_url}/models/status", params={"model_id": model_id}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status", "idle")
        except: pass
        return "error"

    async def process_queue(self):
        print("🚀 [Scheduler] Intelligent Dispatcher Active")
        while True:
            try:
                item = await r.blpop("job_queue", timeout=2)
                if not item: continue

                raw_job = json.loads(item[1])
                job_id = raw_job['id']
                model_id = raw_job['model_id']

                status = await self._check_model_status(model_id)

                if status == 'ready':
                    pass
                elif status == 'processing':
                    print(f"⏳ JIT: Model {model_id} is processing. Re-queueing...")
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue
                elif status == 'failed':
                    print(f"❌ JIT: Model {model_id} failed to shard. Dropping job.")
                    continue
                else:
                    print(f"⬇️ JIT: Triggering download for {model_id}")
                    try:
                        async with aiohttp.ClientSession() as sess:
                            await sess.post(f"{self.registry_url}/models/shard", json={"model_id": model_id})
                    except: pass
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(5)
                    continue

                model_info = None
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.get(f"{self.registry_url}/models/info", params={"model_id": model_id}) as resp:
                            if resp.status == 200: model_info = await resp.json()
                except: pass

                if not model_info:
                    print("⚠️ No model info. Re-queueing.")
                    await r.rpush("job_queue", item[1])
                    await asyncio.sleep(2)
                    continue

                plan = self._plan_job(model_info)

                if not plan:
                    print("⚠️ Planning failed (Insufficient Resources). Re-queueing...")
                    await asyncio.sleep(5)
                    await r.rpush("job_queue", item[1])
                    continue

                job = JobState(raw_job['id'], model_id, raw_job['input_prompt'], raw_job['owner'], plan, max_tokens=raw_job.get('tokens', 50))
                self.active_jobs[job_id] = job
                await self._dispatch(job)

            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _dispatch(self, job: JobState):
        print(f"📦 Dispatching Job {job.id} (Smart Route)")

        first_node_endpoint = job.topology[0]['endpoint']

        # Generate auth token for this job (HMAC or configured provider).
        auth_token = None
        if is_auth_enabled():
            auth_token = get_auth_provider().generate_token(job.id)
            job.auth_token = auth_token

        # === PHASE 1: Send JOB_START to ALL workers with topology ===
        # This triggers the P2P mesh handshake on each worker
        all_peer_urls = list(set(node['endpoint'] for node in job.topology if node['endpoint']))
        print(f"🔗 [Job {job.id[:8]}] Sending job_start to {len(all_peer_urls)} workers for mesh handshake...")

        job_start_tasks = []
        for node in job.topology:
            w = self.workers.get(node['worker_id'])
            if not w:
                print(f"⚠️ Worker {node['worker_id']} not found!")
                continue

            job_start_payload = {
                "type": "JOB_START",
                "job_id": job.id,
                "model_id": job.model_id,
                "topology": all_peer_urls,  # All peer P2P URLs for handshake
                "auth_token": auth_token,
            }
            job_start_tasks.append(w.ws.send_json(job_start_payload))

        # Wait for all workers to receive job_start
        if job_start_tasks:
            await asyncio.gather(*job_start_tasks)
            print(f"🔗 [Job {job.id[:8]}] All workers received job_start, waiting for handshake...")
            # Give workers time to complete mesh handshake
            await asyncio.sleep(2)  # Workers will proceed when ready, this is just safety

        # === PHASE 2: Send EXECUTE messages (original logic) ===
        for i, node in enumerate(job.topology):
            w = self.workers.get(node['worker_id'])
            if not w:
                print(f"⚠️ Worker {node['worker_id']} not found!")
                continue

            is_last_node = (i == len(job.topology) - 1)
            next_hop = job.topology[i + 1]['endpoint'] + "/tensor_in" if not is_last_node else None
            next_layer_idx = job.topology[i + 1]['layer_idx'] if not is_last_node else None
            role = node.get('role')

            print(f"  Layer {i}: type={node['type']}, worker={w.pubkey[:8]}, next_hop={next_hop}")

            payload = {
                "job_id": job.id,
                "model_id": job.model_id,
                "layer_idx": node['layer_idx'],
                "next_hop": next_hop,
                "next_layer_idx": next_layer_idx
            }

            if node['type'] == 'dense':
                payload.update({
                    "type": "EXECUTE_DENSE",
                    "input": job.input_prompt if role == 'embed' else None,
                    "is_first": role == 'embed',
                    "is_last": role == 'decode',
                    "max_tokens": job.max_tokens,
                    "first_node_endpoint": first_node_endpoint if role == 'decode' else None
                })
                await w.ws.send_json(payload)

            elif node['type'] == 'moe':
                payload.update({
                    "type": "EXECUTE_ROUTER",
                    "expert_map": node['expert_map']
                })
                await w.ws.send_json(payload)

                tasks = {}
                for exp_idx, url in node['expert_map'].items():
                    exp_w = next((wk for wk in self.workers.values() if wk.specs.get('p2p_url') == url), None)
                    if exp_w:
                        if exp_w.pubkey not in tasks: tasks[exp_w.pubkey] = []
                        tasks[exp_w.pubkey].append(int(exp_idx))

                for wid, indices in tasks.items():
                    t_w = self.workers.get(wid)
                    for idx in indices:
                        await t_w.ws.send_json({
                            "type": "EXECUTE_EXPERT",
                            "job_id": job.id,
                            "model_id": job.model_id,
                            "layer_idx": node['layer_idx'],
                            "expert_idx": idx,
                            "return_url": node['endpoint']
                        })

    async def handle_result(self, wid: str, data: dict):
        job_id = data.get('job_id')
        if data.get('status') == 'completed':
            print(f"🎉 Job {job_id} Finished")

            # =====================================================
            # PAYMENT PROCESSING - Credit workers for completed work
            # =====================================================
            await self._process_worker_payment(job_id)

            await r.setex(f"result:{job_id}", 3600, json.dumps(data))
            if job_id in self.active_jobs: del self.active_jobs[job_id]

    async def _process_worker_payment(self, job_id: str):
        """
        Credit workers for completed work.

        This is where workers get paid! The scheduler credits the worker's
        internal ledger balance based on the work they completed.
        """
        try:
            # Import payment modules
            from azu.shared.payments import get_payment_provider
            from azu.shared.ledger import get_ledger
            from azu.shared.economics import calculate_worker_payments, WORKER_SHARE
            from azu.shared import TransactionType

            # Get job details
            job = self.active_jobs.get(job_id)
            if not job:
                return

            # Get topology and calculate payments
            topology = job.topology

            # Group workers by payment address
            worker_layers: Dict[str, List[int]] = {}
            for node in topology:
                wid = node['worker_id']
                w = self.workers.get(wid)
                if w and w.payment_address:
                    if w.payment_address not in worker_layers:
                        worker_layers[w.payment_address] = []
                    worker_layers[w.payment_address].append(node['layer_idx'])

            if not worker_layers:
                print(f"⚠️ No workers with payment addresses for job {job_id}")
                return

            # Calculate payments per worker
            est_tokens = job.max_tokens * 2  # Rough estimate
            worker_payments = calculate_worker_payments(
                worker_layers=worker_layers,
                num_tokens=est_tokens,
                is_moe=False,
                num_experts=1
            )

            # Credit each worker
            ledger = await get_ledger(r)
            for payment in worker_payments:
                try:
                    await ledger.credit(
                        address=payment.worker_address,
                        amount=payment.amount,
                        transaction_type=TransactionType.WORKER_CREDIT,
                        job_id=job_id,
                        metadata={
                            "layers": worker_layers.get(payment.worker_address, []),
                            "tokens": est_tokens
                        }
                    )
                    print(f"💰 Credited worker {payment.worker_address[:20]}... with {payment.amount:.8f}")
                except Exception as e:
                    print(f"❌ Failed to credit worker: {e}")

            print(f"✅ Worker payments processed for job {job_id}")

            # =====================================================
            # ON-CHAIN PAYOUT - Transfer real tokens to workers
            # =====================================================
            await self._process_worker_payouts(worker_payments)

        except Exception as e:
            print(f"⚠️ Worker payment processing failed: {e}")

    async def _process_worker_payouts(self, worker_payments: List):
        """
        Process on-chain payouts to workers when threshold is exceeded.

        This is where real tokens get sent from the platform wallet
        to the worker's external address on-chain.
        """
        try:
            from azu.shared.payments import get_payment_provider
            from azu.shared.ledger import get_ledger

            # Get payment provider
            provider = get_payment_provider()
            ledger = await get_ledger(r)

            # Check each worker's balance
            for payment in worker_payments:
                worker_address = payment.worker_address
                balance = await ledger.get_balance(worker_address)

                # Check if balance exceeds payout threshold
                payout_threshold = config.payment.payout_threshold

                if balance.available >= payout_threshold:
                    try:
                        print(f"💸 Initiating on-chain payout to {worker_address[:20]}... ({balance.available:.6f} tokens)")

                        # Execute on-chain transfer
                        payout_info = await provider.payout(
                            recipient_address=worker_address,
                            amount=balance.available,
                            memo=f"Payout for work completed"
                        )

                        if payout_info.status == "confirmed":
                            # Deduct from internal ledger after successful on-chain transfer
                            await ledger.debit(
                                address=worker_address,
                                amount=balance.available,
                                transaction_type=TransactionType.PAYOUT,
                                metadata={
                                    "tx_hash": payout_info.tx_hash,
                                    "status": payout_info.status
                                }
                            )
                            print(f"✅ Payout confirmed: {payout_info.tx_hash[:20]}...")
                        else:
                            print(f"⚠️ Payout pending: {payout_info.tx_hash[:20]}...")

                    except Exception as e:
                        print(f"❌ Payout failed for {worker_address[:20]}...: {e}")
                else:
                    print(f"💰 Worker {worker_address[:20]}... balance ({balance.available:.6f}) below threshold ({payout_threshold})")

        except Exception as e:
            print(f"⚠️ Payout processing failed: {e}")

scheduler = MoEScheduler(f"http://localhost:{config.registry.port}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.process_queue())

@app.websocket("/ws/worker")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    wid = None
    try:
        msg = await ws.receive_json()
        if msg.get('type') == "REGISTER":
            wid = await scheduler.register_worker(ws, msg['specs'])
            while True:
                data = await ws.receive_json()
                if data['type'] == "RESULT":
                    await scheduler.handle_result(wid, data)
                elif data['type'] == "HEARTBEAT":
                    await scheduler.handle_heartbeat(wid, data)
    except:
        if wid: await scheduler.unregister_worker(wid)
