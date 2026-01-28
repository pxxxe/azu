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
