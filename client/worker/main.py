import torch
import asyncio
import websockets
import aiohttp
import json
import base64
import io
import os
import urllib.request
import traceback
from aiohttp import web
from transformers import AutoTokenizer
from layer_loader import LayerLoader

# Configuration
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

class ProductionWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)

        # State
        self.loaded_model_id = None
        self.active_layers = []  # List[torch.nn.Module]
        self.layer_indices = []  # List[int]
        self.embeddings = None
        self.lm_head = None

        # Concurrency
        self.p2p_queue = asyncio.Queue()

        print(f"🚀 [Worker] Initialized on {self.device}")

    def get_public_ip(self):
        try:
            return urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        except:
            return "127.0.0.1"

    def get_p2p_url(self):
        """
        Returns the Reachable URL for this worker.
        CRITICAL for RunPod: Use the Proxy URL.
        """
        # 1. Check for RunPod env
        pod_id = os.getenv("RUNPOD_POD_ID")
        if pod_id:
            # RunPod exposes port 8003 via proxy
            url = f"https://{pod_id}-8003.proxy.runpod.net"
            print(f"🌍 [Net] RunPod Detected. Advertising Proxy: {url}")
            return url

        # 2. Local fallback
        ip = self.get_public_ip()
        return f"http://{ip}:8003"

    # --- P2P SERVER ---
    async def start_p2p_server(self):
        app = web.Application()
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8003)
        await site.start()
        print("👂 [P2P] Listening on 0.0.0.0:8003")

    async def handle_tensor_ingress(self, request):
        """Accepts tensors from upstream workers."""
        try:
            # We support wrapped JSON {"tensor": "base64..."}
            # Or raw binary body (more efficient)
            try:
                data = await request.json()
                if 'tensor' in data:
                    tensor_bytes = base64.b64decode(data['tensor'])
                    buff = io.BytesIO(tensor_bytes)
                else:
                    raise ValueError("JSON missing 'tensor' field")
            except:
                # Fallback to reading body as bytes
                body = await request.read()
                buff = io.BytesIO(body)

            tensor = torch.load(buff, map_location=self.device)

            # Put in queue for the execution loop
            await self.p2p_queue.put(tensor)

            return web.Response(text="Accepted")
        except Exception as e:
            print(f"❌ [P2P] Ingress Error: {e}")
            return web.Response(status=500, text=str(e))

    # --- EXECUTION ---
    async def execute_job(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layers = msg['layers']

        print(f"⚙️ [Job {job_id}] Start. Layers: {layers[0]}-{layers[-1]}")

        try:
            # 1. JIT RESOURCE LOADING
            # Only unload if model changed
            if self.loaded_model_id != model_id:
                print(f"🧹 [Worker] New model {model_id} (was {self.loaded_model_id}). Clearing VRAM.")
                self.active_layers = []
                self.embeddings = None
                self.lm_head = None
                torch.cuda.empty_cache()
                self.loaded_model_id = model_id

            # Check layers
            if self.layer_indices != layers:
                self.active_layers = await self.loader.load_layers(model_id, layers, self.device)
                self.layer_indices = layers

            # Check Heads
            if msg['is_first'] and not self.embeddings:
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)
            if msg['is_last'] and not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            # 2. INPUT PROCESSING
            hidden_states = None

            if msg['is_first']:
                prompt = msg['input']
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    hidden_states = self.embeddings(input_ids)
            else:
                print(f"⏳ [Job {job_id}] Waiting for upstream tensor...")
                # Timeout after 120s to avoid indefinite hang
                hidden_states = await asyncio.wait_for(self.p2p_queue.get(), timeout=120.0)
                print("   ✅ Upstream received")

            # 3. COMPUTE
            with torch.no_grad():
                hidden_states = hidden_states.half()
                for layer in self.active_layers:
                    out = layer(hidden_states)
                    if isinstance(out, tuple):
                        hidden_states = out[0]
                    else:
                        hidden_states = out

            # 4. OUTPUT / FORWARD
            if msg['is_last']:
                with torch.no_grad():
                    logits = self.lm_head(hidden_states)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                    text = tokenizer.decode(next_token)

                # Send back to scheduler
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "status": "completed",
                    "output": text
                }))
                print(f"🏁 [Job {job_id}] Completed. Sent to Scheduler.")

            else:
                next_hop = msg['next_hop']
                print(f"📤 [Job {job_id}] Forwarding to {next_hop}")

                buff = io.BytesIO()
                torch.save(hidden_states, buff)
                # Base64 encode for safe transport
                b64_data = base64.b64encode(buff.getvalue()).decode('utf-8')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(next_hop, json={"tensor": b64_data}) as resp:
                        if resp.status != 200:
                            raise Exception(f"Next hop {next_hop} returned {resp.status}")

        except Exception as e:
            print(f"❌ [Job {job_id}] Failed: {e}")
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    async def run(self):
        await self.start_p2p_server()

        while True:
            try:
                print(f"🔌 Connecting to {SCHEDULER_URL}...")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    print("✅ Connected to Scheduler")

                    # Register
                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": {
                            "pubkey": "Worker_" + os.urandom(4).hex(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 4.0,
                            "p2p_url": self.get_p2p_url()
                        }
                    }))

                    # Listen
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg['type'] == 'EXECUTE':
                            await self.execute_job(msg, ws)
                        elif msg['type'] == 'PAYMENT':
                            print(f"💰 Payment: {msg['amount']} Lamports. Sig: {msg['sig']}")

            except Exception as e:
                print(f"❌ Connection Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(ProductionWorker().run())
