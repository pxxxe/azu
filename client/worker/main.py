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

# Load Env
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

class ProductionWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)

        # State
        self.loaded_model_id = None
        self.active_layers = []  # List of torch modules
        self.layer_indices = []  # List of ints
        self.embeddings = None
        self.lm_head = None

        # Concurrency & P2P
        self.p2p_queue = asyncio.Queue()
        self.current_job_id = None

        print(f"🚀 [Worker] Initialized on {self.device}")

    def get_public_ip(self):
        try:
            return urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        except:
            return "127.0.0.1"

    def get_p2p_url(self):
        """
        Determine the external URL for other workers to reach this one.
        Prioritizes RunPod Proxy if available to ensure connectivity behind NAT.
        """
        # 1. RunPod Environment (Env var is injected by RunPod)
        if os.getenv("RUNPOD_POD_ID"):
            pod_id = os.getenv("RUNPOD_POD_ID")
            # RunPod proxy uses HTTPS for the exposed port
            url = f"https://{pod_id}-8003.proxy.runpod.net"
            print(f"🌍 [Net] Detected RunPod. Advertising: {url}")
            return url

        # 2. Local/Direct Environment
        ip = self.get_public_ip()
        return f"http://{ip}:8003"

    # --- P2P SERVER ---
    async def start_p2p_server(self, port=8003):
        app = web.Application()
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        print(f"👂 [P2P] Listener active on port {port}")

    async def handle_tensor_ingress(self, request):
        """Receives tensor/hidden_states from the previous worker in the chain."""
        try:
            # We accept the connection immediately.
            # If the worker is busy loading layers, this data sits in the queue.

            # Support both raw binary and JSON wrapper
            try:
                json_data = await request.json()
                if 'tensor' in json_data:
                    tensor_bytes = base64.b64decode(json_data['tensor'])
                    buffer = io.BytesIO(tensor_bytes)
                else:
                    raise ValueError("Invalid JSON")
            except:
                # Raw binary fallback
                data = await request.read()
                buffer = io.BytesIO(data)

            tensor = torch.load(buffer, map_location=self.device)

            # Push to queue to be picked up by execute_job
            await self.p2p_queue.put(tensor)

            return web.Response(text="Accepted")

        except Exception as e:
            print(f"❌ [P2P] Error receiving tensor: {e}")
            return web.Response(status=500, text=str(e))

    # --- EXECUTION LOGIC ---

    async def ensure_resources(self, model_id: str, layers: list, is_first: bool, is_last: bool):
        """JIT Resource Loading. Smart diffing."""

        # Check if we need to flush VRAM
        if self.loaded_model_id != model_id:
            print(f"🧹 [Worker] Switching models: {self.loaded_model_id} -> {model_id}")
            self.active_layers = []
            self.embeddings = None
            self.lm_head = None
            torch.cuda.empty_cache()
            self.loaded_model_id = model_id

        # Check if layers match
        if self.layer_indices != layers:
            print(f"📦 [Worker] Loading layers {layers}...")
            self.active_layers = await self.loader.load_layers(model_id, layers, self.device)
            self.layer_indices = layers

        # Check Embeddings (Node 0)
        if is_first and self.embeddings is None:
            print(f"📦 [Worker] Loading Embeddings...")
            self.embeddings = await self.loader.load_embeddings(model_id, self.device)

        # Check Head (Last Node)
        if is_last and self.lm_head is None:
            print(f"📦 [Worker] Loading LM Head...")
            self.lm_head = await self.loader.load_lm_head(model_id, self.device)

    async def execute_job(self, msg, ws):
        job_id = msg['job_id']
        model_id = msg['model_id']
        layers = msg['layers']
        is_first = msg['is_first']
        is_last = msg['is_last']
        next_hop = msg.get('next_hop') # URL of next worker

        print(f"⚙️ [Job {job_id}] Start. Layers: {layers[0]}-{layers[-1]}")
        self.current_job_id = job_id

        try:
            # 1. Prepare Resources (May take time downloading)
            await self.ensure_resources(model_id, layers, is_first, is_last)

            # 2. Get Input
            hidden_states = None

            if is_first:
                # Tokenize Prompt
                prompt = msg['input']
                # Note: In a real optimized system, tokenizer would be loaded once or passed via API.
                # For now, we load from HF or Cache.
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

                with torch.no_grad():
                    hidden_states = self.embeddings(input_ids)
            else:
                # Wait for P2P
                print(f"⏳ [Job {job_id}] Waiting for upstream tensor...")
                hidden_states = await self.p2p_queue.get() # Blocks until data arrives
                print(f"   ✅ [Job {job_id}] Upstream data received")

            # 3. Compute
            with torch.no_grad():
                hidden_states = hidden_states.half()
                for layer in self.active_layers:
                    out = layer(hidden_states)
                    if isinstance(out, tuple): hidden_states = out[0]
                    else: hidden_states = out

            # 4. Output or Forward
            if is_last:
                # Decode
                with torch.no_grad():
                    logits = self.lm_head(hidden_states)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                    output_text = tokenizer.decode(next_token)

                # Report to Scheduler
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "status": "completed",
                    "output": output_text
                }))
                print(f"🏁 [Job {job_id}] Complete. Output sent to Scheduler.")

            else:
                # Forward to Next Worker
                if not next_hop:
                    raise Exception("Topology Error: Not last node, but no next_hop")

                print(f"📤 [Job {job_id}] Forwarding to {next_hop}...")

                buffer = io.BytesIO()
                torch.save(hidden_states, buffer)
                # Send raw binary is faster, but let's use the JSON wrapper for compatibility with the ingress handler above
                tensor_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(next_hop, json={"tensor": tensor_b64}) as resp:
                        if resp.status != 200:
                            raise Exception(f"Next hop rejected data: {resp.status}")

                # Tell Scheduler we are done
                # (Optional, keeps connection alive)
                # await ws.send(json.dumps({"type": "PARTIAL", "job_id": job_id}))

        except Exception as e:
            print(f"❌ [Job {job_id}] Execution Failed: {e}")
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    # --- MAIN LOOP ---
    async def run(self):
        p2p_port = 8003
        await self.start_p2p_server(p2p_port)

        retry_wait = 2
        while True:
            try:
                print(f"🔌 Connecting to Scheduler: {SCHEDULER_URL}")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    print("✅ Connected")

                    # Register
                    # Generate a stable ID based on hardware? For now random is fine per session.
                    specs = {
                        "pubkey": "Worker_" + os.urandom(4).hex(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                        "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 4.0,
                        "p2p_url": self.get_p2p_url()
                    }

                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": specs
                    }))

                    # Loop
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        mtype = msg.get('type')

                        if mtype == 'EXECUTE':
                            # Fire and forget? No, we await to keep flow control simple for now
                            # In real prod, this would spawn a task, but GPU is single-stream anyway
                            await self.execute_job(msg, ws)

                        elif mtype == 'PAYMENT':
                            print(f"💰 Payment Received: {msg['amount']} Lamports. Sig: {msg['sig']}")

            except Exception as e:
                print(f"❌ Connection Lost: {e}")
                await asyncio.sleep(retry_wait)
                retry_wait = min(retry_wait * 2, 30)

if __name__ == "__main__":
    worker = ProductionWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
