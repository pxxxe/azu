import torch
import asyncio
import websockets
import aiohttp
import json
import base64
import io
import os
import urllib.request
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
        self.input_queue = asyncio.Queue() # For P2P tensors

        # State
        self.active_layers = []
        self.embeddings = None
        self.lm_head = None

        print(f"🚀 Worker Initialized on {self.device}")

    def get_public_ip(self):
        """Determines external IP for P2P connection"""
        try:
            return urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        except:
            print("⚠️ Could not resolve Public IP, using localhost")
            return "127.0.0.1"

    # --- P2P SERVER ---
    async def start_p2p_server(self, port=8003):
        app = web.Application()
        # Accept POST requests with binary tensors
        app.router.add_post('/tensor_in', self.handle_tensor_ingress)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        print(f"👂 P2P Tensor Listener active on port {port}")

    async def handle_tensor_ingress(self, request):
        """Receives tensor from previous worker"""
        try:
            # Read binary body directly (more efficient than JSON wrapper for large tensors)
            data = await request.read()

            # If wrapped in JSON/base64 (Scheduler might enforce this format)
            # Let's support the JSON format used in the test harness
            try:
                json_data = await request.json()
                if 'tensor' in json_data:
                    tensor_bytes = base64.b64decode(json_data['tensor'])
                    buffer = io.BytesIO(tensor_bytes)
                    tensor = torch.load(buffer, map_location=self.device)
                    await self.input_queue.put(tensor)
                    return web.Response(text="Accepted")
            except:
                pass

            # Fallback: Raw binary
            buffer = io.BytesIO(data)
            tensor = torch.load(buffer, map_location=self.device)
            await self.input_queue.put(tensor)
            return web.Response(text="Accepted")

        except Exception as e:
            print(f"❌ Error receiving tensor: {e}")
            return web.Response(status=500, text=str(e))

    # --- INFERENCE LOGIC ---
    async def process_job(self, job, ws):
        job_id = job['id']
        model_id = job['model']
        layers = job['assigned_layers']
        is_first = job.get('is_first', False)
        is_last = job.get('is_last', False)
        next_worker_url = job.get('next_worker')

        print(f"⚙️ Processing Job {job_id} | Layers {layers}")

        try:
            # 1. Load Resources (skip if already loaded from PRELOAD)
            if not self.active_layers:
                self.active_layers = await self.loader.load_layers(model_id, layers, self.device)

            if is_first and not self.embeddings:
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)
            if is_last and not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            # 2. Prepare Input
            hidden_states = None

            if is_first:
                # Tokenize (Using HF Tokenizer)
                prompt = job['input']
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

                # Embedding Lookup
                with torch.no_grad():
                    hidden_states = self.embeddings(input_ids)
            else:
                # Wait for tensor from P2P
                print(f"⏳ Waiting for tensor input...")
                hidden_states = await self.input_queue.get()
                print("   ✅ Input received")

            # 3. Execute Layers
            # Real Torch Compute
            with torch.no_grad():
                # Ensure half precision
                hidden_states = hidden_states.half()

                for i, layer in enumerate(self.active_layers):
                    # Transformer layers usually return tuple (hidden_states, ...)
                    out = layer(hidden_states)
                    if isinstance(out, tuple):
                        hidden_states = out[0]
                    else:
                        hidden_states = out

            # 4. Handle Output
            if is_last:
                # LM Head -> Logits -> Text
                with torch.no_grad():
                    logits = self.lm_head(hidden_states)
                    # Simple Greedy Decode for now
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)

                    # We need the tokenizer again to decode
                    # (In efficient systems, we'd cache this or send tokens back)
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                    output_text = tokenizer.decode(next_token)

                # Send Final Result to Scheduler
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "output": output_text,
                    "status": "completed"
                }))
                print(f"🏁 Job {job_id} Completed. Output: {output_text}")

            else:
                # Send to Next Worker
                if not next_worker_url:
                    raise Exception("Not last worker, but no next_worker_url provided")

                print(f"📤 Sending tensor to {next_worker_url}")

                # Serialize
                buffer = io.BytesIO()
                torch.save(hidden_states, buffer)
                tensor_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                async with aiohttp.ClientSession() as sess:
                    async with sess.post(next_worker_url, json={"tensor": tensor_b64}) as resp:
                        if resp.status != 200:
                            raise Exception(f"Next worker rejected tensor: {resp.status}")

                # Notify Scheduler we are done with our part
                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "status": "partial_complete"
                }))

        except Exception as e:
            print(f"❌ Job Execution Failed: {e}")
            import traceback
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    # --- MAIN LOOP ---
    async def run(self):
        # 1. Start P2P
        p2p_port = 8003
        await self.start_p2p_server(p2p_port)

        # 2. Connect to Scheduler
        retry_wait = 2
        while True:
            try:
                print(f"🔌 Connecting to Scheduler: {SCHEDULER_URL}")
                async with websockets.connect(SCHEDULER_URL) as ws:
                    print("✅ Connected")

                    # Register
                    specs = {
                        "pubkey": "Worker_" + os.urandom(4).hex(), # Mock Pubkey for now
                        "gpu": torch.cuda.get_device_name(0),
                        "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                        "public_ip": self.get_public_ip(),
                        "p2p_port": p2p_port
                    }

                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": specs
                    }))

                    # Msg Loop
                    async for msg in ws:
                        data = json.loads(msg)
                        msg_type = data.get('type')

                        if msg_type == 'PRELOAD':
                            # Scheduler tells us to preload layers
                            model_id = data['model_id']
                            layers = data['layers']

                            print(f"📦 Preloading {model_id} layers {layers[0]}-{layers[-1]}...")

                            # Download and load to GPU
                            self.active_layers = await self.loader.load_layers(model_id, layers, self.device)

                            # Load embeddings if first worker
                            if data.get('is_first'):
                                self.embeddings = await self.loader.load_embeddings(model_id, self.device)

                            # Load LM head if last worker
                            if data.get('is_last'):
                                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

                            # Broadcast ready
                            await ws.send(json.dumps({
                                "type": "READY",
                                "model_id": model_id,
                                "layers": layers
                            }))

                            print(f"✅ Ready with {len(self.active_layers)} layers")

                        elif msg_type == 'EXECUTE':
                            # Job execution - layers already loaded from PRELOAD
                            asyncio.create_task(self.process_job(data['job'], ws))

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
