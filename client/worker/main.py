# client/worker/main.py
import asyncio
import json
import torch
import aiohttp
import os
import base64
from solders.keypair import Keypair
from layer_loader import LayerLoader, LayerExecutor

# CONFIG
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")

class GPUWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(registry_url=REGISTRY_URL)
        self.executor = LayerExecutor(device=self.device)

        # What layers we have loaded
        self.assigned_layers = {}  # {model_id: {"layers": [module, ...], "indices": [0,1,2,...]}}

        # Load Solana Wallet
        try:
            with open(os.path.expanduser("~/.config/solana/id.json")) as f:
                self.keypair = Keypair.from_bytes(bytes(json.load(f)))
        except:
            print("⚠️  No wallet found, generating temp one")
            self.keypair = Keypair()

        # Get GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_name = "CPU"
            self.vram_gb = 4.0  # Assume 4GB for CPU

    async def register_with_registry(self, model_id: str):
        """
        Ask registry what layers we should load based on our VRAM.
        Then download and load those layers.
        """
        print(f"📋 Requesting layer assignment for {model_id}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{REGISTRY_URL}/workers/assign_layers",
                json={
                    "model_id": model_id,
                    "vram_gb": self.vram_gb
                }
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Registry assignment failed: {resp.status}")

                assignment = await resp.json()

        layer_indices = assignment['assigned_layers']
        print(f"✅ Assigned layers: {min(layer_indices)}-{max(layer_indices)}")

        # Download and load layers
        layers = await self.loader.load_layers(model_id, layer_indices, self.device)

        self.assigned_layers[model_id] = {
            "layers": layers,
            "indices": layer_indices
        }

        return layer_indices

    async def execute_job(self, job):
        """
        Execute our assigned layers on the input tensor.
        """
        model_id = job['model']

        # Make sure we have this model loaded
        if model_id not in self.assigned_layers:
            await self.register_with_registry(model_id)

        # Get input tensor
        if 'input_tensor_b64' in job:
            # Continuing from previous worker
            input_data = base64.b64decode(job['input_tensor_b64'])
            input_tensor = self.executor.deserialize_tensor(input_data)
        else:
            # We're the first worker - need to tokenize and embed
            # For now, assume input_tensor is provided or we fail
            raise ValueError("First worker must receive pre-embedded input")

        # Execute our layers
        layers = self.assigned_layers[model_id]['layers']
        output_tensor = self.executor.execute_layers(layers, input_tensor)

        # Serialize output
        output_bytes = self.executor.serialize_tensor(output_tensor)
        output_b64 = base64.b64encode(output_bytes).decode()

        return {
            "job_id": job['id'],
            "output_tensor_b64": output_b64,
            "layer_range": self.assigned_layers[model_id]['indices'],
            "worker_id": str(self.keypair.pubkey())
        }

    async def start(self):
        """Connect to scheduler and start processing jobs"""
        print(f"🚀 Starting worker")
        print(f"  GPU: {self.gpu_name}")
        print(f"  VRAM: {self.vram_gb:.1f}GB")
        print(f"  Wallet: {self.keypair.pubkey()}")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(SCHEDULER_URL) as ws:

                # 1. Register with scheduler
                print(f"🔐 Connecting to scheduler...")
                await ws.send_json({
                    "type": "REGISTER",
                    "specs": {
                        "pubkey": str(self.keypair.pubkey()),
                        "gpu": self.gpu_name,
                        "vram_gb": self.vram_gb
                    }
                })

                print("✅ Connected! Waiting for jobs...")

                # 2. Work loop
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)

                        if data['type'] == 'EXECUTE':
                            job = data['job']
                            print(f"⚡ Executing Job {job['id']} ({job['model']})")

                            try:
                                result = await self.execute_job(job)

                                # Send result back
                                await ws.send_json({
                                    "type": "RESULT",
                                    **result,
                                    "cost": job.get('cost', 0)
                                })

                                print(f"✅ Job {job['id']} complete")

                            except Exception as e:
                                print(f"❌ Job {job['id']} failed: {e}")
                                await ws.send_json({
                                    "type": "ERROR",
                                    "job_id": job['id'],
                                    "error": str(e)
                                })

                        elif data['type'] == 'PAYMENT':
                            print(f"💰 Received payment: {data['amount']} lamports")
                            print(f"   Tx: {data['sig']}")

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"❌ WebSocket error: {ws.exception()}")
                        break

async def main():
    worker = GPUWorker()

    while True:
        try:
            await worker.start()
        except Exception as e:
            print(f"❌ Worker crashed: {e}")
            print("🔄 Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
