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

SCHEDULER_URL = os.getenv("SCHEDULER_URL", "ws://localhost:8001/ws/worker")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8002")
HF_TOKEN = os.getenv("HF_TOKEN")

class MoEWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = LayerLoader(REGISTRY_URL)

        self.loaded_model_id = None

        self.dense_layers = {}
        self.moe_routers = {}
        self.moe_experts = {}
        self.moe_shared_experts = {}

        self.embeddings = None
        self.lm_head = None

        self.p2p_queue = asyncio.Queue()

        print(f"🚀 [MoE Worker] Initialized on {self.device}")

    def get_public_ip(self):
        try:
            return urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        except:
            return "127.0.0.1"

    def get_p2p_url(self):
        """Returns the Reachable URL for this worker."""
        pod_id = os.getenv("RUNPOD_POD_ID")
        if pod_id:
            url = f"https://{pod_id}-8003.proxy.runpod.net"
            print(f"🌍 [Net] RunPod Detected. Advertising Proxy: {url}")
            return url

        ip = self.get_public_ip()
        return f"http://{ip}:8003"

    async def start_p2p_server(self):
        """P2P server for receiving tensors from other workers."""
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
            data = await request.json()
            tensor_bytes = base64.b64decode(data['tensor'])
            buff = io.BytesIO(tensor_bytes)
            tensor = torch.load(buff, map_location=self.device)

            attention_mask = None
            if 'attention_mask' in data:
                mask_bytes = base64.b64decode(data['attention_mask'])
                mask_buff = io.BytesIO(mask_bytes)
                attention_mask = torch.load(mask_buff, map_location=self.device)

            await self.p2p_queue.put((tensor, attention_mask))
            return web.Response(text="Accepted")

        except Exception as e:
            print(f"❌ [P2P] Ingress Error: {e}")
            traceback.print_exc()
            return web.Response(status=500, text=str(e))

    async def execute_dense_layer(self, msg, ws):
        """Execute a regular dense transformer layer."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']

        print(f"⚙️ [Job {job_id}] Dense Layer {layer_idx}")

        try:
            if self.loaded_model_id != model_id:
                print(f"🧹 [Worker] New model {model_id}. Clearing cache.")
                self.dense_layers = {}
                self.moe_routers = {}
                self.moe_experts = {}
                self.moe_shared_experts = {}
                self.embeddings = None
                self.lm_head = None
                torch.cuda.empty_cache()
                self.loaded_model_id = model_id

            if layer_idx != -1 and layer_idx not in self.dense_layers:
                self.dense_layers[layer_idx] = await self.loader.load_dense_layer(
                    model_id, layer_idx, self.device
                )

            if msg.get('is_first') and not self.embeddings:
                self.embeddings = await self.loader.load_embeddings(model_id, self.device)
            if msg.get('is_last') and not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            hidden_states = None
            attention_mask = None

            if msg.get('is_first'):
                prompt = msg['input']
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded.input_ids.to(self.device)
                attention_mask = encoded.attention_mask.to(self.device)

                with torch.no_grad():
                    hidden_states = self.embeddings(input_ids)
            else:
                print(f"⏳ Waiting for upstream tensor...")
                hidden_states, attention_mask = await asyncio.wait_for(
                    self.p2p_queue.get(), timeout=120.0
                )

            if layer_idx != -1:
                with torch.no_grad():
                    hidden_states = hidden_states.half()
                    if attention_mask is not None:
                        attention_mask = attention_mask.half()

                    layer = self.dense_layers[layer_idx]

                    if attention_mask is not None:
                        out = layer(hidden_states, attention_mask=attention_mask)
                    else:
                        out = layer(hidden_states)

                    if isinstance(out, tuple):
                        hidden_states = out[0]
                    else:
                        hidden_states = out

            if msg.get('is_last'):
                with torch.no_grad():
                    logits = self.lm_head(hidden_states)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
                    text = tokenizer.decode(next_token, skip_special_tokens=True)

                await ws.send(json.dumps({
                    "type": "RESULT",
                    "job_id": job_id,
                    "status": "completed",
                    "output": text
                }))
                print(f"🏁 [Job {job_id}] Completed")
            else:
                next_hop = msg.get('next_hop')
                if next_hop:
                    print(f"📤 Forwarding to {next_hop}")

                    buff = io.BytesIO()
                    torch.save(hidden_states, buff)
                    payload = {"tensor": base64.b64encode(buff.getvalue()).decode('utf-8')}

                    if attention_mask is not None:
                        mask_buff = io.BytesIO()
                        torch.save(attention_mask, mask_buff)
                        payload["attention_mask"] = base64.b64encode(mask_buff.getvalue()).decode('utf-8')

                    async with aiohttp.ClientSession() as sess:
                        async with sess.post(next_hop, json=payload) as resp:
                            if resp.status != 200:
                                raise Exception(f"Next hop returned {resp.status}")

        except Exception as e:
            print(f"❌ [Job {job_id}] Failed: {e}")
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    async def execute_moe_router(self, msg, ws):
        """Run router to determine top-k experts."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        top_k = msg.get('top_k', 2)

        print(f"🎯 [Job {job_id}] MoE Router for layer {layer_idx}")

        try:
            if self.loaded_model_id != model_id:
                print(f"🧹 [Worker] New model {model_id}. Clearing cache.")
                self.dense_layers = {}
                self.moe_routers = {}
                self.moe_experts = {}
                self.moe_shared_experts = {}
                self.embeddings = None
                self.lm_head = None
                torch.cuda.empty_cache()
                self.loaded_model_id = model_id

            if layer_idx not in self.moe_routers:
                self.moe_routers[layer_idx] = await self.loader.load_moe_router(
                    model_id, layer_idx, self.device
                )

            if msg.get('has_input'):
                hidden_states_b64 = msg['hidden_states']
                hidden_states_bytes = base64.b64decode(hidden_states_b64)
                buff = io.BytesIO(hidden_states_bytes)
                hidden_states = torch.load(buff, map_location=self.device)
            else:
                print(f"⏳ Waiting for upstream tensor...")
                hidden_states, _ = await asyncio.wait_for(
                    self.p2p_queue.get(), timeout=120.0
                )

            hidden_states = hidden_states.half()

            with torch.no_grad():
                router = self.moe_routers[layer_idx]

                batch_size, seq_len, hidden_dim = hidden_states.shape
                flat_hidden = hidden_states.view(-1, hidden_dim)

                router_logits = router(flat_hidden)

                routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

                routing_weights = routing_weights.view(batch_size, seq_len, top_k)
                selected_experts = selected_experts.view(batch_size, seq_len, top_k)

            buff = io.BytesIO()
            torch.save(hidden_states, buff)
            hidden_states_b64 = base64.b64encode(buff.getvalue()).decode('utf-8')

            await ws.send(json.dumps({
                "type": "ROUTER_RESULT",
                "job_id": job_id,
                "layer_idx": layer_idx,
                "selected_experts": selected_experts.cpu().tolist(),
                "routing_weights": routing_weights.cpu().tolist(),
                "hidden_states": hidden_states_b64
            }))

            print(f"   ✅ Router complete")

        except Exception as e:
            print(f"❌ [Job {job_id}] Router failed: {e}")
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    async def execute_moe_expert(self, msg, ws):
        """Execute specific expert(s) on input."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        layer_idx = msg['layer_idx']
        expert_indices = msg['expert_indices']

        print(f"🔬 [Job {job_id}] MoE Experts {expert_indices} for layer {layer_idx}")

        try:
            if self.loaded_model_id != model_id:
                print(f"🧹 [Worker] New model {model_id}. Clearing cache.")
                self.dense_layers = {}
                self.moe_routers = {}
                self.moe_experts = {}
                self.moe_shared_experts = {}
                self.embeddings = None
                self.lm_head = None
                torch.cuda.empty_cache()
                self.loaded_model_id = model_id

            for expert_idx in expert_indices:
                if (layer_idx, expert_idx) not in self.moe_experts:
                    self.moe_experts[(layer_idx, expert_idx)] = await self.loader.load_moe_expert(
                        model_id, layer_idx, expert_idx, self.device
                    )

            if msg.get('has_input'):
                hidden_states_b64 = msg['hidden_states']
                hidden_states_bytes = base64.b64decode(hidden_states_b64)
                buff = io.BytesIO(hidden_states_bytes)
                hidden_states = torch.load(buff, map_location=self.device)
            else:
                print(f"⏳ Waiting for upstream tensor...")
                hidden_states, _ = await asyncio.wait_for(
                    self.p2p_queue.get(), timeout=120.0
                )

            hidden_states = hidden_states.half()

            expert_outputs = []
            with torch.no_grad():
                for expert_idx in expert_indices:
                    expert = self.moe_experts[(layer_idx, expert_idx)]
                    output = expert(hidden_states)
                    expert_outputs.append(output)

            outputs_encoded = []
            for output in expert_outputs:
                buff = io.BytesIO()
                torch.save(output, buff)
                outputs_encoded.append(base64.b64encode(buff.getvalue()).decode('utf-8'))

            await ws.send(json.dumps({
                "type": "EXPERT_RESULT",
                "job_id": job_id,
                "layer_idx": layer_idx,
                "expert_indices": expert_indices,
                "outputs": outputs_encoded
            }))

            print(f"   ✅ Expert execution complete")

        except Exception as e:
            print(f"❌ [Job {job_id}] Expert execution failed: {e}")
            traceback.print_exc()
            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }))

    async def execute_lm_head(self, msg, ws):
        """Execute LM head to convert hidden states to tokens."""
        job_id = msg['job_id']
        model_id = msg['model_id']
        hidden_states_b64 = msg['hidden_states']

        print(f"🏁 [Job {job_id}] LM Head execution")

        try:
            if self.loaded_model_id != model_id:
                print(f"🧹 [Worker] New model {model_id}. Clearing cache.")
                self.dense_layers = {}
                self.moe_routers = {}
                self.moe_experts = {}
                self.moe_shared_experts = {}
                self.embeddings = None
                self.lm_head = None
                torch.cuda.empty_cache()
                self.loaded_model_id = model_id

            if not self.lm_head:
                self.lm_head = await self.loader.load_lm_head(model_id, self.device)

            tensor_bytes = base64.b64decode(hidden_states_b64)
            buff = io.BytesIO(tensor_bytes)
            hidden_states = torch.load(buff, map_location=self.device)

            with torch.no_grad():
                logits = self.lm_head(hidden_states.half())
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    trust_remote_code=True
                )
                output_text = tokenizer.decode(next_token, skip_special_tokens=True)

            await ws.send(json.dumps({
                "type": "RESULT",
                "job_id": job_id,
                "status": "completed",
                "output": output_text
            }))

            print(f"🏁 [Job {job_id}] Completed: {output_text}")

        except Exception as e:
            print(f"❌ [Job {job_id}] LM head failed: {e}")
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

                    await ws.send(json.dumps({
                        "type": "REGISTER",
                        "specs": {
                            "pubkey": "Worker_" + os.urandom(4).hex(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 4.0,
                            "p2p_url": self.get_p2p_url(),
                            "capabilities": ["dense", "moe_router", "moe_expert"]
                        }
                    }))

                    async for raw in ws:
                        msg = json.loads(raw)
                        msg_type = msg.get('type')

                        if msg_type == 'EXECUTE_DENSE':
                            await self.execute_dense_layer(msg, ws)
                        elif msg_type == 'EXECUTE_MOE_ROUTER':
                            await self.execute_moe_router(msg, ws)
                        elif msg_type == 'EXECUTE_MOE_EXPERT':
                            await self.execute_moe_expert(msg, ws)
                        elif msg_type == 'EXECUTE_LM_HEAD':
                            await self.execute_lm_head(msg, ws)
                        elif msg_type == 'PAYMENT':
                            print(f"💰 Payment: {msg['amount']} Lamports. Sig: {msg['sig']}")
                        else:
                            print(f"⚠️ Unknown message type: {msg_type}")

            except Exception as e:
                print(f"❌ Connection Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(MoEWorker().run())
