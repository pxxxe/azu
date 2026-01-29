import torch
import aiohttp
import os
from pathlib import Path
from transformers import AutoConfig
# We need these imports to construct the empty shell on the GPU
# so we can pour the weights into it.
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class LayerLoader:
    def __init__(self, registry_url, cache_dir="./layer_cache"):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_layers = {} # RAM Cache

    async def _download(self, url: str, path: Path):
        """Helper to download a file from Registry to Worker Disk"""
        if path.exists(): return

        print(f"   ⬇️ Downloading {url}...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Download failed [{resp.status}]: {url}")
                data = await resp.read()

                # Atomic write
                temp = path.with_suffix('.tmp')
                with open(temp, 'wb') as f: f.write(data)
                os.rename(temp, path)

    def _get_layer_class(self, config):
        """Map architecture string to actual PyTorch Class"""
        arch = config.architectures[0]
        if "Llama" in arch: return LlamaDecoderLayer
        if "Qwen" in arch: return Qwen2DecoderLayer
        if "Mistral" in arch: return MistralDecoderLayer
        if "GPT2" in arch: return GPT2Block
        raise ValueError(f"Worker does not support architecture: {arch}")

    async def load_layers(self, model_id: str, layer_indices: list, device="cuda"):
        """
        1. Download specific shard (layer_x.pt) from Registry.
        2. Create empty Transformer Layer on GPU.
        3. Load weights into it.
        """
        # RAM Cache check
        key = f"{model_id}_{tuple(layer_indices)}"
        if key in self.loaded_layers: return self.loaded_layers[key]

        print(f"📦 Loading layers {layer_indices} for {model_id}...")

        # 1. Get Config (Tiny JSON file)
        sanitized = model_id.replace("/", "_")
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        LayerClass = self._get_layer_class(config)
        modules = []

        for idx in layer_indices:
            filename = f"layer_{idx}.pt"
            path = self.cache_dir / f"{sanitized}_{filename}"

            # 2. Download Weights
            await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

            # 3. Create Empty Shell on GPU
            layer = LayerClass(config, layer_idx=idx).to(device).half()

            # 4. Fill with Weights
            state_dict = torch.load(path, map_location=device)
            layer.load_state_dict(state_dict)
            layer.eval()
            modules.append(layer)

        self.loaded_layers[key] = modules
        return modules

    async def load_embeddings(self, model_id: str, device="cuda"):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_embeddings.pt"

        # Download
        await self._download(f"{self.registry_url}/layers/{sanitized}/embeddings.pt", path)

        # Load
        config_path = self.cache_dir / f"{sanitized}_config.json"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        emb = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
        emb.load_state_dict(torch.load(path, map_location=device))
        emb.eval()
        return emb

    async def load_lm_head(self, model_id: str, device="cuda"):
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / f"{sanitized}_lm_head.pt"

        # Download
        await self._download(f"{self.registry_url}/layers/{sanitized}/lm_head.pt", path)

        # Load
        config_path = self.cache_dir / f"{sanitized}_config.json"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device).half()
        head.load_state_dict(torch.load(path, map_location=device))
        head.eval()
        return head
