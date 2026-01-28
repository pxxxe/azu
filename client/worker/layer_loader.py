# client/worker/layer_loader.py
import torch
import aiohttp
import asyncio
from pathlib import Path
import json
from transformers import AutoConfig

class LayerLoader:
    """
    Downloads and loads ONLY specific layers from registry.
    Workers use this to avoid loading full models.
    """

    def __init__(self, registry_url, cache_dir="./layer_cache"):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_layers = {}

    async def download_file(self, url: str, save_path: Path):
        """Download file from registry"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Download failed: {resp.status}")

                data = await resp.read()
                save_path.write_bytes(data)
                print(f"  Downloaded {save_path.name} ({len(data) / (1024**2):.1f}MB)")

    async def get_model_config(self, model_id: str):
        """Fetch model config from registry"""
        cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_config.json"

        if not cache_path.exists():
            url = f"{self.registry_url}/download/config/{model_id}"
            await self.download_file(url, cache_path)

        with open(cache_path) as f:
            config_dict = json.load(f)

        return AutoConfig.from_dict(config_dict)

    async def download_layer(self, model_id: str, layer_idx: int):
        """Download specific layer from registry"""
        cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_layer_{layer_idx}.pt"

        if not cache_path.exists():
            url = f"{self.registry_url}/download/layer/{model_id}/{layer_idx}"
            await self.download_file(url, cache_path)

        return cache_path

    async def download_embeddings(self, model_id: str):
        """Download embedding layer"""
        cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_embeddings.pt"

        if not cache_path.exists():
            url = f"{self.registry_url}/download/embeddings/{model_id}"
            await self.download_file(url, cache_path)

        return cache_path

    async def download_lm_head(self, model_id: str):
        """Download LM head"""
        cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_lm_head.pt"

        if not cache_path.exists():
            url = f"{self.registry_url}/download/lm_head/{model_id}"
            await self.download_file(url, cache_path)

        return cache_path

    def _create_layer_module(self, config, layer_idx: int):
        """
        Create an empty layer module based on model architecture.
        This is where we instantiate the right layer class.
        """
        arch = config.architectures[0] if hasattr(config, 'architectures') else None

        if arch and "Llama" in arch:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer(config, layer_idx)

        elif arch and "Qwen" in arch:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return Qwen2DecoderLayer(config, layer_idx)

        elif arch and "Mistral" in arch:
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            return MistralDecoderLayer(config, layer_idx)

        elif arch and "GPT2" in arch:
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            return GPT2Block(config, layer_idx)

        else:
            raise ValueError(f"Unsupported architecture: {arch}. Add support in _create_layer_module()")

    async def load_layers(self, model_id: str, layer_indices: list, device="cuda"):
        """
        Load specific layers onto GPU.
        This is what workers call to get their assigned layers.

        Returns: List of loaded layer modules ready for inference
        """
        cache_key = f"{model_id}_{min(layer_indices)}_{max(layer_indices)}"

        if cache_key in self.loaded_layers:
            print(f"Using cached layers {min(layer_indices)}-{max(layer_indices)}")
            return self.loaded_layers[cache_key]

        print(f"Loading layers {min(layer_indices)}-{max(layer_indices)} for {model_id}")

        # Get config
        config = await self.get_model_config(model_id)

        # Download and load each layer
        layers = []
        for idx in layer_indices:
            # Download layer weights
            layer_path = await self.download_layer(model_id, idx)

            # Create layer module
            layer = self._create_layer_module(config, idx)

            # Load weights
            state_dict = torch.load(layer_path, map_location="cpu")
            layer.load_state_dict(state_dict)

            # Move to GPU and set to eval mode
            layer.to(device)
            layer.half()  # Use FP16
            layer.eval()

            layers.append(layer)

        self.loaded_layers[cache_key] = layers
        print(f"✅ Loaded {len(layers)} layers to {device}")

        return layers

    async def load_embeddings(self, model_id: str, device="cuda"):
        """Load embedding layer"""
        cache_key = f"{model_id}_embeddings"

        if cache_key in self.loaded_layers:
            return self.loaded_layers[cache_key]

        print(f"Loading embeddings for {model_id}")

        config = await self.get_model_config(model_id)
        emb_path = await self.download_embeddings(model_id)

        # Create embedding module
        embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        # Load weights
        state_dict = torch.load(emb_path, map_location="cpu")
        embedding.load_state_dict(state_dict)

        embedding.to(device)
        embedding.half()
        embedding.eval()

        self.loaded_layers[cache_key] = embedding
        return embedding

    async def load_lm_head(self, model_id: str, device="cuda"):
        """Load LM head"""
        cache_key = f"{model_id}_lm_head"

        if cache_key in self.loaded_layers:
            return self.loaded_layers[cache_key]

        print(f"Loading LM head for {model_id}")

        config = await self.get_model_config(model_id)
        head_path = await self.download_lm_head(model_id)

        # Create LM head module
        lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Load weights
        state_dict = torch.load(head_path, map_location="cpu")
        lm_head.load_state_dict(state_dict)

        lm_head.to(device)
        lm_head.half()
        lm_head.eval()

        self.loaded_layers[cache_key] = lm_head
        return lm_head


class LayerExecutor:
    """
    Executes inference through loaded layers.
    Workers use this to run their portion of the model.
    """

    def __init__(self, device="cuda"):
        self.device = device

    def execute_layers(self, layers: list, input_tensor: torch.Tensor):
        """
        Run input through all loaded layers.

        Args:
            layers: List of transformer layer modules
            input_tensor: Input hidden states [batch, seq_len, hidden_size]

        Returns:
            Output hidden states [batch, seq_len, hidden_size]
        """
        with torch.no_grad():
            x = input_tensor.to(self.device).half()

            for layer in layers:
                # Most transformer layers return (hidden_states, ...)
                outputs = layer(x)

                if isinstance(outputs, tuple):
                    x = outputs[0]
                else:
                    x = outputs

            return x.cpu()

    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes for network transmission"""
        import io
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Convert bytes back to tensor"""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer)
