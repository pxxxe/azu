import torch
import aiohttp
import os
import sys
import json
import asyncio  # <--- ADDED
from pathlib import Path
from transformers import AutoConfig

class LayerLoader:
    def __init__(self, registry_url, cache_dir="./layer_cache"):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_cache = {}  # RAM cache
        # 32 parallel downloads is enough to saturate a standard container NIC without killing the Registry.
        self.semaphore = asyncio.Semaphore(32)

    async def _download(self, url: str, path: Path):
        """Helper to download a file from Registry to Worker Disk (STREAMING)"""
        # Acquire semaphore to throttle the thundering herd
        async with self.semaphore:
            if path.exists():
                # Simple check for empty files from failed runs
                if path.stat().st_size > 0:
                    print(f"   ✓ Using cached {path.name}")
                    sys.stdout.flush()
                    return
                else:
                    print(f"   ⚠️ Found 0-byte file {path.name}, removing...")
                    sys.stdout.flush()
                    os.remove(path)

            print(f"   ⬇️ Downloading {url}...")
            sys.stdout.flush()

            # Timeout: Connect fast (10s), but allow time for large file transfer (600s)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)
            temp = path.with_suffix('.tmp')

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise Exception(f"Download failed [{resp.status}]: {url}")

                        # STREAMING DOWNLOAD (Chunked)
                        with open(temp, 'wb') as f:
                            downloaded = 0
                            async for chunk in resp.content.iter_chunked(1024 * 1024): # 1MB chunks
                                f.write(chunk)
                                downloaded += len(chunk)
                                # Print progress every 50MB
                                if downloaded % (50 * 1024 * 1024) == 0:
                                    print(f"      ...{downloaded / (1024*1024):.0f}MB downloaded...")
                                    sys.stdout.flush()

                # Atomic Move
                os.rename(temp, path)
                print(f"   ✅ Downloaded {path.name} ({downloaded / (1024*1024):.2f} MB)")
                sys.stdout.flush()

            except Exception as e:
                print(f"   ❌ Error downloading {url}: {e}")
                sys.stdout.flush()
                if os.path.exists(temp):
                    os.remove(temp)
                raise e

    def _get_layer_class(self, config):
        """
        Map architecture string to actual PyTorch Layer Class.
        Uses dynamic imports to support ALL architectures without hardcoding.
        """
        arch = config.architectures[0]

        # Common architecture -> (module_name, class_name) mappings
        arch_map = {
            "LlamaForCausalLM": ("llama", "LlamaDecoderLayer"),
            "MistralForCausalLM": ("mistral", "MistralDecoderLayer"),
            "MixtralForCausalLM": ("mixtral", "MixtralDecoderLayer"),
            "Qwen2ForCausalLM": ("qwen2", "Qwen2DecoderLayer"),
            "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeDecoderLayer"),
            "GPT2LMHeadModel": ("gpt2", "GPT2Block"),
            "GPTNeoForCausalLM": ("gpt_neo", "GPTNeoBlock"),
            "GPTJForCausalLM": ("gptj", "GPTJBlock"),
            "OPTForCausalLM": ("opt", "OPTDecoderLayer"),
            "BloomForCausalLM": ("bloom", "BloomBlock"),
            "FalconForCausalLM": ("falcon", "FalconDecoderLayer"),
            "MPTForCausalLM": ("mpt", "MPTBlock"),
            "PhiForCausalLM": ("phi", "PhiDecoderLayer"),
            "Phi3ForCausalLM": ("phi3", "Phi3DecoderLayer"),
            "GemmaForCausalLM": ("gemma", "GemmaDecoderLayer"),
            "Gemma2ForCausalLM": ("gemma2", "Gemma2DecoderLayer"),
            "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2DecoderLayer"),
            "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2DecoderLayer"),
        }

        if arch in arch_map:
            module_name, class_name = arch_map[arch]
            try:
                # Dynamic import: from transformers.models.{module_name}.modeling_{module_name} import {class_name}
                import importlib
                full_module = f"transformers.models.{module_name}.modeling_{module_name}"
                mod = importlib.import_module(full_module)
                return getattr(mod, class_name)
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Failed to import {class_name} from {full_module}: {e}")
                sys.stdout.flush()
                # Fall through to generic fallback

        # FALLBACK: Try generic pattern matching for unknown architectures
        # Most architectures follow the pattern: XForCausalLM -> XDecoderLayer
        try:
            # Extract base name (e.g., "Llama" from "LlamaForCausalLM")
            if arch.endswith("ForCausalLM"):
                base = arch[:-len("ForCausalLM")]
            elif arch.endswith("LMHeadModel"):
                base = arch[:-len("LMHeadModel")]
            else:
                base = arch

            # Try to import dynamically
            module_name = base.lower()
            class_name = f"{base}DecoderLayer"

            import importlib
            full_module = f"transformers.models.{module_name}.modeling_{module_name}"
            # print(f"   🔍 Attempting generic import: {full_module}.{class_name}")
            mod = importlib.import_module(full_module)
            layer_class = getattr(mod, class_name)
            # print(f"   ✅ Successfully loaded {class_name} via generic pattern")
            return layer_class
        except Exception as e:
            print(f"   ❌ Generic import failed: {e}")
            sys.stdout.flush()

        # LAST RESORT: Return error with helpful message
        raise ValueError(
            f"Worker does not support architecture: {arch}\n"
            f"To add support, update the arch_map in layer_loader.py with:\n"
            f'  "{arch}": ("<module_name>", "<LayerClassName>")'
        )

    async def load_dense_layer(self, model_id: str, layer_idx: int, device="cuda"):
        """Load a regular dense transformer layer."""
        cache_key = f"{model_id}:dense:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        print(f"📦 Loading dense layer {layer_idx} for {model_id}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # Get config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Download dense layer
        filename = f"layer_{layer_idx}_dense.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        print(f"   🔧 Creating layer class and loading weights...")
        sys.stdout.flush()

        # Create layer on GPU
        LayerClass = self._get_layer_class(config)
        layer = LayerClass(config, layer_idx=layer_idx).to(device).half()

        # Load weights - strict=False for robustness
        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)
        # print(f"   ✓ Loaded state dict, applying to model...")
        # sys.stdout.flush()
        layer.load_state_dict(state_dict, strict=False)
        layer.eval()

        print(f"   ✅ Dense layer {layer_idx} ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = layer
        return layer

    async def load_moe_router(self, model_id: str, layer_idx: int, device="cuda"):
        """Load the router/gate for an MoE layer."""
        cache_key = f"{model_id}:router:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        print(f"📦 Loading router for MoE layer {layer_idx}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # Get config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Download router
        filename = f"layer_{layer_idx}_router.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # Create router (simple Linear layer for most MoE)
        # For Qwen2-MoE and Mixtral: it's a Linear(hidden_size, num_experts)
        num_experts = getattr(config, 'num_local_experts', 8)  # Default to 8
        hidden_size = config.hidden_size

        print(f"   🔧 Creating router ({hidden_size} -> {num_experts})...")
        sys.stdout.flush()

        router = torch.nn.Linear(hidden_size, num_experts, bias=False).to(device).half()

        # Load weights - strict=False
        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)
        # print(f"   ✓ Loaded state dict, applying to model...")
        # sys.stdout.flush()
        router.load_state_dict(state_dict, strict=False)
        router.eval()

        print(f"   ✅ Router ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = router
        return router

    async def load_moe_expert(self, model_id: str, layer_idx: int, expert_idx: int, device="cuda"):
        """Load a specific expert from an MoE layer."""
        cache_key = f"{model_id}:expert:{layer_idx}:{expert_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        print(f"📦 Loading expert {expert_idx} from MoE layer {layer_idx}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # Get config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Download expert
        filename = f"layer_{layer_idx}_expert_{expert_idx}.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # Create expert FFN
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        print(f"   🔧 Creating expert FFN...")
        sys.stdout.flush()

        class ExpertFFN(torch.nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
                self.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.act_fn = torch.nn.SiLU()

            def forward(self, x):
                return self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        expert = ExpertFFN(hidden_size, intermediate_size).to(device).half()

        # Load weights - strict=False
        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)
        # print(f"   ✓ Loaded state dict, applying to model...")
        # sys.stdout.flush()
        expert.load_state_dict(state_dict, strict=False)
        expert.eval()

        print(f"   ✅ Expert {expert_idx} ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = expert
        return expert

    async def load_moe_shared_expert(self, model_id: str, layer_idx: int, device="cuda"):
        """Load the shared expert FFN (Qwen2-MoE specific)."""
        cache_key = f"{model_id}:shared_expert:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        print(f"📦 Loading shared expert for MoE layer {layer_idx}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # Get config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Download shared expert
        filename = f"layer_{layer_idx}_shared_expert.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # Create shared expert FFN
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "shared_expert_intermediate_size", config.intermediate_size)

        print(f"   🔧 Creating shared expert FFN...")
        sys.stdout.flush()

        class SharedExpertFFN(torch.nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
                self.act_fn = torch.nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        shared_expert = SharedExpertFFN(hidden_size, intermediate_size).to(device).half()

        # Load weights - strict=False
        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)
        # print(f"   ✓ Loaded state dict, applying to model...")
        # sys.stdout.flush()
        shared_expert.load_state_dict(state_dict, strict=False)
        shared_expert.eval()

        print(f"   ✅ Shared expert ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = shared_expert
        return shared_expert

    async def load_embeddings(self, model_id: str, device="cuda"):
        cache_key = f"{model_id}:embeddings"
        if cache_key in self.loaded_cache:
            print(f"   ✓ Using cached embeddings")
            sys.stdout.flush()
            return self.loaded_cache[cache_key]

        print(f"📦 Loading embeddings for {model_id}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # CRITICAL FIX: Download config FIRST before embeddings
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Now download embeddings
        path = self.cache_dir / f"{sanitized}_embeddings.pt"
        await self._download(f"{self.registry_url}/layers/{sanitized}/embeddings.pt", path)

        print(f"   🔧 Creating embedding layer (vocab={config.vocab_size}, dim={config.hidden_size})...")
        sys.stdout.flush()

        emb = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()

        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)

        # print(f"   ✓ Loaded state dict ({len(state_dict)} keys), applying to model...")
        # sys.stdout.flush()
        emb.load_state_dict(state_dict)
        emb.eval()

        print(f"   ✅ Embeddings ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = emb
        return emb

    async def load_lm_head(self, model_id: str, device="cuda"):
        cache_key = f"{model_id}:lm_head"
        if cache_key in self.loaded_cache:
            print(f"   ✓ Using cached LM head")
            sys.stdout.flush()
            return self.loaded_cache[cache_key]

        print(f"📦 Loading LM head for {model_id}...")
        sys.stdout.flush()

        sanitized = model_id.replace("/", "_")

        # Download config first
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)

        # print(f"   📋 Loading config from {config_path}...")
        # sys.stdout.flush()
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Now download LM head
        path = self.cache_dir / f"{sanitized}_lm_head.pt"
        await self._download(f"{self.registry_url}/layers/{sanitized}/lm_head.pt", path)

        print(f"   🔧 Creating LM head (dim={config.hidden_size} -> vocab={config.vocab_size})...")
        sys.stdout.flush()

        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device).half()

        # print(f"   📥 torch.load({path.name})...")
        # sys.stdout.flush()
        state_dict = torch.load(path, map_location=device)

        # print(f"   ✓ Loaded state dict ({len(state_dict)} keys), applying to model...")
        # sys.stdout.flush()
        head.load_state_dict(state_dict)
        head.eval()

        print(f"   ✅ LM head ready")
        sys.stdout.flush()

        self.loaded_cache[cache_key] = head
        return head
