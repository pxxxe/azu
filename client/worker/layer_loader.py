import torch
import aiohttp
import os
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
from transformers import AutoConfig

class LayerLoader:
    def __init__(self, registry_url, cache_dir="./layer_cache"):
        self.registry_url = registry_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_cache = {}  # RAM cache

        # --- CONCURRENCY CONTROL ---
        # 1. ThreadPool for blocking torch.load (CPU bound + Disk Read)
        #    prevents the event loop from freezing during 500MB+ loads.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        # 2. Semaphore for Network/Disk Write concurrency
        #    prevents opening too many simultaneous connections
        self.sem = asyncio.Semaphore(10)

        # 3. Shared Session for Connection Pooling (Keep-Alive)
        self.session = None
        self.download_locks: dict[str, asyncio.Lock] = {}

    async def _get_session(self):
        """Lazy-load the shared session with high connection limits."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    async def _load_weights_safe(self, path):
        """
        Offload torch.load to a thread.
        Critical for preventing the P2P heartbeat from dying while loading huge files.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: torch.load(path, map_location="cpu")
        )

    async def _download(self, url: str, path: Path):
        """Helper to download a file from Registry to Worker Disk (STREAMING & NON-BLOCKING)"""

        # 1. Per-file locking to prevent duplicate downloads of the same shard
        lock_key = str(path)
        if lock_key not in self.download_locks:
            self.download_locks[lock_key] = asyncio.Lock()

        async with self.download_locks[lock_key]:
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Using cached {path.name}")
                sys.stdout.flush()
                return

            # 2. Global concurrency semaphore
            async with self.sem:
                print(f"   ⬇️ Downloading {url}...")
                sys.stdout.flush()

                temp = path.with_suffix('.tmp')

                try:
                    session = await self._get_session()
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise Exception(f"Download failed [{resp.status}]: {url}")

                        # 3. STREAMING DOWNLOAD (Chunked + Yield)
                        with open(temp, 'wb') as f:
                            downloaded = 0
                            # CRITICAL FIX: 4MB chunks + sleep(0) prevents event loop starvation
                            async for chunk in resp.content.iter_chunked(4 * 1024 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)
                                await asyncio.sleep(0) # Yield to event loop

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
        """Map architecture string to actual PyTorch Layer Class using dynamic imports."""
        arch = config.architectures[0]

        # Generic fallback pattern: XForCausalLM -> XDecoderLayer
        try:
            if arch.endswith("ForCausalLM"):
                base = arch[:-len("ForCausalLM")]
            elif arch.endswith("LMHeadModel"):
                base = arch[:-len("LMHeadModel")]
            else:
                base = arch

            module_name = base.lower()
            class_name = f"{base}DecoderLayer"

            # Mixtral edge case
            if "Mixtral" in arch:
                class_name = "MixtralDecoderLayer"

            import importlib
            full_module = f"transformers.models.{module_name}.modeling_{module_name}"
            mod = importlib.import_module(full_module)
            return getattr(mod, class_name)
        except Exception as e:
            raise ValueError(f"Could not load layer class for {arch}: {e}")

    async def load_dense_layer(self, model_id: str, layer_idx: int, device="cuda"):
        """Load a regular dense transformer layer."""
        cache_key = f"{model_id}:dense:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        sanitized = model_id.replace("/", "_")

        # 1. Config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # 2. Weights
        filename = f"layer_{layer_idx}_dense.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # 3. Create & Load (Offloaded)
        LayerClass = self._get_layer_class(config)
        layer = LayerClass(config, layer_idx=layer_idx).to(device).half()

        state_dict = await self._load_weights_safe(path)
        layer.load_state_dict(state_dict, strict=False)
        layer.eval()

        self.loaded_cache[cache_key] = layer
        return layer

    async def load_moe_router(self, model_id: str, layer_idx: int, device="cuda"):
        """Load the router/gate for an MoE layer."""
        cache_key = f"{model_id}:router:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        sanitized = model_id.replace("/", "_")

        # 1. Config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # 2. Weights
        filename = f"layer_{layer_idx}_router.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # 3. Create & Load
        num_experts = getattr(config, 'num_local_experts', 8)
        router = torch.nn.Linear(config.hidden_size, num_experts, bias=False).to(device).half()

        state_dict = await self._load_weights_safe(path)
        router.load_state_dict(state_dict, strict=False)
        router.eval()

        self.loaded_cache[cache_key] = router
        return router

    async def load_moe_expert(self, model_id: str, layer_idx: int, expert_idx: int, device="cuda"):
        """Load a specific expert from an MoE layer."""
        cache_key = f"{model_id}:expert:{layer_idx}:{expert_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        sanitized = model_id.replace("/", "_")

        # 1. Config
        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # 2. Weights
        filename = f"layer_{layer_idx}_expert_{expert_idx}.pt"
        path = self.cache_dir / f"{sanitized}_{filename}"
        await self._download(f"{self.registry_url}/layers/{sanitized}/{filename}", path)

        # 3. Create & Load
        # Basic MLP Expert (Mixtral style)
        class ExpertFFN(torch.nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
                self.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.act_fn = torch.nn.SiLU()

            def forward(self, x):
                return self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        expert = ExpertFFN(config.hidden_size, config.intermediate_size).to(device).half()

        state_dict = await self._load_weights_safe(path)
        expert.load_state_dict(state_dict, strict=False)
        expert.eval()

        self.loaded_cache[cache_key] = expert
        return expert

    async def load_embeddings(self, model_id: str, device="cuda"):
        cache_key = f"{model_id}:embeddings"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        sanitized = model_id.replace("/", "_")

        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        path = self.cache_dir / f"{sanitized}_embeddings.pt"
        await self._download(f"{self.registry_url}/layers/{sanitized}/embeddings.pt", path)

        emb = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(device).half()
        state_dict = await self._load_weights_safe(path)
        emb.load_state_dict(state_dict)
        emb.eval()

        self.loaded_cache[cache_key] = emb
        return emb

    async def load_lm_head(self, model_id: str, device="cuda"):
        cache_key = f"{model_id}:lm_head"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        sanitized = model_id.replace("/", "_")

        config_path = self.cache_dir / f"{sanitized}_config.json"
        await self._download(f"{self.registry_url}/layers/{sanitized}/config.json", config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        path = self.cache_dir / f"{sanitized}_lm_head.pt"
        await self._download(f"{self.registry_url}/layers/{sanitized}/lm_head.pt", path)

        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device).half()
        state_dict = await self._load_weights_safe(path)
        head.load_state_dict(state_dict)
        head.eval()

        self.loaded_cache[cache_key] = head
        return head
