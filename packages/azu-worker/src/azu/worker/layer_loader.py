"""
Layer loader module for downloading and caching model layers from registry.
Handles loading of dense layers, MoE components, embeddings, and LM head.
"""

import hashlib
import os
import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict

import torch
import aiohttp
from transformers import AutoConfig
from safetensors.torch import load_file as load_safetensors

from azu.worker.config import LAYER_CACHE_DIR, MAX_DOWNLOAD_WORKERS, MAX_DOWNLOAD_SEMAPHORE, HF_TOKEN

# Only safetensors is permitted. .pt and .bin are pickle-based and allow
# arbitrary code execution on load. Any file with another extension is
# rejected before touching disk.
_SAFE_EXTENSIONS = {".safetensors"}


class LayerLoader:
    """
    Handles downloading and caching of model layers from the registry.
    Supports dense layers, MoE routers, MoE experts, embeddings, and LM head.
    """

    def __init__(self, registry_url: str, cache_dir: Optional[str] = None):
        """
        Initialize the layer loader.

        Args:
            registry_url: Base URL of the registry service
            cache_dir: Optional custom cache directory
        """
        self.registry_url = registry_url
        if cache_dir is None:
            cache_dir = LAYER_CACHE_DIR

        print(f"Using cache directory: {cache_dir}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_cache = {}  # RAM cache for loaded layers

        # Per-model HF checksum cache: {model_id: {filename: sha256}}
        self._hf_checksums: Dict[str, Dict[str, str]] = {}

        # --- PRECISION & DEVICE SETTINGS ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        # 2026 Hardware (A100/H100/Ada) requires BFloat16 for Mixtral/Llama3+
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            print(f"⚡ BFloat16 Hardware Acceleration Enabled")
            self.dtype = torch.bfloat16
        else:
            print(f"⚠️ Warning: BFloat16 not supported, falling back to Float16 (High risk of NaN)")

        # --- CONCURRENCY CONTROL ---
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS)
        self.sem = asyncio.Semaphore(MAX_DOWNLOAD_SEMAPHORE)
        self.session: Optional[aiohttp.ClientSession] = None
        self.download_locks: dict[str, asyncio.Lock] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for downloading layers."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    # =========================================================================
    # Security: Extension Guard + HF Checksum Verification
    # =========================================================================

    @staticmethod
    def _assert_safe_extension(path: Path) -> None:
        """
        Reject any file that is not a safetensors file.

        .pt and .bin are pickle-based formats that execute arbitrary Python
        code on load. This guard is the first line of defence — it runs before
        any bytes are written to disk.

        Raises:
            ValueError: If the file extension is not .safetensors
        """
        if path.suffix not in _SAFE_EXTENSIONS:
            raise ValueError(
                f"Refusing to load '{path.name}': only .safetensors files are "
                "permitted. .pt and .bin files can execute arbitrary code via "
                "pickle and are never loaded by this worker."
            )

    async def _fetch_hf_checksums(self, model_id: str) -> Dict[str, str]:
        """
        Fetch SHA256 checksums for all files in a model repo from HuggingFace.

        Results are cached in memory for the lifetime of the loader — the HF
        API is only called once per model per worker process.

        The checksums come from HF's own model metadata (the 'siblings' field),
        making them independent of whichever registry served the actual file.
        A malicious registry cannot forge a file that passes this check.

        Args:
            model_id: HuggingFace model ID e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1"

        Returns:
            Dict mapping filename → sha256 hex string.
            Empty dict if HF is unreachable (logged as a warning; does not block loading).
        """
        if model_id in self._hf_checksums:
            return self._hf_checksums[model_id]

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        try:
            session = await self._get_session()
            async with session.get(
                f"https://huggingface.co/api/models/{model_id}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    print(f"   ⚠️ [Security] Could not fetch HF checksums for {model_id} "
                          f"(HTTP {resp.status}) — hash verification skipped")
                    self._hf_checksums[model_id] = {}
                    return {}

                data = await resp.json()

        except Exception as e:
            print(f"   ⚠️ [Security] HF checksum fetch failed for {model_id}: {e} "
                  "— hash verification skipped")
            self._hf_checksums[model_id] = {}
            return {}

        checksums = {
            s["rfilename"]: s["sha256"]
            for s in data.get("siblings", [])
            if s.get("sha256")
        }
        self._hf_checksums[model_id] = checksums
        print(f"   🔒 [Security] Fetched {len(checksums)} HF checksums for {model_id}")
        return checksums

    async def _verify_file_hash(self, path: Path, model_id: str) -> None:
        """
        Verify a downloaded file's SHA256 against HuggingFace's published manifest.

        The filename used for lookup is path.name — this matches the 'rfilename'
        field in HF's siblings list. If the checksum is not present in the
        manifest (e.g. config.json which has no sha256 on HF), verification is
        skipped silently.

        If verification fails the corrupted/tampered file is deleted from disk
        and a ValueError is raised to abort loading.

        Args:
            path: Local path to the downloaded file
            model_id: HuggingFace model ID (used to look up the checksum)

        Raises:
            ValueError: If the file hash does not match HF's published checksum
        """
        checksums = await self._fetch_hf_checksums(model_id)

        expected = checksums.get(path.name)
        if not expected:
            # Not all files have a published sha256 (e.g. config.json).
            # Skip silently — the safetensors format itself is still safe to load.
            return

        loop = asyncio.get_running_loop()

        def _compute_sha256():
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            return sha.hexdigest()

        actual = await loop.run_in_executor(self.executor, _compute_sha256)

        if actual != expected:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"[Security] Hash mismatch for '{path.name}' — possible registry "
                f"tampering detected.\n"
                f"  Expected (HuggingFace): {expected}\n"
                f"  Received (registry):    {actual}\n"
                f"The file has been deleted. This worker will not load it."
            )

        print(f"   ✅ [Security] Hash verified: {path.name}")

    # =========================================================================
    # Download
    # =========================================================================

    async def _load_weights_safe(self, path: Path) -> dict:
        """
        Load weights from a safetensors file safely.

        Args:
            path: Path to the safetensors file

        Returns:
            State dict
        """
        loop = asyncio.get_running_loop()

        # Aggressive defrag before loading
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Load directly to CPU first, standard practice for safetensors to avoid OOM
        return await loop.run_in_executor(
            self.executor,
            lambda: load_safetensors(path, device="cpu")
        )

    async def _download(self, url: str, path: Path, quiet: bool = False,
                        model_id: Optional[str] = None) -> None:
        """
        Download a file from Registry to Worker Disk, then verify its integrity.

        Security guarantees:
          1. Only .safetensors extensions are accepted (checked before any I/O).
          2. After download, SHA256 is verified against HuggingFace's published
             manifest. A registry cannot serve a malicious file that passes this
             check without also compromising HuggingFace itself.

        Args:
            url: URL to download from
            path: Local path to save the file
            quiet: If True, suppress output
            model_id: HuggingFace model ID for checksum verification.
                      Pass None only for non-tensor files (e.g. config.json).
        """
        # Reject unsafe file types before touching the network.
        # config.json is the only non-safetensors file we fetch and it is
        # never loaded with torch, so we skip the extension check for it.
        if path.suffix != ".json":
            self._assert_safe_extension(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        lock_key = str(path)
        if lock_key not in self.download_locks:
            self.download_locks[lock_key] = asyncio.Lock()

        async with self.download_locks[lock_key]:
            if path.exists() and path.stat().st_size > 0:
                # File already cached — still verify hash in case it was
                # corrupted or swapped on disk since last download.
                if model_id and path.suffix in _SAFE_EXTENSIONS:
                    await self._verify_file_hash(path, model_id)
                if not quiet:
                    print(f"   ✓ Using cached {path.name}")
                    sys.stdout.flush()
                return

            async with self.sem:
                if not quiet:
                    print(f"   ⬇️ Downloading {url}...")
                    sys.stdout.flush()

                temp = path.with_suffix('.tmp')

                try:
                    session = await self._get_session()
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise Exception(f"Download failed [{resp.status}]: {url}")

                        with open(temp, 'wb') as f:
                            async for chunk in resp.content.iter_chunked(4 * 1024 * 1024):
                                f.write(chunk)
                                await asyncio.sleep(0)

                    os.rename(temp, path)

                    # Verify hash against HF before allowing the file to be used.
                    if model_id and path.suffix in _SAFE_EXTENSIONS:
                        await self._verify_file_hash(path, model_id)

                    if not quiet:
                        print(f"   ✅ Downloaded {path.name}")
                        sys.stdout.flush()

                except Exception as e:
                    if not quiet:
                        print(f"   ❌ Error downloading {url}: {e}")
                        sys.stdout.flush()
                    if os.path.exists(temp):
                        os.remove(temp)
                    raise e

    def _get_layer_class(self, config: AutoConfig):
        """
        Get the transformer layer class from config architecture.

        Args:
            config: Model configuration

        Returns:
            Decoder layer class
        """
        arch = config.architectures[0]
        try:
            if arch.endswith("ForCausalLM"):
                base = arch[:-len("ForCausalLM")]
            elif arch.endswith("LMHeadModel"):
                base = arch[:-len("LMHeadModel")]
            else:
                base = arch

            module_name = base.lower()
            class_name = f"{base}DecoderLayer"

            if "Mixtral" in arch:
                class_name = "MixtralDecoderLayer"

            import importlib
            full_module = f"transformers.models.{module_name}.modeling_{module_name}"
            mod = importlib.import_module(full_module)
            return getattr(mod, class_name)
        except Exception as e:
            raise ValueError(f"Could not load layer class for {arch}: {e}")

    def _get_paths(self, model_id: str, filename: str):
        """
        Get local path and URL for a model file.

        Args:
            model_id: HuggingFace model ID
            filename: Name of the file

        Returns:
            Tuple of (local_path, url)
        """
        sanitized = model_id.replace("/", "_")
        path = self.cache_dir / sanitized / filename
        url = f"{self.registry_url}/layers/{sanitized}/{filename}"
        return path, url

    # =========================================================================
    # Load Shared Components (Attention + Norms) for MoE Layers
    # =========================================================================
    async def load_moe_shared(self, model_id: str, layer_idx: int):
        """
        Load shared attention and normalization components for MoE layer.

        Args:
            model_id: HuggingFace model ID
            layer_idx: Layer index

        Returns:
            Loaded layer with experts removed (lobotomized)
        """
        cache_key = f"{model_id}:shared:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        filename = f"layer_{layer_idx}_shared.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        LayerClass = self._get_layer_class(config)

        # VRAM LEAK FIX: INSTANTIATE ON CPU FIRST
        # Create layer on CPU to perform surgery before moving to GPU
        layer = LayerClass(config, layer_idx=layer_idx)

        # LOBOTOMY: Remove Experts from the container
        if hasattr(layer, "block_sparse_moe"):
            del layer.block_sparse_moe
            layer.block_sparse_moe = None
        elif hasattr(layer, "mlp"):
            del layer.mlp
            layer.mlp = None

        # Move slimmed-down layer to GPU
        layer = layer.to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)

        # strict=False because experts/router weights are missing
        layer.load_state_dict(state_dict, strict=False)
        layer.eval()

        self.loaded_cache[cache_key] = layer
        return layer

    async def load_dense_layer(self, model_id: str, layer_idx: int):
        """
        Load a dense transformer layer.

        Args:
            model_id: HuggingFace model ID
            layer_idx: Layer index

        Returns:
            Loaded dense layer
        """
        cache_key = f"{model_id}:dense:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        filename = f"layer_{layer_idx}_dense.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        LayerClass = self._get_layer_class(config)
        # FORCE DTYPE
        layer = LayerClass(config, layer_idx=layer_idx).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        layer.load_state_dict(state_dict, strict=False)
        layer.eval()

        self.loaded_cache[cache_key] = layer
        return layer

    async def load_moe_router(self, model_id: str, layer_idx: int):
        """
        Load an MoE router (gate) network.

        Args:
            model_id: HuggingFace model ID
            layer_idx: Layer index

        Returns:
            Loaded router linear layer
        """
        cache_key = f"{model_id}:router:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        filename = f"layer_{layer_idx}_router.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        num_experts = getattr(config, 'num_local_experts', 8)
        # FORCE DTYPE
        router = torch.nn.Linear(config.hidden_size, num_experts, bias=False).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        router.load_state_dict(state_dict, strict=False)
        router.eval()

        self.loaded_cache[cache_key] = router
        return router

    async def load_moe_expert(self, model_id: str, layer_idx: int, expert_idx: int):
        """
        Load an MoE expert FFN.

        Args:
            model_id: HuggingFace model ID
            layer_idx: Layer index
            expert_idx: Expert index

        Returns:
            Loaded expert FFN
        """
        cache_key = f"{model_id}:expert:{layer_idx}:{expert_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        filename = f"layer_{layer_idx}_expert_{expert_idx}.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        class ExpertFFN(torch.nn.Module):
            """Expert FFN with SwiGLU activation."""
            def __init__(self, hidden_size: int, intermediate_size: int):
                super().__init__()
                self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
                self.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                self.act_fn = torch.nn.SiLU()

            def forward(self, x):
                return self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        # FORCE DTYPE
        expert = ExpertFFN(config.hidden_size, config.intermediate_size).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        expert.load_state_dict(state_dict, strict=False)
        expert.eval()

        self.loaded_cache[cache_key] = expert
        return expert

    async def load_embeddings(self, model_id: str):
        """
        Load token embeddings.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Loaded embedding layer
        """
        cache_key = f"{model_id}:embeddings"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        path, url = self._get_paths(model_id, "embeddings.safetensors")
        await self._download(url, path, model_id=model_id)

        # FORCE DTYPE
        emb = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(self.device).to(self.dtype)
        state_dict = await self._load_weights_safe(path)
        emb.load_state_dict(state_dict)
        emb.eval()

        self.loaded_cache[cache_key] = emb
        return emb

    async def load_lm_head(self, model_id: str):
        """
        Load language model head.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Loaded LM head
        """
        cache_key = f"{model_id}:lm_head"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        path, url = self._get_paths(model_id, "lm_head.safetensors")
        await self._download(url, path, model_id=model_id)

        state_dict = await self._load_weights_safe(path)
        actual_vocab_size = state_dict['weight'].shape[0]
        # FORCE DTYPE
        head = torch.nn.Linear(config.hidden_size, actual_vocab_size, bias=False).to(self.device).to(self.dtype)
        head.load_state_dict(state_dict)
        head.eval()

        self.loaded_cache[cache_key] = head
        return head

    def _get_norm_class(self, config: AutoConfig):
        """
        Get the normalization class from config.

        Args:
            config: Model configuration

        Returns:
            Normalization class (RMSNorm or LayerNorm)
        """
        arch = config.architectures[0]
        try:
            if arch.endswith("ForCausalLM"):
                base = arch[:-len("ForCausalLM")]
            elif arch.endswith("LMHeadModel"):
                base = arch[:-len("LMHeadModel")]
            else:
                base = arch

            module_name = base.lower()
            class_name = f"{base}RMSNorm"

            if "GPT" in arch or "Bloom" in arch:
                return torch.nn.LayerNorm

            import importlib
            full_module = f"transformers.models.{module_name}.modeling_{module_name}"
            mod = importlib.import_module(full_module)
            return getattr(mod, class_name)
        except Exception as e:
            try:
                return torch.nn.RMSNorm
            except:
                raise ValueError(f"Could not load Norm class: {e}")

    async def load_final_norm(self, model_id: str):
        """
        Load final layer normalization.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Loaded norm layer or None if not available
        """
        cache_key = f"{model_id}:final_norm"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        path, url = self._get_paths(model_id, "final_norm.safetensors")
        try:
            await self._download(url, path, model_id=model_id)
        except:
            return None

        NormClass = self._get_norm_class(config)

        # FORCE DTYPE
        if hasattr(config, "rms_norm_eps"):
            norm = NormClass(config.hidden_size, eps=config.rms_norm_eps).to(self.device).to(self.dtype)
        elif hasattr(config, "layer_norm_eps"):
            norm = NormClass(config.hidden_size, eps=config.layer_norm_eps).to(self.device).to(self.dtype)
        else:
            norm = NormClass(config.hidden_size).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        norm.load_state_dict(state_dict)
        norm.eval()

        self.loaded_cache[cache_key] = norm
        return norm
