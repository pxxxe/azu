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
from typing import Optional, Dict, List, Type

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

        # Per-architecture layer-class list built from a zero-weight meta model.
        # Keys are config.architectures[0]; values are lists of layer classes,
        # one entry per layer index.  Built lazily on first trust_remote_code miss.
        self._layer_class_cache: Dict[str, List[Type[torch.nn.Module]]] = {}

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

    @staticmethod
    def _normalize_config(config) -> None:
        """
        Proxy missing top-level config attributes from a nested text_config.

        Some VLM / multimodal architectures (e.g. Qwen3.5-27B) store the
        language-model config inside config.text_config rather than at the
        top level.  Transformers exposes the outer config object, so attributes
        like vocab_size, hidden_size, max_position_embeddings etc. raise
        AttributeError when accessed directly.

        This method checks for a text_config sub-object and, for each
        attribute that is missing on the outer config, copies the value across
        so the rest of the loader can use config.vocab_size etc. uniformly.

        Mutates config in place; safe to call multiple times (no-op if attrs
        already present).
        """
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            return

        _PROXY_ATTRS = [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "layer_norm_eps",
            "rope_theta",
            "rope_scaling",
            "attention_bias",
            "attention_dropout",
            "hidden_act",
            "initializer_range",
            "tie_word_embeddings",
        ]

        for attr in _PROXY_ATTRS:
            try:
                # Will raise AttributeError if missing on outer config
                _ = getattr(config, attr)
            except AttributeError:
                try:
                    setattr(config, attr, getattr(text_cfg, attr))
                except AttributeError:
                    pass  # not on text_config either — skip

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

        # Load directly to the target device — safetensors supports this natively
        # and avoids staging through CPU RAM entirely.
        return await loop.run_in_executor(
            self.executor,
            lambda: load_safetensors(path, device=self.device)
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

    # =========================================================================
    # Layer class resolution
    # =========================================================================

    def _get_layer_class(self, config: AutoConfig):
        """
        Get the transformer layer class from config architecture.

        Fast path: works for all architectures that live inside
        transformers.models.<name>.modeling_<name>.  Raises ValueError if the
        architecture is not found there (e.g. trust_remote_code models).

        Args:
            config: Model configuration

        Returns:
            Decoder layer class

        Raises:
            ValueError: If the class cannot be resolved via importlib
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

    def _build_layer_class_cache(self, config: AutoConfig) -> None:
        """
        Populate _layer_class_cache for an architecture that cannot be resolved
        via importlib (e.g. trust_remote_code models such as Qwen3.5).

        Instantiates the full model on meta device (zero RAM / zero VRAM) and
        records the concrete Python class of every layer, indexed by position.
        The result is cached by architecture name so this only runs once per
        worker process per model family.

        For homogeneous models (all layers the same class) this degenerates to
        a single repeated entry — no overhead versus the old approach.

        For hybrid models (e.g. Qwen3.5 which alternates Gated-DeltaNet and
        Gated-Attention layers) this gives the exact per-position class needed
        for correct layer instantiation during load_dense_layer.

        Args:
            config: Model configuration (must have been loaded with
                    trust_remote_code=True so custom code is registered)
        """
        arch = config.architectures[0]
        if arch in self._layer_class_cache:
            return

        print(f"   🏗️ Building layer-class cache for {arch} (meta device, one-time)...")

        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM, AutoModel

        # Try CausalLM first; fall back to generic AutoModel for VLMs /
        # conditional-generation models (e.g. Qwen3.5's
        # Qwen3_5ForConditionalGeneration which does not register as ForCausalLM).
        model = None
        for loader_cls in (AutoModelForCausalLM, AutoModel):
            try:
                with init_empty_weights():
                    model = loader_cls.from_config(config, trust_remote_code=True)
                break
            except Exception:
                continue

        if model is None:
            raise ValueError(
                f"Could not instantiate meta model for {arch}. "
                "Ensure the model was downloaded with trust_remote_code=True."
            )

        # Locate the layers list — try standard and VLM layout variants.
        layer_paths = [
            "model.layers",
            "transformer.h",
            "model.decoder.layers",
            "transformer.layers",
            "language_model.model.layers",
            "model.language_model.layers",
        ]

        layers = None
        for attr_path in layer_paths:
            try:
                obj = model
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    layers = obj
                    break
            except AttributeError:
                continue

        if layers is None:
            del model
            raise ValueError(
                f"Could not locate layer list in meta model for {arch}. "
                f"Tried paths: {layer_paths}"
            )

        self._layer_class_cache[arch] = [type(layers[i]) for i in range(len(layers))]
        print(
            f"   ✅ Layer-class cache built: {len(self._layer_class_cache[arch])} layers, "
            f"{len(set(self._layer_class_cache[arch]))} distinct class(es)"
        )
        del model

    def _get_layer_class_for_idx(self, config: AutoConfig, layer_idx: int) -> Type[torch.nn.Module]:
        """
        Return the concrete layer class for a specific layer index.

        For well-known architectures that live in transformers.models.* this
        is the same as _get_layer_class (fast importlib path, O(1)).

        For trust_remote_code / VLM architectures (e.g. Qwen3.5) where
        importlib fails — or for hybrid models where different layer positions
        have different classes — this builds the per-index class list from a
        zero-weight meta model (once per architecture, then cached).

        Args:
            config: Model configuration
            layer_idx: Index of the layer to load

        Returns:
            Concrete nn.Module subclass for that layer position
        """
        # Fast path: importlib resolution works for standard transformers models.
        try:
            return self._get_layer_class(config)
        except ValueError:
            pass

        # Slow path: trust_remote_code or non-standard architecture.
        # Build (or reuse) the per-architecture class list.
        arch = config.architectures[0]
        if arch not in self._layer_class_cache:
            self._build_layer_class_cache(config)

        classes = self._layer_class_cache[arch]
        if layer_idx < len(classes):
            return classes[layer_idx]

        # layer_idx out of range (shouldn't happen in practice) — use last class.
        return classes[-1]

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
        self._normalize_config(config)

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
        self._normalize_config(config)

        filename = f"layer_{layer_idx}_shared.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        # Use per-index class resolution so MoE layers in hybrid architectures
        # get the right class.
        LayerClass = self._get_layer_class_for_idx(config, layer_idx)

        # VRAM LEAK FIX: INSTANTIATE ON CPU FIRST
        # Create layer on CPU to perform surgery before moving to GPU
        try:
            layer = LayerClass(config, layer_idx=layer_idx)
        except TypeError:
            layer = LayerClass(config)

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
        self._normalize_config(config)

        filename = f"layer_{layer_idx}_dense.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        # Per-index class resolution: required for hybrid architectures such as
        # Qwen3.5 where even-positioned layers are Gated-DeltaNet and
        # every-fourth layer is Gated-Attention.
        LayerClass = self._get_layer_class_for_idx(config, layer_idx)

        # Some custom layer constructors accept only (config) without layer_idx.
        try:
            layer = LayerClass(config, layer_idx=layer_idx)
        except TypeError:
            layer = LayerClass(config)

        layer = layer.to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        layer.load_state_dict(state_dict, strict=False)
        layer.eval()

        self.loaded_cache[cache_key] = layer
        return layer

    async def load_moe_router(self, model_id: str, layer_idx: int):
        """
        Load an MoE router (gate) network.
        """
        cache_key = f"{model_id}:router:{layer_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self._normalize_config(config)

        filename = f"layer_{layer_idx}_router.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        num_experts = self._get_num_experts(config)
        hidden_size = config.hidden_size
        router = torch.nn.Linear(hidden_size, num_experts, bias=False).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        router.load_state_dict(state_dict)
        router.eval()

        self.loaded_cache[cache_key] = router
        return router

    def _get_num_experts(self, config):
        """Get number of experts from config."""
        for key in ['num_local_experts', 'num_experts', 'moe_num_experts', 'n_routed_experts']:
            if hasattr(config, key):
                num = getattr(config, key)
                if num > 0:
                    return num
        return 8

    async def load_moe_expert(self, model_id: str, layer_idx: int, expert_idx: int):
        """Load a single MoE expert."""
        cache_key = f"{model_id}:expert:{layer_idx}:{expert_idx}"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self._normalize_config(config)

        filename = f"layer_{layer_idx}_expert_{expert_idx}.safetensors"
        path, url = self._get_paths(model_id, filename)
        await self._download(url, path, model_id=model_id)

        intermediate = getattr(config, 'intermediate_size', config.hidden_size * 4)
        hidden = config.hidden_size

        expert = torch.nn.Sequential(
            torch.nn.Linear(hidden, intermediate, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(intermediate, hidden, bias=False),
        ).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        expert.load_state_dict(state_dict, strict=False)
        expert.eval()

        self.loaded_cache[cache_key] = expert
        return expert

    async def load_embeddings(self, model_id: str):
        """Load token embeddings."""
        cache_key = f"{model_id}:embeddings"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self._normalize_config(config)

        path, url = self._get_paths(model_id, "embeddings.safetensors")
        await self._download(url, path, model_id=model_id)

        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        emb = torch.nn.Embedding(vocab_size, hidden_size).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        emb.load_state_dict(state_dict)
        emb.eval()

        self.loaded_cache[cache_key] = emb
        return emb

    async def load_lm_head(self, model_id: str):
        """Load the language model head."""
        cache_key = f"{model_id}:lm_head"
        if cache_key in self.loaded_cache:
            return self.loaded_cache[cache_key]

        config_path, config_url = self._get_paths(model_id, "config.json")
        await self._download(config_url, config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self._normalize_config(config)

        path, url = self._get_paths(model_id, "lm_head.safetensors")
        await self._download(url, path, model_id=model_id)

        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        head = torch.nn.Linear(hidden_size, vocab_size, bias=False).to(self.device).to(self.dtype)

        state_dict = await self._load_weights_safe(path)
        head.load_state_dict(state_dict)
        head.eval()

        self.loaded_cache[cache_key] = head
        return head
