import torch
import os
import sys
import json
import gc
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import ContextManagers
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors
import glob

class LayerStore:
    def __init__(self, storage_path="/data/layers"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def has_model(self, model_id: str) -> bool:
        """Check if a model is already fully sharded and stored."""
        sanitized = model_id.replace("/", "_")
        # We assume if structure.json exists, the sharding completed successfully
        return (self.storage_path / sanitized / "structure.json").exists()

    def _find_layers(self, model):
        """Find transformer layers in model architecture (works on meta device)."""
        print(f"   🔍 Searching for layers in model structure...")
        possible_paths = ['model.layers', 'transformer.h', 'model.decoder.layers', 'transformer.layers']
        for attr_path in possible_paths:
            try:
                parts = attr_path.split('.')
                obj = model
                for part in parts:
                    obj = getattr(obj, part)
                if hasattr(obj, '__len__'):
                    print(f"   ✅ Found layers at: {attr_path}")
                    return obj, attr_path
            except AttributeError:
                continue
        raise ValueError(f"Could not find layers. Model keys: {list(model.state_dict().keys())[:5]}...")

    def _is_moe_layer(self, layer, config):
        """Check if a layer is an MoE layer."""
        # On meta device, we check structure
        moe_indicators = ['mlp.experts', 'block_sparse_moe', 'moe', 'experts']
        for indicator in moe_indicators:
            parts = indicator.split('.')
            obj = layer
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if obj is not None:
                    return True, indicator
            except AttributeError:
                continue
        return False, None

    def _load_tensor_for_key(self, key, model_path, index, loaded_shards):
        """
        Loads a specific tensor key from the checkpoint files.
        Uses a cache (loaded_shards) to avoid re-reading files unnecessarily.
        """
        filename = index.get(key)
        if not filename:
            # Maybe it's a non-sharded model (bin/safetensors directly)
            # We assume sharded for large models, but handle fallback?
            # For now, rely on index. If index is None, it implies single file.
            pass

        filepath = os.path.join(model_path, filename)

        # Check cache
        if filepath not in loaded_shards:
            # Free memory if we have too many shards loaded?
            # For strict memory, we keep only ONE shard.
            loaded_shards.clear()
            gc.collect()

            print(f"      📖 Loading shard: {filename}")
            if filename.endswith(".safetensors"):
                loaded_shards[filepath] = load_safetensors(filepath)
            else:
                loaded_shards[filepath] = torch.load(filepath, map_location="cpu")

        return loaded_shards[filepath][key]

    def _save_module(self, module, prefix, model_path, index, output_path, loaded_shards):
        """
        Reconstructs a module's state_dict by fetching keys from disk and saving to output_path.
        prefix: e.g. "model.layers.0.mlp.experts.0"
        """
        state_dict = {}
        # Find all keys in the index that start with this prefix
        # We need to map the module's local keys (e.g. "weight") to global keys (e.g. "model.layers.0...")

        # Iterate named parameters of the meta module to know what to look for
        for name, _ in module.named_parameters(recurse=True):
            global_key = f"{prefix}.{name}"
            # Some models have different mapping, but usually simple concatenation works
            if global_key in index:
                state_dict[name] = self._load_tensor_for_key(global_key, model_path, index, loaded_shards)
            else:
                # Try finding without model prefix if wrapped? No, usually precise.
                print(f"      ⚠️ Warning: Missing key {global_key}")

        torch.save(state_dict, output_path)
        return output_path.stat().st_size / (1024**2)

    def shard_model(self, model_id: str, hf_token: str):
        print(f"\n🔪 SHARDING {model_id} (Streaming Mode)")

        # Check disk space
        import shutil
        total, _, free = shutil.disk_usage(self.storage_path)
        if free < 50 * (2**30): # 50GB check
            print(f"   ⚠️ Low disk space: {free // (2**30)}GB free")

        from huggingface_hub import snapshot_download

        try:
            # 1. Download Model Artifacts (Metadata + Weights)
            # We download to cache, then read. snapshot_download handles caching.
            print("   📥 Fetching model artifacts (snapshot)...")
            model_path = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"]
            )
            print(f"   ✅ Model available at: {model_path}")

            # 2. Load Config & Index
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Load index.json for weights mapping
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                index_path = os.path.join(model_path, "pytorch_model.bin.index.json")

            if os.path.exists(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
            else:
                # Single file model
                print("   ℹ️ Single file model detected.")
                # Map all keys to the single file
                files = glob.glob(os.path.join(model_path, "*.safetensors")) or glob.glob(os.path.join(model_path, "*.bin"))
                file_name = os.path.basename(files[0])
                # We need to know keys. Load the meta model first.
                weight_map = None # Will populate later

            # 3. Instantiate Meta Model (Zero RAM)
            print("   🏗️ Building Meta Model...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

            # Populate weight map if single file
            if weight_map is None:
                weight_map = {k: file_name for k in model.state_dict().keys()}

            # 4. Prepare Output Dir
            out_dir = self.storage_path / model_id.replace("/", "_")
            out_dir.mkdir(exist_ok=True, parents=True)

            # 5. Extract Structure
            layers_obj, layer_prefix_base = self._find_layers(model)
            num_layers = len(layers_obj)
            layer_metadata = []
            total_size_mb = 0

            # State for streaming loader
            loaded_shards = {} # path -> dict (cache current file)

            print(f"   💾 Processing {num_layers} layers...")

            for i in range(num_layers):
                layer = layers_obj[i]
                layer_prefix = f"{layer_prefix_base}.{i}"

                is_moe, moe_rel_path = self._is_moe_layer(layer, config)

                if is_moe:
                    # It's MoE. We need to save Router and Experts separately.
                    # Navigate to MoE module
                    moe_module = layer
                    for part in moe_rel_path.split('.'):
                        moe_module = getattr(moe_module, part)

                    moe_prefix = f"{layer_prefix}.{moe_rel_path}"

                    # 1. Router
                    router_attr = "gate" if hasattr(moe_module, "gate") else "router" # generic
                    router = getattr(moe_module, router_attr, None)
                    if router:
                        self._save_module(router, f"{moe_prefix}.{router_attr}", model_path, weight_map, out_dir / f"layer_{i}_router.pt", loaded_shards)

                    # 2. Experts
                    experts = getattr(moe_module, "experts", [])
                    num_experts = len(experts)
                    expert_size_acc = 0

                    for exp_idx, expert in enumerate(experts):
                        sz = self._save_module(expert, f"{moe_prefix}.experts.{exp_idx}", model_path, weight_map, out_dir / f"layer_{i}_expert_{exp_idx}.pt", loaded_shards)
                        expert_size_acc += sz
                        if exp_idx % 4 == 0: sys.stdout.write(".")
                        sys.stdout.flush()

                    print(f" Layer {i} MoE done ({num_experts} experts)")

                    layer_metadata.append({
                        "layer_idx": i, "type": "moe", "size_mb": expert_size_acc, "num_experts": num_experts
                    })
                    total_size_mb += expert_size_acc

                else:
                    # Dense Layer
                    # We save the WHOLE layer as one block
                    sz = self._save_module(layer, layer_prefix, model_path, weight_map, out_dir / f"layer_{i}_dense.pt", loaded_shards)
                    layer_metadata.append({
                        "layer_idx": i, "type": "dense", "size_mb": sz, "num_experts": 0
                    })
                    total_size_mb += sz
                    print(f"   Layer {i} Dense done")

                # GC after every layer to be safe
                loaded_shards.clear()
                gc.collect()

            # 6. Embeddings & Head
            print("   💾 Saving embeddings & head...")
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self._save_module(model.model.embed_tokens, "model.embed_tokens", model_path, weight_map, out_dir / "embeddings.pt", loaded_shards)

            if hasattr(model, "lm_head"):
                self._save_module(model.lm_head, "lm_head", model_path, weight_map, out_dir / "lm_head.pt", loaded_shards)

            # 7. Metadata
            config.save_pretrained(out_dir)
            structure = {
                "model_id": model_id,
                "num_layers": num_layers,
                "architecture": config.architectures[0],
                "hidden_size": config.hidden_size,
                "total_size_mb": total_size_mb,
                "is_moe": any(x['type'] == 'moe' for x in layer_metadata),
                "num_experts_per_tok": getattr(config, "num_experts_per_tok", 2),
                "layer_metadata": layer_metadata
            }

            with open(out_dir / "structure.json", "w") as f:
                json.dump(structure, f, indent=2)

            print(f"✅ Sharding Complete. Total Size: {total_size_mb:.1f}MB")
            return num_layers

        except Exception as e:
            print(f"❌ Sharding Error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            # Cleanup
            if 'loaded_shards' in locals(): loaded_shards.clear()
            if 'model' in locals(): del model
            gc.collect()
