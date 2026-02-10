import torch
import os
import sys
import json
import gc
import shutil
import glob # <--- ADDED for Nuclear Option
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import ContextManagers
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors
# --- CHANGE: Import save_file ---
from safetensors.torch import save_file as save_safetensors

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
        """Find the MoE block inside a layer.

        FIXED VERSION: Checks config FIRST instead of relying on broken meta model structure.

        Returns (is_moe, path_to_moe_block) where path_to_moe_block points to
        the module that CONTAINS both the router (gate) and the experts list —
        e.g. "mlp" for Mixtral, NOT "mlp.experts".
        """

        # ===================================================================
        # STEP 1: Check config to determine if this is an MoE architecture
        # ===================================================================
        # This is CRITICAL - meta models don't reliably expose structure
        is_moe_architecture = False

        # Check for MoE-specific config attributes
        if hasattr(config, 'num_local_experts') and config.num_local_experts > 1:
            is_moe_architecture = True
        elif hasattr(config, 'num_experts') and config.num_experts > 1:
            is_moe_architecture = True
        elif hasattr(config, 'moe_num_experts') and config.moe_num_experts > 1:
            is_moe_architecture = True

        # Also check architecture name for known MoE models
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0].lower()
            if 'mixtral' in arch_name or 'moe' in arch_name:
                is_moe_architecture = True

        # If config says it's NOT MoE, return immediately
        if not is_moe_architecture:
            return False, None

        # ===================================================================
        # STEP 2: If config confirms MoE, find the path to the MoE block
        # ===================================================================
        # We try common paths where the MoE block lives
        candidates = [
            ("block_sparse_moe", ["gate", "router"]),   # some Mixtral variants
            ("mlp",              ["gate", "router"]),   # Mixtral standard (MOST COMMON)
            ("moe",              ["gate", "router"]),   # generic
            ("",                 ["gate", "router"]),   # layer itself is the block
        ]

        for block_path, router_names in candidates:
            try:
                obj = layer
                if block_path:
                    for part in block_path.split('.'):
                        obj = getattr(obj, part)

                # CRITICAL FIX: On meta device, we can't rely on checking if
                # experts has __len__ or if it's empty. Just check EXISTENCE.
                has_router = any(hasattr(obj, name) for name in router_names)
                has_experts = hasattr(obj, 'experts')

                # If we found EITHER router OR experts, this is the MoE block
                if has_router or has_experts:
                    print(f"      🔍 Found MoE block at path: '{block_path or 'ROOT'}'")
                    return True, block_path
            except AttributeError:
                continue

        # ===================================================================
        # STEP 3: Fallback - config says MoE but we can't find structure
        # ===================================================================
        # For Mixtral and most MoE models in 2026, it's always "mlp"
        print(f"      ⚠️  Config indicates MoE but couldn't detect structure. Assuming 'mlp' path.")
        return True, "mlp"

    def _get_num_experts(self, config):
        """
        Get number of experts from config.
        Works for Mixtral and other MoE models.
        """
        # Common config keys for number of experts
        for key in ['num_local_experts', 'num_experts', 'moe_num_experts', 'n_routed_experts']:
            if hasattr(config, key):
                num = getattr(config, key)
                if num > 0:
                    return num

        # Fallback: default to 8 for Mixtral-style models
        print(f"   ⚠️ Warning: Could not find num_experts in config, defaulting to 8")
        return 8

    def _load_tensor_for_key(self, key, model_path, index, loaded_shards):
        """
        Loads a specific tensor key from the checkpoint files.
        Uses a cache (loaded_shards) to avoid re-reading files unnecessarily.
        """
        filename = index.get(key)
        if not filename:
            # Maybe it's a non-sharded model (bin/safetensors directly)
            # We assume sharded for large models, but handle fallback?
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

        INCLUDES FIX FOR MIXTRAL ALIASING (block_sparse_moe <-> mlp)
        """
        state_dict = {}
        # Iterate named parameters of the meta module to know what to look for
        for name, _ in module.named_parameters(recurse=True):
            global_key = f"{prefix}.{name}"

            # --- FIX: Handle Key Aliasing ---
            # If the exact key isn't in the index, try swapping mlp/block_sparse_moe
            target_key = global_key
            if target_key not in index:
                if ".mlp." in target_key:
                    alt = target_key.replace(".mlp.", ".block_sparse_moe.")
                    if alt in index: target_key = alt
                elif ".block_sparse_moe." in target_key:
                    alt = target_key.replace(".block_sparse_moe.", ".mlp.")
                    if alt in index: target_key = alt

            if target_key in index:
                state_dict[name] = self._load_tensor_for_key(target_key, model_path, index, loaded_shards)
            else:
                # Try finding without model prefix if wrapped? No, usually precise.
                print(f"      ⚠️ Warning: Missing key {global_key} (checked alias: {target_key})")

        # --- CHANGE: .pt -> .safetensors ---
        # 1. Ensure contiguous (required for safetensors)
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        # 2. Save
        save_safetensors(state_dict, output_path)
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
            print("   📥 Fetching model artifacts (snapshot)...")
            model_path = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py"]
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
                files = glob.glob(os.path.join(model_path, "*.safetensors")) or glob.glob(os.path.join(model_path, "*.bin"))
                file_name = os.path.basename(files[0])
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

                # Debug: print what we detected
                if is_moe:
                    print(f"   Layer {i}: MoE (path: {moe_rel_path})")
                else:
                    print(f"   Layer {i}: Dense")

                if is_moe:
                    # Navigate to the MoE block (e.g. layer.mlp for Mixtral).
                    moe_module = layer
                    if moe_rel_path:  # guard: empty string means layer itself is the block
                        for part in moe_rel_path.split('.'):
                            moe_module = getattr(moe_module, part)
                        moe_prefix = f"{layer_prefix}.{moe_rel_path}"
                    else:
                        moe_prefix = layer_prefix

                    # 1. Router/Gate — extract from checkpoint
                    router_attr = "gate" if hasattr(moe_module, "gate") else "router"
                    # --- CHANGE: .pt -> .safetensors ---
                    router_file = out_dir / f"layer_{i}_router.safetensors"
                    router_prefix = f"{moe_prefix}.{router_attr}"

                    # Build router state dict manually from weight_map
                    router_state = {}

                    # The router/gate is typically a single weight tensor
                    # In checkpoint it's stored as "layer.X.block_sparse_moe.gate.weight"
                    # We need to find this exact key

                    # Try direct key match first
                    direct_key = f"{router_prefix}.weight"
                    if direct_key in weight_map:
                        router_state["weight"] = self._load_tensor_for_key(direct_key, model_path, weight_map, loaded_shards)
                    else:
                        # Try with aliasing (mlp <-> block_sparse_moe)
                        if ".mlp." in direct_key:
                            alt_key = direct_key.replace(".mlp.", ".block_sparse_moe.")
                        elif ".block_sparse_moe." in direct_key:
                            alt_key = direct_key.replace(".block_sparse_moe.", ".mlp.")
                        else:
                            alt_key = None

                        if alt_key and alt_key in weight_map:
                            router_state["weight"] = self._load_tensor_for_key(alt_key, model_path, weight_map, loaded_shards)
                        else:
                            # Last resort: search for any key containing this layer's gate/router
                            pattern = f"layers.{i}.*.{router_attr}.weight"
                            for key in weight_map.keys():
                                if f"layers.{i}." in key and f"{router_attr}.weight" in key:
                                    param_name = "weight"
                                    router_state[param_name] = self._load_tensor_for_key(key, model_path, weight_map, loaded_shards)
                                    print(f"      🔍 Found router via pattern match: {key}")
                                    break

                    if router_state:
                        # --- CHANGE: save_file + contiguous ---
                        router_state = {k: v.contiguous() for k, v in router_state.items()}
                        save_safetensors(router_state, router_file)
                        if not router_file.exists():
                            raise RuntimeError(f"Router file was not created: {router_file}")
                        print(f"      ✅ Router saved ({router_file.stat().st_size / (1024**2):.2f}MB)")
                    else:
                        # CRITICAL: Router is mandatory for MoE layers
                        print(f"      ❌ CRITICAL: No router weights found!")
                        print(f"      Tried: {direct_key}, {alt_key if 'alt_key' in locals() else 'N/A'}")
                        # Show some sample keys from this layer to debug
                        layer_keys = [k for k in weight_map.keys() if f"layers.{i}." in k][:5]
                        print(f"      Sample keys for layer {i}: {layer_keys}")
                        raise RuntimeError(f"Router missing for MoE layer {i}")

                    # 2. Experts — handle both ModuleList and custom MixtralExperts class
                    num_experts = self._get_num_experts(config)
                    experts_list = getattr(moe_module, 'experts', None)

                    # CRITICAL: In 2026, Mixtral uses MixtralExperts which doesn't have __len__
                    # We can't use the traditional ModuleList[i] approach
                    # Instead, we ALWAYS use the weight_map approach for MoE models

                    expert_size_acc = 0

                    # DEBUG: Show sample expert keys from checkpoint
                    sample_expert_keys = [k for k in weight_map.keys() if f"layers.{i}." in k and "expert" in k.lower()]
                    if sample_expert_keys:
                        print(f"      🔍 Sample expert keys for layer {i}: {sample_expert_keys[:3]}")

                    for exp_idx in range(num_experts):
                        # --- CHANGE: .pt -> .safetensors ---
                        expert_file = out_dir / f"layer_{i}_expert_{exp_idx}.safetensors"

                        # Build state dict by finding all keys for this expert in weight_map
                        expert_state = {}
                        expert_prefix = f"{moe_prefix}.experts.{exp_idx}"

                        # Try both mlp and block_sparse_moe aliases
                        possible_prefixes = [expert_prefix]
                        if ".mlp." in expert_prefix:
                            possible_prefixes.append(expert_prefix.replace(".mlp.", ".block_sparse_moe."))
                        elif ".block_sparse_moe." in expert_prefix:
                            possible_prefixes.append(expert_prefix.replace(".block_sparse_moe.", ".mlp."))

                        for key in weight_map.keys():
                            for prefix in possible_prefixes:
                                if key.startswith(prefix):
                                    # Extract the relative parameter name (everything after "experts.N.")
                                    param_name = key[len(prefix)+1:]  # +1 for the dot
                                    expert_state[param_name] = self._load_tensor_for_key(key, model_path, weight_map, loaded_shards)
                                    break  # Found with this prefix, don't check others

                        if not expert_state:
                            print(f"      ⚠️  WARNING: No weights found for expert {exp_idx}")
                            print(f"      Tried prefixes: {possible_prefixes}")
                            # This is actually a critical error for MoE models
                            raise RuntimeError(f"Expert {exp_idx} has no weights in checkpoint")

                        # --- CHANGE: save_file + contiguous ---
                        expert_state = {k: v.contiguous() for k, v in expert_state.items()}
                        save_safetensors(expert_state, expert_file)

                        sz = expert_file.stat().st_size / (1024**2)
                        expert_size_acc += sz

                        if exp_idx % 4 == 0:
                            sys.stdout.write(".")
                        sys.stdout.flush()

                    print(f" Layer {i} MoE done ({num_experts} experts)")

                    layer_metadata.append({
                        "layer_idx": i, "type": "moe", "size_mb": expert_size_acc, "num_experts": num_experts
                    })
                    total_size_mb += expert_size_acc

                else:
                    # Dense Layer
                    # --- CHANGE: .pt -> .safetensors ---
                    dense_file = out_dir / f"layer_{i}_dense.safetensors"
                    sz = self._save_module(layer, layer_prefix, model_path, weight_map, dense_file, loaded_shards)

                    if not dense_file.exists():
                        raise RuntimeError(f"Dense layer file was not created: {dense_file}")

                    layer_metadata.append({
                        "layer_idx": i, "type": "dense", "size_mb": sz, "num_experts": 0
                    })
                    total_size_mb += sz
                    print(f"   Layer {i} Dense done ({sz:.2f}MB)")

                # GC after every layer to be safe
                loaded_shards.clear()
                gc.collect()

            # 6. Embeddings & Head
            print("   💾 Saving embeddings & head...")
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                # --- CHANGE: .pt -> .safetensors ---
                emb_file = out_dir / "embeddings.safetensors"
                self._save_module(model.model.embed_tokens, "model.embed_tokens", model_path, weight_map, emb_file, loaded_shards)
                if not emb_file.exists():
                    raise RuntimeError(f"Embeddings file was not created: {emb_file}")
                print(f"      ✅ Embeddings saved")

            if hasattr(model, "lm_head"):
                # --- CHANGE: .pt -> .safetensors ---
                head_file = out_dir / "lm_head.safetensors"
                self._save_module(model.lm_head, "lm_head", model_path, weight_map, head_file, loaded_shards)
                if not head_file.exists():
                    raise RuntimeError(f"LM head file was not created: {head_file}")
                print(f"      ✅ LM head saved")

            # 7. Metadata & Tokenizer
            print("   💾 Saving metadata and tokenizer assets...")
            config.save_pretrained(out_dir)

            # A. Try standard save_pretrained
            try:
                from transformers import AutoTokenizer as TokenizerClass
                tokenizer = TokenizerClass.from_pretrained(
                    model_path,
                    token=hf_token,
                    trust_remote_code=True
                )
                tokenizer.save_pretrained(out_dir)
            except Exception as e:
                print(f"      ⚠️ standard save_pretrained failed: {e}")

            # B. "NUCLEAR" COPY STRATEGY
            # Copy ANY file that looks like metadata/config/tokenizer/code
            aux_extensions = ['*.json', '*.model', '*.txt', '*.py']
            for ext in aux_extensions:
                for src_path in Path(model_path).rglob(ext):
                    filename = os.path.basename(src_path)

                    # Skip hidden files or index files if huge
                    if filename.startswith(".") or "index" in filename: continue

                    dst_path = out_dir / filename

                    # Copy if it doesn't exist (e.g. tokenizer.model often missed by save_pretrained)
                    if not dst_path.exists():
                        shutil.copy2(src_path, dst_path)
                        print(f"      📦 Manually copied: {filename}")

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

            # 8. Verify all critical files exist
            print(f"   🔍 Verifying files...")
            critical_files = ["structure.json", "config.json"]
            for cf in critical_files:
                if not (out_dir / cf).exists():
                    raise RuntimeError(f"Critical file missing: {cf}")

            # List all files created
            # --- CHANGE: glob .safetensors ---
            all_files = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.json"))
            print(f"   📁 Created {len(all_files)} files in {out_dir}")

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
