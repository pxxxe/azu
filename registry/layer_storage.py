import torch
import os
import sys
import json
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM

class LayerStore:
    def __init__(self, storage_path="/data/layers"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def _find_layers(self, model):
        """Find transformer layers in model architecture."""
        print(f"   🔍 Searching for layers in model...")

        possible_paths = [
            'model.layers',           # Llama, Mistral, Qwen2
            'transformer.h',          # GPT-2
            'model.decoder.layers',   # Bart, T5
            'transformer.layers',     # Some custom models
        ]

        for attr_path in possible_paths:
            try:
                parts = attr_path.split('.')
                obj = model
                found = True

                for part in parts:
                    if not hasattr(obj, part):
                        found = False
                        break
                    obj = getattr(obj, part)

                if found and hasattr(obj, '__len__') and len(obj) > 0:
                    print(f"   ✅ Found layers at: {attr_path}")
                    return obj
            except AttributeError:
                continue

        raise ValueError(f"Could not find transformer layers in model. Available attributes: {dir(model)}")

    def _is_moe_layer(self, layer, config):
        """Check if a layer is an MoE layer."""
        # Check for common MoE attributes
        moe_indicators = [
            'mlp.experts',      # Qwen2-MoE, DeepSeek-MoE
            'block_sparse_moe', # Mixtral
            'moe',              # Generic
            'experts',          # Some models
        ]

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

    def _extract_moe_experts(self, layer, layer_idx, model_dir, moe_path):
        """Extract individual experts and router from MoE layer."""
        print(f"      🎯 MoE Layer {layer_idx} - Extracting experts...")

        # Navigate to the MoE module
        parts = moe_path.split('.')
        moe_module = layer
        for part in parts:
            moe_module = getattr(moe_module, part)

        # Extract router/gate
        router_attr = None
        for attr in ['gate', 'router', 'gating']:
            if hasattr(moe_module, attr):
                router_attr = attr
                break

        if router_attr:
            router = getattr(moe_module, router_attr)
            router_path = model_dir / f"layer_{layer_idx}_router.pt"
            torch.save(router.state_dict(), router_path)
            router_size = router_path.stat().st_size / (1024**2)
            print(f"         Router: {router_size:.1f}MB")

        # Extract experts
        experts_attr = None
        for attr in ['experts', 'expert']:
            if hasattr(moe_module, attr):
                experts_attr = attr
                break

        num_experts = 0
        total_expert_size = 0

        if experts_attr:
            experts = getattr(moe_module, experts_attr)
            if hasattr(experts, '__len__'):
                num_experts = len(experts)
                print(f"         Found {num_experts} experts")

                for expert_idx, expert in enumerate(experts):
                    expert_path = model_dir / f"layer_{layer_idx}_expert_{expert_idx}.pt"
                    torch.save(expert.state_dict(), expert_path)
                    size = expert_path.stat().st_size / (1024**2)
                    total_expert_size += size

                    if (expert_idx + 1) % 8 == 0:
                        print(f"         Extracted {expert_idx + 1}/{num_experts} experts ({total_expert_size:.1f}MB)")

        # Extract shared expert if exists (Qwen2-MoE has this)
        if hasattr(moe_module, 'shared_expert'):
            shared_expert = moe_module.shared_expert
            shared_path = model_dir / f"layer_{layer_idx}_shared_expert.pt"
            torch.save(shared_expert.state_dict(), shared_path)
            size = shared_path.stat().st_size / (1024**2)
            print(f"         Shared Expert: {size:.1f}MB")

        # Save layer metadata
        metadata = {
            "layer_idx": layer_idx,
            "type": "moe",
            "num_experts": num_experts,
            "has_router": router_attr is not None,
            "has_shared_expert": hasattr(moe_module, 'shared_expert'),
            "router_attr": router_attr,
            "experts_attr": experts_attr,
            "moe_path": moe_path,
            "total_size_mb": total_expert_size
        }

        with open(model_dir / f"layer_{layer_idx}_moe_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return num_experts

    def shard_model(self, model_id: str, hf_token: str):
        """
        Download model and extract individual layers + experts.
        Returns number of layers extracted.
        """
        print(f"\n🔪 SHARDING {model_id}")
        print(f"   Storage path: {self.storage_path}")
        print(f"   HF Token: {'*' * 10}{hf_token[-4:] if len(hf_token) > 4 else '???'}")

        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(self.storage_path)
        print(f"   💾 Disk: {free // (2**30)}GB free / {total // (2**30)}GB total")

        if free < 5 * (2**30):
            raise Exception(f"Insufficient disk space: {free // (2**30)}GB free, need at least 5GB")

        try:
            print(f"\n   📥 Downloading model from HuggingFace...")
            sys.stdout.flush()

            # Load config first to check if MoE
            config = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            print(f"   ✅ Config loaded")

            # Check for MoE indicators
            is_moe = False
            num_experts_per_tok = None

            for attr in ['num_local_experts', 'num_experts', 'moe_num_experts']:
                if hasattr(config, attr):
                    is_moe = True
                    print(f"   🎯 MoE Model Detected! ({attr}={getattr(config, attr)})")
                    break

            if hasattr(config, 'num_experts_per_tok'):
                num_experts_per_tok = config.num_experts_per_tok
                print(f"   🎯 Top-{num_experts_per_tok} expert routing")

            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"   ✅ Model downloaded successfully")

        except Exception as e:
            print(f"\n   ❌ Failed to download model from HuggingFace")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            raise Exception(f"HuggingFace download failed: {str(e)}")

        # Find layer structure
        print(f"\n   🔍 Finding layer structure...")
        try:
            layers = self._find_layers(model)
            num_layers = len(layers)
            print(f"   ✅ Found {num_layers} layers")
        except Exception as e:
            print(f"   ❌ Failed to find layers in model")
            raise Exception(f"Layer detection failed: {str(e)}")

        # Create storage directory
        model_dir = self.storage_path / model_id.replace("/", "_")
        model_dir.mkdir(exist_ok=True, parents=True)
        print(f"   📁 Output directory: {model_dir}")

        print(f"\n   💾 Extracting layers...")

        layer_metadata = []
        total_size_mb = 0

        # Save each layer
        for i, layer in enumerate(layers):
            # Check if this is an MoE layer
            is_moe_layer, moe_path = self._is_moe_layer(layer, config)

            if is_moe_layer:
                # Extract MoE experts separately
                num_experts = self._extract_moe_experts(layer, i, model_dir, moe_path)
                layer_type = "moe"

                # Calculate total size of all expert files
                expert_files = list(model_dir.glob(f"layer_{i}_expert_*.pt"))
                layer_size = sum(f.stat().st_size for f in expert_files) / (1024**2)

            else:
                # Regular dense layer - save whole thing
                layer_path = model_dir / f"layer_{i}_dense.pt"
                torch.save(layer.state_dict(), layer_path)
                layer_size = layer_path.stat().st_size / (1024**2)
                layer_type = "dense"
                num_experts = 0

            total_size_mb += layer_size

            metadata = {
                "layer_idx": i,
                "type": layer_type,
                "size_mb": layer_size,
                "num_experts": num_experts if is_moe_layer else 0
            }
            layer_metadata.append(metadata)

            # Progress indicator
            if (i + 1) % 5 == 0 or i == num_layers - 1:
                print(f"      Progress: {i+1}/{num_layers} layers ({total_size_mb:.1f}MB total)")
                sys.stdout.flush()

        # Save embeddings and heads
        print(f"\n   💾 Saving embeddings and heads...")

        # 1. Embeddings
        if hasattr(model, 'get_input_embeddings'):
            emb = model.get_input_embeddings()
            torch.save(emb.state_dict(), model_dir / "embeddings.pt")
            emb_size = (model_dir / 'embeddings.pt').stat().st_size / (1024**2)
            print(f"      Embeddings: {emb_size:.1f}MB")
        else:
            print("      ⚠️ WARNING: Could not find input embeddings!")

        # 2. LM Head
        if hasattr(model, 'lm_head'):
            torch.save(model.lm_head.state_dict(), model_dir / "lm_head.pt")
            head_size = (model_dir / 'lm_head.pt').stat().st_size / (1024**2)
            print(f"      LM Head: {head_size:.1f}MB")
        else:
            print("      ⚠️ WARNING: Could not find lm_head!")

        # Save config
        config.save_pretrained(model_dir)

        # Save comprehensive structure info
        structure_info = {
            "model_id": model_id,
            "num_layers": num_layers,
            "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown",
            "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
            "total_size_mb": total_size_mb,
            "is_moe": is_moe,
            "num_experts_per_tok": num_experts_per_tok,
            "layer_metadata": layer_metadata
        }

        with open(model_dir / "structure.json", "w") as f:
            json.dump(structure_info, f, indent=2)

        del model
        torch.cuda.empty_cache()

        print(f"\n✅ SHARDING COMPLETE")
        print(f"   Model: {model_id}")
        print(f"   Layers: {num_layers}")
        print(f"   MoE: {'Yes' if is_moe else 'No'}")
        if is_moe:
            moe_layers = [m for m in layer_metadata if m['type'] == 'moe']
            print(f"   MoE Layers: {len(moe_layers)}")
            print(f"   Total Experts: {sum(m['num_experts'] for m in moe_layers)}")
        print(f"   Total size: {total_size_mb:.1f}MB")

        return num_layers

    def get_model_structure(self, model_id: str):
        """Get overall model structure info"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        structure_path = model_dir / "structure.json"

        if not structure_path.exists():
            return None

        with open(structure_path) as f:
            return json.load(f)

    def has_model(self, model_id: str):
        """Check if model is already sharded"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        return (model_dir / "structure.json").exists()
