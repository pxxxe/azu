# registry/layer_storage.py
import torch
from transformers import AutoModel, AutoConfig
from pathlib import Path
import json
import os
import sys

class LayerStore:
    def __init__(self, storage_path="/data/layers"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def _find_layers(self, model):
        """
        Find the transformer layers in the model.
        Returns the actual layer list object.
        """
        # Try common layer attribute names
        for attr_path in [
            'model.layers',           # Llama, Mistral, Qwen
            'transformer.h',          # GPT-2, GPT-Neo
            'encoder.layer',          # BERT
            'decoder.layers',         # T5
            'h',                      # Some GPT models
            'layers'                  # Generic
        ]:
            try:
                obj = model
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)

                # Verify it's a list/ModuleList of layers
                if hasattr(obj, '__len__') and len(obj) > 0:
                    print(f"   ✅ Found layers at: {attr_path}")
                    return obj
            except AttributeError:
                continue

        raise ValueError(f"Could not find transformer layers in model. Available attributes: {dir(model)}")

    def shard_model(self, model_id: str, hf_token: str):
        """
        Download model and extract individual layers.
        Returns number of layers extracted.
        """
        print(f"\n🔪 SHARDING {model_id}")
        print(f"   Storage path: {self.storage_path}")
        print(f"   HF Token: {'*' * 10}{hf_token[-4:] if len(hf_token) > 4 else '???'}")

        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(self.storage_path)
        print(f"   💾 Disk: {free // (2**30)}GB free / {total // (2**30)}GB total")

        if free < 5 * (2**30):  # Less than 5GB
            raise Exception(f"Insufficient disk space: {free // (2**30)}GB free, need at least 5GB")

        try:
            print(f"\n   📥 Downloading model from HuggingFace...")
            sys.stdout.flush()

            # Load full model temporarily (CPU only to save VRAM)
            model = AutoModel.from_pretrained(
                model_id,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"   ✅ Model downloaded successfully")

            print(f"   📥 Loading config...")
            config = AutoConfig.from_pretrained(model_id, token=hf_token)
            print(f"   ✅ Config loaded")

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

        # Save each layer separately
        for i, layer in enumerate(layers):
            layer_path = model_dir / f"layer_{i}.pt"

            # Save layer weights
            torch.save(layer.state_dict(), layer_path)

            # Calculate size
            size_mb = layer_path.stat().st_size / (1024**2)

            # Store metadata
            metadata = {
                "model_id": model_id,
                "layer_idx": i,
                "size_mb": size_mb,
                "dtype": "float16",
                "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown"
            }

            with open(model_dir / f"layer_{i}.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Progress indicator every 5 layers
            if (i + 1) % 5 == 0 or i == num_layers - 1:
                print(f"      Progress: {i+1}/{num_layers} layers ({size_mb:.1f}MB)")
                sys.stdout.flush()

        # Save embeddings separately
        print(f"\n   💾 Saving embeddings and heads...")
        if hasattr(model, 'get_input_embeddings'):
            emb = model.get_input_embeddings()
            torch.save(emb.state_dict(), model_dir / "embeddings.pt")
            emb_size = (model_dir / 'embeddings.pt').stat().st_size / (1024**2)
            print(f"      Embeddings: {emb_size:.1f}MB")

        # Save LM head if exists
        if hasattr(model, 'lm_head'):
            torch.save(model.lm_head.state_dict(), model_dir / "lm_head.pt")
            head_size = (model_dir / 'lm_head.pt').stat().st_size / (1024**2)
            print(f"      LM Head: {head_size:.1f}MB")

        # Save config
        config.save_pretrained(model_dir)

        # Save layer structure info
        total_size_mb = sum((model_dir / f"layer_{i}.pt").stat().st_size for i in range(num_layers)) / (1024**2)
        structure_info = {
            "model_id": model_id,
            "num_layers": num_layers,
            "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown",
            "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
            "total_size_mb": total_size_mb
        }

        with open(model_dir / "structure.json", "w") as f:
            json.dump(structure_info, f, indent=2)

        del model  # Free memory
        torch.cuda.empty_cache()

        print(f"\n✅ SHARDING COMPLETE")
        print(f"   Model: {model_id}")
        print(f"   Layers: {num_layers}")
        print(f"   Total size: {total_size_mb:.1f}MB")

        return num_layers

    def get_layer_path(self, model_id: str, layer_idx: int):
        """Get filesystem path to a layer file"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        return model_dir / f"layer_{layer_idx}.pt"

    def get_layer_metadata(self, model_id: str, layer_idx: int):
        """Get metadata for a specific layer"""
        model_dir = self.storage_path / model_id.replace("/", "_")
        metadata_path = model_dir / f"layer_{layer_idx}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

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
