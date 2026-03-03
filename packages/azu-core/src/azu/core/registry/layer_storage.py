"""
Layer storage module for sharding and storing model weights on the registry.
All architecture-specific logic is delegated to the model driver system.
"""

import torch
import os
import sys
import json
import gc
import shutil
import glob
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from azu.shared.model_drivers import get_driver


class LayerStore:
    def __init__(self, storage_path="/data/layers"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def has_model(self, model_id: str) -> bool:
        """Check if a model is already fully sharded and stored."""
        sanitized = model_id.replace("/", "_")
        return (self.storage_path / sanitized / "structure.json").exists()

    # =========================================================================
    # Layer discovery
    # =========================================================================

    def _find_layers(self, model, driver):
        """
        Find transformer decoder layers using driver-provided paths,
        falling back to a recursive ModuleList scan if explicit paths miss.
        """
        import torch.nn as nn

        print("   🔍 Searching for layers in model structure...")

        def _is_valid(obj):
            try:
                length = len(obj)
            except TypeError:
                return False
            if length == 0:
                return False
            first = obj[0] if hasattr(obj, "__getitem__") else next(iter(obj), None)
            return isinstance(first, nn.Module)

        # Pass 1: driver-provided explicit paths
        for attr_path in driver.layer_module_paths:
            try:
                obj = model
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if _is_valid(obj):
                    print(f"   ✅ Found layers at: {attr_path}")
                    return obj, attr_path
            except AttributeError:
                continue

        # Pass 2: recursive scan — pick the largest valid ModuleList
        print("   🔍 Explicit paths exhausted, falling back to recursive scan...")
        best_obj, best_path, best_len = None, None, 0
        for name, module in model.named_modules():
            if not isinstance(module, (nn.ModuleList, nn.Sequential)):
                continue
            if not _is_valid(module):
                continue
            if len(module) > best_len:
                best_len = len(module)
                best_obj = module
                best_path = name

        if best_obj is not None:
            print(f"   ✅ Found layers via scan at: {best_path} ({best_len} layers)")
            return best_obj, best_path

        raise ValueError(
            f"Could not find layers. Model keys: {list(model.state_dict().keys())[:5]}..."
        )

    # =========================================================================
    # Weight loading helpers
    # =========================================================================

    def _load_tensor_for_key(self, key, model_path, index, loaded_shards):
        """Load a specific tensor key from checkpoint shards (with shard cache)."""
        filename = index.get(key)
        if not filename:
            return None
        filepath = os.path.join(model_path, filename)
        if filepath not in loaded_shards:
            loaded_shards.clear()
            gc.collect()
            print(f"      📖 Loading shard: {filename}")
            if filename.endswith(".safetensors"):
                loaded_shards[filepath] = load_safetensors(filepath)
            else:
                loaded_shards[filepath] = torch.load(filepath, map_location="cpu")
        return loaded_shards[filepath][key]

    def _save_module(self, module, prefix, model_path, index, output_path, loaded_shards, driver):
        """
        Reconstruct a module's state_dict from the checkpoint and save it.
        Uses driver.resolve_weight_key() for all aliasing; falls back to
        suffix-match only as a last resort.
        """
        state_dict = {}
        for name, _ in module.named_parameters(recurse=True):
            global_key = f"{prefix}.{name}"
            resolved = driver.resolve_weight_key(global_key, index)
            if resolved in index:
                state_dict[name] = self._load_tensor_for_key(resolved, model_path, index, loaded_shards)
            else:
                found = False
                for key in index:
                    if key.endswith(f".{name}") or key == name:
                        state_dict[name] = self._load_tensor_for_key(key, model_path, index, loaded_shards)
                        print(f"      🔍 Found key via suffix match: {key}")
                        found = True
                        break
                if not found:
                    print(f"      ⚠️ Missing key {global_key} (resolved: {resolved})")

        state_dict = {k: v.contiguous() for k, v in state_dict.items() if v is not None}
        save_safetensors(state_dict, output_path)
        return output_path.stat().st_size / (1024 ** 2)

    # =========================================================================
    # Main sharding entry point
    # =========================================================================

    def shard_model(self, model_id: str, hf_token: str):
        print(f"\n🔪 SHARDING {model_id} (Streaming Mode)")

        total, _, free = shutil.disk_usage(self.storage_path)
        if free < 50 * (2 ** 30):
            print(f"   ⚠️ Low disk space: {free // (2 ** 30)}GB free")

        from huggingface_hub import snapshot_download

        try:
            # ── 1. Download ──────────────────────────────────────────────────
            print("   📥 Fetching model artifacts (snapshot)...")
            model_path = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py"],
            )
            print(f"   ✅ Model available at: {model_path}")

            # ── 2. Config + driver ───────────────────────────────────────────
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            driver = get_driver(config)
            driver.normalize_config(config)
            print(f"   🔧 Driver: {type(driver).__name__}")

            # ── 3. Weight index ──────────────────────────────────────────────
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                index_path = os.path.join(model_path, "pytorch_model.bin_index.json")

            weight_map = None
            if os.path.exists(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
            else:
                print("   ℹ️ Single file model detected.")
                files = (glob.glob(os.path.join(model_path, "*.safetensors")) or
                         glob.glob(os.path.join(model_path, "*.bin")))
                if not files:
                    raise RuntimeError(f"No model files found in {model_path}")
                file_name = os.path.basename(files[0])
                file_path = os.path.join(model_path, file_name)
                print(f"   🔍 Reading keys directly from {file_name}...")
                if file_name.endswith(".safetensors"):
                    from safetensors import safe_open
                    with safe_open(file_path, framework="pt", device="cpu") as sf:
                        checkpoint_keys = list(sf.keys())
                    weight_map = {k: file_name for k in checkpoint_keys}
                    print(f"   ✅ Found {len(checkpoint_keys)} keys in checkpoint")
                else:
                    with init_empty_weights():
                        temp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                    weight_map = {k: file_name for k in temp_model.state_dict().keys()}
                    del temp_model

            # ── 4. Meta model ────────────────────────────────────────────────
            print("   🏗️ Building Meta Model...")
            _meta_loaders = []
            try:
                from transformers import AutoModelForCausalLM as _CausalLM
                _meta_loaders.append(_CausalLM)
            except ImportError:
                pass
            try:
                from transformers import AutoModelForVision2Seq as _V2S
                _meta_loaders.append(_V2S)
            except ImportError:
                pass
            try:
                from transformers import AutoModel as _AM
                _meta_loaders.append(_AM)
            except ImportError:
                pass

            model = None
            _last_exc = None
            for _loader_cls in _meta_loaders:
                try:
                    with init_empty_weights():
                        model = _loader_cls.from_config(config, trust_remote_code=True)
                    break
                except Exception as _e:
                    _last_exc = _e

            if model is None:
                raise RuntimeError(
                    f"Could not build meta model for {model_id}. Last error: {_last_exc}"
                )

            if weight_map is None:
                weight_map = {k: files[0] for k in model.state_dict().keys()}

            # ── 5. Output directory ──────────────────────────────────────────
            out_dir = self.storage_path / model_id.replace("/", "_")
            out_dir.mkdir(exist_ok=True, parents=True)

            # ── 6. Layer discovery & prefix reconciliation ───────────────────
            layers_obj, layer_prefix_base = self._find_layers(model, driver)
            num_layers = len(layers_obj)

            layer_prefix_base = driver.reconcile_layer_weight_prefix(layer_prefix_base, weight_map)
            print(f"   🔧 Layer weight prefix: {layer_prefix_base}")

            _backbone_weight_prefix = layer_prefix_base.rsplit(".layers", 1)[0]
            _backbone_module = model
            for _part in _backbone_weight_prefix.split("."):
                _backbone_module = getattr(_backbone_module, _part, _backbone_module)

            # ── 7. Shard layers ──────────────────────────────────────────────
            layer_metadata = []
            total_size_mb = 0
            loaded_shards = {}

            print(f"   💾 Processing {num_layers} layers...")

            for i in range(num_layers):
                layer = layers_obj[i]
                layer_prefix = f"{layer_prefix_base}.{i}"
                is_moe = driver.is_moe(config)
                print(f"   Layer {i}: {'MoE' if is_moe else 'Dense'}")

                if is_moe:
                    # ── Shared (attention + norms) ────────────────────────────
                    shared_file = out_dir / f"layer_{i}_shared.safetensors"
                    shared_state = {}
                    for comp_attr, comp_prefix in [
                        ("input_layernorm",         f"{layer_prefix}.input_layernorm"),
                        ("self_attn",               f"{layer_prefix}.self_attn"),
                        ("post_attention_layernorm", f"{layer_prefix}.post_attention_layernorm"),
                    ]:
                        comp_mod = getattr(layer, comp_attr, None)
                        if comp_mod is None:
                            continue
                        for pname, _ in comp_mod.named_parameters(recurse=True):
                            raw = f"{comp_prefix}.{pname}"
                            resolved = driver.resolve_weight_key(raw, weight_map)
                            if resolved in weight_map:
                                shared_state[f"{comp_attr}.{pname}"] = self._load_tensor_for_key(
                                    resolved, model_path, weight_map, loaded_shards
                                )

                    shared_size_mb = 0
                    if shared_state:
                        shared_state = {k: v.contiguous() for k, v in shared_state.items() if v is not None}
                        save_safetensors(shared_state, shared_file)
                        shared_size_mb = shared_file.stat().st_size / (1024 ** 2)
                        print(f"      ✅ Shared saved ({shared_size_mb:.2f}MB)")
                    else:
                        print(f"      ⚠️ WARNING: No shared weights found for layer {i}")

                    # ── Navigate to MoE block ─────────────────────────────────
                    moe_rel_path = driver.moe_router_path(layer)
                    moe_module = layer
                    if moe_rel_path:
                        for part in moe_rel_path.split("."):
                            moe_module = getattr(moe_module, part)
                        moe_prefix = f"{layer_prefix}.{moe_rel_path}"
                    else:
                        moe_prefix = layer_prefix

                    # ── Router ────────────────────────────────────────────────
                    router_attr = "gate" if hasattr(moe_module, "gate") else "router"
                    router_file = out_dir / f"layer_{i}_router.safetensors"
                    router_size_mb = 0
                    router_state = {}

                    direct_key = f"{moe_prefix}.{router_attr}.weight"
                    resolved = driver.resolve_weight_key(direct_key, weight_map)
                    if resolved in weight_map:
                        router_state["weight"] = self._load_tensor_for_key(
                            resolved, model_path, weight_map, loaded_shards
                        )
                    else:
                        for key in weight_map:
                            if f"layers.{i}." in key and f"{router_attr}.weight" in key:
                                router_state["weight"] = self._load_tensor_for_key(
                                    key, model_path, weight_map, loaded_shards
                                )
                                print(f"      🔍 Found router via pattern: {key}")
                                break

                    if router_state:
                        router_state = {k: v.contiguous() for k, v in router_state.items() if v is not None}
                        save_safetensors(router_state, router_file)
                        if not router_file.exists():
                            raise RuntimeError(f"Router file not created: {router_file}")
                        router_size_mb = router_file.stat().st_size / (1024 ** 2)
                        print(f"      ✅ Router saved ({router_size_mb:.2f}MB)")
                    else:
                        sample = [k for k in weight_map if f"layers.{i}." in k][:5]
                        raise RuntimeError(
                            f"Router missing for MoE layer {i}. Tried: {direct_key}. "
                            f"Sample keys: {sample}"
                        )

                    # ── Experts ───────────────────────────────────────────────
                    num_experts = driver.num_experts(config)
                    expert_size_acc = 0

                    sample_exp = [k for k in weight_map if f"layers.{i}." in k and "expert" in k.lower()]
                    if sample_exp:
                        print(f"      🔍 Sample expert keys: {sample_exp[:3]}")

                    for exp_idx in range(num_experts):
                        expert_file = out_dir / f"layer_{i}_expert_{exp_idx}.safetensors"
                        expert_prefix = f"{moe_prefix}.experts.{exp_idx}"
                        expert_state = {}

                        # Collect all weight keys for this expert
                        possible_prefixes = {expert_prefix}
                        # Also check resolved alias
                        probe_resolved = driver.resolve_weight_key(f"{expert_prefix}.weight", weight_map)
                        if "." in probe_resolved:
                            possible_prefixes.add(probe_resolved.rsplit(".", 1)[0])

                        for key in weight_map:
                            for pfx in possible_prefixes:
                                if key.startswith(pfx + "."):
                                    param_name = key[len(pfx) + 1:]
                                    expert_state[param_name] = self._load_tensor_for_key(
                                        key, model_path, weight_map, loaded_shards
                                    )
                                    break

                        if not expert_state:
                            raise RuntimeError(f"Expert {exp_idx} has no weights in checkpoint")

                        expert_state = {k: v.contiguous() for k, v in expert_state.items() if v is not None}
                        save_safetensors(expert_state, expert_file)
                        sz = expert_file.stat().st_size / (1024 ** 2)
                        expert_size_acc += sz

                        if exp_idx % 4 == 0:
                            sys.stdout.write(".")
                        sys.stdout.flush()

                    print(f" Layer {i} MoE done ({num_experts} experts)")

                    total_moe_size = expert_size_acc + shared_size_mb + router_size_mb
                    avg_expert_size = expert_size_acc / max(1, num_experts)
                    layer_metadata.append({
                        "layer_idx": i,
                        "type": "moe",
                        "num_experts": num_experts,
                        "shared_size_mb": shared_size_mb,
                        "router_size_mb": router_size_mb,
                        "expert_size_mb": avg_expert_size,
                        "size_mb": total_moe_size,
                    })
                    total_size_mb += total_moe_size

                else:
                    # ── Dense ─────────────────────────────────────────────────
                    dense_file = out_dir / f"layer_{i}_dense.safetensors"
                    sz = self._save_module(
                        layer, layer_prefix, model_path, weight_map, dense_file, loaded_shards, driver
                    )
                    if not dense_file.exists():
                        raise RuntimeError(f"Dense layer file not created: {dense_file}")
                    layer_metadata.append({
                        "layer_idx": i,
                        "type": "dense",
                        "size_mb": sz,
                        "num_experts": 0,
                    })
                    total_size_mb += sz
                    print(f"   Layer {i} Dense done ({sz:.2f}MB)")

                loaded_shards.clear()
                gc.collect()

            # ── 8. Shared components ─────────────────────────────────────────
            print("   💾 Saving embeddings, final norm & head...")

            # A. Embeddings
            emb_attr = driver.embedding_module_attr
            _emb_module = getattr(_backbone_module, emb_attr, None)
            if _emb_module is not None:
                emb_file = out_dir / "embeddings.safetensors"
                self._save_module(
                    _emb_module,
                    f"{_backbone_weight_prefix}.{emb_attr}",
                    model_path, weight_map, emb_file, loaded_shards, driver,
                )
                if not emb_file.exists():
                    raise RuntimeError(f"Embeddings file not created: {emb_file}")
                print("      ✅ Embeddings saved")
            else:
                print(f"      ⚠️ No {emb_attr} on backbone ({_backbone_weight_prefix})")

            # B. Final norm
            norm_attr = driver.final_norm_module_attr
            final_norm_module = getattr(_backbone_module, norm_attr, None)
            final_norm_prefix = f"{_backbone_weight_prefix}.{norm_attr}" if final_norm_module else None

            if final_norm_module is None and hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                final_norm_module = model.transformer.ln_f
                final_norm_prefix = "transformer.ln_f"

            if final_norm_module:
                norm_file = out_dir / "final_norm.safetensors"
                self._save_module(
                    final_norm_module, final_norm_prefix,
                    model_path, weight_map, norm_file, loaded_shards, driver,
                )
                if not norm_file.exists():
                    raise RuntimeError(f"Final norm file not created: {norm_file}")
                print("      ✅ Final norm saved")
            else:
                print("      ⚠️ No final norm found")

            # C. LM head — driver first, then generic scan, then structural fallback
            lm_head_result = driver.resolve_lm_head(model, weight_map, _backbone_weight_prefix)

            if lm_head_result is None:
                for wk in weight_map:
                    if not wk.endswith("lm_head.weight"):
                        continue
                    candidate_prefix = wk[: -len(".weight")]
                    _mod = None
                    # Try direct navigation
                    try:
                        _mod = model
                        for _p in candidate_prefix.split("."):
                            _mod = getattr(_mod, _p)
                    except AttributeError:
                        _mod = None
                    # Try VLM structural paths
                    if _mod is None:
                        for _path in [
                            "language_model.lm_head",
                            "model.language_model.lm_head",
                            "model.lm_head",
                            f"{_backbone_weight_prefix}.lm_head",
                        ]:
                            try:
                                _vlm = model
                                for _p in _path.split("."):
                                    _vlm = getattr(_vlm, _p)
                                _mod = _vlm
                                print(f"      🔧 lm_head module at '{_path}', "
                                      f"checkpoint prefix '{candidate_prefix}'")
                                break
                            except AttributeError:
                                continue
                    if _mod is not None:
                        lm_head_result = (_mod, candidate_prefix)
                        break

            if lm_head_result is None:
                for _mod, _pfx in [
                    (_backbone_module, f"{_backbone_weight_prefix}.lm_head"),
                    (model, "lm_head"),
                    (getattr(model, "language_model", None), "language_model.lm_head"),
                ]:
                    if _mod is not None and getattr(_mod, "lm_head", None) is not None:
                        lm_head_result = (getattr(_mod, "lm_head"), _pfx)
                        break

            if lm_head_result:
                _lm_mod, _lm_prefix = lm_head_result
                head_file = out_dir / "lm_head.safetensors"
                self._save_module(
                    _lm_mod, _lm_prefix,
                    model_path, weight_map, head_file, loaded_shards, driver,
                )
                if not head_file.exists():
                    raise RuntimeError(f"LM head file not created: {head_file}")
                print(f"      ✅ LM head saved (prefix: {_lm_prefix})")
            else:
                print("      ⚠️ No lm_head found in weight_map or model structure")

            # ── 9. Metadata & tokenizer ──────────────────────────────────────
            print("   💾 Saving metadata and tokenizer assets...")
            config.save_pretrained(out_dir)

            try:
                from transformers import AutoTokenizer as _Tok
                tokenizer = _Tok.from_pretrained(model_path, token=hf_token, trust_remote_code=True)
                tokenizer.save_pretrained(out_dir)
            except Exception as e:
                print(f"      ⚠️ tokenizer save_pretrained failed: {e}")

            for ext in ["*.json", "*.model", "*.txt", "*.py"]:
                for src_path in Path(model_path).rglob(ext):
                    fname = os.path.basename(src_path)
                    if fname.startswith(".") or "index" in fname:
                        continue
                    dst = out_dir / fname
                    if not dst.exists():
                        shutil.copy2(src_path, dst)
                        print(f"      📦 Manually copied: {fname}")

            # ── 10. structure.json ───────────────────────────────────────────
            _hidden_size = (
                getattr(config, "hidden_size", None)
                or getattr(getattr(config, "text_config", None), "hidden_size", None)
                or getattr(getattr(config, "language_config", None), "hidden_size", None)
                or getattr(getattr(config, "llm_config", None), "hidden_size", None)
            )
            if _hidden_size is None:
                raise RuntimeError(f"Could not resolve hidden_size from config for {model_id}")

            structure = {
                "model_id": model_id,
                "num_layers": num_layers,
                "architecture": config.architectures[0],
                "driver": type(driver).__name__,
                "hidden_size": _hidden_size,
                "total_size_mb": total_size_mb,
                "is_moe": any(x["type"] == "moe" for x in layer_metadata),
                "num_experts_per_tok": getattr(config, "num_experts_per_tok", 2),
                "layer_metadata": layer_metadata,
            }

            with open(out_dir / "structure.json", "w") as f:
                json.dump(structure, f, indent=2)

            for cf in ["structure.json", "config.json"]:
                if not (out_dir / cf).exists():
                    raise RuntimeError(f"Critical file missing: {cf}")

            all_files = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.json"))
            print(f"   📁 Created {len(all_files)} files in {out_dir}")
            print(f"✅ Sharding Complete. Total Size: {total_size_mb:.1f}MB")
            return num_layers

        except Exception as e:
            print(f"❌ Sharding Error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if "loaded_shards" in locals():
                loaded_shards.clear()  # type: ignore[name-defined]
            if "model" in locals():
                del model  # type: ignore[name-defined]
            gc.collect()
