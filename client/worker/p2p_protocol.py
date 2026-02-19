"""
P2P protocol module for tensor serialization and deserialization.
Handles binary tensor transfer between workers.
"""

import base64
import json
import numpy as np
import torch
from typing import Dict, Any, Optional


class P2PProtocol:
    """
    Handles serialization and deserialization of tensors for P2P transfer.
    Provides methods for encoding/decoding tensors with metadata headers.
    """

    @staticmethod
    def tensor_to_bytes(tensor: torch.Tensor) -> tuple:
        """
        Convert a tensor to bytes and metadata for transmission.

        Returns:
            tuple: (bytes_data, dtype_str, shape_list)
        """
        # Handle BFloat16 -> Float conversion for numpy compatibility
        # Numpy crashes on BFloat16, so we must detach().cpu().float() first
        np_tensor = tensor.detach().cpu().float().contiguous().numpy()
        dtype_str = str(np_tensor.dtype)
        shape_json = json.dumps(list(np_tensor.shape))
        data_bytes = np_tensor.tobytes()
        return data_bytes, dtype_str, shape_json

    @staticmethod
    def bytes_to_tensor(
        data: bytes,
        dtype_str: str,
        shape: list,
        device: torch.device,
        target_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Reconstruct a tensor from bytes and metadata.

        Args:
            data: Raw bytes from the tensor
            dtype_str: NumPy dtype string (e.g., 'float32')
            shape: Tensor shape list
            device: Target device
            target_dtype: Target torch dtype

        Returns:
            Reconstructed tensor on target device
        """
        dtype = getattr(np, dtype_str)
        # copy() is needed to make the array writable/contiguous for torch
        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        tensor = torch.from_numpy(arr).to(device).to(target_dtype)
        return tensor

    @staticmethod
    def build_headers(payload_meta: Dict[str, Any]) -> Dict[str, str]:
        """
        Build HTTP headers for tensor transmission.

        Args:
            payload_meta: Dictionary with keys: job_id, type, expert_idx, layer_idx, target_layer_idx

        Returns:
            Dictionary of headers
        """
        headers = {
            "x-job-id": payload_meta["job_id"],
            "x-msg-type": payload_meta.get("type", "input"),
            "x-dtype": payload_meta.get("dtype", "float32"),
            "x-shape": payload_meta.get("shape", "[]")
        }

        # Add routing fields if present
        if "expert_idx" in payload_meta:
            headers["x-expert-idx"] = str(payload_meta["expert_idx"])
        if "layer_idx" in payload_meta:
            headers["x-layer-idx"] = str(payload_meta["layer_idx"])
        if "target_layer_idx" in payload_meta and payload_meta["target_layer_idx"] is not None:
            headers["x-target-layer-idx"] = str(payload_meta["target_layer_idx"])

        return headers

    @staticmethod
    def parse_headers(headers) -> Dict[str, Any]:
        """
        Parse HTTP headers for tensor reception.

        Args:
            headers: aiohttp headers object

        Returns:
            Dictionary with parsed values
        """
        job_id = headers.get("x-job-id")
        msg_type = headers.get("x-msg-type", "input")
        dtype_str = headers.get("x-dtype", "float16")
        shape = json.loads(headers.get("x-shape", "[]"))

        result = {
            "job_id": job_id,
            "msg_type": msg_type,
            "dtype": dtype_str,
            "shape": shape
        }

        # Parse optional routing headers
        expert_idx = headers.get("x-expert-idx")
        layer_idx = headers.get("x-layer-idx")
        target_layer_idx = headers.get("x-target-layer-idx")

        if expert_idx is not None:
            result["expert_idx"] = int(expert_idx)
        if layer_idx is not None:
            result["layer_idx"] = int(layer_idx)
        if target_layer_idx is not None:
            result["target_layer_idx"] = int(target_layer_idx)

        return result

    @staticmethod
    def build_payload_meta(
        job_id: str,
        msg_type: str = "input",
        dtype: str = "float32",
        shape: list = None,
        expert_idx: int = None,
        layer_idx: int = None,
        target_layer_idx: int = None
    ) -> Dict[str, Any]:
        """
        Build payload metadata dictionary.

        Args:
            job_id: Job identifier
            msg_type: Message type (input, expert_result)
            dtype: NumPy dtype string
            shape: Tensor shape
            expert_idx: Expert index (for MoE)
            layer_idx: Layer index
            target_layer_idx: Target layer index for routing

        Returns:
            Metadata dictionary
        """
        meta = {
            "job_id": job_id,
            "type": msg_type,
            "dtype": dtype,
            "shape": shape or []
        }

        if expert_idx is not None:
            meta["expert_idx"] = expert_idx
        if layer_idx is not None:
            meta["layer_idx"] = layer_idx
        if target_layer_idx is not None:
            meta["target_layer_idx"] = target_layer_idx

        return meta

    # -------------------------------------------------------------------------
    # CUDA IPC — zero-copy same-machine transfer
    # -------------------------------------------------------------------------

    @staticmethod
    def tensor_to_ipc_handle(tensor: torch.Tensor) -> dict:
        """
        Export a CUDA IPC handle for zero-copy same-machine process transfer.

        The returned dict is JSON-serializable (~200–500 bytes). The receiver
        calls ipc_handle_to_tensor() to reconstruct the tensor directly in its
        own CUDA context — no GPU→CPU copy, no dtype conversion, no network body.

        Only valid when torch.cuda.is_available() and tensor is on CUDA.
        Caller should fall back to tensor_to_bytes() if this raises.
        """
        tensor = tensor.detach().contiguous()
        storage = tensor.untyped_storage()
        raw = storage._share_cuda_()

        # _share_cuda_() returns a tuple that may contain bytes objects.
        # Encode bytes as base64 for JSON transport.
        serialized = []
        for item in raw:
            if isinstance(item, (bytes, bytearray)):
                serialized.append({"t": "b", "v": base64.b64encode(item).decode()})
            else:
                serialized.append({"t": "i", "v": item})

        return {
            "h": serialized,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }

    @staticmethod
    def ipc_handle_to_tensor(handle_data: dict, target_dtype: torch.dtype) -> torch.Tensor:
        """
        Reconstruct a tensor from a CUDA IPC handle dict.

        The resulting tensor shares CUDA memory with the sender — zero copy.
        CUDA ref-counts the underlying allocation, so the sender's Python object
        can be released safely once this returns.
        """
        raw = []
        for item in handle_data["h"]:
            if item["t"] == "b":
                raw.append(base64.b64decode(item["v"]))
            else:
                raw.append(item["v"])

        storage = torch.UntypedStorage._new_shared_cuda(*raw)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        src_dtype = dtype_map.get(handle_data["dtype"], torch.float16)
        shape = handle_data["shape"]
        stride = handle_data["stride"]

        # Reconstruct tensor view over shared CUDA storage — zero copy.
        # device is encoded in the storage (comes from the IPC handle).
        out = torch.empty(shape, dtype=src_dtype, device=storage.device)
        out.set_(storage, 0, shape, stride)

        return out.to(target_dtype) if out.dtype != target_dtype else out
