from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


USE_ALL_GPUS_ENV = "VARIFUSE_USE_ALL_GPUS"
REQUIRED_GPU_COUNT_ENV = "VARIFUSE_REQUIRED_GPU_COUNT"
MAX_GPU_COUNT_ENV = "VARIFUSE_MAX_GPU_COUNT"
GPU_NAME_CONTAINS_ENV = "VARIFUSE_GPU_NAME_CONTAINS"


class SafeDataParallel(nn.DataParallel):
    """DataParallel with a correct singleton-batch fallback.

    PyTorch 2.10 may construct a second replica with keyword arguments but no
    positional tensor when the leading batch dimension is smaller than the
    device count. Fair-ESM then receives ``repr_layers`` without ``tokens``.
    Running that rare undersized batch on the primary device is exact and
    avoids both the missing-argument crash and duplicated scientific rows.
    """

    @staticmethod
    def _leading_batch_size(
        inputs: tuple[Any, ...], module_kwargs: dict[str, Any]
    ) -> int | None:
        for value in (*inputs, *module_kwargs.values()):
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        return None

    def forward(self, *inputs: Any, **module_kwargs: Any) -> Any:
        batch_size = self._leading_batch_size(inputs, module_kwargs)
        if batch_size is not None and batch_size < len(self.device_ids):
            return self.module(*inputs, **module_kwargs)
        return super().forward(*inputs, **module_kwargs)


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _nonnegative_integer(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def selected_cuda_device_ids(device: str) -> list[int]:
    """Resolve and validate the CUDA devices assigned to this process.

    Local/CPU execution retains the historical single-device behaviour. The
    Kaggle runner opts into this contract with two required T4 devices, so a
    notebook cannot silently fall back to one GPU or CPU and produce a partly
    comparable publication run.
    """
    required = _nonnegative_integer(REQUIRED_GPU_COUNT_ENV, 0)
    use_all = _enabled(USE_ALL_GPUS_ENV)
    if device != "cuda":
        if required:
            raise RuntimeError(
                f"VariFuse requires {required} CUDA GPUs, but CUDA is unavailable"
            )
        return []

    available = int(torch.cuda.device_count())
    if required and available < required:
        raise RuntimeError(
            f"VariFuse requires {required} CUDA GPUs, but only {available} are visible"
        )
    if available < 1:
        raise RuntimeError("CUDA was selected but no CUDA device is visible")

    maximum = _nonnegative_integer(MAX_GPU_COUNT_ENV, 0)
    selected_count = available if use_all else 1
    if maximum:
        selected_count = min(selected_count, maximum)
    if required:
        selected_count = max(selected_count, required)
    if selected_count > available:
        raise RuntimeError(
            f"Requested {selected_count} CUDA GPUs, but only {available} are visible"
        )
    device_ids = list(range(selected_count))

    required_name = os.environ.get(GPU_NAME_CONTAINS_ENV, "").strip().casefold()
    if required_name:
        mismatches = [
            f"cuda:{index}={torch.cuda.get_device_name(index)!r}"
            for index in device_ids
            if required_name not in torch.cuda.get_device_name(index).casefold()
        ]
        if mismatches:
            raise RuntimeError(
                f"Selected GPUs must contain {required_name!r} in their names; "
                + ", ".join(mismatches)
            )
    return device_ids


def cuda_inventory(device: str) -> dict[str, Any]:
    """Return JSON-safe hardware metadata for manifests and preflight logs."""
    device_ids = selected_cuda_device_ids(device)
    devices: list[dict[str, Any]] = []
    for index in device_ids:
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_bytes": int(total_bytes),
                "free_bytes": int(free_bytes),
            }
        )
    return {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "selected_cuda_devices": device_ids,
        "data_parallel": len(device_ids) > 1,
        "devices": devices,
    }


def data_parallel(module: nn.Module, device: str) -> nn.Module:
    """Temporarily distribute a module while keeping checkpoints wrapper-free."""
    device_ids = selected_cuda_device_ids(device)
    if len(device_ids) <= 1:
        return module
    if isinstance(module, nn.DataParallel):
        return module
    return SafeDataParallel(
        module,
        device_ids=device_ids,
        output_device=device_ids[0],
    )
