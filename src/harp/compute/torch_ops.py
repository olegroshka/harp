"""PyTorch-based compute operations with automatic CPU/CUDA dispatch."""

import os
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@lru_cache(maxsize=1)
def get_device(force: str | None = None):
    if not _HAS_TORCH:
        logger.info("HARP compute: torch not available, using numpy fallback")
        return None

    if force is not None:
        device_str = force
    else:
        device_str = os.environ.get("SMIM_DEVICE", "auto")

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            logger.info(f"HARP compute: using CUDA device {name}")
        else:
            device = torch.device("cpu")
            logger.info("HARP compute: using CPU (no CUDA available)")
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("HARP compute: using CPU (forced)")
    else:
        device = torch.device(device_str)
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device.index or 0)
            logger.info(f"HARP compute: using CUDA device {name}")
    return device


def ensure_tensor(arr, device=None, dtype=None):
    if not _HAS_TORCH:
        return np.asarray(arr, dtype=float)

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = get_device()
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def to_numpy(t):
    if not _HAS_TORCH:
        return np.asarray(t)
    return t.detach().cpu().numpy()
