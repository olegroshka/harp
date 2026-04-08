"""Linear algebra operations — CPU and optional CUDA via PyTorch."""

import numpy as np
from numpy.typing import NDArray
from .torch_ops import ensure_tensor, to_numpy, get_device, _HAS_TORCH


def svd(A: NDArray, k: int | None = None) -> tuple[NDArray, NDArray, NDArray]:
    """SVD with optional truncation. Uses torch if available, else numpy."""
    if _HAS_TORCH:
        import torch
        A_t = ensure_tensor(A)
        U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
        if k is not None:
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        return to_numpy(U), to_numpy(S), to_numpy(Vh)
    else:
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        if k is not None:
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        return U, S, Vh
