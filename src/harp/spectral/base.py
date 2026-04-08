"""Abstract base class for spectral decomposers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sparse

from harp.interfaces import DecompositionMethod, ModalFrame


class AbstractSpectralDecomposer(ABC):
    """Abstract base for spectral decomposers."""

    @property
    @abstractmethod
    def method(self) -> DecompositionMethod: ...

    @abstractmethod
    def decompose(self, operator: sparse.csr_matrix, k: int) -> ModalFrame: ...

    def _validate_operator(self, op: sparse.csr_matrix) -> np.ndarray:
        if sparse.issparse(op):
            A = op.toarray().astype(float)
        else:
            A = np.asarray(op, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Operator must be square, got shape {A.shape}")
        return A

    def _truncate_modes(self, frame: ModalFrame, k: int) -> ModalFrame:
        K = frame.basis.shape[1]
        if k >= K:
            return frame
        order = np.argsort(-np.abs(frame.eigenvalues))[:k]
        return ModalFrame(
            basis=frame.basis[:, order],
            eigenvalues=frame.eigenvalues[order],
            method=frame.method,
            metadata=frame.metadata,
        )
