"""Dynamic Mode Decomposition (exact DMD)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from harp.interfaces import DecompositionMethod, ModalFrame
from harp.spectral.base import AbstractSpectralDecomposer
from harp.compute.linalg import svd as _compute_svd


def _to_real_modes(
    eigenvalues: np.ndarray,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Convert complex DMD modes to a real-valued basis via conjugate-pair splitting."""
    k = len(eigenvalues)
    imag_tol = 1e-8 * (float(np.abs(eigenvalues).max()) + 1e-12)

    real_cols: list[np.ndarray] = []
    real_eigs: list[complex] = []
    mode_pairs: list[tuple[int, ...]] = []
    col_idx = 0
    i = 0

    while i < k:
        lam = eigenvalues[i]
        phi = modes[:, i]

        if i + 1 < k and abs(lam.imag) > imag_tol:
            lam_next = eigenvalues[i + 1]
            avg_mag = (abs(lam) + abs(lam_next)) / 2.0 + 1e-12
            if abs(lam_next - np.conj(lam)) < 1e-4 * avg_mag:
                if lam.imag < 0:
                    lam, lam_next = lam_next, lam
                    phi = modes[:, i + 1]
                real_cols.append(phi.real)
                real_cols.append(phi.imag)
                real_eigs.extend([complex(lam), complex(np.conj(lam))])
                mode_pairs.append((col_idx, col_idx + 1))
                col_idx += 2
                i += 2
                continue

        real_cols.append(phi.real)
        real_eigs.append(complex(lam))
        mode_pairs.append((col_idx,))
        col_idx += 1
        i += 1

    N = modes.shape[0]
    if real_cols:
        basis = np.column_stack(real_cols)
    else:
        basis = np.zeros((N, 0))
    return basis, np.array(real_eigs), mode_pairs


class ExactDMDDecomposer(AbstractSpectralDecomposer):
    """Exact DMD decomposer."""

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.DMD

    def _validate_operator(self, op) -> np.ndarray:
        if sparse.issparse(op):
            A = op.toarray().astype(float)
        else:
            A = np.asarray(op, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"Operator must be 2-D, got shape {A.shape}")
        return A

    def decompose(self, operator, k: int) -> ModalFrame:
        snapshots = self._validate_operator(operator)
        return self._dmd_from_snapshots(snapshots, k)

    def decompose_snapshots(self, snapshots: np.ndarray, k: int) -> ModalFrame:
        snapshots = np.asarray(snapshots, dtype=float)
        return self._dmd_from_snapshots(snapshots, k)

    def _dmd_from_snapshots(self, snapshots: np.ndarray, k: int) -> ModalFrame:
        N, T = snapshots.shape
        if T < 2:
            raise ValueError(f"Need at least 2 snapshots, got T={T}")
        X = snapshots[:, :-1]
        Y = snapshots[:, 1:]

        U, S, Vh = _compute_svd(X)
        k_svd = min(k, len(S))
        U_r = U[:, :k_svd]
        S_r = S[:k_svd]
        Vh_r = Vh[:k_svd, :]

        S_inv = np.diag(1.0 / S_r)
        Atilde = U_r.T @ Y @ Vh_r.T @ S_inv

        eigenvalues, W = np.linalg.eig(Atilde)

        safe_eigs = np.where(np.abs(eigenvalues) > 1e-12, eigenvalues, 1.0)
        modes = (Y @ Vh_r.T @ S_inv @ W) / safe_eigs[np.newaxis, :]

        order = np.argsort(-np.abs(eigenvalues))
        k_actual = min(k, k_svd)
        idx = order[:k_actual]

        basis, eigs, mode_pairs = _to_real_modes(eigenvalues[idx], modes[:, idx])

        return ModalFrame(
            basis=basis,
            eigenvalues=eigs,
            method=self.method,
            metadata={
                "Atilde": Atilde,
                "U": U_r,
                "S": S_r,
                "Vh": Vh_r,
                "full_eigenvalues": eigenvalues,
                "mode_pairs": mode_pairs,
            },
        )
