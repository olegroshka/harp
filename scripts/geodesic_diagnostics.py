#!/usr/bin/env python
"""
Iteration 6.3 Gate A — Is the Rotation Predictable?

Extracts the quarterly DMD basis sequence U_1,...,U_T, computes rotation
diagnostics, tests temporal structure, and evaluates 10 subspace prediction
models (P0-P9) against persistence.

Kill Rule A: if no model P1-P9 beats persistence P0 on projector Frobenius
error or chordal distance with paired-t CI excluding zero.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_3_gate_a.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes, logm, expm
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.spectral.dmd import ExactDMDDecomposer

INTENSITIES_PATH = PROJECT_ROOT / "data" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_DEFAULT = 8
K_MAX = 15
EWM_HL = 12
T_YR = 5


# ══════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════

def load_panel():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    actor_ids = [a["actor_id"] for a in registry["actors"]]
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel[sorted(set(actor_ids) & set(panel.columns))]


def ewm_demean(obs, hl=EWM_HL):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


# ══════════════════════════════════════════════════════════════════════
#  Grassmannian operations
# ══════════════════════════════════════════════════════════════════════

def principal_angles(U1, U2):
    """All K principal angles between subspaces span(U1) and span(U2).

    Returns angles in radians, sorted descending (largest angle first).
    """
    K = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :K])
    Q2, _ = np.linalg.qr(U2[:, :K])
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svals, -1.0, 1.0))
    return np.sort(angles)[::-1]  # largest first


def chordal_distance(U1, U2):
    """Chordal distance: ||sin(theta)||_2."""
    angles = principal_angles(U1, U2)
    return float(np.linalg.norm(np.sin(angles)))


def projector_frobenius(U1, U2):
    """||P1 - P2||_F where P = U U^T."""
    K = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :K])
    Q2, _ = np.linalg.qr(U2[:, :K])
    # ||P1 - P2||_F^2 = 2 sum sin^2(theta_k)
    angles = principal_angles(Q1, Q2)
    return float(np.sqrt(2 * np.sum(np.sin(angles) ** 2)))


def subspace_correlation(U1, U2):
    """1 - d_chord^2 / K.  Range [0, 1]."""
    K = min(U1.shape[1], U2.shape[1])
    d = chordal_distance(U1, U2)
    return float(1.0 - d ** 2 / K)


def grassmann_log(U_base, U_target):
    """Log map on Gr(K,N): tangent vector at U_base pointing toward U_target.

    Returns Delta (N x K) with U_base^T @ Delta = 0.
    """
    K = U_base.shape[1]
    M = U_base.T @ U_target  # K x K

    # SVD of the overlap matrix
    V, sigma, Wt = np.linalg.svd(M, full_matrices=False)
    W = Wt.T
    theta = np.arccos(np.clip(sigma, -1.0, 1.0))

    # Aligned target and base in principal coordinates
    U_target_W = U_target @ W     # N x K
    U_base_V = U_base @ V         # N x K

    # Direction vectors in the complement (one per principal angle)
    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) > 1e-10, sin_theta, 1.0)
    Q_cs = (U_target_W - U_base_V * np.cos(theta)[np.newaxis, :]) / safe_sin[np.newaxis, :]

    # Zero out directions with negligible angle (no rotation)
    mask = np.abs(theta) < 1e-10
    Q_cs[:, mask] = 0.0

    # Tangent vector: scale by angle, rotate back to original coordinates
    Delta = (Q_cs * theta[np.newaxis, :]) @ V.T
    return Delta


def grassmann_exp(U_base, Delta):
    """Exp map on Gr(K,N): move from U_base along tangent vector Delta.

    Returns U_new (N x K) with orthonormal columns.
    """
    K = U_base.shape[1]

    # Thin SVD of Delta
    Q_d, sigma_d, Vd_t = np.linalg.svd(Delta, full_matrices=False)
    r = len(sigma_d)
    Vd = Vd_t.T  # K x r

    cos_s = np.cos(sigma_d)
    sin_s = np.sin(sigma_d)

    # Geodesic at t=1
    U_new = (U_base @ Vd * cos_s[np.newaxis, :] + Q_d * sin_s[np.newaxis, :]) @ Vd.T
    # Add the unrotated components (principal directions not in Delta)
    U_new += U_base @ (np.eye(K) - Vd @ Vd.T)

    # QR for numerical stability
    U_new, _ = np.linalg.qr(U_new)
    return U_new


def karcher_mean(Us, max_iter=30, tol=1e-8, step=1.0):
    """Fréchet mean on Gr(K,N) via gradient descent in tangent space."""
    U_mean = Us[0].copy()
    for iteration in range(max_iter):
        deltas = [grassmann_log(U_mean, U) for U in Us]
        delta_avg = np.mean(deltas, axis=0)
        norm = np.linalg.norm(delta_avg, 'fro')
        if norm < tol:
            break
        U_mean = grassmann_exp(U_mean, step * delta_avg)
    return U_mean


# ══════════════════════════════════════════════════════════════════════
#  SO(K) rotation operations
# ══════════════════════════════════════════════════════════════════════

def procrustes_rotation(U_prev, U_curr):
    """Extract Procrustes rotation R such that U_prev @ R ≈ U_curr."""
    K = min(U_prev.shape[1], U_curr.shape[1])
    Q1, _ = np.linalg.qr(U_prev[:, :K])
    Q2, _ = np.linalg.qr(U_curr[:, :K])
    R, _ = orthogonal_procrustes(Q1, Q2)
    return R


def rotation_log(R):
    """Matrix logarithm of rotation R ∈ SO(K) → skew-symmetric ω ∈ so(K)."""
    try:
        omega = logm(R.astype(complex)).real
        # Make exactly skew-symmetric
        omega = (omega - omega.T) / 2
        return omega
    except Exception:
        return np.zeros_like(R)


def rotation_exp(omega):
    """Matrix exponential of skew-symmetric ω → rotation R ∈ SO(K)."""
    R = expm(omega).real
    # Re-orthogonalise via SVD
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def rotation_angle(R):
    """Total rotation angle of R ∈ SO(K) in radians."""
    K = R.shape[0]
    cos_total = (np.trace(R) - 1) / (K - 1) if K > 1 else (np.trace(R) - 1)
    cos_total = np.clip(cos_total, -1.0, 1.0)
    return float(np.arccos(cos_total))


# ══════════════════════════════════════════════════════════════════════
#  Step 1: Extract basis sequence
# ══════════════════════════════════════════════════════════════════════

def extract_basis_sequence(panel, K=K_DEFAULT, t_yr=T_YR, ewm_hl=EWM_HL):
    """Extract DMD basis U_t at each quarter, Procrustes-aligned to predecessor.

    Returns list of (quarter_date, U_t) pairs.
    """
    all_quarters = pd.date_range("2010-01-01", "2024-12-31", freq="QS")
    sequence = []
    prev_U = None

    for q_end in all_quarters:
        q_start = q_end - pd.DateOffset(years=t_yr)
        if q_start < pd.Timestamp("2005-01-01"):
            continue

        chunk = panel[(panel.index >= q_start) & (panel.index <= q_end)]
        valid = chunk.columns[chunk.notna().any()]
        chunk = chunk[valid].fillna(chunk[valid].mean())
        N = len(valid)
        vals = chunk.values.astype(np.float64)

        if vals.shape[0] < 4 or N < 10:
            prev_U = None
            continue

        om = ewm_demean(vals, ewm_hl)
        dm = vals - om

        try:
            mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, N))
            ka = min(K, N - 2, mf.K)
            U = mf.metadata["U"][:, :ka].real.copy()
        except Exception:
            prev_U = None
            continue

        # QR-orthonormalise
        U, _ = np.linalg.qr(U)

        # Column-sign alignment: flip column signs to match predecessor.
        # This fixes SVD sign ambiguity WITHOUT removing the true rotation.
        # Full Procrustes would collapse consecutive bases to near-identity.
        if prev_U is not None and prev_U.shape == U.shape:
            signs = np.sign(np.sum(U * prev_U, axis=0))
            signs[signs == 0] = 1.0
            U = U * signs[np.newaxis, :]

        sequence.append((q_end, U, N))
        prev_U = U

    return sequence


# ══════════════════════════════════════════════════════════════════════
#  Step 2: Rotation diagnostics
# ══════════════════════════════════════════════════════════════════════

def compute_rotation_diagnostics(sequence):
    """Compute rotation matrices, principal angles, geodesic distances."""
    rotations = []

    for i in range(1, len(sequence)):
        q_prev, U_prev, _ = sequence[i - 1]
        q_curr, U_curr, _ = sequence[i]

        if U_prev.shape != U_curr.shape:
            continue

        K = U_prev.shape[1]
        R = procrustes_rotation(U_prev, U_curr)
        angles = principal_angles(U_prev, U_curr)
        d_geodesic = float(np.linalg.norm(angles))
        d_deg = float(np.degrees(d_geodesic))
        omega = rotation_log(R)

        rotations.append({
            "quarter": q_curr,
            "R": R,
            "omega": omega,
            "angles_rad": angles,
            "angles_deg": np.degrees(angles),
            "d_geodesic_rad": d_geodesic,
            "d_geodesic_deg": d_deg,
            "mean_angle_deg": float(np.degrees(np.mean(angles))),
        })

    return rotations


def compute_axis_stability(rotations):
    """Measure stability of dominant rotation plane across time."""
    omegas = [r["omega"] for r in rotations]
    cosines = []
    for i in range(1, len(omegas)):
        o1 = omegas[i - 1].ravel()
        o2 = omegas[i].ravel()
        n1, n2 = np.linalg.norm(o1), np.linalg.norm(o2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosines.append(float(np.dot(o1, o2) / (n1 * n2)))
    return np.array(cosines)


# ══════════════════════════════════════════════════════════════════════
#  Step 3: Temporal structure tests
# ══════════════════════════════════════════════════════════════════════

def acf(x, max_lag=8):
    """Sample autocorrelation function."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.sum(x ** 2) / n
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    result = np.zeros(max_lag + 1)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        result[lag] = np.sum(x[:n - lag] * x[lag:]) / (n * var)
    return result


def ljung_box(x, max_lag=8):
    """Ljung-Box Q statistic. H0: x is white noise."""
    n = len(x)
    acf_vals = acf(x, max_lag)
    Q = 0.0
    for k in range(1, min(max_lag + 1, n)):
        Q += acf_vals[k] ** 2 / (n - k)
    Q *= n * (n + 2)
    # p-value from chi2(max_lag)
    from scipy.stats import chi2
    p = 1.0 - chi2.cdf(Q, df=max_lag)
    return float(Q), float(p)


def variance_ratio(x, q=4):
    """Variance ratio VR(q). VR=1 for random walk, <1 for mean-reversion, >1 for momentum."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < q + 1:
        return np.nan
    # Cumulative sum for "random walk" interpretation
    diffs_1 = np.diff(x)
    var_1 = np.var(diffs_1, ddof=1) if len(diffs_1) > 1 else 1e-15
    if var_1 < 1e-15:
        return np.nan
    diffs_q = x[q:] - x[:-q]
    var_q = np.var(diffs_q, ddof=1) if len(diffs_q) > 1 else 1e-15
    return float(var_q / (q * var_1))


def direction_autocorrelation(rotations):
    """Autocorrelation of rotation DIRECTION (not just magnitude).

    Uses cosine similarity of consecutive log-rotation vectors (ω_t, ω_{t+1}).
    """
    omegas_vec = []
    for r in rotations:
        # Upper-triangle of skew-symmetric omega (the independent entries)
        K = r["omega"].shape[0]
        idx = np.triu_indices(K, k=1)
        omegas_vec.append(r["omega"][idx])
    omegas_vec = np.array(omegas_vec)

    if len(omegas_vec) < 3:
        return np.nan
    # Vector autocorrelation at lag 1: mean cos(omega_t, omega_{t+1})
    cosines = []
    for i in range(len(omegas_vec) - 1):
        o1, o2 = omegas_vec[i], omegas_vec[i + 1]
        n1, n2 = np.linalg.norm(o1), np.linalg.norm(o2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosines.append(float(np.dot(o1, o2) / (n1 * n2)))
    return float(np.mean(cosines)) if cosines else np.nan


# ══════════════════════════════════════════════════════════════════════
#  Step 4: Subspace prediction models P0–P9
# ══════════════════════════════════════════════════════════════════════

def _rank_k_subspace(P, K):
    """Extract top-K eigenvectors from a symmetric matrix P."""
    P = (P + P.T) / 2
    eigvals, eigvecs = np.linalg.eigh(P)
    idx = np.argsort(eigvals)[::-1][:K]
    U = eigvecs[:, idx]
    return U


def predict_p0(sequence, t):
    """P0: Persistence. Û_{t+1} = U_t."""
    return sequence[t][1]


def predict_p1(sequence, t, rotations):
    """P1: Last-difference extrapolation in projector space.

    P̂_{t+1} = 2P_t − P_{t-1} (linear extrapolation of projector trajectory).
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 1:
        return U_t
    U_prev = sequence[t - 1][1]
    P_t = U_t @ U_t.T
    P_prev = U_prev @ U_prev.T
    P_pred = 2 * P_t - P_prev
    return _rank_k_subspace(P_pred, K)


def predict_p2(sequence, t, rotations):
    """P2: Mean projector velocity.

    P̂_{t+1} = P_t + mean(ΔP_1,...,ΔP_t).
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 1:
        return U_t
    # Compute mean ΔP
    N = U_t.shape[0]
    delta_mean = np.zeros((N, N))
    for i in range(1, t + 1):
        U_i = sequence[i][1]
        U_im1 = sequence[i - 1][1]
        delta_mean += U_i @ U_i.T - U_im1 @ U_im1.T
    delta_mean /= t
    P_pred = U_t @ U_t.T + delta_mean
    return _rank_k_subspace(P_pred, K)


def predict_p3(sequence, t, rotations, hl=4):
    """P3: EWM projector velocity.

    P̂_{t+1} = P_t + ewm-weighted mean of recent ΔP.
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 1:
        return U_t
    N = U_t.shape[0]
    # Compute EWM of ΔP
    n = t
    w = np.exp(-np.arange(n)[::-1] * np.log(2) / hl)
    w /= w.sum()
    delta_ewm = np.zeros((N, N))
    for i in range(1, t + 1):
        U_i = sequence[i][1]
        U_im1 = sequence[i - 1][1]
        dp = U_i @ U_i.T - U_im1 @ U_im1.T
        delta_ewm += w[i - 1] * dp
    P_pred = U_t @ U_t.T + delta_ewm
    return _rank_k_subspace(P_pred, K)


def predict_p4(sequence, t, rotations, ridge_alpha=0.1):
    """P4: Grassmannian tangent-space AR(1).

    Compute tangent vectors via Log map, fit diagonal AR(1) on the
    principal-angle components, predict, apply via Exp map.
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 3:
        return U_t

    # Collect principal angles (K-dimensional time series)
    angle_series = np.array([rotations[i]["angles_rad"] for i in range(t)])

    # Diagonal AR(1) on each principal angle
    phi = np.zeros(K)
    for k in range(K):
        x = angle_series[:, k]
        if len(x) >= 3 and np.var(x[:-1]) > 1e-15:
            phi[k] = np.clip(
                np.sum(x[:-1] * x[1:]) / (np.sum(x[:-1] ** 2) + ridge_alpha),
                -0.99, 0.99
            )

    # Predict next angles
    pred_angles = phi * angle_series[-1]
    pred_total = float(np.linalg.norm(np.clip(pred_angles, 0, np.pi / 2)))
    if pred_total < 1e-10:
        return U_t

    # Use last tangent vector direction, scaled by predicted magnitude
    U_prev = sequence[t - 1][1]
    Delta_last = grassmann_log(U_prev, U_t)
    delta_norm = np.linalg.norm(Delta_last, 'fro')
    if delta_norm < 1e-10:
        return U_t

    # Scale to predicted magnitude and apply at U_t
    Delta_pred = Delta_last * (pred_total / delta_norm)
    # Project to be tangent at U_t (approximate parallel transport)
    Delta_pred = Delta_pred - U_t @ (U_t.T @ Delta_pred)
    return grassmann_exp(U_t, Delta_pred)


def predict_p5(sequence, t, rotations):
    """P5: Per-angle AR(1) with Grassmannian extrapolation.

    θ̂_{k,t+1} = φ_k · θ_{k,t}. Direction from last tangent vector.
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 3:
        return U_t

    angle_series = np.array([rotations[i]["angles_rad"] for i in range(t)])

    phi = np.zeros(K)
    for k in range(K):
        x = angle_series[:, k]
        if len(x) >= 3 and np.var(x[:-1]) > 1e-15:
            phi[k] = np.clip(
                np.sum(x[:-1] * x[1:]) / np.sum(x[:-1] ** 2), -0.99, 0.99
            )

    pred_total = float(np.linalg.norm(np.clip(phi * angle_series[-1], 0, np.pi / 2)))
    if pred_total < 1e-10:
        return U_t

    # Direction from last step
    U_prev = sequence[t - 1][1]
    Delta_last = grassmann_log(U_prev, U_t)
    delta_norm = np.linalg.norm(Delta_last, 'fro')
    if delta_norm < 1e-10:
        return U_t

    Delta_pred = Delta_last * (pred_total / delta_norm)
    Delta_pred = Delta_pred - U_t @ (U_t.T @ Delta_pred)
    return grassmann_exp(U_t, Delta_pred)


def predict_p6(sequence, t, rotations):
    """P6: Constant angular velocity (mean rate from history).

    Apply mean tangent vector magnitude, direction from last step.
    """
    U_t = sequence[t][1]
    K = U_t.shape[1]
    if t < 2:
        return U_t

    # Mean geodesic distance
    mean_d = np.mean([r["d_geodesic_rad"] for r in rotations[:t]])
    if mean_d < 1e-10:
        return U_t

    U_prev = sequence[t - 1][1]
    Delta_last = grassmann_log(U_prev, U_t)
    delta_norm = np.linalg.norm(Delta_last, 'fro')
    if delta_norm < 1e-10:
        return U_t

    Delta_pred = Delta_last * (mean_d / delta_norm)
    Delta_pred = Delta_pred - U_t @ (U_t.T @ Delta_pred)
    return grassmann_exp(U_t, Delta_pred)


def predict_p7(sequence, t, w=4):
    """P7: Euclidean projector average of last w quarters."""
    N = sequence[t][1].shape[0]
    K = sequence[t][1].shape[1]
    start = max(0, t - w + 1)
    Us = [sequence[i][1] for i in range(start, t + 1)]

    # Average projectors
    P_avg = np.zeros((N, N))
    for U in Us:
        P_avg += U @ U.T
    P_avg /= len(Us)

    # Extract rank-K approximation via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(P_avg)
    idx = np.argsort(eigvals)[::-1][:K]
    U_pred = eigvecs[:, idx]
    return U_pred


def predict_p8(sequence, t, w=4):
    """P8: Karcher mean on Gr(K,N) of last w quarters."""
    start = max(0, t - w + 1)
    Us = [sequence[i][1] for i in range(start, t + 1)]
    if len(Us) < 2:
        return Us[0]
    return karcher_mean(Us)


def predict_p9(sequence, t, max_lags=2):
    """P9: HS-linear forecast on projectors.

    P̂_{t+1} = Σ_k c_k P_{t+1-k}, where c_k are scalar weights fit by OLS.
    """
    N = sequence[t][1].shape[0]
    K = sequence[t][1].shape[1]

    if t < max_lags + 2:
        return sequence[t][1]

    # Compute projectors
    Ps = []
    for i in range(t + 1):
        U = sequence[i][1]
        Ps.append(U @ U.T)

    # Fit scalar weights: for each t', P_{t'+1} ≈ Σ c_k P_{t'-k+1}
    # Frobenius loss → scalar least squares
    n_train = t - max_lags
    if n_train < 3:
        return sequence[t][1]

    # Design matrix: each row is [tr(P_{t'} · P_{t'+1}), tr(P_{t'-1} · P_{t'+1}), ...]
    # Response: ||P_{t'+1}||_F^2 (constant = K for projectors)
    # Actually, simpler: minimise Σ_{t'} ||P_{t'+1} - Σ_k c_k P_{t'-k+1}||_F^2
    # = Σ_{t'} [K - 2 Σ_k c_k tr(P_{t'-k+1} P_{t'+1}) + (Σ_k c_k)^2 K]
    # This is a quadratic in c.

    # Gram matrix approach: let G_{jk} = Σ_{t'} tr(P_{t'-j+1} P_{t'-k+1}) / n
    # and b_j = Σ_{t'} tr(P_{t'-j+1} P_{t'+1}) / n
    G = np.zeros((max_lags, max_lags))
    b = np.zeros(max_lags)
    for tp in range(max_lags, t):
        P_target = Ps[tp + 1] if tp + 1 <= t else Ps[t]
        for j in range(max_lags):
            tr_j = np.trace(Ps[tp - j].T @ P_target)
            b[j] += tr_j
            for k in range(max_lags):
                G[j, k] += np.trace(Ps[tp - j].T @ Ps[tp - k])
    n_pts = t - max_lags
    G /= n_pts
    b /= n_pts

    try:
        c = np.linalg.solve(G + 1e-6 * np.eye(max_lags), b)
    except np.linalg.LinAlgError:
        c = np.ones(max_lags) / max_lags

    # Predict
    P_pred = np.zeros((N, N))
    for k in range(max_lags):
        P_pred += c[k] * Ps[t - k]

    # Symmetrise and extract rank-K
    P_pred = (P_pred + P_pred.T) / 2
    eigvals, eigvecs = np.linalg.eigh(P_pred)
    idx = np.argsort(eigvals)[::-1][:K]
    U_pred = eigvecs[:, idx]
    return U_pred


# ══════════════════════════════════════════════════════════════════════
#  Step 5: Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_all_models(sequence, rotations):
    """Evaluate all prediction models on the basis sequence."""
    T = len(sequence)
    model_names = ["P0_persist", "P1_lastR", "P2_meanR", "P3_ewmR",
                   "P4_tangAR", "P5_angleAR", "P6_constV",
                   "P7_euclAvg", "P8_karcher", "P9_hsLinear"]

    results = {m: {"frob": [], "chordal": [], "angle_rmse": [], "subcorr": []}
               for m in model_names}

    # Evaluate: for each t, predict U_{t+1} and compare to actual U_{t+1}
    for t in range(1, T - 1):  # predict t+1 from t; need t >= 1 for rotations
        U_actual = sequence[t + 1][1]
        quarter = sequence[t + 1][0]

        predictors = {
            "P0_persist":  lambda: predict_p0(sequence, t),
            "P1_lastR":    lambda: predict_p1(sequence, t, rotations),
            "P2_meanR":    lambda: predict_p2(sequence, t, rotations),
            "P3_ewmR":     lambda: predict_p3(sequence, t, rotations),
            "P4_tangAR":   lambda: predict_p4(sequence, t, rotations),
            "P5_angleAR":  lambda: predict_p5(sequence, t, rotations),
            "P6_constV":   lambda: predict_p6(sequence, t, rotations),
            "P7_euclAvg":  lambda: predict_p7(sequence, t),
            "P8_karcher":  lambda: predict_p8(sequence, t),
            "P9_hsLinear": lambda: predict_p9(sequence, t),
        }

        for name in model_names:
            try:
                U_pred = predictors[name]()
                if U_pred.shape != U_actual.shape:
                    raise ValueError("shape mismatch")

                frob = projector_frobenius(U_pred, U_actual)
                chord = chordal_distance(U_pred, U_actual)
                angles = principal_angles(U_pred, U_actual)
                rmse = float(np.sqrt(np.mean(angles ** 2)))
                sc = subspace_correlation(U_pred, U_actual)

                results[name]["frob"].append(frob)
                results[name]["chordal"].append(chord)
                results[name]["angle_rmse"].append(rmse)
                results[name]["subcorr"].append(sc)
            except Exception:
                results[name]["frob"].append(np.nan)
                results[name]["chordal"].append(np.nan)
                results[name]["angle_rmse"].append(np.nan)
                results[name]["subcorr"].append(np.nan)

    return results


# ══════════════════════════════════════════════════════════════════════
#  Statistical tests
# ══════════════════════════════════════════════════════════════════════

def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_t_test(d):
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se < 1e-15:
        return np.nan, np.nan
    t_stat = m / se
    p_val = 2 * t_dist.sf(abs(t_stat), df=len(d) - 1)
    return float(t_stat), float(p_val)


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════

def print_rotation_diagnostics(rotations, axis_cosines):
    """Print rotation diagnostics."""
    print("\n" + "=" * 80)
    print("  ROTATION DIAGNOSTICS")
    print("=" * 80)

    d_series = np.array([r["d_geodesic_deg"] for r in rotations])
    print(f"\n  Rotation magnitude (geodesic distance, degrees):")
    print(f"    Mean: {d_series.mean():.1f}°  Std: {d_series.std():.1f}°"
          f"  CV: {d_series.std() / d_series.mean():.3f}")
    print(f"    Min: {d_series.min():.1f}°  Max: {d_series.max():.1f}°")
    print(f"    Median: {np.median(d_series):.1f}°")

    # Per-principal-angle statistics
    all_angles = np.array([r["angles_deg"] for r in rotations])
    K = all_angles.shape[1]
    print(f"\n  Per-principal-angle means (degrees, largest first):")
    for k in range(K):
        print(f"    θ_{k+1}: mean={all_angles[:, k].mean():.1f}°"
              f"  std={all_angles[:, k].std():.1f}°")

    # Axis stability
    print(f"\n  Rotation axis stability (cosine similarity of consecutive ω):")
    print(f"    Mean: {axis_cosines.mean():.3f}  Std: {axis_cosines.std():.3f}")
    if axis_cosines.mean() > 0.8:
        print(f"    → STABLE axis (>0.8): effective 2D dynamics, predict rate only")
    elif axis_cosines.mean() < 0.5:
        print(f"    → UNSTABLE axis (<0.5): full SO(K) dynamics needed")
    else:
        print(f"    → MODERATE stability: partial direction predictability")


def print_temporal_tests(rotations):
    """Print temporal structure test results."""
    print("\n" + "=" * 80)
    print("  TEMPORAL STRUCTURE TESTS")
    print("=" * 80)

    d_series = np.array([r["d_geodesic_deg"] for r in rotations])

    # ACF of geodesic distance
    acf_d = acf(d_series, max_lag=8)
    print(f"\n  ACF of rotation magnitude d_t:")
    for lag in range(1, 9):
        sig = "*" if abs(acf_d[lag]) > 1.96 / np.sqrt(len(d_series)) else " "
        print(f"    lag {lag}: {acf_d[lag]:+.3f} {sig}")

    # Ljung-Box
    Q_stat, p_lb = ljung_box(d_series, max_lag=4)
    print(f"\n  Ljung-Box (H0: d_t is white noise):")
    print(f"    Q(4) = {Q_stat:.2f}, p = {p_lb:.4f} {'→ REJECT (autocorrelated)' if p_lb < 0.05 else '→ FAIL TO REJECT (no evidence of autocorrelation)'}")

    # Variance ratio
    vr = variance_ratio(d_series, q=4)
    print(f"\n  Variance ratio VR(4): {vr:.3f}")
    if vr < 0.8:
        print(f"    → Mean-reverting (VR < 0.8)")
    elif vr > 1.2:
        print(f"    → Persistent/trending (VR > 1.2)")
    else:
        print(f"    → Near random walk (0.8 ≤ VR ≤ 1.2)")

    # Direction autocorrelation
    dir_acorr = direction_autocorrelation(rotations)
    print(f"\n  Rotation direction autocorrelation (lag-1 cosine of ω vectors):")
    print(f"    Mean cosine: {dir_acorr:.3f}")
    if dir_acorr > 0.5:
        print(f"    → STRONG direction persistence (>0.5): P1 should beat P0")
    elif dir_acorr > 0.2:
        print(f"    → MODERATE direction persistence: P1 may beat P0")
    else:
        print(f"    → WEAK direction persistence (<0.2): P1 unlikely to help")

    # Per-angle ACF
    all_angles = np.array([r["angles_deg"] for r in rotations])
    K = all_angles.shape[1]
    print(f"\n  Per-angle ACF(1):")
    for k in range(K):
        acf_k = acf(all_angles[:, k], max_lag=4)
        sig = "*" if abs(acf_k[1]) > 1.96 / np.sqrt(len(d_series)) else " "
        print(f"    θ_{k+1}: ACF(1)={acf_k[1]:+.3f} {sig}")


def print_model_results(eval_results):
    """Print model comparison table."""
    print("\n" + "=" * 80)
    print("  SUBSPACE PREDICTION MODELS — vs PERSISTENCE (P0)")
    print("=" * 80)

    ref_key = "P0_persist"
    ref_frob = np.array(eval_results[ref_key]["frob"])

    model_order = ["P0_persist", "P1_lastR", "P2_meanR", "P3_ewmR",
                   "P4_tangAR", "P5_angleAR", "P6_constV",
                   "P7_euclAvg", "P8_karcher", "P9_hsLinear"]

    labels = {
        "P0_persist": "P0 Persistence",
        "P1_lastR": "P1 Last-R extrap.",
        "P2_meanR": "P2 Mean-R extrap.",
        "P3_ewmR": "P3 EWM-R extrap.",
        "P4_tangAR": "P4 Tangent AR(1)",
        "P5_angleAR": "P5 Angle AR(1)",
        "P6_constV": "P6 Const velocity",
        "P7_euclAvg": "P7 Eucl. proj avg",
        "P8_karcher": "P8 Karcher mean",
        "P9_hsLinear": "P9 HS-linear",
    }

    print(f"\n  {'Model':<22s} {'Frob':>8s} {'Δ vs P0':>9s} {'t':>7s} {'p':>7s}"
          f" {'CI':>23s} {'Chord':>8s} {'SubCorr':>8s}")
    print(f"  {'-'*100}")

    any_beats_p0 = False

    for name in model_order:
        frob = np.array(eval_results[name]["frob"])
        chord = np.array(eval_results[name]["chordal"])
        sc = np.array(eval_results[name]["subcorr"])

        m_frob = np.nanmean(frob)
        m_chord = np.nanmean(chord)
        m_sc = np.nanmean(sc)

        # Delta vs P0 (negative = better)
        delta_frob = frob - ref_frob
        valid = np.isfinite(delta_frob)
        m_delta = np.nanmean(delta_frob)

        if name == ref_key:
            print(f"  {labels[name]:<22s} {m_frob:8.4f} {'—':>9s} {'—':>7s} {'—':>7s}"
                  f" {'—':>23s} {m_chord:8.4f} {m_sc:8.4f}")
            continue

        t_stat, p_val = paired_t_test(delta_frob[valid])
        ci = bootstrap_ci(delta_frob[valid])

        beats = np.isfinite(ci[1]) and ci[1] < 0  # CI entirely below zero = model is better
        if beats:
            any_beats_p0 = True

        ci_s = f"[{ci[0]:+.5f}, {ci[1]:+.5f}]" if np.isfinite(ci[0]) else "N/A"
        t_s = f"{t_stat:.2f}" if np.isfinite(t_stat) else "N/A"
        p_s = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
        beat_mark = " ★" if beats else ""

        print(f"  {labels[name]:<22s} {m_frob:8.4f} {m_delta:+9.5f} {t_s:>7s} {p_s:>7s}"
              f" {ci_s:>23s} {m_chord:8.4f} {m_sc:8.4f}{beat_mark}")

    return any_beats_p0


def evaluate_kill_rule(eval_results):
    """Evaluate Kill Rule A."""
    print("\n" + "=" * 80)
    print("  KILL RULE A EVALUATION")
    print("=" * 80)

    ref_frob = np.array(eval_results["P0_persist"]["frob"])
    ref_chord = np.array(eval_results["P0_persist"]["chordal"])

    any_frob_win = False
    any_chord_win = False

    models = ["P1_lastR", "P2_meanR", "P3_ewmR", "P4_tangAR",
              "P5_angleAR", "P6_constV", "P7_euclAvg", "P8_karcher", "P9_hsLinear"]

    for name in models:
        frob = np.array(eval_results[name]["frob"])
        chord = np.array(eval_results[name]["chordal"])

        d_frob = frob - ref_frob
        d_chord = chord - ref_chord
        valid_f = np.isfinite(d_frob)
        valid_c = np.isfinite(d_chord)

        ci_f = bootstrap_ci(d_frob[valid_f])
        ci_c = bootstrap_ci(d_chord[valid_c])

        f_win = np.isfinite(ci_f[1]) and ci_f[1] < 0
        c_win = np.isfinite(ci_c[1]) and ci_c[1] < 0

        if f_win:
            any_frob_win = True
            print(f"  {name}: Frobenius WIN — CI [{ci_f[0]:+.4f}, {ci_f[1]:+.4f}]")
        if c_win:
            any_chord_win = True
            print(f"  {name}: Chordal WIN — CI [{ci_c[0]:+.4f}, {ci_c[1]:+.4f}]")

    killed = not (any_frob_win or any_chord_win)

    print(f"\n  ═══════════════════════════════════════════════")
    if killed:
        print(f"  KILL RULE A: TRIGGERED")
        print(f"  No model beats persistence on projector Frobenius or chordal distance.")
        print(f"  Rotation is structurally real but temporally UNPREDICTABLE.")
        print(f"  Report diagnostics as structural findings. Stop geometric forecasting.")
    else:
        print(f"  KILL RULE A: NOT TRIGGERED")
        print(f"  At least one geometric predictor beats persistence.")
        print(f"  Proceed to Gate B (actor-level reconstruction).")
    print(f"  ═══════════════════════════════════════════════")

    return killed


# ══════════════════════════════════════════════════════════════════════
#  Save results
# ══════════════════════════════════════════════════════════════════════

def save_results(rotations, eval_results, sequence):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Diagnostics
    rows = []
    for r in rotations:
        row = {"quarter": r["quarter"], "d_deg": r["d_geodesic_deg"],
               "mean_angle_deg": r["mean_angle_deg"]}
        for k, a in enumerate(r["angles_deg"]):
            row[f"theta_{k+1}_deg"] = a
        rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_3_gate_a_diagnostics.parquet", index=False)

    # Model results
    rows = []
    n_quarters = len(eval_results["P0_persist"]["frob"])
    for name, metrics in eval_results.items():
        for i in range(n_quarters):
            rows.append({
                "model": name,
                "quarter_idx": i,
                "frob": metrics["frob"][i],
                "chordal": metrics["chordal"][i],
                "angle_rmse": metrics["angle_rmse"][i],
                "subcorr": metrics["subcorr"][i],
            })
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_3_gate_a_models.parquet", index=False)
    print(f"\n  Saved: iter6_3_gate_a_{{diagnostics,models}}.parquet")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.3 GATE A — IS THE ROTATION PREDICTABLE?")
    print("  Subspace prediction: 10 models vs persistence on ~52-quarter trajectory")
    print("=" * 80)

    panel = load_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # Step 1: Extract basis sequence
    print("\n  Extracting rolling DMD basis sequence...")
    t0 = time.time()
    sequence = extract_basis_sequence(panel, K=K_DEFAULT, t_yr=T_YR, ewm_hl=EWM_HL)
    print(f"  Got {len(sequence)} quarterly bases ({time.time()-t0:.1f}s)")

    # Consistency check: Log/Exp roundtrip
    if len(sequence) >= 2:
        U0, U1 = sequence[0][1], sequence[1][1]
        delta = grassmann_log(U0, U1)
        U1_reconstructed = grassmann_exp(U0, delta)
        roundtrip_err = projector_frobenius(U1, U1_reconstructed)
        print(f"  Log/Exp roundtrip error: {roundtrip_err:.2e}"
              f" {'OK' if roundtrip_err < 1e-6 else 'WARNING'}")

    # Step 2: Rotation diagnostics
    print("\n  Computing rotation diagnostics...")
    rotations = compute_rotation_diagnostics(sequence)
    axis_cosines = compute_axis_stability(rotations)
    print(f"  Got {len(rotations)} transitions, {len(axis_cosines)} axis stability measurements")

    print_rotation_diagnostics(rotations, axis_cosines)
    print_temporal_tests(rotations)

    # Step 4: Subspace prediction models
    print("\n  Evaluating 10 subspace prediction models...")
    t0 = time.time()
    eval_results = evaluate_all_models(sequence, rotations)
    print(f"  Done ({time.time()-t0:.1f}s)")

    any_beats = print_model_results(eval_results)

    # Kill rule
    killed = evaluate_kill_rule(eval_results)

    # Save
    save_results(rotations, eval_results, sequence)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    return rotations, eval_results, killed


if __name__ == "__main__":
    main()
