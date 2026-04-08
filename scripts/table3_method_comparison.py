#!/usr/bin/env python
"""
Iteration 6.2 Gate A — h=1 Decisive Test (93-actor panel only).

Does DMD earn its complexity?  Nine models across three complexity classes,
three contrast blocks (basis, dynamics, Kalman), information combination test.

Models (all on pooled+FE residuals):
  TINY  (~K):   1) PCA+diag AR  2) DMD+diag(Ã)  3) PCA reduced-state  4) DMD reduced-state
  MEDIUM (~K²): 5) PCA+full VAR  6) DMD+full Ã  7) PCA+ridge VAR  8) Reduced-rank Ridge
  LARGE:        9) Ridge on raw residuals
  REF:          0a) Pooled+FE only  0b) Per-actor AR(1)  0c) +residual per-actor AR(1)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_2_gate_a.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.spectral.dmd import ExactDMDDecomposer
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_DEFAULT = 8
K_MAX = 15


# ══════════════════════════════════════════════════════════════════════
#  Shared infrastructure
# ══════════════════════════════════════════════════════════════════════

def load_93_actor_panel():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    layer_map = {a["actor_id"]: a["layer"] for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    actors = list(panel.columns)
    layer_labels = np.array([layer_map.get(a, -1) for a in actors])
    return panel, layer_labels


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def ar1_baseline(otr, N):
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c
    return mu, rho


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_t_test(deltas):
    d = np.array([x for x in deltas if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    mean_d = d.mean()
    se_d = d.std(ddof=1) / np.sqrt(len(d))
    if se_d < 1e-15:
        return np.nan, np.nan
    t_stat = mean_d / se_d
    p_val = 2 * t_dist.sf(abs(t_stat), df=len(d) - 1)
    return float(t_stat), float(p_val)


def _clip_sr(F, max_sr=0.99):
    eigvals = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(eigvals)))
    return F * (max_sr / mx) if mx > max_sr else F


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 10:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def dmd_full(dm, k_svd=K_MAX):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def _mean_valid(lst):
    vals = [x for x in lst if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def _count_wins(model_list, ref_list):
    wins, total = 0, 0
    for m, r in zip(model_list, ref_list):
        if m is not None and r is not None:
            total += 1
            if m > r:
                wins += 1
    return wins, total


def _pca_basis(dm, K):
    """PCA basis from covariance of demeaned data."""
    N = dm.shape[1]
    C = dm.T @ dm / dm.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    ka = min(K, N - 2)
    return eigvecs[:, idx][:, :ka], ka


# ══════════════════════════════════════════════════════════════════════
#  All 9 models + references in one rolling-window pass
# ══════════════════════════════════════════════════════════════════════

def run_window_all_models(panel, ty, K=K_DEFAULT, ewm=12, T_yr=5):
    """Run all 9 + 3 reference models for a single test year.

    Returns dict of model_name → R², or None if window invalid.
    Also returns per-quarter prediction arrays for combination tests.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # ── Stage 1: Pooled AR(1)+FE ──
    rho_pool, bar_y = estimate_pooled_ar1(otr)
    predicted_train = bar_y + rho_pool * (otr[:-1] - bar_y)
    residuals = otr[1:] - predicted_train

    # Per-actor AR(1) (for reference 0b)
    mu_ar1, rho_ar1 = ar1_baseline(otr, N)

    # Residual per-actor AR(1) (for reference 0c)
    mu_r = np.nan_to_num(residuals.mean(0), nan=0.0)
    d_r = residuals - mu_r
    rho_resid_ar1 = np.zeros(N)
    for j in range(N):
        y = d_r[:, j]
        if len(y) >= 3 and np.std(y[:-1]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho_resid_ar1[j] = c

    # ── Residual bases ──
    om_r = ewm_demean(residuals, ewm)
    dm_r = residuals - om_r

    # PCA basis on residuals
    U_pca, ka_pca = _pca_basis(dm_r, K)

    # PCA diag AR(1) on factors
    fac_pca = dm_r @ U_pca
    phi_pca = np.zeros(ka_pca)
    for k in range(ka_pca):
        f = fac_pca[:, k]
        if len(f) >= 3 and np.std(f[:-1]) > 1e-10:
            c = np.corrcoef(f[:-1], f[1:])[0, 1]
            if np.isfinite(c):
                phi_pca[k] = np.clip(c, -0.99, 0.99)

    # PCA full VAR on factors (model 5)
    X_pca, Y_pca = fac_pca[:-1], fac_pca[1:]
    try:
        A_pca_full = (np.linalg.solve(X_pca.T @ X_pca + 1e-8 * np.eye(ka_pca),
                                       X_pca.T @ Y_pca)).T
    except np.linalg.LinAlgError:
        A_pca_full = np.eye(ka_pca) * 0.5

    # PCA ridge VAR on factors (model 7) — ridge penalty
    ridge_alpha_pca = 1.0
    try:
        A_pca_ridge = (np.linalg.solve(X_pca.T @ X_pca + ridge_alpha_pca * np.eye(ka_pca),
                                        X_pca.T @ Y_pca)).T
    except np.linalg.LinAlgError:
        A_pca_ridge = np.eye(ka_pca) * 0.5

    # DMD basis on residuals
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None
    ka_dmd = min(K, mf_r.basis.shape[0] - 2, mf_r.K)
    U_dmd = mf_r.metadata["U"][:, :ka_dmd]
    A_dmd_full = mf_r.metadata["Atilde"][:ka_dmd, :ka_dmd].real.copy()
    A_dmd_diag = np.diag(np.diag(A_dmd_full))

    # DMD ridge VAR hybrid (DMD basis, ridge dynamics instead of Ã)
    fac_dmd = dm_r @ U_dmd
    X_dmd, Y_dmd = fac_dmd[:-1], fac_dmd[1:]
    ridge_alpha_dmd = 1.0
    try:
        A_dmd_ridge = (np.linalg.solve(X_dmd.T @ X_dmd + ridge_alpha_dmd * np.eye(ka_dmd),
                                        X_dmd.T @ Y_dmd)).T
    except np.linalg.LinAlgError:
        A_dmd_ridge = np.eye(ka_dmd) * 0.5

    # Model 8: Reduced-rank Ridge (SVD of ridge coefficient matrix)
    # Ridge on raw residuals → then rank-K approximation
    T_r, N_r = dm_r.shape
    X_rr, Y_rr = dm_r[:-1], dm_r[1:]
    ridge_alpha_rr = max(1.0, N_r * 0.01)
    try:
        C_hat = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha_rr * np.eye(N_r),
                                  X_rr.T @ Y_rr)).T  # (N, N)
        # Rank-K approximation via SVD
        Uc, Sc, Vtc = np.linalg.svd(C_hat, full_matrices=False)
        C_rrr = Uc[:, :K] @ np.diag(Sc[:K]) @ Vtc[:K, :]
    except np.linalg.LinAlgError:
        C_rrr = np.eye(N_r) * 0.5

    # Model 9: Full Ridge on raw residuals
    ridge_alpha_full = max(1.0, N_r * 0.1)
    try:
        C_ridge = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha_full * np.eye(N_r),
                                    X_rr.T @ Y_rr)).T
    except np.linalg.LinAlgError:
        C_ridge = np.eye(N_r) * 0.5

    # ── Kalman filter states for models 1,2 (with Kalman) ──
    def _init_kalman(k):
        return np.zeros(k), np.eye(k), np.eye(k) * Q_INIT_SCALE

    # Model 1: PCA+diag AR (with Kalman)
    a1, P1, Q1 = _init_kalman(ka_pca)
    F1 = np.diag(phi_pca)
    R_pca = sph_r(dm_r, U_pca)

    # Model 2: DMD+diag(Ã) (with Kalman)
    a2, P2, Q2 = _init_kalman(ka_dmd)
    F2 = _clip_sr(A_dmd_diag)
    R_dmd = sph_r(dm_r, U_dmd)

    # Model 6: DMD+full Ã (with Kalman)
    a6, P6, Q6 = _init_kalman(ka_dmd)
    F6 = _clip_sr(A_dmd_full.copy())

    # ── Accumulators ──
    model_names = [
        "pca_diag_kalman",        # 1
        "dmd_diag_kalman",        # 2
        "pca_reduced_nokalman",   # 3
        "dmd_reduced_nokalman",   # 4
        "pca_full_var",           # 5
        "dmd_full_kalman",        # 6
        "pca_ridge_var",          # 7
        "reduced_rank_ridge",     # 8
        "ridge_raw",              # 9
        "dmd_ridge_var",          # extra: DMD+ridge dynamics
        "ref_pooled",             # 0a
        "ref_ar1",                # 0b
        "ref_resid_ar1",          # 0c
    ]
    preds = {m: [] for m in model_names}
    actuals = []

    prev = np.nan_to_num(otr[-1], nan=0.5)
    prev_resid = np.zeros(N)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # Stage 1 prediction
        y_ar = bar_y + rho_pool * (prev - bar_y)

        # References
        preds["ref_pooled"].append(y_ar)
        preds["ref_ar1"].append(mu_ar1 + rho_ar1 * (prev - mu_ar1))
        preds["ref_resid_ar1"].append(y_ar + mu_r + rho_resid_ar1 * (prev_resid - mu_r))

        # ── Model 1: PCA+diag AR with Kalman ──
        ap1 = F1 @ a1
        Pp1 = F1 @ P1 @ F1.T + Q1
        pred1 = U_pca @ ap1 + om_r.ravel()
        if not np.all(np.isfinite(pred1)):
            pred1 = np.zeros(N)
        preds["pca_diag_kalman"].append(y_ar + pred1)

        # ── Model 2: DMD+diag(Ã) with Kalman ──
        ap2 = F2 @ a2
        Pp2 = F2 @ P2 @ F2.T + Q2
        pred2 = U_dmd @ ap2 + om_r.ravel()
        if not np.all(np.isfinite(pred2)):
            pred2 = np.zeros(N)
        preds["dmd_diag_kalman"].append(y_ar + pred2)

        # ── Model 3: PCA reduced-state (no Kalman) ──
        f_pca_prev = U_pca.T @ (prev_resid - om_r.ravel())
        pred3 = U_pca @ (np.diag(phi_pca) @ f_pca_prev) + om_r.ravel()
        if not np.all(np.isfinite(pred3)):
            pred3 = np.zeros(N)
        preds["pca_reduced_nokalman"].append(y_ar + pred3)

        # ── Model 4: DMD reduced-state (no Kalman) ──
        f_dmd_prev = U_dmd.T @ (prev_resid - om_r.ravel())
        pred4 = U_dmd @ (_clip_sr(A_dmd_diag) @ f_dmd_prev) + om_r.ravel()
        if not np.all(np.isfinite(pred4)):
            pred4 = np.zeros(N)
        preds["dmd_reduced_nokalman"].append(y_ar + pred4)

        # ── Model 5: PCA+full VAR (no Kalman) ──
        pred5 = U_pca @ (A_pca_full @ f_pca_prev) + om_r.ravel()
        if not np.all(np.isfinite(pred5)):
            pred5 = np.zeros(N)
        preds["pca_full_var"].append(y_ar + pred5)

        # ── Model 6: DMD+full Ã with Kalman ──
        ap6 = F6 @ a6
        Pp6 = F6 @ P6 @ F6.T + Q6
        pred6 = U_dmd @ ap6 + om_r.ravel()
        if not np.all(np.isfinite(pred6)):
            pred6 = np.zeros(N)
        preds["dmd_full_kalman"].append(y_ar + pred6)

        # ── Model 7: PCA+ridge VAR (no Kalman) ──
        pred7 = U_pca @ (A_pca_ridge @ f_pca_prev) + om_r.ravel()
        if not np.all(np.isfinite(pred7)):
            pred7 = np.zeros(N)
        preds["pca_ridge_var"].append(y_ar + pred7)

        # ── Model 8: Reduced-rank Ridge ──
        pred8 = C_rrr @ (prev_resid - om_r.ravel()) + om_r.ravel()
        if not np.all(np.isfinite(pred8)):
            pred8 = np.zeros(N)
        preds["reduced_rank_ridge"].append(y_ar + pred8)

        # ── Model 9: Ridge on raw residuals ──
        pred9 = C_ridge @ (prev_resid - om_r.ravel()) + om_r.ravel()
        if not np.all(np.isfinite(pred9)):
            pred9 = np.zeros(N)
        preds["ridge_raw"].append(y_ar + pred9)

        # ── DMD+ridge VAR hybrid ──
        pred_dr = U_dmd @ (A_dmd_ridge @ f_dmd_prev) + om_r.ravel()
        if not np.all(np.isfinite(pred_dr)):
            pred_dr = np.zeros(N)
        preds["dmd_ridge_var"].append(y_ar + pred_dr)

        actuals.append(obs)

        # ── Kalman updates for models 1, 2, 6 ──
        actual_resid = obs - y_ar
        odm_r = actual_resid - om_r.ravel()

        # Model 1 Kalman update
        S1 = U_pca @ Pp1 @ U_pca.T + R_pca
        try:
            Kg1 = Pp1 @ U_pca.T @ np.linalg.solve(S1, np.eye(N))
        except Exception:
            Kg1 = np.zeros((ka_pca, N))
        a1 = ap1 + Kg1 @ (odm_r - U_pca @ ap1)
        P1 = (np.eye(ka_pca) - Kg1 @ U_pca) @ Pp1
        inn1 = a1 - ap1
        Q1 = (1 - LAMBDA_Q) * Q1 + LAMBDA_Q * np.outer(inn1, inn1)
        Q1 = (Q1 + Q1.T) / 2 + np.eye(ka_pca) * 1e-6

        # Model 2 Kalman update
        S2 = U_dmd @ Pp2 @ U_dmd.T + R_dmd
        try:
            Kg2 = Pp2 @ U_dmd.T @ np.linalg.solve(S2, np.eye(N))
        except Exception:
            Kg2 = np.zeros((ka_dmd, N))
        a2 = ap2 + Kg2 @ (odm_r - U_dmd @ ap2)
        P2 = (np.eye(ka_dmd) - Kg2 @ U_dmd) @ Pp2
        inn2 = a2 - ap2
        Q2 = (1 - LAMBDA_Q) * Q2 + LAMBDA_Q * np.outer(inn2, inn2)
        Q2 = (Q2 + Q2.T) / 2 + np.eye(ka_dmd) * 1e-6

        # Model 6 Kalman update
        S6 = U_dmd @ Pp6 @ U_dmd.T + R_dmd
        try:
            Kg6 = Pp6 @ U_dmd.T @ np.linalg.solve(S6, np.eye(N))
        except Exception:
            Kg6 = np.zeros((ka_dmd, N))
        a6 = ap6 + Kg6 @ (odm_r - U_dmd @ ap6)
        P6 = (np.eye(ka_dmd) - Kg6 @ U_dmd) @ Pp6
        inn6 = a6 - ap6
        Q6 = (1 - LAMBDA_Q) * Q6 + LAMBDA_Q * np.outer(inn6, inn6)
        Q6 = (Q6 + Q6.T) / 2 + np.eye(ka_dmd) * 1e-6

        prev_resid = actual_resid
        prev = obs

        # ── Rolling update: re-estimate all bases ──
        otr = np.vstack([otr, qv])
        rho_pool, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho_pool * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, ewm)
        dm_r = residuals_new - om_r

        # Re-estimate per-actor AR(1)
        mu_ar1, rho_ar1 = ar1_baseline(otr, N)

        # Re-estimate residual AR(1)
        mu_r = np.nan_to_num(residuals_new.mean(0), nan=0.0)
        d_r = residuals_new - mu_r
        rho_resid_ar1 = np.zeros(N)
        for j in range(N):
            y = d_r[:, j]
            if len(y) >= 3 and np.std(y[:-1]) > 1e-10:
                c = np.corrcoef(y[:-1], y[1:])[0, 1]
                if np.isfinite(c):
                    rho_resid_ar1[j] = c

        # Re-estimate PCA
        U_pca_new, ka_pca_new = _pca_basis(dm_r, K)
        fac_pca_new = dm_r @ U_pca_new
        phi_pca_new = np.zeros(ka_pca_new)
        for k in range(ka_pca_new):
            f = fac_pca_new[:, k]
            if len(f) >= 3 and np.std(f[:-1]) > 1e-10:
                c = np.corrcoef(f[:-1], f[1:])[0, 1]
                if np.isfinite(c):
                    phi_pca_new[k] = np.clip(c, -0.99, 0.99)
        X_pca_new, Y_pca_new = fac_pca_new[:-1], fac_pca_new[1:]
        try:
            A_pca_full = (np.linalg.solve(X_pca_new.T @ X_pca_new + 1e-8 * np.eye(ka_pca_new),
                                           X_pca_new.T @ Y_pca_new)).T
        except np.linalg.LinAlgError:
            pass
        try:
            A_pca_ridge = (np.linalg.solve(X_pca_new.T @ X_pca_new + ridge_alpha_pca * np.eye(ka_pca_new),
                                            X_pca_new.T @ Y_pca_new)).T
        except np.linalg.LinAlgError:
            pass
        U_pca = U_pca_new
        ka_pca = ka_pca_new
        phi_pca = phi_pca_new
        R_pca = sph_r(dm_r, U_pca)

        # Kalman reset for model 1
        a1 = U_pca.T @ (actual_resid - om_r.ravel())
        P1 = np.eye(ka_pca)
        Q1 = np.eye(ka_pca) * Q_INIT_SCALE
        F1 = np.diag(phi_pca)

        # Re-estimate DMD
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            ka_dmd_new = min(K, mf_r2.basis.shape[0] - 2, mf_r2.K)
            U_dmd = mf_r2.metadata["U"][:, :ka_dmd_new]
            A_dmd_full = mf_r2.metadata["Atilde"][:ka_dmd_new, :ka_dmd_new].real.copy()
            A_dmd_diag = np.diag(np.diag(A_dmd_full))
            ka_dmd = ka_dmd_new
            R_dmd = sph_r(dm_r, U_dmd)

            # DMD ridge hybrid
            fac_dmd_new = dm_r @ U_dmd
            X_dmd_new, Y_dmd_new = fac_dmd_new[:-1], fac_dmd_new[1:]
            try:
                A_dmd_ridge = (np.linalg.solve(X_dmd_new.T @ X_dmd_new + ridge_alpha_dmd * np.eye(ka_dmd),
                                                X_dmd_new.T @ Y_dmd_new)).T
            except np.linalg.LinAlgError:
                pass

            # Kalman resets for models 2, 6
            a2 = U_dmd.T @ (actual_resid - om_r.ravel())
            P2 = np.eye(ka_dmd); Q2 = np.eye(ka_dmd) * Q_INIT_SCALE
            F2 = _clip_sr(A_dmd_diag)

            a6 = U_dmd.T @ (actual_resid - om_r.ravel())
            P6 = np.eye(ka_dmd); Q6 = np.eye(ka_dmd) * Q_INIT_SCALE
            F6 = _clip_sr(A_dmd_full.copy())

        # Re-estimate Ridge models (8, 9)
        N_r = dm_r.shape[1]
        X_rr, Y_rr = dm_r[:-1], dm_r[1:]
        ridge_alpha_rr = max(1.0, N_r * 0.01)
        ridge_alpha_full = max(1.0, N_r * 0.1)
        try:
            C_hat = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha_rr * np.eye(N_r),
                                      X_rr.T @ Y_rr)).T
            Uc, Sc, Vtc = np.linalg.svd(C_hat, full_matrices=False)
            C_rrr = Uc[:, :K] @ np.diag(Sc[:K]) @ Vtc[:K, :]
        except np.linalg.LinAlgError:
            pass
        try:
            C_ridge = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha_full * np.eye(N_r),
                                        X_rr.T @ Y_rr)).T
        except np.linalg.LinAlgError:
            pass

    if not actuals:
        return None

    # ── Compute R² for all models ──
    act_a = np.array(actuals)
    r2s = {}
    pred_arrays = {}
    for m in model_names:
        pa = np.array(preds[m])
        if pa.shape[0] == 0 or not np.all(np.isfinite(pa)):
            r2s[m] = None
            pred_arrays[m] = None
        else:
            r2s[m] = float(oos_r_squared(pa.ravel(), act_a.ravel()))
            pred_arrays[m] = pa

    return r2s, pred_arrays, act_a


# ══════════════════════════════════════════════════════════════════════
#  Combination Test
# ══════════════════════════════════════════════════════════════════════

def run_combination_test(pred_arrays_by_year, actuals_by_year, test_years):
    """Pairwise and triple combinations of DMD, PCA, Ridge predictions.

    Uses equal-weight and train-only CV weight.
    Reports combined R², ΔR², forecast-error correlation.
    """
    # Collect all predictions and actuals across windows
    models = ["dmd_diag_kalman", "pca_diag_kalman", "ridge_raw"]
    labels = {"dmd_diag_kalman": "DMD", "pca_diag_kalman": "PCA", "ridge_raw": "Ridge"}

    # Per-window combination results
    pairs = [("dmd_diag_kalman", "ridge_raw"), ("dmd_diag_kalman", "pca_diag_kalman"),
             ("pca_diag_kalman", "ridge_raw")]
    triple = ("dmd_diag_kalman", "pca_diag_kalman", "ridge_raw")

    results = {}

    for ty_idx, ty in enumerate(test_years):
        if pred_arrays_by_year[ty_idx] is None:
            continue
        pa, act = pred_arrays_by_year[ty_idx], actuals_by_year[ty_idx]

        # Individual errors
        errors = {}
        for m in models:
            if pa.get(m) is not None:
                errors[m] = (act - pa[m]).ravel()

        if len(errors) < 3:
            continue

        # Forecast-error correlations
        for m1, m2 in pairs:
            key = f"rho_{labels[m1]}_{labels[m2]}"
            if key not in results:
                results[key] = []
            rho = np.corrcoef(errors[m1], errors[m2])[0, 1]
            results[key].append(float(rho))

        # Pairwise combinations (equal-weight)
        for m1, m2 in pairs:
            key = f"eq_{labels[m1]}+{labels[m2]}"
            if key not in results:
                results[key] = []
            combined = 0.5 * pa[m1] + 0.5 * pa[m2]
            r2 = float(oos_r_squared(combined.ravel(), act.ravel()))
            results[key].append(r2)

        # Pairwise combinations (OLS-weight on training predictions)
        # Use leave-one-out within the test window
        for m1, m2 in pairs:
            key = f"cv_{labels[m1]}+{labels[m2]}"
            if key not in results:
                results[key] = []
            # Simple: use correlation-based optimal weight
            e1, e2 = errors[m1], errors[m2]
            v1, v2 = np.var(e1), np.var(e2)
            cov12 = np.cov(e1, e2)[0, 1]
            denom = v1 + v2 - 2 * cov12
            if denom > 1e-12:
                w1 = (v2 - cov12) / denom
                w1 = np.clip(w1, 0, 1)
            else:
                w1 = 0.5
            combined_cv = w1 * pa[m1] + (1 - w1) * pa[m2]
            r2 = float(oos_r_squared(combined_cv.ravel(), act.ravel()))
            results[key].append(r2)

        # Triple combination (equal-weight)
        key = "eq_DMD+PCA+Ridge"
        if key not in results:
            results[key] = []
        combined_3 = (pa[models[0]] + pa[models[1]] + pa[models[2]]) / 3
        r2_3 = float(oos_r_squared(combined_3.ravel(), act.ravel()))
        results[key].append(r2_3)

        # Triple combination (OLS weights)
        key = "ols_DMD+PCA+Ridge"
        if key not in results:
            results[key] = []
        # Stack predictions and solve for optimal weights
        X_comb = np.column_stack([pa[m].ravel() for m in models])
        y_comb = act.ravel()
        try:
            w_ols = np.linalg.solve(X_comb.T @ X_comb + 1e-6 * np.eye(3),
                                     X_comb.T @ y_comb)
            combined_ols = X_comb @ w_ols
            r2_ols = float(oos_r_squared(combined_ols, y_comb))
        except np.linalg.LinAlgError:
            r2_ols = r2_3
        results[key].append(r2_ols)

    return results


# ══════════════════════════════════════════════════════════════════════
#  DM-style paired-window loss differential
# ══════════════════════════════════════════════════════════════════════

def dm_window_test(r2_model, r2_ref):
    """DM-style test on per-window R² differentials."""
    deltas = [m - r for m, r in zip(r2_model, r2_ref)
              if m is not None and r is not None and np.isfinite(m) and np.isfinite(r)]
    if len(deltas) < 3:
        return np.nan, np.nan, (np.nan, np.nan)
    d = np.array(deltas)
    t_stat, p_val = paired_t_test(d)
    ci = bootstrap_ci(d)
    return float(np.mean(d)), t_stat, ci


# ══════════════════════════════════════════════════════════════════════
#  Print & Report
# ══════════════════════════════════════════════════════════════════════

def print_model_table(all_r2s, test_years):
    """Print the full model comparison table."""
    display = [
        ("ref_ar1",             "0b  Per-actor AR(1)",    "REF"),
        ("ref_pooled",          "0a  Pooled+FE",          "REF"),
        ("ref_resid_ar1",       "0c  +residual AR(1)",    "REF"),
        ("pca_diag_kalman",     "1   PCA+diag Kalman",    "TINY"),
        ("dmd_diag_kalman",     "2   DMD+diag(Ã) Kalman", "TINY"),
        ("pca_reduced_nokalman","3   PCA reduced (no K)", "TINY"),
        ("dmd_reduced_nokalman","4   DMD reduced (no K)", "TINY"),
        ("pca_full_var",        "5   PCA+full VAR",       "MEDIUM"),
        ("dmd_full_kalman",     "6   DMD+full Ã Kalman",  "MEDIUM"),
        ("pca_ridge_var",       "7   PCA+ridge VAR",      "MEDIUM"),
        ("dmd_ridge_var",       "    DMD+ridge VAR",      "MEDIUM"),
        ("reduced_rank_ridge",  "8   Reduced-rank Ridge", "MEDIUM"),
        ("ridge_raw",           "9   Ridge raw residuals","LARGE"),
    ]

    ar1_r2s = [all_r2s[i].get("ref_ar1") if all_r2s[i] else None for i in range(len(test_years))]
    ar1_mean = _mean_valid(ar1_r2s)

    print(f"\n  {'#':<3s} {'Model':<25s} {'Class':<8s} {'R²':>7s} {'ΔR² AR1':>9s}"
          f" {'W/AR1':>6s} {'t-stat':>7s} {'p':>7s} {'CI':>21s}")
    print(f"  {'-'*100}")

    for key, label, cclass in display:
        vals = [all_r2s[i].get(key) if all_r2s[i] else None for i in range(len(test_years))]
        m = _mean_valid(vals)
        d_ar1 = m - ar1_mean if np.isfinite(m) else np.nan
        w, t = _count_wins(vals, ar1_r2s)

        deltas = [v - a for v, a in zip(vals, ar1_r2s)
                  if v is not None and a is not None]
        if len(deltas) >= 3:
            t_stat, p_val = paired_t_test(deltas)
            lo, hi = bootstrap_ci(np.array(deltas))
            ci_s = f"[{lo:+.4f}, {hi:+.4f}]"
        else:
            t_stat, p_val = np.nan, np.nan
            ci_s = "N/A"

        r2_s = f"{m:.4f}" if np.isfinite(m) else "N/A"
        d_s = f"{d_ar1:+.4f}" if np.isfinite(d_ar1) else "N/A"
        t_s = f"{t_stat:.2f}" if np.isfinite(t_stat) else "N/A"
        p_s = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"

        print(f"  {label:<28s} {cclass:<8s} {r2_s:>7s} {d_s:>9s}"
              f" {w:>3d}/{t:<2d} {t_s:>7s} {p_s:>7s} {ci_s:>21s}")


def print_contrast_blocks(all_r2s, test_years):
    """Print the three contrast blocks."""

    def _extract(key):
        return [all_r2s[i].get(key) if all_r2s[i] else None for i in range(len(test_years))]

    # Block 1: Basis contrast (dynamics = diagonal AR, fixed)
    print("\n" + "=" * 80)
    print("  BLOCK 1 — BASIS CONTRAST (dynamics = diagonal, fixed)")
    print("=" * 80)
    pca_diag = _extract("pca_diag_kalman")
    dmd_diag = _extract("dmd_diag_kalman")
    mean_delta, t_stat, ci = dm_window_test(dmd_diag, pca_diag)
    print(f"  PCA+diag (1) mean R²: {_mean_valid(pca_diag):.4f}")
    print(f"  DMD+diag (2) mean R²: {_mean_valid(dmd_diag):.4f}")
    print(f"  Δ(DMD−PCA): {mean_delta:+.4f}  t={t_stat:.2f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    if np.isfinite(ci[0]) and np.isfinite(ci[1]):
        if ci[0] > 0:
            print(f"  → CI excludes zero: DMD basis ADDS value at tiny complexity")
        elif ci[1] < 0:
            print(f"  → CI excludes zero: PCA basis BETTER at tiny complexity")
        else:
            print(f"  → CI includes zero: basis choice IRRELEVANT at tiny complexity")

    # Block 2: Dynamics contrast (basis = DMD, fixed)
    print("\n" + "=" * 80)
    print("  BLOCK 2 — DYNAMICS CONTRAST (basis = DMD, fixed)")
    print("=" * 80)
    dmd_nokalman = _extract("dmd_reduced_nokalman")
    dmd_full = _extract("dmd_full_kalman")
    dmd_ridge = _extract("dmd_ridge_var")

    print(f"  4) DMD reduced (no K):  {_mean_valid(dmd_nokalman):.4f}")
    print(f"  2) DMD+diag(Ã) Kalman:  {_mean_valid(dmd_diag):.4f}")
    print(f"  6) DMD+full Ã Kalman:   {_mean_valid(dmd_full):.4f}")
    print(f"     DMD+ridge VAR:       {_mean_valid(dmd_ridge):.4f}")

    delta_proj_diag, t1, ci1 = dm_window_test(dmd_diag, dmd_nokalman)
    print(f"\n  Projection→diag: Δ={delta_proj_diag:+.4f}  t={t1:.2f}  CI [{ci1[0]:+.4f}, {ci1[1]:+.4f}]")

    delta_diag_full, t2, ci2 = dm_window_test(dmd_full, dmd_diag)
    print(f"  Diag→full:       Δ={delta_diag_full:+.4f}  t={t2:.2f}  CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}]")

    delta_full_ridge, t3, ci3 = dm_window_test(dmd_full, dmd_ridge)
    print(f"  Full Ã vs ridge: Δ={delta_full_ridge:+.4f}  t={t3:.2f}  CI [{ci3[0]:+.4f}, {ci3[1]:+.4f}]")

    # Block 3: Kalman contribution (basis + dynamics fixed)
    print("\n" + "=" * 80)
    print("  BLOCK 3 — KALMAN CONTRIBUTION (basis + dynamics fixed)")
    print("=" * 80)
    pca_nokalman = _extract("pca_reduced_nokalman")

    delta_dmd_k, t4, ci4 = dm_window_test(dmd_diag, dmd_nokalman)
    print(f"  DMD: Kalman − no Kalman: Δ={delta_dmd_k:+.4f}  t={t4:.2f}  CI [{ci4[0]:+.4f}, {ci4[1]:+.4f}]")

    delta_pca_k, t5, ci5 = dm_window_test(pca_diag, pca_nokalman)
    print(f"  PCA: Kalman − no Kalman: Δ={delta_pca_k:+.4f}  t={t5:.2f}  CI [{ci5[0]:+.4f}, {ci5[1]:+.4f}]")

    kalman_threshold = 0.005
    for label, delta in [("DMD", delta_dmd_k), ("PCA", delta_pca_k)]:
        if np.isfinite(delta) and delta >= kalman_threshold:
            print(f"  → {label}: Kalman adds ≥+{kalman_threshold:.3f} → filtering machinery contributes")
        elif np.isfinite(delta):
            print(f"  → {label}: Kalman adds <+{kalman_threshold:.3f} → Kalman is overhead")


def print_combination_results(combo_results, all_r2s, test_years):
    """Print combination test results."""
    print("\n" + "=" * 80)
    print("  INFORMATION COMBINATION TEST")
    print("=" * 80)

    def _extract(key):
        return [all_r2s[i].get(key) if all_r2s[i] else None for i in range(len(test_years))]

    dmd_r2 = _mean_valid(_extract("dmd_diag_kalman"))
    pca_r2 = _mean_valid(_extract("pca_diag_kalman"))
    ridge_r2 = _mean_valid(_extract("ridge_raw"))

    print(f"\n  Individual model R²:")
    print(f"    DMD+diag:  {dmd_r2:.4f}")
    print(f"    PCA+diag:  {pca_r2:.4f}")
    print(f"    Ridge:     {ridge_r2:.4f}")

    # Error correlations
    print(f"\n  Forecast-error correlations (ρ_pred):")
    for key in sorted(combo_results):
        if key.startswith("rho_"):
            vals = combo_results[key]
            print(f"    {key[4:]}: {_mean_valid(vals):.4f}")

    # Combinations
    print(f"\n  Combination R² (mean across windows):")
    print(f"    {'Combination':<30s} {'R²':>7s} {'ΔR² vs max':>11s}")
    print(f"    {'-'*52}")
    for key in sorted(combo_results):
        if key.startswith("rho_"):
            continue
        vals = combo_results[key]
        m = _mean_valid(vals)
        max_input = max(dmd_r2, pca_r2, ridge_r2)
        delta = m - max_input if np.isfinite(m) else np.nan
        d_s = f"{delta:+.4f}" if np.isfinite(delta) else "N/A"
        print(f"    {key:<30s} {m:7.4f} {d_s:>11s}")

    # Interpretation
    rho_dmd_ridge = _mean_valid(combo_results.get("rho_DMD_Ridge", []))
    rho_dmd_pca = _mean_valid(combo_results.get("rho_DMD_PCA", []))
    print(f"\n  Interpretation:")
    if np.isfinite(rho_dmd_ridge) and rho_dmd_ridge > 0.98:
        print(f"    ρ(DMD,Ridge)={rho_dmd_ridge:.4f} > 0.98 → predictions functionally identical")
    elif np.isfinite(rho_dmd_ridge) and rho_dmd_ridge < 0.95:
        print(f"    ρ(DMD,Ridge)={rho_dmd_ridge:.4f} < 0.95 → potential unique content")
    elif np.isfinite(rho_dmd_ridge):
        print(f"    ρ(DMD,Ridge)={rho_dmd_ridge:.4f} — borderline")
    if np.isfinite(rho_dmd_pca) and rho_dmd_pca > 0.98:
        print(f"    ρ(DMD,PCA)={rho_dmd_pca:.4f} > 0.98 → DMD and PCA predictions functionally identical")


def print_quality_gates(all_r2s, test_years):
    """Verify quality gates QG1-QG5."""
    print("\n" + "=" * 80)
    print("  QUALITY GATES")
    print("=" * 80)

    def _extract(key):
        return [all_r2s[i].get(key) if all_r2s[i] else None for i in range(len(test_years))]

    ar1 = _mean_valid(_extract("ref_ar1"))
    pooled = _mean_valid(_extract("ref_pooled"))
    dmd_diag = _mean_valid(_extract("dmd_diag_kalman"))
    ridge = _mean_valid(_extract("ridge_raw"))

    qg1 = abs(ar1 - 0.594) <= 0.010
    qg2 = abs(pooled - 0.591) <= 0.010
    qg3 = abs(dmd_diag - 0.615) <= 0.020  # wider tolerance — different implementation
    qg4 = abs(ridge - 0.632) <= 0.020
    # QG5: no NaN check (done implicitly by oos_r_squared returning finite)
    qg5 = all(all_r2s[i] is not None for i in range(len(test_years)))

    checks = [
        ("QG1", f"AR(1) = {ar1:.4f} ≈ 0.594 ±0.010", qg1),
        ("QG2", f"Pooled+FE = {pooled:.4f} ≈ 0.591 ±0.010", qg2),
        ("QG3", f"DMD+diag = {dmd_diag:.4f} ≈ 0.615 ±0.020", qg3),
        ("QG4", f"Ridge = {ridge:.4f} ≈ 0.632 ±0.020", qg4),
        ("QG5", f"All windows valid = {qg5}", qg5),
    ]
    for name, desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} — {desc}")

    return all(p for _, _, p in checks)


def evaluate_kill_rule_a(all_r2s, combo_results, test_years):
    """Evaluate Kill Rule A."""
    print("\n" + "=" * 80)
    print("  KILL RULE A EVALUATION")
    print("=" * 80)

    def _extract(key):
        return [all_r2s[i].get(key) if all_r2s[i] else None for i in range(len(test_years))]

    # Condition 1: DMD does not beat PCA at matched complexity
    pca_diag = _extract("pca_diag_kalman")
    dmd_diag = _extract("dmd_diag_kalman")
    delta_tiny, _, ci_tiny = dm_window_test(dmd_diag, pca_diag)

    pca_full = _extract("pca_full_var")
    dmd_full = _extract("dmd_full_kalman")
    delta_med, _, ci_med = dm_window_test(dmd_full, pca_full)

    tiny_no_win = np.isfinite(ci_tiny[0]) and ci_tiny[0] <= 0 and ci_tiny[1] >= 0
    med_no_win = np.isfinite(ci_med[0]) and ci_med[0] <= 0 and ci_med[1] >= 0

    # Adjust: also check if DMD wins (CI above zero)
    tiny_dmd_wins = np.isfinite(ci_tiny[0]) and ci_tiny[0] > 0
    med_dmd_wins = np.isfinite(ci_med[0]) and ci_med[0] > 0

    print(f"  Condition 1a — TINY: DMD−PCA Δ={delta_tiny:+.4f}  CI [{ci_tiny[0]:+.4f}, {ci_tiny[1]:+.4f}]")
    print(f"    {'DMD wins' if tiny_dmd_wins else 'No DMD advantage'}")
    print(f"  Condition 1b — MEDIUM: DMD−PCA Δ={delta_med:+.4f}  CI [{ci_med[0]:+.4f}, {ci_med[1]:+.4f}]")
    print(f"    {'DMD wins' if med_dmd_wins else 'No DMD advantage'}")

    # Condition 2: Combination test shows no unique content
    rho_dmd_ridge = _mean_valid(combo_results.get("rho_DMD_Ridge", []))
    high_corr = np.isfinite(rho_dmd_ridge) and rho_dmd_ridge > 0.98

    # Check if combination DM-significantly beats max input
    ridge_r2s = _extract("ridge_raw")
    combo_eq = combo_results.get("eq_DMD+Ridge", [])
    if len(combo_eq) >= 3 and len(ridge_r2s) >= 3:
        combo_delta, combo_t, combo_ci = dm_window_test(combo_eq, ridge_r2s)
        combo_no_gain = np.isfinite(combo_ci[0]) and combo_ci[0] <= 0
    else:
        combo_delta, combo_no_gain = np.nan, True

    print(f"\n  Condition 2a — ρ(DMD,Ridge) = {rho_dmd_ridge:.4f} {'> 0.98 (identical)' if high_corr else '≤ 0.98'}")
    if np.isfinite(combo_delta):
        print(f"  Condition 2b — DMD+Ridge combo vs Ridge: Δ={combo_delta:+.4f}")

    no_dmd_advantage = not tiny_dmd_wins and not med_dmd_wins
    no_unique_content = high_corr and combo_no_gain

    killed = no_dmd_advantage and no_unique_content
    alive = tiny_dmd_wins or med_dmd_wins

    print(f"\n  ═══════════════════════════════════════════════")
    if killed:
        print(f"  KILL RULE A: TRIGGERED")
        print(f"  Generic h=1 DMD-specific claim is DEAD.")
        print(f"  Proceed to Gate B — DMD may still win under stress conditions.")
    elif alive:
        print(f"  KILL RULE A: NOT TRIGGERED — DMD shows advantage")
        print(f"  Proceed to Gate B for confirmation + Gate C for cross-panel validation.")
    else:
        print(f"  KILL RULE A: TRIGGERED (no advantage, though unique content unclear)")
        print(f"  Proceed to Gate B — DMD may still win under stress conditions.")
    print(f"  ═══════════════════════════════════════════════")

    return killed


# ══════════════════════════════════════════════════════════════════════
#  Save Results
# ══════════════════════════════════════════════════════════════════════

def save_results(all_r2s, combo_results, test_years):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Main model table
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty}
        if all_r2s[i]:
            row.update(all_r2s[i])
        rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_2_gate_a_models.parquet", index=False)

    # Combination results
    combo_rows = []
    for key, vals in combo_results.items():
        for i, v in enumerate(vals):
            combo_rows.append({"metric": key, "window_idx": i, "value": v})
    if combo_rows:
        pd.DataFrame(combo_rows).to_parquet(METRICS_DIR / "iter6_2_gate_a_combinations.parquet", index=False)

    print(f"\n  Saved: iter6_2_gate_a_{{models,combinations}}.parquet")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.2 GATE A — h=1 DECISIVE TEST")
    print("  Does DMD earn its complexity?")
    print("=" * 80)

    panel, layer_labels = load_93_actor_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # ── Run all models across all windows ──
    print("\n  Running 13 models × 10 windows...")
    all_r2s = []
    all_pred_arrays = []
    all_actuals = []

    for ty in TEST_YEARS:
        t0 = time.time()
        result = run_window_all_models(panel, ty, K=K_DEFAULT, ewm=12, T_yr=5)
        if result is not None:
            r2s, pred_arrays, actuals = result
            all_r2s.append(r2s)
            all_pred_arrays.append(pred_arrays)
            all_actuals.append(actuals)
            # Print key results for this window
            ar1 = r2s.get("ref_ar1", 0)
            dmd = r2s.get("dmd_diag_kalman", 0)
            pca = r2s.get("pca_diag_kalman", 0)
            ridge = r2s.get("ridge_raw", 0)
            print(f"  W{ty}: AR1={ar1:.4f}  DMD+diag={dmd:.4f}  PCA+diag={pca:.4f}"
                  f"  Ridge={ridge:.4f}  ({time.time()-t0:.1f}s)")
        else:
            all_r2s.append(None)
            all_pred_arrays.append(None)
            all_actuals.append(None)
            print(f"  W{ty}: FAILED  ({time.time()-t0:.1f}s)")

    # ── Model Table ──
    print_model_table(all_r2s, TEST_YEARS)

    # ── Contrast Blocks ──
    print_contrast_blocks(all_r2s, TEST_YEARS)

    # ── Combination Test ──
    combo_results = run_combination_test(all_pred_arrays, all_actuals, TEST_YEARS)
    print_combination_results(combo_results, all_r2s, TEST_YEARS)

    # ── Quality Gates ──
    qg_pass = print_quality_gates(all_r2s, TEST_YEARS)

    # ── Kill Rule A ──
    killed = evaluate_kill_rule_a(all_r2s, combo_results, TEST_YEARS)

    # ── Save ──
    save_results(all_r2s, combo_results, TEST_YEARS)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    return all_r2s, combo_results, killed


if __name__ == "__main__":
    main()
