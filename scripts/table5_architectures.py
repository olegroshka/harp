#!/usr/bin/env python
"""
Iteration 6.4b — Heterogeneity-Aware Local Decomposition.

Eight architectures on the full 93-actor panel:
  G0:     Pooled-only (no augmentation)
  BA:     Block-specific rho_b + FE (no Stage 2)
  G1:     Global always-on augmentation (C1 full Ã)
  S1:     Selective-off (pooled-only for local blocks, global for REMAINDER)
  M1:     Mixture (local Ridge for local blocks, global for REMAINDER)
  M2:     Mixture (local PCA+ridge for local blocks, global for REMAINDER)
  BA_M2:  Block-specific rho_b + block-specific Stage 2 PCA+ridge  [A-1]
  ENS:    Equal-weighted ensemble of G1 and BA predictions           [C-3]

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b.py
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

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3


# ══════════════════════════════════════════════════════════════════════
#  Data loading + infrastructure
# ══════════════════════════════════════════════════════════════════════

def load_panel_and_meta():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    meta = {a["actor_id"]: a for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel, meta


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    return float(num / den) if den > 1e-12 else 0.0, bar_y


def estimate_block_ar1(otr, v_list, actor_block):
    """Estimate block-specific rho_b and per-actor means."""
    N = otr.shape[1]
    rho_vec = np.zeros(N)
    mean_vec = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    # Group actor indices by block
    block_idx = {}
    for i, a in enumerate(v_list):
        b = actor_block.get(a, "REMAINDER")
        block_idx.setdefault(b, []).append(i)
    for b, idx in block_idx.items():
        bd = otr[:, idx]
        by = np.nan_to_num(bd.mean(axis=0), nan=0.5)
        tl = bd - by
        nm = np.sum(tl[1:] * tl[:-1])
        dn = np.sum(tl[:-1] ** 2)
        rb = float(nm / dn) if dn > 1e-12 else 0.0
        for j, ii in enumerate(idx):
            rho_vec[ii] = rb
            mean_vec[ii] = by[j]
    return rho_vec, mean_vec


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def _clip_sr(F, max_sr=0.99):
    ev = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(ev)))
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
    if N < 5:
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
    return float(m / se), float(2 * t_dist.sf(abs(m / se), df=len(d) - 1))


# ══════════════════════════════════════════════════════════════════════
#  Block definitions (economically pre-specified)
# ══════════════════════════════════════════════════════════════════════

def define_blocks(panel, meta):
    """Pre-specified blocks with economic rationale."""
    actors = list(panel.columns)
    blocks = {}

    # SEC_diversified: heterogeneous sector, global basis misrepresents
    blocks["SEC_diversified"] = [a for a in actors
                                  if meta.get(a, {}).get("sector") == "diversified"]

    # LAYER_macro_inst: different data types (FRED + institutional)
    blocks["LAYER_macro_inst"] = [a for a in actors
                                   if meta.get(a, {}).get("layer", -1) in (0, 1)]

    # MERGED_tech_health: coherent sub-panel (modes 1-2 load here)
    blocks["MERGED_tech_health"] = [a for a in actors
                                     if meta.get(a, {}).get("sector") in ("technology", "healthcare")]

    # REMAINDER: everything else
    local_actors = set()
    for b_actors in blocks.values():
        local_actors.update(b_actors)
    blocks["REMAINDER"] = [a for a in actors if a not in local_actors]

    return blocks


# ══════════════════════════════════════════════════════════════════════
#  Local Ridge model for a block
# ══════════════════════════════════════════════════════════════════════

def fit_local_ridge(residuals_block, alpha):
    """Fit Ridge on block residuals: r_{t+1} = C · r_t.

    Returns C (N_b × N_b).
    """
    X = residuals_block[:-1]  # (T-1, N_b)
    Y = residuals_block[1:]   # (T-1, N_b)
    N_b = X.shape[1]
    try:
        C = (np.linalg.solve(X.T @ X + alpha * np.eye(N_b), X.T @ Y)).T
    except np.linalg.LinAlgError:
        C = np.zeros((N_b, N_b))
    return C


def select_ridge_alpha(residuals_block, N_b):
    """LOO CV to select Ridge α from grid."""
    alphas = [0.1 * N_b, 1.0 * N_b, 10.0 * N_b]
    best_alpha, best_score = alphas[1], -np.inf

    X = residuals_block[:-1]
    Y = residuals_block[1:]
    T_train = X.shape[0]

    if T_train < 4:
        return best_alpha

    for alpha in alphas:
        # Simple hold-last-2 validation
        X_tr, Y_tr = X[:-2], Y[:-2]
        X_val, Y_val = X[-2:], Y[-2:]
        if X_tr.shape[0] < 2:
            continue
        try:
            C = (np.linalg.solve(X_tr.T @ X_tr + alpha * np.eye(N_b), X_tr.T @ Y_tr)).T
            pred = (C @ X_val.T).T
            score = -np.mean((Y_val - pred) ** 2)
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


# ══════════════════════════════════════════════════════════════════════
#  Local PCA+ridge model for a block
# ══════════════════════════════════════════════════════════════════════

def fit_local_pca_ridge(dm_block, K_b):
    """Fit PCA+ridge VAR on block demeaned residuals.

    Returns U_pca (N_b × K_b), A_ridge (K_b × K_b).
    """
    N_b = dm_block.shape[1]
    ka = min(K_b, N_b - 2)

    C = dm_block.T @ dm_block / dm_block.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    U_pca = eigvecs[:, idx][:, :ka]

    factors = dm_block @ U_pca
    X, Y = factors[:-1], factors[1:]
    try:
        A = (np.linalg.solve(X.T @ X + 1.0 * np.eye(ka), X.T @ Y)).T
    except np.linalg.LinAlgError:
        A = np.eye(ka) * 0.5

    return U_pca, A


# ══════════════════════════════════════════════════════════════════════
#  Full-panel window runner with all 5 architectures
# ══════════════════════════════════════════════════════════════════════

def run_window_all_architectures(panel, ty, blocks, T_yr=5):
    """Run all 8 architectures for one test year.

    Returns per-architecture full-panel predictions and actuals.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    # Map actors to blocks
    actor_block = {}
    for bname, bactors in blocks.items():
        for a in bactors:
            if a in v_list:
                actor_block[a] = bname
    # Actors not in any block → REMAINDER
    for a in v_list:
        if a not in actor_block:
            actor_block[a] = "REMAINDER"

    # Block indices
    block_indices = {}
    for bname in blocks:
        block_indices[bname] = [v_list.index(a) for a in v_list if actor_block.get(a) == bname]
    local_block_names = [b for b in blocks if b != "REMAINDER"]

    # ── Stage 1: Global pooled+FE ──
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # ── Stage 1 alt: Block-specific rho_b+FE (for BA, BA_M2, ENS) ──
    rho_b_vec, mean_b_vec = estimate_block_ar1(otr, v_list, actor_block)
    residuals_b = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

    # ── Global C1 augmentation (for G1, S1 remainder, M1 remainder, M2 remainder) ──
    om_r = ewm_demean(residuals, 12)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None

    ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
    A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
    F_r = _clip_sr(A_r)
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    # ── Local models per block ──
    local_models = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        if len(bidx) < 5:
            local_models[bname] = None
            continue

        # Extract block residuals
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        N_b = len(bidx)

        # Ridge
        alpha = select_ridge_alpha(dm_b, N_b)
        C_ridge = fit_local_ridge(dm_b, alpha)

        # PCA+ridge
        K_b = min(4, max(2, N_b // 5))
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        local_models[bname] = {
            "C_ridge": C_ridge, "U_pca": U_pca, "A_pca": A_pca,
            "om_b": om_b, "alpha": alpha, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

    # ── Local models on block-specific residuals (for BA_M2) ──
    local_models_b = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        if len(bidx) < 5:
            local_models_b[bname] = None
            continue
        block_resids_b = residuals_b[:, bidx]
        om_bb = ewm_demean(block_resids_b, 12)
        dm_bb = block_resids_b - om_bb
        N_b = len(bidx)
        K_b = min(4, max(2, N_b // 5))
        U_pca_b, A_pca_b = fit_local_pca_ridge(dm_bb, K_b)
        local_models_b[bname] = {
            "U_pca": U_pca_b, "A_pca": A_pca_b,
            "om_b": om_bb, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

    # ── Rolling test ──
    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    preds = {a: [] for a in archs}
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # Stage 1 prediction (global pooled — same for G0/G1/S1/M1/M2)
        y_pool = bar_y + rho * (prev - bar_y)

        # Stage 1 prediction (block-specific — for BA/BA_M2/ENS)
        y_pool_b = mean_b_vec + rho_b_vec * (prev - mean_b_vec)

        # Global augmented prediction
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_global_aug = y_pool + resid_pred_global

        # G0: pooled-only
        preds["G0"].append(y_pool.copy())

        # G1: global always-on
        preds["G1"].append(y_global_aug.copy())

        # S1: selective-off (pooled for local blocks, global-aug for REMAINDER)
        y_s1 = y_global_aug.copy()
        for bname in local_block_names:
            bidx = block_indices[bname]
            y_s1[bidx] = y_pool[bidx]  # pooled-only for local blocks
        preds["S1"].append(y_s1)

        # M1: mixture with local Ridge
        y_m1 = y_global_aug.copy()  # start with global for REMAINDER
        prev_resid = prev - (bar_y + rho * (np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)) if otr.shape[0] >= 2 else np.zeros(N)
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                y_m1[lm["bidx"]] = y_pool[lm["bidx"]] if lm else y_pool[block_indices[bname]]
                continue
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            local_pred = lm["C_ridge"] @ prev_resid_b + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_m1[bidx] = y_pool[bidx] + local_pred
            else:
                y_m1[bidx] = y_pool[bidx]
        preds["M1"].append(y_m1)

        # M2: mixture with local PCA+ridge
        y_m2 = y_global_aug.copy()
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            f_pca = lm["U_pca"].T @ prev_resid_b
            local_pred = lm["U_pca"] @ (lm["A_pca"] @ f_pca) + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_m2[bidx] = y_pool[bidx] + local_pred
            else:
                y_m2[bidx] = y_pool[bidx]
        preds["M2"].append(y_m2)

        # BA: block-specific rho_b+FE (no Stage 2)
        preds["BA"].append(y_pool_b.copy())

        # BA_M2: block-specific rho_b + block-specific local PCA+ridge Stage 2
        # Remainder gets block-specific pooled only (no global aug — consistent
        # with using rho_b everywhere)
        y_bam2 = y_pool_b.copy()
        prev_resid_block = prev - (mean_b_vec + rho_b_vec * (
            np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - mean_b_vec)
        ) if otr.shape[0] >= 2 else np.zeros(N)
        for bname in local_block_names:
            lmb = local_models_b.get(bname)
            if lmb is None:
                continue
            bidx = lmb["bidx"]
            prev_resid_bb = prev_resid_block[bidx] - lmb["om_b"].ravel()
            f_pca = lmb["U_pca"].T @ prev_resid_bb
            local_pred = lmb["U_pca"] @ (lmb["A_pca"] @ f_pca) + lmb["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_bam2[bidx] = y_pool_b[bidx] + local_pred
        preds["BA_M2"].append(y_bam2)

        # ENS: equal-weighted ensemble of G1 and BA
        preds["ENS"].append(0.5 * y_global_aug + 0.5 * y_pool_b)

        actuals.append(obs)

        # ── Kalman update (global) ──
        actual_resid = obs - y_pool
        odm_r = actual_resid - om_r.ravel()
        S_r = U_r @ Pp_r @ U_r.T + R_r
        try:
            Kg_r = Pp_r @ U_r.T @ np.linalg.solve(S_r, np.eye(N))
        except Exception:
            Kg_r = np.zeros((ka, N))
        a_r = ap_r + Kg_r @ (odm_r - U_r @ ap_r)
        P_r = (np.eye(ka) - Kg_r @ U_r) @ Pp_r
        inn_r = a_r - ap_r
        Q_r = (1 - LAMBDA_Q) * Q_r + LAMBDA_Q * np.outer(inn_r, inn_r)
        Q_r = (Q_r + Q_r.T) / 2 + np.eye(ka) * 1e-6
        prev = obs

        # Rolling updates
        otr = np.vstack([otr, qv])
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, 12)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            k2 = min(K_DEFAULT, mf_r2.basis.shape[0] - 2, mf_r2.K)
            A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
            F_r = _clip_sr(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2

        # Re-estimate block-specific AR(1)
        rho_b_vec, mean_b_vec = estimate_block_ar1(otr, v_list, actor_block)
        residuals_b_new = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

        # Re-estimate local models (on global residuals)
        residuals = residuals_new
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            N_b = lm["N_b"]

            alpha = select_ridge_alpha(dm_b, N_b)
            lm["C_ridge"] = fit_local_ridge(dm_b, alpha)
            lm["om_b"] = om_b
            lm["alpha"] = alpha

            K_b = lm["K_b"]
            lm["U_pca"], lm["A_pca"] = fit_local_pca_ridge(dm_b, K_b)

        # Re-estimate local models on block-specific residuals (for BA_M2)
        residuals_b = residuals_b_new
        for bname in local_block_names:
            lmb = local_models_b.get(bname)
            if lmb is None:
                continue
            bidx = lmb["bidx"]
            block_resids_b = residuals_b[:, bidx]
            om_bb = ewm_demean(block_resids_b, 12)
            dm_bb = block_resids_b - om_bb
            K_b = lmb["K_b"]
            lmb["U_pca"], lmb["A_pca"] = fit_local_pca_ridge(dm_bb, K_b)
            lmb["om_b"] = om_bb

    if not actuals:
        return None

    act_a = np.array(actuals)

    # Compute per-architecture R² (full panel + per block)
    results = {}
    for arch in archs:
        pred_a = np.array(preds[arch])
        if not np.all(np.isfinite(pred_a)):
            results[arch] = {"full": None, "blocks": {}}
            continue

        full_r2 = float(oos_r_squared(pred_a.ravel(), act_a.ravel()))
        block_r2 = {}
        for bname, bidx in block_indices.items():
            if bidx:
                br2 = float(oos_r_squared(pred_a[:, bidx].ravel(), act_a[:, bidx].ravel()))
                block_r2[bname] = br2
        results[arch] = {"full": full_r2, "blocks": block_r2}

    return results


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════

def print_results(all_results, blocks):
    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    labels = {
        "G0": "Pooled-only",
        "BA": "Block-specific ρ_b+FE",
        "G1": "Global always-on",
        "S1": "Selective-off",
        "M1": "Mixture (Ridge)",
        "M2": "Mixture (PCA+ridge)",
        "BA_M2": "Block ρ_b + local S2",
        "ENS": "Ensemble(G1,BA)",
    }

    # Full-panel comparison
    print("\n" + "=" * 90)
    print("  FULL-PANEL R² COMPARISON")
    print("=" * 90)

    g1_r2s = [all_results[i]["G1"]["full"] for i in range(len(TEST_YEARS))
              if all_results[i] is not None]

    print(f"\n  {'Architecture':<25s} {'R²':>8s} {'Δ vs G1':>9s} {'t':>7s} {'p':>7s}"
          f" {'CI':>23s} {'W':>5s}")
    print(f"  {'-'*85}")

    for arch in archs:
        r2s = [all_results[i][arch]["full"] for i in range(len(TEST_YEARS))
               if all_results[i] is not None and all_results[i][arch]["full"] is not None]
        mean_r2 = np.mean(r2s) if r2s else np.nan

        if arch == "G1":
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f} {'—':>9s} {'—':>7s} {'—':>7s}"
                  f" {'—':>23s}")
            continue

        # Paired delta vs G1
        deltas = []
        for i in range(len(TEST_YEARS)):
            if all_results[i] is None:
                continue
            a_r2 = all_results[i][arch]["full"]
            g_r2 = all_results[i]["G1"]["full"]
            if a_r2 is not None and g_r2 is not None:
                deltas.append(a_r2 - g_r2)

        if deltas:
            mean_d = np.mean(deltas)
            t_stat, p_val = paired_t_test(deltas)
            ci = bootstrap_ci(deltas)
            wins = sum(1 for d in deltas if d > 0)
            ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if np.isfinite(ci[0]) else "N/A"
            t_s = f"{t_stat:.2f}" if np.isfinite(t_stat) else "N/A"
            p_s = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
            beat = " ★" if np.isfinite(ci[0]) and ci[0] > 0 else ""
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f} {mean_d:+9.4f} {t_s:>7s} {p_s:>7s}"
                  f" {ci_s:>23s} {wins:>2d}/{len(deltas)}{beat}")
        else:
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f}")

    # Per-block decomposition
    print("\n" + "=" * 90)
    print("  PER-BLOCK R² DECOMPOSITION")
    print("=" * 90)

    for bname in blocks:
        N_b = len(blocks[bname])
        print(f"\n  {bname} (N={N_b}):")
        print(f"    {'Architecture':<25s} {'R²':>8s} {'Δ vs G1':>9s}")
        print(f"    {'-'*44}")

        g1_block = [all_results[i]["G1"]["blocks"].get(bname)
                     for i in range(len(TEST_YEARS)) if all_results[i] is not None]
        g1_mean = np.nanmean(g1_block) if g1_block else np.nan

        for arch in archs:
            block_r2s = [all_results[i][arch]["blocks"].get(bname)
                          for i in range(len(TEST_YEARS)) if all_results[i] is not None]
            mean_br = np.nanmean(block_r2s) if block_r2s else np.nan
            delta = mean_br - g1_mean if np.isfinite(mean_br) and np.isfinite(g1_mean) else np.nan
            d_s = f"{delta:+9.4f}" if np.isfinite(delta) else "      N/A"
            print(f"    {labels[arch]:<25s} {mean_br:8.4f} {d_s}")

    # Diagnostic: M1 vs S1
    print("\n" + "=" * 90)
    print("  DIAGNOSTIC: M1 (local Ridge) vs S1 (selective-off)")
    print("  Does local modelling add value beyond harm-removal?")
    print("=" * 90)

    m1_vs_s1 = []
    for i in range(len(TEST_YEARS)):
        if all_results[i] is None:
            continue
        m1 = all_results[i]["M1"]["full"]
        s1 = all_results[i]["S1"]["full"]
        if m1 is not None and s1 is not None:
            m1_vs_s1.append(m1 - s1)

    if m1_vs_s1:
        mean_d = np.mean(m1_vs_s1)
        t_stat, p_val = paired_t_test(m1_vs_s1)
        ci = bootstrap_ci(m1_vs_s1)
        print(f"  M1 − S1: Δ = {mean_d:+.4f}  t = {t_stat:.2f}  p = {p_val:.4f}"
              f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        if mean_d > 0.002:
            print(f"  → Local Ridge adds value beyond harm-removal")
        elif mean_d < -0.002:
            print(f"  → Local Ridge is WORSE than just turning off augmentation")
        else:
            print(f"  → Local Ridge ≈ selective-off (gain is from harm-removal)")


def evaluate_quality_gates(all_results):
    print("\n" + "=" * 90)
    print("  QUALITY GATES")
    print("=" * 90)

    g1_full = [all_results[i]["G1"]["full"] for i in range(len(TEST_YEARS))
               if all_results[i] is not None and all_results[i]["G1"]["full"] is not None]
    g0_full = [all_results[i]["G0"]["full"] for i in range(len(TEST_YEARS))
               if all_results[i] is not None and all_results[i]["G0"]["full"] is not None]

    g1_mean = np.mean(g1_full) if g1_full else np.nan
    g0_mean = np.mean(g0_full) if g0_full else np.nan

    qg1 = abs(g1_mean - 0.630) <= 0.005
    qg2 = abs(g0_mean - 0.591) <= 0.005
    qg3 = all(all_results[i] is not None for i in range(len(TEST_YEARS)))

    print(f"  QG1: Global aug R² = {g1_mean:.4f} ≈ 0.630 ±0.005 — {'PASS' if qg1 else 'FAIL'}")
    print(f"  QG2: Pooled-only R² = {g0_mean:.4f} ≈ 0.591 ±0.005 — {'PASS' if qg2 else 'FAIL'}")
    print(f"  QG3: All windows valid — {'PASS' if qg3 else 'FAIL'}")


def evaluate_win_condition(all_results):
    print("\n" + "=" * 90)
    print("  WIN CONDITION EVALUATION")
    print("=" * 90)

    for arch, label in [("M1", "Mixture Ridge"), ("M2", "Mixture PCA+ridge"), ("S1", "Selective-off"),
                         ("BA", "Block ρ_b+FE"), ("BA_M2", "Block ρ_b + local S2"), ("ENS", "Ensemble(G1,BA)")]:
        deltas = []
        for i in range(len(TEST_YEARS)):
            if all_results[i] is None:
                continue
            a = all_results[i][arch]["full"]
            g = all_results[i]["G1"]["full"]
            if a is not None and g is not None:
                deltas.append(a - g)

        if deltas:
            mean_d = np.mean(deltas)
            ci = bootstrap_ci(deltas)
            wins = np.isfinite(ci[0]) and ci[0] > 0
            threshold = mean_d >= 0.005

            print(f"\n  {label} vs Global:")
            print(f"    ΔR² = {mean_d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            print(f"    ≥ +0.005? {'YES' if threshold else 'NO'}  |  CI > 0? {'YES' if wins else 'NO'}")

            if wins and threshold:
                print(f"    → WIN CONDITION MET ★")
            elif wins:
                print(f"    → CI excludes zero but below +0.005 threshold")
            else:
                print(f"    → Does not beat global always-on")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 100)
    print("  ITERATION 6.4b — HETEROGENEITY-AWARE LOCAL DECOMPOSITION")
    print("  8 architectures: G0 (pooled) | BA (block ρ_b) | G1 (global) | S1 (selective-off)")
    print("                   M1 (mixture Ridge) | M2 (mixture PCA+ridge)")
    print("                   BA_M2 (block ρ_b + local S2) | ENS (G1+BA ensemble)")
    print("=" * 100)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta)

    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")
    print(f"\nBlocks:")
    for bname, bactors in sorted(blocks.items()):
        print(f"  {bname}: N={len(bactors)}")

    # Run all architectures across all windows
    all_results = []
    for ty in TEST_YEARS:
        t0 = time.time()
        result = run_window_all_architectures(panel, ty, blocks)
        all_results.append(result)
        if result:
            g0 = result["G0"]["full"]
            ba = result["BA"]["full"]
            g1 = result["G1"]["full"]
            s1 = result["S1"]["full"]
            m1 = result["M1"]["full"]
            m2 = result["M2"]["full"]
            bam2 = result["BA_M2"]["full"]
            ens = result["ENS"]["full"]
            print(f"  W{ty}: G0={g0:.4f}  BA={ba:.4f}  G1={g1:.4f}  S1={s1:.4f}"
                  f"  M1={m1:.4f}  M2={m2:.4f}  BA_M2={bam2:.4f}  ENS={ens:.4f}"
                  f"  ({time.time()-t0:.1f}s)")
        else:
            print(f"  W{ty}: FAILED")

    # Report
    print_results(all_results, blocks)
    evaluate_quality_gates(all_results)
    evaluate_win_condition(all_results)

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ty in enumerate(TEST_YEARS):
        if all_results[i] is None:
            continue
        for arch in ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]:
            row = {"year": ty, "architecture": arch,
                   "full_r2": all_results[i][arch]["full"]}
            for bname, br2 in all_results[i][arch].get("blocks", {}).items():
                row[f"block_{bname}"] = br2
            rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_4b.parquet", index=False)
    print(f"\n  Saved: iter6_4b.parquet")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
