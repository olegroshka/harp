#!/usr/bin/env python
"""
Table 5-UK -- 8 architectures on the UK/EU heterogeneous panel (experiment_b1).

Mirrors scripts/table5_architectures.py verbatim except for:
  - reads experiment_b1_intensities.parquet / experiment_b1_registry.json
  - panel window: 2011Q2 -- 2025Q4 (59 quarters)
  - test years: 2017--2024 (8 years, 32 test quarters)
  - block partition: same SEC_diversified / LAYER_macro_inst / MERGED_tech_health
    / REMAINDER scheme as the US panel (transfer test of US-learned blocks)
  - quality gates re-targeted: no US-specific R^2 expectations, instead the
    Phase-7 GO/NO-GO decision on Delta = M2 - G1 per UK_EU_PANEL_PLAN.md

Pre-registered hypothesis (UK_EU_PANEL_PLAN.md Sec. 2):
  H1: M2 > G1 on the full UK/EU panel (DM-HAC one-sided, alpha=0.05).

Usage:
    PYTHONIOENCODING=utf-8 uv run python scripts/table5_uk_eu.py
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

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.spectral.dmd import ExactDMDDecomposer
from harp.validation.metrics import oos_r_squared

# Variant + paths get set in main() based on --variant argparse argument
INTENSITIES_PATH = None  # set in main()
REGISTRY_PATH = None     # set in main()
OUT_BASENAME = None      # set in main()
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2017, 2025))  # 2017..2024 = 8 years, 32 test quarters
PANEL_START = "2011-04-01"
PANEL_END = "2025-12-31"


def _resolve_paths(variant: str, t_yr: int = 5, block_scheme: str = "us_inherited",
                    k_b_override: int = 0) -> tuple[Path, Path, str]:
    suffix = f"_{variant}" if variant else ""
    panel = f"experiment_b1{suffix}"
    out_suffix = "" if t_yr == 5 else f"_T{t_yr}"
    bs_suffix = "" if block_scheme == "us_inherited" else f"_{block_scheme}"
    k_suffix = "" if k_b_override <= 0 else f"_K{k_b_override}"
    return (
        PROJECT_ROOT / "data" / "intensities" / f"{panel}_intensities.parquet",
        PROJECT_ROOT / "data" / "registries" / f"{panel}_registry.json",
        f"table5_uk_eu{suffix}{out_suffix}{bs_suffix}{k_suffix}",
    )

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_B_OVERRIDE = 0  # 0 = use default formula min(4, max(2, N_b // 5)); set in main() from --k-b


# Data loading + infrastructure

def load_panel_and_meta():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    meta = {a["actor_id"]: a for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc[PANEL_START:PANEL_END]
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
    if ts < pd.Timestamp(PANEL_START) - pd.Timedelta(days=365):
        # Allow up to 1 year before panel start; _prepare_window will just use available data
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


# Block definitions -- two schemes available

def define_blocks_us_inherited(panel, meta):
    """Original scheme inherited from experiment_a1 (US-block transfer test).

    SEC_diversified is empty in B1; pipeline handles empty blocks gracefully.
    """
    actors = list(panel.columns)
    blocks = {}
    blocks["SEC_diversified"] = [a for a in actors
                                  if meta.get(a, {}).get("sector") == "diversified"]
    blocks["LAYER_macro_inst"] = [a for a in actors
                                   if meta.get(a, {}).get("layer", -1) in (0, 1)]
    blocks["MERGED_tech_health"] = [a for a in actors
                                     if meta.get(a, {}).get("sector") in ("technology", "healthcare")]
    local_actors = set()
    for b_actors in blocks.values():
        local_actors.update(b_actors)
    blocks["REMAINDER"] = [a for a in actors if a not in local_actors]
    return blocks


def define_blocks_sector_split(panel, meta):
    """Phase 9.4 alternative: each SMIM sector gets its own block.

    LAYER_macro_inst + 6 SMIM sectors {energy, financials, healthcare,
    technology, consumer, industrials}. REMAINDER ends up empty (every actor
    is in some block) since SMIM mapping covers all 100 firms.
    """
    actors = list(panel.columns)
    blocks = {}
    blocks["LAYER_macro_inst"] = [a for a in actors
                                   if meta.get(a, {}).get("layer", -1) in (0, 1)]
    for sec in ["energy", "financials", "healthcare", "technology", "consumer", "industrials"]:
        blocks[f"SEC_{sec}"] = [a for a in actors
                                 if meta.get(a, {}).get("sector") == sec
                                 and meta.get(a, {}).get("layer", -1) == 2]
    local_actors = set()
    for b_actors in blocks.values():
        local_actors.update(b_actors)
    blocks["REMAINDER"] = [a for a in actors if a not in local_actors]
    return blocks


def define_blocks(panel, meta, scheme: str = "us_inherited"):
    if scheme == "us_inherited":
        return define_blocks_us_inherited(panel, meta)
    if scheme == "sector_split":
        return define_blocks_sector_split(panel, meta)
    raise ValueError(f"Unknown block scheme: {scheme}")


# Local Ridge model for a block

def fit_local_ridge(residuals_block, alpha):
    X = residuals_block[:-1]
    Y = residuals_block[1:]
    N_b = X.shape[1]
    try:
        C = (np.linalg.solve(X.T @ X + alpha * np.eye(N_b), X.T @ Y)).T
    except np.linalg.LinAlgError:
        C = np.zeros((N_b, N_b))
    return C


def select_ridge_alpha(residuals_block, N_b):
    alphas = [0.1 * N_b, 1.0 * N_b, 10.0 * N_b]
    best_alpha, best_score = alphas[1], -np.inf

    X = residuals_block[:-1]
    Y = residuals_block[1:]
    T_train = X.shape[0]

    if T_train < 4:
        return best_alpha

    for alpha in alphas:
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


# Local PCA+ridge model for a block

def fit_local_pca_ridge(dm_block, K_b):
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


# Full-panel window runner with all 8 architectures

def run_window_all_architectures(panel, ty, blocks, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    actor_block = {}
    for bname, bactors in blocks.items():
        for a in bactors:
            if a in v_list:
                actor_block[a] = bname
    for a in v_list:
        if a not in actor_block:
            actor_block[a] = "REMAINDER"

    block_indices = {}
    for bname in blocks:
        block_indices[bname] = [v_list.index(a) for a in v_list if actor_block.get(a) == bname]
    local_block_names = [b for b in blocks if b != "REMAINDER"]

    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    rho_b_vec, mean_b_vec = estimate_block_ar1(otr, v_list, actor_block)
    residuals_b = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

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

    local_models = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        if len(bidx) < 5:
            local_models[bname] = None
            continue

        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        N_b = len(bidx)

        alpha = select_ridge_alpha(dm_b, N_b)
        C_ridge = fit_local_ridge(dm_b, alpha)

        K_b = K_B_OVERRIDE if K_B_OVERRIDE > 0 else min(4, max(2, N_b // 5))
        K_b = min(K_b, max(2, N_b - 2))
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        local_models[bname] = {
            "C_ridge": C_ridge, "U_pca": U_pca, "A_pca": A_pca,
            "om_b": om_b, "alpha": alpha, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

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
        K_b = K_B_OVERRIDE if K_B_OVERRIDE > 0 else min(4, max(2, N_b // 5))
        K_b = min(K_b, max(2, N_b - 2))
        U_pca_b, A_pca_b = fit_local_pca_ridge(dm_bb, K_b)
        local_models_b[bname] = {
            "U_pca": U_pca_b, "A_pca": A_pca_b,
            "om_b": om_bb, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    preds = {a: [] for a in archs}
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64) if qd in ad.index else np.zeros((0, N))
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        y_pool = bar_y + rho * (prev - bar_y)
        y_pool_b = mean_b_vec + rho_b_vec * (prev - mean_b_vec)

        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_global_aug = y_pool + resid_pred_global

        preds["G0"].append(y_pool.copy())
        preds["G1"].append(y_global_aug.copy())

        y_s1 = y_global_aug.copy()
        for bname in local_block_names:
            bidx = block_indices[bname]
            y_s1[bidx] = y_pool[bidx]
        preds["S1"].append(y_s1)

        y_m1 = y_global_aug.copy()
        prev_resid = prev - (bar_y + rho * (np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)) if otr.shape[0] >= 2 else np.zeros(N)
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            local_pred = lm["C_ridge"] @ prev_resid_b + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_m1[bidx] = y_pool[bidx] + local_pred
            else:
                y_m1[bidx] = y_pool[bidx]
        preds["M1"].append(y_m1)

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

        preds["BA"].append(y_pool_b.copy())

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

        preds["ENS"].append(0.5 * y_global_aug + 0.5 * y_pool_b)

        actuals.append(obs)

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

        rho_b_vec, mean_b_vec = estimate_block_ar1(otr, v_list, actor_block)
        residuals_b_new = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

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


# Reporting

def print_results(all_results, blocks):
    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    labels = {
        "G0": "Pooled-only",
        "BA": "Block-specific rho_b+FE",
        "G1": "Global always-on",
        "S1": "Selective-off",
        "M1": "Mixture (Ridge)",
        "M2": "Mixture (PCA+ridge)",
        "BA_M2": "Block rho_b + local S2",
        "ENS": "Ensemble(G1,BA)",
    }

    print("\n" + "=" * 90)
    print("  FULL-PANEL R^2 COMPARISON  (UK/EU panel, experiment_b1)")
    print("=" * 90)

    print(f"\n  {'Architecture':<25s} {'R^2':>8s} {'Delta vs G1':>12s} {'t':>7s} {'p':>7s}"
          f" {'CI':>23s} {'W':>5s}")
    print(f"  {'-'*88}")

    for arch in archs:
        r2s = [all_results[i][arch]["full"] for i in range(len(TEST_YEARS))
               if all_results[i] is not None and all_results[i][arch]["full"] is not None]
        mean_r2 = np.mean(r2s) if r2s else np.nan

        if arch == "G1":
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f} {'--':>12s} {'--':>7s} {'--':>7s}"
                  f" {'--':>23s}")
            continue

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
            beat = " *" if np.isfinite(ci[0]) and ci[0] > 0 else ""
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f} {mean_d:+12.4f} {t_s:>7s} {p_s:>7s}"
                  f" {ci_s:>23s} {wins:>2d}/{len(deltas)}{beat}")
        else:
            print(f"  {labels[arch]:<25s} {mean_r2:8.4f}")

    print("\n" + "=" * 90)
    print("  PER-BLOCK R^2 DECOMPOSITION")
    print("=" * 90)

    for bname in blocks:
        N_b = len(blocks[bname])
        if N_b == 0:
            print(f"\n  {bname} (N=0): SKIPPED (empty in B1 -- expected)")
            continue
        print(f"\n  {bname} (N={N_b}):")
        print(f"    {'Architecture':<25s} {'R^2':>8s} {'Delta vs G1':>12s}")
        print(f"    {'-'*47}")

        g1_block = [all_results[i]["G1"]["blocks"].get(bname)
                     for i in range(len(TEST_YEARS)) if all_results[i] is not None]
        g1_mean = np.nanmean(g1_block) if g1_block else np.nan

        for arch in archs:
            block_r2s = [all_results[i][arch]["blocks"].get(bname)
                          for i in range(len(TEST_YEARS)) if all_results[i] is not None]
            mean_br = np.nanmean(block_r2s) if block_r2s else np.nan
            delta = mean_br - g1_mean if np.isfinite(mean_br) and np.isfinite(g1_mean) else np.nan
            d_s = f"{delta:+12.4f}" if np.isfinite(delta) else "         N/A"
            print(f"    {labels[arch]:<25s} {mean_br:8.4f} {d_s}")


def evaluate_g7_gate(all_results):
    """Phase 7 GO/NO-GO gate per UK_EU_PANEL_PLAN.md.

    H1 evaluation prep: Delta = M2 - G1.
      Delta > 0:               GO -- proceed to P8 statistical battery.
      Delta in [-0.010, 0]:    NULL -- proceed to P8, expect non-significant DM-HAC.
      Delta < -0.010:          FUTILITY -- pause and reassess framing.
    """
    print("\n" + "=" * 90)
    print("  GATE G7 -- Phase-7 GO/NO-GO DECISION (UK_EU_PANEL_PLAN.md Sec. 6, Phase 7)")
    print("=" * 90)

    deltas_m2 = []
    for i in range(len(TEST_YEARS)):
        if all_results[i] is None:
            continue
        m2 = all_results[i]["M2"]["full"]
        g1 = all_results[i]["G1"]["full"]
        if m2 is not None and g1 is not None:
            deltas_m2.append(m2 - g1)

    if not deltas_m2:
        print("  ERROR: no usable windows -- cannot evaluate Gate G7.")
        return "ERROR"

    mean_d = float(np.mean(deltas_m2))
    ci = bootstrap_ci(deltas_m2)
    t_stat, p_val = paired_t_test(deltas_m2)
    wins = sum(1 for d in deltas_m2 if d > 0)
    n = len(deltas_m2)

    print(f"  H1 (M2 > G1):  Delta = {mean_d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
          f"  t = {t_stat:.2f}  p = {p_val:.4f}  wins = {wins}/{n}")
    print()

    if mean_d > 0.0:
        verdict = "GO"
        msg = "M2 > G1 in mean -- proceed to Phase 8 (DM-HAC + placebo)."
    elif mean_d >= -0.010:
        verdict = "NULL"
        msg = "Null result (Delta in [-0.010, 0]) -- proceed to Phase 8 but expect non-significant DM-HAC. Still publishable as scope-narrowing."
    else:
        verdict = "FUTILITY"
        msg = "Delta < -0.010 -- futility threshold hit. Pause, reassess framing per Stop-for-futility rule (UK_EU_PANEL_PLAN.md Sec. 2)."

    print(f"  Verdict: {verdict}")
    print(f"  -> {msg}")

    print()
    print(f"  US-panel Delta for context: +0.047 (93-actor, 84Q)")
    return verdict


def evaluate_win_condition(all_results):
    print("\n" + "=" * 90)
    print("  WIN CONDITION -- per architecture vs G1")
    print("=" * 90)

    for arch, label in [("M1", "Mixture Ridge"), ("M2", "Mixture PCA+ridge"), ("S1", "Selective-off"),
                         ("BA", "Block rho_b+FE"), ("BA_M2", "Block rho_b + local S2"), ("ENS", "Ensemble(G1,BA)")]:
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
            print(f"    Delta R^2 = {mean_d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            print(f"    >= +0.005? {'YES' if threshold else 'NO'}  |  CI > 0? {'YES' if wins else 'NO'}")

            if wins and threshold:
                print(f"    -> WIN CONDITION MET *")
            elif wins:
                print(f"    -> CI excludes zero but below +0.005 threshold")
            else:
                print(f"    -> Does not beat global always-on")


# Main

def _summary_dict(all_results, blocks, verdict):
    """Compact dict of headline numbers for JSON dump."""
    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    summary = {
        "panel": "experiment_b1 (UK/EU heterogeneous)",
        "panel_window": f"{PANEL_START} to {PANEL_END}",
        "test_years": TEST_YEARS,
        "n_test_quarters": 4 * len(TEST_YEARS),
        "blocks": {bname: len(bactors) for bname, bactors in blocks.items()},
        "g7_verdict": verdict,
        "architectures": {},
        "deltas_vs_G1": {},
    }
    for arch in archs:
        r2s = [all_results[i][arch]["full"] for i in range(len(TEST_YEARS))
               if all_results[i] is not None and all_results[i][arch]["full"] is not None]
        summary["architectures"][arch] = {
            "mean_full_r2": float(np.mean(r2s)) if r2s else None,
            "per_window_r2": r2s,
            "per_block_mean_r2": {},
        }
        for bname in blocks:
            br2s = [all_results[i][arch]["blocks"].get(bname)
                    for i in range(len(TEST_YEARS))
                    if all_results[i] is not None]
            br2s = [x for x in br2s if x is not None and np.isfinite(x)]
            if br2s:
                summary["architectures"][arch]["per_block_mean_r2"][bname] = float(np.mean(br2s))

    for arch in archs:
        if arch == "G1":
            continue
        deltas = []
        for i in range(len(TEST_YEARS)):
            if all_results[i] is None:
                continue
            a = all_results[i][arch]["full"]
            g = all_results[i]["G1"]["full"]
            if a is not None and g is not None:
                deltas.append(a - g)
        if deltas:
            t_stat, p_val = paired_t_test(deltas)
            ci_lo, ci_hi = bootstrap_ci(deltas)
            summary["deltas_vs_G1"][arch] = {
                "mean": float(np.mean(deltas)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "t": t_stat, "p": p_val,
                "wins": sum(1 for d in deltas if d > 0),
                "n": len(deltas),
            }

    return summary


def main():
    global INTENSITIES_PATH, REGISTRY_PATH, OUT_BASENAME, K_B_OVERRIDE

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="",
                        help="Panel variant suffix (e.g. 'eu_only' for experiment_b1_eu_only_*)")
    parser.add_argument("--t-yr", type=int, default=5,
                        help="Training-window length in years (default 5; Phase 9.2 sweep uses {2,3,5,8})")
    parser.add_argument("--block-scheme", type=str, default="us_inherited",
                        choices=["us_inherited", "sector_split"],
                        help="Block partition: us_inherited (3+REMAINDER, default) or sector_split (LAYER_macro_inst + 6 SMIM sectors)")
    parser.add_argument("--k-b", type=int, default=0,
                        help="Override local-block PCA dimension K_b (0 = use default formula, "
                             "any positive value forces all blocks to K_b=this value, capped at N_b-2)")
    args = parser.parse_args()
    K_B_OVERRIDE = args.k_b
    INTENSITIES_PATH, REGISTRY_PATH, OUT_BASENAME = _resolve_paths(
        args.variant, args.t_yr, args.block_scheme, args.k_b)

    t_start = time.time()

    print("=" * 100)
    print(f"  TABLE 5-UK -- 8 architectures on {INTENSITIES_PATH.stem}")
    print(f"  Variant: '{args.variant or 'full'}' | T_yr: {args.t_yr} | Block scheme: {args.block_scheme}"
          f" | K_b: {'auto' if args.k_b == 0 else args.k_b}")
    print(f"  Output basename: {OUT_BASENAME}")
    print("  Phase 7 / 9.{1,2,4} of UK_EU_PANEL_PLAN.md; pre-registered hypothesis H1: M2 > G1")
    print("=" * 100)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta, scheme=args.block_scheme)

    print(f"\nPanel: {panel.shape[0]}Q x {panel.shape[1]} actors  ({PANEL_START} to {PANEL_END})")
    print(f"Test years: {TEST_YEARS[0]}..{TEST_YEARS[-1]} ({len(TEST_YEARS)} years, {4*len(TEST_YEARS)} quarters)")
    print(f"\nBlocks:")
    for bname, bactors in sorted(blocks.items()):
        print(f"  {bname}: N={len(bactors)}")

    all_results = []
    for ty in TEST_YEARS:
        t0 = time.time()
        result = run_window_all_architectures(panel, ty, blocks, T_yr=args.t_yr)
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

    print_results(all_results, blocks)
    verdict = evaluate_g7_gate(all_results)
    evaluate_win_condition(all_results)

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
    pd.DataFrame(rows).to_parquet(METRICS_DIR / f"{OUT_BASENAME}.parquet", index=False)

    summary = _summary_dict(all_results, blocks, verdict)
    with open(METRICS_DIR / f"{OUT_BASENAME}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: results/metrics/{OUT_BASENAME}.parquet")
    print(f"  Saved: results/metrics/{OUT_BASENAME}.json")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
