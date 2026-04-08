#!/usr/bin/env python
"""
Iteration 6.4b Cross-Panel Validation.

Tests the mixture architecture on:
  - 146-firm CapEx/Revenue (homogeneous — mixture may NOT help)
  - 270-actor multi-ratio (capexrev vs revass — mixture SHOULD help)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_xpanel.py
"""
from __future__ import annotations

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

EDGAR_PATH = PROJECT_ROOT / "data" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3


# ══════════════════════════════════════════════════════════════════════
#  Shared infrastructure
# ══════════════════════════════════════════════════════════════════════

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
#  Panel loaders
# ══════════════════════════════════════════════════════════════════════

def _build_ratio_panel(edgar, num_tag, den_tag):
    num = edgar[edgar["tag"] == num_tag][["ticker", "event_date", "value"]].copy()
    den = edgar[edgar["tag"] == den_tag][["ticker", "event_date", "value"]].copy()
    for df in [num, den]:
        df["q"] = df["event_date"].dt.to_period("Q").dt.to_timestamp()
    num = num.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    den = den.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    m = num.merge(den, on=["ticker", "q"], suffixes=("_n", "_d"))
    m["ratio"] = m["value_n"] / m["value_d"]
    m = m.replace([np.inf, -np.inf], np.nan)
    p = m.pivot_table(index="q", columns="ticker", values="ratio")
    p.index = pd.to_datetime(p.index)
    r = p.rank(axis=1, method="average", pct=True)
    good = r.columns[r.notna().mean() > 0.50]
    return r[good].loc["2005-01-01":"2025-12-31"]


def load_146_panel():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    return _build_ratio_panel(edgar, "PaymentsToAcquirePropertyPlantAndEquipment", "Revenues")


def load_270_panel():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    cr = _build_ratio_panel(edgar, "PaymentsToAcquirePropertyPlantAndEquipment", "Revenues")
    ra = _build_ratio_panel(edgar, "Revenues", "Assets")
    overlap = sorted(set(cr.columns) & set(ra.columns))
    cr = cr[overlap]; ra = ra[overlap]
    common_idx = cr.index.intersection(ra.index)
    cr = cr.loc[common_idx]; ra = ra.loc[common_idx]
    cr.columns = [f"{t}_capexrev" for t in cr.columns]
    ra.columns = [f"{t}_revass" for t in ra.columns]
    return pd.concat([cr, ra], axis=1)


# ══════════════════════════════════════════════════════════════════════
#  Run global + mixture for a panel with given block partition
# ══════════════════════════════════════════════════════════════════════

def run_window_global_and_mixture(panel, ty, block_map, T_yr=5):
    """Run global C1 and mixture PCA+ridge for one window.

    block_map: dict actor_id → block_name. Actors with block_name != "REMAINDER"
    get local PCA+ridge; REMAINDER gets global augmentation.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    # Resolve block indices
    local_blocks = {}
    remainder_idx = []
    for i, a in enumerate(v_list):
        bname = block_map.get(a, "REMAINDER")
        if bname != "REMAINDER":
            local_blocks.setdefault(bname, []).append(i)
        else:
            remainder_idx.append(i)

    # Stage 1: global pooled+FE
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Global C1 augmentation setup
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

    # Local PCA+ridge models per block
    local_models = {}
    for bname, bidx in local_blocks.items():
        N_b = len(bidx)
        if N_b < 5:
            continue
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        K_b = min(4, max(2, N_b // 5))

        # PCA
        C = dm_b.T @ dm_b / dm_b.shape[0]
        eigvals, eigvecs = np.linalg.eigh(C)
        idx_sort = np.argsort(eigvals)[::-1]
        U_pca = eigvecs[:, idx_sort][:, :K_b]

        # Ridge VAR on factors
        fac = dm_b @ U_pca
        X, Y = fac[:-1], fac[1:]
        try:
            A_pca = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
        except np.linalg.LinAlgError:
            A_pca = np.eye(K_b) * 0.5

        local_models[bname] = {"U_pca": U_pca, "A_pca": A_pca, "om_b": om_b,
                                "K_b": K_b, "bidx": bidx, "N_b": N_b}

    # Rolling test
    preds_global, preds_mixture, actuals = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_pool = bar_y + rho * (prev - bar_y)

        # Global augmented
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        y_global = y_pool + resid_pred
        preds_global.append(y_global.copy())

        # Mixture: global for REMAINDER, local PCA+ridge for local blocks
        y_mix = y_global.copy()
        prev_resid = prev - (bar_y + rho * (np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)) if otr.shape[0] >= 2 else np.zeros(N)
        for bname, lm in local_models.items():
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            f_pca = lm["U_pca"].T @ prev_resid_b
            local_pred = lm["U_pca"] @ (lm["A_pca"] @ f_pca) + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_mix[bidx] = y_pool[bidx] + local_pred
            else:
                y_mix[bidx] = y_pool[bidx]
        preds_mixture.append(y_mix)
        actuals.append(obs)

        # Kalman update (global)
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
            F_r = _clip_sr(mf_r2.metadata["Atilde"][:k2, :k2].real.copy())
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2
        residuals = residuals_new

        # Re-estimate local models
        for bname, lm in local_models.items():
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]; N_b = lm["N_b"]
            C2 = dm_b.T @ dm_b / dm_b.shape[0]
            ev2, ec2 = np.linalg.eigh(C2)
            lm["U_pca"] = ec2[:, np.argsort(ev2)[::-1]][:, :K_b]
            fac = dm_b @ lm["U_pca"]
            X, Y = fac[:-1], fac[1:]
            try:
                lm["A_pca"] = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
            except:
                pass
            lm["om_b"] = om_b

    if not actuals:
        return None

    act_a = np.array(actuals)
    glob_a = np.array(preds_global)
    mix_a = np.array(preds_mixture)

    if not np.all(np.isfinite(glob_a)) or not np.all(np.isfinite(mix_a)):
        return None

    return {
        "global_r2": float(oos_r_squared(glob_a.ravel(), act_a.ravel())),
        "mixture_r2": float(oos_r_squared(mix_a.ravel(), act_a.ravel())),
    }


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def run_panel(panel, block_map, panel_name):
    """Run global vs mixture on a panel across all test years."""
    print(f"\n  {panel_name}: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # Count block sizes
    actors = list(panel.columns)
    from collections import Counter
    block_counts = Counter(block_map.get(a, "REMAINDER") for a in actors)
    for bname, cnt in sorted(block_counts.items()):
        print(f"    {bname}: N={cnt}")

    global_r2s, mixture_r2s = [], []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_window_global_and_mixture(panel, ty, block_map)
        if res:
            global_r2s.append(res["global_r2"])
            mixture_r2s.append(res["mixture_r2"])
            print(f"    W{ty}: Global={res['global_r2']:.4f}  Mixture={res['mixture_r2']:.4f}"
                  f"  Δ={res['mixture_r2']-res['global_r2']:+.4f}  ({time.time()-t0:.1f}s)")
        else:
            global_r2s.append(np.nan)
            mixture_r2s.append(np.nan)
            print(f"    W{ty}: FAILED")

    # Summary
    g_mean = np.nanmean(global_r2s)
    m_mean = np.nanmean(mixture_r2s)
    deltas = [m - g for m, g in zip(mixture_r2s, global_r2s)
              if np.isfinite(m) and np.isfinite(g)]

    if deltas:
        mean_d = np.mean(deltas)
        t_stat, p_val = paired_t_test(deltas)
        ci = bootstrap_ci(deltas)
        wins = sum(1 for d in deltas if d > 0)
        ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if np.isfinite(ci[0]) else "N/A"
        t_s = f"{t_stat:.2f}" if np.isfinite(t_stat) else "N/A"
        p_s = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
    else:
        mean_d, t_s, p_s, ci_s, wins = np.nan, "N/A", "N/A", "N/A", 0

    print(f"\n  {panel_name} SUMMARY:")
    print(f"    Global:  R² = {g_mean:.4f}")
    print(f"    Mixture: R² = {m_mean:.4f}")
    print(f"    Δ = {mean_d:+.4f}  t = {t_s}  p = {p_s}  CI {ci_s}  W = {wins}/{len(deltas)}")

    if np.isfinite(ci[0]) and ci[0] > 0:
        print(f"    → REPLICATES ★")
    else:
        print(f"    → Does not replicate")

    return {
        "panel": panel_name,
        "global_mean": g_mean, "mixture_mean": m_mean,
        "delta": mean_d, "ci_lo": ci[0] if np.isfinite(ci[0]) else np.nan,
        "ci_hi": ci[1] if np.isfinite(ci[1]) else np.nan,
        "wins": wins, "total": len(deltas),
    }


def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.4b — CROSS-PANEL VALIDATION")
    print("  Mixture PCA+ridge vs Global always-on")
    print("=" * 80)

    # ── 146-firm CapEx/Revenue ──
    # Homogeneous panel — no natural blocks.
    # Use train-only k-means K=2 on first training window's PCA loadings
    # to create a data-driven 2-block partition.
    p146 = load_146_panel()

    # For 146-firm: split into halves by first-PC loading sign (train-only proxy)
    # Use a simpler pre-specified split: first half vs second half alphabetically
    # (this is a null-partition; mixture should NOT help much)
    actors_146 = list(p146.columns)
    mid = len(actors_146) // 2
    block_map_146 = {}
    for i, a in enumerate(actors_146):
        block_map_146[a] = "HALF_A" if i < mid else "HALF_B"

    res_146 = run_panel(p146, block_map_146, "146-firm CapEx/Revenue (alphabetical halves)")

    # ── 270-actor multi-ratio ──
    # Natural 2-block partition: capexrev vs revass
    p270 = load_270_panel()
    block_map_270 = {}
    for a in p270.columns:
        if "_capexrev" in a:
            block_map_270[a] = "CAPEXREV"
        elif "_revass" in a:
            block_map_270[a] = "REVASS"
        else:
            block_map_270[a] = "REMAINDER"

    res_270 = run_panel(p270, block_map_270, "270-actor multi-ratio (capexrev vs revass)")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  CROSS-PANEL SUMMARY")
    print("=" * 80)
    print(f"\n  {'Panel':<45s} {'Global':>7s} {'Mixture':>8s} {'Δ':>8s} {'Replicates?':>12s}")
    print(f"  {'-'*82}")
    for res in [res_146, res_270]:
        repl = "YES ★" if np.isfinite(res["ci_lo"]) and res["ci_lo"] > 0 else "no"
        print(f"  {res['panel'][:44]:<45s} {res['global_mean']:7.4f} {res['mixture_mean']:8.4f}"
              f" {res['delta']:+8.4f} {repl:>12s}")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res_146, res_270]).to_parquet(
        METRICS_DIR / "iter6_4b_xpanel.parquet", index=False)
    print(f"\n  Saved: iter6_4b_xpanel.parquet")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
