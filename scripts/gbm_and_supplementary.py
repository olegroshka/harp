#!/usr/bin/env python
"""
Supplementary experiments for referee response:
  1. Two-block partition: Tech/Health local + Remainder global (C-4)
  2. Non-linear baseline: GBM on Stage 1 residuals (C-3)
  3. MAE robustness check across all architectures (B-9)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_supplementary.py
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
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3


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
#  Experiment 1: Two-block partition (Tech/Health + Remainder)
# ══════════════════════════════════════════════════════════════════════

def run_two_block_experiment(panel, meta):
    """Two-block partition: Tech/Health local, everything else as remainder."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: TWO-BLOCK PARTITION (Tech/Health + Remainder)")
    print("=" * 80)

    actors = list(panel.columns)
    blocks_2 = {
        "MERGED_tech_health": [a for a in actors
                               if meta.get(a, {}).get("sector") in ("technology", "healthcare")],
    }
    local_set = set(blocks_2["MERGED_tech_health"])
    blocks_2["REMAINDER"] = [a for a in actors if a not in local_set]

    g1_r2s, m2_2b_r2s = [], []

    for ty in TEST_YEARS:
        prep = _prepare_window(panel, ty)
        if prep is None:
            g1_r2s.append(np.nan)
            m2_2b_r2s.append(np.nan)
            continue

        ad, otr, tq, N, v = prep
        v_list = list(v)

        # Block indices
        actor_block = {}
        for bname, bactors in blocks_2.items():
            for a in bactors:
                if a in v_list:
                    actor_block[a] = bname
        for a in v_list:
            if a not in actor_block:
                actor_block[a] = "REMAINDER"

        block_indices = {}
        for bname in blocks_2:
            block_indices[bname] = [v_list.index(a) for a in v_list if actor_block.get(a) == bname]

        # Stage 1
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        # Local model for tech/health only
        th_idx = block_indices["MERGED_tech_health"]
        block_resids = residuals[:, th_idx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        K_b = 4
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        preds_g1, preds_m2_2b, actuals = [], [], []
        prev = np.nan_to_num(otr[-1], nan=0.5)

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0:
                continue
            obs = qv[0]

            y_pool = bar_y + rho * (prev - bar_y)

            # G1: just pooled (no global augmentation for simplicity — use stored values)
            preds_g1.append(y_pool.copy())

            # M2-2block: local PCA+ridge for tech/health, pooled for rest
            y_m2_2b = y_pool.copy()
            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)
            prev_resid_b = prev_resid[th_idx] - om_b.ravel()
            f_pca = U_pca.T @ prev_resid_b
            local_pred = U_pca @ (A_pca @ f_pca) + om_b.ravel()
            if np.all(np.isfinite(local_pred)):
                y_m2_2b[th_idx] = y_pool[th_idx] + local_pred
            preds_m2_2b.append(y_m2_2b)

            actuals.append(obs)
            prev = obs

            # Rolling update
            otr = np.vstack([otr, qv])
            rho, bar_y = estimate_pooled_ar1(otr)
            residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
            block_resids = residuals_new[:, th_idx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        if actuals:
            act_a = np.array(actuals)
            g1_r2 = float(oos_r_squared(np.array(preds_g1).ravel(), act_a.ravel()))
            m2_2b_r2 = float(oos_r_squared(np.array(preds_m2_2b).ravel(), act_a.ravel()))
            g1_r2s.append(g1_r2)
            m2_2b_r2s.append(m2_2b_r2)
            print(f"  W{ty}: G0={g1_r2:.4f}  M2_2block={m2_2b_r2:.4f}  Δ={m2_2b_r2-g1_r2:+.4f}")
        else:
            g1_r2s.append(np.nan)
            m2_2b_r2s.append(np.nan)

    # Note: g1_r2s here is actually G0 (pooled-only) since we don't have global augmentation
    # We need to compare against the stored G1 values for the proper delta
    # Load stored results for proper comparison
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values

    # The 2-block M2 needs to be compared against G1 properly
    # Since our g1_r2s is actually pooled-only (no augmentation), let's re-run with augmentation
    # Actually, for the two-block experiment, the proper comparison is:
    # - Does 2-block (TH local + remainder global aug) achieve close to 3-block M2?
    # Let me just report the delta vs stored G1
    m2_2b = np.array(m2_2b_r2s)
    deltas_vs_g1 = m2_2b - g1_stored
    deltas_vs_m2 = m2_2b - m2_stored

    print(f"\n  Two-block M2 (TH only): mean R² = {np.nanmean(m2_2b):.4f}")
    print(f"  Three-block M2:         mean R² = {np.nanmean(m2_stored):.4f}")

    # But wait — the two-block version uses pooled-only for remainder (no global aug)
    # For a fair comparison, remainder should get global augmentation
    # Let me flag this and report what we have
    print(f"\n  NOTE: Two-block version uses pooled-only for non-TH actors")
    print(f"  (no global augmentation for remainder)")
    print(f"  Δ vs G0 (pooled-only):")
    g0_stored = df_4b[df_4b["architecture"] == "G0"].sort_values("year")["full_r2"].values
    deltas_vs_g0 = m2_2b - g0_stored
    mean_d = np.nanmean(deltas_vs_g0)
    wins = int(np.sum(deltas_vs_g0 > 0))
    t_s, p_s = paired_t_test(deltas_vs_g0)
    ci = bootstrap_ci(deltas_vs_g0)
    print(f"  Δ = {mean_d:+.4f}  t={t_s:.2f}  p={p_s:.4f}"
          f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  W={wins}/10")

    return m2_2b_r2s


# ══════════════════════════════════════════════════════════════════════
#  Experiment 1b: Two-block with global augmentation for remainder
# ══════════════════════════════════════════════════════════════════════

def run_two_block_with_global_aug(panel, meta):
    """Two-block: Tech/Health local PCA+ridge, remainder gets global augmentation.

    This is the proper comparison: same as 3-block M2 but with only 1 local block.
    Uses the single-block diagnostic approach from Appendix G.
    """
    from harp.spectral.dmd import ExactDMDDecomposer

    def sph_r(dm, U):
        N = U.shape[0]
        res = dm - (dm @ U) @ U.T
        return np.eye(N) * max(np.mean(res ** 2), 1e-8)

    def _clip_sr(F, max_sr=0.99):
        ev = np.linalg.eigvals(F)
        mx = float(np.max(np.abs(ev)))
        return F * (max_sr / mx) if mx > max_sr else F

    def dmd_full(dm, k_svd=K_MAX):
        N = dm.shape[1]
        if dm.shape[0] < 3:
            return None
        try:
            return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
        except Exception:
            return None

    print("\n" + "=" * 80)
    print("  EXPERIMENT 1b: TWO-BLOCK (TH local + global aug remainder)")
    print("=" * 80)

    actors = list(panel.columns)
    th_actors = set(a for a in actors
                    if meta.get(a, {}).get("sector") in ("technology", "healthcare"))

    m2_2b_r2s = []

    for ty in TEST_YEARS:
        prep = _prepare_window(panel, ty)
        if prep is None:
            m2_2b_r2s.append(np.nan)
            continue

        ad, otr, tq, N, v = prep
        v_list = list(v)
        th_idx = [i for i, a in enumerate(v_list) if a in th_actors]

        # Stage 1
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        # Global augmentation
        om_r = ewm_demean(residuals, 12)
        dm_r = residuals - om_r
        mf_r = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r is None:
            m2_2b_r2s.append(np.nan)
            continue

        ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
        A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
        F_r = _clip_sr(A_r)
        U_r = mf_r.metadata["U"][:, :ka]
        R_r = sph_r(dm_r, U_r)
        a_r, P_r = np.zeros(ka), np.eye(ka)
        Q_r = np.eye(ka) * Q_INIT_SCALE

        # Local model for tech/health
        block_resids = residuals[:, th_idx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        K_b = 4
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        preds, actuals = [], []
        prev = np.nan_to_num(otr[-1], nan=0.5)

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0:
                continue
            obs = qv[0]

            y_pool = bar_y + rho * (prev - bar_y)

            # Global augmented prediction
            ap_r = F_r @ a_r
            Pp_r = F_r @ P_r @ F_r.T + Q_r
            resid_pred_global = U_r @ ap_r + om_r.ravel()
            if not np.all(np.isfinite(resid_pred_global)):
                resid_pred_global = np.zeros(N)
            y_pred = y_pool + resid_pred_global  # start with global aug for all

            # Override tech/health with local
            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)
            prev_resid_b = prev_resid[th_idx] - om_b.ravel()
            f_pca = U_pca.T @ prev_resid_b
            local_pred = U_pca @ (A_pca @ f_pca) + om_b.ravel()
            if np.all(np.isfinite(local_pred)):
                y_pred[th_idx] = y_pool[th_idx] + local_pred

            preds.append(y_pred)
            actuals.append(obs)

            # Kalman update
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

            # Rolling update
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

            # Re-estimate local
            residuals = residuals_new
            block_resids = residuals[:, th_idx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)

        if actuals:
            act_a = np.array(actuals)
            r2 = float(oos_r_squared(np.array(preds).ravel(), act_a.ravel()))
            m2_2b_r2s.append(r2)
            print(f"  W{ty}: R2={r2:.4f}")
        else:
            m2_2b_r2s.append(np.nan)

    # Compare against stored
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values
    m2_2b = np.array(m2_2b_r2s)

    deltas = m2_2b - g1_stored
    mean_d = np.nanmean(deltas)
    t_s, p_s = paired_t_test(deltas)
    ci = bootstrap_ci(deltas)
    wins = int(np.sum(deltas > 0))

    print(f"\n  Two-block (TH local + global aug): mean R² = {np.nanmean(m2_2b):.4f}")
    print(f"  Three-block M2:                    mean R² = {np.nanmean(m2_stored):.4f}")
    print(f"  G1 (global):                       mean R² = {np.nanmean(g1_stored):.4f}")
    print(f"\n  Δ vs G1: {mean_d:+.4f}  t={t_s:.2f}  p={p_s:.4f}"
          f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  W={wins}/10")

    # Fraction of 3-block gain retained
    m2_gain = np.nanmean(m2_stored) - np.nanmean(g1_stored)
    frac = mean_d / m2_gain if m2_gain > 0 else np.nan
    print(f"  Fraction of 3-block M2 gain retained: {frac:.1%}")

    return m2_2b_r2s


# ══════════════════════════════════════════════════════════════════════
#  Experiment 2: Non-linear baseline (GBM on Stage 1 residuals)
# ══════════════════════════════════════════════════════════════════════

def run_nonlinear_baseline(panel, meta):
    """Gradient boosting on Stage 1 residuals, same train/test protocol."""
    from sklearn.ensemble import GradientBoostingRegressor

    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: NON-LINEAR BASELINE (GBM on Stage 1 residuals)")
    print("=" * 80)

    gbm_r2s = []

    for ty in TEST_YEARS:
        prep = _prepare_window(panel, ty)
        if prep is None:
            gbm_r2s.append(np.nan)
            continue

        ad, otr, tq, N, v = prep
        v_list = list(v)

        rho, bar_y = estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        preds_gbm, actuals = [], []
        prev = np.nan_to_num(otr[-1], nan=0.5)

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0:
                continue
            obs = qv[0]

            y_pool = bar_y + rho * (prev - bar_y)

            # GBM: for each actor, predict r_{i,t+1} from r_t (all actors)
            # Train on residuals: X = residuals[:-1], Y = residuals[1:]
            X_train = residuals[:-1]  # (T-2, N)
            Y_train = residuals[1:]   # (T-2, N)

            if X_train.shape[0] < 4:
                preds_gbm.append(y_pool.copy())
                actuals.append(obs)
                prev = obs
                otr = np.vstack([otr, qv])
                rho, bar_y = estimate_pooled_ar1(otr)
                residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
                continue

            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)

            # Fit one GBM per actor (vectorised would be better but this is clear)
            y_gbm = y_pool.copy()
            for i in range(N):
                try:
                    gbm = GradientBoostingRegressor(
                        n_estimators=50, max_depth=2, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                    gbm.fit(X_train, Y_train[:, i])
                    pred_resid = gbm.predict(prev_resid.reshape(1, -1))[0]
                    if np.isfinite(pred_resid):
                        y_gbm[i] = y_pool[i] + pred_resid
                except Exception:
                    pass

            preds_gbm.append(y_gbm)
            actuals.append(obs)
            prev = obs

            otr = np.vstack([otr, qv])
            rho, bar_y = estimate_pooled_ar1(otr)
            residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        if actuals:
            act_a = np.array(actuals)
            r2 = float(oos_r_squared(np.array(preds_gbm).ravel(), act_a.ravel()))
            gbm_r2s.append(r2)
            print(f"  W{ty}: R2={r2:.4f}")
        else:
            gbm_r2s.append(np.nan)

    # Compare
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values

    gbm_arr = np.array(gbm_r2s)
    deltas_g1 = gbm_arr - g1_stored
    mean_d = np.nanmean(deltas_g1)
    t_s, p_s = paired_t_test(deltas_g1)
    wins = int(np.sum(deltas_g1 > 0))

    print(f"\n  GBM (Stage 1 residuals): mean R² = {np.nanmean(gbm_arr):.4f}")
    print(f"  G1 (global linear):      mean R² = {np.nanmean(g1_stored):.4f}")
    print(f"  M2 (mixture PCA+ridge):  mean R² = {np.nanmean(m2_stored):.4f}")
    print(f"\n  GBM Δ vs G1: {mean_d:+.4f}  t={t_s:.2f}  p={p_s:.4f}  W={wins}/10")
    print(f"  GBM Δ vs M2: {np.nanmean(gbm_arr) - np.nanmean(m2_stored):+.4f}")

    if np.nanmean(gbm_arr) <= np.nanmean(m2_stored):
        print(f"  → GBM does not beat M2: ceiling is architectural, not a linear-method artifact")
    else:
        print(f"  → GBM exceeds M2: non-linear dynamics add value beyond architecture")

    return gbm_r2s


# ══════════════════════════════════════════════════════════════════════
#  Experiment 4: Block-specific GBM (non-linear analog of M2)
# ══════════════════════════════════════════════════════════════════════

def run_block_specific_gbm(panel, meta):
    """GBM estimated separately per block — the non-linear analog of M2.

    For local blocks (Diversified, Macro/Inst, Tech/Health): train a per-actor
    GBM using only within-block residuals as features.
    For the remainder block: use global per-actor GBM (same as Experiment 2).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    print("\n" + "=" * 80)
    print("  EXPERIMENT 4: BLOCK-SPECIFIC GBM (non-linear analog of M2)")
    print("=" * 80)

    actors = list(panel.columns)
    blocks = {
        "SEC_diversified": [a for a in actors
                            if meta.get(a, {}).get("sector") == "diversified"],
        "LAYER_macro_inst": [a for a in actors
                             if meta.get(a, {}).get("layer", -1) in (0, 1)],
        "MERGED_tech_health": [a for a in actors
                               if meta.get(a, {}).get("sector")
                               in ("technology", "healthcare")],
    }
    local_set = set()
    for v in blocks.values():
        local_set.update(v)
    blocks["REMAINDER"] = [a for a in actors if a not in local_set]
    local_block_names = ["SEC_diversified", "LAYER_macro_inst", "MERGED_tech_health"]

    for bname, bactors in blocks.items():
        print(f"  {bname}: {len(bactors)} actors")

    bgbm_r2s = []
    ggbm_r2s = []  # global GBM for comparison

    for ty in TEST_YEARS:
        prep = _prepare_window(panel, ty)
        if prep is None:
            bgbm_r2s.append(np.nan)
            ggbm_r2s.append(np.nan)
            continue

        ad, otr, tq, N, v = prep
        v_list = list(v)

        # Build actor-to-block index mapping
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
            block_indices[bname] = [v_list.index(a) for a in v_list
                                    if actor_block.get(a) == bname]

        rho, bar_y = estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        preds_bgbm, preds_ggbm, actuals = [], [], []
        prev = np.nan_to_num(otr[-1], nan=0.5)

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0:
                continue
            obs = qv[0]

            y_pool = bar_y + rho * (prev - bar_y)

            X_train = residuals[:-1]  # (T-2, N)
            Y_train = residuals[1:]   # (T-2, N)

            if X_train.shape[0] < 4:
                preds_bgbm.append(y_pool.copy())
                preds_ggbm.append(y_pool.copy())
                actuals.append(obs)
                prev = obs
                otr = np.vstack([otr, qv])
                rho, bar_y = estimate_pooled_ar1(otr)
                residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
                continue

            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1],
                              nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)

            # ── Block-specific GBM ──
            y_bgbm = y_pool.copy()
            for bname in list(blocks.keys()):
                bidx = block_indices[bname]
                if len(bidx) < 2:
                    continue

                if bname in local_block_names:
                    # Local: use only within-block residuals as features
                    X_b = X_train[:, bidx]
                    Y_b = Y_train[:, bidx]
                    prev_r_b = prev_resid[bidx]
                else:
                    # Remainder: use all residuals (same as global per-actor)
                    X_b = X_train
                    Y_b = Y_train[:, bidx]
                    prev_r_b = prev_resid

                for j, gi in enumerate(bidx):
                    try:
                        gbm = GradientBoostingRegressor(
                            n_estimators=50, max_depth=2, learning_rate=0.1,
                            subsample=0.8, random_state=42
                        )
                        gbm.fit(X_b, Y_b[:, j])
                        pred_resid = gbm.predict(
                            prev_r_b.reshape(1, -1))[0]
                        if np.isfinite(pred_resid):
                            y_bgbm[gi] = y_pool[gi] + pred_resid
                    except Exception:
                        pass

            # ── Global GBM (for comparison, same as Experiment 2) ──
            y_ggbm = y_pool.copy()
            for i in range(N):
                try:
                    gbm = GradientBoostingRegressor(
                        n_estimators=50, max_depth=2, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                    gbm.fit(X_train, Y_train[:, i])
                    pred_resid = gbm.predict(prev_resid.reshape(1, -1))[0]
                    if np.isfinite(pred_resid):
                        y_ggbm[i] = y_pool[i] + pred_resid
                except Exception:
                    pass

            preds_bgbm.append(y_bgbm)
            preds_ggbm.append(y_ggbm)
            actuals.append(obs)
            prev = obs

            otr = np.vstack([otr, qv])
            rho, bar_y = estimate_pooled_ar1(otr)
            residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        if actuals:
            act_a = np.array(actuals)
            r2_b = float(oos_r_squared(
                np.array(preds_bgbm).ravel(), act_a.ravel()))
            r2_g = float(oos_r_squared(
                np.array(preds_ggbm).ravel(), act_a.ravel()))
            bgbm_r2s.append(r2_b)
            ggbm_r2s.append(r2_g)
            print(f"  W{ty}: block-GBM={r2_b:.4f}  global-GBM={r2_g:.4f}"
                  f"  delta={r2_b - r2_g:+.4f}")
        else:
            bgbm_r2s.append(np.nan)
            ggbm_r2s.append(np.nan)

    # ── Compare ──
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")[
        "full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")[
        "full_r2"].values

    bgbm_arr = np.array(bgbm_r2s)
    ggbm_arr = np.array(ggbm_r2s)

    print(f"\n  {'Model':<30s} {'Mean R²':>8s} {'Δ vs G1':>9s} {'Δ vs M2':>9s}")
    print(f"  {'-' * 60}")
    for label, arr in [("GBM (global per-actor)", ggbm_arr),
                       ("GBM (block-specific)", bgbm_arr),
                       ("G1 (global linear)", g1_stored),
                       ("M2 (mixture PCA+ridge)", m2_stored)]:
        m = np.nanmean(arr)
        print(f"  {label:<30s} {m:8.4f} {m - np.nanmean(g1_stored):+9.4f}"
              f" {m - np.nanmean(m2_stored):+9.4f}")

    # Paired tests
    print(f"\n  Paired comparisons:")

    d1 = bgbm_arr - g1_stored
    t1, p1 = paired_t_test(d1)
    ci1 = bootstrap_ci(d1)
    w1 = int(np.nansum(d1 > 0))
    print(f"  Block-GBM vs G1:         delta={np.nanmean(d1):+.4f}"
          f"  t={t1:.2f}  p={p1:.4f}"
          f"  CI [{ci1[0]:+.4f}, {ci1[1]:+.4f}]  W={w1}/10")

    d2 = bgbm_arr - m2_stored
    t2, p2 = paired_t_test(d2)
    ci2 = bootstrap_ci(d2)
    w2 = int(np.nansum(d2 > 0))
    print(f"  Block-GBM vs M2:         delta={np.nanmean(d2):+.4f}"
          f"  t={t2:.2f}  p={p2:.4f}"
          f"  CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}]  W={w2}/10")

    d3 = bgbm_arr - ggbm_arr
    t3, p3 = paired_t_test(d3)
    ci3 = bootstrap_ci(d3)
    w3 = int(np.nansum(d3 > 0))
    print(f"  Block-GBM vs Global-GBM: delta={np.nanmean(d3):+.4f}"
          f"  t={t3:.2f}  p={p3:.4f}"
          f"  CI [{ci3[0]:+.4f}, {ci3[1]:+.4f}]  W={w3}/10")

    if np.nanmean(bgbm_arr) >= np.nanmean(m2_stored):
        print(f"\n  → Block-specific GBM ≥ M2: gain is purely architectural"
              f" (block decomposition), method-agnostic")
    else:
        print(f"\n  → Block-specific GBM < M2: PCA+ridge captures structure"
              f" that per-actor GBM within blocks cannot")
        if np.nanmean(bgbm_arr) > np.nanmean(ggbm_arr):
            print(f"    But block-GBM > global-GBM: block decomposition"
                  f" still helps non-linear methods")

    return bgbm_r2s, ggbm_r2s


# ══════════════════════════════════════════════════════════════════════
#  Experiment 3: MAE robustness
# ══════════════════════════════════════════════════════════════════════

def run_mae_robustness():
    """Report MAE alongside R² for all architectures."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: MAE ROBUSTNESS CHECK")
    print("=" * 80)

    # Re-run the main experiment but also collect MAE
    # Actually, we can just load stored per-window predictions...
    # But the main script only saves R². We need to re-run.
    # For efficiency, let's load the parquet and recompute from stored R²
    # Actually R² and MAE are different metrics; we can't derive MAE from R²

    # Instead, let's re-run the main experiment and compute both metrics
    # But that's the same as run_iter6_4b.py. Let's just import and modify.
    # For now, let's run a simplified version that computes MAE per window.

    panel, meta = load_panel_and_meta()

    actors = list(panel.columns)
    blocks = {
        "SEC_diversified": [a for a in actors if meta.get(a, {}).get("sector") == "diversified"],
        "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
        "MERGED_tech_health": [a for a in actors
                               if meta.get(a, {}).get("sector") in ("technology", "healthcare")],
    }
    local_set = set()
    for v in blocks.values():
        local_set.update(v)
    blocks["REMAINDER"] = [a for a in actors if a not in local_set]

    # We need predictions to compute MAE. Re-import run_window_all_architectures
    # and modify to also return raw predictions.
    # Simpler: just load iter6_4b.parquet which has per-window R² and compute
    # paired MAE from the running script. But that only stores R², not predictions.

    # Let's do it properly: import and run
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import table5_architectures as r4b

    archs = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]
    mae_results = {a: [] for a in archs}
    r2_results = {a: [] for a in archs}

    for ty in TEST_YEARS:
        prep = r4b._prepare_window(panel, ty)
        if prep is None:
            for a in archs:
                mae_results[a].append(np.nan)
                r2_results[a].append(np.nan)
            continue

        result = r4b.run_window_all_architectures(panel, ty, blocks)
        if result is None:
            for a in archs:
                mae_results[a].append(np.nan)
                r2_results[a].append(np.nan)
            continue

        for a in archs:
            r2_results[a].append(result[a]["full"] if result[a]["full"] is not None else np.nan)

    # For MAE we need the actual predictions. The current runner only returns R².
    # Let's modify the approach: compute MAE from the predictions directly.
    # Actually, the run_window function computes R² internally and discards predictions.
    # We need to modify it to also return MAE.

    # Quick approach: patch to return MAE alongside R²
    # For now, report what we can from the stored R² (which we just recomputed)
    # and note that MAE requires modifying the runner.

    # Actually, let's just add MAE computation to the runner inline
    print("  Re-running all architectures to compute MAE...")

    # Modify to collect predictions
    all_preds = {a: [] for a in archs}
    all_actuals = []

    for ty in TEST_YEARS:
        prep = r4b._prepare_window(panel, ty)
        if prep is None:
            continue

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

        rho, bar_y = r4b.estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        rho_b_vec, mean_b_vec = r4b.estimate_block_ar1(otr, v_list, actor_block)
        residuals_b = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

        om_r = r4b.ewm_demean(residuals, 12)
        dm_r = residuals - om_r
        mf_r = r4b.dmd_full(dm_r, k_svd=K_MAX)
        if mf_r is None:
            continue

        ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
        A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
        F_r = r4b._clip_sr(A_r)
        U_r = mf_r.metadata["U"][:, :ka]
        R_r = r4b.sph_r(dm_r, U_r)
        a_r, P_r = np.zeros(ka), np.eye(ka)
        Q_r = np.eye(ka) * Q_INIT_SCALE

        local_models = {}
        local_models_b = {}
        for bname in local_block_names:
            bidx = block_indices[bname]
            if len(bidx) < 5:
                local_models[bname] = None
                local_models_b[bname] = None
                continue
            block_resids = residuals[:, bidx]
            om_b = r4b.ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            N_b = len(bidx)
            alpha = r4b.select_ridge_alpha(dm_b, N_b)
            C_ridge = r4b.fit_local_ridge(dm_b, alpha)
            K_b = min(4, max(2, N_b // 5))
            U_pca, A_pca = r4b.fit_local_pca_ridge(dm_b, K_b)
            local_models[bname] = {
                "C_ridge": C_ridge, "U_pca": U_pca, "A_pca": A_pca,
                "om_b": om_b, "alpha": alpha, "K_b": K_b, "bidx": bidx, "N_b": N_b,
            }
            # Block-specific residuals version
            block_resids_b = residuals_b[:, bidx]
            om_bb = r4b.ewm_demean(block_resids_b, 12)
            dm_bb = block_resids_b - om_bb
            U_pca_b, A_pca_b = r4b.fit_local_pca_ridge(dm_bb, K_b)
            local_models_b[bname] = {
                "U_pca": U_pca_b, "A_pca": A_pca_b,
                "om_b": om_bb, "K_b": K_b, "bidx": bidx, "N_b": N_b,
            }

        prev = np.nan_to_num(otr[-1], nan=0.5)
        window_preds = {a: [] for a in archs}
        window_actuals = []

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
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

            window_preds["G0"].append(y_pool.copy())
            window_preds["BA"].append(y_pool_b.copy())
            window_preds["G1"].append(y_global_aug.copy())

            y_s1 = y_global_aug.copy()
            for bname in local_block_names:
                bidx = block_indices[bname]
                y_s1[bidx] = y_pool[bidx]
            window_preds["S1"].append(y_s1)

            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)

            y_m1 = y_global_aug.copy()
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
            window_preds["M1"].append(y_m1)

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
            window_preds["M2"].append(y_m2)

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
            window_preds["BA_M2"].append(y_bam2)

            window_preds["ENS"].append(0.5 * y_global_aug + 0.5 * y_pool_b)

            window_actuals.append(obs)

            # Kalman + rolling updates (abbreviated)
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
            rho, bar_y = r4b.estimate_pooled_ar1(otr)
            residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
            om_r = r4b.ewm_demean(residuals_new, 12)
            dm_r = residuals_new - om_r
            mf_r2 = r4b.dmd_full(dm_r, k_svd=K_MAX)
            if mf_r2 is not None:
                k2 = min(K_DEFAULT, mf_r2.basis.shape[0] - 2, mf_r2.K)
                A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
                F_r = r4b._clip_sr(A_r2)
                U_r2 = mf_r2.metadata["U"][:, :k2]
                a_r = U_r2.T @ (actual_resid - om_r.ravel())
                P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
                R_r = r4b.sph_r(dm_r, U_r2); U_r = U_r2; ka = k2

            rho_b_vec, mean_b_vec = r4b.estimate_block_ar1(otr, v_list, actor_block)
            residuals_b_new = otr[1:] - (mean_b_vec + rho_b_vec * (otr[:-1] - mean_b_vec))

            residuals = residuals_new
            for bname in local_block_names:
                lm = local_models.get(bname)
                if lm is None:
                    continue
                bidx = lm["bidx"]
                block_resids = residuals[:, bidx]
                om_b = r4b.ewm_demean(block_resids, 12)
                dm_b = block_resids - om_b
                N_b = lm["N_b"]
                alpha = r4b.select_ridge_alpha(dm_b, N_b)
                lm["C_ridge"] = r4b.fit_local_ridge(dm_b, alpha)
                lm["om_b"] = om_b
                lm["alpha"] = alpha
                K_b = lm["K_b"]
                lm["U_pca"], lm["A_pca"] = r4b.fit_local_pca_ridge(dm_b, K_b)

            residuals_b = residuals_b_new
            for bname in local_block_names:
                lmb = local_models_b.get(bname)
                if lmb is None:
                    continue
                bidx = lmb["bidx"]
                block_resids_b = residuals_b[:, bidx]
                om_bb = r4b.ewm_demean(block_resids_b, 12)
                dm_bb = block_resids_b - om_bb
                K_b = lmb["K_b"]
                lmb["U_pca"], lmb["A_pca"] = r4b.fit_local_pca_ridge(dm_bb, K_b)
                lmb["om_b"] = om_bb

        if window_actuals:
            act_a = np.array(window_actuals)
            for a in archs:
                pa = np.array(window_preds[a])
                mae = float(np.mean(np.abs(pa - act_a)))
                mae_results[a].append(mae)
            all_actuals.extend(window_actuals)
            for a in archs:
                all_preds[a].extend(window_preds[a])

    # Report
    print(f"\n  {'Architecture':<25s} {'R²':>8s} {'MAE':>8s} {'ΔR² vs G1':>10s} {'ΔMAE vs G1':>11s}")
    print(f"  {'-'*65}")
    g1_maes = np.array(mae_results["G1"])
    for a in archs:
        r2s = np.array(r2_results[a])
        maes = np.array(mae_results[a])
        r2_mean = np.nanmean(r2s)
        mae_mean = np.nanmean(maes)

        df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
        g1_r2 = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
        r2_d = r2_mean - np.nanmean(g1_r2)
        mae_d = mae_mean - np.nanmean(g1_maes)  # negative = better

        print(f"  {a:<25s} {r2_mean:8.4f} {mae_mean:8.4f} {r2_d:+10.4f} {mae_d:+11.4f}")

    # Paired test: M2 vs G1 on MAE
    m2_maes = np.array(mae_results["M2"])
    mae_deltas = m2_maes - g1_maes  # negative = M2 better
    t_s, p_s = paired_t_test(-mae_deltas)  # flip sign for "improvement" direction
    wins = int(np.sum(mae_deltas < 0))
    print(f"\n  M2 vs G1 MAE: mean Δ = {np.nanmean(mae_deltas):+.4f}"
          f"  t={t_s:.2f}  p={p_s:.4f}  W(M2 lower)={wins}/10")

    return mae_results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 80)
    print("  SUPPLEMENTARY EXPERIMENTS FOR REFEREE RESPONSE")
    print("=" * 80)

    panel, meta = load_panel_and_meta()

    # C-4: Two-block partition
    run_two_block_with_global_aug(panel, meta)

    # C-3: Non-linear baseline
    run_nonlinear_baseline(panel, meta)

    # C-5: Block-specific GBM (non-linear analog of M2)
    run_block_specific_gbm(panel, meta)

    # B-9: MAE robustness
    run_mae_robustness()

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
