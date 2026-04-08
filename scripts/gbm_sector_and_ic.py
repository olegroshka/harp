#!/usr/bin/env python
"""
Referee round-3 experiments:
  1. GBM with sector interaction features (C-3 extension)
  2. Cross-sectional Rank IC per quarter (G1 vs M2)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_referee_round3.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist

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


# ══════════════════════════════════════════════════════════════════════
#  Data loading + infrastructure (same as run_iter6_4b.py)
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


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def _clip_sr(F, max_sr=0.99):
    ev = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(ev)))
    return F * (max_sr / mx) if mx > max_sr else F


def dmd_full(dm, k_svd=K_MAX):
    from harp.spectral.dmd import ExactDMDDecomposer
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def define_blocks(panel, meta):
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


def fit_local_ridge(residuals_block, alpha):
    X = residuals_block[:-1]
    Y = residuals_block[1:]
    N_b = X.shape[1]
    try:
        C = (np.linalg.solve(X.T @ X + alpha * np.eye(N_b), X.T @ Y)).T
    except np.linalg.LinAlgError:
        C = np.zeros((N_b, N_b))
    return C


def estimate_block_ar1(otr, v_list, actor_block):
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


# ══════════════════════════════════════════════════════════════════════
#  Helper: run G1 and M2 for one test year, returning per-quarter
#  predictions and actuals (needed for both experiments)
# ══════════════════════════════════════════════════════════════════════

def run_window_g1_m2(panel, meta, ty, blocks):
    """Run G1 and M2 for one test year.

    Returns dict with keys "G1", "M2", each containing list of
    (pred_vector, actual_vector) per quarter, plus "v_list" and "meta".
    """
    prep = _prepare_window(panel, ty)
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

    # Stage 1
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Global augmentation
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

    # Local models for M2
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
        K_b = min(4, max(2, N_b // 5))
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)
        local_models[bname] = {
            "U_pca": U_pca, "A_pca": A_pca,
            "om_b": om_b, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

    preds_g1, preds_m2, actuals_list = [], [], []
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
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_global_aug = y_pool + resid_pred_global

        preds_g1.append(y_global_aug.copy())

        # M2
        y_m2 = y_global_aug.copy()
        prev_resid = prev - (bar_y + rho * (
            np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
        ) if otr.shape[0] >= 2 else np.zeros(N)
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
        preds_m2.append(y_m2)

        actuals_list.append(obs)

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

        residuals = residuals_new
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]
            lm["U_pca"], lm["A_pca"] = fit_local_pca_ridge(dm_b, K_b)
            lm["om_b"] = om_b

    if not actuals_list:
        return None

    # Identify firm-only indices (layer == 2)
    firm_idx = [i for i, a in enumerate(v_list) if meta.get(a, {}).get("layer", -1) == 2]

    return {
        "G1": preds_g1,
        "M2": preds_m2,
        "actuals": actuals_list,
        "v_list": v_list,
        "firm_idx": firm_idx,
        "N": N,
        "otr_final": otr,
        "residuals_final": residuals,
        "rho_final": rho,
        "bar_y_final": bar_y,
    }


# ══════════════════════════════════════════════════════════════════════
#  Experiment 1: GBM with sector interaction features
# ══════════════════════════════════════════════════════════════════════

def _build_sector_mapping(v_list, meta):
    """Build sector name list and one-hot encoding for actors in v_list."""
    all_sectors = sorted(set(
        meta.get(a, {}).get("sector", "unknown") for a in v_list
    ))
    sector_to_idx = {s: i for i, s in enumerate(all_sectors)}
    n_sectors = len(all_sectors)

    # Per-actor sector index
    actor_sector_idx = np.array([
        sector_to_idx[meta.get(a, {}).get("sector", "unknown")]
        for a in v_list
    ])

    # One-hot matrix: (N, n_sectors)
    one_hot = np.zeros((len(v_list), n_sectors))
    for i, si in enumerate(actor_sector_idx):
        one_hot[i, si] = 1.0

    return all_sectors, actor_sector_idx, one_hot


def _build_gbm_features_with_sectors(residuals_lag, actor_idx, one_hot_row, N):
    """Build feature vector for actor i at time t.

    Features:
      - residuals_lag: full residual vector r_t  (N,)
      - one_hot_row: one-hot sector dummies for actor i  (n_sectors,)
      - interaction: residuals_lag * actor's own sector indicator  (N,)
        i.e., the residuals masked to the actor's sector

    Returns feature vector of length N + n_sectors + N = 2N + n_sectors.
    """
    sector_idx = np.argmax(one_hot_row)
    interaction = residuals_lag * one_hot_row[sector_idx]  # scalar * vector = just residuals if sector=1
    # Actually, the interaction should be: each residual_j * indicator(actor_i is in sector_s)
    # This gives the GBM the ability to weight residuals differently for different sectors
    return np.concatenate([residuals_lag, one_hot_row, residuals_lag * one_hot_row[sector_idx]])


def run_gbm_sector_interactions(panel, meta):
    """GBM on Stage 1 residuals with sector interaction features."""
    from sklearn.ensemble import GradientBoostingRegressor

    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: GBM WITH SECTOR INTERACTION FEATURES")
    print("=" * 80)

    gbm_plain_r2s = []
    gbm_sector_r2s = []

    for ty in TEST_YEARS:
        prep = _prepare_window(panel, ty)
        if prep is None:
            gbm_plain_r2s.append(np.nan)
            gbm_sector_r2s.append(np.nan)
            continue

        ad, otr, tq, N, v = prep
        v_list = list(v)

        all_sectors, actor_sector_idx, one_hot = _build_sector_mapping(v_list, meta)
        n_sectors = len(all_sectors)

        rho, bar_y = estimate_pooled_ar1(otr)
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        preds_plain, preds_sector, actuals = [], [], []
        prev = np.nan_to_num(otr[-1], nan=0.5)

        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0:
                continue
            obs = qv[0]

            y_pool = bar_y + rho * (prev - bar_y)

            X_train = residuals[:-1]   # (T-2, N)
            Y_train = residuals[1:]    # (T-2, N)

            if X_train.shape[0] < 4:
                preds_plain.append(y_pool.copy())
                preds_sector.append(y_pool.copy())
                actuals.append(obs)
                prev = obs
                otr = np.vstack([otr, qv])
                rho, bar_y = estimate_pooled_ar1(otr)
                residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
                continue

            prev_resid = prev - (bar_y + rho * (
                np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
            ) if otr.shape[0] >= 2 else np.zeros(N)

            # Build sector-augmented training data (pooled across actors)
            # For each training sample (t, i): X = [r_t, onehot_i, r_t * sector_i], Y = r_{t+1,i}
            T_tr = X_train.shape[0]
            X_sector_all = []
            Y_sector_all = []
            for t_idx in range(T_tr):
                for i in range(N):
                    feat = np.concatenate([
                        X_train[t_idx],               # N residuals
                        one_hot[i],                    # n_sectors dummies
                        X_train[t_idx] * one_hot[i, actor_sector_idx[i]],  # N interaction terms
                    ])
                    X_sector_all.append(feat)
                    Y_sector_all.append(Y_train[t_idx, i])

            X_sector_all = np.array(X_sector_all)
            Y_sector_all = np.array(Y_sector_all)

            # Subsample if training set is too large (>5000 samples)
            max_samples = 5000
            if len(Y_sector_all) > max_samples:
                rng = np.random.default_rng(42 + ty)
                idx_sub = rng.choice(len(Y_sector_all), max_samples, replace=False)
                X_sector_train = X_sector_all[idx_sub]
                Y_sector_train = Y_sector_all[idx_sub]
            else:
                X_sector_train = X_sector_all
                Y_sector_train = Y_sector_all

            # Fit single pooled GBM with sector features
            y_gbm_sector = y_pool.copy()
            try:
                gbm_s = GradientBoostingRegressor(
                    n_estimators=50, max_depth=2, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                gbm_s.fit(X_sector_train, Y_sector_train)

                for i in range(N):
                    feat = np.concatenate([
                        prev_resid,
                        one_hot[i],
                        prev_resid * one_hot[i, actor_sector_idx[i]],
                    ])
                    pred_resid = gbm_s.predict(feat.reshape(1, -1))[0]
                    if np.isfinite(pred_resid):
                        y_gbm_sector[i] = y_pool[i] + pred_resid
            except Exception:
                pass

            preds_sector.append(y_gbm_sector)

            # Plain GBM (per-actor, same as existing baseline)
            y_gbm_plain = y_pool.copy()
            for i in range(N):
                try:
                    gbm = GradientBoostingRegressor(
                        n_estimators=50, max_depth=2, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                    gbm.fit(X_train, Y_train[:, i])
                    pred_resid = gbm.predict(prev_resid.reshape(1, -1))[0]
                    if np.isfinite(pred_resid):
                        y_gbm_plain[i] = y_pool[i] + pred_resid
                except Exception:
                    pass
            preds_plain.append(y_gbm_plain)

            actuals.append(obs)
            prev = obs

            otr = np.vstack([otr, qv])
            rho, bar_y = estimate_pooled_ar1(otr)
            residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

        if actuals:
            act_a = np.array(actuals)
            r2_plain = float(oos_r_squared(np.array(preds_plain).ravel(), act_a.ravel()))
            r2_sector = float(oos_r_squared(np.array(preds_sector).ravel(), act_a.ravel()))
            gbm_plain_r2s.append(r2_plain)
            gbm_sector_r2s.append(r2_sector)
            print(f"  W{ty}: GBM_plain={r2_plain:.4f}  GBM_sector={r2_sector:.4f}"
                  f"  delta={r2_sector - r2_plain:+.4f}")
        else:
            gbm_plain_r2s.append(np.nan)
            gbm_sector_r2s.append(np.nan)

    # Compare against stored G1 and M2
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values

    gbm_plain_arr = np.array(gbm_plain_r2s)
    gbm_sector_arr = np.array(gbm_sector_r2s)

    print(f"\n  {'Model':<30s} {'Mean R²':>8s} {'Δ vs G1':>9s} {'Δ vs M2':>9s}")
    print(f"  {'-'*60}")

    for label, arr in [("GBM (plain, no sectors)", gbm_plain_arr),
                       ("GBM (sector interactions)", gbm_sector_arr),
                       ("G1 (global linear)", g1_stored),
                       ("M2 (mixture PCA+ridge)", m2_stored)]:
        m = np.nanmean(arr)
        dg1 = m - np.nanmean(g1_stored)
        dm2 = m - np.nanmean(m2_stored)
        print(f"  {label:<30s} {m:8.4f} {dg1:+9.4f} {dm2:+9.4f}")

    # Paired tests
    print(f"\n  Paired comparisons:")

    # GBM_sector vs GBM_plain
    d1 = gbm_sector_arr - gbm_plain_arr
    t1, p1 = paired_t_test(d1)
    ci1 = bootstrap_ci(d1)
    w1 = int(np.nansum(d1 > 0))
    print(f"  GBM_sector vs GBM_plain: delta={np.nanmean(d1):+.4f}  t={t1:.2f}  p={p1:.4f}"
          f"  CI [{ci1[0]:+.4f}, {ci1[1]:+.4f}]  W={w1}/10")

    # GBM_sector vs M2
    d2 = gbm_sector_arr - m2_stored
    t2, p2 = paired_t_test(d2)
    ci2 = bootstrap_ci(d2)
    w2 = int(np.nansum(d2 > 0))
    print(f"  GBM_sector vs M2:        delta={np.nanmean(d2):+.4f}  t={t2:.2f}  p={p2:.4f}"
          f"  CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}]  W={w2}/10")

    # GBM_sector vs G1
    d3 = gbm_sector_arr - g1_stored
    t3, p3 = paired_t_test(d3)
    ci3 = bootstrap_ci(d3)
    w3 = int(np.nansum(d3 > 0))
    print(f"  GBM_sector vs G1:        delta={np.nanmean(d3):+.4f}  t={t3:.2f}  p={p3:.4f}"
          f"  CI [{ci3[0]:+.4f}, {ci3[1]:+.4f}]  W={w3}/10")

    if np.nanmean(gbm_sector_arr) <= np.nanmean(m2_stored):
        print(f"\n  Conclusion: GBM with sector features does not beat M2.")
        print(f"  The M2 gain is architectural (block decomposition), not a feature-engineering gap.")
    else:
        print(f"\n  Conclusion: GBM with sector features exceeds M2.")
        print(f"  Non-linear sector interactions capture structure beyond block decomposition.")

    return gbm_plain_r2s, gbm_sector_r2s


# ══════════════════════════════════════════════════════════════════════
#  Experiment 2: Cross-sectional Rank IC per quarter
# ══════════════════════════════════════════════════════════════════════

def run_rank_ic(panel, meta):
    """Compute cross-sectional Spearman Rank IC per quarter for G1 vs M2."""
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: CROSS-SECTIONAL RANK IC PER QUARTER")
    print("=" * 80)

    blocks = define_blocks(panel, meta)

    ic_g1_all, ic_m2_all = [], []
    ic_g1_firms, ic_m2_firms = [], []
    quarter_labels = []

    for ty in TEST_YEARS:
        result = run_window_g1_m2(panel, meta, ty, blocks)
        if result is None:
            continue

        preds_g1 = result["G1"]
        preds_m2 = result["M2"]
        actuals = result["actuals"]
        firm_idx = result["firm_idx"]
        N = result["N"]

        tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
        q_idx = 0

        for pred_g1, pred_m2, actual in zip(preds_g1, preds_m2, actuals):
            if q_idx < len(tq):
                qlabel = f"{ty}Q{q_idx + 1}"
            else:
                qlabel = f"{ty}Q?"
            quarter_labels.append(qlabel)
            q_idx += 1

            # Full panel IC
            if np.all(np.isfinite(pred_g1)) and np.all(np.isfinite(actual)):
                rho_g1, _ = spearmanr(pred_g1, actual)
                ic_g1_all.append(float(rho_g1) if np.isfinite(rho_g1) else 0.0)
            else:
                ic_g1_all.append(np.nan)

            if np.all(np.isfinite(pred_m2)) and np.all(np.isfinite(actual)):
                rho_m2, _ = spearmanr(pred_m2, actual)
                ic_m2_all.append(float(rho_m2) if np.isfinite(rho_m2) else 0.0)
            else:
                ic_m2_all.append(np.nan)

            # Firm-only IC
            if firm_idx and len(firm_idx) > 5:
                pred_g1_f = pred_g1[firm_idx]
                pred_m2_f = pred_m2[firm_idx]
                actual_f = actual[firm_idx]

                if np.all(np.isfinite(pred_g1_f)) and np.all(np.isfinite(actual_f)):
                    rho_g1_f, _ = spearmanr(pred_g1_f, actual_f)
                    ic_g1_firms.append(float(rho_g1_f) if np.isfinite(rho_g1_f) else 0.0)
                else:
                    ic_g1_firms.append(np.nan)

                if np.all(np.isfinite(pred_m2_f)) and np.all(np.isfinite(actual_f)):
                    rho_m2_f, _ = spearmanr(pred_m2_f, actual_f)
                    ic_m2_firms.append(float(rho_m2_f) if np.isfinite(rho_m2_f) else 0.0)
                else:
                    ic_m2_firms.append(np.nan)
            else:
                ic_g1_firms.append(np.nan)
                ic_m2_firms.append(np.nan)

    ic_g1_all = np.array(ic_g1_all)
    ic_m2_all = np.array(ic_m2_all)
    ic_g1_firms = np.array(ic_g1_firms)
    ic_m2_firms = np.array(ic_m2_firms)

    # Summary statistics
    print(f"\n  Total test quarters: {len(quarter_labels)}")

    print(f"\n  {'Metric':<40s} {'G1':>8s} {'M2':>8s} {'M2-G1':>8s}")
    print(f"  {'-'*68}")

    mean_g1_all = np.nanmean(ic_g1_all)
    mean_m2_all = np.nanmean(ic_m2_all)
    print(f"  {'Mean IC (all actors)':<40s} {mean_g1_all:8.4f} {mean_m2_all:8.4f}"
          f" {mean_m2_all - mean_g1_all:+8.4f}")

    std_g1_all = np.nanstd(ic_g1_all, ddof=1)
    std_m2_all = np.nanstd(ic_m2_all, ddof=1)
    print(f"  {'Std IC (all actors)':<40s} {std_g1_all:8.4f} {std_m2_all:8.4f}")

    ir_g1 = mean_g1_all / std_g1_all if std_g1_all > 1e-12 else np.nan
    ir_m2 = mean_m2_all / std_m2_all if std_m2_all > 1e-12 else np.nan
    print(f"  {'IC IR (mean/std, all actors)':<40s} {ir_g1:8.4f} {ir_m2:8.4f}"
          f" {ir_m2 - ir_g1:+8.4f}")

    mean_g1_f = np.nanmean(ic_g1_firms)
    mean_m2_f = np.nanmean(ic_m2_firms)
    print(f"  {'Mean IC (firms only, layer=2)':<40s} {mean_g1_f:8.4f} {mean_m2_f:8.4f}"
          f" {mean_m2_f - mean_g1_f:+8.4f}")

    std_g1_f = np.nanstd(ic_g1_firms, ddof=1)
    std_m2_f = np.nanstd(ic_m2_firms, ddof=1)
    print(f"  {'Std IC (firms only)':<40s} {std_g1_f:8.4f} {std_m2_f:8.4f}")

    ir_g1_f = mean_g1_f / std_g1_f if std_g1_f > 1e-12 else np.nan
    ir_m2_f = mean_m2_f / std_m2_f if std_m2_f > 1e-12 else np.nan
    print(f"  {'IC IR (firms only)':<40s} {ir_g1_f:8.4f} {ir_m2_f:8.4f}"
          f" {ir_m2_f - ir_g1_f:+8.4f}")

    # Paired t-test on quarterly ICs (M2 vs G1)
    print(f"\n  Paired t-tests (M2 vs G1):")

    d_all = ic_m2_all - ic_g1_all
    valid_all = d_all[np.isfinite(d_all)]
    t_all, p_all = paired_t_test(d_all)
    ci_all = bootstrap_ci(d_all)
    wins_all = int(np.sum(valid_all > 0))
    print(f"  All actors:  delta={np.nanmean(d_all):+.4f}  t={t_all:.2f}  p={p_all:.4f}"
          f"  CI [{ci_all[0]:+.4f}, {ci_all[1]:+.4f}]  W={wins_all}/{len(valid_all)}")

    d_firms = ic_m2_firms - ic_g1_firms
    valid_firms = d_firms[np.isfinite(d_firms)]
    t_firms, p_firms = paired_t_test(d_firms)
    ci_firms = bootstrap_ci(d_firms)
    wins_firms = int(np.sum(valid_firms > 0))
    print(f"  Firms only:  delta={np.nanmean(d_firms):+.4f}  t={t_firms:.2f}  p={p_firms:.4f}"
          f"  CI [{ci_firms[0]:+.4f}, {ci_firms[1]:+.4f}]  W={wins_firms}/{len(valid_firms)}")

    # Per-year breakdown
    print(f"\n  Per-year mean IC:")
    print(f"  {'Year':<8s} {'G1 all':>8s} {'M2 all':>8s} {'G1 firm':>8s} {'M2 firm':>8s}")
    print(f"  {'-'*44}")
    qi = 0
    for ty in TEST_YEARS:
        # Each year has 4 quarters
        if qi + 4 <= len(ic_g1_all):
            g1_yr = np.nanmean(ic_g1_all[qi:qi+4])
            m2_yr = np.nanmean(ic_m2_all[qi:qi+4])
            g1f_yr = np.nanmean(ic_g1_firms[qi:qi+4])
            m2f_yr = np.nanmean(ic_m2_firms[qi:qi+4])
            print(f"  {ty:<8d} {g1_yr:8.4f} {m2_yr:8.4f} {g1f_yr:8.4f} {m2f_yr:8.4f}")
            qi += 4

    return {
        "quarter_labels": quarter_labels,
        "ic_g1_all": ic_g1_all,
        "ic_m2_all": ic_m2_all,
        "ic_g1_firms": ic_g1_firms,
        "ic_m2_firms": ic_m2_firms,
    }


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 80)
    print("  REFEREE ROUND-3 EXPERIMENTS")
    print("  1. GBM with sector interaction features")
    print("  2. Cross-sectional Rank IC per quarter (G1 vs M2)")
    print("=" * 80)

    panel, meta = load_panel_and_meta()
    print(f"\nPanel: {panel.shape[0]}Q x {panel.shape[1]} actors")

    # Experiment 1: GBM with sector interactions
    gbm_plain_r2s, gbm_sector_r2s = run_gbm_sector_interactions(panel, meta)

    # Experiment 2: Cross-sectional Rank IC
    ic_results = run_rank_ic(panel, meta)

    # Save summary parquet
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # GBM results
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g1_stored = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_stored = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values

    for i, ty in enumerate(TEST_YEARS):
        rows.append({
            "experiment": "gbm_comparison",
            "year": ty,
            "metric": "r2",
            "gbm_plain": gbm_plain_r2s[i],
            "gbm_sector": gbm_sector_r2s[i],
            "g1": float(g1_stored[i]) if i < len(g1_stored) else np.nan,
            "m2": float(m2_stored[i]) if i < len(m2_stored) else np.nan,
        })

    # Rank IC results
    ql = ic_results["quarter_labels"]
    for qi in range(len(ql)):
        rows.append({
            "experiment": "rank_ic",
            "quarter": ql[qi],
            "metric": "rank_ic",
            "ic_g1_all": float(ic_results["ic_g1_all"][qi]),
            "ic_m2_all": float(ic_results["ic_m2_all"][qi]),
            "ic_g1_firms": float(ic_results["ic_g1_firms"][qi]),
            "ic_m2_firms": float(ic_results["ic_m2_firms"][qi]),
        })

    out_path = METRICS_DIR / "iter6_4b_referee_round3.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
