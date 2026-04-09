"""Extract per-actor, per-quarter predictions from the mixture architecture pipeline.

Runs the same estimation as table5_architectures.py but saves the full
prediction matrix for each architecture and test quarter. This is the
input for the portfolio backtest.

Output: results/portfolio/predictions.parquet
    Columns: year, quarter, actor_id, actual, pred_g0, pred_g1, pred_m2, pred_ar1
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from table5_architectures import (
    load_panel_and_meta,
    define_blocks,
    _prepare_window,
    K_DEFAULT,
    K_MAX,
    Q_INIT_SCALE,
    LAMBDA_Q,
    TEST_YEARS,
    estimate_pooled_ar1,
    estimate_block_ar1,
    ewm_demean,
    dmd_full,
    sph_r,
    _clip_sr,
    fit_local_pca_ridge,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "portfolio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_predictions_for_window(panel, ty, blocks, meta, T_yr=5):
    """Run one test year and return per-actor predictions as a DataFrame."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    # Identify firm-layer actors only (layer 2)
    firm_actors = set()
    for a in v_list:
        m = meta.get(a, {})
        layer = m.get("layer", -1)
        if layer == 2:
            firm_actors.add(a)

    # Map actors to blocks
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

    # Stage 1: Global pooled+FE
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Per-actor AR(1)
    mu_ar1 = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tl = otr - mu_ar1
    rho_ar1 = np.zeros(N)
    for i in range(N):
        denom = np.sum(tl[:-1, i] ** 2)
        if denom > 1e-12:
            rho_ar1[i] = np.sum(tl[1:, i] * tl[:-1, i]) / denom

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

    # Local models (M2)
    local_models = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        N_b = len(bidx)
        if N_b < 3:
            continue
        resid_b = residuals[:, bidx]
        om_bb = ewm_demean(resid_b, 12)
        dm_bb = resid_b - om_bb
        K_b = min(4, max(2, N_b // 5))
        U_pca_b, A_pca_b = fit_local_pca_ridge(dm_bb, K_b)
        local_models[bname] = {
            "U_pca": U_pca_b, "A_pca": A_pca_b,
            "om_b": om_bb, "K_b": K_b, "bidx": bidx, "N_b": N_b,
        }

    # Rolling test — collect predictions
    rows = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # G0: pooled-only
        y_pool = bar_y + rho * (prev - bar_y)

        # AR(1): per-actor
        y_ar1 = mu_ar1 + rho_ar1 * (prev - mu_ar1)

        # G1: global augmented
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_g1 = y_pool + resid_pred_global

        # M2: mixture with local PCA+ridge
        y_m2 = y_g1.copy()
        prev_resid = prev - (bar_y + rho * (
            np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y
        )) if otr.shape[0] >= 2 else np.zeros(N)
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

        # Save per-actor predictions (firms only)
        for j, actor_id in enumerate(v_list):
            if actor_id not in firm_actors:
                continue
            rows.append({
                "year": ty,
                "quarter": qd,
                "actor_id": actor_id,
                "sector": meta.get(actor_id, {}).get("sector", "unknown"),
                "actual": float(obs[j]),
                "pred_g0": float(y_pool[j]),
                "pred_g1": float(y_g1[j]),
                "pred_m2": float(y_m2[j]),
                "pred_ar1": float(y_ar1[j]),
            })

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

        # Rolling update
        otr = np.vstack([otr, qv])
        residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals, 12)
        dm_r = residuals - om_r
        mf_new = dmd_full(dm_r, k_svd=K_MAX)
        if mf_new is not None:
            ka_new = min(K_DEFAULT, mf_new.basis.shape[0] - 2, mf_new.K)
            A_r = mf_new.metadata["Atilde"][:ka_new, :ka_new].real.copy()
            F_r = _clip_sr(A_r)
            U_r = mf_new.metadata["U"][:, :ka_new]
            R_r = sph_r(dm_r, U_r)
            ka = ka_new
            a_r = np.zeros(ka)
            P_r = np.eye(ka)
            Q_r = np.eye(ka) * Q_INIT_SCALE

        # Update local models
        for bname in local_block_names:
            bidx = block_indices[bname]
            resid_b = residuals[:, bidx]
            om_bb = ewm_demean(resid_b, 12)
            dm_bb = resid_b - om_bb
            K_b = min(4, max(2, len(bidx) // 5))
            U_pca_b, A_pca_b = fit_local_pca_ridge(dm_bb, K_b)
            local_models[bname] = {
                "U_pca": U_pca_b, "A_pca": A_pca_b,
                "om_b": om_bb, "K_b": K_b, "bidx": bidx, "N_b": len(bidx),
            }

        # Update per-actor AR(1)
        mu_ar1 = np.nan_to_num(otr.mean(axis=0), nan=0.5)
        tl = otr - mu_ar1
        for i in range(N):
            denom = np.sum(tl[:-1, i] ** 2)
            if denom > 1e-12:
                rho_ar1[i] = np.sum(tl[1:, i] * tl[:-1, i]) / denom

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  EXTRACT PER-ACTOR PREDICTIONS (82 firms, 2015-2024)")
    print("=" * 70)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta)

    all_dfs = []
    for ty in TEST_YEARS:
        print(f"  Window {ty}...", end=" ", flush=True)
        df = extract_predictions_for_window(panel, ty, blocks, meta)
        if df is not None:
            all_dfs.append(df)
            n_firms = df["actor_id"].nunique()
            print(f"{len(df)} rows, {n_firms} firms")
        else:
            print("SKIP")

    predictions = pd.concat(all_dfs, ignore_index=True)
    out_path = OUTPUT_DIR / "predictions.parquet"
    predictions.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Actors: {predictions['actor_id'].nunique()}")
    print(f"  Quarters: {predictions['quarter'].nunique()}")

    # Quick sanity: per-window R² should match table5
    print("\n  Sanity check (R² by year, should match Table 5):")
    from harp.validation.metrics import oos_r_squared
    for ty in TEST_YEARS:
        wd = predictions[predictions["year"] == ty]
        for arch in ["pred_g0", "pred_g1", "pred_m2", "pred_ar1"]:
            r2 = oos_r_squared(wd[arch].values, wd["actual"].values)
        g1 = oos_r_squared(wd["pred_g1"].values, wd["actual"].values)
        m2 = oos_r_squared(wd["pred_m2"].values, wd["actual"].values)
        print(f"    {ty}: G1={g1:.3f}  M2={m2:.3f}  Δ={m2 - g1:+.3f}")


if __name__ == "__main__":
    main()
