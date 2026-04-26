#!/usr/bin/env python
"""
Placebo test for UK/EU panel (experiment_b1) -- mirrors table7_placebo.py.

Phase 8 of UK_EU_PANEL_PLAN.md. Runs N_PLACEBO=1000 random block partitions
with the same block sizes as the real economic blocks, comparing the resulting
mean Delta = mixture - global to the real Delta. If the real Delta sits in
the upper tail of the placebo distribution, the economic block assignment
captures genuine heterogeneity rather than arbitrary structure.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/table7_placebo_uk_eu.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.spectral.dmd import ExactDMDDecomposer
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = None
REGISTRY_PATH = None
OUT_BASENAME = None
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2017, 2025))
PANEL_START = "2011-04-01"
PANEL_END = "2025-12-31"


def _resolve_paths(variant: str, t_yr: int = 5, block_scheme: str = "us_inherited"):
    suffix = f"_{variant}" if variant else ""
    panel = f"experiment_b1{suffix}"
    out_suffix = "" if t_yr == 5 else f"_T{t_yr}"
    bs_suffix = "" if block_scheme == "us_inherited" else f"_{block_scheme}"
    return (
        PROJECT_ROOT / "data" / "intensities" / f"{panel}_intensities.parquet",
        PROJECT_ROOT / "data" / "registries" / f"{panel}_registry.json",
        f"placebo_uk_eu{suffix}{out_suffix}{bs_suffix}",
    )

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
N_PLACEBO = 1000


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


def run_window_mixture(panel, ty, local_actor_sets, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    local_indices = []
    for block_actors in local_actor_sets:
        bidx = [v_list.index(a) for a in block_actors if a in v_list]
        if len(bidx) >= 5:
            local_indices.append(bidx)

    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    om_r = ewm_demean(residuals, 12)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None
    ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
    F_r = _clip_sr(mf_r.metadata["Atilde"][:ka, :ka].real.copy())
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    local_models = []
    for bidx in local_indices:
        N_b = len(bidx)
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        K_b = min(4, max(2, N_b // 5))
        C = dm_b.T @ dm_b / dm_b.shape[0]
        ev, ec = np.linalg.eigh(C)
        U_pca = ec[:, np.argsort(ev)[::-1]][:, :K_b]
        fac = dm_b @ U_pca
        X, Y = fac[:-1], fac[1:]
        try:
            A_pca = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
        except np.linalg.LinAlgError:
            A_pca = np.eye(K_b) * 0.5
        local_models.append({"U_pca": U_pca, "A_pca": A_pca, "om_b": om_b,
                              "K_b": K_b, "bidx": bidx, "N_b": N_b})

    preds_global, preds_mixture, actuals = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64) if qd in ad.index else np.zeros((0, N))
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_pool = bar_y + rho * (prev - bar_y)

        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        y_global = y_pool + resid_pred
        preds_global.append(y_global.copy())

        y_mix = y_global.copy()
        prev_resid = prev - (bar_y + rho * (np.nan_to_num(
            otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y)
        ) if otr.shape[0] >= 2 else np.zeros(N)
        for lm in local_models:
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
            F_r = _clip_sr(mf_r2.metadata["Atilde"][:k2, :k2].real.copy())
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2
        residuals = residuals_new

        for lm in local_models:
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]
            C2 = dm_b.T @ dm_b / dm_b.shape[0]
            ev2, ec2 = np.linalg.eigh(C2)
            lm["U_pca"] = ec2[:, np.argsort(ev2)[::-1]][:, :K_b]
            fac = dm_b @ lm["U_pca"]
            X, Y = fac[:-1], fac[1:]
            try:
                lm["A_pca"] = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
            except Exception:
                pass
            lm["om_b"] = om_b

    if not actuals:
        return None
    act_a = np.array(actuals)
    glob_a = np.array(preds_global)
    mix_a = np.array(preds_mixture)
    if not np.all(np.isfinite(glob_a)) or not np.all(np.isfinite(mix_a)):
        return None
    return (float(oos_r_squared(glob_a.ravel(), act_a.ravel())),
            float(oos_r_squared(mix_a.ravel(), act_a.ravel())))


def define_real_blocks(panel, meta, scheme: str = "us_inherited"):
    actors = list(panel.columns)
    blocks = []
    if scheme == "us_inherited":
        blocks.append([a for a in actors if meta.get(a, {}).get("sector") == "diversified"])
        blocks.append([a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)])
        blocks.append([a for a in actors if meta.get(a, {}).get("sector") in ("technology", "healthcare")])
    elif scheme == "sector_split":
        blocks.append([a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)])
        for sec in ["energy", "financials", "healthcare", "technology", "consumer", "industrials"]:
            blocks.append([a for a in actors
                           if meta.get(a, {}).get("sector") == sec
                           and meta.get(a, {}).get("layer", -1) == 2])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return blocks


def main():
    global INTENSITIES_PATH, REGISTRY_PATH, OUT_BASENAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--t-yr", type=int, default=5,
                        help="Training-window length passed through to run_window_mixture")
    parser.add_argument("--block-scheme", type=str, default="us_inherited",
                        choices=["us_inherited", "sector_split"],
                        help="Block partition scheme; 'sector_split' uses 7 blocks (macro_inst + 6 SMIM sectors)")
    args = parser.parse_args()
    INTENSITIES_PATH, REGISTRY_PATH, OUT_BASENAME = _resolve_paths(args.variant, args.t_yr, args.block_scheme)

    t_start = time.time()

    print("=" * 80)
    print(f"  PLACEBO TEST -- {INTENSITIES_PATH.stem}")
    print(f"  Variant: '{args.variant or 'full'}' | T_yr: {args.t_yr} | Output: {OUT_BASENAME}.{{parquet,json}}")
    print(f"  {N_PLACEBO} random block partitions vs real economic blocks")
    print("=" * 80)

    panel, meta = load_panel_and_meta()
    actors = list(panel.columns)
    N_total = len(actors)
    print(f"\nPanel: {panel.shape[0]}Q x {N_total} actors")

    real_blocks = define_real_blocks(panel, meta, scheme=args.block_scheme)
    real_sizes = [len(b) for b in real_blocks]
    real_local_n = sum(real_sizes)
    print(f"\nReal blocks: sizes {real_sizes}, total local = {real_local_n}, "
          f"remainder = {N_total - real_local_n}")
    # Drop empty blocks (SEC_diversified is empty in B1) -- the placebo scheme
    # samples ONLY the non-empty real-block sizes so the comparison is fair.
    nonempty_sizes = [s for s in real_sizes if s > 0]
    print(f"Non-empty block sizes used for placebo sampling: {nonempty_sizes}")

    print("\n  Running REAL mixture...")
    real_deltas = []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel, ty, real_blocks, T_yr=args.t_yr)
        if res:
            real_deltas.append(res[1] - res[0])
    real_mean_delta = np.mean(real_deltas) if real_deltas else np.nan
    print(f"  Real mixture: mean Delta = {real_mean_delta:+.4f} ({len(real_deltas)} windows)")

    print(f"\n  Running {N_PLACEBO} PLACEBO mixtures (random blocks, sizes {nonempty_sizes})...")
    rng = np.random.default_rng(42)
    placebo_deltas = []
    t_loop = time.time()

    for p_idx in range(N_PLACEBO):
        perm = rng.permutation(N_total)
        random_blocks = []
        offset = 0
        for sz in nonempty_sizes:
            block_actors = [actors[perm[i]] for i in range(offset, offset + sz)]
            random_blocks.append(block_actors)
            offset += sz

        deltas = []
        for ty in TEST_YEARS:
            res = run_window_mixture(panel, ty, random_blocks, T_yr=args.t_yr)
            if res:
                deltas.append(res[1] - res[0])
        if deltas:
            placebo_deltas.append(np.mean(deltas))

        if (p_idx + 1) % 50 == 0:
            elapsed = time.time() - t_loop
            rate = (p_idx + 1) / elapsed
            eta = (N_PLACEBO - p_idx - 1) / rate
            print(f"    Placebo {p_idx+1}/{N_PLACEBO}: mean Delta = {placebo_deltas[-1]:+.4f}"
                  f"  ({rate:.1f}/s, ETA {eta:.0f}s)")

    placebo_arr = np.array(placebo_deltas)

    print("\n" + "=" * 80)
    print("  PLACEBO TEST RESULTS")
    print("=" * 80)
    print(f"\n  Real economic blocks:  mean Delta = {real_mean_delta:+.4f}")
    print(f"  Placebo distribution:  mean = {placebo_arr.mean():+.4f}"
          f"  std = {placebo_arr.std():.4f}")
    print(f"  Placebo range:         [{placebo_arr.min():+.4f}, {placebo_arr.max():+.4f}]")
    print(f"  Placebo 95th pct:      {np.percentile(placebo_arr, 95):+.4f}")
    print(f"  Placebo 99th pct:      {np.percentile(placebo_arr, 99):+.4f}")

    p_placebo = float(np.mean(placebo_arr >= real_mean_delta))
    print(f"\n  Placebo p-value: {p_placebo:.4f} ({int(p_placebo * N_PLACEBO)}/{N_PLACEBO}"
          f" placebos >= real)")

    if placebo_arr.std() > 1e-10:
        z = float((real_mean_delta - placebo_arr.mean()) / placebo_arr.std())
        print(f"  Z-score (real vs placebo): {z:.2f}")
    else:
        z = float("nan")

    print()
    if p_placebo < 0.05:
        verdict = "PASSED"
        msg = ("The real economic block assignment produces gains that random blocks of the "
               "same size cannot replicate. Block structure captures genuine heterogeneity.")
    else:
        verdict = "FAILED"
        msg = ("Random blocks produce comparable gains. The gain may come from local estimation "
               "per se, not from the specific economic block assignment.")
    print(f"  PLACEBO TEST: {verdict}")
    print(f"  --> {msg}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "real", "mean_delta": real_mean_delta, "p_placebo": p_placebo}]
    for i, d in enumerate(placebo_deltas):
        rows.append({"type": "placebo", "placebo_idx": i, "mean_delta": d})
    pd.DataFrame(rows).to_parquet(METRICS_DIR / f"{OUT_BASENAME}.parquet", index=False)

    summary = {
        "panel": "experiment_b1",
        "n_placebo": N_PLACEBO,
        "test_years": TEST_YEARS,
        "real_block_sizes": real_sizes,
        "nonempty_block_sizes_used": nonempty_sizes,
        "real_mean_delta": float(real_mean_delta),
        "placebo_mean": float(placebo_arr.mean()),
        "placebo_std": float(placebo_arr.std()),
        "placebo_min": float(placebo_arr.min()),
        "placebo_max": float(placebo_arr.max()),
        "placebo_p95": float(np.percentile(placebo_arr, 95)),
        "placebo_p99": float(np.percentile(placebo_arr, 99)),
        "placebo_p_value": p_placebo,
        "z_score": z,
        "verdict": verdict,
    }
    with open(METRICS_DIR / f"{OUT_BASENAME}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_BASENAME}.{{parquet,json}}")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
