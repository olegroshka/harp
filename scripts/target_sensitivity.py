#!/usr/bin/env python
"""
Iteration 6.4 Gate A — Target / Noise Audit.

Tests whether the R² ceiling is an artifact of target construction.
Target variants, change-vs-level, split-half reliability, perturbation audit.

Kill Rule A: no variant changes augmentation gain by more than ±0.005.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4_gate_a.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist

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
#  Shared infrastructure (from 6.2)
# ══════════════════════════════════════════════════════════════════════

def load_panel_and_registry():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    sector_map = {a["actor_id"]: a["sector"] for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel, sector_map


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


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


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


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _mean_valid(lst):
    vals = [x for x in lst if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════
#  C1 augmentation runner (simplified from 6.2)
# ══════════════════════════════════════════════════════════════════════

def run_window_augmented(panel, ty, K=K_DEFAULT, ewm=12, T_yr=5):
    """Run C1 two-stage augmentation: pooled+FE → DMD/Kalman full Ã on residuals.

    Returns dict with ar1_r2, pooled_r2, augmented_r2, or None.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # Stage 1: pooled+FE
    rho, bar_y = estimate_pooled_ar1(otr)
    predicted_train = bar_y + rho * (otr[:-1] - bar_y)
    residuals = otr[1:] - predicted_train

    # Per-actor AR(1)
    mu_ar1 = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu_ar1
    rho_ar1 = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho_ar1[j] = c

    # Stage 2: DMD on residuals
    om_r = ewm_demean(residuals, ewm)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None

    ka = min(K, mf_r.basis.shape[0] - 2, mf_r.K)
    A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
    F_r = _clip_sr(A_r)
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    ps_ar1, ps_pool, ps_aug, ac = [], [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # AR(1)
        ps_ar1.append(mu_ar1 + rho_ar1 * (prev - mu_ar1))

        # Pooled+FE
        y_ar = bar_y + rho * (prev - bar_y)
        ps_pool.append(y_ar)

        # Augmented
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        ps_aug.append(y_ar + resid_pred)
        ac.append(obs)

        # Kalman update
        actual_resid = obs - y_ar
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
        mu_ar1 = np.nan_to_num(otr.mean(0), nan=0.5)
        d = otr - mu_ar1
        rho_ar1 = np.zeros(N)
        for j in range(N):
            y = d[:, j]
            if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
                c = np.corrcoef(y[:-1], y[1:])[0, 1]
                if np.isfinite(c):
                    rho_ar1[j] = c
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, ewm)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            k2 = min(K, mf_r2.basis.shape[0] - 2, mf_r2.K)
            A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
            F_r = _clip_sr(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2

    if not ps_ar1:
        return None
    ar1_a = np.array(ps_ar1)
    pool_a = np.array(ps_pool)
    aug_a = np.array(ps_aug)
    act_a = np.array(ac)
    if not np.all(np.isfinite(aug_a)):
        return None

    return {
        "ar1_r2": float(oos_r_squared(ar1_a.ravel(), act_a.ravel())),
        "pooled_r2": float(oos_r_squared(pool_a.ravel(), act_a.ravel())),
        "augmented_r2": float(oos_r_squared(aug_a.ravel(), act_a.ravel())),
    }


# ══════════════════════════════════════════════════════════════════════
#  Target transformation functions
# ══════════════════════════════════════════════════════════════════════

def transform_panel(panel, method, sector_map=None):
    """Transform the panel values. All transformations are monotonic/reversible."""
    if method == "ranks":
        return panel  # already [0,1] ranks

    elif method == "normal_quantiles":
        # Inverse normal CDF of ranks → z-scores
        return panel.apply(lambda col: norm_dist.ppf(col.clip(0.005, 0.995)))

    elif method == "winsorized_z":
        z = panel.apply(lambda col: norm_dist.ppf(col.clip(0.005, 0.995)))
        return z.clip(-2, 2)

    elif method == "sector_relative":
        # Subtract per-quarter sector mean
        result = panel.copy()
        for q_idx in range(len(panel)):
            row = panel.iloc[q_idx]
            for sector in set(sector_map.values()):
                actors_in_sector = [a for a in panel.columns if sector_map.get(a) == sector]
                actors_present = [a for a in actors_in_sector if a in panel.columns]
                if actors_present:
                    sec_mean = row[actors_present].mean()
                    result.iloc[q_idx, result.columns.isin(actors_present)] -= sec_mean
        return result

    elif method == "ma2":
        return panel.rolling(2, min_periods=1).mean()

    elif method == "ma4":
        return panel.rolling(4, min_periods=1).mean()

    elif method == "changes":
        return panel.diff()

    else:
        raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════
#  M1. Target-construction sensitivity
# ══════════════════════════════════════════════════════════════════════

def run_target_sensitivity(panel, sector_map):
    """M1: Compare augmentation gain across target variants."""
    methods = ["ranks", "normal_quantiles", "winsorized_z", "sector_relative",
               "ma2", "ma4", "changes"]

    print("\n  M1 — TARGET-CONSTRUCTION SENSITIVITY")
    print("  " + "=" * 70)

    results = {}
    for method in methods:
        t0 = time.time()
        try:
            transformed = transform_panel(panel, method, sector_map)
            # Drop NaN rows from changes/MA
            transformed = transformed.dropna(how="all")
        except Exception as e:
            print(f"    {method}: FAILED ({e})")
            continue

        r2s = []
        for ty in TEST_YEARS:
            res = run_window_augmented(transformed, ty)
            if res:
                r2s.append(res)

        if not r2s:
            print(f"    {method}: no valid windows")
            continue

        ar1_mean = _mean_valid([r["ar1_r2"] for r in r2s])
        pool_mean = _mean_valid([r["pooled_r2"] for r in r2s])
        aug_mean = _mean_valid([r["augmented_r2"] for r in r2s])
        gain = aug_mean - ar1_mean if np.isfinite(aug_mean) and np.isfinite(ar1_mean) else np.nan

        results[method] = {
            "ar1": ar1_mean, "pooled": pool_mean,
            "augmented": aug_mean, "gain": gain, "n_windows": len(r2s),
        }
        print(f"    {method:<20s}: AR1={ar1_mean:.4f}  Pool={pool_mean:.4f}"
              f"  Aug={aug_mean:.4f}  Gain={gain:+.4f}  ({time.time()-t0:.1f}s)")

    return results


# ══════════════════════════════════════════════════════════════════════
#  M2. Change-vs-level formulation
# ══════════════════════════════════════════════════════════════════════

def run_change_vs_level(panel):
    """M2: Compare level prediction vs change prediction."""
    print("\n  M2 — CHANGE-VS-LEVEL FORMULATION")
    print("  " + "=" * 70)

    # Level prediction (standard)
    level_r2s = []
    for ty in TEST_YEARS:
        res = run_window_augmented(panel, ty)
        if res:
            level_r2s.append(res)

    # Change prediction: predict Δy, then reconstruct level
    change_panel = panel.diff().dropna(how="all")
    change_r2s = []
    for ty in TEST_YEARS:
        res = run_window_augmented(change_panel, ty)
        if res:
            change_r2s.append(res)

    level_aug = _mean_valid([r["augmented_r2"] for r in level_r2s])
    level_gain = _mean_valid([r["augmented_r2"] - r["ar1_r2"] for r in level_r2s])
    change_aug = _mean_valid([r["augmented_r2"] for r in change_r2s])
    change_gain = _mean_valid([r["augmented_r2"] - r["ar1_r2"] for r in change_r2s])

    print(f"    Level:  Aug R²={level_aug:.4f}  Gain={level_gain:+.4f}")
    print(f"    Change: Aug R²={change_aug:.4f}  Gain={change_gain:+.4f}")

    return {"level": {"aug": level_aug, "gain": level_gain},
            "change": {"aug": change_aug, "gain": change_gain}}


# ══════════════════════════════════════════════════════════════════════
#  M3. Low-frequency / high-frequency decomposition
# ══════════════════════════════════════════════════════════════════════

def run_freq_decomposition(panel):
    """M3: Augmentation on 4Q-MA target (low-freq) vs quarterly residual (high-freq)."""
    print("\n  M3 — LOW-FREQUENCY vs HIGH-FREQUENCY TARGET")
    print("  " + "=" * 70)

    low_freq = panel.rolling(4, min_periods=1).mean()
    high_freq = panel - low_freq

    results = {}
    for label, p in [("low_freq_4Q_MA", low_freq), ("high_freq_residual", high_freq)]:
        p = p.dropna(how="all")
        r2s = []
        for ty in TEST_YEARS:
            res = run_window_augmented(p, ty)
            if res:
                r2s.append(res)
        aug = _mean_valid([r["augmented_r2"] for r in r2s])
        gain = _mean_valid([r["augmented_r2"] - r["ar1_r2"] for r in r2s])
        results[label] = {"aug": aug, "gain": gain}
        print(f"    {label:<25s}: Aug R²={aug:.4f}  Gain={gain:+.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  M4. Split-half reliability
# ══════════════════════════════════════════════════════════════════════

def run_split_half_reliability(panel, n_splits=50):
    """M4: Split actors into random halves, estimate reliability and noise ceiling."""
    print("\n  M4 — SPLIT-HALF RELIABILITY")
    print("  " + "=" * 70)

    rng = np.random.default_rng(42)
    actors = list(panel.columns)
    N = len(actors)

    half_r2_pairs = []
    for s in range(n_splits):
        perm = rng.permutation(N)
        half1 = [actors[i] for i in perm[:N // 2]]
        half2 = [actors[i] for i in perm[N // 2:]]

        r2_h1, r2_h2 = [], []
        for ty in TEST_YEARS:
            res1 = run_window_augmented(panel[half1], ty)
            res2 = run_window_augmented(panel[half2], ty)
            if res1 and res2:
                r2_h1.append(res1["augmented_r2"])
                r2_h2.append(res2["augmented_r2"])

        if len(r2_h1) >= 5:
            rho_split = np.corrcoef(r2_h1, r2_h2)[0, 1]
            if np.isfinite(rho_split):
                half_r2_pairs.append(rho_split)

    if half_r2_pairs:
        mean_rho = np.mean(half_r2_pairs)
        # Spearman-Brown correction for full-length reliability
        reliability = 2 * mean_rho / (1 + mean_rho) if mean_rho > 0 else 0
        # Observed R² on full panel
        full_r2s = [run_window_augmented(panel, ty) for ty in TEST_YEARS]
        full_aug = _mean_valid([r["augmented_r2"] for r in full_r2s if r])
        noise_ceiling = full_aug / np.sqrt(reliability) if reliability > 0.01 else np.nan

        print(f"    Split-half ρ (mean of {len(half_r2_pairs)} splits): {mean_rho:.3f}")
        print(f"    Spearman-Brown reliability: {reliability:.3f}")
        print(f"    Full-panel augmented R²: {full_aug:.4f}")
        print(f"    Noise-corrected ceiling ≈ {noise_ceiling:.4f}")
        return {"rho": mean_rho, "reliability": reliability,
                "full_aug": full_aug, "ceiling": noise_ceiling}
    else:
        print(f"    FAILED — not enough valid splits")
        return None


# ══════════════════════════════════════════════════════════════════════
#  M5. Perturbation audit
# ══════════════════════════════════════════════════════════════════════

def run_perturbation_audit(panel):
    """M5: Add noise to targets and measure R² sensitivity."""
    print("\n  M5 — PERTURBATION AUDIT")
    print("  " + "=" * 70)

    sigmas = [0.0, 0.01, 0.02, 0.05, 0.10]
    rng = np.random.default_rng(42)
    results = {}

    for sigma in sigmas:
        t0 = time.time()
        if sigma == 0:
            perturbed = panel
        else:
            noise = pd.DataFrame(
                rng.normal(0, sigma, panel.shape),
                index=panel.index, columns=panel.columns
            )
            perturbed = (panel + noise).clip(0, 1)

        r2s = []
        for ty in TEST_YEARS:
            res = run_window_augmented(perturbed, ty)
            if res:
                r2s.append(res)

        aug = _mean_valid([r["augmented_r2"] for r in r2s])
        gain = _mean_valid([r["augmented_r2"] - r["ar1_r2"] for r in r2s])

        # Rank stability: Spearman ρ between original and perturbed cross-sections
        if sigma > 0:
            from scipy.stats import spearmanr
            rank_stab = []
            for q in range(min(20, len(panel))):
                orig = panel.iloc[q].dropna()
                pert = perturbed.iloc[q].reindex(orig.index).dropna()
                common = orig.index.intersection(pert.index)
                if len(common) > 5:
                    rho_s, _ = spearmanr(orig[common], pert[common])
                    if np.isfinite(rho_s):
                        rank_stab.append(rho_s)
            mean_rank_stab = np.mean(rank_stab) if rank_stab else np.nan
        else:
            mean_rank_stab = 1.0

        results[sigma] = {"aug": aug, "gain": gain, "rank_stability": mean_rank_stab}
        print(f"    σ={sigma:.2f}: Aug R²={aug:.4f}  Gain={gain:+.4f}"
              f"  Rank ρ={mean_rank_stab:.4f}  ({time.time()-t0:.1f}s)")

    return results


# ══════════════════════════════════════════════════════════════════════
#  Kill Rule A
# ══════════════════════════════════════════════════════════════════════

def evaluate_kill_rule_a(m1_results):
    """Kill Rule A: no variant changes augmentation gain by >±0.005."""
    print("\n" + "=" * 80)
    print("  KILL RULE A EVALUATION")
    print("=" * 80)

    ref_gain = m1_results.get("ranks", {}).get("gain", np.nan)
    max_delta = 0.0
    best_variant = "ranks"

    for method, res in m1_results.items():
        if method == "ranks":
            continue
        g = res.get("gain", np.nan)
        if np.isfinite(g) and np.isfinite(ref_gain):
            delta = g - ref_gain
            print(f"    {method:<20s}: Δgain = {delta:+.4f}")
            if abs(delta) > abs(max_delta):
                max_delta = delta
                best_variant = method

    killed = abs(max_delta) <= 0.005
    print(f"\n  Max |Δgain| = {abs(max_delta):.4f} (variant: {best_variant})")
    print(f"\n  ═══════════════════════════════════════════════")
    if killed:
        print(f"  KILL RULE A: TRIGGERED — ceiling is NOT target-specific")
        print(f"  No target variant changes augmentation gain by >±0.005.")
    else:
        print(f"  KILL RULE A: NOT TRIGGERED — target variant '{best_variant}' changes gain by {max_delta:+.4f}")
        print(f"  Consider re-running later gates with the better target.")
    print(f"  ═══════════════════════════════════════════════")
    return killed, best_variant, max_delta


# ══════════════════════════════════════════════════════════════════════
#  Save & Main
# ══════════════════════════════════════════════════════════════════════

def save_results(m1, m2, m3, m4, m5):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, res in m1.items():
        rows.append({"test": "M1_target", "variant": method, **res})
    for label, res in m2.items():
        rows.append({"test": "M2_change_level", "variant": label, **res})
    for label, res in m3.items():
        rows.append({"test": "M3_freq", "variant": label, **res})
    if m4:
        rows.append({"test": "M4_reliability", "variant": "split_half", **m4})
    for sigma, res in m5.items():
        rows.append({"test": "M5_perturbation", "variant": f"sigma_{sigma}", **res})
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_4_gate_a.parquet", index=False)
    print(f"\n  Saved: iter6_4_gate_a.parquet")


def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.4 GATE A — TARGET / NOISE AUDIT")
    print("  Is the R² ceiling an artifact of target construction?")
    print("=" * 80)

    panel, sector_map = load_panel_and_registry()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # M1: Target-construction sensitivity
    m1 = run_target_sensitivity(panel, sector_map)

    # M2: Change vs level
    m2 = run_change_vs_level(panel)

    # M3: Frequency decomposition
    m3 = run_freq_decomposition(panel)

    # M4: Split-half reliability
    m4 = run_split_half_reliability(panel, n_splits=30)

    # M5: Perturbation audit
    m5 = run_perturbation_audit(panel)

    # Kill rule
    killed, best_var, max_delta = evaluate_kill_rule_a(m1)

    # Save
    save_results(m1, m2, m3, m4, m5)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    return m1, m2, m3, m4, m5, killed


if __name__ == "__main__":
    main()
