#!/usr/bin/env python
"""
Iteration 6.4 Gate D — Conditional / Quarter-Level Gating.

Tests whether augmentation should be activated only in certain states.
Includes NCD-gated augmentation as a primary candidate gate.

Kill Rule D: no causal gate beats always-on by >+0.005.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4_gate_d.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
#  Shared infrastructure (copy from Gate A)
# ══════════════════════════════════════════════════════════════════════

def load_panel():
    df = pd.read_parquet(INTENSITIES_PATH)
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index().loc["2005-01-01":"2025-12-31"]


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


def tercile_encode(vec):
    t1, t2 = np.percentile(vec, [33.3, 66.7])
    symbols = np.zeros(len(vec), dtype=np.uint8)
    symbols[vec > t1] = 1
    symbols[vec > t2] = 2
    return symbols.tobytes()


def ncd(x_bytes, y_bytes):
    cx = len(gzip.compress(x_bytes, compresslevel=9))
    cy = len(gzip.compress(y_bytes, compresslevel=9))
    cxy = len(gzip.compress(x_bytes + y_bytes, compresslevel=9))
    denom = max(cx, cy)
    return (cxy - min(cx, cy)) / denom if denom > 0 else 1.0


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
#  Per-quarter gated augmentation
# ══════════════════════════════════════════════════════════════════════

def run_window_gated(panel, ty, K=K_DEFAULT, T_yr=5):
    """Run per-quarter augmentation with multiple gating strategies.

    For each quarter, computes train-only diagnostic variables, then
    decides whether to use augmented or pooled-only prediction.

    Returns per-quarter results for each gating strategy.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # Stage 1: pooled+FE
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Stage 2 setup
    om_r = ewm_demean(residuals, 12)
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

    # Track per-quarter diagnostics and predictions
    quarter_data = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_ar = bar_y + rho * (prev - bar_y)

        # Augmented prediction
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        y_aug = y_ar + resid_pred

        # ── Train-only diagnostic variables (for gating) ──
        # 1. Cross-sectional dispersion (IQR of current cross-section)
        dispersion = float(np.percentile(prev, 75) - np.percentile(prev, 25))

        # 2. Effective rank of residual covariance
        if dm_r.shape[0] >= 3:
            _, S, _ = np.linalg.svd(dm_r, full_matrices=False)
            S2 = S ** 2
            total = S2.sum()
            if total > 1e-15:
                p = S2 / total
                p = p[p > 1e-15]
                eff_rank = float(np.exp(-np.sum(p * np.log(p))))
            else:
                eff_rank = np.nan
        else:
            eff_rank = np.nan

        # 3. Residual persistence
        resid_rhos = []
        for j in range(min(N, residuals.shape[1])):
            col = residuals[:, j]
            if len(col) >= 3 and np.std(col[:-1]) > 1e-10:
                c = np.corrcoef(col[:-1], col[1:])[0, 1]
                if np.isfinite(c):
                    resid_rhos.append(c)
        persistence = float(np.mean(resid_rhos)) if resid_rhos else 0.0

        # 4. Temporal NCD (between last two training snapshots)
        if otr.shape[0] >= 2:
            x = tercile_encode(otr[-2])
            y = tercile_encode(otr[-1])
            ncd_val = ncd(x, y)
        else:
            ncd_val = 1.0

        # 5. Explained variance concentration
        if dm_r.shape[0] >= 3 and total > 1e-15:
            expl_var = float(S2[:ka].sum() / total)
        else:
            expl_var = np.nan

        quarter_data.append({
            "quarter": qd,
            "actual": obs,
            "pred_pooled": y_ar,
            "pred_augmented": y_aug,
            "dispersion": dispersion,
            "eff_rank": eff_rank,
            "persistence": persistence,
            "ncd": ncd_val,
            "expl_var": expl_var,
        })

        # Kalman update
        actual_resid = obs - y_ar
        odm_r = actual_resid - om_r.ravel()
        S_r = U_r @ Pp_r @ U_r.T + R_r
        try:
            Kg_r = Pp_r @ U_r.T @ np.linalg.solve(S_r, np.eye(N))
        except:
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
            k2 = min(K, mf_r2.basis.shape[0] - 2, mf_r2.K)
            A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
            F_r = _clip_sr(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2
        residuals = residuals_new

    return quarter_data


def apply_gating_policies(all_quarter_data):
    """Apply gating policies to per-quarter predictions.

    Each policy is a function: diagnostics → bool (True = use augmented).
    """
    # Flatten all quarter data across windows
    all_qd = []
    for wd in all_quarter_data:
        if wd is not None:
            all_qd.extend(wd)

    if not all_qd:
        return {}

    # Compute running medians for causal thresholds
    # At each quarter, use the median of ALL PAST diagnostics as threshold
    policies = {
        "always_on": lambda qd, hist: True,
        "pooled_only": lambda qd, hist: False,
        "ncd_gate": lambda qd, hist: qd["ncd"] < np.median([h["ncd"] for h in hist]) if hist else True,
        "dispersion_gate": lambda qd, hist: qd["dispersion"] > np.median([h["dispersion"] for h in hist]) if hist else True,
        "persistence_gate": lambda qd, hist: qd["persistence"] > np.median([h["persistence"] for h in hist]) if hist else True,
        "eff_rank_gate": lambda qd, hist: qd["eff_rank"] < np.median([h["eff_rank"] for h in hist]) if np.isfinite(qd["eff_rank"]) and hist else True,
        "combined_gate": lambda qd, hist: (
            qd["ncd"] < np.median([h["ncd"] for h in hist]) and
            qd["persistence"] > np.median([h["persistence"] for h in hist])
        ) if hist else True,
    }

    results = {p: {"preds": [], "actuals": []} for p in policies}

    history = []
    for qd in all_qd:
        actual = qd["actual"]
        for pname, pfunc in policies.items():
            try:
                use_aug = pfunc(qd, history)
            except Exception:
                use_aug = True
            pred = qd["pred_augmented"] if use_aug else qd["pred_pooled"]
            results[pname]["preds"].append(pred)
            results[pname]["actuals"].append(actual)
        history.append(qd)

    # Compute R² for each policy
    r2s = {}
    for pname, data in results.items():
        try:
            preds = np.concatenate([np.asarray(p).ravel() for p in data["preds"]])
            actuals = np.concatenate([np.asarray(a).ravel() for a in data["actuals"]])
            if len(preds) > 0 and np.all(np.isfinite(preds)):
                r2s[pname] = float(oos_r_squared(preds, actuals))
            else:
                r2s[pname] = np.nan
        except Exception:
            r2s[pname] = np.nan
    return r2s


def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.4 GATE D — CONDITIONAL / QUARTER-LEVEL GATING")
    print("  Should augmentation be state-dependent?")
    print("=" * 80)

    panel = load_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # Collect per-quarter data across all windows
    print("\n  Running per-quarter diagnostics + predictions...")
    all_quarter_data = []
    for ty in TEST_YEARS:
        t0 = time.time()
        qd = run_window_gated(panel, ty)
        all_quarter_data.append(qd)
        if qd:
            ncds = [q["ncd"] for q in qd]
            print(f"  W{ty}: {len(qd)}Q  NCD=[{min(ncds):.3f}, {max(ncds):.3f}]  ({time.time()-t0:.1f}s)")
        else:
            print(f"  W{ty}: FAILED")

    # Apply gating policies
    print("\n  Applying gating policies...")
    r2s = apply_gating_policies(all_quarter_data)

    # Report
    print("\n" + "=" * 80)
    print("  GATING POLICY COMPARISON")
    print("=" * 80)

    ref = r2s.get("always_on", np.nan)
    print(f"\n  {'Policy':<25s} {'R²':>8s} {'Δ vs always-on':>15s}")
    print(f"  {'-'*50}")
    for pname in ["pooled_only", "always_on", "ncd_gate", "dispersion_gate",
                   "persistence_gate", "eff_rank_gate", "combined_gate"]:
        r2 = r2s.get(pname, np.nan)
        delta = r2 - ref if np.isfinite(r2) and np.isfinite(ref) else np.nan
        d_s = f"{delta:+15.4f}" if np.isfinite(delta) else "            N/A"
        mark = " ★" if np.isfinite(delta) and delta > 0.005 else ""
        print(f"  {pname:<25s} {r2:8.4f} {d_s}{mark}")

    # Kill rule
    print(f"\n  ═══════════════════════════════════════════════")
    best_gate = max((p for p in r2s if p not in ("always_on", "pooled_only")),
                    key=lambda p: r2s.get(p, -np.inf), default=None)
    if best_gate:
        best_delta = r2s[best_gate] - ref
        if best_delta > 0.005:
            print(f"  KILL RULE D: NOT TRIGGERED — {best_gate} beats always-on by {best_delta:+.4f}")
        else:
            print(f"  KILL RULE D: TRIGGERED — best gate ({best_gate}) only {best_delta:+.4f}")
            print(f"  Predictability is not meaningfully state-dependent.")
    else:
        print(f"  KILL RULE D: TRIGGERED — no valid gates")
    print(f"  ═══════════════════════════════════════════════")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"policy": p, "r2": r2s[p]} for p in r2s]
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_4_gate_d.parquet", index=False)
    # Also save per-quarter diagnostics
    all_qd_flat = []
    for wd in all_quarter_data:
        if wd:
            for qd in wd:
                all_qd_flat.append({k: v for k, v in qd.items()
                                    if k not in ("actual", "pred_pooled", "pred_augmented")})
    if all_qd_flat:
        pd.DataFrame(all_qd_flat).to_parquet(METRICS_DIR / "iter6_4_gate_d_diagnostics.parquet", index=False)
    print(f"\n  Saved: iter6_4_gate_d*.parquet")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
