#!/usr/bin/env python
"""
Phase 9.3 -- Regime sub-window analysis for UK/EU panel (experiment_b1).

Splits the 8 test years 2017-2024 into:
  - pre_brexit_completion: 2017-2019 (3 windows)
  - covid:                 2020-2021 (2 windows)
  - post_covid:            2022-2024 (3 windows)

Also runs a sliding 3-year window: (2017-19), (2018-20), ..., (2022-24).

Reads:  results/metrics/table5_uk_eu_<variant>{,_T?}.parquet
Writes: results/metrics/regime_uk_eu_<variant>{,_T?}.json

Usage:
    uv run python scripts/regime_subwindows_uk_eu.py --variant eu_only --t-yr 5
    uv run python scripts/regime_subwindows_uk_eu.py --variant eu_only --t-yr 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

REGIMES = {
    "pre_brexit_completion": list(range(2017, 2020)),
    "covid": [2020, 2021],
    "post_covid": list(range(2022, 2025)),
}

ARCHS = ["G0", "BA", "G1", "S1", "M1", "M2", "BA_M2", "ENS"]


def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 42):
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_t(d: np.ndarray):
    if len(d) < 2:
        return float("nan"), float("nan")
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if se < 1e-15:
        return float("nan"), float("nan")
    t = float(d.mean() / se)
    p = float(2 * t_dist.sf(abs(t), len(d) - 1))
    return t, p


def analyze(df_t5: pd.DataFrame, label: str) -> dict:
    print()
    print("=" * 110)
    print(f"  {label}")
    print("=" * 110)

    pivot = df_t5.pivot(index="year", columns="architecture", values="full_r2")
    pivot = pivot.reindex(columns=[a for a in ARCHS if a in pivot.columns])
    years = sorted(pivot.index.tolist())
    print(f"  Test years available: {years} (n={len(years)})")

    deltas = pd.DataFrame({a: pivot[a] - pivot["G1"] for a in pivot.columns}, index=pivot.index)

    print()
    print("  Per-year R^2 (full panel):")
    print("  " + f"{'year':>6s}  " + "  ".join(f"{a:>8s}" for a in pivot.columns))
    for y in years:
        line = "  " + f"{y:>6d}  " + "  ".join(f"{pivot.loc[y, a]:>8.4f}" for a in pivot.columns)
        print(line)

    print()
    print("  Per-year Delta vs G1:")
    print("  " + f"{'year':>6s}  " + "  ".join(f"{a:>8s}" for a in deltas.columns))
    for y in years:
        line = "  " + f"{y:>6d}  " + "  ".join(f"{deltas.loc[y, a]:>+8.4f}" for a in deltas.columns)
        print(line)

    summary = {
        "label": label,
        "test_years": years,
        "per_year": {int(y): {a: float(pivot.loc[y, a]) for a in pivot.columns} for y in years},
        "per_year_delta_vs_G1": {int(y): {a: float(deltas.loc[y, a]) for a in deltas.columns}
                                  for y in years},
        "regimes": {},
        "sliding": {},
    }

    # Discrete regimes
    print()
    print("  Delta_M2 / Delta_M1 / Delta_BA_M2 / Delta_ENS by REGIME (mean, 95% CI, t, p, wins):")
    print("  " + f"{'regime':<25s}  {'n':>3s}  "
          + "  ".join(f"{'D_'+a:>27s}" for a in ["M1", "M2", "BA_M2", "ENS"]))
    for reg, reg_years in REGIMES.items():
        avail = [y for y in reg_years if y in pivot.index]
        if not avail:
            continue
        line = "  " + f"{reg:<25s}  {len(avail):>3d}  "
        reg_data = {}
        for a in ARCHS:
            d = deltas.loc[avail, a].values
            mean_d = float(d.mean())
            ci_lo, ci_hi = bootstrap_ci(d)
            t_stat, p_val = paired_t(d)
            wins = int((d > 0).sum())
            reg_data[a] = {"mean": mean_d, "ci_lo": ci_lo, "ci_hi": ci_hi,
                           "t": t_stat, "p": p_val, "wins": wins, "n": len(d)}
            if a in ["M1", "M2", "BA_M2", "ENS"]:
                line += f"  {mean_d:>+7.4f} [{ci_lo:>+6.4f},{ci_hi:>+6.4f}] {wins}/{len(d)}"
        print(line)
        summary["regimes"][reg] = {"years": avail, "deltas": reg_data}

    # Sliding 3-year window
    print()
    print("  Delta by SLIDING 3-year window:")
    print("  " + f"{'window':<14s}  {'n':>3s}  "
          + "  ".join(f"{'D_'+a:>27s}" for a in ["M1", "M2", "BA_M2", "ENS"]))
    if years:
        for start in range(min(years), max(years) - 1):
            win = list(range(start, start + 3))
            avail = [y for y in win if y in pivot.index]
            if len(avail) < 2:
                continue
            line = "  " + f"{start}-{start+2:<8d}  {len(avail):>3d}  "
            win_data = {}
            for a in ARCHS:
                d = deltas.loc[avail, a].values
                mean_d = float(d.mean())
                ci_lo, ci_hi = bootstrap_ci(d)
                wins = int((d > 0).sum())
                win_data[a] = {"mean": mean_d, "ci_lo": ci_lo, "ci_hi": ci_hi,
                               "wins": wins, "n": len(d)}
                if a in ["M1", "M2", "BA_M2", "ENS"]:
                    line += f"  {mean_d:>+7.4f} [{ci_lo:>+6.4f},{ci_hi:>+6.4f}] {wins}/{len(d)}"
            print(line)
            summary["sliding"][f"{start}-{start+2}"] = {"years": avail, "deltas": win_data}

    # Homogeneity test: are the three regimes' Delta_M2 means consistent?
    print()
    print("  Regime homogeneity test (one-way ANOVA-like over regime Delta_M2):")
    reg_means = {r: summary["regimes"][r]["deltas"]["M2"]["mean"]
                 for r in summary["regimes"]}
    print(f"    Per-regime Delta_M2: {reg_means}")
    if len(reg_means) >= 2:
        spread = max(reg_means.values()) - min(reg_means.values())
        print(f"    Spread (max - min):  {spread:+.4f}")
        summary["regime_M2_spread"] = float(spread)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="eu_only",
                        help="Panel variant (e.g. 'eu_only')")
    parser.add_argument("--t-yr", type=int, default=5,
                        help="Training-window length matching the table5 file")
    args = parser.parse_args()

    suffix = f"_{args.variant}" if args.variant else ""
    t_suffix = "" if args.t_yr == 5 else f"_T{args.t_yr}"
    table5_base = f"table5_uk_eu{suffix}{t_suffix}"
    in_path = METRICS_DIR / f"{table5_base}.parquet"
    out_path = METRICS_DIR / f"regime_uk_eu{suffix}{t_suffix}.json"

    if not in_path.exists():
        print(f"ERROR: required input does not exist: {in_path}")
        print(f"Run scripts/table5_uk_eu.py with --variant {args.variant} --t-yr {args.t_yr} first.")
        sys.exit(1)

    df = pd.read_parquet(in_path)
    summary = analyze(df, f"REGIME ANALYSIS -- {table5_base} (variant={args.variant}, T_yr={args.t_yr})")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
