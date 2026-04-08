#!/usr/bin/env python
"""
Filing-lag robustness check.

SEC firm filings are typically available 30-45 days after quarter-end.
This script lags all firm-layer (layer=2) actor data by one quarter,
keeping macro/institutional actors at their original timing, then reruns
the G1 and M2 pipelines to check whether the headline differential survives.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_filing_lag_robustness.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from table7_placebo import (
    load_panel_and_meta,
    run_window_mixture,
    define_real_blocks,
)

TEST_YEARS = list(range(2015, 2025))


def lag_firm_data(panel, meta, lag_quarters=1):
    """Shift firm-layer actors back by lag_quarters.

    At each date t, firm actors see data from t - lag_quarters.
    Macro/institutional actors (layers 0, 1) are unchanged.
    """
    result = panel.copy()
    firm_cols = [c for c in panel.columns
                 if meta.get(c, {}).get("layer", -1) == 2]
    macro_cols = [c for c in panel.columns if c not in firm_cols]

    print(f"  Lagging {len(firm_cols)} firm actors by {lag_quarters}Q")
    print(f"  Keeping {len(macro_cols)} macro/inst actors at original timing")

    # Shift firm columns down by lag_quarters rows
    result[firm_cols] = result[firm_cols].shift(lag_quarters)

    return result


def run_experiment(panel, meta, local_blocks, label):
    """Run G1 and M2, return (global_r2, mixture_r2, delta) per window."""
    deltas, globals_, mixtures = [], [], []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel, ty, local_blocks)
        if res:
            globals_.append(res[0])
            mixtures.append(res[1])
            deltas.append(res[1] - res[0])
    if deltas:
        arr = np.array(deltas)
        g_mean = np.mean(globals_)
        m_mean = np.mean(mixtures)
        d_mean = arr.mean()
        pos = int(np.sum(arr > 0))
        print(f"  {label}:")
        print(f"    G1 R²={g_mean:.4f}  M2 R²={m_mean:.4f}  Δ={d_mean:+.4f}  {pos}/10 positive")
        return g_mean, m_mean, d_mean, pos
    return None, None, None, 0


def main():
    t_start = time.time()

    print("=" * 80)
    print("  FILING-LAG ROBUSTNESS CHECK")
    print("  Firm data lagged by 1 quarter (real-time availability)")
    print("=" * 80)

    panel, meta = load_panel_and_meta()
    real_blocks = define_real_blocks(panel, meta)

    # Baseline (no lag)
    print("\n── 1. BASELINE (contemporaneous, no lag) ──")
    g0, m0, d0, w0 = run_experiment(panel, meta, real_blocks, "No lag")

    # 1-quarter lag on firm data
    print("\n── 2. ONE-QUARTER LAG ON FIRM DATA ──")
    panel_lagged = lag_firm_data(panel, meta, lag_quarters=1)
    g1, m1, d1, w1 = run_experiment(panel_lagged, meta, real_blocks, "1Q firm lag")

    # Comparison
    print("\n── 3. COMPARISON ──")
    print(f"  {'Metric':<20} {'No lag':>10} {'1Q lag':>10} {'Change':>10}")
    print(f"  {'-'*50}")
    if g0 and g1:
        print(f"  {'G1 R²':<20} {g0:>10.4f} {g1:>10.4f} {g1-g0:>+10.4f}")
        print(f"  {'M2 R²':<20} {m0:>10.4f} {m1:>10.4f} {m1-m0:>+10.4f}")
        print(f"  {'Δ (M2-G1)':<20} {d0:>+10.4f} {d1:>+10.4f} {d1-d0:>+10.4f}")
        print(f"  {'Windows positive':<20} {w0:>10} {w1:>10}")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")

    # Verdict
    print("\n" + "=" * 80)
    if d1 is not None and d1 > 0 and w1 >= 7:
        print(f"  VERDICT: Filing-lag correction preserves the mixture gain.")
        pct = (d1 / d0 * 100) if d0 else 0
        print(f"  Δ under lag: {d1:+.4f} ({pct:.0f}% of baseline {d0:+.4f})")
        print(f"  Absolute R² drops as expected (firm data is staler).")
    else:
        print(f"  WARNING: Gain does not clearly survive the lag correction.")
    print("=" * 80)


if __name__ == "__main__":
    main()
