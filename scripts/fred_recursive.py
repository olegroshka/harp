#!/usr/bin/env python
"""
FRED Recursive Normalization Robustness Check.

The main pipeline uses full-sample min-max normalization for 7 macro actors,
introducing look-ahead bias. This script re-normalizes macro actors using
strictly expanding-window min-max bounds (only data up to each quarter),
then reruns the mixture pipeline (M2) and global (G1) to check whether
the headline differential survives.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_fred_recursive_robustness.py
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

# Reuse mixture runner
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from table7_placebo import (
    load_panel_and_meta,
    run_window_mixture,
    define_real_blocks,
)

# Also need the raw FRED data to re-normalize
FRED_PIT_PATH = PROJECT_ROOT / "data" / "smim" / "pit_store" / "fred.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "experiment_a1_registry.json"

TEST_YEARS = list(range(2015, 2025))


def get_macro_actor_ids(meta):
    """Get actor IDs for layer 0 and layer 1 actors (the FRED-normalized ones)."""
    return [aid for aid, m in meta.items() if m.get("layer", -1) in (0, 1)]


def recursive_minmax_normalize(panel, macro_cols):
    """Replace full-sample min-max with expanding-window min-max for macro columns only.

    For each macro column, at each time t, the normalized value is:
        (x_t - min(x_1..x_t)) / (max(x_1..x_t) - min(x_1..x_t))

    This ensures no future information enters the normalization.
    Firm columns (cross-sectional ranks) are left unchanged.
    """
    result = panel.copy()
    for col in macro_cols:
        if col not in result.columns:
            continue
        raw = result[col].values.astype(np.float64)
        normalized = np.full_like(raw, np.nan)

        running_min = np.inf
        running_max = -np.inf

        for t in range(len(raw)):
            if np.isnan(raw[t]):
                continue
            running_min = min(running_min, raw[t])
            running_max = max(running_max, raw[t])
            span = running_max - running_min
            if span > 1e-10:
                normalized[t] = (raw[t] - running_min) / span
            else:
                normalized[t] = 0.5  # degenerate case early in series

        result[col] = normalized

    return result


def load_panel_with_raw_fred():
    """Load panel but replace FRED actors with raw (un-normalized) values.

    Then we can apply recursive normalization ourselves.
    """
    # Load the standard panel (has full-sample min-max for macro)
    panel, meta = load_panel_and_meta()
    macro_ids = get_macro_actor_ids(meta)

    # Load raw FRED data from PIT store
    if not FRED_PIT_PATH.exists():
        print(f"  WARNING: FRED PIT store not found at {FRED_PIT_PATH}")
        print(f"  Falling back to reverse-engineering raw values from the panel")
        # We can't get raw values without the PIT store, but we CAN
        # apply expanding-window normalization to the already-normalized values.
        # This is a second-order approximation: expanding-window of full-sample-normalized
        # values != expanding-window of raw values, but it tests the direction.
        return panel, meta, macro_ids, False

    # Load raw FRED and build a mapping
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    fred_df = pd.read_parquet(FRED_PIT_PATH)

    # Build actor_id -> FRED series_id mapping from registry
    actor_to_series = {}
    for actor in registry["actors"]:
        if actor.get("layer", -1) in (0, 1):
            # Extract FRED series ID from data_config
            dc = actor.get("data_config", {})
            sid = dc.get("fred_series_id") or dc.get("series_id")
            if sid:
                actor_to_series[actor["actor_id"]] = sid

    # For each macro actor, replace panel values with raw FRED values
    quarter_dates = panel.index
    for actor_id, series_id in actor_to_series.items():
        if actor_id not in panel.columns:
            continue
        series_data = fred_df[fred_df["signal_id"] == series_id].copy()
        if series_data.empty:
            continue
        series_data["date"] = pd.to_datetime(series_data["date"])
        series_data = series_data.sort_values("date")

        # Resample to quarterly (last observation per quarter)
        series_data = series_data.set_index("date")["value"]
        quarterly = series_data.resample("QS").last()

        for qd in quarter_dates:
            if qd in quarterly.index and actor_id in panel.columns:
                panel.at[qd, actor_id] = quarterly[qd]

    return panel, meta, macro_ids, True


def main():
    t_start = time.time()

    print("=" * 80)
    print("  FRED RECURSIVE NORMALIZATION ROBUSTNESS CHECK")
    print("=" * 80)

    # Load standard panel first for baseline
    panel_std, meta = load_panel_and_meta()
    macro_ids_all = get_macro_actor_ids(meta)
    macro_ids = [a for a in macro_ids_all if a in panel_std.columns]
    real_blocks = define_real_blocks(panel_std, meta)

    print(f"\nPanel: {panel_std.shape[0]}Q x {len(panel_std.columns)} actors")
    print(f"Macro actors in panel: {len(macro_ids)} (of {len(macro_ids_all)} in registry)")
    print(f"  IDs: {macro_ids}")

    # ── 1. Baseline: original full-sample min-max ──
    print("\n── 1. BASELINE (full-sample min-max, original) ──")
    baseline_deltas = []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel_std, ty, real_blocks)
        if res:
            baseline_deltas.append(res[1] - res[0])
    baseline_mean = np.mean(baseline_deltas)
    baseline_pos = int(np.sum(np.array(baseline_deltas) > 0))
    print(f"  Global R2 (mean): computed from pipeline")
    print(f"  M2 Delta (mean): {baseline_mean:+.4f}, {baseline_pos}/10 positive")

    # ── 2. Recursive min-max normalization ──
    print("\n── 2. RECURSIVE EXPANDING-WINDOW MIN-MAX ──")

    # Apply expanding-window normalization to the macro columns
    # We work with the panel that already has full-sample min-max values.
    # To get the "raw" values back, we'd need the original FRED data.
    # Instead, we apply expanding-window normalization to the existing panel,
    # which re-normalizes macro actors using only past data at each point.
    panel_recursive = recursive_minmax_normalize(panel_std, macro_ids)

    # Verify the transformation changed something
    diff = (panel_std[macro_ids] - panel_recursive[macro_ids]).abs()
    mean_diff = diff.mean().mean()
    max_diff = diff.max().max()
    print(f"  Mean absolute change in macro values: {mean_diff:.4f}")
    print(f"  Max absolute change: {max_diff:.4f}")

    recursive_deltas = []
    recursive_globals = []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel_recursive, ty, real_blocks)
        if res:
            recursive_deltas.append(res[1] - res[0])
            recursive_globals.append(res[0])
    recursive_mean = np.mean(recursive_deltas)
    recursive_pos = int(np.sum(np.array(recursive_deltas) > 0))
    recursive_global_mean = np.mean(recursive_globals) if recursive_globals else np.nan

    print(f"  Global R2 (mean): {recursive_global_mean:.4f}")
    print(f"  M2 Delta (mean): {recursive_mean:+.4f}, {recursive_pos}/10 positive")

    # ── 3. Comparison ──
    print("\n── 3. COMPARISON ──")
    print(f"  Baseline M2 Delta:   {baseline_mean:+.4f} ({baseline_pos}/10)")
    print(f"  Recursive M2 Delta:  {recursive_mean:+.4f} ({recursive_pos}/10)")
    diff_delta = recursive_mean - baseline_mean
    print(f"  Difference:          {diff_delta:+.4f}")

    # ── 4. Additional: exclude macro actors entirely ──
    print("\n── 4. MACRO-EXCLUDED PANEL (firm-only, 82 actors) ──")
    firm_cols = [c for c in panel_std.columns if c not in set(macro_ids)]
    panel_firms = panel_std[firm_cols]
    # Only diversified and tech/health blocks (macro/inst block is excluded)
    firm_blocks = [
        [a for a in real_blocks[0] if a in firm_cols],  # diversified
        [a for a in real_blocks[2] if a in firm_cols],  # tech/health
    ]
    firm_deltas = []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel_firms, ty, firm_blocks)
        if res:
            firm_deltas.append(res[1] - res[0])
    firm_mean = np.mean(firm_deltas) if firm_deltas else np.nan
    firm_pos = int(np.sum(np.array(firm_deltas) > 0)) if firm_deltas else 0
    print(f"  M2 Delta (firms only): {firm_mean:+.4f}, {firm_pos}/10 positive")

    # ── Verdict ──
    print("\n" + "=" * 80)
    pct_retained = (recursive_mean / baseline_mean * 100) if baseline_mean > 0 else 0
    print(f"  VERDICT: Recursive normalization retains {pct_retained:.0f}% of the gain.")
    if recursive_pos >= 8:
        print(f"  The FRED look-ahead bias does NOT materially affect the result.")
    else:
        print(f"  WARNING: Window consistency degraded ({recursive_pos}/10).")
    print(f"  Firm-only panel (no macro actors at all): {firm_mean:+.4f} ({firm_pos}/10)")
    print("=" * 80)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
