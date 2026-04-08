#!/usr/bin/env python
"""
Paper robustness checks requested by referees:
  1. Drop tech/health block → does mixture gain survive?
  2. Block boundary sensitivity → move 3 borderline actors
  3. Local treatment of remainder block → does it help or hurt?
  4. Report all 10 candidate blocks' per-block R²

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_paper_robustness.py
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

# Reuse the mixture runner from the placebo script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from table7_placebo import (
    load_panel_and_meta,
    run_window_mixture,
)

TEST_YEARS = list(range(2015, 2025))


def run_experiment(panel, meta, local_blocks, label, T_yr=5):
    """Run mixture pipeline with given blocks, return mean delta and per-window."""
    deltas = []
    for ty in TEST_YEARS:
        res = run_window_mixture(panel, ty, local_blocks, T_yr=T_yr)
        if res:
            deltas.append(res[1] - res[0])
    if deltas:
        arr = np.array(deltas)
        mean_d = arr.mean()
        pos = int(np.sum(arr > 0))
        print(f"  {label}: mean Δ = {mean_d:+.4f}, {pos}/10 windows positive")
        return mean_d, pos, arr
    return None, 0, np.array([])


def main():
    t_start = time.time()
    panel, meta = load_panel_and_meta()
    actors = list(panel.columns)

    # Define actor groups
    diversified = [a for a in actors if meta.get(a, {}).get("sector") == "diversified"]
    macro_inst = [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)]
    tech_health = [a for a in actors if meta.get(a, {}).get("sector") in ("technology", "healthcare")]
    energy = [a for a in actors if meta.get(a, {}).get("sector") == "energy"
              and meta.get(a, {}).get("layer", -1) == 2]
    financials = [a for a in actors if meta.get(a, {}).get("sector") == "financials"
                  and meta.get(a, {}).get("layer", -1) == 2]
    industrials = [a for a in actors if meta.get(a, {}).get("sector") == "industrials"]
    remainder = [a for a in actors if a not in set(diversified + macro_inst + tech_health)]

    print("=" * 80)
    print("  PAPER ROBUSTNESS CHECKS")
    print("=" * 80)
    print(f"\nPanel: {panel.shape[0]}Q × {len(actors)} actors")
    print(f"Blocks: diversified={len(diversified)}, macro_inst={len(macro_inst)}, "
          f"tech_health={len(tech_health)}, remainder={len(remainder)}")
    print(f"Remainder breakdown: energy={len(energy)}, fin={len(financials)}, ind={len(industrials)}")

    # ── 0. Baseline: original M2 ──
    print("\n── 0. BASELINE (original M2) ──")
    real_blocks = [diversified, macro_inst, tech_health]
    run_experiment(panel, meta, real_blocks, "M2 (original)")

    # ── 1. Drop tech/health ──
    print("\n── 1. DROP TECH/HEALTH ──")
    # Remove tech/health actors from the panel entirely
    panel_no_th = panel[[a for a in actors if a not in set(tech_health)]]
    remaining_blocks = [diversified, macro_inst]
    run_experiment(panel_no_th, meta, remaining_blocks, "M2 (no tech/health)")

    # ── 2. Block boundary sensitivity ──
    print("\n── 2. BLOCK BOUNDARY SENSITIVITY ──")
    # Move 3 borderline actors:
    # - 1 diversified conglomerate → industrials (out of diversified block)
    # - 1 fintech → tech/health (into tech/health block)
    # - 1 healthcare supplier → diversified (swap between blocks)
    # Pick the first actor from each group for perturbation
    if len(diversified) > 1 and len(financials) > 0:
        perturbed_div = diversified[1:]  # remove 1 from diversified
        perturbed_th = tech_health + [financials[0]]  # add 1 fintech to tech/health
        perturbed_macro = macro_inst
        run_experiment(panel, meta, [perturbed_div, perturbed_macro, perturbed_th],
                       "M2 (move 1 div→rem, 1 fin→th)")

    if len(diversified) > 2 and len(financials) > 1:
        perturbed_div2 = diversified[2:]  # remove 2 from diversified
        perturbed_th2 = tech_health + financials[:2]  # add 2 financials to tech/health
        run_experiment(panel, meta, [perturbed_div2, macro_inst, perturbed_th2],
                       "M2 (move 2 div→rem, 2 fin→th)")

    if len(tech_health) > 1:
        perturbed_th3 = tech_health[1:]  # remove 1 from tech/health
        perturbed_div3 = diversified + [tech_health[0]]  # add to diversified
        run_experiment(panel, meta, [perturbed_div3, macro_inst, perturbed_th3],
                       "M2 (move 1 th→div)")

    # ── 3. Local treatment of remainder ──
    print("\n── 3. LOCAL TREATMENT OF REMAINDER ──")
    # Add remainder as a 4th local block
    all_four_blocks = [diversified, macro_inst, tech_health, remainder]
    run_experiment(panel, meta, all_four_blocks, "M2 (all 4 blocks local)")

    # Remainder only (just the 34-actor block gets local, others stay global)
    run_experiment(panel, meta, [remainder], "M2 (only remainder local)")

    # ── 4. All 10 candidate blocks ──
    print("\n── 4. ALL 10 CANDIDATE BLOCKS (single-block local treatment) ──")
    candidates = {
        "SEC_diversified": diversified,
        "SEC_energy": energy,
        "SEC_financials": financials,
        "SEC_industrials": industrials,
        "SEC_technology": [a for a in actors if meta.get(a, {}).get("sector") == "technology"],
        "SEC_healthcare": [a for a in actors if meta.get(a, {}).get("sector") == "healthcare"],
        "MERGED_tech_health": tech_health,
        "MERGED_ind_energy": industrials + energy,
        "LAYER_macro_inst": macro_inst,
        "LAYER_firms": [a for a in actors if meta.get(a, {}).get("layer", -1) == 2],
    }
    for name, block_actors in candidates.items():
        run_experiment(panel, meta, [block_actors], f"{name} (N={len(block_actors)})")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
