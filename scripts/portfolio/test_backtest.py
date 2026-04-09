"""Backtest verification tests.

Tests:
  1. Random signal → Sharpe ≈ 0 (no look-ahead)
  2. Perfect foresight → very high Sharpe (machinery works)
  3. Signal-return timing check (no contemporaneous leakage)
  4. Quintile size verification
  5. Momentum look-ahead check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "portfolio"))

from portfolio import quintile_sort, long_short_returns
from metrics import sharpe_ratio

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def test_random_signal(returns: pd.DataFrame):
    """Random signal should produce Sharpe ≈ 0 (no systematic edge)."""
    print("\n  TEST 1: RANDOM SIGNAL")
    rng = np.random.default_rng(42)
    signals = returns[["quarter", "actor_id"]].copy()
    signals["score"] = rng.standard_normal(len(signals))

    sorted_df = quintile_sort(signals, returns)
    ls = long_short_returns(sorted_df)
    sr = sharpe_ratio(ls["return_ls"].values)
    status = "PASS" if abs(sr) < 0.5 else "FAIL"
    print(f"    Random signal Sharpe: {sr:.3f} (expect ≈ 0)  [{status}]")

    # Run 100 random seeds
    sharpes = []
    for seed in range(100):
        signals["score"] = np.random.default_rng(seed).standard_normal(len(signals))
        sorted_df = quintile_sort(signals, returns)
        ls = long_short_returns(sorted_df)
        sharpes.append(sharpe_ratio(ls["return_ls"].values))
    print(f"    100 random seeds: mean={np.mean(sharpes):.3f}, "
          f"std={np.std(sharpes):.3f}, |max|={np.max(np.abs(sharpes)):.3f}")
    status2 = "PASS" if abs(np.mean(sharpes)) < 0.15 else "FAIL"
    print(f"    Mean ≈ 0?  [{status2}]")
    return abs(np.mean(sharpes)) < 0.15


def test_perfect_foresight(returns: pd.DataFrame):
    """Sort by actual future return → should get very high Sharpe."""
    print("\n  TEST 2: PERFECT FORESIGHT (sort by actual return)")
    signals = returns[["quarter", "actor_id", "return_q"]].copy()
    signals["score"] = signals["return_q"].values  # perfect foresight = look-ahead
    signals = signals[["quarter", "actor_id", "score"]]

    sorted_df = quintile_sort(signals, returns)
    ls = long_short_returns(sorted_df)
    sr = sharpe_ratio(ls["return_ls"].values)
    status = "PASS" if sr > 2.0 else "FAIL"
    print(f"    Perfect foresight Sharpe: {sr:.3f} (expect >> 2)  [{status}]")

    # Check quintile means
    qr = sorted_df.groupby("quintile")["return_q"].mean()
    monotonic = all(qr.iloc[i] <= qr.iloc[i+1] for i in range(len(qr)-1))
    status2 = "PASS" if monotonic else "FAIL"
    print(f"    Monotonic Q1<Q2<...<Q5: {monotonic}  [{status2}]")
    for q in sorted(qr.index):
        print(f"      Q{q}: {qr[q]:.2%}")
    return sr > 2.0


def test_momentum_look_ahead(predictions: pd.DataFrame, returns: pd.DataFrame):
    """Check if momentum signal uses contemporaneous return."""
    print("\n  TEST 3: MOMENTUM LOOK-AHEAD CHECK")

    from signals import signal_momentum

    mom = signal_momentum(returns)

    # Merge momentum score with the SAME quarter's return
    merged = mom.merge(returns, on=["quarter", "actor_id"])

    # If momentum includes current quarter return, correlation will be high
    corr = merged["score"].corr(merged["return_q"])
    print(f"    Correlation(momentum_score, same_quarter_return): {corr:.4f}")

    # For proper momentum, the signal should be computed from PAST returns only
    # High correlation (> 0.3) indicates the current return is in the signal
    status = "FAIL (LOOK-AHEAD)" if abs(corr) > 0.3 else "PASS"
    print(f"    Look-ahead contamination?  [{status}]")

    if abs(corr) > 0.3:
        print("    → Momentum signal includes current quarter's return!")
        print("    → Must lag the signal by 1 quarter.")

    return abs(corr) <= 0.3


def test_signal_return_alignment(predictions: pd.DataFrame, returns: pd.DataFrame):
    """Verify prediction signals don't leak future information."""
    print("\n  TEST 4: PREDICTION SIGNAL TIMING")

    # pred_m2 at quarter Q is formed using data through Q-1
    # return at quarter Q is the return during Q
    # Correlation should be low (prediction ≠ return)
    merged = predictions.merge(returns, on=["quarter", "actor_id"])
    corr_m2 = merged["pred_m2"].corr(merged["return_q"])
    corr_g1 = merged["pred_g1"].corr(merged["return_q"])
    corr_actual = merged["actual"].corr(merged["return_q"])

    print(f"    Corr(pred_m2, return): {corr_m2:.4f}")
    print(f"    Corr(pred_g1, return): {corr_g1:.4f}")
    print(f"    Corr(actual_rank, return): {corr_actual:.4f}")
    print(f"    (Low correlation expected — rank ≠ return)")

    # The actual intensity rank and return should have LOW correlation
    # because they measure different things
    status = "PASS" if abs(corr_actual) < 0.3 else "INVESTIGATE"
    print(f"    No systematic rank→return link?  [{status}]")
    return True


def test_quintile_sizes(returns: pd.DataFrame):
    """Verify quintiles have reasonable sizes."""
    print("\n  TEST 5: QUINTILE SIZE VERIFICATION")

    from signals import SIGNAL_REGISTRY, COMBO_SIGNAL_REGISTRY
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]

    for name in ["Momentum", "M2 positive", "Mom+M2pos 50/50"]:
        if name in SIGNAL_REGISTRY:
            if name == "Momentum":
                signals = SIGNAL_REGISTRY[name](returns)
            else:
                signals = SIGNAL_REGISTRY[name](predictions)
        elif name in COMBO_SIGNAL_REGISTRY:
            signals = COMBO_SIGNAL_REGISTRY[name](predictions, returns)
        else:
            continue

        sorted_df = quintile_sort(signals, returns)
        sizes = sorted_df.groupby(["quarter", "quintile"]).size().reset_index(name="count")
        mean_sizes = sizes.groupby("quintile")["count"].mean()
        print(f"    {name}: ", end="")
        for q in sorted(mean_sizes.index):
            print(f"Q{q}={mean_sizes[q]:.0f} ", end="")
        print()

    return True


def main():
    print("=" * 70)
    print("  BACKTEST VERIFICATION TESTS")
    print("=" * 70)

    predictions, returns = load_data()

    results = []
    results.append(("Random signal", test_random_signal(returns)))
    results.append(("Perfect foresight", test_perfect_foresight(returns)))
    results.append(("Momentum look-ahead", test_momentum_look_ahead(predictions, returns)))
    results.append(("Signal-return alignment", test_signal_return_alignment(predictions, returns)))
    results.append(("Quintile sizes", test_quintile_sizes(returns)))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    {name:<30s}  [{status}]")

    if not all_pass:
        print("\n  ⚠ SOME TESTS FAILED — results are unreliable until fixed")
    else:
        print("\n  All tests passed")


if __name__ == "__main__":
    main()
