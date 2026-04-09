"""Portfolio backtest: does better rank prediction translate to better returns?

Runs quintile-sort portfolios using M2, G1, AR(1), momentum, and naive
investment factor signals. Reports Sharpe ratios, drawdowns, information
ratios, and probabilistic Sharpe ratios.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/run_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "portfolio"))

from signals import SIGNAL_REGISTRY, COMBO_SIGNAL_REGISTRY, signal_from_predictions
from portfolio import quintile_sort, long_short_returns, long_only_returns, compute_turnover
from metrics import full_metrics, sharpe_ratio, information_ratio

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    """Load predictions and returns."""
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    # Align: keep only actors present in both
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def run_strategy(
    name: str,
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    ew_returns: np.ndarray | None = None,
) -> dict | None:
    """Run one strategy and return metrics."""
    sorted_df = quintile_sort(signals, returns)
    if sorted_df.empty or sorted_df["quintile"].nunique() < 2:
        return None
    ls = long_short_returns(sorted_df)
    lo = long_only_returns(sorted_df)
    turnover = compute_turnover(sorted_df, quintile=5)

    ls_ret = ls["return_ls"].values
    lo_ret = lo["return_top_q"].values

    result = full_metrics(ls_ret, label=f"{name} L/S")
    result["mean_turnover"] = turnover["turnover"].iloc[1:].mean()  # skip first

    # Long-only metrics
    lo_metrics = full_metrics(lo_ret, label=f"{name} Long")
    result["lo_ann_return"] = lo_metrics["ann_return"]
    result["lo_sharpe"] = lo_metrics["sharpe"]

    if ew_returns is not None and len(lo_ret) > 0:
        n = min(len(lo_ret), len(ew_returns))
        if n > 0:
            result["info_ratio"] = information_ratio(lo_ret[-n:], ew_returns[-n:])

    # Per-quintile returns
    qr = sorted_df.groupby("quintile")["return_q"].mean()
    for q in range(1, 6):
        result[f"Q{q}_mean_ret"] = float(qr.get(q, np.nan))

    # Monotonicity: Q5 > Q4 > ... > Q1?
    q_means = [qr.get(q, np.nan) for q in range(1, 6)]
    result["monotonic"] = all(q_means[i] <= q_means[i + 1] for i in range(4))

    return result


def run_subsample_analysis(
    name: str,
    signals: pd.DataFrame,
    returns: pd.DataFrame,
):
    """Run strategy on pre/post COVID subsamples."""
    results = {}
    for period, start, end in [
        ("Full", "2015-01-01", "2024-12-31"),
        ("Pre-COVID", "2015-01-01", "2019-12-31"),
        ("Post-COVID", "2020-01-01", "2024-12-31"),
    ]:
        sig_sub = signals[
            (signals["quarter"] >= start) & (signals["quarter"] <= end)
        ]
        ret_sub = returns[
            (returns["quarter"] >= start) & (returns["quarter"] <= end)
        ]
        if len(sig_sub) < 20:
            continue
        sorted_df = quintile_sort(sig_sub, ret_sub)
        ls = long_short_returns(sorted_df)
        sr = sharpe_ratio(ls["return_ls"].values)
        results[period] = sr
    return results


def main():
    print("=" * 80)
    print("  PORTFOLIO BACKTEST: DOES BETTER RANK PREDICTION → BETTER RETURNS?")
    print("=" * 80)

    predictions, returns = load_data()
    n_actors = predictions["actor_id"].nunique()
    n_quarters = returns["quarter"].nunique()
    print(f"\n  Universe: {n_actors} US firms, {n_quarters} quarters")
    print(f"  Period: {returns['quarter'].min().date()} to {returns['quarter'].max().date()}")

    # Equal-weight benchmark
    ew = returns.groupby("quarter")["return_q"].mean().sort_index()
    ew_ret = ew.values
    ew_metrics = full_metrics(ew_ret, label="Equal Weight")
    print(f"\n  Equal-weight benchmark: Ann.Ret={ew_metrics['ann_return']:.1%}, "
          f"Sharpe={ew_metrics['sharpe']:.2f}")

    # ── Run all strategies ──
    all_results = [ew_metrics]

    for name, signal_fn in SIGNAL_REGISTRY.items():
        print(f"\n  Running {name}...", end=" ", flush=True)
        if name in ("Momentum",):
            signals = signal_fn(returns)
        else:
            signals = signal_fn(predictions)

        result = run_strategy(name, signals, returns, ew_ret)
        if result is None:
            print("SKIP (insufficient quintile spread)")
            continue
        all_results.append(result)
        print(f"L/S Sharpe={result['sharpe']:.2f}, "
              f"PSR={result['psr']:.1%}, "
              f"MaxDD={result['max_dd']:.1%}")

    # ── Run combo signals (momentum + prediction) ──
    for name, signal_fn in COMBO_SIGNAL_REGISTRY.items():
        print(f"\n  Running {name}...", end=" ", flush=True)
        signals = signal_fn(predictions, returns)
        result = run_strategy(name, signals, returns, ew_ret)
        if result is None:
            print("SKIP (insufficient quintile spread)")
            continue
        all_results.append(result)
        print(f"L/S Sharpe={result['sharpe']:.2f}, "
              f"PSR={result['psr']:.1%}, "
              f"MaxDD={result['max_dd']:.1%}")

    # ── Summary Table: Long-Short ──
    print("\n" + "=" * 80)
    print("  LONG-SHORT PORTFOLIO RESULTS (Q5 - Q1, equal-weight)")
    print("=" * 80)
    print(f"\n  {'Strategy':<15s} {'Ann.Ret':>8s} {'Vol':>7s} {'Sharpe':>7s} "
          f"{'[CI lo':>7s} {'hi]':>6s} {'PSR':>6s} {'MaxDD':>7s} "
          f"{'Calmar':>7s} {'Hit%':>5s} {'TO':>5s}")
    print(f"  {'-' * 95}")

    for r in all_results:
        label = r["label"]
        if "L/S" not in label and label != "Equal Weight":
            continue
        name = label.replace(" L/S", "")
        turnover = r.get("mean_turnover", 0)
        print(f"  {name:<15s} {r['ann_return']:>7.1%} {r['ann_vol']:>7.1%} "
              f"{r['sharpe']:>7.2f} {r.get('sharpe_ci_lo', 0):>7.2f} "
              f"{r.get('sharpe_ci_hi', 0):>5.2f} {r.get('psr', 0):>6.1%} "
              f"{r['max_dd']:>7.1%} {r.get('calmar', 0):>7.2f} "
              f"{r['hit_rate']:>5.0%} {turnover:>5.1%}")

    # ── Quintile Spread ──
    print("\n" + "=" * 80)
    print("  QUINTILE MEAN RETURNS (quarterly)")
    print("=" * 80)
    print(f"\n  {'Strategy':<15s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s} {'Q5':>8s} {'Mono?':>6s}")
    print(f"  {'-' * 65}")
    for r in all_results:
        if "L/S" not in r["label"]:
            continue
        name = r["label"].replace(" L/S", "")
        qs = [r.get(f"Q{i}_mean_ret", np.nan) for i in range(1, 6)]
        mono = "Yes" if r.get("monotonic", False) else "No"
        print(f"  {name:<15s} " + " ".join(f"{q:>8.2%}" for q in qs) + f" {mono:>6s}")

    # ── M2 vs G1 Head-to-Head ──
    print("\n" + "=" * 80)
    print("  M2 vs G1: ARCHITECTURE EDGE")
    print("=" * 80)
    m2_result = next(r for r in all_results if r["label"] == "M2 L/S")
    g1_result = next(r for r in all_results if r["label"] == "G1 L/S")
    print(f"\n  M2 L/S Sharpe: {m2_result['sharpe']:.3f}  [{m2_result['sharpe_ci_lo']:.3f}, {m2_result['sharpe_ci_hi']:.3f}]")
    print(f"  G1 L/S Sharpe: {g1_result['sharpe']:.3f}  [{g1_result['sharpe_ci_lo']:.3f}, {g1_result['sharpe_ci_hi']:.3f}]")
    print(f"  Δ Sharpe:      {m2_result['sharpe'] - g1_result['sharpe']:+.3f}")
    print(f"  M2 PSR:        {m2_result['psr']:.1%}")
    print(f"  G1 PSR:        {g1_result['psr']:.1%}")

    # ── Subsample Analysis ──
    print("\n" + "=" * 80)
    print("  SUBSAMPLE ANALYSIS (L/S Sharpe)")
    print("=" * 80)
    print(f"\n  {'Strategy':<15s} {'Full':>8s} {'Pre-COVID':>10s} {'Post-COVID':>11s}")
    print(f"  {'-' * 50}")
    for name, signal_fn in SIGNAL_REGISTRY.items():
        if name in ("Momentum",):
            signals = signal_fn(returns)
        else:
            signals = signal_fn(predictions)
        subs = run_subsample_analysis(name, signals, returns)
        vals = [subs.get(p, np.nan) for p in ["Full", "Pre-COVID", "Post-COVID"]]
        print(f"  {name:<15s} " + " ".join(f"{v:>8.2f}" if np.isfinite(v) else f"{'n/a':>8s}" for v in vals))
    for name, signal_fn in COMBO_SIGNAL_REGISTRY.items():
        signals = signal_fn(predictions, returns)
        subs = run_subsample_analysis(name, signals, returns)
        vals = [subs.get(p, np.nan) for p in ["Full", "Pre-COVID", "Post-COVID"]]
        print(f"  {name:<15s} " + " ".join(f"{v:>8.2f}" if np.isfinite(v) else f"{'n/a':>8s}" for v in vals))

    # ── Transaction Cost Sensitivity ──
    print("\n" + "=" * 80)
    print("  TRANSACTION COST SENSITIVITY (M2 L/S)")
    print("=" * 80)
    m2_signals = SIGNAL_REGISTRY["M2"](predictions)
    sorted_m2 = quintile_sort(m2_signals, returns)
    ls_m2 = long_short_returns(sorted_m2)
    ls_raw = ls_m2["return_ls"].values
    to = compute_turnover(sorted_m2, quintile=5)
    mean_to = to["turnover"].iloc[1:].mean()

    print(f"\n  Mean quarterly turnover (Q5): {mean_to:.1%}")
    print(f"  {'Cost (bps)':>12s} {'Sharpe':>8s} {'Ann.Ret':>8s}")
    print(f"  {'-' * 32}")
    for cost_bps in [0, 5, 10, 20, 50]:
        cost_frac = cost_bps / 10000
        # Approximate: each quarter, turnover * 2 sides * cost
        tc_drag = mean_to * 2 * cost_frac
        adjusted = ls_raw - tc_drag
        sr = sharpe_ratio(adjusted)
        ann = float(np.mean(adjusted) * 4) if len(adjusted) > 0 else 0
        print(f"  {cost_bps:>8d} bps {sr:>8.2f} {ann:>8.1%}")

    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
