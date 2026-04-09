"""Final analysis: G1-M2 disagreement as a long-only stock selection signal.

Computes active returns (signal portfolio minus equal-weight benchmark)
with proper attribution and statistical tests.
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

from metrics import (
    sharpe_ratio, sharpe_ratio_bootstrap_ci, probabilistic_sharpe_ratio,
    max_drawdown, hit_rate, annualised_return, annualised_volatility,
    information_ratio,
)

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def build_signal(predictions: pd.DataFrame, acc_window: int | None = None) -> pd.DataFrame:
    """Build G1-M2 disagreement signal, optionally with accuracy scaling."""
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]

    if acc_window is not None:
        df["err_g1"] = (df["pred_g1"] - df["actual"]).abs()
        df["err_m2"] = (df["pred_m2"] - df["actual"]).abs()
        df["m2_better"] = (df["err_m2"] < df["err_g1"]).astype(float)
        df["m2_hit"] = df.groupby("actor_id")["m2_better"].transform(
            lambda x: x.rolling(acc_window, min_periods=1).mean().shift(1)
        )
        df["score"] = df["raw_disagr"] * df["m2_hit"].fillna(0.5)
    else:
        df["score"] = df["raw_disagr"]

    return df[["quarter", "actor_id", "score"]].dropna()


def long_only_portfolio(
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    N: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Top-N long-only portfolio vs equal-weight benchmark.

    Returns: (portfolio_returns, ew_returns, quarters)
    """
    merged = signal_df.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())
    port_ret, ew_ret, qs = [], [], []
    for q in quarters:
        qd = merged[merged["quarter"] == q].dropna(subset=["score"])
        if len(qd) < N:
            continue
        ranked = qd.sort_values("score")
        port_ret.append(ranked.tail(N)["return_q"].mean())
        ew_ret.append(qd["return_q"].mean())
        qs.append(q)
    return np.array(port_ret), np.array(ew_ret), qs


def report_strategy(name: str, port: np.ndarray, ew: np.ndarray, quarters: list):
    """Full report for one strategy."""
    active = port - ew

    sr_p, lo_p, hi_p = sharpe_ratio_bootstrap_ci(port)
    sr_ew, _, _ = sharpe_ratio_bootstrap_ci(ew)
    psr_p = probabilistic_sharpe_ratio(port)

    ir = information_ratio(port, ew)
    psr_active = probabilistic_sharpe_ratio(active, sr_benchmark=0.0)

    print(f"\n  {name}")
    print(f"  {'─' * 60}")
    print(f"  Portfolio:     Ann.Ret={annualised_return(port):+.1%}  "
          f"Vol={annualised_volatility(port):.1%}  "
          f"Sharpe={sr_p:+.2f} [{lo_p:+.2f},{hi_p:+.2f}]  "
          f"PSR={psr_p:.0%}")
    print(f"  Equal-weight:  Ann.Ret={annualised_return(ew):+.1%}  "
          f"Vol={annualised_volatility(ew):.1%}  "
          f"Sharpe={sr_ew:+.2f}")
    print(f"  Active return: Ann={np.mean(active)*4:+.1%}  "
          f"Vol={np.std(active, ddof=1)*2:.1%}  "
          f"IR={ir:+.2f}  "
          f"PSR(active)={psr_active:.0%}")
    print(f"  MaxDD(port):   {max_drawdown(port):.0%}  "
          f"MaxDD(active): {max_drawdown(active):.0%}  "
          f"Hit(active): {hit_rate(active):.0%}")
    print(f"  Quarters:      {len(quarters)}")

    return {
        "name": name,
        "ann_ret": annualised_return(port),
        "sharpe": sr_p,
        "sharpe_ci": (lo_p, hi_p),
        "psr": psr_p,
        "active_ann": np.mean(active) * 4,
        "ir": ir,
        "psr_active": psr_active,
        "max_dd": max_drawdown(port),
        "hit_active": hit_rate(active),
        "n_quarters": len(quarters),
        "port_returns": port,
        "ew_returns": ew,
        "active_returns": active,
        "quarters": quarters,
    }


def subsample_report(results: list[dict]):
    """Print subsample Sharpe and IR for each strategy."""
    print(f"\n  {'Strategy':<30s} {'Full':>6s} {'Pre-C':>6s} {'Post-C':>7s} "
          f"{'Full':>6s} {'Pre-C':>6s} {'Post-C':>7s}")
    print(f"  {'':30s} {'── Sharpe ──':>20s} {'── IR (active) ──':>20s}")
    print(f"  {'─' * 70}")

    for r in results:
        port = r["port_returns"]
        ew = r["ew_returns"]
        active = r["active_returns"]
        qs = r["quarters"]

        # Find split point
        split = pd.Timestamp("2020-01-01")
        pre_idx = [i for i, q in enumerate(qs) if q < split]
        post_idx = [i for i, q in enumerate(qs) if q >= split]

        sr_full = sharpe_ratio(port)
        sr_pre = sharpe_ratio(port[pre_idx]) if len(pre_idx) > 3 else np.nan
        sr_post = sharpe_ratio(port[post_idx]) if len(post_idx) > 3 else np.nan

        ir_full = information_ratio(port, ew)
        ir_pre = information_ratio(port[pre_idx], ew[pre_idx]) if len(pre_idx) > 3 else np.nan
        ir_post = information_ratio(port[post_idx], ew[post_idx]) if len(post_idx) > 3 else np.nan

        def fmt(v):
            return f"{v:+6.2f}" if np.isfinite(v) else f"{'n/a':>6s}"

        print(f"  {r['name']:<30s} {fmt(sr_full)} {fmt(sr_pre)} {fmt(sr_post)} "
              f"{fmt(ir_full)} {fmt(ir_pre)} {fmt(ir_post)}")


def main():
    predictions, returns = load_data()
    n_actors = returns["actor_id"].nunique()

    print("=" * 80)
    print("  FINAL ANALYSIS: G1-M2 DISAGREEMENT — LONG-ONLY STOCK SELECTION")
    print("=" * 80)
    print(f"\n  Universe: {n_actors} US firms, "
          f"{returns['quarter'].min().date()} to {returns['quarter'].max().date()}")

    results = []

    # ── Equal-weight baseline ──
    ew_full = returns.groupby("quarter")["return_q"].mean().sort_index().values
    sr_ew, _, _ = sharpe_ratio_bootstrap_ci(ew_full)
    print(f"\n  Equal-weight benchmark: Sharpe={sr_ew:+.2f}, "
          f"Ann.Ret={annualised_return(ew_full):+.1%}")

    # ── G1-M2 disagreement (raw) ──
    for N in [10, 15, 20]:
        sig = build_signal(predictions, acc_window=None)
        port, ew, qs = long_only_portfolio(sig, returns, N)
        r = report_strategy(f"G1-M2 disagr, top-{N}", port, ew, qs)
        results.append(r)

    # ── G1-M2 acc-scaled disagreement ──
    for N in [10, 15, 20]:
        sig = build_signal(predictions, acc_window=4)
        port, ew, qs = long_only_portfolio(sig, returns, N)
        r = report_strategy(f"G1-M2 acc-sc disagr, top-{N}", port, ew, qs)
        results.append(r)

    # ── G1-M2 acc-scaled, acc_window=3 (best from grid) ──
    sig = build_signal(predictions, acc_window=3)
    port, ew, qs = long_only_portfolio(sig, returns, 15)
    r = report_strategy("G1-M2 acc-sc(3Q) disagr, top-15", port, ew, qs)
    results.append(r)

    # ── Subsample analysis ──
    print("\n" + "=" * 80)
    print("  SUBSAMPLE STABILITY")
    print("=" * 80)
    subsample_report(results)

    # ── Per-year active returns ──
    print("\n" + "=" * 80)
    print("  PER-YEAR ACTIVE RETURNS (best config: G1-M2 acc-sc disagr, top-15)")
    print("=" * 80)

    best = next(r for r in results if "acc-sc disagr, top-15" in r["name"]
                and "3Q" not in r["name"])
    qs = best["quarters"]
    active = best["active_returns"]
    port = best["port_returns"]
    ew = best["ew_returns"]

    print(f"\n  {'Year':<8s} {'Port':>7s} {'EW':>7s} {'Active':>7s} {'Cum.Active':>11s}")
    print(f"  {'─' * 45}")
    cum = 1.0
    for q, p, e, a in zip(qs, port, ew, active):
        cum *= (1 + a)
        yr = pd.Timestamp(q).year
        qn = (pd.Timestamp(q).month - 1) // 3 + 1
        print(f"  {yr}Q{qn}   {p:+7.1%} {e:+7.1%} {a:+7.1%}    {cum - 1:+.1%}")

    # ── Statistical test: is active return significantly > 0? ──
    print("\n" + "=" * 80)
    print("  STATISTICAL TEST: IS ACTIVE RETURN > 0?")
    print("=" * 80)

    from scipy import stats as scipy_stats
    t_stat, p_val = scipy_stats.ttest_1samp(active, 0)
    print(f"\n  Mean active return (quarterly): {np.mean(active):+.2%}")
    print(f"  Std:  {np.std(active, ddof=1):.2%}")
    print(f"  t-stat: {t_stat:.3f}")
    print(f"  p-value (two-sided): {p_val:.3f}")
    print(f"  p-value (one-sided): {p_val/2:.3f}")
    print(f"  n quarters: {len(active)}")

    if p_val / 2 < 0.05:
        print(f"\n  Active return is statistically significant at 5% (one-sided)")
    else:
        print(f"\n  Active return is NOT statistically significant at 5%")
        print(f"  (need more data or stronger signal)")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
