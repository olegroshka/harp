"""Timing analysis: when is the signal formed vs when is the return earned?

Critical question: the local component (pred_m2 - pred_g1) at quarter Q
uses predictions made at the START of Q from data through Q-1.
The return at quarter Q is earned DURING Q.

So the contemporaneous signal is legitimate IF predictions are formed
before the quarter's return is realised. But we must verify:
  1. The prediction doesn't use any Q data
  2. The return is the FULL quarter return (not partial)
  3. Lagged versions (signal from Q-1, return at Q) still work

This script tests every timing variant systematically.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "portfolio"))

from metrics import (
    sharpe_ratio, probabilistic_sharpe_ratio, max_drawdown,
    information_ratio, hit_rate, annualised_return,
)

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def topN_portfolio(signal_df: pd.DataFrame, returns: pd.DataFrame, frac: float = 0.33):
    """Top-fraction portfolio, returns active and benchmark arrays."""
    merged = signal_df.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())
    port_ret, ew_ret = [], []
    for q in quarters:
        qd = merged[merged["quarter"] == q].dropna(subset=["score"])
        if len(qd) < 4:
            continue
        N = max(3, int(len(qd) * frac))
        ranked = qd.sort_values("score")
        port_ret.append(ranked.tail(N)["return_q"].mean())
        ew_ret.append(qd["return_q"].mean())
    return np.array(port_ret), np.array(ew_ret)


def report_line(name: str, port: np.ndarray, bench: np.ndarray):
    if len(port) < 4:
        print(f"    {name:<50s}  n={len(port)} (too short)")
        return
    active = port - bench
    sr = sharpe_ratio(port)
    ir = information_ratio(port, bench)
    psr = probabilistic_sharpe_ratio(active)
    t, p = ttest_1samp(active, 0)
    hr = hit_rate(active)
    print(f"    {name:<50s}  SR={sr:+.2f}  IR={ir:+.2f}  "
          f"PSR={psr:.0%}  t={t:+.2f}  p={p:.3f}  Hit={hr:.0%}  n={len(port)}")


def main():
    predictions, returns = load_data()

    sector_map = predictions.groupby("actor_id")["sector"].first().to_dict()
    ret_actors = set(returns["actor_id"].unique())
    th_actors = [a for a, s in sector_map.items()
                 if s in ("technology", "healthcare") and a in ret_actors]

    th_pred = predictions[predictions["actor_id"].isin(th_actors)].copy()
    th_ret = returns[returns["actor_id"].isin(th_actors)].copy()

    th_pred = th_pred.sort_values(["actor_id", "quarter"])
    th_pred["local_comp"] = th_pred["pred_m2"] - th_pred["pred_g1"]
    th_pred["lag_actual"] = th_pred.groupby("actor_id")["actual"].shift(1)
    th_pred["pred_change"] = th_pred["pred_m2"] - th_pred["lag_actual"]

    print("=" * 95)
    print("  TIMING ANALYSIS: TECH/HEALTH BLOCK")
    print("=" * 95)
    print(f"\n  Actors: {len(th_actors)}, Quarters: {th_ret['quarter'].nunique()}")

    # ════════════════════════════════════════════════════════════
    # 1. VERIFY PREDICTION TIMING
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  1. PREDICTION TIMING VERIFICATION")
    print(f"{'─' * 95}")
    print("     Predictions at Q are formed BEFORE observing Q's actual value.")
    print("     If correlation(pred_Q, actual_Q) ≈ correlation(pred_Q, actual_Q-1),")
    print("     then predictions aren't leaking contemporaneous data.\n")

    merged = th_pred.merge(th_ret, on=["quarter", "actor_id"], how="inner")

    # Correlation of prediction with SAME quarter actual
    r_same, p_same = pearsonr(merged["pred_m2"], merged["actual"])
    # Correlation of prediction with LAGGED actual
    merged_lag = merged.copy()
    merged_lag["lag_actual_1q"] = merged_lag.groupby("actor_id")["actual"].shift(1)
    valid = merged_lag.dropna(subset=["lag_actual_1q"])
    r_lag, p_lag = pearsonr(valid["pred_m2"], valid["lag_actual_1q"])

    print(f"    Corr(pred_m2_Q, actual_Q):   {r_same:+.4f}  (should be moderate — pred targets Q)")
    print(f"    Corr(pred_m2_Q, actual_Q-1): {r_lag:+.4f}  (should be similar — pred uses Q-1 data)")
    print(f"    If pred leaks Q data, first corr would be >> second.")
    print(f"    Ratio: {r_same / (r_lag + 1e-8):.2f}  (expect ~1.0 if no leak, >>1 if leak)")

    # Correlation of local component with SAME quarter return
    r_lc_ret, p_lc_ret = pearsonr(merged["local_comp"], merged["return_q"])
    print(f"\n    Corr(local_comp_Q, return_Q): {r_lc_ret:+.4f}  p={p_lc_ret:.3f}")
    print(f"    (This is the contemporaneous signal-return relationship)")

    # ════════════════════════════════════════════════════════════
    # 2. ALL TIMING VARIANTS OF LOCAL COMPONENT
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  2. LOCAL COMPONENT (M2-G1) — ALL TIMING VARIANTS")
    print(f"{'─' * 95}")
    print("     Signal_lag=0: signal at Q, return at Q (contemporaneous)")
    print("     Signal_lag=1: signal at Q-1, return at Q (1-quarter ahead)")
    print("     Signal_lag=2: signal at Q-2, return at Q (2-quarter ahead)\n")

    for lag in [0, 1, 2, 3]:
        df = th_pred.copy()
        if lag > 0:
            df["score"] = df.groupby("actor_id")["local_comp"].shift(lag)
        else:
            df["score"] = df["local_comp"]
        df = df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]

        port, ew = topN_portfolio(df, th_ret)
        report_line(f"Local comp, lag={lag}Q (high→long)", port, ew)

    # Reverse direction
    for lag in [0, 1, 2]:
        df = th_pred.copy()
        if lag > 0:
            df["score"] = -df.groupby("actor_id")["local_comp"].shift(lag)
        else:
            df["score"] = -df["local_comp"]
        df = df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]

        port, ew = topN_portfolio(df, th_ret)
        report_line(f"Local comp, lag={lag}Q (low→long)", port, ew)

    # ════════════════════════════════════════════════════════════
    # 3. ALL TIMING VARIANTS OF PREDICTED CHANGE
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  3. PREDICTED CHANGE (pred_m2 - lagged_actual) — ALL TIMING VARIANTS")
    print(f"{'─' * 95}\n")

    for lag in [0, 1, 2, 3]:
        df = th_pred.copy()
        if lag > 0:
            df["score"] = df.groupby("actor_id")["pred_change"].shift(lag)
        else:
            df["score"] = df["pred_change"]
        df = df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]

        port, ew = topN_portfolio(df, th_ret)
        report_line(f"Pred change (ramp→long), lag={lag}Q", port, ew)

    for lag in [0, 1, 2]:
        df = th_pred.copy()
        if lag > 0:
            df["score"] = -df.groupby("actor_id")["pred_change"].shift(lag)
        else:
            df["score"] = -df["pred_change"]
        df = df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]

        port, ew = topN_portfolio(df, th_ret)
        report_line(f"Pred change (cut→long), lag={lag}Q", port, ew)

    # ════════════════════════════════════════════════════════════
    # 4. M2 PREDICTION LEVEL — TIMING VARIANTS
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  4. M2 PREDICTION LEVEL — TIMING VARIANTS")
    print(f"{'─' * 95}\n")

    for lag in [0, 1, 2]:
        for direction, sign in [("high→long", 1), ("low→long", -1)]:
            df = th_pred.copy()
            if lag > 0:
                df["score"] = sign * df.groupby("actor_id")["pred_m2"].shift(lag)
            else:
                df["score"] = sign * df["pred_m2"]
            df = df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]

            port, ew = topN_portfolio(df, th_ret)
            report_line(f"M2 level ({direction}), lag={lag}Q", port, ew)

    # ════════════════════════════════════════════════════════════
    # 5. FORWARD-LOOKING LEAK TEST
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  5. LEAK TEST: use FUTURE return as signal (should show high IR)")
    print(f"{'─' * 95}")
    print("     If our machinery is correct, only this should show strong IR.\n")

    df = th_ret.copy()
    df = df.sort_values(["actor_id", "quarter"])
    # Perfect foresight (use same-quarter return)
    df["score"] = df["return_q"]
    df_pf = df[["quarter", "actor_id", "score"]].dropna()
    port, ew = topN_portfolio(df_pf, th_ret)
    report_line("PERFECT FORESIGHT (same Q return)", port, ew)

    # Use NEXT quarter return (future leak)
    df["score"] = df.groupby("actor_id")["return_q"].shift(-1)
    df_fl = df[["quarter", "actor_id", "score"]].dropna()
    port, ew = topN_portfolio(df_fl, th_ret)
    report_line("FUTURE LEAK (next Q return as signal)", port, ew)

    # Random signal
    rng = np.random.default_rng(42)
    df["score"] = rng.standard_normal(len(df))
    df_rand = df[["quarter", "actor_id", "score"]]
    port, ew = topN_portfolio(df_rand, th_ret)
    report_line("RANDOM SIGNAL", port, ew)

    # ════════════════════════════════════════════════════════════
    # 6. CORRELATION MATRIX: signal lags vs forward returns
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 95}")
    print("  6. CROSS-CORRELATION: local component lag vs forward return")
    print(f"{'─' * 95}\n")

    merged = th_pred.merge(th_ret, on=["quarter", "actor_id"], how="inner")
    merged = merged.sort_values(["actor_id", "quarter"])

    print(f"    {'Signal lag':<20s} {'Corr':>7s} {'p-val':>7s} {'Sig?':>5s}")
    print(f"    {'─' * 45}")

    for lag in range(-2, 5):
        mc = merged.copy()
        if lag >= 0:
            mc["sig"] = mc.groupby("actor_id")["local_comp"].shift(lag)
            label = f"local_comp(t-{lag})" if lag > 0 else "local_comp(t)"
        else:
            # Future signal (leak test)
            mc["sig"] = mc.groupby("actor_id")["local_comp"].shift(lag)
            label = f"local_comp(t+{-lag}) [FUTURE]"

        valid = mc.dropna(subset=["sig"])
        if len(valid) < 10:
            continue
        r, p = pearsonr(valid["sig"], valid["return_q"])
        sig = "*" if p < 0.05 else ("." if p < 0.10 else " ")
        print(f"    {label:<20s} {r:+7.4f} {p:7.3f}   {sig}")

    print(f"\n{'=' * 95}")
    print("  TIMING ANALYSIS COMPLETE")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
