"""Deep dive into G1-M2 disagreement signal variants.

Focused exploration of the accuracy-scaled disagreement signal:
  - Asymmetric decomposition (long-only vs short-only)
  - Persistence filter (2+ quarters same direction)
  - Concentrated portfolios (top/bottom N instead of quintiles)
  - Accuracy window sensitivity (2Q, 4Q, 8Q)
  - Volatility regime scaling
  - Lagged disagreement
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

from metrics import sharpe_ratio, sharpe_ratio_bootstrap_ci, probabilistic_sharpe_ratio, max_drawdown, hit_rate

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def build_base_signal(predictions: pd.DataFrame, acc_window: int = 4) -> pd.DataFrame:
    """Build accuracy-scaled G1-M2 disagreement signal."""
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]

    # Accuracy scaling
    df["err_g1"] = (df["pred_g1"] - df["actual"]).abs()
    df["err_m2"] = (df["pred_m2"] - df["actual"]).abs()
    df["m2_better"] = (df["err_m2"] < df["err_g1"]).astype(float)
    df["m2_hit"] = df.groupby("actor_id")["m2_better"].transform(
        lambda x: x.rolling(acc_window, min_periods=1).mean().shift(1)
    )
    df["score"] = df["raw_disagr"] * df["m2_hit"].fillna(0.5)
    return df


def topN_long_short(signal_df: pd.DataFrame, returns: pd.DataFrame, N: int):
    """Top-N / Bottom-N long-short portfolio (equal weight)."""
    merged = signal_df.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())
    ls_returns = []
    for q in quarters:
        qd = merged[merged["quarter"] == q].dropna(subset=["score"])
        if len(qd) < 2 * N:
            continue
        ranked = qd.sort_values("score")
        short_ret = ranked.head(N)["return_q"].mean()
        long_ret = ranked.tail(N)["return_q"].mean()
        ls_returns.append(long_ret - short_ret)
    return np.array(ls_returns)


def long_only_topN(signal_df: pd.DataFrame, returns: pd.DataFrame, N: int):
    """Top-N long-only portfolio returns."""
    merged = signal_df.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())
    long_returns = []
    ew_returns = []
    for q in quarters:
        qd = merged[merged["quarter"] == q].dropna(subset=["score"])
        if len(qd) < N:
            continue
        ranked = qd.sort_values("score")
        long_returns.append(ranked.tail(N)["return_q"].mean())
        ew_returns.append(qd["return_q"].mean())
    return np.array(long_returns), np.array(ew_returns)


def short_only_bottomN(signal_df: pd.DataFrame, returns: pd.DataFrame, N: int):
    """Bottom-N short-only portfolio returns (negative = profit on short)."""
    merged = signal_df.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())
    short_returns = []
    ew_returns = []
    for q in quarters:
        qd = merged[merged["quarter"] == q].dropna(subset=["score"])
        if len(qd) < N:
            continue
        ranked = qd.sort_values("score")
        short_returns.append(-ranked.head(N)["return_q"].mean())  # profit from shorting
        ew_returns.append(qd["return_q"].mean())
    return np.array(short_returns), np.array(ew_returns)


def report(name: str, returns: np.ndarray, indent: str = "    "):
    """Print metrics for a return series."""
    if len(returns) < 4:
        print(f"{indent}{name:<35s}  n={len(returns)} (too short)")
        return
    sr, lo, hi = sharpe_ratio_bootstrap_ci(returns)
    psr = probabilistic_sharpe_ratio(returns)
    mdd = max_drawdown(returns)
    hr = hit_rate(returns)
    ann = float((np.prod(1 + returns)) ** (4 / len(returns)) - 1)
    print(f"{indent}{name:<35s}  Sharpe={sr:+.2f} [{lo:+.2f},{hi:+.2f}] "
          f"PSR={psr:.0%} MaxDD={mdd:.0%} Hit={hr:.0%} Ann={ann:+.1%} n={len(returns)}")


def main():
    predictions, returns = load_data()

    print("=" * 80)
    print("  DEEP DIVE: G1-M2 DISAGREEMENT SIGNAL")
    print("=" * 80)

    # ── 1. Asymmetric decomposition ──
    print("\n  1. ASYMMETRIC DECOMPOSITION (which side drives returns?)")
    print("     Long = top-N highest score, Short = bottom-N lowest score\n")

    sig = build_base_signal(predictions, acc_window=4)
    sig_simple = sig[["quarter", "actor_id", "score"]].dropna()

    for N in [6, 10, 15, 20]:
        ls = topN_long_short(sig_simple, returns, N)
        lo, ew = long_only_topN(sig_simple, returns, N)
        sh, _ = short_only_bottomN(sig_simple, returns, N)

        report(f"L/S top/bot-{N}", ls)
        report(f"Long-only top-{N}", lo)
        report(f"Short-only bot-{N} (neg ret)", sh)
        print()

    # ── 2. Persistence filter ──
    print("\n  2. PERSISTENCE FILTER (same direction 2+ quarters)")

    sig_pers = sig.copy()
    sig_pers["sign"] = np.sign(sig_pers["raw_disagr"])
    sig_pers["prev_sign"] = sig_pers.groupby("actor_id")["sign"].shift(1)
    sig_pers["persistent"] = sig_pers["sign"] == sig_pers["prev_sign"]
    sig_pers["score_pers"] = np.where(sig_pers["persistent"], sig_pers["score"], 0.0)
    sig_pers_df = sig_pers[["quarter", "actor_id", "score_pers"]].rename(
        columns={"score_pers": "score"}).dropna()

    for N in [6, 10, 15]:
        ls = topN_long_short(sig_pers_df, returns, N)
        report(f"Persistent L/S top/bot-{N}", ls)

    # ── 3. Accuracy window sensitivity ──
    print("\n\n  3. ACCURACY WINDOW SENSITIVITY")

    for window in [2, 3, 4, 6, 8]:
        sig_w = build_base_signal(predictions, acc_window=window)
        sig_w_df = sig_w[["quarter", "actor_id", "score"]].dropna()
        ls = topN_long_short(sig_w_df, returns, 10)
        report(f"AccWindow={window}Q, L/S top/bot-10", ls)

    # ── 4. Concentrated vs broad ──
    print("\n\n  4. PORTFOLIO CONCENTRATION")

    for N in [3, 5, 6, 8, 10, 12, 15, 20, 25]:
        ls = topN_long_short(sig_simple, returns, N)
        sr = sharpe_ratio(ls) if len(ls) > 3 else np.nan
        print(f"    N={N:2d}: Sharpe={sr:+.2f}, n_quarters={len(ls)}")

    # ── 5. Volatility regime scaling ──
    print("\n\n  5. VOLATILITY REGIME SCALING")

    # Market-level quarterly vol
    ew_q = returns.groupby("quarter")["return_q"].mean().sort_index()
    ew_vol = ew_q.rolling(4).std()
    median_vol = ew_vol.median()
    vol_scale = median_vol / (ew_vol + 1e-8)
    vol_scale = vol_scale.clip(0.5, 2.0)  # bound scaling

    sig_vol = sig_simple.merge(
        vol_scale.reset_index().rename(columns={"return_q": "vol_scale"}),
        on="quarter", how="left"
    )
    sig_vol["score"] = sig_vol["score"] * sig_vol["vol_scale"].fillna(1.0)
    sig_vol_df = sig_vol[["quarter", "actor_id", "score"]]

    for N in [6, 10, 15]:
        ls = topN_long_short(sig_vol_df, returns, N)
        report(f"VolScaled L/S top/bot-{N}", ls)

    # ── 6. Lagged disagreement ──
    print("\n\n  6. LAGGED DISAGREEMENT (does signal need time to work?)")

    sig_lag = sig.copy()
    sig_lag["score_lag1"] = sig_lag.groupby("actor_id")["score"].shift(1)
    sig_lag1 = sig_lag[["quarter", "actor_id", "score_lag1"]].rename(
        columns={"score_lag1": "score"}).dropna()

    for N in [6, 10, 15]:
        ls_lag = topN_long_short(sig_lag1, returns, N)
        ls_raw = topN_long_short(sig_simple, returns, N)
        report(f"Lagged-1Q L/S top/bot-{N}", ls_lag)
        report(f"Raw       L/S top/bot-{N}", ls_raw)
        print()

    # ── 7. Best configuration summary ──
    print("\n" + "=" * 80)
    print("  BEST CONFIGURATIONS")
    print("=" * 80)

    configs = []
    for window in [2, 3, 4, 6, 8]:
        for N in [6, 8, 10, 12, 15]:
            sig_w = build_base_signal(predictions, acc_window=window)
            sig_w_df = sig_w[["quarter", "actor_id", "score"]].dropna()
            ls = topN_long_short(sig_w_df, returns, N)
            if len(ls) > 3:
                sr = sharpe_ratio(ls)
                psr = probabilistic_sharpe_ratio(ls)
                mdd = max_drawdown(ls)
                configs.append((window, N, sr, psr, mdd, len(ls)))

    configs.sort(key=lambda x: -x[2])
    print(f"\n    {'AccW':>4s} {'N':>3s} {'Sharpe':>7s} {'PSR':>6s} {'MaxDD':>6s} {'nQ':>4s}")
    print(f"    {'-' * 35}")
    for w, n, sr, psr, mdd, nq in configs[:15]:
        print(f"    {w:4d} {n:3d} {sr:+7.2f} {psr:5.0%} {mdd:6.0%} {nq:4d}")


if __name__ == "__main__":
    main()
