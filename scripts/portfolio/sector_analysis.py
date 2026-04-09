"""G1-M2 disagreement portfolio analysis by sector.

Tests whether the architecture disagreement signal works differently
across the block structure identified in the paper.
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


def build_signal(predictions: pd.DataFrame, acc_window: int | None = 4) -> pd.DataFrame:
    """G1-M2 disagreement, optionally accuracy-scaled."""
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "actual", "sector"]].copy()
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

    return df


def run_sector_portfolio(
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    actor_subset: list[str] | None = None,
    top_frac: float = 0.4,
):
    """Long-only top-fraction portfolio for a sector subset.

    Uses top_frac instead of fixed N since sector sizes vary.
    """
    sig = signal_df[["quarter", "actor_id", "score"]].dropna()
    if actor_subset is not None:
        sig = sig[sig["actor_id"].isin(actor_subset)]

    merged = sig.merge(returns, on=["quarter", "actor_id"], how="inner")
    quarters = sorted(merged["quarter"].unique())

    port_ret, ew_ret = [], []
    for q in quarters:
        qd = merged[merged["quarter"] == q]
        if len(qd) < 3:
            continue
        N = max(2, int(len(qd) * top_frac))
        ranked = qd.sort_values("score")
        port_ret.append(ranked.tail(N)["return_q"].mean())
        ew_ret.append(qd["return_q"].mean())

    return np.array(port_ret), np.array(ew_ret), len(quarters)


def report(name: str, port: np.ndarray, ew: np.ndarray, n_stocks: int):
    """Print one-line report."""
    if len(port) < 4:
        print(f"  {name:<35s}  n_q={len(port):2d}  (too short)")
        return

    active = port - ew
    sr_p = sharpe_ratio(port)
    sr_ew = sharpe_ratio(ew)
    psr = probabilistic_sharpe_ratio(port)
    ir = information_ratio(port, ew) if len(port) == len(ew) else np.nan
    mdd_p = max_drawdown(port)
    mdd_ew = max_drawdown(ew)
    psr_act = probabilistic_sharpe_ratio(active)
    ann_act = np.mean(active) * 4

    print(f"  {name:<35s}  N={n_stocks:2d}  "
          f"SR={sr_p:+.2f} (ew={sr_ew:+.2f})  "
          f"IR={ir:+.2f}  PSR_act={psr_act:.0%}  "
          f"DD={mdd_p:.0%}(ew={mdd_ew:.0%})  "
          f"Act={ann_act:+.1%}")


def main():
    predictions, returns = load_data()

    # Get sector mapping
    sector_map = predictions.groupby("actor_id")["sector"].first().to_dict()
    actors_by_sector = {}
    for actor, sector in sector_map.items():
        actors_by_sector.setdefault(sector, []).append(actor)

    print("=" * 90)
    print("  G1-M2 DISAGREEMENT: SECTOR-LEVEL ANALYSIS")
    print("=" * 90)

    print(f"\n  Sector composition (US firms with return data):")
    for sec in sorted(actors_by_sector.keys()):
        n = len(actors_by_sector[sec])
        actors = actors_by_sector[sec]
        # Check how many have returns
        ret_actors = set(returns["actor_id"].unique())
        matched = [a for a in actors if a in ret_actors]
        print(f"    {sec:<20s}: {len(matched):2d} firms")

    # ── Paper block groups ──
    block_groups = {
        "Tech/Health": ["technology", "healthcare"],
        "Diversified": ["diversified"],
        "Energy": ["energy"],
        "Financials": ["financials"],
        "Industrials": ["industrials"],
        "Remainder (E+F+I)": ["energy", "financials", "industrials"],
    }

    ret_actors = set(returns["actor_id"].unique())

    # ── Raw G1-M2 disagreement by sector ──
    print(f"\n{'─' * 90}")
    print("  G1-M2 disagr (raw) — by sector")
    print(f"{'─' * 90}")

    sig_raw = build_signal(predictions, acc_window=None)

    for group_name, sectors in block_groups.items():
        actors = [a for sec in sectors for a in actors_by_sector.get(sec, []) if a in ret_actors]
        if len(actors) < 3:
            print(f"  {group_name:<35s}  {len(actors)} firms (too few)")
            continue
        port, ew, nq = run_sector_portfolio(sig_raw, returns, actors)
        report(f"G1-M2 disagr / {group_name}", port, ew, len(actors))

    # Full universe for comparison
    port_all, ew_all, _ = run_sector_portfolio(sig_raw, returns)
    report("G1-M2 disagr / ALL", port_all, ew_all, len(ret_actors))

    # ── Accuracy-scaled by sector ──
    print(f"\n{'─' * 90}")
    print("  G1-M2 acc-sc disagr — by sector")
    print(f"{'─' * 90}")

    sig_acc = build_signal(predictions, acc_window=4)

    for group_name, sectors in block_groups.items():
        actors = [a for sec in sectors for a in actors_by_sector.get(sec, []) if a in ret_actors]
        if len(actors) < 3:
            print(f"  {group_name:<35s}  {len(actors)} firms (too few)")
            continue
        port, ew, nq = run_sector_portfolio(sig_acc, returns, actors)
        report(f"G1-M2 acc-sc / {group_name}", port, ew, len(actors))

    port_all, ew_all, _ = run_sector_portfolio(sig_acc, returns)
    report("G1-M2 acc-sc / ALL", port_all, ew_all, len(ret_actors))

    # ── Per-sector: does the architecture add value? ──
    print(f"\n{'─' * 90}")
    print("  ARCHITECTURE VALUE: sector-level M2 vs G1 prediction accuracy")
    print(f"{'─' * 90}")
    print(f"\n  {'Sector':<20s} {'N':>3s} {'M2 R²':>7s} {'G1 R²':>7s} {'ΔR²':>7s} "
          f"{'M2>G1':>6s} {'|disagr|':>8s}")
    print(f"  {'─' * 65}")

    from harp.validation.metrics import oos_r_squared
    for group_name, sectors in sorted(block_groups.items()):
        actors = [a for sec in sectors for a in actors_by_sector.get(sec, []) if a in ret_actors]
        if not actors:
            continue
        sub = predictions[predictions["actor_id"].isin(actors)]
        r2_m2 = oos_r_squared(sub["pred_m2"].values, sub["actual"].values)
        r2_g1 = oos_r_squared(sub["pred_g1"].values, sub["actual"].values)
        mean_abs_disagr = (sub["pred_g1"] - sub["pred_m2"]).abs().mean()
        # Per-quarter M2 wins
        wins = 0
        total = 0
        for q in sub["quarter"].unique():
            qd = sub[sub["quarter"] == q]
            r2_m2_q = oos_r_squared(qd["pred_m2"].values, qd["actual"].values)
            r2_g1_q = oos_r_squared(qd["pred_g1"].values, qd["actual"].values)
            if r2_m2_q > r2_g1_q:
                wins += 1
            total += 1
        print(f"  {group_name:<20s} {len(actors):3d} {r2_m2:7.3f} {r2_g1:7.3f} "
              f"{r2_m2 - r2_g1:+7.3f} {wins}/{total:>3d} {mean_abs_disagr:8.4f}")

    # ── Subsample by sector ──
    print(f"\n{'─' * 90}")
    print("  SUBSAMPLE: G1-M2 acc-sc disagr Sharpe by sector × period")
    print(f"{'─' * 90}")
    print(f"\n  {'Group':<25s} {'Full':>6s} {'Pre-C':>6s} {'Post-C':>7s}")
    print(f"  {'─' * 50}")

    for group_name, sectors in block_groups.items():
        actors = [a for sec in sectors for a in actors_by_sector.get(sec, []) if a in ret_actors]
        if len(actors) < 3:
            continue
        sig_sub = sig_acc[sig_acc["actor_id"].isin(actors)]

        sharpes = {}
        for period, start, end in [
            ("Full", "2015-01-01", "2024-12-31"),
            ("Pre-C", "2015-01-01", "2019-12-31"),
            ("Post-C", "2020-01-01", "2024-12-31"),
        ]:
            s = sig_sub[(sig_sub["quarter"] >= start) & (sig_sub["quarter"] <= end)]
            r = returns[(returns["quarter"] >= start) & (returns["quarter"] <= end)]
            r = r[r["actor_id"].isin(actors)]
            port, ew, _ = run_sector_portfolio(s, r, actors)
            sharpes[period] = sharpe_ratio(port) if len(port) > 3 else np.nan

        def fmt(v):
            return f"{v:+6.2f}" if np.isfinite(v) else f"{'n/a':>6s}"
        print(f"  {group_name:<25s} {fmt(sharpes['Full'])} "
              f"{fmt(sharpes['Pre-C'])} {fmt(sharpes['Post-C'])}")

    print(f"\n{'=' * 90}")
    print("  SECTOR ANALYSIS COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
