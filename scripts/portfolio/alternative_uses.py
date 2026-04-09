"""Alternative ways to use the mixture architecture for returns.

Tests whether the architecture's edge translates to returns through
channels other than stock-level quintile sorting.
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
    max_drawdown, annualised_return, information_ratio, hit_rate,
)

DATA_DIR = PROJECT_ROOT / "results" / "portfolio"


def load_data():
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")
    common = set(predictions["actor_id"]) & set(returns["actor_id"])
    predictions = predictions[predictions["actor_id"].isin(common)]
    returns = returns[returns["actor_id"].isin(common)]
    return predictions, returns


def report(name: str, port: np.ndarray, bench: np.ndarray):
    if len(port) < 4:
        print(f"    {name:<45s}  n={len(port)} (too short)")
        return
    active = port - bench
    sr = sharpe_ratio(port)
    sr_b = sharpe_ratio(bench)
    ir = information_ratio(port, bench)
    psr_act = probabilistic_sharpe_ratio(active)
    mdd = max_drawdown(port)
    hr = hit_rate(active)
    print(f"    {name:<45s}  SR={sr:+.2f}(bm={sr_b:+.2f})  IR={ir:+.2f}  "
          f"PSR_act={psr_act:.0%}  DD={mdd:.0%}  Hit={hr:.0%}  n={len(port)}")


def main():
    predictions, returns = load_data()

    sector_map = predictions.groupby("actor_id")["sector"].first().to_dict()
    ret_actors = set(returns["actor_id"].unique())
    th_actors = [a for a, s in sector_map.items()
                 if s in ("technology", "healthcare") and a in ret_actors]
    remainder_actors = [a for a, s in sector_map.items()
                        if s not in ("technology", "healthcare") and a in ret_actors]

    print("=" * 90)
    print("  ALTERNATIVE USES OF THE MIXTURE ARCHITECTURE")
    print("=" * 90)
    print(f"\n  Tech/Health: {len(th_actors)} firms, Remainder: {len(remainder_actors)} firms")

    quarters = sorted(returns["quarter"].unique())

    # ════════════════════════════════════════════════════════════
    # 1. SECTOR ROTATION: overweight Tech/Health based on aggregate M2 signal
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print("  1. SECTOR ROTATION (aggregate block-level signal)")
    print(f"{'─' * 90}")
    print("     Signal = mean M2 predicted rank for Tech/Health block.")
    print("     High → Tech/Health is expected to invest more → overweight or underweight?\n")

    th_pred = predictions[predictions["actor_id"].isin(th_actors)]
    th_ret = returns[returns["actor_id"].isin(th_actors)]
    rem_ret = returns[returns["actor_id"].isin(remainder_actors)]

    # Compute per-quarter aggregate signals
    q_signal = th_pred.groupby("quarter").agg(
        mean_m2=("pred_m2", "mean"),
        mean_g1=("pred_g1", "mean"),
        mean_actual=("actual", "mean"),
        mean_disagr=("pred_m2", lambda x: 0),  # placeholder
    ).reset_index()
    # Compute disagreement properly
    th_pred_q = th_pred.copy()
    th_pred_q["disagr"] = th_pred_q["pred_g1"] - th_pred_q["pred_m2"]
    q_signal["mean_disagr"] = th_pred_q.groupby("quarter")["disagr"].mean().values

    # Compute per-quarter sector returns
    th_qret = th_ret.groupby("quarter")["return_q"].mean().reset_index().rename(
        columns={"return_q": "ret_th"})
    rem_qret = rem_ret.groupby("quarter")["return_q"].mean().reset_index().rename(
        columns={"return_q": "ret_rem"})
    ew_qret = returns.groupby("quarter")["return_q"].mean().reset_index().rename(
        columns={"return_q": "ret_ew"})

    merged = q_signal.merge(th_qret, on="quarter").merge(rem_qret, on="quarter").merge(
        ew_qret, on="quarter")
    merged = merged.sort_values("quarter")

    # Lag signals
    merged["lag_m2"] = merged["mean_m2"].shift(1)
    merged["lag_disagr"] = merged["mean_disagr"].shift(1)
    merged["lag_actual"] = merged["mean_actual"].shift(1)
    merged = merged.dropna()

    # Strategy: when lagged aggregate M2 is above median → overweight Tech/Health
    median_m2 = merged["lag_m2"].median()
    median_disagr = merged["lag_disagr"].median()

    for signal_name, signal_col, thresh in [
        ("M2 agg (high→OW TH)", "lag_m2", median_m2),
        ("M2 agg (low→OW TH)", "lag_m2", median_m2),
        ("G1-M2 agg disagr", "lag_disagr", median_disagr),
    ]:
        if "low→OW" in signal_name:
            ow_th = merged[signal_col] < thresh
        else:
            ow_th = merged[signal_col] >= thresh

        # 70/30 TH when overweight, 30/70 when underweight (vs 42/58 neutral)
        th_wt = np.where(ow_th, 0.70, 0.30)
        rem_wt = 1 - th_wt
        port = th_wt * merged["ret_th"].values + rem_wt * merged["ret_rem"].values
        bench = merged["ret_ew"].values
        report(signal_name, port, bench)

    # ════════════════════════════════════════════════════════════
    # 2. WITHIN TECH/HEALTH: M2 prediction as stock selection
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print("  2. WITHIN TECH/HEALTH: stock selection using M2 prediction")
    print(f"{'─' * 90}")
    print("     M2 has ΔR²=+0.249 in this block. Does prediction → returns?\n")

    th_pred_ret = th_pred.merge(th_ret, on=["quarter", "actor_id"], how="inner")
    th_pred_ret = th_pred_ret.sort_values(["actor_id", "quarter"])

    # Both directions
    for signal_name, score_expr in [
        ("M2 pred (high inv→long)", "pred_m2"),
        ("M2 pred (low inv→long)", "-pred_m2"),
        ("M2 pred change (ramp→long)", "delta_m2"),
        ("M2 pred change (cut→long)", "-delta_m2"),
        ("G1-M2 disagr within TH", "disagr"),
        ("M2-G1 disagr within TH", "-disagr"),
    ]:
        df = th_pred_ret.copy()
        df["disagr"] = df["pred_g1"] - df["pred_m2"]
        df["lag_actual"] = df.groupby("actor_id")["actual"].shift(1)
        df["delta_m2"] = df["pred_m2"] - df["lag_actual"]

        if score_expr.startswith("-"):
            df["score"] = -df[score_expr[1:]]
        else:
            df["score"] = df[score_expr]

        df = df.dropna(subset=["score"])
        port_ret, ew_ret = [], []
        for q in sorted(df["quarter"].unique()):
            qd = df[df["quarter"] == q]
            if len(qd) < 6:
                continue
            N = max(3, len(qd) // 3)
            ranked = qd.sort_values("score")
            port_ret.append(ranked.tail(N)["return_q"].mean())
            ew_ret.append(qd["return_q"].mean())

        report(signal_name, np.array(port_ret), np.array(ew_ret))

    # ════════════════════════════════════════════════════════════
    # 3. LOCAL RESIDUAL COMPONENT as signal
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print("  3. LOCAL RESIDUAL COMPONENT (what the local model adds)")
    print(f"{'─' * 90}")
    print("     For TH stocks: local_component = pred_m2 - pred_g1")
    print("     This is the unique information from block-specific estimation.\n")

    df = th_pred_ret.copy()
    df["local_component"] = df["pred_m2"] - df["pred_g1"]
    df["lag_local"] = df.groupby("actor_id")["local_component"].shift(1)

    for signal_name, score_col in [
        ("Local comp (high→long)", "local_component"),
        ("Local comp (low→long)", "local_component_neg"),
        ("Lagged local comp (high→long)", "lag_local"),
        ("Lagged local comp (low→long)", "lag_local_neg"),
    ]:
        dfc = df.copy()
        if "_neg" in score_col:
            dfc["score"] = -dfc[score_col.replace("_neg", "")]
        else:
            dfc["score"] = dfc[score_col]

        dfc = dfc.dropna(subset=["score"])
        port_ret, ew_ret = [], []
        for q in sorted(dfc["quarter"].unique()):
            qd = dfc[dfc["quarter"] == q]
            if len(qd) < 6:
                continue
            N = max(3, len(qd) // 3)
            ranked = qd.sort_values("score")
            port_ret.append(ranked.tail(N)["return_q"].mean())
            ew_ret.append(qd["return_q"].mean())

        report(signal_name, np.array(port_ret), np.array(ew_ret))

    # ════════════════════════════════════════════════════════════
    # 4. PREDICTION ACCURACY AS REGIME INDICATOR
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print("  4. REGIME INDICATOR (is the architecture 'working' this quarter?)")
    print(f"{'─' * 90}")
    print("     When M2 is more accurate than G1 (trailing), does TH outperform?\n")

    df = th_pred.copy()
    df["err_m2"] = (df["pred_m2"] - df["actual"]).abs()
    df["err_g1"] = (df["pred_g1"] - df["actual"]).abs()
    df["m2_wins"] = (df["err_m2"] < df["err_g1"]).astype(float)

    q_accuracy = df.groupby("quarter")["m2_wins"].mean().reset_index()
    q_accuracy = q_accuracy.sort_values("quarter")
    q_accuracy["lag_acc"] = q_accuracy["m2_wins"].rolling(4, min_periods=1).mean().shift(1)

    regime = q_accuracy.merge(th_qret, on="quarter").merge(rem_qret, on="quarter").merge(
        ew_qret, on="quarter").dropna()

    median_acc = regime["lag_acc"].median()

    # When M2 accuracy is high → TH block dynamics are predictable → overweight TH
    ow = regime["lag_acc"] >= median_acc
    port_ow = np.where(ow, 0.70, 0.30) * regime["ret_th"].values + \
              np.where(ow, 0.30, 0.70) * regime["ret_rem"].values
    report("High M2 acc → OW Tech/Health", port_ow, regime["ret_ew"].values)

    # When M2 accuracy is LOW → block dynamics are unstable → underweight TH
    port_uw = np.where(~ow, 0.70, 0.30) * regime["ret_th"].values + \
              np.where(~ow, 0.30, 0.70) * regime["ret_rem"].values
    report("Low M2 acc → OW Tech/Health", port_uw, regime["ret_ew"].values)

    # ════════════════════════════════════════════════════════════
    # 5. CORRELATION ANALYSIS: what DOES correlate with TH returns?
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─' * 90}")
    print("  5. CORRELATION: what predicts Tech/Health returns?")
    print(f"{'─' * 90}\n")

    corr_df = merged.copy()
    corr_df["fwd_th_ret"] = corr_df["ret_th"]  # already aligned (signal lagged)

    candidates = {
        "Lagged M2 agg prediction": "lag_m2",
        "Lagged G1-M2 agg disagreement": "lag_disagr",
        "Lagged actual agg rank": "lag_actual",
        "Lagged TH return": None,  # compute separately
    }

    corr_df["lag_th_ret"] = corr_df["ret_th"].shift(1)

    print(f"    {'Signal':<40s} {'Corr':>6s} {'p-val':>7s}")
    print(f"    {'─' * 55}")
    from scipy.stats import pearsonr
    for name, col in candidates.items():
        if col is None:
            col = "lag_th_ret"
        valid = corr_df[[col, "fwd_th_ret"]].dropna()
        if len(valid) < 5:
            continue
        r, p = pearsonr(valid[col], valid["fwd_th_ret"])
        sig = "*" if p < 0.05 else " "
        print(f"    {name:<40s} {r:+6.3f} {p:7.3f} {sig}")

    print(f"\n{'=' * 90}")
    print("  ANALYSIS COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
