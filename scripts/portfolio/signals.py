"""Signal generation for the portfolio backtest.

Each signal function takes predictions and/or returns and produces
a cross-sectional score per actor per quarter. Higher score = more
desirable for the long side.

All signals return a DataFrame with columns: quarter, actor_id, score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_from_predictions(
    predictions: pd.DataFrame,
    pred_column: str,
) -> pd.DataFrame:
    """Score = predicted intensity rank (higher predicted rank = higher score).

    The investment factor literature (Hou-Xue-Zhang 2015) shows that
    conservative investors (low CapEx/Assets) tend to outperform.
    So we REVERSE the signal: low predicted intensity → high score.
    """
    df = predictions[["quarter", "actor_id", pred_column]].copy()
    # Reverse: firms predicted to have LOW investment intensity get HIGH score
    df["score"] = -df[pred_column]
    return df[["quarter", "actor_id", "score"]]


def signal_m2(predictions: pd.DataFrame) -> pd.DataFrame:
    """M2 (mixture PCA+ridge) signal."""
    return signal_from_predictions(predictions, "pred_m2")


def signal_g1(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1 (global augmentation) signal."""
    return signal_from_predictions(predictions, "pred_g1")


def signal_g0(predictions: pd.DataFrame) -> pd.DataFrame:
    """G0 (pooled-only) signal."""
    return signal_from_predictions(predictions, "pred_g0")


def signal_ar1(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-actor AR(1) signal."""
    return signal_from_predictions(predictions, "pred_ar1")


def signal_naive_investment(predictions: pd.DataFrame) -> pd.DataFrame:
    """Naive investment factor: sort by LAGGED actual CapEx/Assets rank.

    Uses the actual value from the prior quarter (no model needed).
    Lower lagged intensity → higher score (conservative investor premium).
    """
    df = predictions[["quarter", "actor_id", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["lagged_actual"] = df.groupby("actor_id")["actual"].shift(1)
    df["score"] = -df["lagged_actual"]
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_momentum(returns: pd.DataFrame) -> pd.DataFrame:
    """12-month momentum: trailing 4-quarter cumulative return, LAGGED by 1Q.

    Signal at quarter Q uses returns from Q-5 through Q-1 (not Q itself).
    This avoids look-ahead bias from including the current quarter's return.
    """
    df = returns[["quarter", "actor_id", "return_q"]].copy()
    df = df.sort_values(["actor_id", "quarter"])

    # Rolling 4-quarter cumulative return (compounded), then lag by 1
    df["cum_4q"] = (
        df.groupby("actor_id")["return_q"]
        .transform(lambda x: (1 + x).rolling(4).apply(np.prod, raw=True) - 1)
    )
    # Lag: use LAST quarter's momentum as THIS quarter's signal
    df["score"] = df.groupby("actor_id")["cum_4q"].shift(1)
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_equal_weight() -> str:
    """Marker for equal-weight (no signal). Handled in portfolio construction."""
    return "equal_weight"


# ── Creative signals ─────────────────────────────────────────────

def signal_rank_change_m2(predictions: pd.DataFrame) -> pd.DataFrame:
    """Predicted rank CHANGE: pred_m2 - lagged_actual.

    Firms predicted to CUT investment (negative delta) → high score
    (conservative investor premium on predicted changes, not levels).
    """
    df = predictions[["quarter", "actor_id", "pred_m2", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["lagged_actual"] = df.groupby("actor_id")["actual"].shift(1)
    df["delta"] = df["pred_m2"] - df["lagged_actual"]
    df["score"] = -df["delta"]  # predicted CUT → high score
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_rank_change_m2_positive(predictions: pd.DataFrame) -> pd.DataFrame:
    """Predicted rank CHANGE, POSITIVE direction.

    Firms predicted to INCREASE investment → high score.
    Rationale: firms ramping up see good opportunities → future growth.
    """
    df = predictions[["quarter", "actor_id", "pred_m2", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["lagged_actual"] = df.groupby("actor_id")["actual"].shift(1)
    df["delta"] = df["pred_m2"] - df["lagged_actual"]
    df["score"] = df["delta"]  # predicted RAMP UP → high score
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_disagreement(predictions: pd.DataFrame) -> pd.DataFrame:
    """M2-G1 disagreement: where block-specific dynamics diverge from global.

    score = pred_m2 - pred_g1.
    Positive: mixture sees more investment than global model predicts
    (block-specific upward dynamic). We go long these — the local
    information the global model misses may reflect real opportunities.
    """
    df = predictions[["quarter", "actor_id", "pred_m2", "pred_g1"]].copy()
    df["score"] = df["pred_m2"] - df["pred_g1"]
    return df[["quarter", "actor_id", "score"]]


def signal_disagreement_reversed(predictions: pd.DataFrame) -> pd.DataFrame:
    """M2-G1 disagreement, reversed sign."""
    df = predictions[["quarter", "actor_id", "pred_m2", "pred_g1"]].copy()
    df["score"] = df["pred_g1"] - df["pred_m2"]
    return df[["quarter", "actor_id", "score"]]


def signal_abs_disagreement(predictions: pd.DataFrame) -> pd.DataFrame:
    """Absolute M2-G1 disagreement: where the architecture matters most.

    High |disagreement| → the block-specific model sees something the global
    model doesn't. Go long the stocks where the architecture edge is largest,
    regardless of direction. Short the stocks where models agree (no edge).
    """
    df = predictions[["quarter", "actor_id", "pred_m2", "pred_g1"]].copy()
    df["score"] = np.abs(df["pred_m2"] - df["pred_g1"])
    return df[["quarter", "actor_id", "score"]]


def signal_sector_neutral_m2(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sector-neutral M2: rank within sector, then sort.

    Removes sector bets, isolates within-sector stock selection.
    Uses negative predicted rank (conservative premium) within each sector.
    """
    df = predictions[["quarter", "actor_id", "pred_m2", "sector"]].copy()
    # Within-sector z-score of predicted rank each quarter
    df["score"] = (
        df.groupby(["quarter", "sector"])["pred_m2"]
        .transform(lambda x: -(x - x.mean()) / (x.std() + 1e-8))
    )
    return df[["quarter", "actor_id", "score"]]


def signal_m2_positive(predictions: pd.DataFrame) -> pd.DataFrame:
    """M2 signal, POSITIVE direction (high predicted intensity → high score).

    Opposite of the conservative premium hypothesis.
    Test: maybe high-investing firms outperform in this universe/period?
    """
    df = predictions[["quarter", "actor_id", "pred_m2"]].copy()
    df["score"] = df["pred_m2"]
    return df[["quarter", "actor_id", "score"]]


def signal_g1_positive(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1 signal, POSITIVE direction."""
    df = predictions[["quarter", "actor_id", "pred_g1"]].copy()
    df["score"] = df["pred_g1"]
    return df[["quarter", "actor_id", "score"]]


# ── Momentum + prediction combinations ──────────────────────────

def _zscore_within_quarter(df: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional z-score within each quarter."""
    return df.groupby("quarter")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


def signal_mom_plus_disagr(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Momentum + G1-M2 disagreement (equal-weighted z-score blend).

    Combines the strongest return signal (momentum) with the strongest
    prediction-based signal (G1-M2 disagreement).
    """
    mom = signal_momentum(returns)
    dis = signal_disagreement_reversed(predictions)

    merged = mom.merge(dis, on=["quarter", "actor_id"], suffixes=("_mom", "_dis"))
    merged["z_mom"] = _zscore_within_quarter(merged, "score_mom")
    merged["z_dis"] = _zscore_within_quarter(merged, "score_dis")
    merged["score"] = 0.5 * merged["z_mom"] + 0.5 * merged["z_dis"]
    return merged[["quarter", "actor_id", "score"]]


def signal_mom_plus_m2pos(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Momentum + M2 positive (high predicted investment → long).

    Tilt momentum toward firms the mixture model predicts will invest heavily.
    """
    mom = signal_momentum(returns)
    m2p = signal_m2_positive(predictions)

    merged = mom.merge(m2p, on=["quarter", "actor_id"], suffixes=("_mom", "_m2"))
    merged["z_mom"] = _zscore_within_quarter(merged, "score_mom")
    merged["z_m2"] = _zscore_within_quarter(merged, "score_m2")
    merged["score"] = 0.5 * merged["z_mom"] + 0.5 * merged["z_m2"]
    return merged[["quarter", "actor_id", "score"]]


def signal_mom_plus_delta(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Momentum + predicted rank change (positive direction).

    Momentum + firms predicted to ramp up investment.
    """
    mom = signal_momentum(returns)
    delta = signal_rank_change_m2_positive(predictions)

    merged = mom.merge(delta, on=["quarter", "actor_id"], suffixes=("_mom", "_delta"))
    merged["z_mom"] = _zscore_within_quarter(merged, "score_mom")
    merged["z_delta"] = _zscore_within_quarter(merged, "score_delta")
    merged["score"] = 0.5 * merged["z_mom"] + 0.5 * merged["z_delta"]
    return merged[["quarter", "actor_id", "score"]]


def signal_mom_confirmed(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Momentum filtered by prediction agreement.

    Take momentum positions only when M2 predicted change agrees
    with the momentum direction (both positive or both negative).
    Disagreements get score = 0 (neutral).
    """
    mom = signal_momentum(returns)
    delta = signal_rank_change_m2_positive(predictions)

    merged = mom.merge(delta, on=["quarter", "actor_id"], suffixes=("_mom", "_delta"))
    # Agreement: momentum and predicted change have the same sign
    agrees = np.sign(merged["score_mom"]) == np.sign(merged["score_delta"])
    merged["score"] = np.where(agrees, merged["score_mom"], 0.0)
    return merged[["quarter", "actor_id", "score"]]


def signal_mom_tilt_70_30(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """70% momentum + 30% G1-M2 disagreement."""
    mom = signal_momentum(returns)
    dis = signal_disagreement_reversed(predictions)

    merged = mom.merge(dis, on=["quarter", "actor_id"], suffixes=("_mom", "_dis"))
    merged["z_mom"] = _zscore_within_quarter(merged, "score_mom")
    merged["z_dis"] = _zscore_within_quarter(merged, "score_dis")
    merged["score"] = 0.7 * merged["z_mom"] + 0.3 * merged["z_dis"]
    return merged[["quarter", "actor_id", "score"]]


# ── Price-based signals from OHLCV ──────────────────────────────

def signal_reversal_1q(returns: pd.DataFrame) -> pd.DataFrame:
    """1-quarter reversal: last quarter's losers → this quarter's winners.

    Short-term reversal is well-documented (Jegadeesh 1990).
    Score = -lagged_1q_return.
    """
    df = returns[["quarter", "actor_id", "return_q"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["score"] = -df.groupby("actor_id")["return_q"].shift(1)
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_low_volatility(returns: pd.DataFrame) -> pd.DataFrame:
    """Low realised volatility: trailing 4-quarter return std, LAGGED.

    Low-vol anomaly (Ang et al. 2006): low-vol stocks outperform risk-adjusted.
    Score = -lagged_vol (lower vol → higher score).
    """
    df = returns[["quarter", "actor_id", "return_q"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["vol_4q"] = (
        df.groupby("actor_id")["return_q"]
        .transform(lambda x: x.rolling(4).std())
    )
    # Lag by 1 to avoid look-ahead
    df["score"] = -df.groupby("actor_id")["vol_4q"].shift(1)
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


# ── Combinations with G1-M2 disagreement ───────────────────────

def signal_rev_plus_disagr(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Reversal + G1-M2 disagreement (50/50 z-score blend)."""
    rev = signal_reversal_1q(returns)
    dis = signal_disagreement_reversed(predictions)
    merged = rev.merge(dis, on=["quarter", "actor_id"], suffixes=("_rev", "_dis"))
    merged["z_rev"] = _zscore_within_quarter(merged, "score_rev")
    merged["z_dis"] = _zscore_within_quarter(merged, "score_dis")
    merged["score"] = 0.5 * merged["z_rev"] + 0.5 * merged["z_dis"]
    return merged[["quarter", "actor_id", "score"]]


def signal_lowvol_plus_disagr(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Low-vol + G1-M2 disagreement (50/50 z-score blend)."""
    lv = signal_low_volatility(returns)
    dis = signal_disagreement_reversed(predictions)
    merged = lv.merge(dis, on=["quarter", "actor_id"], suffixes=("_lv", "_dis"))
    merged["z_lv"] = _zscore_within_quarter(merged, "score_lv")
    merged["z_dis"] = _zscore_within_quarter(merged, "score_dis")
    merged["score"] = 0.5 * merged["z_lv"] + 0.5 * merged["z_dis"]
    return merged[["quarter", "actor_id", "score"]]


def signal_rev_lowvol_disagr(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Three-way: reversal + low-vol + G1-M2 disagreement (equal z-score)."""
    rev = signal_reversal_1q(returns)
    lv = signal_low_volatility(returns)
    dis = signal_disagreement_reversed(predictions)

    merged = rev.merge(lv, on=["quarter", "actor_id"], suffixes=("_rev", "_lv"))
    merged = merged.merge(dis, on=["quarter", "actor_id"])
    merged = merged.rename(columns={"score": "score_dis"})
    merged["z_rev"] = _zscore_within_quarter(merged, "score_rev")
    merged["z_lv"] = _zscore_within_quarter(merged, "score_lv")
    merged["z_dis"] = _zscore_within_quarter(merged, "score_dis")
    merged["score"] = (merged["z_rev"] + merged["z_lv"] + merged["z_dis"]) / 3
    return merged[["quarter", "actor_id", "score"]]


def signal_rev_plus_m2pos(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Reversal + M2 positive (50/50 z-score blend)."""
    rev = signal_reversal_1q(returns)
    m2p = signal_m2_positive(predictions)
    merged = rev.merge(m2p, on=["quarter", "actor_id"], suffixes=("_rev", "_m2"))
    merged["z_rev"] = _zscore_within_quarter(merged, "score_rev")
    merged["z_m2"] = _zscore_within_quarter(merged, "score_m2")
    merged["score"] = 0.5 * merged["z_rev"] + 0.5 * merged["z_m2"]
    return merged[["quarter", "actor_id", "score"]]


# ── Enhanced G1-M2 disagreement variants ───────────────────────

def signal_disagr_self_zscore(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1-M2 disagreement, self-normalized per actor.

    For each actor, z-score the disagreement against its own trailing
    history. A "surprise" disagreement (unusual for this actor) gets
    higher weight than a habitual one.
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2"]].copy()
    df = df.sort_values(["actor_id", "quarter"])
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]

    # Trailing 4Q expanding mean and std per actor
    df["roll_mean"] = df.groupby("actor_id")["raw_disagr"].transform(
        lambda x: x.expanding(min_periods=2).mean().shift(1)
    )
    df["roll_std"] = df.groupby("actor_id")["raw_disagr"].transform(
        lambda x: x.expanding(min_periods=2).std().shift(1)
    )
    df["score"] = (df["raw_disagr"] - df["roll_mean"]) / (df["roll_std"] + 1e-8)
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_disagr_block_only(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1-M2 disagreement, restricted to LOCAL-BLOCK actors only.

    For REMAINDER actors, M2 = G1 by construction (both use global aug).
    Trading only local-block actors removes noise from the remainder.
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "sector"]].copy()
    local_sectors = {"diversified", "technology", "healthcare"}
    # Include macro/institutional by checking if actor is Layer 0/1
    # (they don't have .L suffix and aren't in any sector — but we don't have layer in predictions)
    # Approximate: actors where |G1-M2| > 0 at least sometimes
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]
    actor_max_disagr = df.groupby("actor_id")["raw_disagr"].transform(
        lambda x: x.abs().max()
    )
    # Only keep actors that ever show meaningful disagreement
    df = df[actor_max_disagr > 0.005].copy()
    df["score"] = df["raw_disagr"]
    return df[["quarter", "actor_id", "score"]]


def signal_disagr_top_quartile(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1-M2 disagreement, only trade when |disagreement| is extreme.

    Score = disagreement direction, but only for actors in the top 25%
    of |disagreement| each quarter. Others get score = 0 (neutral).
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2"]].copy()
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]
    df["abs_disagr"] = df["raw_disagr"].abs()

    # Top quartile threshold per quarter
    threshold = df.groupby("quarter")["abs_disagr"].transform(
        lambda x: x.quantile(0.75)
    )
    df["score"] = np.where(df["abs_disagr"] >= threshold, df["raw_disagr"], 0.0)
    return df[["quarter", "actor_id", "score"]]


def signal_disagr_accuracy_scaled(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1-M2 disagreement, scaled by recent M2 accuracy advantage.

    For each actor, compute trailing 4Q: was M2 closer to actual than G1?
    Scale disagreement by this accuracy ratio. If M2 has been more accurate,
    trust the disagreement more.
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "actual"]].copy()
    df = df.sort_values(["actor_id", "quarter"])

    # Per-actor, per-quarter: absolute error
    df["err_g1"] = (df["pred_g1"] - df["actual"]).abs()
    df["err_m2"] = (df["pred_m2"] - df["actual"]).abs()
    df["m2_better"] = (df["err_m2"] < df["err_g1"]).astype(float)

    # Trailing 4Q hit rate of M2 being better (lagged to avoid look-ahead)
    df["m2_hit_rate"] = df.groupby("actor_id")["m2_better"].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Scale disagreement by confidence
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]
    df["score"] = df["raw_disagr"] * df["m2_hit_rate"]
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


def signal_disagr_conviction(predictions: pd.DataFrame) -> pd.DataFrame:
    """G1-M2 disagreement, scaled by cross-sectional dispersion.

    When the architecture "has strong opinions" (high cross-sectional
    dispersion of disagreement), scale up. When disagreements are small
    across the board, scale down.
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2"]].copy()
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]

    # Cross-sectional dispersion of disagreement per quarter
    q_std = df.groupby("quarter")["raw_disagr"].transform("std")
    q_median_std = q_std.median()

    # Scale: dispersion / median_dispersion (>1 = high conviction quarter)
    df["conviction"] = q_std / (q_median_std + 1e-8)
    df["score"] = df["raw_disagr"] * df["conviction"]
    return df[["quarter", "actor_id", "score"]]


def signal_disagr_combined(predictions: pd.DataFrame) -> pd.DataFrame:
    """Best-of-breed: block-only + accuracy-scaled + conviction.

    Only trade local-block actors, weight by M2 accuracy advantage,
    scale by cross-sectional conviction.
    """
    df = predictions[["quarter", "actor_id", "pred_g1", "pred_m2", "actual", "sector"]].copy()
    df = df.sort_values(["actor_id", "quarter"])

    # Block filter: only actors with meaningful disagreement
    df["raw_disagr"] = df["pred_g1"] - df["pred_m2"]
    actor_max = df.groupby("actor_id")["raw_disagr"].transform(lambda x: x.abs().max())
    df = df[actor_max > 0.005].copy()
    if df.empty:
        return pd.DataFrame(columns=["quarter", "actor_id", "score"])

    # Accuracy scaling
    df["err_g1"] = (df["pred_g1"] - df["actual"]).abs()
    df["err_m2"] = (df["pred_m2"] - df["actual"]).abs()
    df["m2_better"] = (df["err_m2"] < df["err_g1"]).astype(float)
    df["m2_hit"] = df.groupby("actor_id")["m2_better"].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Conviction scaling
    q_std = df.groupby("quarter")["raw_disagr"].transform("std")
    q_med = q_std.median()
    df["conviction"] = q_std / (q_med + 1e-8)

    df["score"] = df["raw_disagr"] * df["m2_hit"].fillna(0.5) * df["conviction"]
    return df.dropna(subset=["score"])[["quarter", "actor_id", "score"]]


COMBO_SIGNAL_REGISTRY = {
    # Enhanced G1-M2 disagreement
    "Disagr self-z": lambda p, r: signal_disagr_self_zscore(p),
    "Disagr block": lambda p, r: signal_disagr_block_only(p),
    "Disagr top-Q": lambda p, r: signal_disagr_top_quartile(p),
    "Disagr acc-sc": lambda p, r: signal_disagr_accuracy_scaled(p),
    "Disagr convic": lambda p, r: signal_disagr_conviction(p),
    "Disagr combo": lambda p, r: signal_disagr_combined(p),
    # Price-based standalone
    "Rev 1Q": lambda p, r: signal_reversal_1q(r),
    "LowVol": lambda p, r: signal_low_volatility(r),
    # Best combo attempts
    "Rev+Disagr": signal_rev_plus_disagr,
    "Rev+Disagr": signal_rev_plus_disagr,
    "Rev+M2pos": signal_rev_plus_m2pos,
}


SIGNAL_REGISTRY = {
    # Original signals (conservative premium: low predicted rank → long)
    "M2": signal_m2,
    "G1": signal_g1,
    "G0": signal_g0,
    "AR(1)": signal_ar1,
    "Naive Inv": signal_naive_investment,
    "Momentum": signal_momentum,
    # Creative signals
    "M2 Δrank-": signal_rank_change_m2,
    "M2 Δrank+": signal_rank_change_m2_positive,
    "M2-G1 disagr": signal_disagreement,
    "G1-M2 disagr": signal_disagreement_reversed,
    "|M2-G1|": signal_abs_disagreement,
    "M2 sec-neut": signal_sector_neutral_m2,
    "M2 positive": signal_m2_positive,
    "G1 positive": signal_g1_positive,
}
