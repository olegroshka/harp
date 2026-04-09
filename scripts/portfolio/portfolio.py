"""Portfolio construction from cross-sectional signals.

Pure functions: signal scores + returns → portfolio returns.
No state, no side effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quintile_sort(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Sort actors into quintiles by signal score each quarter.

    Args:
        signals: DataFrame with (quarter, actor_id, score).
        returns: DataFrame with (quarter, actor_id, return_q).
        n_quantiles: Number of quantile buckets (default 5).

    Returns:
        DataFrame with (quarter, actor_id, quintile, score, return_q).
        Quintile 1 = lowest score, quintile 5 = highest score.
    """
    merged = signals.merge(returns, on=["quarter", "actor_id"], how="inner")

    # Assign quintiles within each quarter
    merged["quintile"] = (
        merged.groupby("quarter")["score"]
        .transform(lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop") + 1)
    )

    return merged


def long_short_returns(
    sorted_df: pd.DataFrame,
    long_q: int = 5,
    short_q: int = 1,
) -> pd.DataFrame:
    """Compute long-short portfolio returns (equal-weight within quintile).

    Args:
        sorted_df: Output of quintile_sort().
        long_q: Quintile to go long (default 5 = highest score).
        short_q: Quintile to go short (default 1 = lowest score).

    Returns:
        DataFrame with (quarter, return_long, return_short, return_ls).
    """
    qr = (
        sorted_df.groupby(["quarter", "quintile"])["return_q"]
        .mean()
        .reset_index()
    )

    # Use max/min quintile if exact long_q/short_q not available
    max_q = int(qr["quintile"].max())
    min_q = int(qr["quintile"].min())
    actual_long = long_q if long_q in qr["quintile"].values else max_q
    actual_short = short_q if short_q in qr["quintile"].values else min_q

    long_ret = qr[qr["quintile"] == actual_long][["quarter", "return_q"]].rename(
        columns={"return_q": "return_long"}
    )
    short_ret = qr[qr["quintile"] == actual_short][["quarter", "return_q"]].rename(
        columns={"return_q": "return_short"}
    )

    result = long_ret.merge(short_ret, on="quarter", how="inner")
    result["return_ls"] = result["return_long"] - result["return_short"]
    return result.sort_values("quarter").reset_index(drop=True)


def long_only_returns(sorted_df: pd.DataFrame, long_q: int = 5) -> pd.DataFrame:
    """Compute long-only top quintile returns (equal-weight).

    Returns:
        DataFrame with (quarter, return_top_q, return_equal_weight).
    """
    # Top quintile return
    top = (
        sorted_df[sorted_df["quintile"] == long_q]
        .groupby("quarter")["return_q"].mean()
        .reset_index()
        .rename(columns={"return_q": "return_top_q"})
    )

    # Equal-weight benchmark (all stocks)
    ew = (
        sorted_df.groupby("quarter")["return_q"].mean()
        .reset_index()
        .rename(columns={"return_q": "return_ew"})
    )

    result = top.merge(ew, on="quarter", how="inner")
    result["return_active"] = result["return_top_q"] - result["return_ew"]
    return result.sort_values("quarter").reset_index(drop=True)


def compute_turnover(sorted_df: pd.DataFrame, quintile: int = 5) -> pd.DataFrame:
    """Compute quarterly turnover for a given quintile.

    Turnover = fraction of holdings that change from one quarter to the next.

    Returns:
        DataFrame with (quarter, turnover).
    """
    quarters = sorted(sorted_df["quarter"].unique())
    rows = []
    prev_holdings = set()

    for q in quarters:
        current = sorted_df[
            (sorted_df["quarter"] == q) & (sorted_df["quintile"] == quintile)
        ]["actor_id"].values
        current_set = set(current)

        if prev_holdings:
            stayed = len(current_set & prev_holdings)
            total = max(len(current_set), len(prev_holdings))
            turnover = 1.0 - stayed / total if total > 0 else 0.0
        else:
            turnover = 1.0  # first quarter — full portfolio construction

        rows.append({"quarter": q, "turnover": turnover})
        prev_holdings = current_set

    return pd.DataFrame(rows)
