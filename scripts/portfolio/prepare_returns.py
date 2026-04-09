"""Prepare quarterly equity returns for the portfolio backtest.

Loads OHLCV data, computes quarterly total returns for the firm-layer
actors, and merges with the prediction data.

Output: results/portfolio/returns.parquet
    Columns: quarter, actor_id, return_q (quarterly holding-period return)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BTEST_ROOT = Path("C:/Users/olegr/PycharmProjects/btest")

OUTPUT_DIR = PROJECT_ROOT / "results" / "portfolio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv(tickers: list[str]) -> pd.DataFrame:
    """Download daily adjusted close from Yahoo Finance for the given tickers.

    Caches to results/portfolio/ohlcv_cache.parquet to avoid re-downloading.
    """
    cache_path = OUTPUT_DIR / "ohlcv_cache.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_tickers = set(df["ticker"].unique())
        if set(tickers).issubset(cached_tickers):
            print(f"  Using cached OHLCV ({len(cached_tickers)} tickers)")
            return df

    print(f"  Downloading OHLCV for {len(tickers)} tickers (2014-2025)...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: uv add yfinance")

    data = yf.download(
        tickers, start="2014-01-01", end="2025-06-01",
        auto_adjust=True, progress=False,
    )
    # yf.download returns MultiIndex columns (Price, Ticker)
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    records = []
    for ticker in close.columns:
        s = close[ticker].dropna()
        for date, price in s.items():
            records.append({"date": date, "ticker": ticker, "close": float(price)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached: {cache_path} ({len(df)} rows, {df['ticker'].nunique()} tickers)")
    return df


def compute_quarterly_returns(ohlcv: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute quarterly holding-period returns from daily close prices.

    For each quarter, return = close_last / close_first - 1,
    using the last trading day of the quarter vs last trading day of prior quarter.
    """
    df = ohlcv[ohlcv["ticker"].isin(tickers)].copy()
    df = df.sort_values(["ticker", "date"])

    # Assign quarter
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    # Get last close per ticker per quarter
    quarter_close = (
        df.groupby(["ticker", "quarter"])["close"]
        .last()
        .reset_index()
        .sort_values(["ticker", "quarter"])
    )

    # Compute return: close_t / close_{t-1} - 1
    quarter_close["return_q"] = (
        quarter_close.groupby("ticker")["close"]
        .pct_change()
    )

    # Drop first quarter per ticker (no prior close)
    returns = quarter_close.dropna(subset=["return_q"])[["ticker", "quarter", "return_q"]]
    returns = returns.rename(columns={"ticker": "actor_id"})

    return returns


def main():
    print("=" * 70)
    print("  PREPARE QUARTERLY RETURNS")
    print("=" * 70)

    # Load predictions to get the actor list
    pred_path = OUTPUT_DIR / "predictions.parquet"
    if not pred_path.exists():
        print("  ERROR: Run extract_predictions.py first")
        sys.exit(1)
    predictions = pd.read_parquet(pred_path)
    all_actors = predictions["actor_id"].unique()

    # Filter to US firms only (no .L suffix)
    us_actors = [a for a in all_actors if not a.endswith(".L")]
    print(f"  US firm actors: {len(us_actors)}")

    # Load OHLCV and compute returns
    ohlcv = load_ohlcv(us_actors)
    available_tickers = set(ohlcv["ticker"].unique())
    matched = [a for a in us_actors if a in available_tickers]
    missing = [a for a in us_actors if a not in available_tickers]
    print(f"  Matched in OHLCV: {len(matched)}")
    if missing:
        print(f"  Missing ({len(missing)}): {missing}")

    returns = compute_quarterly_returns(ohlcv, matched)

    # Filter to test period (2015-2024)
    returns = returns[
        (returns["quarter"] >= "2015-01-01") &
        (returns["quarter"] <= "2024-12-31")
    ]

    print(f"  Quarterly returns: {len(returns)} rows")
    print(f"  Actors: {returns['actor_id'].nunique()}")
    print(f"  Quarters: {returns['quarter'].nunique()}")
    print(f"  Date range: {returns['quarter'].min()} to {returns['quarter'].max()}")

    # Summary stats
    print(f"\n  Return stats:")
    print(f"    Mean:   {returns['return_q'].mean():.4f}")
    print(f"    Std:    {returns['return_q'].std():.4f}")
    print(f"    Min:    {returns['return_q'].min():.4f}")
    print(f"    Max:    {returns['return_q'].max():.4f}")

    out_path = OUTPUT_DIR / "returns.parquet"
    returns.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
