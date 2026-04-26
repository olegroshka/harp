"""Fetch daily EOD prices for qualifying EU firms from EODHD.

Only fetches prices for firms that passed the >=56Q coverage threshold
in the fundamentals pull. Cache-aware: skips firms already in the output.

Usage:
    EODHD_API_KEY=xxx uv run python scripts/data_pipeline/fetch_eodhd_eu_prices.py

Outputs:
    data/raw/eodhd/prices_daily.parquet — daily OHLCV for qualifying firms
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw" / "eodhd"
COVERAGE_PATH = RAW_DIR / "coverage_summary.csv"
PRICES_PATH = RAW_DIR / "prices_daily.parquet"

EODHD_BASE = "https://eodhd.com/api"
HTTP_TIMEOUT = 60
DELAY = 0.12
FLUSH_EVERY = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eodhd_prices")


def main() -> None:
    api_key = os.environ.get("EODHD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EODHD_API_KEY not set")

    if not COVERAGE_PATH.exists():
        raise RuntimeError(f"Run fetch_eodhd_eu_fundamentals.py first — {COVERAGE_PATH} not found")

    cov = pd.read_csv(COVERAGE_PATH)
    qualifying = cov[cov["both_60q"] == 1][["ticker", "exchange"]].values.tolist()
    log.info("Qualifying firms for price pull: %d", len(qualifying))

    # Cache-aware: load existing
    cached: set[tuple[str, str]] = set()
    existing: pd.DataFrame | None = None
    if PRICES_PATH.exists():
        existing = pd.read_parquet(PRICES_PATH)
        cached = set(zip(existing["ticker"], existing["exchange"]))
        log.info("Cached prices for %d firms — will skip", len(cached))

    session = requests.Session()
    session.params = {"api_token": api_key}
    new_frames: list[pd.DataFrame] = []
    fetched = 0
    skipped = 0

    def _flush():
        nonlocal existing
        if not new_frames:
            return
        new_df = pd.concat(new_frames, ignore_index=True)
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_parquet(PRICES_PATH, index=False)
        existing = merged
        log.info("  [FLUSH] %d total price rows on disk (%d firms)",
                 len(merged), merged[["ticker", "exchange"]].drop_duplicates().shape[0])
        new_frames.clear()

    for i, (ticker, exchange) in enumerate(qualifying):
        if (ticker, exchange) in cached:
            continue

        if fetched > 0 and fetched % 50 == 0:
            log.info("Progress: %d/%d (fetched=%d, skipped=%d)",
                     i + 1, len(qualifying), fetched, skipped)

        url = f"{EODHD_BASE}/eod/{ticker}.{exchange}"
        try:
            r = session.get(url, params={"from": "2005-01-01", "to": "2025-12-31", "period": "d", "fmt": "json"},
                            timeout=HTTP_TIMEOUT)
        except requests.RequestException:
            skipped += 1
            time.sleep(DELAY)
            continue

        if r.status_code != 200:
            skipped += 1
            time.sleep(DELAY)
            continue

        try:
            data = r.json()
        except Exception:
            skipped += 1
            continue

        if not data or not isinstance(data, list):
            skipped += 1
            time.sleep(DELAY)
            continue

        df = pd.DataFrame(data)
        df["ticker"] = ticker
        df["exchange"] = exchange
        new_frames.append(df)
        cached.add((ticker, exchange))
        fetched += 1

        if fetched % FLUSH_EVERY == 0:
            _flush()

        time.sleep(DELAY)

    _flush()
    log.info("Done: fetched=%d, skipped=%d", fetched, skipped)

    if PRICES_PATH.exists():
        final = pd.read_parquet(PRICES_PATH)
        print(f"\nPrices: {len(final):,} rows, {final[['ticker','exchange']].drop_duplicates().shape[0]} firms")
        print(f"Date range: {final['date'].min()} -> {final['date'].max()}")


if __name__ == "__main__":
    main()
