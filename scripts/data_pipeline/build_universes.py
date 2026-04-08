#!/usr/bin/env python3
"""
scripts/smim/smim_build_universes.py — SMIM Session 1: Build equity universes + OHLCV.

STEP 1 — Build universe CSV files in data/smim/universes/:
  US-LC         top-200 S&P 500 by market cap
  US-LC-ENERGY  S&P 500 energy sector (GICS 10)
  US-LC-TECH    S&P 500 information technology (GICS 45)
  US-LC-FINS    S&P 500 financials (GICS 40)
  US-LC-HEALTH  S&P 500 health care (GICS 35)
  US-LC-INDUS   S&P 500 industrials (GICS 20)
  US-MC         200 S&P 400 mid-cap names
  US-SC         200 Russell 2000 small-cap (stratified by sector, seed=42)
  UK-LC         100 FTSE 100 names (Yahoo Finance .L suffix)
  UK-MC         100 FTSE 250 ex-100 names
  MIXED-200     US + UK energy equities (MVP Layer-2 actor universe)

STEP 2 — Download OHLCV for all universes (2005-01-01 to 2025-12-31).
  Output: equities/smim/{universe_id}/ (parquet, long-form)

STEP 3 — Verify data quality.

Usage:
  uv run python scripts/smim/smim_build_universes.py              # all steps
  uv run python scripts/smim/smim_build_universes.py --step 1     # universes only
  uv run python scripts/smim/smim_build_universes.py --step 2     # OHLCV only
  uv run python scripts/smim/smim_build_universes.py --step 3     # verify only
  uv run python scripts/smim/smim_build_universes.py --skip-market-cap  # use CSV order for US-LC
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Bootstrap — allow importing helpers from download_sp500_to_parquet.py
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from download_sp500_to_parquet import (  # noqa: E402
    _chunk_list,
    _sleep_backoff,
    download_ohlcv,
    build_long_table,
)

log = logging.getLogger("smim.universes")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
LOCAL_SP500_CSV = Path("data/sp500_tickers.csv")
UNIVERSES_DIR = Path("data/smim/universes")
OHLCV_DIR = Path("equities/smim")

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_FTSE100_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
WIKI_FTSE250_URL = "https://en.wikipedia.org/wiki/FTSE_250_Index"

# Wikipedia requires a browser-like User-Agent to avoid 403 errors
_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

GICS_NAME_TO_CODE: Dict[str, int] = {
    "Energy": 10,
    "Materials": 15,
    "Industrials": 20,
    "Consumer Discretionary": 25,
    "Consumer Staples": 30,
    "Health Care": 35,
    "Financials": 40,
    "Information Technology": 45,
    "Communication Services": 50,
    "Utilities": 55,
    "Real Estate": 60,
}

# Output column order for all universe CSVs
CSV_COLS = ["ticker", "name", "sector", "gics_code"]


# ---------------------------------------------------------------------------
# STEP 1 — constituent fetchers
# ---------------------------------------------------------------------------

def _fetch_wiki_tables(url: str) -> list:
    """Fetch Wikipedia page and parse all HTML tables, using a browser User-Agent."""
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _normalise_ticker(t: str) -> str:
    """Yahoo Finance compatible: replace dots with hyphens."""
    return str(t).strip().replace(".", "-")


def _add_lse_suffix(t: str) -> str:
    """Append .L for LSE-listed UK tickers if not already present."""
    t = str(t).strip()
    if not t.endswith(".L"):
        t = t + ".L"
    return t


def _parse_wiki_table(tables: list, min_rows: int = 20) -> Optional[pd.DataFrame]:
    """
    Scan Wikipedia tables for one that looks like a constituent list.
    Returns a normalised DataFrame with columns [ticker, name, sector, gics_code] or None.
    """
    col_synonyms = {
        "ticker": ("symbol", "ticker", "epic", "tidm", "ticker_symbol", "code"),
        "name": ("company", "security", "name", "company_name"),
        "sector": ("gics_sector", "sector", "industry", "gics_economic_sector"),
    }

    for t in tables:
        if len(t) < min_rows:
            continue
        df = t.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("\xa0", "_") for c in df.columns]

        rename: Dict[str, str] = {}
        for target, synonyms in col_synonyms.items():
            for col in df.columns:
                if col in synonyms and target not in rename.values():
                    rename[col] = target
        df = df.rename(columns=rename)

        if "ticker" not in df.columns:
            continue

        df["ticker"] = df["ticker"].astype(str).str.strip()
        if "name" not in df.columns:
            df["name"] = pd.NA
        if "sector" not in df.columns:
            df["sector"] = pd.NA
        df["gics_code"] = df["sector"].map(GICS_NAME_TO_CODE)
        return df[CSV_COLS].dropna(subset=["ticker"]).reset_index(drop=True)

    return None


def load_sp500_local() -> pd.DataFrame:
    """Load S&P 500 from the project-local CSV at data/sp500_tickers.csv."""
    df = pd.read_csv(LOCAL_SP500_CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
    })
    df["ticker"] = df["ticker"].apply(_normalise_ticker)
    df["gics_code"] = df["sector"].map(GICS_NAME_TO_CODE)
    return df[CSV_COLS].dropna(subset=["ticker"]).reset_index(drop=True)


def fetch_sp500_wikipedia() -> pd.DataFrame:
    """Fallback: fetch S&P 500 from Wikipedia."""
    log.info("Fetching S&P 500 from Wikipedia (fallback)...")
    tables = _fetch_wiki_tables(WIKI_SP500_URL)
    result = _parse_wiki_table(tables, min_rows=400)
    if result is None:
        raise RuntimeError("Could not parse S&P 500 table from Wikipedia")
    result["ticker"] = result["ticker"].apply(_normalise_ticker)
    return result


def fetch_sp400_wikipedia() -> pd.DataFrame:
    """Fetch S&P 400 mid-cap constituents from Wikipedia."""
    log.info("Fetching S&P 400 from Wikipedia...")
    try:
        tables = _fetch_wiki_tables(WIKI_SP400_URL)
        result = _parse_wiki_table(tables, min_rows=300)
        if result is not None:
            result["ticker"] = result["ticker"].apply(_normalise_ticker)
            log.info(f"  S&P 400: {len(result)} components")
            return result
    except Exception as e:
        log.warning(f"  S&P 400 Wikipedia fetch failed: {e}")
    return pd.DataFrame(columns=CSV_COLS)


def fetch_russell2000_ishares() -> pd.DataFrame:
    """
    Attempt to fetch Russell 2000 holdings from iShares IWM ETF.
    The iShares CSV has metadata rows before the actual data.
    """
    import urllib.request

    log.info("Fetching Russell 2000 from iShares IWM holdings...")
    url = (
        "https://www.ishares.com/us/products/239710/IWM"
        "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "research-bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"iShares IWM fetch failed: {e}") from e

    lines = raw.splitlines()
    # Find the header row — it contains the literal word "Ticker"
    header_idx = next((i for i, ln in enumerate(lines) if "Ticker" in ln), None)
    if header_idx is None:
        raise ValueError("Could not locate header row in iShares CSV")

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    ticker_col = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    sector_col = next((c for c in df.columns if "sector" in c.lower()), None)

    if ticker_col is None:
        raise ValueError("No ticker column in iShares CSV")

    rename: Dict[str, str] = {ticker_col: "ticker"}
    if name_col:
        rename[name_col] = "name"
    if sector_col:
        rename[sector_col] = "sector"
    df = df.rename(columns=rename)

    df["ticker"] = df["ticker"].astype(str).str.strip()
    # Keep only plain equity tickers (letters only, no dashes/numbers/cash)
    df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")].copy()
    df["ticker"] = df["ticker"].apply(_normalise_ticker)
    if "name" not in df.columns:
        df["name"] = pd.NA
    if "sector" not in df.columns:
        df["sector"] = pd.NA
    df["gics_code"] = df["sector"].map(GICS_NAME_TO_CODE)

    log.info(f"  Russell 2000: {len(df)} equity components")
    return df[CSV_COLS].dropna(subset=["ticker"]).reset_index(drop=True)


def fetch_ftse_wikipedia(url: str, label: str) -> pd.DataFrame:
    """Fetch FTSE 100 or FTSE 250 from Wikipedia."""
    log.info(f"Fetching {label} from Wikipedia...")
    try:
        tables = _fetch_wiki_tables(url)
        result = _parse_wiki_table(tables, min_rows=50)
        if result is not None:
            # Apply .L suffix for Yahoo Finance
            result["ticker"] = result["ticker"].apply(_add_lse_suffix)
            # Keep only valid LSE ticker pattern
            result = result[result["ticker"].str.match(r"^[A-Z0-9]{1,6}\.L$")].copy()
            log.info(f"  {label}: {len(result)} components")
            return result.reset_index(drop=True)
    except Exception as e:
        log.warning(f"  {label} Wikipedia fetch failed: {e}")
    return pd.DataFrame(columns=CSV_COLS)


def get_market_caps(tickers: List[str], delay: float = 0.3) -> Dict[str, float]:
    """
    Fetch market caps from Yahoo Finance fast_info for S&P 500 market-cap ordering.
    Returns dict {ticker: market_cap_usd}. Missing tickers get 0.
    """
    log.info(f"Fetching market caps for {len(tickers)} S&P 500 tickers...")
    caps: Dict[str, float] = {}

    for i, sym in enumerate(tickers, start=1):
        try:
            fi = yf.Ticker(sym).fast_info
            # fast_info returns FastInfo object (yfinance >= 0.2); access as attribute
            mc = getattr(fi, "market_cap", None)
            if mc is None:
                mc = fi.get("marketCap") if hasattr(fi, "get") else None
            caps[sym] = float(mc) if mc and mc > 0 else 0.0
        except Exception:
            caps[sym] = 0.0

        if i % 50 == 0:
            log.info(f"  Market cap progress: {i}/{len(tickers)}")
            time.sleep(delay)

    n_got = sum(1 for v in caps.values() if v > 0)
    log.info(f"  Got market caps for {n_got}/{len(tickers)} tickers")
    return caps


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    sector_col: str = "sector",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a stratified random sample of n rows from df.
    Stratifies by sector_col; sectors with fewer than 1 expected row get
    at least 1. Fills to exactly n with a random draw from the remainder.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    if len(df) <= n:
        return df.copy()

    df = df.copy().reset_index(drop=True)
    df["_sector_"] = df[sector_col].fillna("Unknown")
    sectors = df["_sector_"].unique()

    per_sector = {s: max(1, round(n * (df["_sector_"] == s).sum() / len(df))) for s in sectors}
    # Adjust to sum to n
    total = sum(per_sector.values())
    while total > n:
        biggest = max(per_sector, key=per_sector.__getitem__)
        per_sector[biggest] -= 1
        total -= 1
    while total < n:
        # Add to the most represented sector
        biggest = max(per_sector, key=per_sector.__getitem__)
        per_sector[biggest] += 1
        total += 1

    sampled_idx: List[int] = []
    for s, k in per_sector.items():
        pool = df[df["_sector_"] == s].index.tolist()
        chosen = rng.sample(pool, min(k, len(pool)))
        sampled_idx.extend(chosen)

    # If short (some sectors had fewer members than allocated), top up randomly
    if len(sampled_idx) < n:
        remaining = [i for i in df.index if i not in set(sampled_idx)]
        top_up = rng.sample(remaining, min(n - len(sampled_idx), len(remaining)))
        sampled_idx.extend(top_up)

    result = df.loc[sampled_idx[:n]].drop(columns=["_sector_"]).reset_index(drop=True)
    return result


def _print_universe_summary(name: str, df: pd.DataFrame) -> None:
    """Print a one-line summary + sector breakdown for a universe."""
    print(f"\n{'-'*60}")
    print(f"  {name:15s}  {len(df):4d} tickers")
    if "sector" in df.columns and df["sector"].notna().any():
        breakdown = df["sector"].value_counts().head(8)
        for sector, count in breakdown.items():
            print(f"    {sector:<35s} {count:4d}")


# ---------------------------------------------------------------------------
# STEP 1 — build and save all universe CSVs
# ---------------------------------------------------------------------------

def step1_build_universes(skip_market_cap: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Build all universe DataFrames and save to data/smim/universes/.
    Returns dict {universe_id: DataFrame}.
    """
    UNIVERSES_DIR.mkdir(parents=True, exist_ok=True)

    universes: Dict[str, pd.DataFrame] = {}

    # ── S&P 500 ──────────────────────────────────────────────────────────────
    log.info("Loading S&P 500 constituents...")
    if LOCAL_SP500_CSV.exists():
        sp500 = load_sp500_local()
        log.info(f"  Loaded {len(sp500)} tickers from local CSV")
    else:
        sp500 = fetch_sp500_wikipedia()
        log.info(f"  Loaded {len(sp500)} tickers from Wikipedia")

    # US-LC-ENERGY
    energy = sp500[sp500["gics_code"] == 10].copy()
    universes["US-LC-ENERGY"] = energy
    log.info(f"  US-LC-ENERGY: {len(energy)} tickers")

    # US-LC-TECH
    tech = sp500[sp500["gics_code"] == 45].copy()
    universes["US-LC-TECH"] = tech
    log.info(f"  US-LC-TECH: {len(tech)} tickers")

    # US-LC-FINS
    fins = sp500[sp500["gics_code"] == 40].copy()
    universes["US-LC-FINS"] = fins
    log.info(f"  US-LC-FINS: {len(fins)} tickers")

    # US-LC-HEALTH
    health = sp500[sp500["gics_code"] == 35].copy()
    universes["US-LC-HEALTH"] = health
    log.info(f"  US-LC-HEALTH: {len(health)} tickers")

    # US-LC-INDUS
    indus = sp500[sp500["gics_code"] == 20].copy()
    universes["US-LC-INDUS"] = indus
    log.info(f"  US-LC-INDUS: {len(indus)} tickers")

    # US-LC — top 200 by market cap
    if skip_market_cap:
        log.info("  --skip-market-cap: using first 200 tickers from S&P 500 list for US-LC")
        us_lc = sp500.head(200).copy()
    else:
        caps = get_market_caps(sp500["ticker"].tolist())
        sp500_mc = sp500.copy()
        sp500_mc["market_cap"] = sp500_mc["ticker"].map(caps).fillna(0.0)
        sp500_mc = sp500_mc.sort_values("market_cap", ascending=False)
        us_lc = sp500_mc.head(200).drop(columns=["market_cap"]).reset_index(drop=True)
    universes["US-LC"] = us_lc
    log.info(f"  US-LC: {len(us_lc)} tickers")

    # ── S&P 400 (US-MC) ──────────────────────────────────────────────────────
    sp400 = fetch_sp400_wikipedia()
    if len(sp400) >= 200:
        us_mc = sp400.head(200).copy()
    elif len(sp400) > 0:
        log.warning(f"  S&P 400 only has {len(sp400)} tickers; using all")
        us_mc = sp400.copy()
    else:
        log.warning("  S&P 400 unavailable — US-MC will be empty")
        us_mc = pd.DataFrame(columns=CSV_COLS)
    universes["US-MC"] = us_mc
    log.info(f"  US-MC: {len(us_mc)} tickers")

    # ── Russell 2000 (US-SC) ─────────────────────────────────────────────────
    try:
        r2000 = fetch_russell2000_ishares()
    except Exception as e:
        log.warning(f"  Russell 2000 fetch failed: {e}")
        log.warning("  US-SC will be empty. Populate data/smim/universes/russell2000.csv manually.")
        r2000 = pd.DataFrame(columns=CSV_COLS)

    if len(r2000) > 0:
        us_sc = stratified_sample(r2000, n=200, sector_col="sector", seed=42)
        log.info(f"  US-SC: stratified sample of 200 from {len(r2000)} Russell 2000 tickers")
    else:
        us_sc = pd.DataFrame(columns=CSV_COLS)
    universes["US-SC"] = us_sc
    log.info(f"  US-SC: {len(us_sc)} tickers")

    # ── FTSE 100 / 250 ───────────────────────────────────────────────────────
    ftse100 = fetch_ftse_wikipedia(WIKI_FTSE100_URL, "FTSE 100")
    uk_lc = ftse100.head(100).copy()
    universes["UK-LC"] = uk_lc
    log.info(f"  UK-LC: {len(uk_lc)} tickers")

    ftse250_all = fetch_ftse_wikipedia(WIKI_FTSE250_URL, "FTSE 250")
    # Exclude tickers already in FTSE 100
    ftse100_set = set(ftse100["ticker"].tolist())
    ftse250_ex = ftse250_all[~ftse250_all["ticker"].isin(ftse100_set)].reset_index(drop=True)
    uk_mc = ftse250_ex.head(100).copy()
    universes["UK-MC"] = uk_mc
    log.info(f"  UK-MC: {len(uk_mc)} tickers (FTSE 250 ex-100)")

    # ── MIXED-200 — MVP energy actor universe ────────────────────────────────
    # Layer 2 equities: US energy (S&P 500 sector 10) + UK energy (FTSE 100 energy)
    # UK energy: FTSE 100 doesn't use GICS codes, so include known UK energy names manually
    # plus all FTSE 100 with sector tagged "Oil & Gas" / "Energy"
    uk_energy_tickers = [
        "SHEL.L",  # Shell plc
        "BP.L",    # BP plc
        "SSE.L",   # SSE plc
        "NG.L",    # National Grid
        "CNA.L",   # Centrica
        "SVCO.L",  # Saville Group / placeholder — see below
        "DGEN.L",  # placeholder
    ]
    # Build from FTSE 100 using ICB sector if available, else known list
    uk_energy_rows: List[pd.DataFrame] = []

    # Sector-tagged FTSE 100 energy names (ICB = Oil, Gas & Coal or Utilities)
    if "sector" in ftse100.columns and ftse100["sector"].notna().any():
        energy_keywords = ["oil", "gas", "energy", "utilities", "coal", "petroleum"]
        mask = ftse100["sector"].str.lower().str.contains("|".join(energy_keywords), na=False)
        uk_energy_rows.append(ftse100[mask])

    # Always include known UK energy majors if present in FTSE 100
    known_uk_energy = {"SHEL.L", "BP.L", "SSE.L", "NG.L", "CNA.L", "RDSB.L", "RDSA.L"}
    known_uk_df = ftse100[ftse100["ticker"].isin(known_uk_energy)]
    uk_energy_rows.append(known_uk_df)

    uk_energy = (
        pd.concat(uk_energy_rows, ignore_index=True).drop_duplicates(subset=["ticker"])
        if uk_energy_rows else pd.DataFrame(columns=CSV_COLS)
    )

    # Combine US energy + UK energy
    us_energy_mixed = energy.copy()
    us_energy_mixed["geography"] = "US"
    uk_energy_mixed = uk_energy.copy()
    uk_energy_mixed["geography"] = "UK"

    mixed = pd.concat([us_energy_mixed, uk_energy_mixed], ignore_index=True)
    mixed = mixed.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    # Drop the helper geography column before saving
    if "geography" in mixed.columns:
        mixed = mixed.drop(columns=["geography"])
    universes["MIXED-200"] = mixed
    log.info(f"  MIXED-200: {len(mixed)} tickers (US energy + UK energy)")

    # ── Save all CSVs ────────────────────────────────────────────────────────
    log.info(f"\nSaving universe CSVs to {UNIVERSES_DIR}/")
    for uid, df in universes.items():
        path = UNIVERSES_DIR / f"{uid}.csv"
        # Ensure all CSV_COLS exist
        for col in CSV_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df[CSV_COLS].to_csv(path, index=False)
        log.info(f"  Saved {path.name} ({len(df)} rows)")

    # ── Print summaries ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 SUMMARY - Universe CSVs")
    print("=" * 60)
    for uid, df in universes.items():
        _print_universe_summary(uid, df)
    print()

    return universes


# ---------------------------------------------------------------------------
# STEP 2 — download OHLCV for each universe
# ---------------------------------------------------------------------------

def _download_yf_outer(
    tickers: List[str],
    start: str,
    end: str,
    chunk_size: int = 50,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Download OHLCV for *tickers* using yfinance with an outer-join on dates.

    Each ticker retains its own date range — no inner-join truncation.
    Returns a long-form DataFrame with columns:
        date, ticker, open, high, low, close, volume
    Missing values appear as NaN (not dropped rows).
    """
    chunks = _chunk_list(tickers, chunk_size)
    long_parts: List[pd.DataFrame] = []

    for i, chunk in enumerate(chunks, start=1):
        log.info(f"  Chunk {i}/{len(chunks)}: {len(chunk)} tickers")
        for attempt in range(1, 4):
            try:
                raw = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if raw.empty:
                    log.warning(f"    chunk {i} attempt {attempt}: empty result")
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    continue

                # yf.download with multiple tickers returns MultiIndex columns:
                # (field, ticker). With a single ticker it returns flat columns.
                if isinstance(raw.columns, pd.MultiIndex):
                    # Stack ticker level → long form
                    raw.index.name = "date"
                    raw.columns.names = ["field", "ticker"]
                    long_chunk = (
                        raw.stack(level="ticker", future_stack=True)
                        .reset_index()
                    )
                    # Rename yfinance columns to our schema
                    col_map = {
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    }
                    long_chunk = long_chunk.rename(columns=col_map)
                else:
                    # Single ticker — flat columns
                    assert len(chunk) == 1
                    long_chunk = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
                    long_chunk["ticker"] = chunk[0]
                    col_map = {
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    }
                    long_chunk = long_chunk.rename(columns=col_map)

                # Keep only rows where close is not NaN (ticker had no data that day)
                keep_cols = [c for c in ("date", "ticker", "open", "high", "low", "close", "volume") if c in long_chunk.columns]
                long_chunk = long_chunk[keep_cols].dropna(subset=["close"])
                long_chunk["date"] = pd.to_datetime(long_chunk["date"])

                if not long_chunk.empty:
                    long_parts.append(long_chunk)
                    log.info(f"    chunk {i}: {long_chunk['ticker'].nunique()} tickers, {len(long_chunk)} rows")
                break

            except Exception as e:
                log.warning(f"    chunk {i} attempt {attempt} failed: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)

        if i < len(chunks):
            time.sleep(delay)

    if not long_parts:
        raise RuntimeError("All chunks returned no data.")

    result = pd.concat(long_parts, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def step2_download_ohlcv(
    universes: Dict[str, pd.DataFrame],
    start: str = START_DATE,
    end: str = END_DATE,
    chunk_size: int = 50,
    retries: int = 3,
    inter_universe_delay: float = 2.0,
) -> None:
    """Download OHLCV for all universes and save to equities/smim/{universe_id}/."""
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}

    for uid, df in universes.items():
        if df.empty:
            log.warning(f"  {uid}: empty universe, skipping OHLCV download")
            continue

        tickers = df["ticker"].dropna().unique().tolist()
        log.info(f"\n{'-'*60}")
        log.info(f"  Downloading OHLCV for {uid} ({len(tickers)} tickers, {start} to {end})")

        out_path = OHLCV_DIR / uid
        out_path.mkdir(parents=True, exist_ok=True)
        parquet_path = out_path / "ohlcv.parquet"

        try:
            long = _download_yf_outer(tickers, start, end, chunk_size=chunk_size)

            if long.empty:
                log.warning(f"  {uid}: no usable rows returned")
                results[uid] = {"downloaded": 0, "attempted": len(tickers), "failed": True}
                continue

            # Attach sector info from universe CSV
            long = long.merge(df[["ticker", "sector"]].drop_duplicates(), on="ticker", how="left")

            long.to_parquet(parquet_path, index=False)

            n_tickers = long["ticker"].nunique()
            dt_min = long["date"].min().date()
            dt_max = long["date"].max().date()
            log.info(f"  {uid}: {n_tickers}/{len(tickers)} tickers, {dt_min} to {dt_max}, saved to {parquet_path}")

            results[uid] = {
                "downloaded": n_tickers,
                "attempted": len(tickers),
                "date_min": dt_min,
                "date_max": dt_max,
                "failed": False,
            }

        except Exception as e:
            log.error(f"  {uid}: OHLCV download failed: {e}")
            results[uid] = {"downloaded": 0, "attempted": len(tickers), "error": str(e), "failed": True}

        time.sleep(inter_universe_delay)

    # Summary
    print("\n" + "=" * 60)
    print("  STEP 2 SUMMARY - OHLCV Downloads")
    print("=" * 60)
    for uid, r in results.items():
        status = "OK" if not r.get("failed") else "FAIL"
        pct = f"{100*r['downloaded']/r['attempted']:.0f}%" if r["attempted"] else "n/a"
        date_rng = f"{r.get('date_min','')} to {r.get('date_max','')}" if not r.get("failed") else r.get("error", "")
        print(f"  {uid:15s}  {status:4s}  {r['downloaded']:4d}/{r['attempted']:4d} ({pct})  {date_rng}")
    print()


# ---------------------------------------------------------------------------
# STEP 3 — data quality verification
# ---------------------------------------------------------------------------

def step3_verify(universes: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Verify downloaded OHLCV quality for all universes.
    Loads from equities/smim/{uid}/ohlcv.parquet if not passed in memory.
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — Data Quality Verification")
    print("=" * 60)

    if universes is None:
        # Load universe CSVs from disk
        universes = {}
        for csv_path in sorted(UNIVERSES_DIR.glob("*.csv")):
            uid = csv_path.stem
            universes[uid] = pd.read_csv(csv_path)

    issues: List[str] = []

    for uid in sorted(universes.keys()):
        parquet_path = OHLCV_DIR / uid / "ohlcv.parquet"
        if not parquet_path.exists():
            print(f"\n  {uid}: NO PARQUET FILE (skipped or failed)")
            issues.append(f"{uid}: parquet missing")
            continue

        long = pd.read_parquet(parquet_path)
        if long.empty:
            print(f"\n  {uid}: EMPTY PARQUET")
            issues.append(f"{uid}: empty parquet")
            continue

        universe_df = universes[uid]
        attempted = universe_df["ticker"].nunique()
        downloaded = long["ticker"].nunique()
        dt_min = pd.to_datetime(long["date"]).min().date()
        dt_max = pd.to_datetime(long["date"]).max().date()

        # Trading day coverage per ticker
        all_dates = pd.to_datetime(long["date"]).dt.date.unique()
        n_trading_days = len(all_dates)

        per_ticker_pct = (
            long.groupby("ticker")["close"].apply(lambda s: s.notna().sum()) / n_trading_days * 100
        )
        avg_coverage = per_ticker_pct.mean()
        sparse_tickers = per_ticker_pct[per_ticker_pct < 50].index.tolist()

        print(f"\n  {uid}")
        print(f"    Tickers downloaded : {downloaded}/{attempted}")
        print(f"    Date range         : {dt_min} to {dt_max}")
        print(f"    Trading days       : {n_trading_days}")
        print(f"    Avg coverage/ticker: {avg_coverage:.1f}%")
        if sparse_tickers:
            sample = sparse_tickers[:5]
            print(f"    Sparse tickers (>50% missing): {len(sparse_tickers)} — e.g. {sample}")
            issues.append(f"{uid}: {len(sparse_tickers)} sparse tickers")

    print()
    if issues:
        print(f"  Issues found ({len(issues)}):")
        for iss in issues:
            print(f"    [!] {iss}")
    else:
        print("  All universe parquets look healthy.")
    print()


# ---------------------------------------------------------------------------
# .gitignore helper — add OHLCV parquet paths
# ---------------------------------------------------------------------------

def _ensure_gitignore_excludes_parquet() -> None:
    """Add equities/smim/ to .gitignore if not already present."""
    gi_path = Path(".gitignore")
    entry = "equities/smim/"
    if gi_path.exists():
        content = gi_path.read_text(encoding="utf-8")
        if entry not in content:
            with gi_path.open("a", encoding="utf-8") as f:
                f.write(f"\n# SMIM OHLCV data (large binary, not committed)\n{entry}\n")
            log.info(f"  Added '{entry}' to .gitignore")
    else:
        log.warning(".gitignore not found — parquet files at equities/smim/ will not be excluded from git")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SMIM equity universes and download OHLCV.")
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Which steps to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--skip-market-cap",
        action="store_true",
        help="Skip Yahoo Finance market-cap fetching for US-LC; use CSV order instead",
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"OHLCV start date (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"OHLCV end date (default: {END_DATE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Tickers per download chunk (default: 50)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="UNIVERSE_ID",
        help="Restrict step 2 to these universe IDs only (e.g. --only US-LC US-SC)",
    )
    args = parser.parse_args()

    steps = set(args.step)
    universes: Optional[Dict[str, pd.DataFrame]] = None

    if 1 in steps:
        universes = step1_build_universes(skip_market_cap=args.skip_market_cap)

    if 2 in steps:
        if universes is None:
            # Load from disk if step 1 was not run this session
            universes = {}
            for csv_path in sorted(UNIVERSES_DIR.glob("*.csv")):
                universes[csv_path.stem] = pd.read_csv(csv_path)
        if args.only:
            universes = {k: v for k, v in universes.items() if k in args.only}
            log.info(f"--only filter: downloading {list(universes.keys())}")
        _ensure_gitignore_excludes_parquet()
        step2_download_ohlcv(
            universes,
            start=args.start,
            end=args.end,
            chunk_size=args.chunk_size,
        )

    if 3 in steps:
        step3_verify(universes)

    log.info("Done.")


if __name__ == "__main__":
    main()
