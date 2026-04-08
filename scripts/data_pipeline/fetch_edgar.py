"""Fetch SEC EDGAR XBRL balance sheet data for all US equity universes.

Fetches company-level XBRL facts for US tickers across US-LC, US-MC, US-SC,
and all sector-slice universes. No API key required — SEC only needs a
descriptive User-Agent header.

Usage:
    uv run python scripts/smim/smim_fetch_edgar.py

Outputs:
    data/smim/raw/edgar/<CIK>.json           — raw per-company JSON (optional cache)
    data/smim/processed/edgar_balance_sheet.parquet — normalised tidy table
    data/smim/pit_store/edgar.parquet        — PIT store shard (A1-compliant)

Rate limit: SEC requires ≤10 req/s. We sleep 0.15s between requests (~6.7 req/s).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Ensure src package is importable when run as a script
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from harp.data.pit_store import PointInTimeStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_USER_AGENT = "SMIM Research oleg@example.com"
_HEADERS = {"User-Agent": _USER_AGENT}
_SLEEP = 0.15  # seconds between requests — keeps us safely under 10 req/s

_TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# XBRL tags to extract (matches SmimConfig.data.edgar.xbrl_tags + extras from spec)
XBRL_TAGS: list[str] = [
    # Capital expenditure (several common aliases)
    "CapitalExpenditures",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    # R&D
    "ResearchAndDevelopmentExpense",
    # Balance sheet
    "Assets",
    # Revenue (several common aliases)
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    # Debt / equity
    "LongTermDebt",
    "StockholdersEquity",
]

FILING_TYPES: set[str] = {"10-K", "10-Q"}

# US universe CSV files (excludes UK universes — EDGAR covers US only)
_UNIVERSE_DIR = _ROOT / "data" / "universes"
US_UNIVERSE_FILES: list[str] = [
    "US-LC.csv",
    "US-MC.csv",
    "US-SC.csv",
    "US-LC-ENERGY.csv",
    "US-LC-TECH.csv",
    "US-LC-FINS.csv",
    "US-LC-HEALTH.csv",
    "US-LC-INDUS.csv",
]

# Output paths
RAW_DIR = _ROOT / "data" / "raw" / "edgar"
PROCESSED_PATH = _ROOT / "data" / "processed" / "edgar_balance_sheet.parquet"
PIT_DIR = _ROOT / "data" / "pit_store"

OBS_START = pd.Timestamp("2005-01-01")


# ── Step 1: Collect unique US tickers ─────────────────────────────────────────

def collect_us_tickers() -> set[str]:
    """Read all US universe CSVs and return a set of unique tickers."""
    tickers: set[str] = set()
    for fname in US_UNIVERSE_FILES:
        path = _UNIVERSE_DIR / fname
        if not path.exists():
            log.warning("Universe file not found, skipping: %s", path)
            continue
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            log.warning("No 'ticker' column in %s, skipping", fname)
            continue
        count = df["ticker"].nunique()
        tickers.update(df["ticker"].dropna().str.strip().tolist())
        log.info("  %s: %d tickers", fname, count)
    log.info("Total unique US tickers: %d", len(tickers))
    return tickers


# ── Step 2a: CIK mapping via SEC company_tickers.json ─────────────────────────

def build_cik_map(tickers: set[str]) -> dict[str, str]:
    """Download SEC company_tickers.json and build ticker → zero-padded CIK map."""
    log.info("Downloading CIK mapping from SEC …")
    resp = requests.get(_TICKERS_JSON_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(_SLEEP)

    data = resp.json()
    # SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    ticker_to_cik: dict[str, str] = {}
    for entry in data.values():
        t = str(entry.get("ticker", "")).strip().upper()
        c = str(entry.get("cik_str", "")).strip()
        if t and c:
            ticker_to_cik[t] = c.zfill(10)

    # Filter to our universe
    mapped: dict[str, str] = {}
    missing: list[str] = []
    for tk in sorted(tickers):
        cik = ticker_to_cik.get(tk.upper())
        if cik:
            mapped[tk] = cik
        else:
            missing.append(tk)

    log.info("CIK mapping: %d found, %d not found", len(mapped), len(missing))
    if missing:
        log.warning("Tickers with no CIK mapping: %s", missing)
    return mapped


# ── Step 2b: Fetch XBRL facts per company ─────────────────────────────────────

def _parse_company_facts(
    data: dict,
    ticker: str,
    cik: str,
) -> list[dict]:
    """Extract configured XBRL tags from a company-facts JSON payload."""
    records: list[dict] = []
    facts = data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})

    for tag in XBRL_TAGS:
        tag_data = us_gaap.get(tag, {})
        units = tag_data.get("units", {})
        for unit_values in units.values():
            for obs in unit_values:
                form = obs.get("form", "")
                if form not in FILING_TYPES:
                    continue
                end_date = obs.get("end")
                filed_date = obs.get("filed")
                val = obs.get("val")
                period = obs.get("frame", "")  # e.g. "CY2022Q3I"
                if end_date and filed_date and val is not None:
                    records.append({
                        "ticker": ticker,
                        "cik": cik,
                        "event_date": end_date,
                        "pub_date": filed_date,
                        "tag": tag,
                        "value": float(val),
                        "form_type": form,
                        "period": period,
                    })
    return records


def fetch_xbrl_facts(
    ticker_cik_map: dict[str, str],
) -> pd.DataFrame:
    """Fetch company facts for all tickers. Returns raw long-format DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    no_data: list[str] = []
    total = len(ticker_cik_map)

    for i, (ticker, cik) in enumerate(sorted(ticker_cik_map.items()), 1):
        if i % 50 == 0 or i == 1:
            log.info("Progress: %d / %d tickers …", i, total)

        url = f"{_FACTS_BASE}/CIK{cik}.json"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Failed to fetch %s (CIK %s): %s", ticker, cik, exc)
            no_data.append(ticker)
            time.sleep(_SLEEP)
            continue

        records = _parse_company_facts(data, ticker, cik)
        if records:
            all_records.extend(records)
        else:
            no_data.append(ticker)

        time.sleep(_SLEEP)

    log.info(
        "Fetch complete: %d tickers with data, %d with no relevant filings",
        total - len(no_data), len(no_data),
    )
    if no_data:
        log.info("No-data tickers: %s", no_data)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["value"] = df["value"].astype("float64")

    # Filter to OBS_START by event_date
    df = df[df["event_date"] >= OBS_START].reset_index(drop=True)

    return df


# ── Step 3: Ingest into PIT store ─────────────────────────────────────────────

def ingest_to_pit(edgar_df: pd.DataFrame) -> None:
    """Ingest edgar DataFrame into the PIT store."""
    pit_df = edgar_df.copy()
    # PIT schema requires signal_id (= xbrl tag), actor_id (= ticker), source
    pit_df = pit_df.rename(columns={"tag": "signal_id", "ticker": "actor_id"})
    pit_df["source"] = "edgar"
    pit_df["vintage_id"] = None

    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated at %s", PIT_DIR)


# ── Step 4: Summary ───────────────────────────────────────────────────────────

def print_summary(
    edgar_df: pd.DataFrame,
    ticker_cik_map: dict[str, str],
    attempted: int,
) -> None:
    if edgar_df.empty:
        print("\nNo EDGAR data retrieved.")
        return

    tickers_with_data = edgar_df["ticker"].nunique()
    total_records = len(edgar_df)
    date_min = edgar_df["event_date"].min().date()
    date_max = edgar_df["event_date"].max().date()

    print("\n" + "=" * 60)
    print("SEC EDGAR fetch summary")
    print("=" * 60)
    print(f"  Tickers with data        : {tickers_with_data} / {attempted} attempted")
    print(f"  Total filing records     : {total_records:,}")
    print(f"  Date range               : {date_min} to {date_max}")
    print()
    print("  Coverage by tag:")
    for tag in XBRL_TAGS:
        n = edgar_df[edgar_df["tag"] == tag]["ticker"].nunique()
        print(f"    {tag:<55s}: {n:>4d} tickers")

    tickers_attempted = set(ticker_cik_map.keys())
    tickers_found = set(edgar_df["ticker"].unique())
    no_data = sorted(tickers_attempted - tickers_found)
    if no_data:
        print()
        print(f"  Tickers with NO filings found ({len(no_data)}):")
        # Print in rows of 10
        for i in range(0, len(no_data), 10):
            print("    " + "  ".join(no_data[i:i + 10]))

    print(f"\n  Processed parquet        : {PROCESSED_PATH}")
    print(f"  PIT store                : {PIT_DIR}")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== SMIM EDGAR Fetch ===")

    # Step 1: Collect tickers
    log.info("Step 1: Collecting unique US tickers …")
    tickers = collect_us_tickers()
    if not tickers:
        log.error("No tickers found — check universe CSV files.")
        sys.exit(1)

    # Step 2: Build CIK map
    log.info("Step 2a: Building CIK map …")
    ticker_cik_map = build_cik_map(tickers)
    if not ticker_cik_map:
        log.error("Could not map any tickers to CIKs — aborting.")
        sys.exit(1)

    # Step 2b: Fetch XBRL facts
    log.info(
        "Step 2b: Fetching XBRL facts for %d tickers (est. ~%.0f s min) …",
        len(ticker_cik_map),
        len(ticker_cik_map) * _SLEEP,
    )
    edgar_df = fetch_xbrl_facts(ticker_cik_map)

    if edgar_df.empty:
        log.error("No EDGAR data retrieved — aborting.")
        sys.exit(1)

    # Save processed parquet
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    edgar_df.to_parquet(PROCESSED_PATH, index=False)
    log.info("Saved %d records to %s", len(edgar_df), PROCESSED_PATH)

    # Step 3: PIT ingest
    log.info("Step 3: Ingesting into PIT store …")
    ingest_to_pit(edgar_df)

    # Step 4: Summary
    print_summary(edgar_df, ticker_cik_map, len(ticker_cik_map))


if __name__ == "__main__":
    main()
