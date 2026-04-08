"""Fetch FRED/ALFRED macro signals and ingest into the SMIM PIT store.

Usage:
    uv run python scripts/smim/smim_fetch_fred.py

Requires:
    FRED_API_KEY environment variable set.
    pip install fredapi (already in project deps)

Outputs:
    data/smim/raw/fred/<series_id>.parquet      — raw FRED responses
    data/smim/raw/fred/<series_id>_alfred.parquet — ALFRED vintage releases
    data/smim/processed/fred_signals.parquet    — unified normalised table
    data/smim/pit_store/fred.parquet            — ingested into PIT store
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure the src package is importable when run as a script
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from harp.data.pit_store import PointInTimeStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Series lists ──────────────────────────────────────────────────────────────

MACRO_SERIES: list[str] = [
    # GDP / Output
    "GDP",
    "GDPC1",
    # Industrial production
    "INDPRO",
    # Employment / PMI proxy
    "MANEMP",
    # Unemployment
    "UNRATE",
    # Inflation
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    # Interest rates
    "FEDFUNDS",
    "DFF",
    # Yield curve
    "GS10",
    "GS2",
    "T10Y2Y",
    # Credit spreads
    "BAA10Y",
    "BAMLH0A0HYM2",
    # Volatility / stress
    "VIXCLS",
    "STLFSI2",
    # Sentiment / leading
    "UMCSENT",
    "USSLIND",
    # Housing
    "HOUST",
    # Money supply / credit
    "M2SL",
    "TOTBKCR",
    # Global liquidity proxy
    "DTWEXBGS",
    # Energy
    "DCOILWTICO",
    "DCOILBRENTEU",
    "GASREGW",
    # Sector-specific
    "CPIMEDSL",      # CPI medical care (Healthcare)
    "DRCCLACBS",     # CC delinquency rate (Financials)
]

# Series for which we pull full ALFRED vintage history
ALFRED_SERIES: list[str] = [
    "GDP",
    "UNRATE",
    "CPIAUCSL",
    "INDPRO",
    "FEDFUNDS",
]

OBS_START = "2000-01-01"

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR = _ROOT / "data" / "raw" / "fred"
PROCESSED_PATH = _ROOT / "data" / "processed" / "fred_signals.parquet"
PIT_DIR = _ROOT / "data" / "pit_store"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def _fetch_current_series(fred: object, series_id: str) -> pd.DataFrame | None:
    """Fetch current (non-vintage) observations for one series.

    Returns a tidy DataFrame with columns:
        series_id, event_date, value
    or None if the series is unavailable.
    """
    try:
        s = fred.get_series(series_id, observation_start=OBS_START)  # type: ignore[attr-defined]
    except Exception as exc:
        log.warning("FRED: failed to fetch %s — %s", series_id, exc)
        return None

    if s is None or len(s) == 0:
        log.warning("FRED: %s returned no data", series_id)
        return None

    s.index = pd.to_datetime(s.index).tz_localize(None)
    df = s.reset_index()
    df.columns = pd.Index(["event_date", "value"])
    df["series_id"] = series_id
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["series_id", "event_date", "value"]]


def _fetch_alfred_vintages(fred: object, series_id: str) -> pd.DataFrame | None:
    """Fetch all ALFRED vintages for one series via get_series_all_releases().

    Returns a tidy DataFrame with columns:
        series_id, event_date, value, pub_date, vintage_id
    or None on failure.
    """
    try:
        df = fred.get_series_all_releases(series_id)  # type: ignore[attr-defined]
    except Exception as exc:
        log.warning("ALFRED: failed to fetch vintages for %s — %s", series_id, exc)
        return None

    if df is None or df.empty:
        log.warning("ALFRED: %s returned no vintage data", series_id)
        return None

    # fredapi columns: date, realtime_start, realtime_end, value
    df = df.rename(columns={"date": "event_date", "realtime_start": "pub_date"})
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    df["vintage_id"] = df["pub_date"].dt.strftime("%Y-%m-%d")

    # Filter to OBS_START
    df = df[df["event_date"] >= OBS_START]
    return df[["series_id", "event_date", "value", "pub_date", "vintage_id"]]


# ── Step 1: Fetch ─────────────────────────────────────────────────────────────

def fetch_all(fred: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch current series and ALFRED vintages.

    Returns:
        (current_df, alfred_df) — both in tidy format.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    current_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for sid in MACRO_SERIES:
        log.info("Fetching current: %s", sid)
        df = _fetch_current_series(fred, sid)
        if df is not None:
            raw_path = RAW_DIR / f"{sid}.parquet"
            df.to_parquet(raw_path, index=False)
            current_frames.append(df)
        else:
            failed.append(sid)

    if failed:
        log.warning("Failed series (will be skipped): %s", failed)

    alfred_frames: list[pd.DataFrame] = []

    for sid in ALFRED_SERIES:
        log.info("Fetching ALFRED vintages: %s", sid)
        df = _fetch_alfred_vintages(fred, sid)
        if df is not None:
            raw_path = RAW_DIR / f"{sid}_alfred.parquet"
            df.to_parquet(raw_path, index=False)
            alfred_frames.append(df)

    current_df = pd.concat(current_frames, ignore_index=True) if current_frames else pd.DataFrame()
    alfred_df = pd.concat(alfred_frames, ignore_index=True) if alfred_frames else pd.DataFrame()

    return current_df, alfred_df


# ── Step 2: Normalise ─────────────────────────────────────────────────────────

def build_unified(current_df: pd.DataFrame, alfred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge current and ALFRED data into the unified schema.

    Columns: series_id, event_date, value, pub_date, vintage_id

    For non-vintaged series: pub_date = event_date + 30 days (conservative).
    For ALFRED series: pub_date comes from the vintage realtime_start.
    """
    rows: list[pd.DataFrame] = []

    # Current series — exclude those that have ALFRED coverage (ALFRED is more precise)
    alfred_ids = set(alfred_df["series_id"].unique()) if not alfred_df.empty else set()
    non_alfred = (
        current_df[~current_df["series_id"].isin(alfred_ids)].copy()
        if not current_df.empty else pd.DataFrame()
    )
    if not non_alfred.empty:
        non_alfred["pub_date"] = non_alfred["event_date"] + pd.DateOffset(days=30)
        non_alfred["vintage_id"] = None
        rows.append(non_alfred)

    # ALFRED vintages
    if not alfred_df.empty:
        rows.append(alfred_df)

    if not rows:
        return pd.DataFrame()

    unified = pd.concat(rows, ignore_index=True)
    unified["event_date"] = pd.to_datetime(unified["event_date"]).dt.tz_localize(None)
    unified["pub_date"] = pd.to_datetime(unified["pub_date"]).dt.tz_localize(None)
    unified["value"] = unified["value"].astype("float64")

    # Rename series_id → signal_id to match PIT schema
    unified = unified.rename(columns={"series_id": "signal_id"})
    return unified[["signal_id", "event_date", "value", "pub_date", "vintage_id"]]


# ── Step 3: PIT ingest ────────────────────────────────────────────────────────

def ingest_to_pit(unified: pd.DataFrame) -> None:
    """Ingest unified DataFrame into the PIT store."""
    pit_df = unified.copy()
    pit_df["actor_id"] = "MACRO"  # macro/global series have no single actor
    pit_df["source"] = "fred"

    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated at %s", PIT_DIR)


# ── Step 4: Summary ───────────────────────────────────────────────────────────

def print_summary(current_df: pd.DataFrame, alfred_df: pd.DataFrame, unified: pd.DataFrame) -> None:
    fetched_ids = (
        set(current_df["series_id"].unique()) if not current_df.empty else set()
    )
    total_series = len(fetched_ids)
    total_obs = len(unified)
    alfred_series = (
        alfred_df["series_id"].nunique() if not alfred_df.empty else 0
    )

    if not unified.empty:
        earliest = unified["event_date"].min().date()
        latest = unified["event_date"].max().date()
    else:
        earliest = latest = "n/a"

    failed = set(MACRO_SERIES) - fetched_ids

    print("\n" + "=" * 60)
    print("FRED/ALFRED fetch summary")
    print("=" * 60)
    print(f"  Total series fetched     : {total_series} / {len(MACRO_SERIES)}")
    print(f"  Total observations       : {total_obs:,}")
    print(f"  Date range               : {earliest} to {latest}")
    print(f"  ALFRED vintage series    : {alfred_series}")
    print(f"  Processed parquet        : {PROCESSED_PATH}")
    print(f"  PIT store                : {PIT_DIR}")
    if failed:
        print(f"  Failed / skipped series  : {sorted(failed)}")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = _load_api_key()

    try:
        from fredapi import Fred  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "fredapi is required. Install with: pip install fredapi"
        ) from exc

    fred = Fred(api_key=api_key)

    # Step 1: Fetch
    log.info("Step 1: Fetching %d FRED series + %d ALFRED vintage series …",
             len(MACRO_SERIES), len(ALFRED_SERIES))
    current_df, alfred_df = fetch_all(fred)

    # Step 2: Normalise
    log.info("Step 2: Normalising to unified schema …")
    unified = build_unified(current_df, alfred_df)

    if unified.empty:
        log.error("No data retrieved — aborting.")
        sys.exit(1)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(PROCESSED_PATH, index=False)
    log.info("Saved processed signals to %s", PROCESSED_PATH)

    # Step 3: PIT ingest
    log.info("Step 3: Ingesting into PIT store …")
    ingest_to_pit(unified)

    # Step 4: Summary
    print_summary(current_df, alfred_df, unified)


if __name__ == "__main__":
    main()
