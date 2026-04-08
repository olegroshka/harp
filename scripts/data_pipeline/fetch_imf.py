"""Fetch IMF macro signals via the IMF DataMapper API and ingest into the SMIM PIT store.

Uses the IMF DataMapper API (no key required):
    https://www.imf.org/external/datamapper/api/v1/{indicator}/{country}

Coverage:
  - Real GDP growth (NGDP_RPCH)      annual, US/UK/DE/JP  → Layer 0 exogenous
  - CPI inflation (PCPIPCH)          annual, US/UK/DE/JP  → Layer 0 exogenous
  - Current account % GDP (BCA_NGDPDP) annual, US/UK     → BOP signal
  - General govt balance % GDP (GGB_NGDP) annual, US/UK  → fiscal signal
  - Unemployment rate (LUR)          annual, US/UK        → labour market
  - Total investment % GDP (NID_NGDP) annual, US/UK/DE/JP → investment signal

Note: DataMapper provides annual WEO projections + history (2000–2030).
For higher-frequency IFS data (monthly/quarterly), the IMF SDMX endpoint is
sometimes inaccessible. If IFS becomes accessible at:
  https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/...
the script will automatically use it; otherwise it falls back to DataMapper only.

Usage:
    uv run python scripts/smim/smim_fetch_imf.py

Outputs:
    data/smim/raw/imf/<indicator>.parquet   — per-indicator raw data
    data/smim/processed/imf_macro.parquet  — unified normalised table
    data/smim/pit_store/imf.parquet        — PIT store shard
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from harp.data.pit_store import PointInTimeStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── IMF DataMapper API ─────────────────────────────────────────────────────────
# Annual WEO indicators, 1980–2030 (includes projections)

_DATAMAPPER_BASE = "https://www.imf.org/external/datamapper/api/v1"

# (indicator_code, description, countries_csv, pub_lag_days)
DATAMAPPER_SERIES: list[tuple[str, str, str, int]] = [
    # Real GDP growth (%): quarterly feel via FRED GDP is better for signal, but WEO is authoritative
    ("NGDP_RPCH",   "real_gdp_growth_pct",        "USA,GBR,DEU,JPN", 365),
    # CPI inflation (%)
    ("PCPIPCH",     "cpi_inflation_pct",           "USA,GBR,DEU,JPN", 365),
    # Current account balance (USD billions) — BCA works; BCA_NGDPDP not in DataMapper
    ("BCA",          "current_account_usd_bn",       "USA,GBR",         365),
    # General government net lending (% of GDP) — fiscal balance
    ("GGXCNL_NGDP",  "govt_net_lending_pct_gdp",    "USA,GBR",         365),
    # General government gross debt (% of GDP)
    ("GGXWDG_NGDP",  "govt_gross_debt_pct_gdp",     "USA,GBR",         365),
    # Unemployment rate
    ("LUR",          "unemployment_rate",            "USA,GBR",         365),
    # GDP (PPP, international $, billions) — cross-country comparison
    ("PPPGDP",       "gdp_ppp_bn_intl",             "USA,GBR,DEU,JPN", 365),
]

# ── Optional: IMF CompactData SDMX (higher-frequency IFS data) ────────────────
# Tried only if the endpoint is accessible (timeouts are common).

_SDMX_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"

IFS_SERIES: list[tuple[str, str, str, str, int]] = [
    # (freq, countries, indicator, label, pub_lag_days)
    ("Q", "US+GB+DE+JP", "NGDP_R_XDC", "ifs_real_gdp",     90),
    ("M", "US+GB+DE+JP", "PCPI_IX",    "ifs_cpi_index",     30),
    ("M", "US+GB+DE+JP", "FPOLM_PA",   "ifs_policy_rate",   30),
    ("Q", "US+GB",        "BCA_BP6_USD","ifs_ca_usd",        90),
]
_SDMX_TIMEOUT = 20   # keep low so we don't block if unreachable

OBS_START_YEAR = 2000

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR = _ROOT / "data" / "raw" / "imf"
PROCESSED_PATH = _ROOT / "data" / "processed" / "imf_macro.parquet"
PIT_DIR = _ROOT / "data" / "pit_store"

# ── DataMapper fetch ───────────────────────────────────────────────────────────

def _datamapper_country_map() -> dict[str, str]:
    """Map ISO 3166-1 alpha-3 → alpha-2 for consistent actor_id."""
    return {"USA": "US", "GBR": "GB", "DEU": "DE", "JPN": "JP",
            "FRA": "FR", "CAN": "CA", "CHN": "CN", "BRA": "BR"}


def fetch_datamapper(session: requests.Session) -> pd.DataFrame:
    """Fetch all WEO indicators via DataMapper API. Returns combined DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    country_map = _datamapper_country_map()
    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for indicator, label, countries_csv, pub_lag in DATAMAPPER_SERIES:
        countries = countries_csv.replace(",", "/")
        url = f"{_DATAMAPPER_BASE}/{indicator}/{countries}"
        log.info("IMF DataMapper: %s (%s) …", indicator, label)
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as exc:
            log.warning("  network error: %s", exc)
            failed.append(indicator)
            continue

        if resp.status_code != 200:
            log.warning("  HTTP %d", resp.status_code)
            failed.append(indicator)
            continue

        try:
            data = resp.json()
        except Exception as exc:
            log.warning("  JSON parse error: %s", exc)
            failed.append(indicator)
            continue

        ind_values = data.get("values", {}).get(indicator, {})
        if not ind_values:
            log.warning("  no values in response")
            failed.append(indicator)
            continue

        rows: list[dict] = []
        for country_code, year_dict in ind_values.items():
            actor_id = country_map.get(country_code, country_code)
            for year_str, val in year_dict.items():
                try:
                    year = int(year_str)
                except ValueError:
                    continue
                if year < OBS_START_YEAR or val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "actor_id":   actor_id,
                    "signal_id":  indicator,
                    "event_date": pd.Timestamp(f"{year}-12-31"),
                    "value":      fval,
                })

        if not rows:
            log.warning("  no rows after parsing")
            failed.append(indicator)
            continue

        df = pd.DataFrame(rows)
        df["pub_date"] = df["event_date"] + pd.Timedelta(days=pub_lag)
        df["vintage_id"] = None

        raw_path = RAW_DIR / f"{indicator}.parquet"
        df.to_parquet(raw_path, index=False)
        all_frames.append(df)
        log.info("  → %d rows, countries=%s, years=%d–%d",
                 len(df),
                 sorted(df["actor_id"].unique()),
                 df["event_date"].dt.year.min(),
                 df["event_date"].dt.year.max())
        time.sleep(0.3)

    if failed:
        log.warning("Failed DataMapper indicators: %s", failed)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# ── SDMX fetch (best-effort, may time out) ────────────────────────────────────

def _parse_compact_sdmx(data: dict, indicator_code: str, pub_lag: int) -> list[dict]:
    """Parse IMF CompactData SDMX-JSON response. Returns list of row dicts."""
    rows: list[dict] = []
    try:
        dataset = data["CompactData"]["DataSet"]
    except (KeyError, TypeError):
        return rows

    series_list = dataset.get("Series") or []
    if isinstance(series_list, dict):
        series_list = [series_list]

    for series in series_list:
        meta = {k.lstrip("@"): v for k, v in series.items() if k.startswith("@")}
        obs_list = series.get("Obs") or []
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        country = meta.get("REF_AREA", "UNKNOWN")
        for obs in obs_list:
            period = obs.get("@TIME_PERIOD")
            value  = obs.get("@OBS_VALUE")
            if period is None or value is None:
                continue
            # Parse period: "2005-Q1" → first day of quarter, "2005-01" → first day
            try:
                if "Q" in period:
                    y, q = period.split("-Q")
                    month = (int(q) - 1) * 3 + 1
                    ts = pd.Timestamp(f"{y}-{month:02d}-01")
                elif len(period) == 4:
                    ts = pd.Timestamp(f"{period}-01-01")
                else:
                    ts = pd.to_datetime(period)
            except Exception:
                continue
            try:
                fval = float(value)
            except (ValueError, TypeError):
                continue
            rows.append({
                "actor_id":   country,
                "signal_id":  indicator_code,
                "event_date": ts,
                "pub_date":   ts + pd.Timedelta(days=pub_lag),
                "value":      fval,
                "vintage_id": None,
            })
    return rows


def fetch_ifs_sdmx(session: requests.Session) -> pd.DataFrame:
    """Try to fetch IFS quarterly data via CompactData SDMX. Best-effort only."""
    frames: list[pd.DataFrame] = []

    for freq, countries, indicator, label, pub_lag in IFS_SERIES:
        key = f"{freq}.{countries}.{indicator}"
        url = f"{_SDMX_BASE}/IFS/{key}?startPeriod=2000&endPeriod=2026"
        log.info("IMF SDMX IFS: %s (%s) …", indicator, label)
        try:
            resp = session.get(url, headers={"Accept": "application/json"},
                               timeout=_SDMX_TIMEOUT)
        except requests.RequestException as exc:
            log.info("  IFS SDMX unreachable (skipping): %s", exc)
            continue

        if resp.status_code != 200:
            log.info("  IFS SDMX HTTP %d (skipping)", resp.status_code)
            continue

        try:
            data = resp.json()
        except Exception:
            log.info("  IFS SDMX non-JSON response (skipping)")
            continue

        rows = _parse_compact_sdmx(data, indicator, pub_lag)
        if rows:
            df = pd.DataFrame(rows)
            frames.append(df)
            log.info("  → %d rows", len(df))
        else:
            log.info("  → no observations")
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Normalise ──────────────────────────────────────────────────────────────────

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct column types and remove duplicates."""
    if df.empty:
        return df
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["value"] = df["value"].astype("float64")
    df["actor_id"] = df["actor_id"].astype(str)
    df["signal_id"] = df["signal_id"].astype(str)
    if "vintage_id" not in df.columns:
        df["vintage_id"] = None
    return df


# ── PIT ingest ─────────────────────────────────────────────────────────────────

def ingest_to_pit(df: pd.DataFrame) -> None:
    pit_df = df.copy()
    pit_df["source"] = "imf"
    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated at %s", PIT_DIR)


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n{'=' * 60}")
        print("IMF fetch summary: NO DATA RETRIEVED")
        print("=" * 60 + "\n")
        return

    series_count = df["signal_id"].nunique()
    obs_count = len(df)
    countries = sorted(df["actor_id"].unique())
    earliest = df["event_date"].min().date()
    latest = df["event_date"].max().date()
    sources_used = sorted(df.get("source_type", pd.Series(["datamapper"])).unique())

    print(f"\n{'=' * 60}")
    print("IMF fetch summary")
    print("=" * 60)
    print(f"  Series fetched       : {series_count}")
    print(f"  Total observations   : {obs_count:,}")
    print(f"  Countries            : {countries}")
    print(f"  Event date range     : {earliest} to {latest}")
    print(f"  Processed parquet    : {PROCESSED_PATH}")
    print(f"  PIT store            : {PIT_DIR}")
    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "SMIM-DataFetch/1.0 (research)"})

    log.info("Step 1a: Fetching IMF WEO indicators via DataMapper API …")
    dm_df = fetch_datamapper(session)

    log.info("Step 1b: Attempting IMF IFS SDMX (best-effort, may be unreachable) …")
    ifs_df = fetch_ifs_sdmx(session)

    frames = [f for f in [dm_df, ifs_df] if not f.empty]
    if not frames:
        log.error("No IMF data retrieved from either DataMapper or IFS SDMX.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = normalise(combined)

    # De-duplicate: prefer IFS (higher freq) over DataMapper for same signal/date
    combined = combined.sort_values(
        ["actor_id", "signal_id", "event_date", "pub_date"]
    ).drop_duplicates(
        subset=["actor_id", "signal_id", "event_date"], keep="last"
    )

    log.info("Step 2: Saving %d rows to processed parquet …", len(combined))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PROCESSED_PATH, index=False)

    log.info("Step 3: Ingesting into PIT store …")
    ingest_to_pit(combined)

    print_summary(combined)


if __name__ == "__main__":
    main()
