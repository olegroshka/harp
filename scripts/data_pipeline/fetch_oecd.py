"""Fetch OECD SDMX 3.0 macro signals and ingest into the SMIM PIT store.

Uses OECD SDMX 3.0 REST API (no key required) with **explicit dimension keys**.
The previous "all" key approach returned a server-limited subset (244 rows vs ~2,000
expected). Explicit keys are required to retrieve the full time series.

Fetches:
  DF_CLI (Composite Leading Indicators) — monthly for USA + GBR
    Key: USA+GBR.M.LI+BCICP+CCICP.IX._Z.AA.IX._Z.H
    Measures: LI, BCICP, CCICP (Amplitude Adjusted, Methodology H)
    Expected: ~1,700+ rows, 2000-01 to present

  DF_QNA_EXPENDITURE_CAPITA (Quarterly GDP per capita) — quarterly for USA + GBR
    Key: Q.Y.USA+GBR.S1.S1.B1GQ_POP._Z._Z._Z.USD_PPP_PS.LR.LA.T0102
    Transaction: B1GQ_POP, Chain-linked volumes, Log-annual transformation
    Expected: ~200 rows, 2000-Q1 to present

Usage:
    uv run python scripts/smim/smim_fetch_oecd.py

Outputs:
    data/smim/raw/oecd/<dataflow>.parquet           — raw per-dataflow data
    data/smim/processed/oecd_macro.parquet          — unified normalised table
    data/smim/pit_store/oecd.parquet                — PIT store shard
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

# ── API endpoints ──────────────────────────────────────────────────────────────

_BASE = "https://sdmx.oecd.org/public/rest/data"
_HEADERS = {"Accept": "application/json"}
_RATE_LIMIT_PAUSE = 1.0

# ── Series definitions ─────────────────────────────────────────────────────────
# Each entry specifies a dataflow to fetch via the `all` key, then filter to
# the desired REF_AREA, MEASURE, ADJUSTMENT etc. post-fetch.

OECD_DATAFLOWS: list[dict] = [
    {
        "agency":       "OECD.SDD.STES",
        "dataflow":     "DSD_STES@DF_CLI",
        "version":      "4.0",
        "label":        "composite_leading_indicators",
        "freq":         "M",
        "start":        "2000-01",
        "end":          "2026-03",
        "pub_lag":      45,
        # Explicit dimension key (9 dims): REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.TRANSFORMATION.TIME_HORIZ.METHODOLOGY
        # METHODOLOGY=H is required for USA/GBR CLI data; "all" key silently omitted these series.
        "explicit_key": "USA+GBR.M.LI+BCICP+CCICP.IX._Z.AA.IX._Z.H",
        "filters": {
            "REF_AREA":    {"USA", "GBR"},
            "FREQ":        {"M"},
            "MEASURE":     {"LI", "BCICP", "CCICP"},
            "ADJUSTMENT":  {"AA"},
        },
        "actor_dim":    "REF_AREA",
        "signal_dim":   "MEASURE",
    },
    {
        "agency":       "OECD.SDD.NAD",
        "dataflow":     "DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA",
        "version":      "1.1",
        "label":        "qna_gdp_per_capita",
        "freq":         "Q",
        "start":        "2000-Q1",
        "end":          "2026-Q1",
        "pub_lag":      75,
        # Explicit key (13 dims): FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
        # TRANSFORMATION=LA (log-annual); original config had LG which is not a valid value.
        "explicit_key": "Q.Y.USA+GBR.S1.S1.B1GQ_POP._Z._Z._Z.USD_PPP_PS.LR.LA.T0102",
        "filters": {
            "REF_AREA":     {"USA", "GBR"},
            "FREQ":         {"Q"},
            "TRANSACTION":  {"B1GQ_POP"},
            "PRICE_BASE":   {"LR"},
        },
        "actor_dim":    "REF_AREA",
        "signal_dim":   "TRANSACTION",
    },
]

# Country code normalisation (OECD uses alpha-3, we store alpha-2)
_COUNTRY_MAP = {
    "USA": "US", "GBR": "GB", "DEU": "DE", "JPN": "JP",
    "FRA": "FR", "CAN": "CA", "AUS": "AU", "NZL": "NZ",
}

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR = _ROOT / "data" / "raw" / "oecd"
PROCESSED_PATH = _ROOT / "data" / "processed" / "oecd_macro.parquet"
PIT_DIR = _ROOT / "data" / "pit_store"


# ── SDMX 3.0 parser ───────────────────────────────────────────────────────────

def _decode_series_key(
    key_str: str,
    series_dims: list[dict],
) -> dict[str, str]:
    """Decode a colon-separated SDMX 3.0 series key to a dict of {dim_id: code}."""
    indices = [int(x) for x in key_str.split(":")]
    result: dict[str, str] = {}
    for i, dim in enumerate(series_dims):
        if i >= len(indices):
            break
        idx = indices[i]
        values = dim.get("values", [])
        if idx < len(values):
            result[dim["id"]] = values[idx].get("id", "")
    return result


def _period_to_timestamp(period: str) -> pd.Timestamp | None:
    """Convert OECD period strings to Timestamp (start of period)."""
    period = period.strip()
    try:
        if "-Q" in period:
            y, q = period.split("-Q")
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(f"{y}-{month:02d}-01")
        if len(period) == 4:
            return pd.Timestamp(f"{period}-01-01")
        return pd.to_datetime(period)
    except Exception:
        return None


def _passes_filters(dims: dict[str, str], filters: dict[str, set[str]]) -> bool:
    """Return True if all filter dimensions match."""
    for dim_id, allowed in filters.items():
        if dim_id in dims and dims[dim_id] not in allowed:
            return False
    return True


def parse_sdmx3_response(
    data: dict,
    cfg: dict,
) -> list[dict]:
    """Parse OECD SDMX 3.0 JSON response into flat row dicts.

    Returns rows with: actor_id, signal_id, event_date, value, dim_*
    """
    rows: list[dict] = []

    structures = data.get("data", {}).get("structures", [])
    datasets = data.get("data", {}).get("dataSets", [])

    if not structures or not datasets:
        return rows

    struct = structures[0]
    series_dims = struct.get("dimensions", {}).get("series", [])
    obs_dims    = struct.get("dimensions", {}).get("observation", [])

    # Find which observation dimension is TIME_PERIOD
    time_dim_idx = next(
        (i for i, d in enumerate(obs_dims) if "TIME" in d.get("id", "").upper()),
        0,
    )
    time_values = obs_dims[time_dim_idx].get("values", []) if obs_dims else []
    time_map = {str(i): v.get("id", "") for i, v in enumerate(time_values)}

    filters   = cfg.get("filters", {})
    actor_dim = cfg.get("actor_dim", "REF_AREA")
    sig_dim   = cfg.get("signal_dim", "MEASURE")
    pub_lag   = cfg.get("pub_lag", 30)

    for dataset in datasets:
        series_dict = dataset.get("series", {}) or dataset.get("observations", {})
        for key_str, series_data in series_dict.items():
            dim_values = _decode_series_key(key_str, series_dims)

            if not _passes_filters(dim_values, filters):
                continue

            raw_country = dim_values.get(actor_dim, "UNKNOWN")
            actor_id    = _COUNTRY_MAP.get(raw_country, raw_country)
            signal_id   = dim_values.get(sig_dim, "UNKNOWN")

            # Observations: {"0": [value, attr...], "1": [...]}
            obs = series_data.get("observations", {}) if isinstance(series_data, dict) else {}
            for time_idx_str, obs_data in obs.items():
                period_str = time_map.get(time_idx_str, "")
                ts = _period_to_timestamp(period_str)
                if ts is None:
                    continue
                val = obs_data[0] if obs_data else None
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue
                rows.append({
                    "actor_id":   actor_id,
                    "signal_id":  signal_id,
                    "event_date": ts,
                    "pub_date":   ts + pd.Timedelta(days=pub_lag),
                    "value":      fval,
                    "vintage_id": None,
                })

    return rows


# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_dataflow(cfg: dict, session: requests.Session) -> pd.DataFrame:
    """Fetch one OECD SDMX 3.0 dataflow using an explicit dimension key.

    Explicit keys are required because the "all" key returns a server-limited subset
    that silently omits USA/GBR CLI series with METHODOLOGY=H.
    """
    agency       = cfg["agency"]
    dataflow     = cfg["dataflow"]
    version      = cfg["version"]
    label        = cfg["label"]
    start        = cfg["start"]
    end          = cfg["end"]
    explicit_key = cfg.get("explicit_key", "all")

    url = (
        f"{_BASE}/{agency},{dataflow},{version}/{explicit_key}"
        f"?format=jsondata&startPeriod={start}&endPeriod={end}"
    )
    log.info("OECD: fetching %s …", label)
    log.debug("  GET %s", url)

    try:
        resp = session.get(url, headers=_HEADERS, timeout=90)
    except requests.RequestException as exc:
        log.warning("  network error: %s", exc)
        return pd.DataFrame()

    if resp.status_code != 200:
        log.warning("  HTTP %d", resp.status_code)
        return pd.DataFrame()

    try:
        data = resp.json()
    except Exception as exc:
        log.warning("  JSON parse error: %s", exc)
        return pd.DataFrame()

    rows = parse_sdmx3_response(data, cfg)
    if not rows:
        log.warning("  No rows after parsing / filtering")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info(
        "  → %d rows, actors=%s, signals=%s",
        len(df),
        sorted(df["actor_id"].unique()),
        sorted(df["signal_id"].unique()),
    )
    return df


# ── Step 1: Fetch all ──────────────────────────────────────────────────────────

def fetch_all() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "SMIM-DataFetch/1.0 (research)"})
    all_frames: list[pd.DataFrame] = []

    for cfg in OECD_DATAFLOWS:
        df = fetch_dataflow(cfg, session)
        time.sleep(_RATE_LIMIT_PAUSE)
        if not df.empty:
            raw_path = RAW_DIR / f"{cfg['dataflow'].replace('@','_').replace('/','_')}.parquet"
            df.to_parquet(raw_path, index=False)
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined["event_date"] = pd.to_datetime(combined["event_date"]).dt.tz_localize(None)
    combined["pub_date"]   = pd.to_datetime(combined["pub_date"]).dt.tz_localize(None)
    combined["value"]      = combined["value"].astype("float64")
    return combined


# ── PIT ingest ─────────────────────────────────────────────────────────────────

def ingest_to_pit(df: pd.DataFrame) -> None:
    pit_df = df.copy()
    pit_df["source"] = "oecd"
    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated at %s", PIT_DIR)


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n{'=' * 60}")
        print("OECD SDMX fetch summary: NO DATA RETRIEVED")
        print("=" * 60 + "\n")
        return

    series_count = df["signal_id"].nunique()
    obs_count = len(df)
    countries = sorted(df["actor_id"].unique())
    earliest = df["event_date"].min().date()
    latest = df["event_date"].max().date()

    print(f"\n{'=' * 60}")
    print("OECD SDMX fetch summary")
    print("=" * 60)
    print(f"  Series (unique signal_id) : {series_count}")
    print(f"  Total observations        : {obs_count:,}")
    print(f"  Countries (actor_id)      : {countries}")
    print(f"  Event date range          : {earliest} to {latest}")
    print(f"  Processed parquet         : {PROCESSED_PATH}")
    print(f"  PIT store                 : {PIT_DIR}")
    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Step 1: Fetching OECD CLI + QNA via SDMX 3.0 …")
    df = fetch_all()

    if df.empty:
        log.error("No OECD data retrieved.")
        log.error("  API: https://sdmx.oecd.org/public/rest/")
        sys.exit(1)

    log.info("Step 2: Saving %d rows to processed parquet …", len(df))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    log.info("Step 3: Ingesting into PIT store …")
    ingest_to_pit(df)

    print_summary(df)


if __name__ == "__main__":
    main()
