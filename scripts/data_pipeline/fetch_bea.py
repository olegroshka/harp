"""Fetch BEA Input-Output tables and ingest supply-chain coefficients into the SMIM PIT store.

Two modes:
  1. API (requires BEA_API_KEY env var) — fetches Use Table (TableID=259) via JSON API
  2. Direct download (no key needed)    — downloads published Excel files from BEA website

The "Use Table" (Before Redefinitions, Producers' Prices) gives industry × commodity
flow amounts. We derive direct-requirements coefficients:
    coeff[i→j] = flow[i,j] / total_output[j]

Sector mapping (BEA NAICS-based industry codes → SMIM sector labels):
    Energy      : 211,213,324
    Technology  : 334,511,518,519
    Financials  : 521,522,523,524,525
    Healthcare  : 621,622,623,624
    Industrials : 331,332,333,336,337

A1 compliance: pub_date = year-end + 548 days (~18-month BEA publication lag).

Usage:
    uv run python scripts/smim/smim_fetch_bea.py

    # With API key (more complete data):
    BEA_API_KEY=<your_key> uv run python scripts/smim/smim_fetch_bea.py

Outputs:
    data/smim/raw/bea/use_table_<year>.parquet       — raw per-year use tables
    data/smim/raw/bea/direct_req_<year>.parquet      — derived coefficients
    data/smim/processed/bea_io_tables.parquet        — sector-mapped coefficient table
    data/smim/pit_store/bea.parquet                  — PIT store shard
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

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

# ── Configuration ──────────────────────────────────────────────────────────────

BEA_API_URL = "https://apps.bea.gov/api/data"

# TableID 259 = "Use of Commodities by Industries, Before Redefinitions (Producers' Prices)"
# TableID 56  = "Use of Commodities by Industries, After Redefinitions (Producers' Prices)"
USE_TABLE_ID = "259"

YEAR_START = 2010
YEAR_END   = 2024

# Publication lag: BEA releases I/O tables ~18 months after the reference year end
PUB_LAG_DAYS = 548

# ── Sector mapping (BEA NAICS-based 6-digit industry codes → sector labels) ───

# BEA uses hierarchical codes; we match by prefix
SECTOR_MAP: dict[str, str] = {
    # Energy
    "211": "Energy",      # Oil and gas extraction
    "213": "Energy",      # Support activities for mining
    "324": "Energy",      # Petroleum and coal products manufacturing
    # Technology
    "334": "Technology",  # Computer and electronic product manufacturing
    "511": "Technology",  # Publishing industries (including software)
    "518": "Technology",  # Data processing, hosting, and related services
    "519": "Technology",  # Other information services
    # Financials
    "521": "Financials",  # Monetary authorities-central bank
    "522": "Financials",  # Credit intermediation and related activities
    "523": "Financials",  # Securities, commodity contracts, investments
    "524": "Financials",  # Insurance carriers and related activities
    "525": "Financials",  # Funds, trusts, and other financial vehicles
    # Healthcare
    "621": "Healthcare",  # Ambulatory health care services
    "622": "Healthcare",  # Hospitals
    "623": "Healthcare",  # Nursing and residential care facilities
    "624": "Healthcare",  # Social assistance
    # Industrials
    "331": "Industrials", # Primary metal manufacturing
    "332": "Industrials", # Fabricated metal product manufacturing
    "333": "Industrials", # Machinery manufacturing
    "336": "Industrials", # Transportation equipment manufacturing
    "337": "Industrials", # Furniture and related product manufacturing
}

# ── Direct-download URLs (no API key needed) ───────────────────────────────────
# BEA publishes annual Use Tables as Excel workbooks.
# The 2017-framework summary table covers ~2017–2022 (most recent publication).
# We try these in order; use the first that downloads successfully.

DIRECT_DOWNLOAD_URLS: list[tuple[str, str]] = [
    # (url, description)
    (
        "https://apps.bea.gov/industry/xls/io-annual/Use_SUT_Framework_2017_2022_DET.xlsx",
        "BEA Use Table 2017 framework 2017–2022 (detailed)",
    ),
    (
        "https://apps.bea.gov/industry/xls/io-annual/Use_SUT_Framework_2007_2012_DET.xlsx",
        "BEA Use Table 2007 framework 2007–2012 (detailed)",
    ),
    (
        "https://apps.bea.gov/industry/xls/io-annual/IxC_2007_2012_DOM_DET.xlsx",
        "BEA Direct Requirements Table 2007–2012 (detailed)",
    ),
]

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR = _ROOT / "data" / "raw" / "bea"
PROCESSED_PATH = _ROOT / "data" / "processed" / "bea_io_tables.parquet"
PIT_DIR = _ROOT / "data" / "pit_store"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _map_sector(code: str) -> str | None:
    """Return sector label for a BEA industry code, or None if unmapped."""
    code = str(code).strip()
    for prefix, sector in SECTOR_MAP.items():
        if code.startswith(prefix):
            return sector
    return None


def _compute_pub_date(year: int) -> pd.Timestamp:
    """Return A1-compliant pub_date: year-end + 548 days."""
    return pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=PUB_LAG_DAYS)


def _parse_bea_api_response(data: dict) -> pd.DataFrame:
    """Parse BEA JSON API response into a flat DataFrame.

    BEA API returns Results as a list; each element has a "Data" array.
    Actual column names from API: RowCode, RowDescr, ColCode, ColDescr, DataValue.
    """
    results = data.get("BEAAPI", {}).get("Results")
    if results is None:
        raise ValueError("Missing BEAAPI.Results in response")

    # Results is a list; first element contains the actual data
    if isinstance(results, list):
        result_entry = results[0] if results else {}
    else:
        result_entry = results

    rows = result_entry.get("Data") or result_entry.get("Dimensions")
    if rows is None:
        raise ValueError(f"No Data key in Results. Available keys: {list(result_entry.keys())}")

    if isinstance(rows, dict):
        rows = [rows]

    records: list[dict] = []
    for row in rows:
        try:
            year = int(row.get("Year", 0) or 0)
            src = str(row.get("RowCode", "")).strip()
            tgt = str(row.get("ColCode", "")).strip()
            # API uses RowDescr/ColDescr (not RowDescription/ColDescription)
            src_desc = str(row.get("RowDescr", row.get("RowDescription", ""))).strip()
            tgt_desc = str(row.get("ColDescr", row.get("ColDescription", ""))).strip()
            val_str = str(row.get("DataValue", "0")).replace(",", "").strip()
            if not val_str or val_str in ("---", ""):
                continue
            records.append({
                "year":            year,
                "source_industry": src,
                "source_desc":     src_desc,
                "target_industry": tgt,
                "target_desc":     tgt_desc,
                "coefficient":     float(val_str),
                "table_id":        str(row.get("TableID", USE_TABLE_ID)),
            })
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(records)


# ── Mode A: BEA JSON API ───────────────────────────────────────────────────────

def fetch_via_api(api_key: str, session: requests.Session) -> pd.DataFrame:
    """Fetch Use Table for all years via BEA JSON API."""
    years = list(range(YEAR_START, min(YEAR_END, 2024) + 1))
    all_frames: list[pd.DataFrame] = []

    log.info("Fetching BEA Use Table (TableID=%s) for years %d–%d …",
             USE_TABLE_ID, years[0], years[-1])

    # BEA API supports up to ~20 years per request; request all at once
    params = {
        "UserID":      api_key,
        "method":      "GetData",
        "DataSetName": "InputOutput",
        "TableID":     USE_TABLE_ID,
        "Year":        ",".join(str(y) for y in years),
        "ResultFormat": "JSON",
    }
    url = f"{BEA_API_URL}?{urlencode(params)}"
    log.info("  GET %s …", url[:120])

    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("BEA API request failed: %s", exc)
        return pd.DataFrame()

    try:
        data = resp.json()
    except Exception as exc:
        log.error("BEA API JSON parse error: %s", exc)
        return pd.DataFrame()

    # Check for API-level errors
    api_error = data.get("BEAAPI", {}).get("Error")
    if api_error:
        log.error("BEA API error: %s", api_error)
        return pd.DataFrame()

    df = _parse_bea_api_response(data)
    if df.empty:
        log.warning("BEA API returned no rows")
        return df

    log.info("  Retrieved %d rows (year range %d–%d)", len(df), df["year"].min(), df["year"].max())

    # Save raw per-year files
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, grp in df.groupby("year"):
        grp.to_parquet(RAW_DIR / f"use_table_{year}.parquet", index=False)

    return df


# ── Mode B: Direct Excel download ─────────────────────────────────────────────

def _try_download_excel(url: str, desc: str, session: requests.Session) -> bytes | None:
    """Download an Excel file from BEA; return raw bytes or None on failure."""
    log.info("  Trying: %s", desc)
    log.info("  URL: %s", url)
    try:
        resp = session.get(url, timeout=120, stream=True)
    except requests.RequestException as exc:
        log.warning("  Download failed: %s", exc)
        return None
    if resp.status_code != 200:
        log.warning("  HTTP %d", resp.status_code)
        return None
    content = resp.content
    if len(content) < 1024:
        log.warning("  File too small (%d bytes) — likely not a real Excel file", len(content))
        return None
    log.info("  Downloaded %.1f MB", len(content) / 1_048_576)
    return content


def _parse_bea_use_excel(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a BEA Use Table Excel file into a tidy long-format DataFrame.

    BEA Use Tables are formatted as:
      - Rows = industries (source)
      - Columns = commodities (target)
      - Cell = dollar flow (millions)
      - First row(s) = header with column codes
      - First column(s) = row codes

    We detect the data rectangle by finding the first numeric-value cell.
    """
    try:
        xls = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
    except Exception as exc:
        log.error("Failed to open Excel file: %s", exc)
        return pd.DataFrame()

    # Try sheets that are likely to contain the main use table
    candidate_sheets = [
        s for s in xls.sheet_names
        if any(kw in s.upper() for kw in ("USE", "IO", "DETAIL", "INDUSTRY"))
    ]
    if not candidate_sheets:
        candidate_sheets = xls.sheet_names[:3]   # try first 3 sheets

    for sheet in candidate_sheets:
        log.info("  Parsing sheet: %s", sheet)
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
        except Exception as exc:
            log.warning("  Sheet parse error: %s", exc)
            continue

        # Find the header row (contains "Code" or column codes starting with digits)
        header_row_idx = None
        col_codes_row: pd.Series | None = None
        for i, row in raw.iterrows():
            non_empty = row.dropna()
            if len(non_empty) > 10:
                # Check if this row has many numeric-looking or code-like values
                codes = [str(v).strip() for v in non_empty.values[2:10]]
                if any(c[:3].isdigit() for c in codes):
                    header_row_idx = i
                    col_codes_row = row
                    break

        if col_codes_row is None:
            log.warning("  Could not detect header row in sheet %s", sheet)
            continue

        # Collect records
        records: list[dict] = []
        for i, row in raw.iterrows():
            if i <= header_row_idx:
                continue
            row_code = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
            if not row_code or not row_code[:3].replace(".", "").isdigit():
                continue
            for j, col_code in enumerate(col_codes_row):
                col_code_str = str(col_code).strip()
                if not col_code_str or not col_code_str[:3].replace(".", "").isdigit():
                    continue
                val = row.iloc[j]
                if pd.isna(val):
                    continue
                try:
                    fval = float(str(val).replace(",", ""))
                except ValueError:
                    continue
                records.append({
                    "source_industry": row_code[:6],
                    "target_industry": col_code_str[:6],
                    "flow": fval,
                })

        if records:
            log.info("  Parsed %d flow records from sheet %s", len(records), sheet)
            return pd.DataFrame(records)

    log.warning("Could not parse any usable data from the Excel file")
    return pd.DataFrame()


def _flows_to_coefficients(flow_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Convert flow matrix to direct-requirements coefficients.

    coeff[source→target] = flow[source, target] / total_output[target]

    total_output[target] = sum of all flows into target (column sum).
    """
    if flow_df.empty:
        return pd.DataFrame()

    # Pivot to matrix
    try:
        matrix = flow_df.pivot_table(
            index="source_industry",
            columns="target_industry",
            values="flow",
            aggfunc="sum",
            fill_value=0.0,
        )
    except Exception as exc:
        log.warning("Could not pivot flow matrix: %s", exc)
        return pd.DataFrame()

    col_totals = matrix.sum(axis=0)
    col_totals = col_totals.replace(0, float("nan"))
    coeff_matrix = matrix.div(col_totals, axis=1)

    # Melt back to long form
    coeff_long = coeff_matrix.reset_index().melt(
        id_vars="source_industry",
        var_name="target_industry",
        value_name="coefficient",
    )
    coeff_long = coeff_long.dropna(subset=["coefficient"])
    coeff_long = coeff_long[coeff_long["coefficient"] > 0]
    coeff_long["year"] = year
    coeff_long["table_id"] = USE_TABLE_ID
    return coeff_long


def fetch_via_direct_download(session: requests.Session) -> pd.DataFrame:
    """Fetch BEA I/O tables by downloading published Excel files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for url, desc in DIRECT_DOWNLOAD_URLS:
        content = _try_download_excel(url, desc, session)
        if content is None:
            continue

        # Cache the raw file
        filename = url.split("/")[-1]
        raw_path = RAW_DIR / filename
        raw_path.write_bytes(content)
        log.info("  Cached to %s", raw_path)

        flow_df = _parse_bea_use_excel(content)
        if flow_df.empty:
            continue

        # Try to infer years from the URL / filename
        inferred_years: list[int] = []
        import re
        m = re.findall(r"20\d\d", filename)
        if m:
            try:
                y1, y2 = int(m[0]), int(m[-1])
                inferred_years = [y for y in range(y1, y2 + 1) if YEAR_START <= y <= YEAR_END]
            except ValueError:
                pass
        if not inferred_years:
            # Default: assume most-recent year
            inferred_years = [2022]
            log.warning("  Could not infer year from filename; assuming %d", inferred_years[0])

        for year in inferred_years:
            coeff_df = _flows_to_coefficients(flow_df, year)
            if not coeff_df.empty:
                coeff_df["source_desc"] = ""
                coeff_df["target_desc"] = ""
                all_frames.append(coeff_df)

        # Only need one successful download
        break

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    log.info("Direct download: %d coefficient records", len(combined))
    return combined


# ── Normalise + sector mapping ─────────────────────────────────────────────────

def normalise(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add sector labels, A1 pub_date, and filter to mapped sectors.

    Columns out:
        source_industry, source_sector, target_industry, target_sector,
        coefficient, year, pub_date, table_id
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
    df = df.dropna(subset=["coefficient", "year"])
    df = df[df["year"] >= YEAR_START]

    # Add sector labels
    df["source_sector"] = df["source_industry"].apply(_map_sector)
    df["target_sector"] = df["target_industry"].apply(_map_sector)

    # Keep only rows where at least one end is in a mapped sector
    sector_mask = df["source_sector"].notna() | df["target_sector"].notna()
    df = df[sector_mask].copy()
    log.info("After sector filter: %d rows (from %d)", len(df), len(raw_df))

    # Add A1-compliant pub_date
    df["pub_date"] = df["year"].apply(_compute_pub_date)

    # Ensure required columns exist
    for col in ("source_desc", "target_desc"):
        if col not in df.columns:
            df[col] = ""

    return df[[
        "source_industry", "source_sector", "source_desc",
        "target_industry", "target_sector", "target_desc",
        "coefficient", "year", "pub_date", "table_id",
    ]]


# ── PIT ingest ─────────────────────────────────────────────────────────────────

def ingest_to_pit(normalised: pd.DataFrame) -> None:
    """Ingest supply-chain coefficients into the PIT store.

    Each (source_sector, target_sector, year) tuple → one observation.
    actor_id = "source_sector→target_sector"
    signal_id = "io_coefficient"
    value = mean coefficient for that sector pair / year
    """
    if normalised.empty:
        log.warning("No data to ingest into PIT store")
        return

    # Aggregate to sector-pair level (mean coefficient across all industry pairs)
    agg = (
        normalised
        .dropna(subset=["source_sector", "target_sector"])
        .groupby(["source_sector", "target_sector", "year", "pub_date"])["coefficient"]
        .mean()
        .reset_index()
    )

    pit_df = pd.DataFrame({
        "actor_id":   agg["source_sector"] + "→" + agg["target_sector"],
        "signal_id":  "io_coefficient",
        "event_date": pd.to_datetime(agg["year"].astype(str) + "-12-31"),
        "pub_date":   agg["pub_date"],
        "value":      agg["coefficient"],
        "source":     "bea",
        "vintage_id": None,
    })

    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated with %d sector-pair observations", len(pit_df))


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(raw_df: pd.DataFrame, norm_df: pd.DataFrame, mode: str) -> None:
    print(f"\n{'=' * 60}")
    print("BEA I/O fetch summary")
    print("=" * 60)
    print(f"  Mode                 : {mode}")
    if raw_df.empty:
        print("  Status               : NO DATA RETRIEVED")
        print(
            "  Note: Register for a free BEA API key at "
            "https://apps.bea.gov/API/signup/"
        )
    else:
        years = sorted(raw_df["year"].dropna().unique().astype(int)) if "year" in raw_df.columns else []
        print(f"  Raw rows             : {len(raw_df):,}")
        print(f"  Year range           : {years[0] if years else 'n/a'} – {years[-1] if years else 'n/a'}")
        print(f"  Sector-mapped rows   : {len(norm_df):,}")
        sectors = sorted(set(
            list(norm_df["source_sector"].dropna().unique())
            + list(norm_df["target_sector"].dropna().unique())
        )) if not norm_df.empty else []
        print(f"  Sectors covered      : {sectors}")
    print(f"  Processed parquet    : {PROCESSED_PATH}")
    print(f"  PIT store            : {PIT_DIR}")
    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "SMIM-DataFetch/1.0 (research)"})

    api_key = os.environ.get("BEA_API_KEY", "").strip()

    if api_key:
        log.info("BEA_API_KEY found — using BEA JSON API …")
        raw_df = fetch_via_api(api_key, session)
        mode = f"API (TableID={USE_TABLE_ID})"
    else:
        log.info("BEA_API_KEY not set — using direct Excel download …")
        log.info("  (Register for a free key at https://apps.bea.gov/API/signup/ for more complete data)")
        raw_df = fetch_via_direct_download(session)
        mode = "Direct Excel download"

    if raw_df.empty:
        log.error("No BEA data retrieved via %s", mode)
        print_summary(raw_df, pd.DataFrame(), mode)
        sys.exit(1)

    log.info("Normalising and mapping to sectors …")
    norm_df = normalise(raw_df)

    if norm_df.empty:
        log.warning("No sector-mapped rows after normalisation (all industry codes unmapped)")
        log.warning("  Raw rows: %d, year range: %s",
                    len(raw_df), sorted(raw_df.get("year", pd.Series(dtype=int)).unique()))

    log.info("Saving processed parquet …")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save full normalised table (all sector pairs present)
    norm_df.to_parquet(PROCESSED_PATH, index=False)
    log.info("Saved %d rows to %s", len(norm_df), PROCESSED_PATH)

    log.info("Ingesting into PIT store …")
    ingest_to_pit(norm_df)

    print_summary(raw_df, norm_df, mode)


if __name__ == "__main__":
    main()
