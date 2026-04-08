"""Fetch GDELT narrative intensity signals for SMIM from raw GKG 2.0 CSV files.

PRIMARY DATA SOURCE
    GDELT Global Knowledge Graph (GKG) 2.0 raw CSV files.
    Base URL: http://data.gdeltproject.org/gdeltv2/
    Files updated every 15 minutes, publicly accessible, no authentication.

WHY RAW GKG CSV INSTEAD OF DOC API
    The GDELT DOC API has two fatal problems for this backfill:
      1. Rate-limited to ~1 req/5s per IP; consecutive year-by-year absolute-date
         queries for common theme codes trigger 429s lasting hours.
      2. Several theme codes (ECON_ENERGY, ENV_ENERGY, etc.) return empty timelines
         via the DOC API even when present in the raw GKG data.
    Direct GKG CSV downloads have neither limitation.

SAMPLING STRATEGY
    For each UTC calendar day, one representative GKG snapshot is selected.
    We choose the file with a timestamp closest to 12:00 UTC for that date,
    trying slots in order of proximity to noon (12:00, 12:15, 11:45, 12:30,
    11:30, ...) and taking the first HTTP 200 response.

    MAX_SLOT_TRIES (default 20) bounds the probe count. If no file is found
    within that window (~±2.5 hours from noon), the day is recorded as missing.

    Selection types logged per run:
      exact_noon     : 12:00:00 UTC file found on first try
      fallback_nearest : a different slot found within the probe window
      missing        : no GKG file found for that UTC day

WEEKLY AGGREGATION (mathematically correct)
    The canonical weekly panel is derived from daily data:

      weekly_article_count = sum(daily_article_count)
      weekly_avg_tone      = sum(daily_avg_tone * daily_article_count)
                             / sum(daily_article_count)          [weighted mean]
      weekly_intensity     = sum(daily_article_count)
                             / sum(daily_total_docs)

    Note: weekly_avg_tone uses article-count weighting; days with zero matched
    articles are excluded from the weighted mean denominator.
    Note: weekly_intensity is NOT a simple mean of daily intensities.

COVERAGE
    GDELT GKG 2.0 began 2015-02-19. Default range: 2015-02-19 – 2025-12-31.

OUTPUTS
    data/smim/processed/gdelt_narrative_daily.parquet  — daily panel
    data/smim/processed/gdelt_narrative.parquet         — weekly panel (daily-derived)
    data/smim/pit_store/gdelt.parquet                   — PIT store (weekly)
    data/smim/cache/gdelt/daily_aggregates/             — per-day stat cache
    data/smim/cache/gdelt/daily_file_index.parquet      — file selection log

    Before overwriting the canonical weekly outputs, the previous files are moved
    to timestamped archive paths under:
      data/smim/processed/old/
      data/smim/pit_store/old/

RESUMABILITY
    Per-day aggregate parquets in daily_aggregates/ act as the cache.
    Reruns skip days whose cache file already exists (unless --force-refetch).
    Use --weekly-only to rebuild weekly from cache without any new downloads.
    Use --rebuild to reprocess all outputs from cache without re-downloading.

GKG FIELD REFERENCE (column indices, 0-based, TSV file):
    4  DocumentIdentifier  — article URL
    8  V2Themes            — "CODE,offset;CODE2,offset;..."
    14 V2Organizations     — "OrgName,offset;OrgName2,offset;..."
    15 V2Tone              — "tone,pos,neg,polarity,actref,sgref,wordcount"

Usage:
    uv run python scripts/smim/smim_fetch_gdelt.py
    uv run python scripts/smim/smim_fetch_gdelt.py --start-date 2023-01-01 --end-date 2023-12-31
    uv run python scripts/smim/smim_fetch_gdelt.py --force-refetch
    uv run python scripts/smim/smim_fetch_gdelt.py --rebuild
    uv run python scripts/smim/smim_fetch_gdelt.py --daily-only
    uv run python scripts/smim/smim_fetch_gdelt.py --weekly-only
    uv run python scripts/smim/smim_fetch_gdelt.py --workers 8
    uv run python scripts/smim/smim_fetch_gdelt.py --validate-only

Requirements:
    pip install httpx pyarrow pandas
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from harp.data.pit_store import PointInTimeStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Date range ─────────────────────────────────────────────────────────────────
GKG_V2_START  = pd.Timestamp("2015-02-19")   # GKG 2.0 true start
DEFAULT_START = pd.Timestamp("2015-02-19")
DEFAULT_END   = pd.Timestamp("2025-12-31")

# ── GKG file settings ──────────────────────────────────────────────────────────
GKG_BASE_URL   = "http://data.gdeltproject.org/gdeltv2"
MAX_SLOT_TRIES = 20  # max 15-min slots to probe per day before declaring missing

# GKG TSV column indices (0-based)
COL_DOCID  = 4
COL_THEMES = 8
COL_ORGS   = 14
COL_TONE   = 15

# ── Sector theme baskets ───────────────────────────────────────────────────────
# GKG 2.0 V2EnhancedThemes codes (column 8 in the TSV).
#
# IMPORTANT: The old simple codes (OIL, GAS, TECH, HEALTH, ECON_BANKING, etc.)
# do NOT appear in GKG 2.0 files — the GDELT taxonomy was replaced by a
# hierarchical WB_/ENV_/EPU_/TAX_ scheme starting in Feb 2015.
# The correct codes below were verified from actual GKG 2.0 file inspection.
#
# Codes verified present in GKG 2.0 files (count per ~1300-doc snapshot):
#   ENV_OIL (540), ECON_OILPRICE (120), FUELPRICES (79), ENV_NATURALGAS (54)
#   WB_507_ENERGY_AND_EXTRACTIVES (387)
#   WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES (453)
#   SOC_INNOVATION (149), TECH_AUTOMATION (46), CYBER_ATTACK (20)
#   ECON_STOCKMARKET (425), WB_1920_FINANCIAL_SECTOR_DEVELOPMENT (164)
#   EPU_CATS_FINANCIAL_REGULATION (122), ECON_DEBT (31)
#   GENERAL_HEALTH (572), MEDICAL (552), WB_1350_PHARMACEUTICALS (405)
#   UNGP_HEALTHCARE (331)
#   ECON_INFLATION (55), WB_442_INFLATION (54)
#   WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT (94)

SECTOR_CODES: dict[str, set[str]] = {
    "sector_energy": {
        "ENV_OIL",
        "ECON_OILPRICE",
        "FUELPRICES",
        "ENV_NATURALGAS",
        "WB_507_ENERGY_AND_EXTRACTIVES",
    },
    "sector_technology": {
        "WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES",
        "SOC_INNOVATION",
        "TECH_AUTOMATION",
        "CYBER_ATTACK",
    },
    "sector_financials": {
        "ECON_STOCKMARKET",
        "WB_1920_FINANCIAL_SECTOR_DEVELOPMENT",
        "EPU_CATS_FINANCIAL_REGULATION",
        "ECON_DEBT",
    },
    "sector_healthcare": {
        "GENERAL_HEALTH",
        "MEDICAL",
        "WB_1350_PHARMACEUTICALS",
        "UNGP_HEALTHCARE",
    },
    "sector_macro": {
        "ECON_INFLATION",
        "WB_442_INFLATION",
        "WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT",
    },
}

# ── Actor alias lists ──────────────────────────────────────────────────────────
# Matched case-insensitively as substrings in normalised V2Organizations entries.
# "central bank" excluded from actor_FED — too ambiguous.
# actor_SEC: GKG NLP drops "and", producing "securities exchange commission".
ACTOR_ALIASES: dict[str, list[str]] = {
    "actor_FED": [
        "federal reserve",
        "board of governors of the federal reserve",
        "fomc",
    ],
    "actor_SEC": [
        "securities exchange commission",
        "securities and exchange commission",
        "u.s. securities",
        "us securities",
    ],
    "actor_IMF": [
        "international monetary fund",
    ],
    "actor_BOE": [
        "bank of england",
    ],
}

ALL_SIGNALS = list(SECTOR_CODES.keys()) + list(ACTOR_ALIASES.keys())

# ── Paths ──────────────────────────────────────────────────────────────────────
DAILY_CACHE_DIR  = _ROOT / "data" / "cache" / "gdelt" / "daily_aggregates"
FILE_INDEX_PATH  = _ROOT / "data" / "cache" / "gdelt" / "daily_file_index.parquet"
DAILY_PROCESSED  = _ROOT / "data" / "processed" / "gdelt_narrative_daily.parquet"
WEEKLY_PROCESSED = _ROOT / "data" / "processed" / "gdelt_narrative.parquet"
PIT_DIR          = _ROOT / "data" / "pit_store"
ARCHIVE_PROC_DIR = _ROOT / "data" / "processed" / "old"
ARCHIVE_PIT_DIR  = _ROOT / "data" / "pit_store" / "old"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_client():
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("httpx required: pip install httpx") from exc
    return httpx.Client(timeout=60.0, follow_redirects=True)


def _noon_ordered_slots() -> list[str]:
    """Return all 96 daily 15-min HHMMSS slots sorted by proximity to 12:00 UTC."""
    noon = 720  # minutes from midnight
    pairs: list[tuple[int, str]] = []
    for i in range(96):
        m = i * 15
        slot = f"{m // 60:02d}{m % 60:02d}00"
        pairs.append((abs(m - noon), slot))
    return [s for _, s in sorted(pairs)]


# Pre-compute once at module load — used in every download call.
_NOON_SLOTS: list[str] = _noon_ordered_slots()


def _download_gkg_file_for_day(
    client: object,
    date: pd.Timestamp,
    max_tries: int = MAX_SLOT_TRIES,
) -> tuple[bytes | None, str | None, str]:
    """Download the GKG zip for a UTC date, choosing the slot nearest to noon.

    Returns:
        (zip_bytes, slot_hhmmss, selection_type)
        selection_type: "exact_noon" | "fallback_nearest" | "missing"
    """
    date_str = date.strftime("%Y%m%d")
    for slot in _NOON_SLOTS[:max_tries]:
        url = f"{GKG_BASE_URL}/{date_str}{slot}.gkg.csv.zip"
        try:
            resp = client.get(url, timeout=60.0)  # type: ignore[attr-defined]
            if resp.status_code == 200 and len(resp.content) > 1000:
                sel_type = "exact_noon" if slot == "120000" else "fallback_nearest"
                return resp.content, slot, sel_type
        except Exception:
            continue
    return None, None, "missing"


# ── Parse one GKG file ─────────────────────────────────────────────────────────

def _extract_theme_codes(v2themes: str) -> set[str]:
    """Extract distinct theme codes from V2Themes field.

    V2Themes format: "CODE1,offset1;CODE2,offset2;..."
    Returns set of CODE strings (part before the comma).
    """
    if not v2themes or pd.isna(v2themes):
        return set()
    codes: set[str] = set()
    for entry in v2themes.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        code = entry.split(",")[0].strip()
        if code:
            codes.add(code)
    return codes


def _extract_org_names(v2orgs: str) -> list[str]:
    """Extract normalised organisation names from V2Organizations field.

    V2Organizations format: "OrgName,offset;OrgName2,offset;..."
    Returns list of lowercased, whitespace-collapsed names.
    """
    if not v2orgs or pd.isna(v2orgs):
        return []
    names: list[str] = []
    for entry in v2orgs.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name = entry.split(",")[0].strip().lower()
        name = " ".join(name.split())
        if name:
            names.append(name)
    return names


def _extract_tone(v2tone: str) -> float | None:
    """Extract primary tone scalar from V2Tone field (first CSV value)."""
    if not v2tone or pd.isna(v2tone):
        return None
    try:
        return float(v2tone.split(",")[0])
    except (ValueError, IndexError):
        return None


def _parse_gkg_bytes(data: bytes) -> pd.DataFrame | None:
    """Unzip and parse a GKG CSV zip file.

    Returns a DataFrame with columns: doc_id, themes (set), orgs (list), tone.
    Returns None if the file cannot be parsed or is empty.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as fh:
                raw = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    dtype=str,
                    on_bad_lines="skip",
                    engine="python",
                )
    except Exception as exc:
        log.warning("parse error: %s", exc)
        return None

    if raw.shape[1] <= COL_TONE:
        log.warning("unexpected column count: %d", raw.shape[1])
        return None

    records = []
    for _, row in raw.iterrows():
        doc_id = str(row.iloc[COL_DOCID]).strip() if pd.notna(row.iloc[COL_DOCID]) else None
        if not doc_id or doc_id == "nan":
            continue
        themes = _extract_theme_codes(str(row.iloc[COL_THEMES]))
        orgs   = _extract_org_names(str(row.iloc[COL_ORGS]))
        tone   = _extract_tone(str(row.iloc[COL_TONE]))
        records.append({"doc_id": doc_id, "themes": themes, "orgs": orgs, "tone": tone})

    if not records:
        return None
    return pd.DataFrame(records)


# ── Per-snapshot signal statistics ────────────────────────────────────────────

def _compute_snapshot_stats(
    docs: pd.DataFrame,
    sector_codes: dict[str, set[str]],
    actor_aliases: dict[str, list[str]],
) -> dict[str, dict]:
    """Compute per-signal stats from one GKG snapshot (any frequency).

    Returns: {signal_id: {"article_count": int, "avg_tone": float|nan, "total_docs": int}}
    total_docs is the distinct document count in the snapshot (denominator for intensity).
    """
    docs = docs.drop_duplicates(subset=["doc_id"])
    total_docs = len(docs)
    if total_docs == 0:
        return {}

    result: dict[str, dict] = {}

    for sig, basket in sector_codes.items():
        mask  = docs["themes"].apply(lambda t: bool(t & basket))
        sub   = docs[mask]
        count = len(sub)
        tones = sub["tone"].dropna()
        result[sig] = {
            "article_count": count,
            "avg_tone":      float(tones.mean()) if len(tones) else float("nan"),
            "total_docs":    total_docs,
        }

    for sig, aliases in actor_aliases.items():
        # Capture aliases via default argument to avoid closure-over-loop-variable.
        def _matches(org_list: list[str], _a: list[str] = aliases) -> bool:
            for org in org_list:
                for alias in _a:
                    if alias in org:
                        return True
            return False

        mask  = docs["orgs"].apply(_matches)
        sub   = docs[mask]
        count = len(sub)
        tones = sub["tone"].dropna()
        result[sig] = {
            "article_count": count,
            "avg_tone":      float(tones.mean()) if len(tones) else float("nan"),
            "total_docs":    total_docs,
        }

    return result


# ── Daily cache helpers ────────────────────────────────────────────────────────

def _daily_cache_path(date: pd.Timestamp) -> Path:
    return DAILY_CACHE_DIR / f"{date.strftime('%Y-%m-%d')}.parquet"


def _load_daily_cache(date: pd.Timestamp) -> dict[str, dict] | None:
    p = _daily_cache_path(date)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        return {
            row["signal"]: {
                "article_count": row["article_count"],
                "avg_tone":      row["avg_tone"],
                "total_docs":    row["total_docs"],
            }
            for _, row in df.iterrows()
        }
    except Exception:
        return None


def _save_daily_cache(date: pd.Timestamp, stats: dict[str, dict]) -> None:
    if not stats:
        return
    rows = [{"signal": sig, **vals} for sig, vals in stats.items()]
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_daily_cache_path(date), index=False)


# ── File index helpers ─────────────────────────────────────────────────────────

def _load_file_index() -> pd.DataFrame:
    if FILE_INDEX_PATH.exists():
        try:
            return pd.read_parquet(FILE_INDEX_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "slot_time", "selection_type", "url"])


def _save_file_index(index_df: pd.DataFrame) -> None:
    FILE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(FILE_INDEX_PATH, index=False)


# ── Fetch one day ──────────────────────────────────────────────────────────────

def fetch_day(
    date: pd.Timestamp,
    force: bool = False,
) -> tuple[dict[str, dict] | None, str]:
    """Fetch and process one day's GKG file. Returns (stats, selection_type).

    selection_type: "exact_noon" | "fallback_nearest" | "missing" | "cached"

    Uses on-disk daily cache; returns (cached_stats, "cached") if already
    processed (unless force=True).
    """
    if not force:
        cached = _load_daily_cache(date)
        if cached is not None:
            return cached, "cached"

    client = _get_client()
    data, slot, sel_type = _download_gkg_file_for_day(client, date)
    if data is None:
        return None, "missing"

    docs = _parse_gkg_bytes(data)
    if docs is None or docs.empty:
        log.warning("  %s — GKG file parsed to 0 rows (slot %s)", date.date(), slot)
        return None, "missing"

    stats = _compute_snapshot_stats(docs, SECTOR_CODES, ACTOR_ALIASES)
    _save_daily_cache(date, stats)
    return stats, sel_type


# ── Build processed DataFrames ────────────────────────────────────────────────

def build_daily_processed(
    daily_results: dict[pd.Timestamp, dict[str, dict]],
) -> pd.DataFrame:
    """Convert {date: {signal: stats}} → daily long DataFrame.

    Schema:
        theme_or_actor : str
        event_date     : datetime64[ns]
        article_count  : float64  — distinct matching docs in that daily snapshot
        avg_tone       : float64  — mean V2Tone[0] of matching docs
        intensity      : float64  — article_count / total_docs_day
        total_docs_day : float64  — total distinct docs in that daily snapshot
    """
    rows: list[dict] = []
    for date, sig_stats in daily_results.items():
        for sig in ALL_SIGNALS:
            s = sig_stats.get(sig)
            if s is None:
                continue
            total = s["total_docs"]
            ac    = s["article_count"]
            rows.append({
                "theme_or_actor": sig,
                "event_date":     date,
                "article_count":  float(ac),
                "avg_tone":       s["avg_tone"],
                "intensity":      float(ac) / total if total > 0 else float("nan"),
                "total_docs_day": float(total),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    for col in ("article_count", "avg_tone", "intensity", "total_docs_day"):
        df[col] = df[col].astype("float64")
    return (
        df[["theme_or_actor", "event_date", "article_count", "avg_tone",
            "intensity", "total_docs_day"]]
        .sort_values(["theme_or_actor", "event_date"])
        .reset_index(drop=True)
    )


def aggregate_daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily panel to weekly using mathematically correct formulas.

    week_start is the Monday of each ISO week (dayofweek == 0).

    Aggregation rules (NOT simple means):
      weekly_article_count = sum(daily_article_count)
      weekly_avg_tone      = weighted mean using daily_article_count as weights
                             (NaN days and zero-article days excluded from mean)
      weekly_intensity     = sum(daily_article_count) / sum(daily_total_docs)
    """
    df = daily_df.copy()
    # Subtract dayofweek days to get Monday of the ISO week for each date.
    df["week_start"] = df["event_date"] - pd.to_timedelta(
        df["event_date"].dt.dayofweek, unit="D"
    )

    rows: list[dict] = []
    for (signal, week_start), grp in df.groupby(["theme_or_actor", "week_start"]):
        weekly_ac    = grp["article_count"].sum()
        weekly_total = grp["total_docs_day"].sum()

        # Weighted mean tone: only include days with valid tone AND matched articles.
        tone_mask = grp["avg_tone"].notna() & (grp["article_count"] > 0)
        tone_weight_sum = grp.loc[tone_mask, "article_count"].sum()
        if tone_weight_sum > 0:
            weekly_tone = float(
                (grp.loc[tone_mask, "avg_tone"] * grp.loc[tone_mask, "article_count"]).sum()
                / tone_weight_sum
            )
        else:
            weekly_tone = float("nan")

        rows.append({
            "theme_or_actor": signal,
            "week_start":     week_start,
            "article_count":  float(weekly_ac),
            "avg_tone":       weekly_tone,
            "intensity":      float(weekly_ac) / weekly_total if weekly_total > 0 else float("nan"),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["week_start"] = pd.to_datetime(out["week_start"])
    for col in ("article_count", "avg_tone", "intensity"):
        out[col] = out[col].astype("float64")
    return (
        out[["theme_or_actor", "week_start", "article_count", "avg_tone", "intensity"]]
        .sort_values(["theme_or_actor", "week_start"])
        .reset_index(drop=True)
    )


# ── Archive old outputs ────────────────────────────────────────────────────────

def archive_old_outputs() -> None:
    """Move existing canonical output files to timestamped archive paths."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = [
        (
            WEEKLY_PROCESSED,
            ARCHIVE_PROC_DIR / f"gdelt_narrative_weekly_snapshot_{ts}.parquet",
        ),
        (
            PIT_DIR / "gdelt.parquet",
            ARCHIVE_PIT_DIR / f"gdelt_weekly_snapshot_{ts}.parquet",
        ),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            log.info("Archived  %s  →  %s", src.name, dst.name)


# ── PIT ingest ─────────────────────────────────────────────────────────────────

def ingest_to_pit(weekly_df: pd.DataFrame) -> None:
    """Ingest the weekly processed panel into the PIT store.

    PIT semantics:
      actor_id   = theme_or_actor
      signal_id  = gdelt_article_count | gdelt_avg_tone | gdelt_intensity
      event_date = week_start
      pub_date   = week_start + 7 days  (GDELT data complete at end of week)
      source     = "gdelt"
      vintage_id = None  (GDELT is append-only, no revisions)
    """
    pit_rows: list[pd.DataFrame] = []
    for metric in ("article_count", "avg_tone", "intensity"):
        chunk = weekly_df[["theme_or_actor", "week_start", metric]].dropna(
            subset=[metric]
        ).copy()
        chunk = chunk.rename(
            columns={"theme_or_actor": "actor_id", "week_start": "event_date",
                     metric: "value"}
        )
        chunk["signal_id"]  = f"gdelt_{metric}"
        chunk["pub_date"]   = chunk["event_date"] + pd.Timedelta(days=7)
        chunk["source"]     = "gdelt"
        chunk["vintage_id"] = None
        pit_rows.append(chunk)

    pit_df = pd.concat(pit_rows, ignore_index=True)
    store  = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store: %d rows → %s", len(pit_df), PIT_DIR)


# ── Validate ───────────────────────────────────────────────────────────────────

def run_validation() -> None:
    """Download yesterday's GKG file and show sample signal values."""
    log.info("Validation: downloading recent GKG file …")
    today     = pd.Timestamp.utcnow().normalize().tz_localize(None)
    test_date = today - pd.Timedelta(days=1)

    client = _get_client()
    data, slot, sel_type = _download_gkg_file_for_day(client, test_date)
    if data is None:
        log.warning("Could not download validation file for %s", test_date.date())
        return

    docs = _parse_gkg_bytes(data)
    if docs is None:
        log.warning("Could not parse validation file")
        return

    stats = _compute_snapshot_stats(docs, SECTOR_CODES, ACTOR_ALIASES)
    total = next(iter(stats.values()), {}).get("total_docs", 0) if stats else 0
    log.info(
        "Validation file: %d docs  (date=%s  slot=%s  selection=%s)",
        total, test_date.date(), slot, sel_type,
    )
    for sig, s in stats.items():
        log.info(
            "  %-28s  %3d docs  intensity=%.4f  tone=%.2f",
            sig, s["article_count"], s["article_count"] / max(total, 1),
            s["avg_tone"] if not pd.isna(s["avg_tone"]) else 0.0,
        )


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    n_exact_noon: int,
    n_fallback: int,
    n_missing: int,
) -> None:
    if weekly_df.empty:
        print("\nNo data.\n")
        return

    fetched      = set(weekly_df["theme_or_actor"].unique())
    missing_sigs = [s for s in ALL_SIGNALS if s not in fetched]
    n_days       = daily_df["event_date"].nunique() if not daily_df.empty else 0

    print("\n" + "=" * 70)
    print("GDELT GKG 2.0 raw-file narrative signals — summary")
    print("=" * 70)
    print(f"  Source          : {GKG_BASE_URL}")
    print(f"  Method          : 1 GKG file per UTC day (slot nearest 12:00 UTC)")
    print(f"                    daily -> weekly via weighted aggregation")
    print(f"  Days processed  : {n_days:,}")
    print(f"  Exact noon      : {n_exact_noon:,}  (12:00:00 UTC file found)")
    print(f"  Fallback nearest: {n_fallback:,}  (nearest slot within probe window)")
    print(f"  Missing days    : {n_missing:,}  (no GKG file found)")
    print(f"  Signals         : {len(fetched)} / {len(ALL_SIGNALS)}")
    print(f"  Daily rows      : {len(daily_df):,}")
    print(f"  Weekly rows     : {len(weekly_df):,}")
    if not weekly_df.empty:
        print(f"  Week range      : {weekly_df['week_start'].min().date()} "
              f"→ {weekly_df['week_start'].max().date()}")
    print(f"  article_count   : sum of daily matched docs across the week")
    print(f"  avg_tone        : article-count-weighted mean daily tone per week")
    print(f"  intensity       : sum(daily_matched) / sum(daily_total_docs)")
    print(f"  Daily artifact  : {DAILY_PROCESSED}")
    print(f"  Weekly artifact : {WEEKLY_PROCESSED}  [DAILY-DERIVED — canonical]")
    print(f"  PIT store       : {PIT_DIR}")
    if missing_sigs:
        print(f"  WARN missing    : {missing_sigs}")
    print("=" * 70 + "\n")


# ── Write outputs helper ───────────────────────────────────────────────────────

def _write_outputs(
    daily_results: dict[pd.Timestamp, dict[str, dict]],
    args: argparse.Namespace,
    n_exact: int = 0,
    n_fallback: int = 0,
    n_missing: int = 0,
) -> None:
    """Build processed artifacts from daily_results and write to disk."""
    log.info("Building daily processed artifact (%d days) …", len(daily_results))
    daily_df = build_daily_processed(daily_results)

    if daily_df.empty:
        log.error("No daily data — aborting.")
        sys.exit(1)

    # Log per-signal daily coverage
    for sig in sorted(daily_df["theme_or_actor"].unique()):
        sub = daily_df[daily_df["theme_or_actor"] == sig]
        tone_vals = sub["avg_tone"].dropna()
        log.info(
            "  %-28s  %4d days  intensity [%.4f–%.4f]  tone [%.1f–%.1f]",
            sig, len(sub),
            sub["intensity"].min(), sub["intensity"].max(),
            float(tone_vals.min()) if len(tone_vals) else 0.0,
            float(tone_vals.max()) if len(tone_vals) else 0.0,
        )

    # Write daily artifact (skip only in --weekly-only mode)
    if not getattr(args, "weekly_only", False):
        DAILY_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_parquet(DAILY_PROCESSED, index=False)
        log.info("Saved daily: %d rows → %s", len(daily_df), DAILY_PROCESSED)

    if getattr(args, "daily_only", False):
        log.info("--daily-only: skipping weekly aggregation and PIT.")
        return

    # Aggregate daily → weekly
    log.info("Aggregating daily → weekly …")
    weekly_df = aggregate_daily_to_weekly(daily_df)

    if weekly_df.empty:
        log.error("Weekly aggregation produced no data — aborting.")
        sys.exit(1)

    # Archive old canonical outputs before overwriting
    if not getattr(args, "no_archive", False):
        archive_old_outputs()

    # Write canonical weekly artifact
    WEEKLY_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_parquet(WEEKLY_PROCESSED, index=False)
    log.info("Saved weekly: %d rows → %s", len(weekly_df), WEEKLY_PROCESSED)

    if getattr(args, "skip_pit", False):
        log.info("--skip-pit: skipping PIT ingest.")
    else:
        log.info("Ingesting into PIT store …")
        ingest_to_pit(weekly_df)

    print_summary(daily_df, weekly_df, n_exact, n_fallback, n_missing)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GDELT GKG 2.0 narrative signals for SMIM "
            "(one representative file per UTC day, aggregated to weekly)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START.strftime("%Y-%m-%d"),
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START.date()}.",
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END.strftime("%Y-%m-%d"),
        help=f"End date (YYYY-MM-DD). Default: {DEFAULT_END.date()}.",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Concurrent download workers. Default: 5.",
    )
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="Ignore daily cache and re-download all GKG files.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help=(
            "Recompute processed outputs from existing daily cache. "
            "No new downloads. Respects --daily-only / --weekly-only."
        ),
    )
    parser.add_argument(
        "--skip-pit", action="store_true",
        help="Skip PIT store ingestion.",
    )
    parser.add_argument(
        "--sectors-only", action="store_true",
        help="Only compute sector signals (skip actor signals).",
    )
    parser.add_argument(
        "--actors-only", action="store_true",
        help="Only compute actor signals (skip sector signals).",
    )
    parser.add_argument(
        "--daily-only", action="store_true",
        help="Build/refresh only the daily artifact; skip weekly aggregation and PIT.",
    )
    parser.add_argument(
        "--weekly-only", action="store_true",
        help=(
            "Rebuild weekly from existing daily cache. "
            "No new downloads. Skips writing the daily artifact."
        ),
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Skip archiving old canonical outputs (overwrite in place).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Download yesterday's GKG file and show signal stats. No full fetch.",
    )
    args = parser.parse_args()

    if args.validate_only:
        run_validation()
        return

    start = max(pd.Timestamp(args.start_date), GKG_V2_START)
    end   = pd.Timestamp(args.end_date)

    # Apply --sectors-only / --actors-only filters by restricting the module-level
    # signal sets. This is done by narrowing SECTOR_CODES and ACTOR_ALIASES
    # temporarily in the per-signal loop inside build_daily_processed (ALL_SIGNALS
    # is module-level; we override it locally here).
    global SECTOR_CODES, ACTOR_ALIASES, ALL_SIGNALS
    if args.sectors_only:
        ACTOR_ALIASES = {}
        ALL_SIGNALS = list(SECTOR_CODES.keys())
    elif args.actors_only:
        SECTOR_CODES = {}
        ALL_SIGNALS = list(ACTOR_ALIASES.keys())

    # ── Weekly-only mode: rebuild from cached daily data, no new downloads ────
    if args.weekly_only:
        log.info("--weekly-only: loading daily cache and rebuilding weekly artifact …")
        dates = pd.date_range(start=start, end=end, freq="D")
        daily_results: dict[pd.Timestamp, dict[str, dict]] = {}
        for d in dates:
            cached = _load_daily_cache(d)
            if cached is not None:
                daily_results[d] = cached
        log.info(
            "Loaded %d / %d days from daily cache.", len(daily_results), len(dates)
        )
        if not daily_results:
            log.error("No daily cache found in range. Run without --weekly-only first.")
            sys.exit(1)
        _write_outputs(daily_results, args)
        return

    # ── Normal fetch loop (or --rebuild) ──────────────────────────────────────
    dates = pd.date_range(start=start, end=end, freq="D")
    log.info(
        "Processing %d UTC days (%s → %s) with %d workers …",
        len(dates), dates[0].date(), dates[-1].date(), args.workers,
    )

    if args.rebuild:
        log.info("--rebuild: loading all stats from daily cache (no downloads).")
    elif not args.force_refetch:
        n_cached = sum(1 for d in dates if _daily_cache_path(d).exists())
        log.info(
            "Cache hits: %d / %d  (downloading %d files)",
            n_cached, len(dates), len(dates) - n_cached,
        )

    daily_results: dict[pd.Timestamp, dict[str, dict]] = {}
    sel_counts: dict[str, int] = {
        "exact_noon": 0, "fallback_nearest": 0, "missing": 0, "cached": 0
    }
    # Per-date selection records for the file index (only non-cached results).
    new_index_rows: list[dict] = []
    completed = 0

    def _do_day(d: pd.Timestamp) -> tuple[pd.Timestamp, dict | None, str]:
        if args.rebuild:
            cached = _load_daily_cache(d)
            return (d, cached, "cached") if cached is not None else (d, None, "missing")
        stats, sel_type = fetch_day(d, force=args.force_refetch)
        return d, stats, sel_type

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_day, d): d for d in dates}
        for fut in as_completed(futures):
            date, stats, sel_type = fut.result()
            completed += 1
            sel_counts[sel_type] = sel_counts.get(sel_type, 0) + 1
            if stats is not None:
                daily_results[date] = stats
            if sel_type != "cached":
                new_index_rows.append({
                    "date":           date.strftime("%Y-%m-%d"),
                    "slot_time":      "",
                    "selection_type": sel_type,
                    "url":            "",
                })
            if completed % 100 == 0 or completed == len(dates):
                log.info(
                    "  Progress: %d / %d  (missing: %d)",
                    completed, len(dates), sel_counts.get("missing", 0),
                )

    n_exact    = sel_counts.get("exact_noon", 0)
    n_fallback = sel_counts.get("fallback_nearest", 0)
    n_missing  = sel_counts.get("missing", 0)
    log.info(
        "Selection summary: exact_noon=%d  fallback=%d  missing=%d  cached=%d",
        n_exact, n_fallback, n_missing, sel_counts.get("cached", 0),
    )

    # Persist file index: merge new rows with existing, dedup on date.
    if new_index_rows:
        existing_index = _load_file_index()
        merged = pd.concat(
            [existing_index, pd.DataFrame(new_index_rows)], ignore_index=True
        ).drop_duplicates(subset=["date"], keep="last")
        _save_file_index(merged)

    _write_outputs(daily_results, args, n_exact, n_fallback, n_missing)


if __name__ == "__main__":
    main()
