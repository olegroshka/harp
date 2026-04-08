"""Comprehensive data coverage audit for SMIM.

Loads every data source and universe, computes coverage metrics, and
generates docs/smim/reports/data_audit.md plus data/smim/universes/manifest.json.

Usage:
    uv run python scripts/smim/smim_data_audit.py

Outputs:
    docs/smim/reports/data_audit.md     — coverage report with Gate G1 checklist
    data/smim/universes/manifest.json   — clean universe manifest with regimes
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

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

# ── Paths ──────────────────────────────────────────────────────────────────────

UNIVERSE_DIR   = _ROOT / "data" / "universes"
OHLCV_BASE     = _ROOT / "equities" / "smim"
PIT_DIR        = _ROOT / "data" / "pit_store"
REPORT_DIR     = _ROOT / "docs" / "smim" / "reports"
REPORT_PATH    = REPORT_DIR / "data_audit.md"
MANIFEST_PATH  = UNIVERSE_DIR / "manifest.json"

# Gate G1 thresholds
G1_MISSINGNESS_THRESHOLD = 0.20   # max missing fraction per signal
G1_TICKER_COVERAGE_MIN   = 0.80   # min fraction of tickers in a universe that must have data

# OHLCV regime thresholds (years)
REGIME_GOLD   = 15
REGIME_SILVER = 10
REGIME_BRONZE  = 5

# Planned FRED series (29 total; NAPM and CUSR0000SAM may have failed)
FRED_PLANNED: list[str] = [
    "GDP", "GDPC1", "INDPRO", "MANEMP", "NAPM", "UNRATE",
    "CPIAUCSL", "CPILFESL", "PCEPI",
    "FEDFUNDS", "DFF", "GS10", "GS2", "T10Y2Y",
    "BAA10Y", "BAMLH0A0HYM2", "VIXCLS", "STLFSI2",
    "UMCSENT", "USSLIND", "HOUST", "M2SL", "TOTBKCR",
    "DTWEXBGS", "DCOILWTICO", "DCOILBRENTEU", "GASREGW",
    "CUSR0000SAM", "DRCCLACBS",
]
ALFRED_SERIES: list[str] = ["GDP", "UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS"]
QUARTERLY_SERIES: list[str] = ["GDP", "GDPC1", "FEDFUNDS"]

# GDELT themes tracked
GDELT_THEMES: list[str] = [
    "TAX_FNCACT_ENERGY_SECTOR", "TAX_FNCACT_TECH_SECTOR",
    "TAX_FNCACT_FINANCIAL_SECTOR", "TAX_FNCACT_HEALTHCARE",
    "TAX_FNCACT_INDUSTRIAL", "ECONOMY_HISTORIC", "ECON_BANKRUPTCY",
    "ECON_MONOPOLY", "REGULATION",
]

# IMF/OECD expected indicators
IMF_INDICATORS: list[str] = [
    "NGDP_RPCH", "PCPIPCH", "BCA", "GGXCNL_NGDP", "GGXWDG_NGDP", "LUR", "PPPGDP",
]
IMF_COUNTRIES: list[str] = ["US", "GB", "DE", "JP"]
OECD_INDICATORS: list[str] = ["LI", "BCICP", "CCICP", "B1GQ_POP"]
OECD_COUNTRIES: list[str] = ["US", "GB"]

# BEA sectors
BEA_SECTORS: list[str] = ["Energy", "Technology", "Financials", "Healthcare", "Industrials"]


# ── Universe loader ────────────────────────────────────────────────────────────

def load_universes() -> dict[str, pd.DataFrame]:
    """Load all universe CSVs. Returns {universe_name: df}."""
    result: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(UNIVERSE_DIR.glob("*.csv")):
        name = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            result[name] = df
            log.info("  Universe %s: %d tickers", name, len(df))
        except Exception as exc:
            log.warning("  Failed to load %s: %s", csv_path.name, exc)
    return result


# ── PIT store loader ───────────────────────────────────────────────────────────

def load_pit_source(source: str) -> pd.DataFrame:
    """Load one PIT source shard. Returns empty DataFrame if missing."""
    path = PIT_DIR / f"{source}.parquet"
    if not path.exists():
        log.warning("  PIT shard missing: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info("  PIT %s: %d rows, %d actors, %d signals",
             source, len(df),
             df["actor_id"].nunique() if "actor_id" in df.columns else 0,
             df["signal_id"].nunique() if "signal_id" in df.columns else 0)
    return df


# ── SECTION 1: Equity OHLCV audit ─────────────────────────────────────────────

def audit_ohlcv(universes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Per-universe OHLCV coverage: regime classification and gap detection."""
    results: dict[str, Any] = {}

    for uni_name, uni_df in universes.items():
        tickers = uni_df["ticker"].tolist()
        ohlcv_path = OHLCV_BASE / uni_name / "ohlcv.parquet"

        if not ohlcv_path.exists():
            results[uni_name] = {
                "status": "MISSING",
                "tickers_total": len(tickers),
                "tickers_with_data": 0,
                "regime": "Sparse",
                "date_range": None,
                "regime_counts": {"Gold": 0, "Silver": 0, "Bronze": 0, "Sparse": len(tickers)},
                "sparse_tickers": tickers,
            }
            log.warning("  OHLCV missing for universe %s", uni_name)
            continue

        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])

        ticker_col = "ticker" if "ticker" in ohlcv.columns else ohlcv.columns[1]
        grouped = ohlcv.groupby(ticker_col)["date"]

        ticker_regimes: dict[str, str] = {}
        sparse_tickers: list[str] = []

        for ticker in tickers:
            if ticker not in grouped.groups:
                ticker_regimes[ticker] = "Sparse"
                sparse_tickers.append(ticker)
                continue
            dates = grouped.get_group(ticker)
            span_years = (dates.max() - dates.min()).days / 365.25
            if span_years >= REGIME_GOLD:
                ticker_regimes[ticker] = "Gold"
            elif span_years >= REGIME_SILVER:
                ticker_regimes[ticker] = "Silver"
            elif span_years >= REGIME_BRONZE:
                ticker_regimes[ticker] = "Bronze"
            else:
                ticker_regimes[ticker] = "Sparse"
                sparse_tickers.append(ticker)

        regime_counts = {r: sum(1 for v in ticker_regimes.values() if v == r)
                         for r in ["Gold", "Silver", "Bronze", "Sparse"]}

        tickers_with_data = len(tickers) - len(sparse_tickers)
        overall_min = ohlcv["date"].min()
        overall_max = ohlcv["date"].max()
        overall_span = (overall_max - overall_min).days / 365.25

        if overall_span >= REGIME_GOLD:
            overall_regime = "Gold"
        elif overall_span >= REGIME_SILVER:
            overall_regime = "Silver"
        elif overall_span >= REGIME_BRONZE:
            overall_regime = "Bronze"
        else:
            overall_regime = "Sparse"

        results[uni_name] = {
            "status": "OK",
            "tickers_total": len(tickers),
            "tickers_with_data": tickers_with_data,
            "coverage_pct": tickers_with_data / len(tickers) * 100 if tickers else 0,
            "regime": overall_regime,
            "date_range": (overall_min.date().isoformat(), overall_max.date().isoformat()),
            "regime_counts": regime_counts,
            "sparse_tickers": sparse_tickers[:20],  # cap list for readability
            "sparse_count": len(sparse_tickers),
        }

    return results


# ── SECTION 2: FRED macro audit ────────────────────────────────────────────────

def audit_fred(pit_df: pd.DataFrame) -> dict[str, Any]:
    """Check FRED series coverage vs planned 29."""
    if pit_df.empty:
        return {"status": "MISSING", "fetched": [], "missing": FRED_PLANNED, "gaps": {}}

    fetched = sorted(pit_df["signal_id"].unique().tolist())
    missing = [s for s in FRED_PLANNED if s not in fetched]
    unexpected = [s for s in fetched if s not in FRED_PLANNED]

    # Gap detection for key quarterly series
    gaps: dict[str, list[str]] = {}
    for series_id in QUARTERLY_SERIES:
        sub = pit_df[pit_df["signal_id"] == series_id].copy()
        if sub.empty:
            gaps[series_id] = ["NO DATA"]
            continue
        sub["event_date"] = pd.to_datetime(sub["event_date"])
        sub = sub.sort_values("event_date")
        # Check for consecutive quarter gaps (>100 days between observations)
        diffs = sub["event_date"].diff().dt.days.dropna()
        big_gaps = diffs[diffs > 100]
        if len(big_gaps) > 0:
            gap_dates = [sub["event_date"].iloc[i].strftime("%Y-%m-%d")
                         for i in big_gaps.index if i < len(sub)]
            gaps[series_id] = gap_dates[:5]
        else:
            gaps[series_id] = []

    # ALFRED vintage check
    alfred_ok = [s for s in ALFRED_SERIES if s in fetched]

    earliest = pit_df["event_date"].min() if "event_date" in pit_df.columns else None
    latest = pit_df["event_date"].max() if "event_date" in pit_df.columns else None

    return {
        "status": "OK",
        "planned": len(FRED_PLANNED),
        "fetched_count": len(fetched),
        "fetched": fetched,
        "missing": missing,
        "unexpected": unexpected,
        "alfred_coverage": alfred_ok,
        "alfred_missing": [s for s in ALFRED_SERIES if s not in fetched],
        "quarterly_gaps": gaps,
        "date_range": (
            str(pd.to_datetime(earliest).date()) if earliest is not None else None,
            str(pd.to_datetime(latest).date()) if latest is not None else None,
        ),
        "total_obs": len(pit_df),
    }


# ── SECTION 3: EDGAR audit ────────────────────────────────────────────────────

def audit_edgar(pit_df: pd.DataFrame, universes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Per-universe EDGAR filing coverage."""
    if pit_df.empty:
        return {"status": "MISSING"}

    pit_df = pit_df.copy()
    pit_df["event_date"] = pd.to_datetime(pit_df["event_date"])

    xbrl_tags = sorted(pit_df["signal_id"].unique().tolist())

    # Tickers with any filing since 2010
    cutoff = pd.Timestamp("2010-01-01")
    recent = pit_df[pit_df["event_date"] >= cutoff]
    tickers_with_data = set(recent["actor_id"].unique())

    universe_coverage: dict[str, dict] = {}
    for uni_name, uni_df in universes.items():
        tickers = set(uni_df["ticker"].tolist())
        covered = tickers & tickers_with_data
        no_data = sorted(tickers - tickers_with_data)
        pct = len(covered) / len(tickers) * 100 if tickers else 0
        universe_coverage[uni_name] = {
            "tickers_total": len(tickers),
            "tickers_covered": len(covered),
            "coverage_pct": pct,
            "no_data": no_data[:20],
            "no_data_count": len(no_data),
        }

    # Filing count by year
    pit_df["year"] = pit_df["event_date"].dt.year
    filings_by_year = pit_df.groupby("year")["actor_id"].nunique().to_dict()

    return {
        "status": "OK",
        "total_obs": len(pit_df),
        "total_actors": pit_df["actor_id"].nunique(),
        "xbrl_tags": xbrl_tags,
        "xbrl_tag_count": len(xbrl_tags),
        "universe_coverage": universe_coverage,
        "filings_by_year": {str(k): v for k, v in sorted(filings_by_year.items())},
        "date_range": (
            str(pit_df["event_date"].min().date()),
            str(pit_df["event_date"].max().date()),
        ),
    }


# ── SECTION 4: GDELT audit ────────────────────────────────────────────────────

def audit_gdelt(pit_df: pd.DataFrame) -> dict[str, Any]:
    """GDELT weekly coverage: continuity and zero-article gap detection."""
    if pit_df.empty:
        return {"status": "MISSING"}

    pit_df = pit_df.copy()
    pit_df["event_date"] = pd.to_datetime(pit_df["event_date"])

    actors = sorted(pit_df["actor_id"].unique().tolist())
    signals = sorted(pit_df["signal_id"].unique().tolist())

    # Coverage since 2015
    cutoff = pd.Timestamp("2015-01-01")
    recent = pit_df[pit_df["event_date"] >= cutoff]

    # Per-actor weekly continuity: find stretches with >4 consecutive missing weeks
    theme_gaps: dict[str, list[str]] = {}
    for actor in actors:
        sub = recent[recent["actor_id"] == actor].sort_values("event_date")
        if sub.empty:
            theme_gaps[actor] = ["NO DATA since 2015"]
            continue
        # Build expected weekly grid
        date_min = sub["event_date"].min()
        date_max = sub["event_date"].max()
        expected_weeks = pd.date_range(date_min, date_max, freq="W-MON")
        # Use ISO year-week strings consistently
        actual_weeks = set(
            sub["event_date"].apply(lambda d: d.strftime("%G-W%V"))
        )
        gap_starts: list[str] = []
        consecutive = 0
        gap_start_date: str | None = None
        for wk in expected_weeks:
            if wk.strftime("%G-W%V") not in actual_weeks:
                if consecutive == 0:
                    gap_start_date = wk.strftime("%Y-%m-%d")
                consecutive += 1
                if consecutive == 4:
                    gap_starts.append(f"{gap_start_date} (+4wk)")
            else:
                consecutive = 0
        theme_gaps[actor] = gap_starts[:5]

    earliest = pit_df["event_date"].min()
    latest = pit_df["event_date"].max()

    # Count weeks since 2015
    weeks_since_2015 = len(recent["event_date"].dt.to_period("W").unique())

    return {
        "status": "OK",
        "total_obs": len(pit_df),
        "actors": actors,
        "signals": signals,
        "weeks_since_2015": weeks_since_2015,
        "actor_gaps": theme_gaps,
        "date_range": (str(earliest.date()), str(latest.date())),
    }


# ── SECTION 5: IMF/OECD audit ─────────────────────────────────────────────────

def audit_imf(pit_df: pd.DataFrame) -> dict[str, Any]:
    """IMF indicator × country coverage matrix."""
    if pit_df.empty:
        return {"status": "MISSING"}

    pit_df = pit_df.copy()
    pit_df["event_date"] = pd.to_datetime(pit_df["event_date"])

    matrix: dict[str, dict[str, Any]] = {}
    all_actors = sorted(pit_df["actor_id"].unique())
    all_signals = sorted(pit_df["signal_id"].unique())

    for indicator in IMF_INDICATORS:
        matrix[indicator] = {}
        for country in IMF_COUNTRIES:
            sub = pit_df[(pit_df["signal_id"] == indicator) & (pit_df["actor_id"] == country)]
            if sub.empty:
                matrix[indicator][country] = "MISSING"
            else:
                max_year = sub["event_date"].dt.year.max()
                matrix[indicator][country] = f"{sub['event_date'].dt.year.min()}–{max_year}"
                if max_year < 2024:
                    matrix[indicator][country] += " ⚠️ stops before 2024"

    missing_indicators = [i for i in IMF_INDICATORS if i not in all_signals]
    missing_countries  = [c for c in IMF_COUNTRIES if c not in all_actors]

    return {
        "status": "OK",
        "total_obs": len(pit_df),
        "signals": all_signals,
        "actors": all_actors,
        "matrix": matrix,
        "missing_indicators": missing_indicators,
        "missing_countries": missing_countries,
        "date_range": (
            str(pit_df["event_date"].min().date()),
            str(pit_df["event_date"].max().date()),
        ),
    }


def audit_oecd(pit_df: pd.DataFrame) -> dict[str, Any]:
    """OECD indicator × country coverage matrix."""
    if pit_df.empty:
        return {"status": "MISSING"}

    pit_df = pit_df.copy()
    pit_df["event_date"] = pd.to_datetime(pit_df["event_date"])

    matrix: dict[str, dict[str, Any]] = {}
    all_actors = sorted(pit_df["actor_id"].unique())
    all_signals = sorted(pit_df["signal_id"].unique())

    for indicator in OECD_INDICATORS:
        matrix[indicator] = {}
        for country in OECD_COUNTRIES:
            sub = pit_df[(pit_df["signal_id"] == indicator) & (pit_df["actor_id"] == country)]
            if sub.empty:
                matrix[indicator][country] = "MISSING"
            else:
                max_year = sub["event_date"].dt.year.max()
                matrix[indicator][country] = f"{sub['event_date'].dt.year.min()}–{max_year}"
                if max_year < 2024:
                    matrix[indicator][country] += " ⚠️ stops before 2024"

    missing_indicators = [i for i in OECD_INDICATORS if i not in all_signals]
    missing_countries  = [c for c in OECD_COUNTRIES if c not in all_actors]

    return {
        "status": "OK",
        "total_obs": len(pit_df),
        "signals": all_signals,
        "actors": all_actors,
        "matrix": matrix,
        "missing_indicators": missing_indicators,
        "missing_countries": missing_countries,
        "date_range": (
            str(pit_df["event_date"].min().date()),
            str(pit_df["event_date"].max().date()),
        ),
    }


# ── SECTION 6: BEA audit ──────────────────────────────────────────────────────

def audit_bea(pit_df: pd.DataFrame) -> dict[str, Any]:
    """BEA I/O supply-chain coverage: years and sector pairs."""
    if pit_df.empty:
        return {"status": "MISSING"}

    pit_df = pit_df.copy()
    pit_df["event_date"] = pd.to_datetime(pit_df["event_date"])

    years = sorted(pit_df["event_date"].dt.year.unique().tolist())
    actors = sorted(pit_df["actor_id"].unique().tolist())  # "SectorA→SectorB"

    # Extract unique sectors mentioned
    sectors_seen: set[str] = set()
    for actor in actors:
        if "→" in actor:
            src, tgt = actor.split("→", 1)
            sectors_seen.add(src.strip())
            sectors_seen.add(tgt.strip())

    missing_sectors = [s for s in BEA_SECTORS if s not in sectors_seen]

    # Count pairs per year
    pairs_by_year = (
        pit_df.groupby(pit_df["event_date"].dt.year)["actor_id"]
        .nunique()
        .to_dict()
    )

    return {
        "status": "OK",
        "total_obs": len(pit_df),
        "years": years,
        "sector_pairs": len(actors),
        "sectors_mapped": sorted(sectors_seen),
        "sectors_missing": missing_sectors,
        "pairs_by_year": {str(k): v for k, v in sorted(pairs_by_year.items())},
        "date_range": (
            str(pit_df["event_date"].min().date()),
            str(pit_df["event_date"].max().date()),
        ),
    }


# ── SECTION 7: A1 leak detection ──────────────────────────────────────────────

def run_leak_detection() -> dict[str, Any]:
    """Run A1 leak check across all PIT shards."""
    store = PointInTimeStore(root_dir=PIT_DIR)
    sources = ["fred", "edgar", "gdelt", "imf", "oecd", "bea"]

    total_checked = 0
    violations: list[dict] = []

    for source in sources:
        path = PIT_DIR / f"{source}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "event_date" not in df.columns or "pub_date" not in df.columns:
            continue
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["pub_date"] = pd.to_datetime(df["pub_date"])
        leaks = df[df["pub_date"] < df["event_date"]]
        total_checked += len(df)
        for _, row in leaks.iterrows():
            violations.append({
                "source": source,
                "actor_id": row.get("actor_id", "?"),
                "signal_id": row.get("signal_id", "?"),
                "event_date": str(row["event_date"].date()),
                "pub_date": str(row["pub_date"].date()),
            })

    return {
        "rows_checked": total_checked,
        "violations": violations,
        "passed": len(violations) == 0,
    }


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_manifest(
    universes: dict[str, pd.DataFrame],
    ohlcv_audit: dict[str, Any],
    edgar_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build the clean universe manifest."""
    manifest: dict[str, Any] = {}

    for uni_name, uni_df in universes.items():
        tickers = uni_df["ticker"].tolist()
        ohlcv = ohlcv_audit.get(uni_name, {})
        edgar = edgar_audit.get("universe_coverage", {}).get(uni_name, {})

        sparse = ohlcv.get("sparse_tickers", [])
        sparse_count = ohlcv.get("sparse_count", 0)

        # Dropped = tickers with <5yr OHLCV
        dropped = sparse[:sparse_count] if sparse_count else sparse

        regime = ohlcv.get("regime", "Sparse")
        date_range = ohlcv.get("date_range", None)

        manifest[uni_name] = {
            "tickers_total": len(tickers),
            "tickers": tickers,
            "dropped": dropped,
            "dropped_reason": "OHLCV span < 5 years" if dropped else "",
            "data_regime": regime,
            "date_range": date_range,
            "edgar_coverage_pct": edgar.get("coverage_pct", 0),
            "ohlcv_coverage_pct": ohlcv.get("coverage_pct", 100 if ohlcv.get("status") == "OK" else 0),
        }

    return manifest


# ── Report renderer ───────────────────────────────────────────────────────────

def _regime_badge(regime: str) -> str:
    return {"Gold": "🥇 Gold", "Silver": "🥈 Silver", "Bronze": "🥉 Bronze", "Sparse": "⚠️ Sparse"}.get(regime, regime)


def render_report(
    ohlcv: dict,
    fred: dict,
    edgar: dict,
    gdelt: dict,
    imf: dict,
    oecd: dict,
    bea: dict,
    leak: dict,
    universes: dict,
    manifest: dict,
) -> str:
    lines: list[str] = []
    lines += [
        "# SMIM Data Coverage Audit",
        "",
        f"> Generated: 2026-03-23  |  PIT store: `{PIT_DIR.relative_to(_ROOT)}`",
        "",
    ]

    # ── Summary table ──────────────────────────────────────────────────────────
    lines += [
        "## Summary: Universe × Source Coverage",
        "",
        "| Universe | Tickers | OHLCV | Regime | EDGAR % | Notes |",
        "|----------|---------|-------|--------|---------|-------|",
    ]
    for uni_name in sorted(universes.keys()):
        o = ohlcv.get(uni_name, {})
        e = edgar.get("universe_coverage", {}).get(uni_name, {})
        tickers = o.get("tickers_total", len(universes[uni_name]))
        regime = _regime_badge(o.get("regime", "?"))
        ohlcv_pct = f"{o.get('coverage_pct', 0):.0f}%" if o.get("status") == "OK" else "MISSING"
        edgar_pct = f"{e.get('coverage_pct', 0):.0f}%" if e else "?"
        dr = o.get("date_range")
        notes = f"{dr[0]}–{dr[1]}" if dr else "no OHLCV"
        lines.append(f"| {uni_name} | {tickers} | {ohlcv_pct} | {regime} | {edgar_pct} | {notes} |")
    lines.append("")

    # ── OHLCV section ──────────────────────────────────────────────────────────
    lines += ["## 1. Equity OHLCV", ""]
    for uni_name in sorted(ohlcv.keys()):
        o = ohlcv[uni_name]
        if o.get("status") == "MISSING":
            lines.append(f"- **{uni_name}**: ❌ OHLCV file missing")
            continue
        rc = o.get("regime_counts", {})
        lines.append(
            f"- **{uni_name}**: {o['tickers_with_data']}/{o['tickers_total']} tickers  "
            f"({o.get('coverage_pct',0):.0f}%)  |  "
            f"Gold={rc.get('Gold',0)}, Silver={rc.get('Silver',0)}, "
            f"Bronze={rc.get('Bronze',0)}, Sparse={rc.get('Sparse',0)}"
        )
        if o.get("sparse_count", 0):
            lines.append(
                f"  - Sparse/missing ({o['sparse_count']}): "
                + ", ".join(o.get("sparse_tickers", []))
                + ("…" if o["sparse_count"] > 20 else "")
            )
    lines.append("")

    # ── FRED section ──────────────────────────────────────────────────────────
    lines += ["## 2. FRED Macro Signals", ""]
    if fred.get("status") == "MISSING":
        lines.append("❌ FRED PIT shard missing.\n")
    else:
        lines += [
            f"- Planned: {fred['planned']}  |  Fetched: {fred['fetched_count']}  |  Missing: {len(fred['missing'])}",
            f"- Observations: {fred['total_obs']:,}",
            f"- Date range: {fred['date_range'][0]} – {fred['date_range'][1]}",
            f"- ALFRED vintages: {fred['alfred_coverage']} ✅",
        ]
        if fred["missing"]:
            lines.append(f"- **Missing series**: {fred['missing']}")
        for series_id, gaps in fred.get("quarterly_gaps", {}).items():
            if gaps:
                lines.append(f"- {series_id} gaps: {gaps}")
        lines.append("")

    # ── EDGAR section ──────────────────────────────────────────────────────────
    lines += ["## 3. EDGAR Filings", ""]
    if edgar.get("status") == "MISSING":
        lines.append("❌ EDGAR PIT shard missing.\n")
    else:
        lines += [
            f"- Total observations: {edgar['total_obs']:,}",
            f"- Total actors (tickers): {edgar['total_actors']:,}",
            f"- XBRL tags ({edgar['xbrl_tag_count']}): {edgar['xbrl_tags']}",
            f"- Date range: {edgar['date_range'][0]} – {edgar['date_range'][1]}",
            "",
            "| Universe | Tickers | With EDGAR | Coverage |",
            "|----------|---------|-----------|---------|",
        ]
        for uni_name, ucov in sorted(edgar.get("universe_coverage", {}).items()):
            nd = ucov.get("no_data_count", 0)
            lines.append(
                f"| {uni_name} | {ucov['tickers_total']} | {ucov['tickers_covered']} | "
                f"{ucov['coverage_pct']:.0f}% |"
            )
        lines.append("")
        # Tickers with no data across all universes
        no_data_all: set[str] = set()
        for ucov in edgar.get("universe_coverage", {}).values():
            no_data_all.update(ucov.get("no_data", []))
        if no_data_all:
            lines.append(f"- Tickers with NO EDGAR data: {sorted(no_data_all)[:30]}")
        lines.append("")

    # ── GDELT section ──────────────────────────────────────────────────────────
    lines += ["## 4. GDELT Narrative Signals", ""]
    if gdelt.get("status") == "MISSING":
        lines.append("❌ GDELT PIT shard missing.\n")
    else:
        lines += [
            f"- Total observations: {gdelt['total_obs']:,}",
            f"- Actors: {gdelt['actors']}",
            f"- Signals: {gdelt['signals']}",
            f"- Weeks since 2015: {gdelt['weeks_since_2015']}",
            f"- Date range: {gdelt['date_range'][0]} – {gdelt['date_range'][1]}",
        ]
        gaps = gdelt.get("actor_gaps", {})
        has_gaps = {a: g for a, g in gaps.items() if g}
        if has_gaps:
            lines.append(f"- **Actors with >4 consecutive week gaps**: {list(has_gaps.keys())}")
            for actor, gs in has_gaps.items():
                lines.append(f"  - {actor}: {gs}")
        else:
            lines.append("- No actors with >4 consecutive week gaps ✅")
        lines.append("")

    # ── IMF section ───────────────────────────────────────────────────────────
    lines += ["## 5. IMF Macro Signals (WEO)", ""]
    if imf.get("status") == "MISSING":
        lines.append("❌ IMF PIT shard missing.\n")
    else:
        lines += [
            f"- Total observations: {imf['total_obs']:,}",
            f"- Date range: {imf['date_range'][0]} – {imf['date_range'][1]}",
            "",
            "| Indicator | US | GB | DE | JP |",
            "|-----------|----|----|----|----|",
        ]
        for ind in IMF_INDICATORS:
            row = imf.get("matrix", {}).get(ind, {})
            cells = [row.get(c, "—") for c in IMF_COUNTRIES]
            lines.append(f"| {ind} | " + " | ".join(cells) + " |")
        lines.append("")
        if imf.get("missing_indicators"):
            lines.append(f"- **Missing indicators**: {imf['missing_indicators']}")
        if imf.get("missing_countries"):
            lines.append(f"- **Missing countries**: {imf['missing_countries']}")
        lines.append("")

    # ── OECD section ──────────────────────────────────────────────────────────
    lines += ["## 6. OECD Macro Signals (CLI + QNA)", ""]
    if oecd.get("status") == "MISSING":
        lines.append("❌ OECD PIT shard missing.\n")
    else:
        lines += [
            f"- Total observations: {oecd['total_obs']:,}",
            f"- Date range: {oecd['date_range'][0]} – {oecd['date_range'][1]}",
            "",
            "| Indicator | US | GB |",
            "|-----------|----|----|",
        ]
        for ind in OECD_INDICATORS:
            row = oecd.get("matrix", {}).get(ind, {})
            cells = [row.get(c, "—") for c in OECD_COUNTRIES]
            lines.append(f"| {ind} | " + " | ".join(cells) + " |")
        lines.append("")
        if oecd.get("missing_indicators"):
            lines.append(f"- **Missing indicators**: {oecd['missing_indicators']}")
        lines.append("")

    # ── BEA section ───────────────────────────────────────────────────────────
    lines += ["## 7. BEA Input-Output Supply-Chain", ""]
    if bea.get("status") == "MISSING":
        lines.append("❌ BEA PIT shard missing.\n")
    else:
        lines += [
            f"- Total observations: {bea['total_obs']:,}",
            f"- Sector pairs: {bea['sector_pairs']}",
            f"- Sectors mapped: {bea['sectors_mapped']}",
            f"- Years covered: {bea['years']}",
            f"- Date range: {bea['date_range'][0]} – {bea['date_range'][1]}",
        ]
        if bea.get("sectors_missing"):
            lines.append(f"- **Missing sectors**: {bea['sectors_missing']}")
        else:
            lines.append("- All 5 SMIM sectors present ✅")
        lines.append("")

    # ── A1 leak detection ─────────────────────────────────────────────────────
    lines += ["## 8. A1 Leak Detection (pub_date < event_date)", ""]
    if leak["passed"]:
        lines += [
            f"✅ **PASSED** — 0 violations across {leak['rows_checked']:,} rows checked.",
            "",
        ]
    else:
        lines += [
            f"❌ **FAILED** — {len(leak['violations'])} violations in {leak['rows_checked']:,} rows.",
            "",
            "| Source | Actor | Signal | event_date | pub_date |",
            "|--------|-------|--------|------------|----------|",
        ]
        for v in leak["violations"][:20]:
            lines.append(
                f"| {v['source']} | {v['actor_id']} | {v['signal_id']} "
                f"| {v['event_date']} | {v['pub_date']} |"
            )
        lines.append("")

    # ── Gate G1 checklist ─────────────────────────────────────────────────────
    lines += ["## Gate G1 Checklist", ""]

    # G1-1: Leak check
    leak_ok = leak["passed"]
    lines.append(f"- [{'x' if leak_ok else ' '}] **G1-1** A1 compliance: 0 pub_date < event_date leaks")

    # G1-2: FRED coverage ≥80%
    fred_pct = (fred.get("fetched_count", 0) / fred.get("planned", 1) * 100) if fred.get("status") == "OK" else 0
    fred_ok = fred_pct >= 80
    lines.append(f"- [{'x' if fred_ok else ' '}] **G1-2** FRED: {fred.get('fetched_count',0)}/{fred.get('planned',0)} series ({fred_pct:.0f}%) ≥ 80%")

    # G1-3: EDGAR coverage
    total_edgar_tickers = 0
    covered_edgar_tickers = 0
    for ucov in edgar.get("universe_coverage", {}).values():
        total_edgar_tickers += ucov.get("tickers_total", 0)
        covered_edgar_tickers += ucov.get("tickers_covered", 0)
    edgar_pct = covered_edgar_tickers / total_edgar_tickers * 100 if total_edgar_tickers else 0
    edgar_ok = edgar_pct >= 80
    lines.append(f"- [{'x' if edgar_ok else ' '}] **G1-3** EDGAR: {covered_edgar_tickers}/{total_edgar_tickers} tickers ({edgar_pct:.0f}%) ≥ 80%")

    # G1-4: GDELT continuity
    gdelt_ok = gdelt.get("status") == "OK" and not any(v for v in gdelt.get("actor_gaps", {}).values())
    lines.append(f"- [{'x' if gdelt_ok else ' '}] **G1-4** GDELT: weekly continuity since 2015, no >4-week gaps")

    # G1-5: IMF completeness
    imf_missing = len(imf.get("missing_indicators", [])) + len(imf.get("missing_countries", []))
    imf_ok = imf.get("status") == "OK" and imf_missing == 0
    lines.append(f"- [{'x' if imf_ok else ' '}] **G1-5** IMF: all {len(IMF_INDICATORS)} indicators × {len(IMF_COUNTRIES)} countries present")

    # G1-6: OECD completeness
    oecd_missing = len(oecd.get("missing_indicators", [])) + len(oecd.get("missing_countries", []))
    oecd_ok = oecd.get("status") == "OK" and oecd_missing == 0
    lines.append(f"- [{'x' if oecd_ok else ' '}] **G1-6** OECD: all {len(OECD_INDICATORS)} indicators × {len(OECD_COUNTRIES)} countries present")

    # G1-7: BEA sectors
    bea_ok = bea.get("status") == "OK" and not bea.get("sectors_missing")
    lines.append(f"- [{'x' if bea_ok else ' '}] **G1-7** BEA: all {len(BEA_SECTORS)} SMIM sectors mapped")

    # G1-8: OHLCV Gold/Silver ≥60% of tickers
    gold_silver_count = 0
    total_ohlcv = 0
    for o in ohlcv.values():
        rc = o.get("regime_counts", {})
        gold_silver_count += rc.get("Gold", 0) + rc.get("Silver", 0)
        total_ohlcv += o.get("tickers_total", 0)
    gs_pct = gold_silver_count / total_ohlcv * 100 if total_ohlcv else 0
    gs_ok = gs_pct >= 60
    lines.append(f"- [{'x' if gs_ok else ' '}] **G1-8** OHLCV: {gold_silver_count}/{total_ohlcv} tickers ({gs_pct:.0f}%) Gold/Silver regime ≥ 60%")

    all_ok = all([leak_ok, fred_ok, edgar_ok, gdelt_ok, imf_ok, oecd_ok, bea_ok, gs_ok])
    lines += [
        "",
        f"### Overall: {'✅ GATE G1 PASSED' if all_ok else '❌ GATE G1 NOT YET PASSED'}",
        "",
    ]

    # ── Recommendations ───────────────────────────────────────────────────────
    lines += ["## Recommendations", ""]
    recs: list[str] = []

    if not leak_ok:
        recs.append("**CRITICAL** Fix A1 leak violations before any backtest.")
    if not fred_ok:
        recs.append(f"Re-run `smim_fetch_fred.py` — missing series: {fred.get('missing', [])}.")
    if not edgar_ok:
        tickers_missing_edgar = sorted({
            t for ucov in edgar.get("universe_coverage", {}).values()
            for t in ucov.get("no_data", [])
        })
        recs.append(f"EDGAR: {len(tickers_missing_edgar)} tickers lack any filing. "
                    f"Consider dropping from universe or using SIC proxies.")
    if not gdelt_ok:
        recs.append("GDELT: some actors have >4 consecutive week gaps — backfill or interpolate.")
    if not imf_ok:
        recs.append(f"IMF: missing indicators/countries: {imf.get('missing_indicators',[])} / {imf.get('missing_countries',[])}.")
    if not oecd_ok:
        recs.append(f"OECD: missing indicators/countries: {oecd.get('missing_indicators',[])} / {oecd.get('missing_countries',[])}.")
    if not bea_ok:
        recs.append(f"BEA: missing sectors: {bea.get('sectors_missing',[])}. Re-run `smim_fetch_bea.py`.")
    if not gs_ok:
        recs.append("OHLCV: many tickers in Sparse regime. Consider restricting to post-2010 backtest window.")

    if not recs:
        recs.append("All checks passed — data is ready for Gate G1 experiment programme.")

    for r in recs:
        lines.append(f"1. {r}")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("SMIM Data Coverage Audit")
    log.info("=" * 60)

    # Load universes
    log.info("\nLoading universe CSVs …")
    universes = load_universes()
    log.info("  → %d universes loaded", len(universes))

    # Load PIT shards
    log.info("\nLoading PIT store shards …")
    pit_fred  = load_pit_source("fred")
    pit_edgar = load_pit_source("edgar")
    pit_gdelt = load_pit_source("gdelt")
    pit_imf   = load_pit_source("imf")
    pit_oecd  = load_pit_source("oecd")
    pit_bea   = load_pit_source("bea")

    # Run audits
    log.info("\nAuditing OHLCV …")
    ohlcv_audit = audit_ohlcv(universes)

    log.info("Auditing FRED …")
    fred_audit = audit_fred(pit_fred)

    log.info("Auditing EDGAR …")
    edgar_audit = audit_edgar(pit_edgar, universes)

    log.info("Auditing GDELT …")
    gdelt_audit = audit_gdelt(pit_gdelt)

    log.info("Auditing IMF …")
    imf_audit = audit_imf(pit_imf)

    log.info("Auditing OECD …")
    oecd_audit = audit_oecd(pit_oecd)

    log.info("Auditing BEA …")
    bea_audit = audit_bea(pit_bea)

    log.info("Running A1 leak detection …")
    leak_result = run_leak_detection()
    if leak_result["passed"]:
        log.info("  ✅ A1 leak check PASSED (%d rows, 0 violations)", leak_result["rows_checked"])
    else:
        log.warning("  ❌ A1 leak check FAILED — %d violations", len(leak_result["violations"]))

    # Build manifest
    log.info("Building universe manifest …")
    manifest = build_manifest(universes, ohlcv_audit, edgar_audit)

    # Render report
    log.info("Rendering audit report …")
    report_md = render_report(
        ohlcv_audit, fred_audit, edgar_audit, gdelt_audit,
        imf_audit, oecd_audit, bea_audit, leak_result,
        universes, manifest,
    )

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    log.info("Report written → %s", REPORT_PATH.relative_to(_ROOT))

    # Save manifest
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("Manifest written → %s", MANIFEST_PATH.relative_to(_ROOT))

    # Print Gate G1 summary to stdout
    print("\n" + "=" * 60)
    print("GATE G1 CHECKLIST")
    print("=" * 60)
    leak_ok  = leak_result["passed"]
    fred_pct = (fred_audit.get("fetched_count", 0) / fred_audit.get("planned", 1) * 100) if fred_audit.get("status") == "OK" else 0
    fred_ok  = fred_pct >= 80

    total_et = sum(v.get("tickers_total", 0) for v in edgar_audit.get("universe_coverage", {}).values())
    covered_et = sum(v.get("tickers_covered", 0) for v in edgar_audit.get("universe_coverage", {}).values())
    edgar_pct = covered_et / total_et * 100 if total_et else 0
    edgar_ok = edgar_pct >= 80

    gdelt_ok = gdelt_audit.get("status") == "OK" and not any(v for v in gdelt_audit.get("actor_gaps", {}).values())
    imf_ok = imf_audit.get("status") == "OK" and not (imf_audit.get("missing_indicators") or imf_audit.get("missing_countries"))
    oecd_ok = oecd_audit.get("status") == "OK" and not (oecd_audit.get("missing_indicators") or oecd_audit.get("missing_countries"))
    bea_ok = bea_audit.get("status") == "OK" and not bea_audit.get("sectors_missing")

    gs_count = sum(o.get("regime_counts", {}).get("Gold", 0) + o.get("regime_counts", {}).get("Silver", 0) for o in ohlcv_audit.values())
    gs_total = sum(o.get("tickers_total", 0) for o in ohlcv_audit.values())
    gs_pct = gs_count / gs_total * 100 if gs_total else 0
    gs_ok = gs_pct >= 60

    checks = [
        (leak_ok,   f"G1-1  A1 leak: 0 violations / {leak_result['rows_checked']:,} rows"),
        (fred_ok,   f"G1-2  FRED: {fred_audit.get('fetched_count',0)}/{fred_audit.get('planned',0)} series ({fred_pct:.0f}%)"),
        (edgar_ok,  f"G1-3  EDGAR: {covered_et}/{total_et} tickers ({edgar_pct:.0f}%)"),
        (gdelt_ok,  f"G1-4  GDELT: weekly continuity since 2015"),
        (imf_ok,    f"G1-5  IMF: {len(IMF_INDICATORS)} indicators × {len(IMF_COUNTRIES)} countries"),
        (oecd_ok,   f"G1-6  OECD: {len(OECD_INDICATORS)} indicators × {len(OECD_COUNTRIES)} countries"),
        (bea_ok,    f"G1-7  BEA: {len(BEA_SECTORS)} sectors mapped"),
        (gs_ok,     f"G1-8  OHLCV Gold/Silver: {gs_count}/{gs_total} ({gs_pct:.0f}%)"),
    ]
    for ok, desc in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {desc}")

    all_ok = all(ok for ok, _ in checks)
    print("=" * 60)
    print(f"  {'GATE G1 PASSED' if all_ok else 'GATE G1 NOT YET PASSED'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
