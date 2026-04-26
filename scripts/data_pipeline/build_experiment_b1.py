"""Build experiment_b1 — UK/EU heterogeneous panel.

End-to-end pipeline:
  1. Select ~100 EU firms by coverage, market cap, and sector balance
  2. Compute CapEx/Assets cross-sectional percentile ranks (Layer 2)
  3. Build macro shock actors from FRED series (Layer 0)
  4. Build institutional proxy actors (Layer 1)
  5. Output registry JSON + intensities parquet matching experiment_a1 schema

Usage:
    uv run python scripts/data_pipeline/build_experiment_b1.py

    # Override firm count target
    uv run python scripts/data_pipeline/build_experiment_b1.py --n-firms 80

Outputs:
    data/registries/experiment_b1_registry.json
    data/intensities/experiment_b1_intensities.parquet
    data/audit/experiment_b1_firm_selection.csv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

RAW_DIR = _ROOT / "data" / "raw" / "eodhd"
REGISTRY_DIR = _ROOT / "data" / "registries"
INTENSITY_DIR = _ROOT / "data" / "intensities"
AUDIT_DIR = _ROOT / "data" / "audit"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_b1")

# ── Config ────────────────────────────────────────────────────────────────────

PANEL_START = pd.Timestamp("2011-04-01")
PANEL_END = pd.Timestamp("2025-12-31")
QUARTER_STARTS = pd.date_range(PANEL_START, PANEL_END, freq="QS")
N_QUARTERS = len(QUARTER_STARTS)

DEFAULT_N_FIRMS = 100
MIN_QUARTERS_BOTH = 50
MIN_SECTORS = 6
MIN_FIRMS_PER_SECTOR = 3

EU_UK_ISIN_PREFIXES = {
    "GB", "DE", "FR", "ES", "IT", "NL", "BE", "AT", "FI", "SE", "DK", "NO",
    "CH", "IE", "LU", "PT", "GR", "PL", "CZ", "HU", "IS", "LI", "SK", "SI",
    "EE", "LV", "LT", "MT", "CY", "RO", "BG", "HR",
}

EODHD_TO_SMIM_SECTOR = {
    "Energy": "energy",
    "Financial Services": "financials",
    "Healthcare": "healthcare",
    "Technology": "technology",
    "Communication Services": "technology",
    "Consumer Cyclical": "consumer",
    "Consumer Defensive": "consumer",
    "Industrials": "industrials",
    "Basic Materials": "industrials",
    "Real Estate": "financials",
    "Utilities": "energy",
    "Other": "diversified",
}

# Layer 0 macro shocks — actor_id -> FRED series in macro_eu.parquet
MACRO_ACTORS = {
    "shock_brent_crude": {"source_id": "shock_brent", "name": "Brent Crude Oil", "sector": "energy"},
    "shock_ecb_rate": {"source_id": "shock_ecb_rate", "name": "ECB Deposit Facility Rate", "sector": "macro"},
    "shock_vix": {"source_id": "shock_vix", "name": "CBOE VIX", "sector": "macro"},
    "shock_eu_10y_yield": {"source_id": "shock_eu_10y_yield", "name": "Euro Area 10Y Yield", "sector": "macro"},
    "shock_eur_usd": {"source_id": "shock_eur_usd", "name": "EUR/USD Exchange Rate", "sector": "macro"},
    "shock_eu_hicp": {"source_id": "shock_eu_hicp", "name": "Euro Area HICP Inflation", "sector": "macro"},
}

# Layer 1 institutional proxies — built from macro data with different transforms
INST_ACTORS = {
    "inst_ecb": {"source_id": "shock_ecb_rate", "name": "ECB (policy stance)", "sector": "macro",
                 "transform": "cumulative_change"},
    "inst_boe": {"source_id": "__boe_bank_rate__", "name": "Bank of England (policy stance)", "sector": "macro",
                 "transform": "cumulative_change"},
    "inst_imf": {"source_id": "macro_eu_gdp_growth", "name": "IMF / EU GDP Growth", "sector": "macro",
                 "transform": "level"},
}

BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?"
    "csv.x=yes&Datefrom=01/Jan/2000&Dateto=31/Dec/2025&SeriesCodes=IUDBEDR"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo or pd.isna(hi - lo):
        return series.where(series.isna(), 0.5)
    return (series - lo) / (hi - lo)


def cross_section_rank(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.rank(axis=1, pct=True, na_option="keep")


# ── Step 1: Firm selection ────────────────────────────────────────────────────

def select_firms(n_target: int, eu_only: bool = False) -> pd.DataFrame:
    log.info("Step 1: Firm selection (target %d, eu_only=%s)", n_target, eu_only)
    cov = pd.read_csv(RAW_DIR / "coverage_summary.csv")
    meta = pd.read_parquet(RAW_DIR / "firm_metadata.parquet")
    fund = pd.read_parquet(RAW_DIR / "fundamentals_quarterly.parquet")

    merged = cov.merge(meta, left_on=["ticker", "exchange"], right_on=["ticker", "exchange_code"],
                       how="left", suffixes=("", "_meta"))

    # Filter
    has_sector = merged["sector"].fillna("").str.strip() != ""
    has_coverage = merged["both_60q"] == 1
    not_iob = ~((merged["exchange"] == "LSE") & merged["ticker"].str.match(r"^0[A-Z0-9]"))
    candidates = merged[has_sector & has_coverage & not_iob].copy()
    log.info("  Candidates after listing+sector+coverage filters: %d", len(candidates))

    if eu_only:
        candidates["__isin_prefix"] = candidates["isin"].fillna("").str[:2]
        before = len(candidates)
        candidates = candidates[candidates["__isin_prefix"].isin(EU_UK_ISIN_PREFIXES)].copy()
        log.info("  After EU/UK ISIN filter: %d (dropped %d non-EU/UK)",
                 len(candidates), before - len(candidates))
        candidates = candidates.drop(columns=["__isin_prefix"])

    # Map to SMIM sectors
    candidates["smim_sector"] = candidates["sector"].map(EODHD_TO_SMIM_SECTOR).fillna("diversified")

    # Use latest totalAssets as size proxy (market_cap is often empty in EODHD metadata)
    bs_recent = fund[(fund["statement"] == "BS") & (fund["date"] >= "2024-01-01")].copy()
    latest_assets = bs_recent.groupby(["ticker", "exchange"])["totalAssets"].last().reset_index()
    latest_assets.columns = ["ticker", "exchange", "latest_assets"]
    candidates = candidates.merge(latest_assets, on=["ticker", "exchange"], how="left")
    candidates["size_proxy"] = candidates["latest_assets"].fillna(0)

    # Deduplicate share classes: group by ISIN, keep the one with most CapEx quarters
    candidates["isin_clean"] = candidates["isin"].fillna("").str.strip()
    has_isin = candidates["isin_clean"] != ""
    # For firms with ISIN: keep best coverage per ISIN
    deduped_parts = []
    if has_isin.any():
        isin_group = candidates[has_isin].sort_values("n_capex_2011_2025", ascending=False)
        isin_deduped = isin_group.drop_duplicates(subset="isin_clean", keep="first")
        deduped_parts.append(isin_deduped)
    if (~has_isin).any():
        deduped_parts.append(candidates[~has_isin])
    candidates = pd.concat(deduped_parts, ignore_index=True)
    log.info("  After deduplication: %d", len(candidates))

    # Sort by size (descending)
    candidates = candidates.sort_values("size_proxy", ascending=False)

    # Geographic + sector balanced selection
    MAX_PER_EXCHANGE_PCT = 0.35  # no exchange > 35% of panel
    max_per_exchange = max(10, int(n_target * MAX_PER_EXCHANGE_PCT))

    sectors = sorted(candidates["smim_sector"].unique())
    n_sectors = len(sectors)
    base_per_sector = max(MIN_FIRMS_PER_SECTOR, (n_target + n_sectors - 1) // n_sectors)

    selected: list[pd.DataFrame] = []
    exchange_counts: dict[str, int] = {}

    for sector in sectors:
        sector_firms = candidates[candidates["smim_sector"] == sector].copy()
        n_taken = 0
        for _, row in sector_firms.iterrows():
            if n_taken >= base_per_sector:
                break
            ex = row["exchange"]
            if exchange_counts.get(ex, 0) >= max_per_exchange:
                continue
            selected.append(row.to_frame().T)
            exchange_counts[ex] = exchange_counts.get(ex, 0) + 1
            n_taken += 1

    selection = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    # Fill remaining slots from largest unselected firms (with geographic cap)
    if len(selection) < n_target:
        selected_tickers = set(zip(selection["ticker"], selection["exchange"]))
        remaining = candidates[
            candidates.apply(lambda r: (r["ticker"], r["exchange"]) not in selected_tickers, axis=1)
        ]
        for _, row in remaining.iterrows():
            if len(selection) >= n_target:
                break
            ex = row["exchange"]
            if exchange_counts.get(ex, 0) >= max_per_exchange:
                continue
            selection = pd.concat([selection, row.to_frame().T], ignore_index=True)
            exchange_counts[ex] = exchange_counts.get(ex, 0) + 1

    # Trim if over target
    if len(selection) > n_target:
        selection = selection.sort_values("size_proxy", ascending=False).head(n_target)

    log.info("  Selected %d firms across %d SMIM sectors",
             len(selection), selection["smim_sector"].nunique())
    for sec in sorted(selection["smim_sector"].unique()):
        n = (selection["smim_sector"] == sec).sum()
        log.info("    %-15s: %d firms", sec, n)

    log.info("  Geographic mix:")
    for ex in sorted(selection["exchange"].unique()):
        n = (selection["exchange"] == ex).sum()
        log.info("    %-6s: %d firms", ex, n)

    return selection


# ── Step 2: Compute firm intensities (Layer 2) ───────────────────────────────

def build_firm_intensities(selection: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 2: Computing CapEx/Assets cross-sectional ranks")
    fund = pd.read_parquet(RAW_DIR / "fundamentals_quarterly.parquet")

    tickers = set(zip(selection["ticker"], selection["exchange"]))
    bs = fund[(fund["statement"] == "BS")].copy()
    cf = fund[(fund["statement"] == "CF")].copy()

    # Filter to selected firms
    bs = bs[bs.apply(lambda r: (r["ticker"], r["exchange"]) in tickers, axis=1)]
    cf = cf[cf.apply(lambda r: (r["ticker"], r["exchange"]) in tickers, axis=1)]

    # Create unique actor_id: ticker.exchange
    bs["actor_id"] = bs["ticker"] + "." + bs["exchange"]
    cf["actor_id"] = cf["ticker"] + "." + cf["exchange"]

    # Parse dates and align to quarter starts
    bs["date"] = pd.to_datetime(bs["date"])
    cf["date"] = pd.to_datetime(cf["date"])
    bs["quarter"] = bs["date"].dt.to_period("Q").dt.start_time
    cf["quarter"] = cf["date"].dt.to_period("Q").dt.start_time

    # Pivot to (T x N) panels
    assets_panel = bs.pivot_table(index="quarter", columns="actor_id", values="totalAssets", aggfunc="last")
    capex_panel = cf.pivot_table(index="quarter", columns="actor_id", values="capitalExpenditures", aggfunc="last")

    # Align to our quarter grid
    assets_panel = assets_panel.reindex(QUARTER_STARTS).ffill(limit=1)
    capex_panel = capex_panel.reindex(QUARTER_STARTS).ffill(limit=1)

    # CapEx/Assets ratio (CapEx is positive in EODHD, Assets positive)
    assets_clean = assets_panel.where(assets_panel > 0)
    ratio = capex_panel.abs().divide(assets_clean)
    ratio = ratio.clip(lower=0, upper=1)

    # Cross-sectional percentile rank per quarter
    intensity = cross_section_rank(ratio)

    log.info("  Firm intensity panel: %d quarters x %d firms", intensity.shape[0], intensity.shape[1])
    log.info("  Non-NaN coverage: %.1f%%", 100 * intensity.notna().mean().mean())

    # Convert to long format
    rows = []
    for quarter in intensity.index:
        for actor_id in intensity.columns:
            val = intensity.loc[quarter, actor_id]
            if pd.notna(val):
                rows.append({
                    "actor_id": actor_id,
                    "period": quarter,
                    "intensity_value": float(val),
                    "normalisation_method": "capex_assets_xsrank",
                })

    return pd.DataFrame(rows)


# ── Step 3: Build macro intensities (Layer 0) ────────────────────────────────

def build_macro_intensities() -> pd.DataFrame:
    log.info("Step 3: Building macro shock intensities (Layer 0)")
    macro = pd.read_parquet(RAW_DIR / "macro_eu.parquet")

    rows = []
    for actor_id, cfg in MACRO_ACTORS.items():
        source_id = cfg["source_id"]
        sub = macro[macro["actor_id"] == source_id].copy()
        if sub.empty:
            log.warning("  %s: no data for source %s", actor_id, source_id)
            continue
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date")

        # Resample to quarterly (last observation per quarter)
        sub = sub.set_index("date")
        quarterly = sub["value"].resample("QS").last()
        quarterly = quarterly.reindex(QUARTER_STARTS).ffill()

        # Min-max normalize
        normalized = minmax_normalize(quarterly)

        for period, val in normalized.items():
            if pd.notna(val):
                rows.append({
                    "actor_id": actor_id,
                    "period": period,
                    "intensity_value": float(val),
                    "normalisation_method": "fred_minmax",
                })

        log.info("  %-25s n=%d quarters, range [%.3f, %.3f]",
                 actor_id, normalized.notna().sum(), normalized.min(), normalized.max())

    return pd.DataFrame(rows)


# ── Step 4: Build institutional intensities (Layer 1) ─────────────────────────

def _fetch_boe_bank_rate() -> pd.Series:
    try:
        r = requests.get(BOE_URL, timeout=30, headers={"User-Agent": "HARP Research"})
        if r.status_code != 200:
            log.warning("BoE IADB returned HTTP %s", r.status_code)
            return pd.Series(dtype=float)
        df = pd.read_csv(io.StringIO(r.text))
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna()
        return df.set_index(date_col)[val_col].sort_index()
    except Exception as e:
        log.warning("BoE fetch failed: %s", e)
        return pd.Series(dtype=float)


def build_institutional_intensities() -> pd.DataFrame:
    log.info("Step 4: Building institutional intensities (Layer 1)")
    macro = pd.read_parquet(RAW_DIR / "macro_eu.parquet")

    # Fetch BoE Bank Rate
    boe_series = _fetch_boe_bank_rate()
    if not boe_series.empty:
        log.info("  BoE Bank Rate: %d obs, %s -> %s",
                 len(boe_series), boe_series.index[0].strftime("%Y-%m-%d"),
                 boe_series.index[-1].strftime("%Y-%m-%d"))

    rows = []
    for actor_id, cfg in INST_ACTORS.items():
        source_id = cfg["source_id"]
        transform = cfg["transform"]

        if source_id == "__boe_bank_rate__":
            if boe_series.empty:
                log.warning("  %s: BoE data unavailable, skipping", actor_id)
                continue
            quarterly = boe_series.resample("QS").last().reindex(QUARTER_STARTS).ffill()
        else:
            sub = macro[macro["actor_id"] == source_id].copy()
            if sub.empty:
                log.warning("  %s: no data for %s", actor_id, source_id)
                continue
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date").set_index("date")
            quarterly = sub["value"].resample("QS").last().reindex(QUARTER_STARTS).ffill()

        if transform == "cumulative_change":
            first_valid = quarterly.first_valid_index()
            if first_valid is not None:
                quarterly = quarterly - quarterly.loc[first_valid]

        normalized = minmax_normalize(quarterly)

        for period, val in normalized.items():
            if pd.notna(val):
                rows.append({
                    "actor_id": actor_id,
                    "period": period,
                    "intensity_value": float(val),
                    "normalisation_method": "institutional_minmax",
                })

        log.info("  %-25s n=%d quarters, transform=%s",
                 actor_id, normalized.notna().sum(), transform)

    return pd.DataFrame(rows)


# ── Step 5: Build registry JSON ───────────────────────────────────────────────

def build_registry(selection: pd.DataFrame) -> dict:
    log.info("Step 5: Building registry JSON")
    actors = []

    # Layer 0 — macros
    for actor_id, cfg in MACRO_ACTORS.items():
        actors.append({
            "actor_id": actor_id,
            "name": cfg["name"],
            "actor_type": "global_shock",
            "layer": 0,
            "geography": "EU",
            "sector": cfg["sector"],
            "external_ids": {"fred_series": cfg["source_id"]},
        })

    # Layer 1 — institutions
    for actor_id, cfg in INST_ACTORS.items():
        actors.append({
            "actor_id": actor_id,
            "name": cfg["name"],
            "actor_type": "central_bank" if "ecb" in actor_id or "boe" in actor_id else "intl_org",
            "layer": 1,
            "geography": "EU" if "ecb" in actor_id else ("UK" if "boe" in actor_id else "GLOBAL"),
            "sector": cfg["sector"],
            "external_ids": {},
        })

    # Layer 2 — firms
    for _, row in selection.iterrows():
        actor_id = f"{row['ticker']}.{row['exchange']}"
        actors.append({
            "actor_id": actor_id,
            "name": row.get("name", row["ticker"]),
            "actor_type": "sector_leader",
            "layer": 2,
            "geography": row.get("country", "EU"),
            "sector": row.get("smim_sector", "diversified"),
            "external_ids": {
                "eodhd_ticker": row["ticker"],
                "eodhd_exchange": row["exchange"],
                "isin": str(row.get("isin", "")),
            },
        })

    return {"actors": actors}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-firms", type=int, default=DEFAULT_N_FIRMS)
    parser.add_argument("--eu-only", action="store_true",
                        help="Filter firms to EU/UK ISIN prefixes only")
    parser.add_argument("--variant", type=str, default="",
                        help="Suffix for output files (e.g. 'eu_only' produces experiment_b1_eu_only_*)")
    args = parser.parse_args()

    suffix = f"_{args.variant}" if args.variant else ""
    panel_name = f"experiment_b1{suffix}"

    log.info("=" * 70)
    log.info("Building %s — UK/EU Heterogeneous Panel", panel_name)
    log.info("Window: %s to %s (%d quarters)",
             PANEL_START.strftime("%Y-%m-%d"), PANEL_END.strftime("%Y-%m-%d"), N_QUARTERS)
    log.info("EU-only ISIN filter: %s", args.eu_only)
    log.info("=" * 70)

    # Step 1
    selection = select_firms(args.n_firms, eu_only=args.eu_only)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(AUDIT_DIR / f"{panel_name}_firm_selection.csv", index=False)

    # Steps 2-4
    firm_df = build_firm_intensities(selection)
    macro_df = build_macro_intensities()
    inst_df = build_institutional_intensities()

    # Merge all intensities
    all_intensities = pd.concat([macro_df, inst_df, firm_df], ignore_index=True)
    all_intensities["period"] = pd.to_datetime(all_intensities["period"])

    # Step 5: Registry
    registry = build_registry(selection)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = REGISTRY_DIR / f"{panel_name}_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)
    log.info("Saved %s (%d actors)", reg_path.name, len(registry["actors"]))

    # Save intensities
    INTENSITY_DIR.mkdir(parents=True, exist_ok=True)
    int_path = INTENSITY_DIR / f"{panel_name}_intensities.parquet"
    all_intensities.to_parquet(int_path, index=False)
    log.info("Saved %s (%d rows)", int_path.name, len(all_intensities))

    # ── Summary ───────────────────────────────────────────────────────────────

    n_actors = all_intensities["actor_id"].nunique()
    n_quarters_actual = all_intensities["period"].nunique()

    print("\n" + "=" * 70)
    print("EXPERIMENT_B1 BUILD SUMMARY")
    print("=" * 70)
    print(f"  Total actors:      {n_actors}")
    print(f"  Layer 0 (macro):   {len(MACRO_ACTORS)}")
    print(f"  Layer 1 (inst):    {len(INST_ACTORS)}")
    print(f"  Layer 2 (firms):   {len(selection)}")
    print(f"  Quarters:          {n_quarters_actual}")
    print(f"  Total intensity rows: {len(all_intensities):,}")
    print(f"  Intensity methods: {all_intensities['normalisation_method'].unique().tolist()}")
    print(f"  Period range:      {all_intensities['period'].min()} -> {all_intensities['period'].max()}")
    print(f"")
    print(f"  Registry:    {reg_path.relative_to(_ROOT)}")
    print(f"  Intensities: {int_path.relative_to(_ROOT)}")
    print(f"  Selection:   {(AUDIT_DIR / f'{panel_name}_firm_selection.csv').relative_to(_ROOT)}")

    # Validate
    issues = []
    if n_actors < 50:
        issues.append(f"Too few actors: {n_actors}")
    if n_quarters_actual < 40:
        issues.append(f"Too few quarters: {n_quarters_actual}")
    sector_counts = selection["smim_sector"].value_counts()
    if len(sector_counts) < MIN_SECTORS:
        issues.append(f"Only {len(sector_counts)} sectors (need {MIN_SECTORS})")
    for sec, n in sector_counts.items():
        if n < MIN_FIRMS_PER_SECTOR:
            issues.append(f"Sector '{sec}' has only {n} firms (need {MIN_FIRMS_PER_SECTOR})")

    if issues:
        print(f"\n  WARNINGS:")
        for iss in issues:
            print(f"    ! {iss}")
    else:
        print(f"\n  >> ALL VALIDATION CHECKS PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
