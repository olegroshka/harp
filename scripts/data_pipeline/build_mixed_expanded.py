"""Build expanded MIXED-200 registry (~120 actors, multi-sector, cross-geography).

Selection:
  - Keep all 38 actors from current experiment_a1_registry.json
  - Add top 15 US-LC-TECH large_firm (by EDGAR CapEx coverage)
  - Add top 10 US-LC-FINS bank (by EDGAR Assets coverage)
  - Add top 10 US-LC-HEALTH large_firm (by EDGAR CapEx coverage)
  - Add top 10 US-LC-INDUS large_firm (by EDGAR CapEx coverage)
  - Add top 15 UK-LC large_firm (by OHLCV return intensity coverage)
  - Deduplicate by actor_id

Writes:
  data/smim/registries/experiment_a1_registry.json  (updated, all layers)
  data/smim/registries/MIXED-200_registry.json       (equity + shock, L0+L2)
  data/smim/universes/MIXED-200.csv                  (equity universe CSV)

Usage:
    uv run python scripts/smim/smim_build_mixed_expanded.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REGISTRY_DIR   = _ROOT / "data" / "registries"
UNIVERSE_DIR   = _ROOT / "data" / "universes"
INTENSITY_DIR  = _ROOT / "data" / "intensities"
EDGAR_PATH     = _ROOT / "data" / "processed" / "edgar_balance_sheet.parquet"

CAPEX_TAG  = "PaymentsToAcquirePropertyPlantAndEquipment"
ASSETS_TAG = "Assets"


# ── Coverage helpers ──────────────────────────────────────────────────────────

def edgar_coverage(tickers: list[str], tag: str) -> dict[str, int]:
    """Return {ticker: n_distinct_quarters_with_data} from EDGAR PIT store."""
    edgar = pd.read_parquet(EDGAR_PATH)
    sub = edgar[edgar["tag"] == tag].copy()
    sub["event_date"] = pd.to_datetime(sub["event_date"])
    sub["qtr"] = sub["event_date"].dt.to_period("Q")
    coverage = (
        sub[sub["ticker"].isin(tickers)]
        .groupby("ticker")["qtr"]
        .nunique()
        .to_dict()
    )
    return coverage


def ohlcv_coverage(tickers: list[str], uni_id: str) -> dict[str, int]:
    """Return {ticker: n_non_null_quarters} from return intensity parquet."""
    return_path = INTENSITY_DIR / f"{uni_id}_return_intensities.parquet"
    if not return_path.exists():
        log.warning("  No return intensities for %s — using alphabetical order", uni_id)
        return {t: 0 for t in tickers}
    df = pd.read_parquet(return_path)
    coverage = (
        df[df["actor_id"].isin(tickers)]
        .groupby("actor_id")["intensity_value"]
        .count()
        .to_dict()
    )
    return coverage


def top_by_coverage(actors: list[dict], coverage: dict[str, int], n: int) -> list[dict]:
    """Return top-n actors sorted descending by coverage count."""
    scored = sorted(actors, key=lambda a: coverage.get(a["actor_id"], 0), reverse=True)
    return scored[:n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load base experiment_a1 actors ────────────────────────────────
    a1_path = REGISTRY_DIR / "experiment_a1_registry.json"
    a1_data = json.loads(a1_path.read_text())
    base_actors: list[dict] = list(a1_data["actors"])
    seen_ids: set[str] = {a["actor_id"] for a in base_actors}
    log.info("Base experiment_a1 actors: %d", len(base_actors))

    # ── Step 2: US-LC-TECH large_firm top 15 by CapEx coverage ───────────────
    tech_data = json.loads((REGISTRY_DIR / "US-LC-TECH_registry.json").read_text())
    tech_large_firm = [a for a in tech_data["actors"] if a["actor_type"] == "large_firm"]
    tech_tickers = [a["actor_id"] for a in tech_large_firm]
    tech_cov = edgar_coverage(tech_tickers, CAPEX_TAG)
    tech_top = top_by_coverage(tech_large_firm, tech_cov, 15)
    tech_added = [a for a in tech_top if a["actor_id"] not in seen_ids]
    seen_ids.update(a["actor_id"] for a in tech_added)
    log.info("US-LC-TECH large_firm: selected %d/%d (top 15 by CapEx coverage)",
             len(tech_added), len(tech_large_firm))

    # ── Step 3: US-LC-FINS bank top 10 by Assets coverage ────────────────────
    fins_data = json.loads((REGISTRY_DIR / "US-LC-FINS_registry.json").read_text())
    fins_banks = [a for a in fins_data["actors"] if a["actor_type"] == "bank"]
    fins_tickers = [a["actor_id"] for a in fins_banks]
    fins_cov = edgar_coverage(fins_tickers, ASSETS_TAG)
    fins_top = top_by_coverage(fins_banks, fins_cov, 10)
    fins_added = [a for a in fins_top if a["actor_id"] not in seen_ids]
    seen_ids.update(a["actor_id"] for a in fins_added)
    log.info("US-LC-FINS bank: selected %d/%d (top 10 by Assets coverage)",
             len(fins_added), len(fins_banks))

    # ── Step 4: US-LC-HEALTH large_firm top 10 by CapEx coverage ─────────────
    health_data = json.loads((REGISTRY_DIR / "US-LC-HEALTH_registry.json").read_text())
    health_large_firm = [a for a in health_data["actors"] if a["actor_type"] == "large_firm"]
    health_tickers = [a["actor_id"] for a in health_large_firm]
    health_cov = edgar_coverage(health_tickers, CAPEX_TAG)
    health_top = top_by_coverage(health_large_firm, health_cov, 10)
    health_added = [a for a in health_top if a["actor_id"] not in seen_ids]
    seen_ids.update(a["actor_id"] for a in health_added)
    log.info("US-LC-HEALTH large_firm: selected %d/%d (top 10 by CapEx coverage)",
             len(health_added), len(health_large_firm))

    # ── Step 5: US-LC-INDUS large_firm top 10 by CapEx coverage ──────────────
    indus_data = json.loads((REGISTRY_DIR / "US-LC-INDUS_registry.json").read_text())
    indus_large_firm = [a for a in indus_data["actors"] if a["actor_type"] == "large_firm"]
    indus_tickers = [a["actor_id"] for a in indus_large_firm]
    indus_cov = edgar_coverage(indus_tickers, CAPEX_TAG)
    indus_top = top_by_coverage(indus_large_firm, indus_cov, 12)
    indus_added = [a for a in indus_top if a["actor_id"] not in seen_ids]
    seen_ids.update(a["actor_id"] for a in indus_added)
    log.info("US-LC-INDUS large_firm: selected %d/%d (top 12 by CapEx coverage)",
             len(indus_added), len(indus_large_firm))

    # ── Step 6: UK-LC large_firm top 15 by OHLCV return coverage ─────────────
    uklc_data = json.loads((REGISTRY_DIR / "UK-LC_registry.json").read_text())
    uklc_large_firm = [a for a in uklc_data["actors"] if a["actor_type"] == "large_firm"]
    uklc_tickers = [a["actor_id"] for a in uklc_large_firm]
    uklc_cov = ohlcv_coverage(uklc_tickers, "UK-LC")
    uklc_top = top_by_coverage(uklc_large_firm, uklc_cov, 20)
    uklc_added = [a for a in uklc_top if a["actor_id"] not in seen_ids]
    seen_ids.update(a["actor_id"] for a in uklc_added)
    log.info("UK-LC large_firm: selected %d/%d (top 20 by OHLCV coverage, de-duped)",
             len(uklc_added), len(uklc_large_firm))

    # ── Step 7: Assemble full registry ───────────────────────────────────────
    all_actors = (
        base_actors
        + tech_added
        + fins_added
        + health_added
        + indus_added
        + uklc_added
    )

    total = len(all_actors)
    log.info("Total actors: %d", total)
    log.info("By type: %s", dict(Counter(a["actor_type"] for a in all_actors)))
    log.info("By geo: %s", dict(Counter(a.get("geography", "?") for a in all_actors)))
    log.info("By sector: %s", dict(Counter(a.get("sector", "?") for a in all_actors)))

    assert 100 <= total <= 150, (
        f"Registry size {total} outside target range [100, 150]. "
        "Adjust selection counts in script."
    )

    # ── Step 8: Write experiment_a1_registry.json ─────────────────────────────
    a1_out = {"actors": all_actors}
    a1_path.write_text(json.dumps(a1_out, indent=2), encoding="utf-8")
    log.info("Written experiment_a1_registry.json (%d actors)", total)

    # ── Step 9: Write MIXED-200_registry.json (L0 shocks + L2 equity only) ───
    equity_types = {"large_firm", "bank", "sector_leader", "sme", "retail_investor"}
    shock_types  = {"global_shock"}
    mixed_actors = [
        a for a in all_actors
        if a["actor_type"] in equity_types or a["actor_type"] in shock_types
    ]
    mixed_path = REGISTRY_DIR / "MIXED-200_registry.json"
    mixed_path.write_text(json.dumps({"actors": mixed_actors}, indent=2), encoding="utf-8")
    log.info("Written MIXED-200_registry.json (%d actors, equity+shock)", len(mixed_actors))

    # ── Step 10: Write MIXED-200.csv (equity actors only for universe CSV) ───
    equity_actors_for_csv = [
        a for a in mixed_actors
        if a["actor_type"] in equity_types
    ]
    csv_rows = []
    for a in equity_actors_for_csv:
        csv_rows.append({
            "ticker": a["actor_id"],
            "name": a.get("name", a["actor_id"]),
            "sector": a.get("sector", ""),
            "gics_code": a.get("gics_code", ""),
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = UNIVERSE_DIR / "MIXED-200.csv"
    csv_df.to_csv(csv_path, index=False)
    log.info("Written MIXED-200.csv (%d equity actors)", len(csv_rows))

    # ── Summary ───────────────────────────────────────────────────────────────
    equity_count = sum(1 for a in all_actors if a["actor_type"] in equity_types)
    institutional_count = total - equity_count
    sectors = set(a.get("sector", "") for a in all_actors if a["actor_type"] in equity_types)
    geos = set(a.get("geography", "") for a in all_actors if a["actor_type"] in equity_types)
    layers = set(a.get("layer", "?") for a in all_actors)

    print(f"\nMIXED-200 expanded registry summary:")
    print(f"  Total actors:      {total}")
    print(f"  Equity (L2):       {equity_count}")
    print(f"  Institutional:     {institutional_count}")
    print(f"  Sectors:           {sorted(s for s in sectors if s)}")
    print(f"  Geographies:       {sorted(g for g in geos if g)}")
    print(f"  Layers:            {sorted(layers)}")
    print(f"\n  experiment_a1_registry.json ->{a1_path.relative_to(_ROOT)}")
    print(f"  MIXED-200_registry.json     ->{mixed_path.relative_to(_ROOT)}")
    print(f"  MIXED-200.csv               ->{csv_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
