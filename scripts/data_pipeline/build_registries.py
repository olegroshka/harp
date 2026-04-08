"""Build ActorRegistry JSON files for all SMIM experiment universes.

For each universe in data/smim/universes/manifest.json, assigns actors to
layers and types, resolves external IDs (CIK from EDGAR, FRED series, GDELT
themes), and writes a registry JSON consumed by the SMIM pipeline.

Also builds three combined registries for experiment configurations:
  - experiment_a1:    MIXED-200 equities + INST-US + INST-UK + Layer-0 shocks
  - experiment_phased: US-LC equities + INST-US + Layer-0 shocks
  - experiment_fast:  US-LC equities + INST-MINIMAL + Layer-0 shocks (quick runs)

Usage:
    uv run python scripts/smim/smim_build_registries.py

Outputs:
    data/smim/registries/{universe_id}_registry.json   — per-universe
    data/smim/registries/experiment_a1_registry.json
    data/smim/registries/experiment_phased_registry.json
    data/smim/registries/experiment_fast_registry.json
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

from harp.data.actor_registry import ActorRegistry  # noqa: E402
from harp.interfaces import Actor, ActorType, Layer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

UNIVERSE_DIR   = _ROOT / "data" / "universes"
MANIFEST_PATH  = UNIVERSE_DIR / "manifest.json"
PIT_DIR        = _ROOT / "data" / "pit_store"
REGISTRY_DIR   = _ROOT / "data" / "registries"

# ── Sector normalization ───────────────────────────────────────────────────────

_GICS_TO_SMIM: dict[str, str] = {
    "Energy":                    "energy",
    "Financials":                "financials",
    "Health Care":               "healthcare",
    "Information Technology":    "technology",
    "Industrials":               "industrials",
    "Communication Services":    "technology",
    "Consumer Discretionary":    "consumer",
    "Consumer Staples":          "consumer",
    "Materials":                 "industrials",
    "Real Estate":               "financials",
    "Utilities":                 "energy",
    "Communication":             "technology",
    "Cash and/or Derivatives":   "macro",
    "Unknown":                   "diversified",
}

def _smim_sector(raw: str) -> str:
    return _GICS_TO_SMIM.get(str(raw).strip(), "diversified")


# ── Layer / ActorType assignment ───────────────────────────────────────────────

def _assign_layer_and_type(
    ticker: str,
    sector: str,
    universe_id: str,
    rank: int,           # 0-based position in universe (sorted by market cap proxy)
    universe_size: int,
    sector_rank: int = 0,   # position within same sector (0 = largest)
) -> tuple[Layer, ActorType]:
    """Assign layer and actor_type based on universe tier, sector, and rank.

    Ensures ≥2 actor types per populated layer:
    - Layer 2: top-2 per sector → SECTOR_LEADER; Financials → BANK; rest → LARGE_FIRM
    - Layer 3: top 60% → SME; bottom 40% → RETAIL_INVESTOR
    """
    smim_sec = _smim_sector(sector)

    # Universe-tier → default layer
    if universe_id in ("US-SC",):
        base_layer = Layer.DOWNSTREAM
    elif universe_id in ("US-MC", "UK-MC"):
        # Top half → transmission, bottom half → downstream
        base_layer = Layer.TRANSMISSION if rank < universe_size // 2 else Layer.DOWNSTREAM
    else:
        # US-LC, UK-LC, sector sub-universes, MIXED-200
        base_layer = Layer.TRANSMISSION

    # ActorType assignment — ensures ≥2 types in each layer
    if base_layer == Layer.DOWNSTREAM:
        # Bottom 40% by rank → RETAIL_INVESTOR, rest → SME
        downstream_actors = (universe_size - universe_size // 2) if universe_id in ("US-MC", "UK-MC") else universe_size
        retail_cutoff = int(downstream_actors * 0.6)
        if universe_id in ("US-SC",):
            retail_cutoff = int(universe_size * 0.6)
        downstream_rank = rank - (universe_size // 2) if universe_id in ("US-MC", "UK-MC") else rank
        if downstream_rank >= retail_cutoff:
            return Layer.DOWNSTREAM, ActorType.RETAIL_INVESTOR
        return Layer.DOWNSTREAM, ActorType.SME
    else:
        # Top 2 per sector → SECTOR_LEADER
        if sector_rank < 2:
            return Layer.TRANSMISSION, ActorType.SECTOR_LEADER
        if smim_sec == "financials":
            return Layer.TRANSMISSION, ActorType.BANK
        return Layer.TRANSMISSION, ActorType.LARGE_FIRM


# ── Pre-defined institutional actor sets ──────────────────────────────────────

# Layer 0: global macro shocks (virtual actors backed by FRED series)
LAYER0_ACTORS: list[dict] = [
    {
        "actor_id":   "shock_brent_crude",
        "name":       "Brent Crude Spot Price",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "GLOBAL",
        "sector":     "energy",
        "external_ids": {"fred_series": "DCOILBRENTEU"},
    },
    {
        "actor_id":   "shock_wti_crude",
        "name":       "WTI Crude Oil Spot Price",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "GLOBAL",
        "sector":     "energy",
        "external_ids": {"fred_series": "DCOILWTICO"},
    },
    {
        "actor_id":   "shock_fed_funds",
        "name":       "Federal Funds Effective Rate",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {"fred_series": "FEDFUNDS"},
    },
    {
        "actor_id":   "shock_vix",
        "name":       "CBOE VIX Volatility Index",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {"fred_series": "VIXCLS"},
    },
    {
        "actor_id":   "shock_usd_index",
        "name":       "US Dollar Broad Trade-Weighted Index",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {"fred_series": "DTWEXBGS"},
    },
    {
        "actor_id":   "shock_yield_spread",
        "name":       "10Y-2Y Treasury Yield Spread",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {"fred_series": "T10Y2Y"},
    },
    {
        "actor_id":   "shock_hy_spread",
        "name":       "ICE BofA US High Yield Option-Adjusted Spread",
        "actor_type": ActorType.GLOBAL_SHOCK,
        "layer":      Layer.EXOGENOUS,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {"fred_series": "BAMLH0A0HYM2"},
    },
]

# Layer 1: US institutional actors
INST_US_ACTORS: list[dict] = [
    {
        "actor_id":   "inst_fed_us",
        "name":       "Federal Reserve System",
        "actor_type": ActorType.CENTRAL_BANK,
        "layer":      Layer.UPSTREAM,
        "geography":  "US",
        "sector":     "macro",
        "external_ids": {
            "fred_series": "FEDFUNDS",
            "gdelt_actor": "actor_FED",
        },
    },
    {
        "actor_id":   "inst_sec_us",
        "name":       "U.S. Securities and Exchange Commission",
        "actor_type": ActorType.REGULATOR,
        "layer":      Layer.UPSTREAM,
        "geography":  "US",
        "sector":     "financials",
        "external_ids": {
            "gdelt_actor": "actor_SEC",
        },
    },
    {
        "actor_id":   "inst_imf",
        "name":       "International Monetary Fund",
        "actor_type": ActorType.INTL_ORG,
        "layer":      Layer.UPSTREAM,
        "geography":  "GLOBAL",
        "sector":     "macro",
        "external_ids": {
            "gdelt_actor": "actor_IMF",
            "imf_indicator": "NGDP_RPCH",
        },
    },
]

# Layer 1: UK institutional actors
INST_UK_ACTORS: list[dict] = [
    {
        "actor_id":   "inst_boe_uk",
        "name":       "Bank of England",
        "actor_type": ActorType.CENTRAL_BANK,
        "layer":      Layer.UPSTREAM,
        "geography":  "UK",
        "sector":     "macro",
        "external_ids": {
            "gdelt_actor": "actor_BOE",
            "imf_indicator": "PCPIPCH:GBR",
        },
    },
    {
        "actor_id":   "inst_fca_uk",
        "name":       "Financial Conduct Authority",
        "actor_type": ActorType.REGULATOR,
        "layer":      Layer.UPSTREAM,
        "geography":  "UK",
        "sector":     "financials",
        "external_ids": {},
    },
]

# INST-MINIMAL: minimal institutional set for fast runs (2 central banks + 1 intl org)
INST_MINIMAL_ACTORS: list[dict] = [
    INST_US_ACTORS[0],   # fed_us (central_bank)
    INST_UK_ACTORS[0],   # boe_uk (central_bank)
    INST_US_ACTORS[2],   # inst_imf (intl_org) — second type for Layer 1
]


def _make_actor(d: dict) -> Actor:
    return Actor(
        actor_id=d["actor_id"],
        name=d["name"],
        actor_type=d["actor_type"],
        layer=d["layer"],
        geography=d["geography"],
        sector=d["sector"],
        external_ids=d.get("external_ids", {}),
    )


# ── CIK lookup table ───────────────────────────────────────────────────────────

def build_cik_map() -> dict[str, str]:
    """Build ticker → CIK mapping from EDGAR PIT shard."""
    path = PIT_DIR / "edgar.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["actor_id", "cik"])
    cik = df[["actor_id", "cik"]].drop_duplicates().dropna()
    return dict(zip(cik["actor_id"], cik["cik"]))


# ── FRED series presence ───────────────────────────────────────────────────────

def build_fred_series_set() -> set[str]:
    """Set of signal_ids actually present in the FRED PIT shard."""
    path = PIT_DIR / "fred.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["signal_id"])
    return set(df["signal_id"].unique())


# ── GDELT actor presence ───────────────────────────────────────────────────────

def build_gdelt_actor_set() -> set[str]:
    """Set of actor_ids in the GDELT PIT shard."""
    path = PIT_DIR / "gdelt.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["actor_id"])
    return set(df["actor_id"].unique())


# ── EDGAR actor presence ───────────────────────────────────────────────────────

def build_edgar_actor_set() -> set[str]:
    path = PIT_DIR / "edgar.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["actor_id"])
    return set(df["actor_id"].unique())


# ── Build equity actors for one universe ─────────────────────────────────────

def build_equity_actors(
    universe_id: str,
    manifest_entry: dict,
    cik_map: dict[str, str],
) -> list[Actor]:
    """Build Actor list for equity tickers in a universe."""
    # Load the CSV for fresh sector data
    csv_path = UNIVERSE_DIR / f"{universe_id}.csv"
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
        csv_df["sector"] = csv_df["sector"].fillna("Unknown").astype(str)
        ticker_sector = dict(zip(csv_df["ticker"], csv_df["sector"]))
        ticker_name = dict(zip(csv_df["ticker"], csv_df["name"]))
    else:
        ticker_sector = {}
        ticker_name = {}

    tickers = manifest_entry.get("tickers", [])
    dropped = set(manifest_entry.get("dropped", []))
    universe_size = len(tickers)

    # Track per-sector rank for SECTOR_LEADER assignment
    sector_counters: dict[str, int] = {}

    actors: list[Actor] = []
    for rank, ticker in enumerate(tickers):
        if ticker in dropped:
            continue
        raw_sector = ticker_sector.get(ticker, "Unknown")
        smim_sec = _smim_sector(raw_sector)
        name = ticker_name.get(ticker, ticker)
        geo = "UK" if ticker.endswith(".L") else "US"

        sector_rank = sector_counters.get(smim_sec, 0)
        sector_counters[smim_sec] = sector_rank + 1

        layer, actor_type = _assign_layer_and_type(
            ticker, raw_sector, universe_id, rank, universe_size,
            sector_rank=sector_rank,
        )

        ext_ids: dict[str, str] = {}
        if cik := cik_map.get(ticker):
            ext_ids["cik"] = cik

        actors.append(Actor(
            actor_id=ticker,
            name=name,
            actor_type=actor_type,
            layer=layer,
            geography=geo,
            sector=smim_sec,
            external_ids=ext_ids,
        ))

    return actors


# ── Build one registry ─────────────────────────────────────────────────────────

def build_universe_registry(
    universe_id: str,
    manifest_entry: dict,
    cik_map: dict[str, str],
    include_layer0: bool = False,
    include_layer1: list[dict] | None = None,
) -> ActorRegistry:
    """Build registry for a single equity universe (optional +institutional layers)."""
    actors: list[Actor] = []

    if include_layer0:
        actors.extend(_make_actor(d) for d in LAYER0_ACTORS)

    if include_layer1:
        actors.extend(_make_actor(d) for d in include_layer1)

    actors.extend(build_equity_actors(universe_id, manifest_entry, cik_map))
    return ActorRegistry(actors=actors)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_registry(
    registry: ActorRegistry,
    universe_id: str,
    expected_min_equity: int,
    cik_map: dict[str, str],
    fred_series: set[str],
    gdelt_actors: set[str],
    edgar_actors: set[str],
) -> dict[str, Any]:
    """Validate a registry against data presence in PIT store."""
    issues: list[str] = []

    # Check N ≥ expected
    equity_count = len([a for a in registry.actors
                        if a.layer in (Layer.TRANSMISSION, Layer.DOWNSTREAM)])
    if equity_count < expected_min_equity:
        issues.append(
            f"Equity count {equity_count} < expected {expected_min_equity}"
        )

    # Check ≥2 actor types per populated layer.
    # Layer.EXOGENOUS is exempt: taxonomy defines only GLOBAL_SHOCK for Layer 0.
    for layer in Layer:
        if layer == Layer.EXOGENOUS:
            continue
        layer_actors = registry.actors_in_layer(layer)
        if not layer_actors:
            continue
        types_in_layer = {a.actor_type for a in layer_actors}
        if len(types_in_layer) < 2:
            issues.append(
                f"Layer {layer.name} has only {len(types_in_layer)} actor type(s): "
                f"{[t.value for t in types_in_layer]}"
            )

    # External ID resolution
    unresolved_cik = 0
    unresolved_fred = 0
    unresolved_gdelt = 0
    for actor in registry.actors:
        if "cik" in actor.external_ids:
            # Check ticker in EDGAR actor set
            if actor.actor_id not in edgar_actors:
                unresolved_cik += 1
        if "fred_series" in actor.external_ids:
            series = actor.external_ids["fred_series"]
            if series not in fred_series:
                unresolved_fred += 1
        if "gdelt_actor" in actor.external_ids:
            ga = actor.external_ids["gdelt_actor"]
            if ga not in gdelt_actors:
                unresolved_gdelt += 1

    total_with_cik = sum(1 for a in registry.actors if "cik" in a.external_ids)
    total_with_fred = sum(1 for a in registry.actors if "fred_series" in a.external_ids)
    total_with_gdelt = sum(1 for a in registry.actors if "gdelt_actor" in a.external_ids)

    return {
        "universe_id": universe_id,
        "N": registry.N,
        "equity_count": equity_count,
        "issues": issues,
        "passed": len(issues) == 0,
        "cik_coverage": f"{total_with_cik - unresolved_cik}/{total_with_cik}",
        "fred_coverage": f"{total_with_fred - unresolved_fred}/{total_with_fred}",
        "gdelt_coverage": f"{total_with_gdelt - unresolved_gdelt}/{total_with_gdelt}",
        "layer_breakdown": {
            layer.name: len(registry.actors_in_layer(layer))
            for layer in Layer
        },
        "actor_types": sorted({a.actor_type.value for a in registry.actors}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("SMIM Registry Builder")
    log.info("=" * 60)

    # Load manifest
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    log.info("Loaded manifest: %d universes", len(manifest))

    # Build lookup tables from PIT store
    log.info("Building external ID maps from PIT store …")
    cik_map      = build_cik_map()
    fred_series  = build_fred_series_set()
    gdelt_actors = build_gdelt_actor_set()
    edgar_actors = build_edgar_actor_set()
    log.info("  CIK map: %d tickers | FRED: %d series | GDELT: %d actors | EDGAR: %d actors",
             len(cik_map), len(fred_series), len(gdelt_actors), len(edgar_actors))

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    all_validation: list[dict] = []

    # ── Per-universe registries (equity only, no institutional) ───────────────
    log.info("\nBuilding per-universe equity registries …")
    per_universe_registries: dict[str, ActorRegistry] = {}

    for uni_id, entry in sorted(manifest.items()):
        dropped = set(entry.get("dropped", []))
        expected_equity = entry["tickers_total"] - len(dropped)

        registry = build_universe_registry(
            uni_id, entry, cik_map,
            include_layer0=False,
            include_layer1=None,
        )
        per_universe_registries[uni_id] = registry

        out_path = REGISTRY_DIR / f"{uni_id}_registry.json"
        registry.to_json(
            out_path,
            universe_id=uni_id,
            description=f"Equity-only registry for {uni_id}",
            data_regime=entry.get("data_regime", "unknown"),
            date_range=entry.get("date_range"),
        )

        val = validate_registry(
            registry, uni_id, expected_equity,
            cik_map, fred_series, gdelt_actors, edgar_actors,
        )
        all_validation.append(val)
        status = "OK" if val["passed"] else f"WARN({len(val['issues'])} issues)"
        log.info("  %-20s  N=%3d  layers=%s  [%s]",
                 uni_id, registry.N,
                 {k: v for k, v in val["layer_breakdown"].items() if v > 0},
                 status)
        if not val["passed"]:
            for iss in val["issues"]:
                log.warning("    ! %s", iss)

    # ── Combined registries ───────────────────────────────────────────────────
    log.info("\nBuilding combined experiment registries …")

    def _combined(
        name: str,
        description: str,
        equity_universes: list[str],
        layer1_actors: list[dict],
    ) -> ActorRegistry:
        actors: list[Actor] = []
        # Layer 0
        actors.extend(_make_actor(d) for d in LAYER0_ACTORS)
        # Layer 1
        actors.extend(_make_actor(d) for d in layer1_actors)
        # Equity layers (deduplicate by actor_id)
        seen: set[str] = {a.actor_id for a in actors}
        for uni_id in equity_universes:
            if uni_id not in manifest:
                log.warning("  Combined %s: universe %s not in manifest, skipping", name, uni_id)
                continue
            entry = manifest[uni_id]
            eq_actors = build_equity_actors(uni_id, entry, cik_map)
            for a in eq_actors:
                if a.actor_id not in seen:
                    actors.append(a)
                    seen.add(a.actor_id)
        reg = ActorRegistry(actors=actors)
        out = REGISTRY_DIR / f"{name}_registry.json"
        reg.to_json(
            out,
            universe_id=name,
            description=description,
            equity_universes=equity_universes,
        )
        log.info("  %-30s  N=%d", name, reg.N)
        return reg

    exp_a1 = _combined(
        "experiment_a1",
        "MIXED-200 equities + INST-US + INST-UK + Layer-0 shocks (Experiment A1)",
        ["MIXED-200"],
        INST_US_ACTORS + INST_UK_ACTORS,
    )

    exp_phased = _combined(
        "experiment_phased",
        "US-LC equities + INST-US + Layer-0 shocks (Phase D experiments)",
        ["US-LC"],
        INST_US_ACTORS,
    )

    exp_fast = _combined(
        "experiment_fast",
        "US-LC equities + INST-MINIMAL + Layer-0 shocks (fast experiment runs)",
        ["US-LC"],
        INST_MINIMAL_ACTORS,
    )

    # Validate combined registries
    for name, reg in [
        ("experiment_a1", exp_a1),
        ("experiment_phased", exp_phased),
        ("experiment_fast", exp_fast),
    ]:
        val = validate_registry(
            reg, name, 0,
            cik_map, fred_series, gdelt_actors, edgar_actors,
        )
        all_validation.append(val)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REGISTRY BUILD SUMMARY")
    print("=" * 70)
    print(f"{'Universe':<26} {'N':>5} {'L0':>4} {'L1':>4} {'L2':>5} {'L3':>5}  "
          f"{'CIK':>9}  {'FRED':>6}  {'GDELT':>7}  Status")
    print("-" * 70)

    for val in all_validation:
        lb = val["layer_breakdown"]
        l0 = lb.get("EXOGENOUS", 0)
        l1 = lb.get("UPSTREAM", 0)
        l2 = lb.get("TRANSMISSION", 0)
        l3 = lb.get("DOWNSTREAM", 0)
        status = "PASS" if val["passed"] else f"WARN"
        print(f"  {val['universe_id']:<24} {val['N']:>5} {l0:>4} {l1:>4} {l2:>5} {l3:>5}  "
              f"{val['cik_coverage']:>9}  {val['fred_coverage']:>6}  "
              f"{val['gdelt_coverage']:>7}  [{status}]")
        if not val["passed"]:
            for iss in val["issues"]:
                print(f"    ! {iss}")

    n_pass = sum(1 for v in all_validation if v["passed"])
    n_total = len(all_validation)
    print("=" * 70)
    print(f"  {n_pass}/{n_total} registries passed validation")
    print(f"  Registry files written to: {REGISTRY_DIR.relative_to(_ROOT)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
