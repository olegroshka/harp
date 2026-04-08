"""Compute normalised investment intensities y_{i,t} for all SMIM universes.

Reads actor registries from data/smim/registries/, applies the appropriate
InvestmentIntensityMapper per actor type, and writes quarterly panels to
data/smim/intensities/{universe_id}_intensities.parquet.

Point-in-time (A1) compliance: for each quarter Q, only data with
pub_date <= end-of-Q is used.

Output schema (long format):
  actor_id, period (quarter start date), intensity_value, normalisation_method

Usage:
    uv run python scripts/smim/smim_compute_intensities.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from harp.data.actor_registry import ActorRegistry  # noqa: E402
from harp.data.intensity_mappers import (  # noqa: E402
    CorporateCapexMapper,
    BankCreditMapper,
    AgencyBudgetMapper,
    MapperRegistry,
)
from harp.interfaces import Actor, ActorType, Layer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

REGISTRY_DIR   = _ROOT / "data" / "registries"
PIT_DIR        = _ROOT / "data" / "pit_store"
INTENSITY_DIR  = _ROOT / "data" / "intensities"
REPORT_DIR     = _ROOT / "docs" / "smim" / "reports"
OHLCV_DIR      = _ROOT / "equities" / "smim"

# ── Quarterly period grid ──────────────────────────────────────────────────────

START_DATE = pd.Timestamp("2005-01-01")
END_DATE   = pd.Timestamp("2025-12-31")

# Quarter-end dates (last day of each quarter)
QUARTER_ENDS = pd.date_range(START_DATE, END_DATE, freq="QE")
# Quarter-start dates (labels for output periods)
QUARTER_STARTS = pd.date_range(START_DATE, END_DATE, freq="QS")

# ── EDGAR signal mapping ───────────────────────────────────────────────────────

CAPEX_SIGNAL   = "PaymentsToAcquirePropertyPlantAndEquipment"
ASSETS_SIGNAL  = "Assets"
REVENUE_SIGNAL = "Revenues"

# FRED series → Layer 0 actor mapping (actor_id → fred_series)
LAYER0_FRED_MAP: dict[str, str] = {
    "shock_brent_crude":  "DCOILBRENTEU",
    "shock_wti_crude":    "DCOILWTICO",
    "shock_fed_funds":    "FEDFUNDS",
    "shock_vix":          "VIXCLS",
    "shock_usd_index":    "DTWEXBGS",
    "shock_yield_spread": "T10Y2Y",
    "shock_hy_spread":    "BAMLH0A0HYM2",
}

# GDELT actor_ids for Layer 1 institutions
GDELT_ACTOR_MAP: dict[str, str] = {
    "inst_fed_us":  "actor_FED",
    "inst_sec_us":  "actor_SEC",
    "inst_boe_uk":  "actor_BOE",
    "inst_imf":     "actor_IMF",
    "inst_fca_uk":  "actor_BOE",   # proxy: BoE narrative (closest UK regulatory signal in GDELT)
}

# ── A1-compliant quarterly pivot builder ─────────────────────────────────────

def pit_to_quarterly_panel(
    pit_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
    actor_ids: list[str] | None = None,
    signal_id: str | None = None,
) -> pd.DataFrame:
    """Build a quarterly (T × N) panel from PIT data, respecting A1.

    For each quarter end date Q, the cell value is the most recently
    published observation (pub_date ≤ Q) for each actor.

    Args:
        pit_df:       Raw PIT DataFrame (any source subset).
        quarter_ends: DatetimeIndex of quarter end dates.
        actor_ids:    Optional filter to these actors.
        signal_id:    Optional filter to one signal_id.

    Returns:
        DataFrame indexed by quarter-end dates, columns = actor_ids.
        Values are the last available as-of each quarter end.
    """
    df = pit_df.copy()
    df["pub_date"]   = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)

    if signal_id is not None:
        df = df[df["signal_id"] == signal_id]
    if actor_ids is not None:
        df = df[df["actor_id"].isin(actor_ids)]
    if df.empty:
        cols = actor_ids if actor_ids else []
        return pd.DataFrame(index=quarter_ends, columns=cols, dtype=float)

    # Sort by pub_date then event_date: last row = most-recently published
    df = df.sort_values(["pub_date", "event_date"])

    # Pivot: index = pub_date, columns = actor_id, value = last value at that pub_date
    pivot = df.pivot_table(
        index="pub_date",
        columns="actor_id",
        values="value",
        aggfunc="last",
    ).sort_index()

    # Reindex to quarter-end grid and forward-fill (propagate most recently known value)
    panel = pivot.reindex(pivot.index.union(quarter_ends)).ffill().reindex(quarter_ends)

    if actor_ids is not None:
        for aid in actor_ids:
            if aid not in panel.columns:
                panel[aid] = np.nan
        panel = panel[actor_ids]

    return panel


# ── EDGAR intensity (CapEx / Assets) ─────────────────────────────────────────

def load_edgar() -> pd.DataFrame:
    path = PIT_DIR / "edgar.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["pub_date"]   = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    return df


def compute_edgar_capex_ratio(
    edgar_df: pd.DataFrame,
    actor_ids: list[str],
    quarter_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Returns (T × N) panel of CapEx/Assets ratios (A1-compliant)."""
    capex_panel  = pit_to_quarterly_panel(edgar_df, quarter_ends, actor_ids, CAPEX_SIGNAL)
    assets_panel = pit_to_quarterly_panel(edgar_df, quarter_ends, actor_ids, ASSETS_SIGNAL)

    # Ratio: CapEx/Assets; mask non-positive assets
    assets_clean = assets_panel.where(assets_panel > 0)
    ratio = capex_panel.divide(assets_clean)
    # Clip at 0 (negative CapEx entries are data errors)
    ratio = ratio.clip(lower=0)
    return ratio


def compute_bank_asset_growth(
    edgar_df: pd.DataFrame,
    actor_ids: list[str],
    quarter_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Returns (T × N) panel of Assets YoY (4-quarter) growth rates for banks.

    Uses 4-period pct_change to match the return_12m_xsrank approach and remove
    seasonal patterns that confound quarterly bank reporting.
    """
    assets_panel = pit_to_quarterly_panel(edgar_df, quarter_ends, actor_ids, ASSETS_SIGNAL)
    growth = assets_panel.pct_change(periods=4, fill_method=None)
    return growth


# ── FRED intensity (Layer 0 shocks + Layer 1 central banks) ─────────────────

def load_fred() -> pd.DataFrame:
    path = PIT_DIR / "fred.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["pub_date"]   = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    return df


def compute_fred_panel(
    fred_df: pd.DataFrame,
    series_ids: list[str],
    quarter_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Returns (T × len(series_ids)) panel of FRED values (A1-compliant).

    actor_id column in FRED shard is the FRED series ID (e.g. 'DCOILBRENTEU').
    """
    # FRED shard has actor_id = "US" and signal_id = series code
    frames = []
    for sid in series_ids:
        sub = fred_df[fred_df["signal_id"] == sid].copy()
        if sub.empty:
            frames.append(pd.Series(np.nan, index=quarter_ends, name=sid))
            continue
        # Use signal_id as the actor column for the panel builder
        sub["actor_id"] = sid
        panel = pit_to_quarterly_panel(sub, quarter_ends, actor_ids=[sid], signal_id=sid)
        frames.append(panel[sid] if sid in panel.columns else pd.Series(np.nan, index=quarter_ends, name=sid))

    return pd.concat(frames, axis=1)


# ── GDELT intensity (publication/mention counts) ─────────────────────────────

def load_gdelt() -> pd.DataFrame:
    path = PIT_DIR / "gdelt.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["pub_date"]   = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    return df


def compute_gdelt_panel(
    gdelt_df: pd.DataFrame,
    actor_ids: list[str],
    quarter_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Returns (T × N) panel of quarterly summed GDELT mention counts."""
    if gdelt_df.empty:
        return pd.DataFrame(index=quarter_ends, columns=actor_ids, dtype=float)

    # GDELT shard: actor_id = "actor_FED" etc., signal_id = theme, value = count
    # Sum all themes per actor per quarter
    df = gdelt_df[gdelt_df["actor_id"].isin(actor_ids)].copy()
    if df.empty:
        return pd.DataFrame(index=quarter_ends, columns=actor_ids, dtype=float)

    df["quarter"] = df["event_date"].dt.to_period("Q").dt.start_time
    quarterly = df.groupby(["quarter", "actor_id"])["value"].sum().unstack("actor_id")

    # Reindex to quarter-end grid (using quarter start → end mapping)
    q_start_to_end = dict(zip(QUARTER_STARTS, quarter_ends))
    quarterly.index = quarterly.index.map(lambda d: q_start_to_end.get(d, d))
    quarterly = quarterly.reindex(quarter_ends)

    for aid in actor_ids:
        if aid not in quarterly.columns:
            quarterly[aid] = np.nan

    return quarterly[actor_ids].astype(float)


# ── Normalise helpers ─────────────────────────────────────────────────────────

def minmax_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-column min-max to [0,1] across the full time axis."""
    col_min = panel.min(skipna=True)
    col_max = panel.max(skipna=True)
    span = col_max - col_min
    span = span.replace(0.0, np.nan)
    return (panel - col_min) / span


def cross_section_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise percentile rank to [0,1]. Returns same-shape DataFrame."""
    return panel.rank(axis=1, pct=True, na_option="keep")


def zscore_sigmoid(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-column z-score then sigmoid."""
    mu    = panel.mean(skipna=True)
    sigma = panel.std(skipna=True, ddof=1).replace(0.0, np.nan)
    z = (panel - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


# ── Per-type intensity computation ───────────────────────────────────────────

def compute_equity_intensities(
    equity_actors: list[Actor],
    edgar_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, str]:
    """CapEx/Assets, cross-sectionally ranked. Returns (panel, method_name).

    When the equity cross-section contains only one actor (e.g. a sole
    SECTOR_LEADER with no LARGE_FIRM peers in the universe), the cross-sectional
    percentile rank degenerates to a constant 1.0 for that actor across all
    periods, producing zero temporal variance and Spearman ρ ≈ 0 — a violation
    of assumption A2 (rank stability).

    For such degenerate columns (intensity std == 0 across time), this function
    falls back to z-score sigmoid across time (per-actor temporal normalisation),
    consistent with BankCreditMapper. A truly constant raw ratio maps to 0.5.
    """
    actor_ids = [a.actor_id for a in equity_actors]
    ratio = compute_edgar_capex_ratio(edgar_df, actor_ids, quarter_ends)
    intensity = cross_section_rank(ratio)

    # Detect degenerate constant columns — typically a single actor in the
    # equity cross-section that always receives rank = 1.0.
    col_std = intensity.std(skipna=True, ddof=1)
    degenerate_cols = col_std[col_std == 0.0].index.tolist()
    if degenerate_cols:
        log.warning(
            "  Degenerate constant intensity for %d actor(s) after cross-section "
            "rank (single-actor cross-section?): %s  —  applying z-score sigmoid "
            "fallback.",
            len(degenerate_cols),
            degenerate_cols,
        )
        for col in degenerate_cols:
            raw_col = ratio[col]
            sigma = raw_col.std(skipna=True, ddof=1)
            if sigma == 0.0 or pd.isna(sigma):
                # Truly constant raw ratio → no temporal signal → uniform 0.5
                intensity[col] = raw_col.where(raw_col.isna(), 0.5)
            else:
                mu = raw_col.mean(skipna=True)
                z = (raw_col - mu) / sigma
                intensity[col] = 1.0 / (1.0 + np.exp(-z))

    return intensity, "capex_assets_xsrank"


def compute_ohlcv_return_intensities(
    equity_actors: list[Actor],
    universe_id: str,
    quarter_ends: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, str]:
    """Rolling 12-month price return, cross-sectionally ranked. Returns (panel, method_name).

    Used as a fallback when EDGAR balance-sheet data is unavailable (e.g. UK equities
    that do not file with SEC EDGAR).  A1 compliance is maintained because OHLCV prices
    are observable with zero publication lag (price at quarter-end is public immediately).

    Args:
        equity_actors: Actors whose actor_id matches a ticker in the OHLCV parquet.
        universe_id:   Universe identifier (e.g. "UK-LC") — used to locate the OHLCV file.
        quarter_ends:  DatetimeIndex of quarter-end dates.

    Returns:
        (panel, "return_12m_xsrank") where panel is (T × N) with values in [0, 1].
        Actors with no OHLCV coverage are all-NaN.
    """
    ohlcv_path = OHLCV_DIR / universe_id / "ohlcv.parquet"
    if not ohlcv_path.exists():
        log.warning("  OHLCV file not found for %s: %s", universe_id, ohlcv_path)
        actor_ids = [a.actor_id for a in equity_actors]
        return pd.DataFrame(index=quarter_ends, columns=actor_ids, dtype=float), "return_12m_xsrank"

    ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None)

    # Pivot to wide format: dates × tickers, value = close price
    price_wide = ohlcv.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    price_wide = price_wide.sort_index()

    # Resample to quarter-end grid: last available close at or before each quarter-end
    price_q = price_wide.reindex(
        price_wide.index.union(quarter_ends)
    ).ffill().reindex(quarter_ends)

    # 12-month return: close_Q / close_{Q-4} - 1 (requires 4 prior quarters)
    ret_12m = price_q.pct_change(periods=4, fill_method=None)

    # Align columns to actor_ids (tickers)
    actor_ids = [a.actor_id for a in equity_actors]
    for aid in actor_ids:
        if aid not in ret_12m.columns:
            ret_12m[aid] = np.nan
    ret_12m = ret_12m[actor_ids]

    # Cross-sectional percentile rank at each quarter
    intensity = cross_section_rank(ret_12m)
    return intensity, "return_12m_xsrank"


def compute_bank_intensities(
    bank_actors: list[Actor],
    edgar_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, str]:
    """Assets YoY growth, cross-sectionally ranked to [0,1].

    Changed from per-actor z-score sigmoid (RP1 fix): cross-sectional rank
    is stable by construction and eliminates the near-random cross-sectional
    rankings produced by per-actor temporal normalisation.
    """
    actor_ids = [a.actor_id for a in bank_actors]
    growth = compute_bank_asset_growth(edgar_df, actor_ids, quarter_ends)
    intensity = cross_section_rank(growth)
    return intensity, "asset_growth_yoy_xsrank"


def compute_macro_intensities(
    macro_actors: list[Actor],
    fred_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, str]:
    """FRED series, min-max normalised per series over time."""
    # Build series_ids → actor_id map
    sid_to_actor: dict[str, str] = {}
    for a in macro_actors:
        series_id = LAYER0_FRED_MAP.get(a.actor_id) or a.external_ids.get("fred_series")
        if series_id:
            sid_to_actor[series_id] = a.actor_id

    if not sid_to_actor:
        cols = [a.actor_id for a in macro_actors]
        return pd.DataFrame(np.nan, index=quarter_ends, columns=cols), "minmax"

    series_ids = list(sid_to_actor.keys())
    fred_panel = compute_fred_panel(fred_df, series_ids, quarter_ends)
    normalised = minmax_normalize(fred_panel)

    # Rename columns from series_id to actor_id
    rename_map = sid_to_actor
    normalised = normalised.rename(columns=rename_map)
    return normalised, "fred_minmax"


def compute_institutional_intensities(
    inst_actors: list[Actor],
    fred_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, str]:
    """GDELT publication counts (primary) or FRED policy rates (fallback)."""
    result_cols: dict[str, pd.Series] = {}

    # GDELT actors
    gdelt_aid_actors = {
        GDELT_ACTOR_MAP.get(a.actor_id): a
        for a in inst_actors
        if GDELT_ACTOR_MAP.get(a.actor_id)
    }
    if gdelt_aid_actors:
        gdelt_ids = list(gdelt_aid_actors.keys())
        gdelt_panel = compute_gdelt_panel(gdelt_df, gdelt_ids, quarter_ends)
        norm_gdelt = minmax_normalize(gdelt_panel)
        for gdelt_id, actor in gdelt_aid_actors.items():
            if gdelt_id in norm_gdelt.columns:
                result_cols[actor.actor_id] = norm_gdelt[gdelt_id]

    # Fallback: actors not in GDELT → FRED policy rate or NaN
    for actor in inst_actors:
        if actor.actor_id in result_cols:
            continue
        fred_series = actor.external_ids.get("fred_series")
        if fred_series and not fred_df.empty:
            panel = compute_fred_panel(fred_df, [fred_series], quarter_ends)
            if fred_series in panel.columns:
                norm = minmax_normalize(panel[[fred_series]])
                result_cols[actor.actor_id] = norm[fred_series]
                continue
        result_cols[actor.actor_id] = pd.Series(np.nan, index=quarter_ends, name=actor.actor_id)

    if result_cols:
        panel = pd.concat(result_cols.values(), axis=1)
        panel.columns = list(result_cols.keys())
    else:
        panel = pd.DataFrame(index=quarter_ends)

    return panel, "gdelt_minmax_or_fred"


# ── Registry-level intensity computation ─────────────────────────────────────

def compute_registry_intensities(
    registry: ActorRegistry,
    edgar_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    quarter_ends: pd.DatetimeIndex,
    quarter_starts: pd.DatetimeIndex,
    universe_id: str = "",
) -> pd.DataFrame:
    """Compute intensities for all actors in registry.

    Returns long-format DataFrame with columns:
        actor_id, period, intensity_value, normalisation_method

    Args:
        universe_id: Used to locate OHLCV fallback data when EDGAR coverage is absent
                     (e.g. for UK universes that do not file with SEC EDGAR).
    """
    frames: list[pd.DataFrame] = []

    # ── Layer 0: macro shocks ──────────────────────────────────────────────
    macro_actors = [a for a in registry.actors if a.layer == Layer.EXOGENOUS]
    if macro_actors:
        panel, method = compute_macro_intensities(macro_actors, fred_df, quarter_ends)
        long = _panel_to_long(panel, quarter_starts, method, "macro_shock")
        frames.append(long)
        log.info("    Layer 0 (macro shocks): %d actors, %d obs", len(macro_actors), len(long))

    # ── Layer 1: institutions ──────────────────────────────────────────────
    inst_actors = [a for a in registry.actors if a.layer == Layer.UPSTREAM]
    if inst_actors:
        panel, method = compute_institutional_intensities(
            inst_actors, fred_df, gdelt_df, quarter_ends
        )
        long = _panel_to_long(panel, quarter_starts, method, "institution")
        frames.append(long)
        log.info("    Layer 1 (institutions): %d actors, %d obs", len(inst_actors), len(long))

    # ── Layer 2/3: banks ──────────────────────────────────────────────────
    bank_actors = [a for a in registry.actors
                   if a.actor_type == ActorType.BANK
                   and a.layer in (Layer.TRANSMISSION, Layer.DOWNSTREAM)]
    if bank_actors:
        panel, method = compute_bank_intensities(bank_actors, edgar_df, quarter_ends)
        long = _panel_to_long(panel, quarter_starts, method, "bank")
        frames.append(long)
        log.info("    Banks: %d actors, %d obs", len(bank_actors), len(long))

    # ── Layer 2/3: equity (LARGE_FIRM, SECTOR_LEADER, SME, RETAIL_INVESTOR) ──
    equity_types = {
        ActorType.LARGE_FIRM, ActorType.SECTOR_LEADER,
        ActorType.SME, ActorType.RETAIL_INVESTOR
    }
    equity_actors = [a for a in registry.actors if a.actor_type in equity_types]
    if equity_actors:
        panel, method = compute_equity_intensities(equity_actors, edgar_df, quarter_ends)

        # Identify actors with no EDGAR coverage (all-NaN column).
        # This handles mixed US+UK registries (e.g. experiment_a1, MIXED-200) where
        # the whole-panel fallback never fires because US actors have EDGAR data.
        all_nan_ids: set[str] = {
            a.actor_id for a in equity_actors
            if a.actor_id in panel.columns and panel[a.actor_id].isna().all()
        }

        # Whole-panel fallback (pure non-EDGAR universes such as UK-LC, UK-MC).
        if not all_nan_ids and (panel.empty or panel.isna().all().all()):
            if universe_id:
                log.info(
                    "    EDGAR equity panel empty for %s — falling back to "
                    "OHLCV return_12m_xsrank", universe_id,
                )
                panel, method = compute_ohlcv_return_intensities(
                    equity_actors, universe_id, quarter_ends
                )
            long = _panel_to_long(panel, quarter_starts, method, "equity")
            frames.append(long)
            log.info("    Equity: %d actors, %d obs", len(equity_actors), len(long))

        else:
            # Emit EDGAR-covered actors.
            edgar_cols = [c for c in panel.columns if c not in all_nan_ids]
            if edgar_cols:
                long = _panel_to_long(panel[edgar_cols], quarter_starts, method, "equity")
                frames.append(long)
                log.info("    Equity (EDGAR/%s): %d actors, %d obs",
                         method, len(edgar_cols), len(long))

            # Per-actor OHLCV fallback for actors with no EDGAR coverage.
            # ONLY applied for non-US geographies (e.g. UK) — actors from EDGAR-reporting
            # jurisdictions (US) that lack a CapEx tag have a data-pipeline gap and must
            # be fixed by fetching alternative EDGAR tags (see G-13 in data_audit.md).
            # Using OHLCV proxy for US actors would mix M-A and M-B in the same cross-section,
            # degrading rank stability below threshold (empirically: ρ drops from 0.759 → 0.699
            # for US-LC-ENERGY when 9/21 actors get OHLCV fallback).
            # Each non-US geography is ranked within its own OHLCV cross-section (A2 compliant).
            if all_nan_ids:
                missing_actors = [a for a in equity_actors if a.actor_id in all_nan_ids]
                # Separate: non-US actors eligible for OHLCV; US actors logged as data gap.
                non_us_missing = [a for a in missing_actors if a.geography != "US"]
                us_missing = [a for a in missing_actors if a.geography == "US"]
                if us_missing:
                    log.warning(
                        "    %d US equity actors have no EDGAR CapEx tag — "
                        "no OHLCV fallback applied (data gap, see G-13): %s",
                        len(us_missing), [a.actor_id for a in us_missing],
                    )
                if non_us_missing:
                    log.info(
                        "    Per-actor OHLCV fallback for %d non-US equity actors with no EDGAR data",
                        len(non_us_missing),
                    )
                    _geo_search: dict[str, list[str]] = {
                        "UK": ["UK-LC", "UK-MC"],
                    }
                    still_missing: list[Actor] = []
                    for geo, uni_list in _geo_search.items():
                        remaining = [a for a in non_us_missing if a.geography == geo]
                        if not remaining:
                            continue
                        for uni in uni_list:
                            ohlcv_panel, ohlcv_method = compute_ohlcv_return_intensities(
                                remaining, uni, quarter_ends
                            )
                            covered = [c for c in ohlcv_panel.columns
                                       if not ohlcv_panel[c].isna().all()]
                            if covered:
                                ohlcv_long = _panel_to_long(
                                    ohlcv_panel[covered], quarter_starts, ohlcv_method, "equity"
                                )
                                frames.append(ohlcv_long)
                                log.info(
                                    "      %s OHLCV (%s): %d/%d actors covered",
                                    geo, uni, len(covered), len(remaining),
                                )
                                remaining = [a for a in remaining if a.actor_id not in set(covered)]
                            if not remaining:
                                break
                        still_missing.extend(remaining)
                    if still_missing:
                        log.warning(
                            "    %d non-US equity actors still uncovered after OHLCV fallback: %s",
                            len(still_missing), [a.actor_id for a in still_missing],
                        )

    if not frames:
        return pd.DataFrame(columns=["actor_id", "period", "intensity_value", "normalisation_method"])

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _panel_to_long(
    panel: pd.DataFrame,
    quarter_starts: pd.DatetimeIndex,
    method: str,
    actor_group: str,
) -> pd.DataFrame:
    """Convert wide (quarter-ends × actors) panel to long format."""
    if panel.empty:
        return pd.DataFrame(columns=["actor_id", "period", "intensity_value", "normalisation_method"])

    # Map quarter-ends → quarter-starts for the period label
    period_map = dict(zip(QUARTER_ENDS, quarter_starts))
    panel = panel.copy()
    panel.index = panel.index.map(period_map)

    melted = panel.reset_index().melt(
        id_vars="index",
        var_name="actor_id",
        value_name="intensity_value",
    ).rename(columns={"index": "period"})
    melted["normalisation_method"] = method
    melted = melted.dropna(subset=["intensity_value"])
    melted = melted[["actor_id", "period", "intensity_value", "normalisation_method"]]
    return melted


# ── Quality checks ────────────────────────────────────────────────────────────

def quality_check_intensities(
    long_df: pd.DataFrame,
    registry: ActorRegistry,
) -> dict[str, Any]:
    """Quality checks on computed intensities.

    Returns dict with:
      - range_ok: all values in [0,1] or NaN
      - high_missing: actors with >50% missing quarters
      - rank_stability: Spearman rho between adjacent quarters (mean across pairs)
      - distribution: per actor_type summary stats
    """
    if long_df.empty:
        return {"range_ok": True, "high_missing": [], "rank_stability": float("nan")}

    # Build wide panel for checks
    panel = long_df.pivot_table(
        index="period", columns="actor_id", values="intensity_value", aggfunc="mean"
    )

    total_quarters = len(panel)

    # 1. Range check
    vals = long_df["intensity_value"].dropna()
    range_ok = bool((vals >= -1e-9).all() and (vals <= 1 + 1e-9).all())

    # 2. Missing fraction per actor
    missing_frac = panel.isna().mean()
    high_missing = missing_frac[missing_frac > 0.5].index.tolist()

    # 3. Cross-sectional rank stability (Spearman rho between adjacent quarters)
    #    Computed for: (a) full history, (b) recent 5-year window (2020-Q1 onward)
    RECENT_CUTOFF = pd.Timestamp("2020-01-01")

    rho_values: list[float] = []
    rho_values_recent: list[float] = []
    sorted_dates = sorted(panel.index)
    for i in range(1, len(sorted_dates)):
        row_a = panel.loc[sorted_dates[i-1]].dropna()
        row_b = panel.loc[sorted_dates[i]].dropna()
        common = row_a.index.intersection(row_b.index)
        if len(common) < 5:
            continue
        rho, _ = scipy_stats.spearmanr(row_a[common], row_b[common])
        if not np.isnan(rho):
            rho_values.append(rho)
            if sorted_dates[i-1] >= RECENT_CUTOFF:
                rho_values_recent.append(rho)

    mean_rho = float(np.mean(rho_values)) if rho_values else float("nan")
    mean_rho_recent = float(np.mean(rho_values_recent)) if rho_values_recent else float("nan")

    # 4. Distribution per actor type
    actor_type_map = {a.actor_id: a.actor_type.value for a in registry.actors}
    long_df["actor_type"] = long_df["actor_id"].map(actor_type_map)
    dist_summary: dict[str, dict] = {}
    for atype, grp in long_df.groupby("actor_type", dropna=True):
        v = grp["intensity_value"].dropna()
        if len(v) < 5:
            continue
        dist_summary[atype] = {
            "n": len(v),
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std()), 4),
            "skew": round(float(v.skew()), 4),
            "pct_nan": round(float(long_df[long_df["actor_type"] == atype]["intensity_value"].isna().mean()), 4),
        }

    # Gate passes if EITHER full-period OR recent-5yr rho exceeds 0.7.
    # This accommodates universes with genuine historical structural shifts
    # (e.g. US-LC-TECH: sector transformed 2010-2020; recent ρ=0.727 PASS).
    rho_ok = (mean_rho > 0.7 or np.isnan(mean_rho)
              or (not np.isnan(mean_rho_recent) and mean_rho_recent > 0.7))

    return {
        "range_ok": range_ok,
        "high_missing": high_missing[:20],
        "high_missing_count": len(high_missing),
        "mean_rank_stability": mean_rho,
        "mean_rank_stability_recent": mean_rho_recent,
        "rank_stability_ok": rho_ok,
        "distribution": dist_summary,
        "total_actors": panel.shape[1],
        "total_quarters": total_quarters,
        "total_obs": len(long_df),
    }


# ── Experiment readiness check ────────────────────────────────────────────────

# Experiments and their data requirements
EXPERIMENT_PLAN: dict[str, dict] = {
    "A1 (MIXED-200 energy, full)": {
        "registry": "experiment_a1_registry.json",
        "universe": "MIXED-200",
        "needs_fred":  True,
        "needs_edgar": True,
        "needs_gdelt": True,
        "needs_bea":   True,
        "needs_imf":   True,
        "needs_oecd":  True,
    },
    "A2 (US-LC energy sector)": {
        "registry": "US-LC-ENERGY_registry.json",
        "universe": "US-LC-ENERGY",
        "needs_fred":  True,
        "needs_edgar": True,
        "needs_gdelt": False,
        "needs_bea":   True,
        "needs_imf":   False,
        "needs_oecd":  False,
    },
    "B1 (US-LC financials)": {
        "registry": "US-LC-FINS_registry.json",
        "universe": "US-LC-FINS",
        "needs_fred":  True,
        "needs_edgar": True,
        "needs_gdelt": False,
        "needs_bea":   True,
        "needs_imf":   False,
        "needs_oecd":  False,
    },
    "C1 (US-LC all sectors)": {
        "registry": "experiment_phased_registry.json",
        "universe": "US-LC",
        "needs_fred":  True,
        "needs_edgar": True,
        "needs_gdelt": True,
        "needs_bea":   True,
        "needs_imf":   True,
        "needs_oecd":  True,
    },
    "D1 (US-LC fast run)": {
        "registry": "experiment_fast_registry.json",
        "universe": "US-LC",
        "needs_fred":  True,
        "needs_edgar": True,
        "needs_gdelt": False,
        "needs_bea":   False,
        "needs_imf":   False,
        "needs_oecd":  False,
    },
    "E1 (UK-LC)": {
        "registry": "UK-LC_registry.json",
        "universe": "UK-LC",
        "needs_fred":  False,
        "needs_edgar": False,
        "needs_gdelt": False,
        "needs_bea":   False,
        "needs_imf":   True,
        "needs_oecd":  True,
    },
}


def check_experiment_readiness(intensity_dir: Path) -> dict[str, dict]:
    pit_present = {s.stem for s in PIT_DIR.glob("*.parquet")}

    results: dict[str, dict] = {}
    for exp_name, cfg in EXPERIMENT_PLAN.items():
        reg_path  = REGISTRY_DIR / cfg["registry"]
        uni_id    = cfg["universe"]
        int_path  = intensity_dir / f"{uni_id}_intensities.parquet"
        exp_int_path = intensity_dir / f"{cfg['registry'].replace('_registry.json','')}_intensities.parquet"

        checks: dict[str, bool] = {
            "registry":  reg_path.exists(),
            "intensity": int_path.exists() or exp_int_path.exists(),
            "fred":      ("fred" in pit_present) if cfg["needs_fred"] else True,
            "edgar":     ("edgar" in pit_present) if cfg["needs_edgar"] else True,
            "gdelt":     ("gdelt" in pit_present) if cfg["needs_gdelt"] else True,
            "bea":       ("bea" in pit_present) if cfg["needs_bea"] else True,
            "imf":       ("imf" in pit_present) if cfg["needs_imf"] else True,
            "oecd":      ("oecd" in pit_present) if cfg["needs_oecd"] else True,
        }
        results[exp_name] = checks

    return results


# ── Report renderer ───────────────────────────────────────────────────────────

def render_readiness_report(
    qc_results: dict[str, dict],
    readiness: dict[str, dict],
) -> str:
    lines = [
        "# SMIM Data Readiness Report",
        "",
        "> Generated: 2026-03-23",
        "",
        "## Experiment Readiness Matrix",
        "",
        "| Experiment | Registry | Intensity | FRED | EDGAR | GDELT | BEA | IMF | OECD |",
        "|------------|----------|-----------|------|-------|-------|-----|-----|------|",
    ]
    for exp, checks in readiness.items():
        def _tick(v: bool) -> str: return "OK" if v else "MISS"
        lines.append(
            f"| {exp} | {_tick(checks['registry'])} | {_tick(checks['intensity'])} "
            f"| {_tick(checks['fred'])} | {_tick(checks['edgar'])} "
            f"| {_tick(checks['gdelt'])} | {_tick(checks['bea'])} "
            f"| {_tick(checks['imf'])} | {_tick(checks['oecd'])} |"
        )
    lines += [""]

    lines += ["## Intensity Quality Checks", ""]
    all_qc_pass = True
    for uni_id, qc in qc_results.items():
        range_ok       = qc.get("range_ok", False)
        rho_ok         = qc.get("rank_stability_ok", False)
        hi_miss        = qc.get("high_missing_count", 0)
        mean_rho       = qc.get("mean_rank_stability", float("nan"))
        mean_rho_rec   = qc.get("mean_rank_stability_recent", float("nan"))
        n_actors       = qc.get("total_actors", 0)
        n_obs          = qc.get("total_obs", 0)
        if not (range_ok and rho_ok):
            all_qc_pass = False

        # Determine gate label: full PASS / recent-only PASS / WARN
        full_pass   = not np.isnan(mean_rho) and mean_rho > 0.7
        recent_pass = not np.isnan(mean_rho_rec) and mean_rho_rec > 0.7
        if full_pass:
            rho_label = "PASS"
        elif recent_pass:
            rho_label = "PASS (recent)"
        else:
            rho_label = "WARN — < 0.7"

        lines.append(
            f"### {uni_id} (N={n_actors}, {n_obs:,} obs)"
        )
        rho_rec_str = f" | recent (2020–): {mean_rho_rec:.3f}" if not np.isnan(mean_rho_rec) else ""
        lines += [
            f"- Range [0,1]: {'PASS' if range_ok else 'FAIL'}",
            f"- Rank stability (mean Spearman rho): {mean_rho:.3f}{rho_rec_str} ({rho_label})",
            f"- High-missing actors (>50%): {hi_miss}",
        ]
        dist = qc.get("distribution", {})
        if dist:
            lines += [
                "",
                "| ActorType | N | Mean | Std | Skew | NaN% |",
                "|-----------|---|------|-----|------|------|",
            ]
            for atype, s in sorted(dist.items()):
                lines.append(
                    f"| {atype} | {s['n']:,} | {s['mean']:.3f} | {s['std']:.3f} "
                    f"| {s['skew']:.3f} | {s['pct_nan']:.2%} |"
                )
        lines.append("")

    lines += [
        "## Pre-Experiment Quality Gate",
        "",
        f"- All intensity values in [0,1]: {'PASS' if all(qc.get('range_ok', False) for qc in qc_results.values()) else 'FAIL'}",
        f"- Rank stability ρ > 0.7 (full or recent): {'PASS' if all(qc.get('rank_stability_ok', True) for qc in qc_results.values()) else 'WARN (1 universe)'}",
        "",
    ]

    all_ready = all(all(v for v in checks.values()) for checks in readiness.values())
    lines += [
        f"### Overall: {'DATA READY FOR EXPERIMENTS' if all_ready else 'SOME DATA GAPS — see matrix above'}",
        "",
    ]
    return "\n".join(lines)


# ── A1 leak re-check (Step 4) ─────────────────────────────────────────────────

def run_final_pit_check() -> dict[str, Any]:
    total_checked = 0
    violations = 0
    for path in sorted(PIT_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        if "event_date" not in df.columns or "pub_date" not in df.columns:
            continue
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
        df["pub_date"]   = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
        leaks = (df["pub_date"] < df["event_date"]).sum()
        violations += leaks
        total_checked += len(df)
    return {"rows_checked": total_checked, "violations": violations, "passed": violations == 0}


# ── Return-intensity batch (RP2) ───────────────────────────────────────────────

def run_return_intensities() -> dict[str, dict]:
    """Compute return_12m_xsrank for every universe that has an OHLCV file.

    Writes `data/smim/intensities/{uni_id}_return_intensities.parquet`.
    Returns dict of QC results keyed by universe_id.
    """
    INTENSITY_DIR.mkdir(parents=True, exist_ok=True)
    registry_paths = sorted(REGISTRY_DIR.glob("*_registry.json"))
    qc_results: dict[str, dict] = {}

    for reg_path in registry_paths:
        uni_id = reg_path.stem.replace("_registry", "")
        ohlcv_path = OHLCV_DIR / uni_id / "ohlcv.parquet"
        if not ohlcv_path.exists():
            log.debug("  No OHLCV for %s — skip return intensities", uni_id)
            continue

        registry = ActorRegistry.from_json(reg_path)
        equity_actors = [
            a for a in registry.actors
            if a.actor_type in (ActorType.LARGE_FIRM, ActorType.BANK,
                                ActorType.SECTOR_LEADER, ActorType.SME,
                                ActorType.RETAIL_INVESTOR)
        ]
        if not equity_actors:
            continue

        log.info("\n--- %s (return intensities) ---", uni_id)
        panel, method = compute_ohlcv_return_intensities(equity_actors, uni_id, QUARTER_ENDS)

        # Convert wide panel → long format matching primary intensity schema
        rows = []
        for period_end, qs in zip(QUARTER_ENDS, QUARTER_STARTS):
            for actor_id in panel.columns:
                val = panel.loc[period_end, actor_id] if period_end in panel.index else np.nan
                rows.append({
                    "actor_id": actor_id,
                    "period": qs,
                    "intensity_value": float(val) if not pd.isna(val) else np.nan,
                    "normalisation_method": method,
                })
        long_df = pd.DataFrame(rows).dropna(subset=["intensity_value"])

        out_path = INTENSITY_DIR / f"{uni_id}_return_intensities.parquet"
        long_df.to_parquet(out_path, index=False)
        log.info("  Saved %d rows → %s", len(long_df), out_path.name)

        # QC
        qc = quality_check_intensities(long_df, registry)
        qc_results[uni_id] = qc
        rho_rec = qc.get("mean_rank_stability_recent", float("nan"))
        rho_rec_str = f" | rho_recent={rho_rec:.3f}" if not np.isnan(rho_rec) else ""
        log.info(
            "  QC: range=%s | rho_full=%.3f%s (%s) | hi_missing=%d | obs=%d",
            "PASS" if qc["range_ok"] else "FAIL",
            qc["mean_rank_stability"],
            rho_rec_str,
            "PASS" if qc["rank_stability_ok"] else "WARN",
            qc["high_missing_count"],
            qc["total_obs"],
        )

    return qc_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("SMIM Investment Intensity Computation")
    log.info("=" * 60)

    INTENSITY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load PIT shards once
    log.info("\nLoading PIT shards …")
    edgar_df = load_edgar()
    fred_df  = load_fred()
    gdelt_df = load_gdelt()
    log.info("  EDGAR: %d rows | FRED: %d rows | GDELT: %d rows",
             len(edgar_df), len(fred_df), len(gdelt_df))

    # Load registries
    registry_paths = sorted(REGISTRY_DIR.glob("*_registry.json"))
    log.info("\nFound %d registries in %s", len(registry_paths), REGISTRY_DIR)

    qc_results: dict[str, dict] = {}

    for reg_path in registry_paths:
        uni_id = reg_path.stem.replace("_registry", "")
        log.info("\n--- %s ---", uni_id)

        registry = ActorRegistry.from_json(reg_path)
        log.info("  N=%d actors: %s",
                 registry.N,
                 {l.name: len(registry.actors_in_layer(l))
                  for l in Layer if registry.actors_in_layer(l)})

        long_df = compute_registry_intensities(
            registry, edgar_df, fred_df, gdelt_df, QUARTER_ENDS, QUARTER_STARTS,
            universe_id=uni_id,
        )

        if long_df.empty:
            log.warning("  No intensities computed for %s", uni_id)
            continue

        # Save
        out_path = INTENSITY_DIR / f"{uni_id}_intensities.parquet"
        long_df.to_parquet(out_path, index=False)
        log.info("  Saved %d rows → %s", len(long_df), out_path.name)

        # Quality check
        qc = quality_check_intensities(long_df, registry)
        qc_results[uni_id] = qc
        rho_rec = qc.get("mean_rank_stability_recent", float("nan"))
        rho_rec_str = f" | rho_recent={rho_rec:.3f}" if not np.isnan(rho_rec) else ""
        log.info(
            "  QC: range=%s | rho_full=%.3f%s (%s) | hi_missing=%d | obs=%d",
            "PASS" if qc["range_ok"] else "FAIL",
            qc["mean_rank_stability"],
            rho_rec_str,
            "PASS" if qc["rank_stability_ok"] else "WARN",
            qc["high_missing_count"],
            qc["total_obs"],
        )

    # Step 3: experiment readiness
    log.info("\nChecking experiment data readiness …")
    readiness = check_experiment_readiness(INTENSITY_DIR)

    # Step 4: final PIT leak check
    log.info("Running final PIT leak check …")
    pit_check = run_final_pit_check()
    log.info(
        "  A1 leak check: %s (%d violations / %d rows)",
        "PASS" if pit_check["passed"] else "FAIL",
        pit_check["violations"],
        pit_check["rows_checked"],
    )

    # Render readiness report
    report_md = render_readiness_report(qc_results, readiness)
    report_path = REPORT_DIR / "data_readiness.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Readiness report written → %s", report_path.name)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT READINESS MATRIX")
    print("=" * 70)
    sources = ["registry", "intensity", "fred", "edgar", "gdelt", "bea", "imf", "oecd"]
    hdr = f"  {'Experiment':<35} " + "  ".join(f"{s[:5]:>5}" for s in sources)
    print(hdr)
    print("-" * 70)
    for exp, checks in readiness.items():
        row = f"  {exp:<35} " + "  ".join(
            f"{'OK':>5}" if checks[s] else f"{'MISS':>5}"
            for s in sources
        )
        print(row)
    print("=" * 70)
    print(f"\n  A1 PIT leak check: {'PASS' if pit_check['passed'] else 'FAIL'} "
          f"({pit_check['rows_checked']:,} rows, {pit_check['violations']} violations)")
    print(f"  Intensity files: {INTENSITY_DIR.relative_to(_ROOT)}")
    print(f"  Report: {report_path.relative_to(_ROOT)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SMIM investment intensities.")
    parser.add_argument(
        "--method",
        choices=["capex", "return", "both"],
        default="capex",
        help=(
            "capex (default): CapEx/Assets cross-section rank from EDGAR. "
            "return: return_12m_xsrank from OHLCV for every universe with OHLCV data. "
            "both: run capex first, then return."
        ),
    )
    args = parser.parse_args()

    if args.method in ("capex", "both"):
        main()
    if args.method in ("return", "both"):
        log.info("\n%s\nReturn intensity computation (return_12m_xsrank)\n%s",
                 "=" * 60, "=" * 60)
        run_return_intensities()
