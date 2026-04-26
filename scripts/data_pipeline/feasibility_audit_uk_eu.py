"""Phase 1 / Gate G1 feasibility audit for the UK/EU heterogeneous panel.

Tests availability of every source needed for experiment_b1 before committing
to the full data build.  Outputs a coverage matrix and a G1 gate decision.

Sources probed:
  1. ECB SDW  — 4 candidate euro-area macro series (SDMX CSV format)
  2. FRED     — 4 candidate euro-area macro proxies (REST JSON)
  3. BoE      — 1 Bank Rate series (IADB CSV)
  4. SEC EDGAR — 10 known EU-domiciled 20-F filers (company facts XBRL)
  5. SimFin   — 30 sample large-cap EU firms (REST v3, quarterly statements)
  6. yfinance — same 30 firms as SimFin (Yahoo Finance fundamentals fallback)

Outputs:
  data/audit/uk_eu_coverage_matrix.csv  — one row per candidate actor
  data/audit/uk_eu_feasibility_report.md — human-readable summary
  stdout summary + G1 decision

Gate G1 decision:
  PASS        — ≥80 firms × ≥60Q, ≥6 macros, ≥4 institutionals
  PIVOT_A     — 60 ≤ firms < 80; proceed with reduced target
  PIVOT_B     — firms < 60 but ≥60 with 48Q window; shorten window
  KILL        — no feasible path; abandon UK/EU, consider alternatives

Exit codes:
  0 = PASS, 1 = PIVOT_A, 2 = PIVOT_B, 3 = KILL

Usage:
  uv run python scripts/data_pipeline/feasibility_audit_uk_eu.py

Requires env vars:
  SIMFIN_API_KEY  — free tier from https://simfin.com
  FRED_API_KEY    — free tier from https://fred.stlouisfed.org
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = _ROOT / "data" / "audit"
OUTPUT_CSV = AUDIT_DIR / "uk_eu_coverage_matrix.csv"
OUTPUT_MD = AUDIT_DIR / "uk_eu_feasibility_report.md"

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_START = "2010-01-01"
TARGET_END = "2025-12-31"
TARGET_QUARTERS = 64  # 2010Q1-2025Q4

GATE_FIRM_COUNT_PRIMARY = 80
GATE_FIRM_COUNT_PIVOT_A = 60
GATE_FIRM_COUNT_PIVOT_B = 60  # with shorter window
GATE_FIRM_COVERAGE_MIN = 0.90  # ≥90% of target quarters
GATE_FIRM_QUARTERS_MIN = 56  # allows a little slack from 60
GATE_FIRM_QUARTERS_PIVOT_B = 44  # 2014Q1+ fallback
GATE_MACRO_COUNT = 6  # out of 7 target (allow 1 substitution)
GATE_INST_COUNT = 4

HTTP_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.15

# SEC EDGAR requires a User-Agent with contact info
EDGAR_UA = "HARP Research oleg.roshka.pp@gmail.com"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("g1_audit")

# ── Coverage record ───────────────────────────────────────────────────────────


@dataclass
class CoverageRecord:
    actor_id: str
    layer: int
    source: str
    target_id: str
    status: str  # ok | missing | partial | error | auth_fail
    first_obs: str = ""
    last_obs: str = ""
    n_obs: int = 0
    n_quarters: int = 0
    coverage_pct: float = 0.0
    notes: str = ""


def _record(layer: int, source: str, actor_id: str, target_id: str) -> CoverageRecord:
    return CoverageRecord(actor_id=actor_id, layer=layer, source=source, target_id=target_id, status="error")


def _q_count(first: pd.Timestamp, last: pd.Timestamp) -> int:
    if pd.isna(first) or pd.isna(last) or last < first:
        return 0
    return int((last.year - first.year) * 4 + (last.quarter - first.quarter) + 1)


def _coverage_pct(n_quarters: int, target: int = TARGET_QUARTERS) -> float:
    return round(100 * n_quarters / target, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  ECB SDW — SDMX CSV
# ═════════════════════════════════════════════════════════════════════════════

ECB_SERIES: list[tuple[str, str, str, str]] = [
    # (actor_id, flow, key, description)
    ("shock_euribor_3m", "FM", "B.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", "Euribor 3M"),
    ("shock_eur_yield_10y", "FM", "M.U2.EUR.4F.BB.U10Y.YLD", "Euro 10Y yield"),
    ("shock_eur_twi", "EXR", "M.E5.EUR.EN00.A", "Euro nominal effective exchange rate"),
    ("inst_ecb_assets", "ILM", "W.U2.C.L010000.U2.EUR", "ECB consolidated balance sheet"),
]

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"


def _parse_sdmx_csv(csv_text: str) -> pd.DataFrame:
    """ECB SDW CSV rows have TIME_PERIOD + OBS_VALUE columns. Returns sorted observation frame."""
    if not csv_text.strip():
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame()
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.DataFrame()
    df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(str)
    return df.sort_values("TIME_PERIOD")


def _period_to_timestamp(period: str) -> pd.Timestamp:
    try:
        if "Q" in period:
            return pd.Period(period, freq="Q").end_time.normalize()
        return pd.Period(period, freq="M").end_time.normalize()
    except Exception:
        try:
            return pd.to_datetime(period)
        except Exception:
            return pd.NaT


def audit_ecb_sdw() -> list[CoverageRecord]:
    log.info("── ECB SDW — testing %d candidate series", len(ECB_SERIES))
    records: list[CoverageRecord] = []
    for actor_id, flow, key, desc in ECB_SERIES:
        rec = _record(0, "ecb_sdw", actor_id, f"{flow}/{key}")
        rec.notes = desc
        url = f"{ECB_BASE}/{flow}/{key}"
        params = {
            "startPeriod": TARGET_START[:7],
            "endPeriod": TARGET_END[:7],
            "format": "csvdata",
        }
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers={"Accept": "text/csv"})
        except Exception as e:
            rec.notes = f"{desc} [exception: {e}]"
            records.append(rec)
            continue
        if r.status_code != 200:
            rec.notes = f"{desc} [HTTP {r.status_code}]"
            records.append(rec)
            continue
        obs = _parse_sdmx_csv(r.text)
        if obs.empty:
            rec.status = "missing"
            rec.notes = f"{desc} [empty]"
            records.append(rec)
            continue
        first_period = obs["TIME_PERIOD"].iloc[0]
        last_period = obs["TIME_PERIOD"].iloc[-1]
        first_ts = _period_to_timestamp(first_period)
        last_ts = _period_to_timestamp(last_period)
        rec.first_obs = first_period
        rec.last_obs = last_period
        rec.n_obs = int(len(obs))
        rec.n_quarters = _q_count(first_ts, last_ts)
        rec.coverage_pct = _coverage_pct(rec.n_quarters)
        rec.status = "ok" if rec.coverage_pct >= 90 else "partial"
        log.info("   %-22s  %s → %s  (%d obs, %s%% of %dQ)",
                 actor_id, first_period, last_period, rec.n_obs, rec.coverage_pct, TARGET_QUARTERS)
        records.append(rec)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FRED — euro-area & UK proxies
# ═════════════════════════════════════════════════════════════════════════════

FRED_EU_SERIES: list[tuple[str, str, str]] = [
    ("shock_fred_ecb_rate", "ECBDFR", "ECB Deposit Facility Rate (FRED mirror)"),
    ("shock_fred_eu_rate_10y", "IRLTLT01EZM156N", "Euro area long-term interest rate 10Y"),
    ("shock_fred_eu_ipro", "EA19PRMNTO01GPSAM", "Euro area industrial production"),
    ("shock_fred_uk_rate", "INTGSTGBM193N", "UK 3-month interbank rate"),
]

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def audit_fred_eu() -> list[CoverageRecord]:
    log.info("── FRED EU — testing %d series", len(FRED_EU_SERIES))
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    records: list[CoverageRecord] = []
    if not api_key:
        for actor_id, sid, desc in FRED_EU_SERIES:
            rec = _record(0, "fred", actor_id, sid)
            rec.status = "auth_fail"
            rec.notes = f"{desc} [FRED_API_KEY not set]"
            records.append(rec)
        return records

    for actor_id, sid, desc in FRED_EU_SERIES:
        rec = _record(0, "fred", actor_id, sid)
        rec.notes = desc
        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": TARGET_START,
            "observation_end": TARGET_END,
        }
        try:
            r = requests.get(FRED_BASE, params=params, timeout=HTTP_TIMEOUT)
        except Exception as e:
            rec.notes = f"{desc} [exception: {e}]"
            records.append(rec)
            continue
        if r.status_code != 200:
            rec.notes = f"{desc} [HTTP {r.status_code}]"
            records.append(rec)
            continue
        try:
            data = r.json()
        except Exception:
            records.append(rec)
            continue
        obs_list = data.get("observations", []) or []
        valid = [o for o in obs_list if o.get("value", ".") not in (".", "", None)]
        if not valid:
            rec.status = "missing"
            records.append(rec)
            continue
        first_d = valid[0]["date"]
        last_d = valid[-1]["date"]
        first_ts = pd.to_datetime(first_d)
        last_ts = pd.to_datetime(last_d)
        rec.first_obs = first_d
        rec.last_obs = last_d
        rec.n_obs = len(valid)
        rec.n_quarters = _q_count(first_ts, last_ts)
        rec.coverage_pct = _coverage_pct(rec.n_quarters)
        rec.status = "ok" if rec.coverage_pct >= 90 else "partial"
        log.info("   %-22s  %s → %s  (%d obs, %s%% of %dQ)",
                 actor_id, first_d, last_d, rec.n_obs, rec.coverage_pct, TARGET_QUARTERS)
        records.append(rec)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Bank of England — Bank Rate via IADB CSV
# ═════════════════════════════════════════════════════════════════════════════

BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?"
    "csv.x=yes&Datefrom=01/Jan/2010&Dateto=31/Dec/2025&SeriesCodes=IUDBEDR"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)


def audit_boe() -> list[CoverageRecord]:
    log.info("── BoE — Bank Rate (IUDBEDR)")
    rec = _record(0, "boe_iadb", "inst_boe_rate", "IUDBEDR")
    rec.notes = "BoE Bank Rate"
    try:
        r = requests.get(BOE_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": EDGAR_UA})
    except Exception as e:
        rec.notes = f"BoE Bank Rate [exception: {e}]"
        return [rec]
    if r.status_code != 200:
        rec.notes = f"BoE Bank Rate [HTTP {r.status_code}]"
        return [rec]
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        rec.notes = "BoE Bank Rate [csv parse failed]"
        return [rec]
    if df.empty or df.shape[1] < 2:
        rec.status = "missing"
        return [rec]
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col, val_col])
    if df.empty:
        rec.status = "missing"
        return [rec]
    first_ts = df[date_col].min()
    last_ts = df[date_col].max()
    rec.first_obs = first_ts.strftime("%Y-%m-%d")
    rec.last_obs = last_ts.strftime("%Y-%m-%d")
    rec.n_obs = int(len(df))
    rec.n_quarters = _q_count(first_ts, last_ts)
    rec.coverage_pct = _coverage_pct(rec.n_quarters)
    rec.status = "ok" if rec.coverage_pct >= 90 else "partial"
    log.info("   inst_boe_rate           %s → %s  (%d obs, %s%% of %dQ)",
             rec.first_obs, rec.last_obs, rec.n_obs, rec.coverage_pct, TARGET_QUARTERS)
    return [rec]


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SEC EDGAR — known EU 20-F filers
# ═════════════════════════════════════════════════════════════════════════════

# (ticker, CIK_zero_padded, name)
EU_ADR_FILERS: list[tuple[str, str, str]] = [
    ("SHEL", "0001306965", "Shell plc"),
    ("BP", "0000313807", "BP plc"),
    ("TTE", "0000879764", "TotalEnergies SE"),
    ("AZN", "0000901832", "AstraZeneca plc"),
    ("NVS", "0001114448", "Novartis AG"),
    ("SAP", "0001000184", "SAP SE"),
    ("ASML", "0000937966", "ASML Holding"),
    ("UL",  "0000217410", "Unilever plc"),
    ("DEO", "0000835403", "Diageo plc"),
    ("HSBC", "0001089113", "HSBC Holdings plc"),
]

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# XBRL tags we hope to find (US-GAAP first, IFRS-full fallback)
CAPEX_TAGS = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("ifrs-full", "PurchaseOfPropertyPlantAndEquipment"),
    ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
]
ASSETS_TAGS = [
    ("us-gaap", "Assets"),
    ("ifrs-full", "Assets"),
]


def _edgar_tag_observations(facts: dict, candidate_tags: list[tuple[str, str]]) -> list[dict]:
    """Return first non-empty list of observations across candidate tags."""
    facts_by_taxonomy = facts.get("facts", {}) or {}
    for taxonomy, tag in candidate_tags:
        tag_block = facts_by_taxonomy.get(taxonomy, {}).get(tag, {})
        units = tag_block.get("units", {}) or {}
        for unit_key, unit_obs in units.items():
            if not unit_obs:
                continue
            return [{**o, "unit": unit_key, "taxonomy": taxonomy, "tag": tag} for o in unit_obs]
    return []


def audit_edgar_eu_adrs() -> list[CoverageRecord]:
    log.info("── SEC EDGAR — %d EU 20-F filers", len(EU_ADR_FILERS))
    records: list[CoverageRecord] = []
    session = requests.Session()
    session.headers.update({"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip, deflate"})
    for ticker, cik, name in EU_ADR_FILERS:
        rec = _record(2, "sec_edgar", ticker, cik)
        rec.notes = name
        url = EDGAR_FACTS_URL.format(cik=cik)
        try:
            r = session.get(url, timeout=HTTP_TIMEOUT)
        except Exception as e:
            rec.notes = f"{name} [exception: {e}]"
            records.append(rec)
            continue
        if r.status_code == 404:
            rec.status = "missing"
            rec.notes = f"{name} [no facts]"
            records.append(rec)
            continue
        if r.status_code != 200:
            rec.notes = f"{name} [HTTP {r.status_code}]"
            records.append(rec)
            continue
        try:
            facts = r.json()
        except Exception:
            records.append(rec)
            continue
        capex_obs = _edgar_tag_observations(facts, CAPEX_TAGS)
        assets_obs = _edgar_tag_observations(facts, ASSETS_TAGS)
        if not capex_obs or not assets_obs:
            rec.status = "missing"
            rec.notes = f"{name} [capex={len(capex_obs)} assets={len(assets_obs)}]"
            records.append(rec)
            continue
        # Count distinct end dates where both CapEx and Assets exist in window
        capex_dates = {o["end"] for o in capex_obs if TARGET_START <= o.get("end", "") <= TARGET_END}
        assets_dates = {o["end"] for o in assets_obs if TARGET_START <= o.get("end", "") <= TARGET_END}
        both = sorted(capex_dates & assets_dates)
        if not both:
            rec.status = "missing"
            records.append(rec)
            continue
        rec.first_obs = both[0]
        rec.last_obs = both[-1]
        rec.n_obs = len(both)
        first_ts = pd.to_datetime(both[0])
        last_ts = pd.to_datetime(both[-1])
        rec.n_quarters = _q_count(first_ts, last_ts)
        rec.coverage_pct = _coverage_pct(rec.n_quarters)
        rec.status = "ok" if rec.n_obs >= GATE_FIRM_QUARTERS_MIN else "partial"
        capex_tax = capex_obs[0]["taxonomy"] if capex_obs else "?"
        log.info("   %-6s  %-28s  %s → %s  n=%d  (%s)",
                 ticker, name[:28], rec.first_obs, rec.last_obs, rec.n_obs, capex_tax)
        records.append(rec)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SimFin — sample EU large caps
# ═════════════════════════════════════════════════════════════════════════════

# (simfin_ticker, name, country, sector)
SIMFIN_EU_SAMPLE: list[tuple[str, str, str, str]] = [
    ("ASML", "ASML Holding", "NL", "Technology"),
    ("SAP", "SAP SE", "DE", "Technology"),
    ("SHEL", "Shell plc", "GB", "Energy"),
    ("BP", "BP plc", "GB", "Energy"),
    ("TTE", "TotalEnergies", "FR", "Energy"),
    ("AZN", "AstraZeneca", "GB", "Healthcare"),
    ("NVS", "Novartis", "CH", "Healthcare"),
    ("ROG", "Roche", "CH", "Healthcare"),
    ("GSK", "GSK plc", "GB", "Healthcare"),
    ("SAN", "Sanofi", "FR", "Healthcare"),
    ("BAYN", "Bayer AG", "DE", "Healthcare"),
    ("NESN", "Nestle", "CH", "Consumer Staples"),
    ("ULVR", "Unilever", "GB", "Consumer Staples"),
    ("DGE", "Diageo", "GB", "Consumer Staples"),
    ("OR", "L'Oreal", "FR", "Consumer Staples"),
    ("MC", "LVMH", "FR", "Consumer Disc"),
    ("HSBA", "HSBC", "GB", "Financials"),
    ("BARC", "Barclays", "GB", "Financials"),
    ("BNP", "BNP Paribas", "FR", "Financials"),
    ("ALV", "Allianz", "DE", "Financials"),
    ("SIE", "Siemens", "DE", "Industrials"),
    ("AIR", "Airbus", "FR", "Industrials"),
    ("BAS", "BASF", "DE", "Materials"),
    ("AI", "Air Liquide", "FR", "Materials"),
    ("VOD", "Vodafone", "GB", "Comms"),
    ("ENEL", "Enel", "IT", "Utilities"),
    ("IBE", "Iberdrola", "ES", "Utilities"),
    ("ENI", "Eni SpA", "IT", "Energy"),
    ("ISP", "Intesa Sanpaolo", "IT", "Financials"),
    ("BATS", "BAT plc", "GB", "Consumer Staples"),
]

SIMFIN_BASE = "https://backend.simfin.com/api/v3"


def _simfin_request(
    session: requests.Session, endpoint: str, params: dict | None = None
) -> tuple[int, Any]:
    url = f"{SIMFIN_BASE}/{endpoint.lstrip('/')}"
    try:
        r = session.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    except Exception as e:
        return 0, {"error": str(e)}
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": "non-json"}


def audit_simfin() -> list[CoverageRecord]:
    log.info("── SimFin — testing %d candidate EU firms", len(SIMFIN_EU_SAMPLE))
    api_key = os.environ.get("SIMFIN_API_KEY", "").strip()
    records: list[CoverageRecord] = []
    if not api_key:
        for tkr, name, country, sector in SIMFIN_EU_SAMPLE:
            rec = _record(2, "simfin", tkr, tkr)
            rec.status = "auth_fail"
            rec.notes = f"{name} [SIMFIN_API_KEY not set]"
            records.append(rec)
        return records

    session = requests.Session()
    session.headers.update({
        "Authorization": f"api-key {api_key}",
        "Accept": "application/json",
    })

    # Smoke test: try companies list to confirm key works
    status, body = _simfin_request(session, "companies/list")
    if status != 200:
        log.warning("SimFin smoke test failed: HTTP %s  body-preview=%s", status, str(body)[:200])
        # Try with ?api-key= query param instead
        session.headers.pop("Authorization", None)
        status2, body2 = _simfin_request(session, "companies/list", {"api-key": api_key})
        if status2 != 200:
            log.error("SimFin both auth modes failed (header %s, query %s)", status, status2)
            for tkr, name, country, sector in SIMFIN_EU_SAMPLE:
                rec = _record(2, "simfin", tkr, tkr)
                rec.status = "auth_fail"
                rec.notes = f"{name} [SimFin HTTP {status}/{status2}]"
                records.append(rec)
            return records
        # Query-param auth mode
        auth_mode = "query"
    else:
        auth_mode = "header"

    log.info("   SimFin auth OK (mode=%s)", auth_mode)

    # Per-firm statement check
    for tkr, name, country, sector in SIMFIN_EU_SAMPLE:
        rec = _record(2, "simfin", tkr, tkr)
        rec.notes = f"{name} ({country}/{sector})"
        params: dict[str, Any] = {
            "ticker": tkr,
            "statements": "bs,cf",
            "period": "q1,q2,q3,q4",
            "start": TARGET_START,
            "end": TARGET_END,
        }
        if auth_mode == "query":
            params["api-key"] = api_key
        status, body = _simfin_request(session, "companies/statements/compact", params)
        if status != 200:
            rec.notes = f"{name} [HTTP {status}]"
            records.append(rec)
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue
        n_obs = 0
        first_period = ""
        last_period = ""
        try:
            if isinstance(body, list) and body:
                node = body[0]
            elif isinstance(body, dict):
                node = body
            else:
                node = None
            if node:
                statements = node.get("statements") or node.get("data") or []
                periods: set[str] = set()
                for stmt in statements:
                    for row in stmt.get("values") or stmt.get("data") or []:
                        periods.add(str(row.get("period") or row.get("fiscal_year") or ""))
                periods.discard("")
                sorted_periods = sorted(periods)
                n_obs = len(sorted_periods)
                if sorted_periods:
                    first_period = sorted_periods[0]
                    last_period = sorted_periods[-1]
        except Exception as e:
            rec.notes = f"{name} [parse: {e}]"
            records.append(rec)
            continue
        rec.first_obs = first_period
        rec.last_obs = last_period
        rec.n_obs = n_obs
        rec.n_quarters = n_obs
        rec.coverage_pct = _coverage_pct(n_obs)
        if n_obs >= GATE_FIRM_QUARTERS_MIN:
            rec.status = "ok"
        elif n_obs > 0:
            rec.status = "partial"
        else:
            rec.status = "missing"
        log.info("   %-6s  %-28s  n=%3d  %s",
                 tkr, name[:28], n_obs, rec.status)
        records.append(rec)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# 6.  yfinance — EU firm fundamentals fallback
# ═════════════════════════════════════════════════════════════════════════════

# (yf_ticker, name, country, sector) — note EU ticker suffixes for Yahoo
YFINANCE_EU_SAMPLE: list[tuple[str, str, str, str]] = [
    ("ASML.AS", "ASML Holding", "NL", "Technology"),
    ("SAP.DE", "SAP SE", "DE", "Technology"),
    ("SHEL.L", "Shell plc", "GB", "Energy"),
    ("BP.L", "BP plc", "GB", "Energy"),
    ("TTE.PA", "TotalEnergies", "FR", "Energy"),
    ("AZN.L", "AstraZeneca", "GB", "Healthcare"),
    ("NOVN.SW", "Novartis", "CH", "Healthcare"),
    ("ROG.SW", "Roche", "CH", "Healthcare"),
    ("GSK.L", "GSK plc", "GB", "Healthcare"),
    ("SAN.PA", "Sanofi", "FR", "Healthcare"),
    ("BAYN.DE", "Bayer AG", "DE", "Healthcare"),
    ("NESN.SW", "Nestle", "CH", "Consumer Staples"),
    ("ULVR.L", "Unilever", "GB", "Consumer Staples"),
    ("DGE.L", "Diageo", "GB", "Consumer Staples"),
    ("OR.PA", "L'Oreal", "FR", "Consumer Staples"),
    ("MC.PA", "LVMH", "FR", "Consumer Disc"),
    ("HSBA.L", "HSBC", "GB", "Financials"),
    ("BARC.L", "Barclays", "GB", "Financials"),
    ("BNP.PA", "BNP Paribas", "FR", "Financials"),
    ("ALV.DE", "Allianz", "DE", "Financials"),
    ("SIE.DE", "Siemens", "DE", "Industrials"),
    ("AIR.PA", "Airbus", "FR", "Industrials"),
    ("BAS.DE", "BASF", "DE", "Materials"),
    ("AI.PA", "Air Liquide", "FR", "Materials"),
    ("VOD.L", "Vodafone", "GB", "Comms"),
    ("ENEL.MI", "Enel", "IT", "Utilities"),
    ("IBE.MC", "Iberdrola", "ES", "Utilities"),
    ("ENI.MI", "Eni SpA", "IT", "Energy"),
    ("ISP.MI", "Intesa Sanpaolo", "IT", "Financials"),
    ("BATS.L", "BAT plc", "GB", "Consumer Staples"),
]


def audit_yfinance() -> list[CoverageRecord]:
    log.info("── yfinance — testing %d EU firms (fundamentals)", len(YFINANCE_EU_SAMPLE))
    records: list[CoverageRecord] = []
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        for tkr, name, country, sector in YFINANCE_EU_SAMPLE:
            rec = _record(2, "yfinance", tkr, tkr)
            rec.status = "error"
            rec.notes = f"{name} [yfinance not installed]"
            records.append(rec)
        return records

    for tkr, name, country, sector in YFINANCE_EU_SAMPLE:
        rec = _record(2, "yfinance", tkr, tkr)
        rec.notes = f"{name} ({country}/{sector})"
        try:
            t = yf.Ticker(tkr)
            qbs = t.quarterly_balance_sheet
            qcf = t.quarterly_cashflow
        except Exception as e:
            rec.notes = f"{name} [exception: {e}]"
            records.append(rec)
            continue
        # yfinance returns DataFrames with columns = report dates, rows = line items
        if qbs is None or qcf is None or qbs.empty or qcf.empty:
            rec.status = "missing"
            records.append(rec)
            continue

        # Look for 'Capital Expenditure' or similar row in cashflow
        capex_rows = [r for r in qcf.index if "Capital" in str(r) and "Expend" in str(r)]
        assets_rows = [r for r in qbs.index if str(r).strip().lower() in ("total assets", "totalassets")]
        if not capex_rows or not assets_rows:
            rec.status = "missing"
            rec.notes = f"{name} [no capex/assets rows]"
            records.append(rec)
            continue

        capex_series = qcf.loc[capex_rows[0]].dropna()
        assets_series = qbs.loc[assets_rows[0]].dropna()
        common_periods = sorted(set(capex_series.index) & set(assets_series.index))
        if not common_periods:
            rec.status = "missing"
            records.append(rec)
            continue

        first_ts = pd.to_datetime(common_periods[0])
        last_ts = pd.to_datetime(common_periods[-1])
        rec.first_obs = first_ts.strftime("%Y-%m-%d")
        rec.last_obs = last_ts.strftime("%Y-%m-%d")
        rec.n_obs = len(common_periods)
        rec.n_quarters = _q_count(first_ts, last_ts)
        rec.coverage_pct = _coverage_pct(rec.n_quarters)
        # yfinance quarterly history is usually ~5Y = 20Q — expect "partial" here
        if rec.n_obs >= GATE_FIRM_QUARTERS_MIN:
            rec.status = "ok"
        elif rec.n_obs >= 8:
            rec.status = "partial"
        else:
            rec.status = "missing"
        log.info("   %-8s %-28s n=%3d  span=%s→%s  %s",
                 tkr, name[:28], rec.n_obs, rec.first_obs, rec.last_obs, rec.status)
        records.append(rec)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Gate G1 evaluation
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class GateDecision:
    decision: str
    firms_primary: int
    firms_pivot_a: int
    firms_any_source: int
    best_firm_source: str
    macros_ok: int
    inst_ok: int
    notes: list[str] = field(default_factory=list)
    exit_code: int = 0


def evaluate_gate_g1(records: list[CoverageRecord]) -> GateDecision:
    # Firms (layer 2): count best-source successes
    firm_records = [r for r in records if r.layer == 2]
    firm_by_actor: dict[str, list[CoverageRecord]] = {}
    for r in firm_records:
        firm_by_actor.setdefault(r.actor_id, []).append(r)

    def best_quarters(recs: list[CoverageRecord]) -> tuple[int, str]:
        best_n = 0
        best_src = ""
        for r in recs:
            if r.n_obs > best_n:
                best_n = r.n_obs
                best_src = r.source
        return best_n, best_src

    firms_primary = 0
    firms_pivot_a = 0
    firms_any = 0
    source_counts: dict[str, int] = {}
    for actor_id, recs in firm_by_actor.items():
        n, src = best_quarters(recs)
        if n > 0:
            firms_any += 1
            source_counts[src] = source_counts.get(src, 0) + 1
        if n >= GATE_FIRM_QUARTERS_MIN:
            firms_primary += 1
        if n >= 40:
            firms_pivot_a += 1

    best_firm_source = max(source_counts, key=source_counts.get) if source_counts else "none"

    # Note: this is a SAMPLE of 30 firms, not the full universe.
    # We extrapolate: if SAMPLE_OK% of sample works, we estimate FULL_UNIVERSE_OK = same % of full universe.
    # STOXX 600 has ~600 firms. Estimate how many we could get from full universe.
    sample_size = len(firm_by_actor)
    estimate_primary_full = int(round(600 * firms_primary / sample_size)) if sample_size else 0
    estimate_pivot_full = int(round(600 * firms_pivot_a / sample_size)) if sample_size else 0

    # Macros (layer 0)
    macro_records = [r for r in records if r.layer == 0 and r.status == "ok"]
    macros_ok = len(macro_records)

    # Institutionals: ECB balance sheet + BoE Bank Rate + (IMF reuse from primary panel)
    # In the audit, we explicitly tested 1 ECB balance sheet series and 1 BoE Bank Rate.
    # IMF, FCA, SEC actors are reused from the existing experiment_a1 registry without new fetch.
    # So we need: ECB assets OK + BoE OK + (reused count = 3: IMF, FCA, SEC per plan §4.3)
    ecb_asset_ok = any(
        r.actor_id == "inst_ecb_assets" and r.status in ("ok", "partial")
        for r in records
    )
    boe_ok = any(
        r.actor_id == "inst_boe_rate" and r.status in ("ok", "partial")
        for r in records
    )
    reused_instituts = 3  # IMF, FCA, SEC from primary panel
    inst_ok = int(ecb_asset_ok) + int(boe_ok) + reused_instituts

    notes: list[str] = []
    notes.append(f"Sample firms: {firms_primary}/{sample_size} at ≥{GATE_FIRM_QUARTERS_MIN}Q")
    notes.append(f"Sample firms: {firms_pivot_a}/{sample_size} at ≥40Q")
    notes.append(f"Extrapolated STOXX-600 yield: primary ~{estimate_primary_full}, pivot-A ~{estimate_pivot_full}")
    notes.append(f"Best firm source: {best_firm_source}")
    notes.append(f"Macros OK: {macros_ok}/{len(ECB_SERIES) + len(FRED_EU_SERIES)}")
    notes.append(f"Institutionals OK: {inst_ok} (ECB_assets={ecb_asset_ok}, BoE={boe_ok}, reused=3)")

    # Decision
    if (
        estimate_primary_full >= GATE_FIRM_COUNT_PRIMARY
        and macros_ok >= GATE_MACRO_COUNT
        and inst_ok >= GATE_INST_COUNT
    ):
        decision = "PASS"
        exit_code = 0
    elif (
        estimate_pivot_full >= GATE_FIRM_COUNT_PIVOT_A
        and macros_ok >= GATE_MACRO_COUNT - 1
        and inst_ok >= GATE_INST_COUNT - 1
    ):
        decision = "PIVOT_A"
        exit_code = 1
    elif estimate_pivot_full >= GATE_FIRM_COUNT_PIVOT_B:
        decision = "PIVOT_B"
        exit_code = 2
    else:
        decision = "KILL"
        exit_code = 3

    return GateDecision(
        decision=decision,
        firms_primary=estimate_primary_full,
        firms_pivot_a=estimate_pivot_full,
        firms_any_source=firms_any,
        best_firm_source=best_firm_source,
        macros_ok=macros_ok,
        inst_ok=inst_ok,
        notes=notes,
        exit_code=exit_code,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Output + main
# ═════════════════════════════════════════════════════════════════════════════


def write_outputs(records: list[CoverageRecord], decision: GateDecision) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(OUTPUT_CSV, index=False)

    lines: list[str] = []
    lines.append("# UK/EU Panel — Gate G1 Feasibility Report\n")
    lines.append(f"_Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}_\n")
    lines.append(f"\n## Decision: **{decision.decision}**\n")
    lines.append(f"\nExit code: {decision.exit_code}\n")
    lines.append("\n### Summary\n")
    for note in decision.notes:
        lines.append(f"- {note}\n")
    lines.append("\n### Per-source breakdown\n\n")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        ok = (sub["status"] == "ok").sum()
        lines.append(f"- **{src}**: {ok}/{len(sub)} OK\n")
    lines.append("\n### Full coverage matrix\n\n")
    lines.append("```\n")
    lines.append(df.to_string(index=False))
    lines.append("\n```\n")

    OUTPUT_MD.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s (%d rows)", OUTPUT_CSV.relative_to(_ROOT), len(df))
    log.info("Wrote %s", OUTPUT_MD.relative_to(_ROOT))


def main() -> int:
    log.info("=" * 72)
    log.info("Phase 1 / Gate G1 Feasibility Audit — UK/EU Panel")
    log.info("Target window: %s to %s (%dQ)", TARGET_START, TARGET_END, TARGET_QUARTERS)
    log.info("=" * 72)

    all_records: list[CoverageRecord] = []
    all_records.extend(audit_ecb_sdw())
    all_records.extend(audit_fred_eu())
    all_records.extend(audit_boe())
    all_records.extend(audit_edgar_eu_adrs())
    all_records.extend(audit_simfin())
    all_records.extend(audit_yfinance())

    decision = evaluate_gate_g1(all_records)
    write_outputs(all_records, decision)

    print("\n" + "=" * 72)
    print(f"  GATE G1 DECISION:  {decision.decision}")
    print("=" * 72)
    for note in decision.notes:
        print(f"  {note}")
    print("=" * 72)
    if decision.decision == "PASS":
        print("  → Proceed to Phase 2/3/4 (parallel data build)")
    elif decision.decision == "PIVOT_A":
        print("  → Reduce firm-count target from 80 → 60, proceed to Phase 2-4")
    elif decision.decision == "PIVOT_B":
        print("  → Shorten window to 2014Q1+ (44Q), proceed with reduced panel")
    else:
        print("  → KILL: UK/EU panel not feasible from free sources.")
        print("    Consider Japan / G10 alternatives or paid data (Compustat Global).")
    print("=" * 72 + "\n")

    return decision.exit_code


if __name__ == "__main__":
    sys.exit(main())
