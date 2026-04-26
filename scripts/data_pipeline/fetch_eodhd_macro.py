"""Fetch EU macroeconomic indicators from EODHD + FRED.

Probes EODHD macro API for EU indicators, then fills gaps with FRED.
Stores all macro data in a single parquet for downstream panel building.

Usage:
    EODHD_API_KEY=xxx uv run python scripts/data_pipeline/fetch_eodhd_macro.py

Outputs:
    data/raw/eodhd/macro_eu.parquet — all EU macro series (quarterly)
    data/raw/eodhd/macro_coverage.csv — per-series coverage stats
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw" / "eodhd"

EODHD_BASE = "https://eodhd.com/api"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
HTTP_TIMEOUT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eodhd_macro")

# Target macro indicators — EODHD country codes + indicator names
# See: https://eodhd.com/financial-apis/macroeconomics-data-api
EODHD_MACRO_SERIES = [
    # (actor_id, country, indicator, description)
    ("macro_eu_gdp_growth", "EMU", "gdp_growth_annual", "Euro area GDP growth"),
    ("macro_eu_inflation", "EMU", "inflation_consumer_prices_annual", "Euro area CPI inflation"),
    ("macro_eu_unemployment", "EMU", "unemployment_total", "Euro area unemployment rate"),
    ("macro_uk_gdp_growth", "GBR", "gdp_growth_annual", "UK GDP growth"),
    ("macro_uk_inflation", "GBR", "inflation_consumer_prices_annual", "UK CPI inflation"),
    ("macro_uk_unemployment", "GBR", "unemployment_total", "UK unemployment rate"),
    ("macro_de_gdp_growth", "DEU", "gdp_growth_annual", "Germany GDP growth"),
    ("macro_de_industrial_prod", "DEU", "industrial_production", "Germany industrial production"),
    ("macro_fr_gdp_growth", "FRA", "gdp_growth_annual", "France GDP growth"),
]

FRED_MACRO_SERIES = [
    # (actor_id, series_id, description)
    ("shock_ecb_rate", "ECBDFR", "ECB Deposit Facility Rate"),
    ("shock_eu_10y_yield", "IRLTLT01EZM156N", "Euro area long-term rate 10Y"),
    ("shock_brent", "DCOILBRENTEU", "Brent crude oil spot price"),
    ("shock_eu_hicp", "CP0000EZ19M086NEST", "Euro area HICP all items"),
    ("shock_vix", "VIXCLS", "CBOE VIX (global risk proxy)"),
    ("shock_eur_usd", "DEXUSEU", "EUR/USD exchange rate"),
    ("shock_eu_m3", "MYAGM3EZM196N", "Euro area M3 money supply"),
]


def fetch_eodhd_macro(session: requests.Session, country: str, indicator: str) -> pd.DataFrame:
    url = f"{EODHD_BASE}/macro-indicator/{country}"
    try:
        r = session.get(url, params={"indicator": indicator, "fmt": "json"}, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        log.warning("EODHD macro failed: %s/%s — %s", country, indicator, e)
        return pd.DataFrame()
    if r.status_code != 200:
        log.debug("EODHD macro HTTP %s: %s/%s", r.status_code, country, indicator)
        return pd.DataFrame()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df


def fetch_fred_series(api_key: str, series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2000-01-01",
        "observation_end": "2025-12-31",
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        log.warning("FRED failed: %s — %s", series_id, e)
        return pd.DataFrame()
    if r.status_code != 200:
        return pd.DataFrame()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame()
    df = pd.DataFrame(obs)
    df = df[df["value"] != "."]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "value"]].dropna()


def main() -> None:
    eodhd_key = os.environ.get("EODHD_API_KEY", "").strip()
    fred_key = os.environ.get("FRED_API_KEY", "").strip()

    if not eodhd_key:
        raise RuntimeError("EODHD_API_KEY not set")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.params = {"api_token": eodhd_key}

    all_series: list[pd.DataFrame] = []
    coverage: list[dict] = []

    # EODHD macro
    log.info("=== EODHD macro indicators (%d series) ===", len(EODHD_MACRO_SERIES))
    for actor_id, country, indicator, desc in EODHD_MACRO_SERIES:
        df = fetch_eodhd_macro(session, country, indicator)
        if df.empty:
            log.info("  %-30s EMPTY", actor_id)
            coverage.append({"actor_id": actor_id, "source": "eodhd", "n_obs": 0,
                             "first": "", "last": "", "description": desc})
            continue
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        val_col = "Value" if "Value" in df.columns else df.columns[-1]
        df = df.rename(columns={date_col: "date", val_col: "value"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["actor_id"] = actor_id
        df["source"] = "eodhd"
        all_series.append(df[["actor_id", "date", "value", "source"]])
        first_d = df["date"].min().strftime("%Y-%m-%d")
        last_d = df["date"].max().strftime("%Y-%m-%d")
        log.info("  %-30s n=%3d  %s -> %s", actor_id, len(df), first_d, last_d)
        coverage.append({"actor_id": actor_id, "source": "eodhd", "n_obs": len(df),
                         "first": first_d, "last": last_d, "description": desc})
        time.sleep(0.15)

    # FRED macro
    if fred_key:
        log.info("\n=== FRED macro indicators (%d series) ===", len(FRED_MACRO_SERIES))
        for actor_id, series_id, desc in FRED_MACRO_SERIES:
            df = fetch_fred_series(fred_key, series_id)
            if df.empty:
                log.info("  %-30s EMPTY", actor_id)
                coverage.append({"actor_id": actor_id, "source": "fred", "n_obs": 0,
                                 "first": "", "last": "", "description": desc})
                continue
            df["actor_id"] = actor_id
            df["source"] = "fred"
            all_series.append(df[["actor_id", "date", "value", "source"]])
            first_d = df["date"].min().strftime("%Y-%m-%d")
            last_d = df["date"].max().strftime("%Y-%m-%d")
            log.info("  %-30s n=%5d  %s -> %s", actor_id, len(df), first_d, last_d)
            coverage.append({"actor_id": actor_id, "source": "fred", "n_obs": len(df),
                             "first": first_d, "last": last_d, "description": desc})
            time.sleep(0.15)
    else:
        log.warning("FRED_API_KEY not set — skipping FRED series")

    # Save
    if all_series:
        macro_df = pd.concat(all_series, ignore_index=True)
        out_path = RAW_DIR / "macro_eu.parquet"
        macro_df.to_parquet(out_path, index=False)
        log.info("\nSaved %s (%d rows, %d series)", out_path.name,
                 len(macro_df), macro_df["actor_id"].nunique())

    cov_df = pd.DataFrame(coverage)
    cov_path = RAW_DIR / "macro_coverage.csv"
    cov_df.to_csv(cov_path, index=False)
    log.info("Saved %s", cov_path.name)

    print("\n" + "=" * 60)
    print("MACRO FETCH SUMMARY")
    print("=" * 60)
    for _, r in cov_df.iterrows():
        status = "OK" if r["n_obs"] > 0 else "EMPTY"
        print(f"  {r['source']:<6} {r['actor_id']:<30} n={r['n_obs']:>5}  [{status}]")
    n_ok = (cov_df["n_obs"] > 0).sum()
    print(f"\n  {n_ok}/{len(cov_df)} series with data")
    print("=" * 60)


if __name__ == "__main__":
    main()
