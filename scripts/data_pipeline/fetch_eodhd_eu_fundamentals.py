"""Fetch quarterly fundamentals for EU-listed firms from EODHD.

Pulls Balance Sheet and Cash Flow quarterly data for firms on target
European exchanges. Stores raw data in parquet files for downstream
normalization into the experiment_b1 panel.

Usage:
    # Smoke test (20 firms)
    EODHD_API_KEY=xxx uv run python scripts/data_pipeline/fetch_eodhd_eu_fundamentals.py --smoke

    # Full pull (all common stocks on target exchanges)
    EODHD_API_KEY=xxx uv run python scripts/data_pipeline/fetch_eodhd_eu_fundamentals.py

    # Limit to N firms per exchange
    EODHD_API_KEY=xxx uv run python scripts/data_pipeline/fetch_eodhd_eu_fundamentals.py --limit 50

Outputs:
    data/raw/eodhd/tickers_{exchange}.parquet    — ticker lists per exchange
    data/raw/eodhd/fundamentals_quarterly.parquet — all quarterly BS + CF data
    data/raw/eodhd/firm_metadata.parquet          — firm general info (name, sector, country)
    data/raw/eodhd/coverage_summary.csv           — per-firm coverage stats
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw" / "eodhd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eodhd_fetch")

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_EXCHANGES = {
    "LSE":   "London Stock Exchange",
    "XETRA": "Frankfurt / Xetra",
    "PA":    "Euronext Paris",
    "AS":    "Euronext Amsterdam",
    "SW":    "SIX Swiss Exchange",
    "MI":    "Borsa Italiana",
    "MC":    "Bolsa de Madrid",
    "ST":    "Stockholm Exchange",
    "OL":    "Oslo Stock Exchange",
    "HE":    "Helsinki Exchange",
    "CO":    "Copenhagen Exchange",
    "VI":    "Vienna Exchange",
}

EODHD_BASE = "https://eodhd.com/api"
HTTP_TIMEOUT = 60
DELAY_BETWEEN_CALLS = 0.12  # ~8 req/s, well within limits

# Fields to extract from quarterly statements
BS_FIELDS = [
    "totalAssets",
    "totalCurrentAssets",
    "totalNonCurrentAssets",
    "totalLiab",
    "totalStockholderEquity",
    "propertyPlantEquipment",
    "intangibleAssets",
    "goodWill",
    "longTermDebt",
    "shortTermDebt",
    "cash",
    "netReceivables",
    "inventory",
]

CF_FIELDS = [
    "capitalExpenditures",
    "totalCashFromOperatingActivities",
    "totalCashflowsFromInvestingActivities",
    "totalCashFromFinancingActivities",
    "freeCashFlow",
    "netIncome",
    "depreciation",
    "changeInWorkingCapital",
]

IS_FIELDS = [
    "totalRevenue",
    "grossProfit",
    "operatingIncome",
    "ebit",
    "ebitda",
    "netIncome",
    "researchDevelopment",
]

# Smoke test tickers (verified to have data)
SMOKE_TICKERS = [
    ("SHEL", "LSE"), ("BP", "LSE"), ("AZN", "LSE"), ("HSBA", "LSE"),
    ("GSK", "LSE"), ("ULVR", "LSE"), ("DGE", "LSE"), ("BARC", "LSE"),
    ("SAP", "XETRA"), ("SIE", "XETRA"), ("BAS", "XETRA"), ("BAYN", "XETRA"),
    ("BMW", "XETRA"), ("ALV", "XETRA"),
    ("TTE", "PA"), ("BNP", "PA"), ("MC", "PA"), ("SAN", "PA"),
    ("NESN", "SW"), ("NOVN", "SW"),
]


# ── API helpers ───────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("EODHD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EODHD_API_KEY env var not set. "
            "Get a key at https://eodhd.com/register"
        )
    return key


def _api_get(session: requests.Session, endpoint: str, params: dict | None = None) -> dict | list | None:
    url = f"{EODHD_BASE}/{endpoint}"
    all_params = {"fmt": "json"}
    if params:
        all_params.update(params)
    try:
        r = session.get(url, params=all_params, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        log.warning("Request failed: %s — %s", endpoint, e)
        return None
    if r.status_code == 429:
        log.warning("Rate limited on %s, sleeping 5s", endpoint)
        time.sleep(5)
        try:
            r = session.get(url, params=all_params, timeout=HTTP_TIMEOUT)
        except requests.RequestException:
            return None
    if r.status_code != 200:
        log.debug("HTTP %s for %s", r.status_code, endpoint)
        return None
    try:
        return r.json()
    except Exception:
        return None


# ── Step 1: Fetch exchange ticker lists ───────────────────────────────────────

def fetch_exchange_tickers(session: requests.Session, exchange: str) -> pd.DataFrame:
    log.info("Fetching ticker list for %s (%s)", exchange, TARGET_EXCHANGES.get(exchange, ""))
    data = _api_get(session, f"exchange-symbol-list/{exchange}")
    if not data or not isinstance(data, list):
        log.warning("No tickers for %s", exchange)
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Filter to common stocks
    if "Type" in df.columns:
        df = df[df["Type"] == "Common Stock"].copy()
    log.info("  %s: %d common stocks", exchange, len(df))
    time.sleep(DELAY_BETWEEN_CALLS)
    return df


# ── Step 2: Fetch fundamentals for one firm ───────────────────────────────────

def fetch_fundamentals(session: requests.Session, ticker: str, exchange: str) -> dict | None:
    endpoint = f"fundamentals/{ticker}.{exchange}"
    return _api_get(session, endpoint)


def extract_general_info(raw: dict) -> dict:
    gen = raw.get("General", {}) or {}
    return {
        "name": gen.get("Name", ""),
        "country": gen.get("CountryISO", gen.get("Country", "")),
        "currency": gen.get("CurrencyCode", gen.get("CurrencySymbol", "")),
        "exchange": gen.get("Exchange", ""),
        "sector": gen.get("GicsSector", gen.get("Sector", "")),
        "industry": gen.get("GicsIndustry", gen.get("Industry", "")),
        "isin": gen.get("ISIN", ""),
        "market_cap": gen.get("MarketCapitalization", None),
    }


def extract_quarterly_statements(raw: dict, ticker: str, exchange: str) -> pd.DataFrame:
    fin = raw.get("Financials", {}) or {}
    rows: list[dict] = []

    for stmt_name, stmt_key, fields in [
        ("BS", "Balance_Sheet", BS_FIELDS),
        ("CF", "Cash_Flow", CF_FIELDS),
        ("IS", "Income_Statement", IS_FIELDS),
    ]:
        stmt = fin.get(stmt_key, {}) or {}
        quarterly = stmt.get("quarterly", {}) or {}
        currency = stmt.get("currency_symbol", "")
        for date_key, values in quarterly.items():
            if not isinstance(values, dict):
                continue
            row = {
                "ticker": ticker,
                "exchange": exchange,
                "statement": stmt_name,
                "date": date_key,
                "filing_date": values.get("filing_date", ""),
                "currency": values.get("currency_symbol", currency),
            }
            for field in fields:
                val = values.get(field)
                if val is not None and val != "None" and val != "":
                    try:
                        row[field] = float(val)
                    except (ValueError, TypeError):
                        row[field] = None
                else:
                    row[field] = None
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Step 3: Coverage analysis ─────────────────────────────────────────────────

def compute_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    bs = df[df["statement"] == "BS"].copy()
    cf = df[df["statement"] == "CF"].copy()

    def _summarize(sub: pd.DataFrame, prefix: str) -> dict:
        if sub.empty:
            return {f"{prefix}_n": 0, f"{prefix}_first": "", f"{prefix}_last": ""}
        dates = sorted(sub["date"].dropna().unique())
        return {
            f"{prefix}_n": len(dates),
            f"{prefix}_first": dates[0] if dates else "",
            f"{prefix}_last": dates[-1] if dates else "",
        }

    records = []
    for (ticker, exchange), group in df.groupby(["ticker", "exchange"]):
        bs_sub = group[group["statement"] == "BS"]
        cf_sub = group[group["statement"] == "CF"]
        has_assets = bs_sub["totalAssets"].notna().sum() if "totalAssets" in bs_sub.columns else 0
        has_capex = cf_sub["capitalExpenditures"].notna().sum() if "capitalExpenditures" in cf_sub.columns else 0
        rec = {
            "ticker": ticker,
            "exchange": exchange,
            **_summarize(bs_sub, "bs"),
            **_summarize(cf_sub, "cf"),
            "n_assets": int(has_assets),
            "n_capex": int(has_capex),
        }
        # Coverage in our target window (2011-2025)
        bs_in_window = bs_sub[(bs_sub["date"] >= "2011-01-01") & (bs_sub["date"] <= "2025-12-31")]
        cf_in_window = cf_sub[(cf_sub["date"] >= "2011-01-01") & (cf_sub["date"] <= "2025-12-31")]
        assets_in_window = bs_in_window["totalAssets"].notna().sum() if "totalAssets" in bs_in_window.columns else 0
        capex_in_window = cf_in_window["capitalExpenditures"].notna().sum() if "capitalExpenditures" in cf_in_window.columns else 0
        rec["n_assets_2011_2025"] = int(assets_in_window)
        rec["n_capex_2011_2025"] = int(capex_in_window)
        rec["both_60q"] = int(min(assets_in_window, capex_in_window) >= 56)
        records.append(rec)

    return pd.DataFrame(records).sort_values(["exchange", "ticker"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EODHD EU fundamentals")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 20 known tickers only")
    parser.add_argument("--limit", type=int, default=0, help="Max firms per exchange (0=all)")
    parser.add_argument("--exchanges", nargs="+", default=list(TARGET_EXCHANGES.keys()),
                        help="Exchange codes to fetch")
    args = parser.parse_args()

    api_key = _get_api_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.params = {"api_token": api_key}  # type: ignore[assignment]
    session.headers.update({"Accept": "application/json"})

    FLUSH_EVERY = 200

    # ── Load existing data (cache-aware resume) ───────────────────────────────

    fund_path = RAW_DIR / "fundamentals_quarterly.parquet"
    meta_path = RAW_DIR / "firm_metadata.parquet"

    existing_fund: pd.DataFrame | None = None
    existing_meta: pd.DataFrame | None = None
    cached_tickers: set[tuple[str, str]] = set()

    if fund_path.exists() and not args.smoke:
        existing_fund = pd.read_parquet(fund_path)
        cached_tickers = set(
            zip(existing_fund["ticker"], existing_fund["exchange"])
        )
        log.info("Loaded %d cached firms from %s — will skip these",
                 len(cached_tickers), fund_path.name)
    if meta_path.exists() and not args.smoke:
        existing_meta = pd.read_parquet(meta_path)

    new_fundamentals: list[pd.DataFrame] = []
    new_metadata: list[dict] = []
    total_fetched = 0
    total_skipped = 0
    total_cached = 0

    if args.smoke:
        log.info("=" * 60)
        log.info("SMOKE TEST -- 20 known tickers")
        log.info("=" * 60)
        tickers_to_fetch = SMOKE_TICKERS
    else:
        log.info("=" * 60)
        log.info("FULL PULL -- exchanges: %s", args.exchanges)
        log.info("=" * 60)
        tickers_to_fetch = []
        for exchange in args.exchanges:
            if exchange not in TARGET_EXCHANGES:
                log.warning("Unknown exchange: %s, skipping", exchange)
                continue
            # Reuse saved ticker list if available
            ticker_path = RAW_DIR / f"tickers_{exchange}.parquet"
            if ticker_path.exists():
                ticker_df = pd.read_parquet(ticker_path)
                if "Type" in ticker_df.columns:
                    ticker_df = ticker_df[ticker_df["Type"] == "Common Stock"]
                log.info("  %s: loaded %d common stocks from cache", exchange, len(ticker_df))
            else:
                ticker_df = fetch_exchange_tickers(session, exchange)
                if ticker_df.empty:
                    continue
                ticker_df.to_parquet(ticker_path, index=False)
                log.info("  %s: fetched and saved %d common stocks", exchange, len(ticker_df))

            code_col = "Code" if "Code" in ticker_df.columns else ticker_df.columns[0]
            codes = ticker_df[code_col].tolist()
            if args.limit > 0:
                codes = codes[:args.limit]
            tickers_to_fetch.extend([(c, exchange) for c in codes])

    log.info("Total tickers to process: %d (cached: %d)",
             len(tickers_to_fetch),
             sum(1 for t, e in tickers_to_fetch if (t, e) in cached_tickers))

    def _flush_to_disk() -> None:
        """Merge new data with existing and write to disk."""
        nonlocal existing_fund, existing_meta
        if not new_fundamentals:
            return
        new_df = pd.concat(new_fundamentals, ignore_index=True)
        if existing_fund is not None and not existing_fund.empty:
            merged = pd.concat([existing_fund, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_parquet(fund_path, index=False)
        existing_fund = merged

        if new_metadata:
            new_meta_df = pd.DataFrame(new_metadata)
            if existing_meta is not None and not existing_meta.empty:
                merged_meta = pd.concat([existing_meta, new_meta_df], ignore_index=True)
            else:
                merged_meta = new_meta_df
            merged_meta.to_parquet(meta_path, index=False)
            existing_meta = merged_meta

        log.info("  [FLUSH] Saved %d total rows (%d firms) to disk",
                 len(merged), merged[["ticker", "exchange"]].drop_duplicates().shape[0])
        new_fundamentals.clear()
        new_metadata.clear()

    for i, (ticker, exchange) in enumerate(tickers_to_fetch):
        # Skip cached
        if (ticker, exchange) in cached_tickers:
            total_cached += 1
            continue

        if (total_fetched + total_skipped) % 50 == 0 and (total_fetched + total_skipped) > 0:
            log.info("Progress: %d / %d (fetched=%d, skipped=%d, cached=%d)",
                     i + 1, len(tickers_to_fetch), total_fetched, total_skipped, total_cached)

        raw = fetch_fundamentals(session, ticker, exchange)
        if not raw or not isinstance(raw, dict):
            total_skipped += 1
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        gen = extract_general_info(raw)
        gen["ticker"] = ticker
        gen["exchange_code"] = exchange
        new_metadata.append(gen)

        stmt_df = extract_quarterly_statements(raw, ticker, exchange)
        if not stmt_df.empty:
            new_fundamentals.append(stmt_df)
            total_fetched += 1
            cached_tickers.add((ticker, exchange))
        else:
            total_skipped += 1

        # Periodic flush
        if total_fetched > 0 and total_fetched % FLUSH_EVERY == 0:
            _flush_to_disk()

        time.sleep(DELAY_BETWEEN_CALLS)

    # Final flush
    _flush_to_disk()
    log.info("Fetch complete: fetched=%d, skipped=%d, cached=%d",
             total_fetched, total_skipped, total_cached)

    # ── Reload final state for coverage analysis ──────────────────────────────

    if fund_path.exists():
        fundamentals_df = pd.read_parquet(fund_path)
    else:
        fundamentals_df = pd.DataFrame()
        log.warning("No fundamentals data on disk")

    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        log.info("Metadata on disk: %d firms", len(meta_df))

    # ── Coverage analysis ─────────────────────────────────────────────────────

    if not fundamentals_df.empty:
        coverage = compute_coverage(fundamentals_df)
        out_path = RAW_DIR / "coverage_summary.csv"
        coverage.to_csv(out_path, index=False)

        n_both_60q = coverage["both_60q"].sum()
        n_with_assets = (coverage["n_assets_2011_2025"] > 0).sum()
        n_with_capex = (coverage["n_capex_2011_2025"] > 0).sum()

        print("\n" + "=" * 70)
        print("EODHD EU FUNDAMENTALS — FETCH SUMMARY")
        print("=" * 70)
        print(f"  Firms fetched:               {total_fetched}")
        print(f"  Firms skipped (no data):      {total_skipped}")
        print(f"  Firms with any Assets data:   {n_with_assets}")
        print(f"  Firms with any CapEx data:    {n_with_capex}")
        print(f"  Firms with both >=56Q (2011-2025): {n_both_60q}")
        print(f"  Output: {RAW_DIR.relative_to(_ROOT)}/")
        if n_both_60q >= 80:
            print(f"\n  >> GATE G1 THRESHOLD MET: {n_both_60q} >= 80 firms")
        elif n_both_60q >= 60:
            print(f"\n  >> PIVOT A: {n_both_60q} firms (60-79 range)")
        else:
            print(f"\n  >> BELOW THRESHOLD: {n_both_60q} firms")

        # Top firms by coverage
        top = coverage.nlargest(20, "n_capex_2011_2025")
        print("\nTop 20 firms by CapEx coverage (2011-2025):")
        for _, r in top.iterrows():
            print(f"  {r['exchange']:<6} {r['ticker']:<8} assets={r['n_assets_2011_2025']:>3}Q  "
                  f"capex={r['n_capex_2011_2025']:>3}Q  range={r['bs_first']}->{r['bs_last']}")

        # Coverage by exchange
        print("\nCoverage by exchange:")
        for ex in sorted(coverage["exchange"].unique()):
            sub = coverage[coverage["exchange"] == ex]
            n60 = sub["both_60q"].sum()
            print(f"  {ex:<6}: {len(sub):>4} firms total, {n60:>3} with >=56Q both")

        print("=" * 70 + "\n")
        log.info("Saved %s (%d rows)", out_path.name, len(coverage))


if __name__ == "__main__":
    main()
