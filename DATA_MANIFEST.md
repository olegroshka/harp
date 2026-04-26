# Data Manifest

**Last updated:** 2026-04-25
**Project:** HARP — UK/EU Panel Extension (experiment_b1)

All paths are relative to the harp project root (`C:\Users\olegr\PycharmProjects\harp\`).

---

## 1. Primary Panel (experiment_a1) — Pre-existing

These files shipped with the original HARP replication package. **Do not modify.**

| File | Format | Rows | Description |
|------|--------|------|-------------|
| `data/intensities/experiment_a1_intensities.parquet` | parquet (long) | ~7,800 | 93 actors x 84 quarters (2005Q1-2025Q4). Columns: `actor_id`, `period`, `intensity_value`, `normalisation_method`. Layer 0 macros are min-max normalised; Layer 2 firms are cross-sectional percentile ranks of CapEx/Assets. |
| `data/registries/experiment_a1_registry.json` | JSON | 93 actors | Actor registry. Per actor: `actor_id`, `name`, `actor_type`, `layer` (0/1/2), `geography`, `sector`, `external_ids`. |
| `data/processed/edgar_balance_sheet.parquet` | parquet | — | SEC EDGAR balance sheet data for 146-firm and 270-actor cross-panel tests. |

---

## 2. EODHD Raw Data — Fetched 2026-04-25

**Source:** EODHD All-in-One API (`https://eodhd.com/api/fundamentals/{TICKER}.{EXCHANGE}`)
**API key:** env var `EODHD_API_KEY` (All-in-One subscription, active until ~2026-05-25)
**Fetch script:** `scripts/data_pipeline/fetch_eodhd_eu_fundamentals.py`
**API calls consumed:** ~5,939 fundamentals + 7 ticker-list = ~5,946 calls
**Exchanges covered:** LSE, XETRA, PA (Paris), AS (Amsterdam), SW (Swiss), MC (Madrid). MI (Milan) ticker list failed — 0 Italian firms.

### 2.1 Ticker Lists

| File | Rows | Description |
|------|------|-------------|
| `data/raw/eodhd/tickers_LSE.parquet` | 3,994 | LSE common stocks. Columns: `Code`, `Name`, `Country`, `Exchange`, `Currency`, `Type`, `Isin`. |
| `data/raw/eodhd/tickers_XETRA.parquet` | 727 | Frankfurt/Xetra common stocks. Same schema. |
| `data/raw/eodhd/tickers_PA.parquet` | 640 | Euronext Paris common stocks. |
| `data/raw/eodhd/tickers_AS.parquet` | 112 | Euronext Amsterdam common stocks. |
| `data/raw/eodhd/tickers_SW.parquet` | 228 | SIX Swiss Exchange common stocks. |
| `data/raw/eodhd/tickers_MC.parquet` | 238 | Bolsa de Madrid common stocks. |
| *(missing)* `tickers_MI.parquet` | — | Borsa Italiana — ticker list API returned empty. Exchange code "MI" may be wrong; needs investigation if Italian firms are needed. |

### 2.2 Quarterly Fundamentals

| File | Size | Rows | Firms | Description |
|------|------|------|-------|-------------|
| `data/raw/eodhd/fundamentals_quarterly.parquet` | 33.7 MB | 981,328 | 5,581 | All quarterly Balance Sheet, Cash Flow, and Income Statement data for 5,581 EU-listed firms. |

**Schema (33 columns):**

| Column | Type | Source statement | Description |
|--------|------|-----------------|-------------|
| `ticker` | str | — | Firm ticker (e.g., "SHEL", "SAP") |
| `exchange` | str | — | Exchange code (e.g., "LSE", "XETRA") |
| `statement` | str | — | "BS" (Balance Sheet), "CF" (Cash Flow), or "IS" (Income Statement) |
| `date` | str | — | Quarter-end date, YYYY-MM-DD |
| `filing_date` | str | — | Filing/reporting date |
| `currency` | str | — | Reporting currency (e.g., "USD", "GBP", "EUR") |
| `totalAssets` | float | BS | **Key field** — denominator for CapEx/Assets intensity |
| `totalCurrentAssets` | float | BS | |
| `totalNonCurrentAssets` | float | BS | |
| `totalLiab` | float | BS | |
| `totalStockholderEquity` | float | BS | |
| `propertyPlantEquipment` | float | BS | |
| `intangibleAssets` | float | BS | |
| `goodWill` | float | BS | |
| `longTermDebt` | float | BS | |
| `shortTermDebt` | float | BS | |
| `cash` | float | BS | |
| `netReceivables` | float | BS | |
| `inventory` | float | BS | |
| `capitalExpenditures` | float | CF | **Key field** — numerator for CapEx/Assets intensity |
| `totalCashFromOperatingActivities` | float | CF | |
| `totalCashflowsFromInvestingActivities` | float | CF | |
| `totalCashFromFinancingActivities` | float | CF | |
| `freeCashFlow` | float | CF | |
| `netIncome` | float | CF/IS | |
| `depreciation` | float | CF | |
| `changeInWorkingCapital` | float | CF | |
| `totalRevenue` | float | IS | |
| `grossProfit` | float | IS | |
| `operatingIncome` | float | IS | |
| `ebit` | float | IS | |
| `ebitda` | float | IS | |
| `researchDevelopment` | float | IS | |

### 2.3 Firm Metadata

| File | Rows | Description |
|------|------|-------------|
| `data/raw/eodhd/firm_metadata.parquet` | 5,939 | General info for all tickers (including those with no fundamentals). |

**Schema (10 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Firm ticker |
| `exchange_code` | str | Exchange code |
| `name` | str | Company name |
| `country` | str | ISO country code (GB, DE, FR, NL, CH, ES) |
| `currency` | str | Reporting currency |
| `exchange` | str | Exchange name (human-readable) |
| `sector` | str | GICS sector (13 unique values) |
| `industry` | str | GICS industry |
| `isin` | str | ISIN identifier |
| `market_cap` | float | Market capitalization (may be stale) |

### 2.4 Coverage Summary

| File | Rows | Description |
|------|------|-------------|
| `data/raw/eodhd/coverage_summary.csv` | 5,695 | Per-firm coverage stats. |

**Schema (13 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | |
| `exchange` | str | |
| `bs_n` | int | Total quarterly BS observations |
| `bs_first` | str | Earliest BS date |
| `bs_last` | str | Latest BS date |
| `cf_n` | int | Total quarterly CF observations |
| `cf_first` | str | Earliest CF date |
| `cf_last` | str | Latest CF date |
| `n_assets` | int | Non-null totalAssets count (all time) |
| `n_capex` | int | Non-null capitalExpenditures count (all time) |
| `n_assets_2011_2025` | int | Non-null totalAssets in target window |
| `n_capex_2011_2025` | int | Non-null capitalExpenditures in target window |
| `both_60q` | int | 1 if min(n_assets, n_capex) in 2011-2025 >= 56 |

**Key coverage stats:**

| Metric | Value |
|--------|-------|
| Total firms with any data | 5,695 |
| Firms with Assets data (2011-2025) | 5,369 |
| Firms with CapEx data (2011-2025) | 5,554 |
| **Firms with both >= 56Q (2011-2025)** | **1,202** |

**By exchange:**

| Exchange | Total firms | >= 56Q both |
|----------|------------|-------------|
| LSE | 3,868 | 762 |
| XETRA | 697 | 329 |
| MC | 218 | 35 |
| PA | 585 | 30 |
| SW | 219 | 24 |
| AS | 108 | 22 |

### 2.5 Daily Prices (fetched 2026-04-25, in progress)

| File | Size | Description |
|------|------|-------------|
| `data/raw/eodhd/prices_daily.parquet` | ~200+ MB (est.) | Daily OHLCV for all 1,202 qualifying firms (>=56Q both). Columns: `date`, `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, `ticker`, `exchange`. Date range: 2005-01-01 to 2025-12-31. |

**Script:** `scripts/data_pipeline/fetch_eodhd_eu_prices.py`
**API calls:** ~1,202 (one per firm). Cache-aware + flush every 100.

### 2.6 EU Macro Data (fetched 2026-04-25)

| File | Rows | Series | Description |
|------|------|--------|-------------|
| `data/raw/eodhd/macro_eu.parquet` | 30,398 | 13 | EU macro indicators from EODHD + FRED combined |
| `data/raw/eodhd/macro_coverage.csv` | 16 | — | Per-series coverage stats |

**Script:** `scripts/data_pipeline/fetch_eodhd_macro.py`
**API calls:** 9 (EODHD) + 7 (FRED) = 16 total.

**Series coverage:**

| Source | Actor ID | Observations | Range | Status |
|--------|----------|-------------|-------|--------|
| EODHD | macro_eu_gdp_growth | 64 | 1961-2024 | OK (annual) |
| EODHD | macro_eu_inflation | 65 | 1960-2024 | OK (annual) |
| EODHD | macro_eu_unemployment | 0 | — | EMPTY |
| EODHD | macro_uk_gdp_growth | 64 | 1961-2024 | OK (annual) |
| EODHD | macro_uk_inflation | 65 | 1960-2024 | OK (annual) |
| EODHD | macro_uk_unemployment | 0 | — | EMPTY |
| EODHD | macro_de_gdp_growth | 64 | 1961-2024 | OK (annual) |
| EODHD | macro_de_industrial_prod | 0 | — | EMPTY |
| EODHD | macro_fr_gdp_growth | 64 | 1961-2024 | OK (annual) |
| FRED | shock_ecb_rate (ECBDFR) | 9,497 | 2000-2025 | OK (daily) |
| FRED | shock_eu_10y_yield | 312 | 2000-2025 | OK (monthly) |
| FRED | shock_brent (DCOILBRENTEU) | 6,599 | 2000-2025 | OK (daily) |
| FRED | shock_eu_hicp | 312 | 2000-2025 | OK (monthly) |
| FRED | shock_vix (VIXCLS) | 6,567 | 2000-2025 | OK (daily) |
| FRED | shock_eur_usd (DEXUSEU) | 6,518 | 2000-2025 | OK (daily) |
| FRED | shock_eu_m3 | 207 | 2000-2017 | OK but ends 2017 |

---

## 3. Data Quality Report (2026-04-25)

Quality analysis run on the 1,202 qualifying firms (>=56Q both CapEx + Assets in 2011-2025).

### 3.1 Strengths

| Check | Result |
|-------|--------|
| Duplicates | **Zero** — no duplicate (ticker, exchange, date) rows in BS or CF |
| CapEx missingness (2011-2025) | **0.0%** — perfect coverage for the key numerator field |
| totalAssets missingness | **3.3%** — very low |
| CapEx sign convention | **100% non-negative** (84.8% positive, 15.2% zero, 0% negative) — clean, no sign-flip needed |
| CapEx/Assets ratio range | Mean 1.1%, median 0.6%, P5 0.0%, P95 3.7% — realistic |
| Ratio outliers (>50%) | **Only 6 rows** out of 73,047 — negligible |

### 3.2 Issues Requiring Attention

#### CRITICAL: GICS Sector Labels — 43% Missing

513 of 1,202 qualifying firms (43%) have **empty sector labels** in EODHD metadata. The block partition for the mixture architecture requires sector assignments. Options:
- Source sectors from ISIN -> GICS mapping via a free reference (e.g., Wikipedia constituent lists, OpenFIGI)
- Use EODHD's `industry` field as fallback (different granularity)
- Accept reduced universe: 689 firms with sectors, still >> 80 target

**Distribution of firms WITH sectors:**

| Sector | Firms |
|--------|-------|
| Financial Services | 178 |
| Industrials | 110 |
| Technology | 80 |
| Basic Materials | 79 |
| Consumer Cyclical | 73 |
| Healthcare | 63 |
| Communication Services | 43 |
| Energy | 37 |
| Consumer Defensive | 32 |
| Real Estate | 15 |
| Utilities | 14 |

11 sectors represented with >=14 firms each — sufficient for block partition even without the 513 empty-sector firms.

#### MODERATE: Currency Diversity

30+ reporting currencies across qualifying firms. Top 5: USD (545), EUR (472), GBP (231), empty (183), GBX (51). Also CAD, CHF, SEK, NOK, CNY, KRW, JPY, etc.

**Impact:** None for the panel — CapEx/Assets ratio is dimensionless, and cross-sectional percentile ranks are currency-invariant. But 183 firms with empty currency is a metadata quality flag.

#### MODERATE: Non-Standard Fiscal Year-Ends

~16% of BS rows have quarter-end dates in non-standard months (1,2,4,5,7,8,10,11 instead of 3,6,9,12). These are firms with non-calendar fiscal years (common for UK firms: Jan/Jul year-end).

**Impact:** Need to align to calendar quarters when building the panel. Either: (a) map each firm's fiscal Q to the nearest calendar Q, or (b) use the fiscal quarter as-is and accept slight cross-sectional misalignment (standard practice in empirical finance).

#### LOW: Non-EU Domiciled Firms on EU Exchanges

545 firms report in USD — many are US-domiciled cross-listings on LSE/XETRA. The `country` field in metadata can filter these, but we should audit how many of the 1,202 qualifying firms are genuinely EU-domiciled vs foreign cross-listings.

#### INFO: Borsa Italiana Not Available

EODHD does not list Borsa Italiana (no exchange code found — tested MI, BIT, MIL, XMIL). Italian firms (ENI, Enel, Intesa Sanpaolo, etc.) are absent. Not a blocker given 1,202 qualifying firms from 6 other exchanges.

---

## 4. Feasibility Audit Outputs — 2026-04-15 (Historical)

These are artefacts from the initial Gate G1 audit (pre-EODHD). Retained for the decision record.

| File | Description |
|------|-------------|
| `data/audit/uk_eu_coverage_matrix.csv` | 54-row coverage matrix testing ECB SDW, FRED, BoE, EDGAR, SimFin, yfinance |
| `data/audit/uk_eu_feasibility_report.md` | Gate G1 FAIL report (pre-EODHD) |
| `data/audit/simfin_eu_coverage.csv` | All 142 SimFin EU-ISIN firms — documenting SimFin's abandoned EU coverage |

---

## 5. Experiment B1 Panel (built 2026-04-25)

| File | Rows | Description |
|------|------|-------------|
| `data/registries/experiment_b1_registry.json` | 109 actors | Actor registry matching experiment_a1 schema |
| `data/intensities/experiment_b1_intensities.parquet` | 6,420 | Long-format: actor_id, period, intensity_value, normalisation_method |
| `data/audit/experiment_b1_firm_selection.csv` | 100 | Firm selection audit trail with sector, exchange, size_proxy |

**Build script:** `scripts/data_pipeline/build_experiment_b1.py`

**Panel composition:**

| Layer | Actors | Method | Description |
|-------|--------|--------|-------------|
| 0 — Macro | 6 | fred_minmax | Brent, ECB rate, VIX, EU 10Y yield, EUR/USD, EU HICP |
| 1 — Inst | 3 | institutional_minmax | ECB (cumulative rate change), BoE (cumulative rate change), IMF/EU GDP growth |
| 2 — Firms | 100 | capex_assets_xsrank | CapEx/Assets cross-sectional percentile rank per quarter |
| **Total** | **109** | | |

**Window:** 2011Q2 to 2025Q4 (59 quarters)

**Firm selection criteria:**
- >=56Q both CapEx + Assets in 2011-2025
- Has GICS sector label (mapped to SMIM 6-sector scheme)
- Deduplicated by ISIN (one share class per company)
- Excluded LSE IOB cross-listings (ticker starting with "0")
- Sorted by latest totalAssets (size proxy, market_cap was unavailable)
- Geographic cap: no exchange >35% of panel
- Sector balanced: 16-17 firms per SMIM sector

**Geographic distribution:**

| Exchange | Firms | Country |
|----------|-------|---------|
| XETRA | 35 | Germany |
| ST | 26 | Sweden |
| LSE | 15 | UK |
| CO | 7 | Denmark |
| OL | 5 | Norway |
| PA | 3 | France |
| HE | 3 | Finland |
| MC | 3 | Spain |
| SW | 3 | Switzerland |

**Sector distribution:**

| SMIM sector | Firms |
|-------------|-------|
| consumer | 17 |
| energy | 17 |
| financials | 17 |
| healthcare | 16 |
| industrials | 17 |
| technology | 16 |

---

## 6. Data Lineage (updated)

```
EODHD API                              FRED API              BoE IADB
  |                                       |                     |
  v                                       v                     v
fetch_eodhd_eu_fundamentals.py     fetch_eodhd_macro.py    (inline in build_b1)
fetch_eodhd_eu_prices.py                  |                     |
  |                                       v                     |
  +-> data/raw/eodhd/              macro_eu.parquet             |
  |   tickers_*.parquet                   |                     |
  |   fundamentals_quarterly.parquet      |                     |
  |   firm_metadata.parquet               |                     |
  |   prices_daily.parquet                |                     |
  |   coverage_summary.csv                |                     |
  |                                       |                     |
  +---------------------------------------+---------------------+
                          |
                          v
              build_experiment_b1.py
                          |
  +-> data/registries/experiment_b1_registry.json
  +-> data/intensities/experiment_b1_intensities.parquet
  +-> data/audit/experiment_b1_firm_selection.csv
                          |
                          v               [NEXT STEPS]
              table5_uk_eu.py  (8 architectures)
              dm_hac_uk_eu.py  (DM-HAC inference)
              table7_placebo_uk_eu.py  (1000 random partitions)
```
  +-> compute CapEx/Assets ratio
  +-> cross-sectional percentile rank per quarter
  +-> filter to >=56Q balanced panel, >=6 GICS sectors
  +-> data/registries/experiment_b1_firm_layer.json
  |
  v
build_experiment_b1.py
  |
  +-> merge macro + institutional + firm layers
  +-> data/registries/experiment_b1_registry.json
  +-> data/intensities/experiment_b1_intensities.parquet
```

---

## 7. Licensing & Attribution

| Source | License | Redistribution | Attribution required |
|--------|---------|---------------|---------------------|
| EODHD | Commercial subscription (All-in-One) | **Derived data only** (CapEx/Assets ranks, not raw). Do not commit raw EODHD JSON or raw fundamentals_quarterly.parquet to public repos. | Yes — acknowledged as data source in paper, replication README, and Data Availability statement. Agreed with EODHD support (Levon V., 2026-04-17). |
| FRED | Public domain | Yes | Standard citation |
| BoE IADB | Public domain | Yes | Standard citation |
| SEC EDGAR | Public domain | Yes | Standard citation |

**Replication package should ship:** derived intensities parquet + registry JSON + the `fetch_eodhd_eu_fundamentals.py` script (so users can re-pull with their own EODHD key). Do **not** ship raw fundamentals_quarterly.parquet.
