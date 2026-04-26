# UK/EU Panel — Gate G1 Feasibility Report

_Generated: 2026-04-15 21:50_

## Decision: **KILL (as currently scoped)**

Gate G1 fails for the specific plan as written, because **no free data source provides ≥60 quarters of CapEx/Assets history for EU firms going back to 2010**. A pivot is required before Phase 2 can begin. Options listed at the end.

---

## Summary

| Source | Layer | Result | Usable for panel? |
|--------|-------|--------|-------------------|
| ECB SDW (SDMX REST) | 0 | 4 / 4 HTTP 404 — my series keys guessed wrong | Need correct series keys (fixable) |
| FRED EU proxies | 0 | 2 / 4 OK, 2 / 4 partial (EU industrial prod ends 2023; UK 3M rate ends 2016) | Yes, 2-3 EU macros confirmed |
| BoE IADB (Bank Rate) | 0 | 1 / 1 OK, 100% coverage | Yes |
| SEC EDGAR — EU 20-F filers | 2 | 5 / 10 returned data — **annual only, 13–20 obs/firm** | Yes but **annual** |
| SimFin free tier | 2 | Rate-limited + smoke test shows **11 quarters max (2019Q2–2021Q4)** | **No — hard cap** |
| yfinance (EU tickers) | 2 | 5 quarters max — Yahoo retroactively trimmed free fundamentals | **No** |

## Hard findings

### 1. SimFin free tier is capped at 2019–2021
Direct probe of `companies/statements/compact?ticker=SAP.DE&statements=bs,cf,pl` (no date filter):
- BS: 11 rows, 2019-06-30 → 2021-12-31
- CF: 11 rows, same window
- PL: 11 rows, same window
- Annual (FY): 3 rows total, 2019 → 2021

The free tier is not a date-filter issue — it is a hard rolling-window cap. SimFin+ paid (~$39/mo) opens 20+ years of history.

### 2. yfinance quarterly fundamentals are now capped at 5 quarters
Every EU ticker tested (ASML.AS, SAP.DE, SHEL.L, BP.L, etc.) returned exactly 5 quarterly observations from 2024-12-31 onward. This is a recent Yahoo-side change — yfinance is no longer a viable multi-year fundamentals source.

### 3. SEC EDGAR works but only at annual frequency for EU filers
EU-domiciled firms file Form **20-F**, which is an **annual** XBRL filing. The `companyfacts` endpoint returns annual observations for CapEx and Assets, not quarterly. Sample results:

| Firm | CIK | Taxonomy | n obs | Span |
|------|-----|----------|-------|------|
| ASML | 0000937966 | us-gaap | 16 | 2010-12 → 2025-12 |
| AstraZeneca | 0000901832 | ifrs-full | 20 | 2015-12 → 2025-12 |
| HSBC | 0001089113 | ifrs-full | 18 | 2015-12 → 2025-12 |
| Unilever | 0000217410 | ifrs-full | 13 | 2016-12 → 2025-12 |
| Novartis | 0001114448 | ifrs-full | 18 | 2016-12 → 2025-12 |
| Shell / BP / Total / SAP / Diageo | — | — | 0 capex | missing capex tag |

Only half of the sample report CapEx in their 20-F XBRL. Coverage is annual, not quarterly.

### 4. Macro and institutional layers are **not** the blocker
FRED alone gives us:
- ECB Deposit Facility Rate (ECBDFR) — 100% coverage ✓
- Euro area 10Y yield (IRLTLT01EZM156N) — 100% coverage ✓
- Euro area industrial production — 87.5% ✓ (partial but workable)
- UK 3M interbank rate — 42% ✗ (discontinued 2016, need alternative)

Plus BoE Bank Rate via IADB at 100%. ECB SDW has correct series codes somewhere — my audit used guessed keys that 404'd. Fixing these is ~2 hours of work once the firm-layer blocker is resolved.

**The macro layer is feasible. The firm layer is not (as scoped).**

---

## Pivot options (ranked)

### Option 1: Pay for SimFin+ for one month (~$39) ★ recommended
- **Unblock:** 20+ years quarterly fundamentals for ~800+ EU firms.
- **Effort:** Phase 2-4 proceeds as written; no methodology change.
- **Timeline:** No delay beyond plan (5-6 weeks still realistic).
- **Tradeoff:** Raw data can't be redistributed; replication package ships derived intensities + a `fetch_simfin.py` script users run with their own key.
- **One-time cost. Cancel after the data pull.** Cleanest path.

### Option 2: Shift UK/EU panel to **annual frequency** (zero cost)
- Use SEC EDGAR 20-F annual filings for ~40-60 EU firms (filter on CapEx+Assets presence).
- Primary US panel also rerun at annual frequency for matched-frequency comparison.
- **Effort:** Rerun `table5_architectures.py` on annualized US panel (~1 day) + build 20-F annual panel (~1 week).
- **Timeline:** Probably matches plan.
- **Tradeoff:** Paper must be reframed — new claim is "architecture survives frequency collapse and cross-regime replication." Actually more compelling scientifically. The US primary result stays unchanged; the new contribution is the 2×2 (freq × region).
- **Risk:** Annual T=15 is thin; DM-HAC power drops. May need to present as "directional evidence" rather than significance.

### Option 3: **EU-domiciled US filers** subset (free, sector-biased)
- Firms like Liberty Global, QIAGEN, Garmin, Amdocs, Check Point, CRH, Covestro file 10-K with SEC → quarterly us-gaap XBRL via existing `fetch_edgar.py`.
- Estimated 30-50 such firms. Reuses existing infrastructure 100%.
- **Tradeoff:** These aren't a representative European panel. They're small/mid cap, tech/biotech heavy, and mostly post-2014 IPOs. A reviewer will note the selection bias immediately.
- **Not recommended as primary** — usable as a robustness check only.

### Option 4: Pivot extension target to **Japan** via EDINET
- Japan mandates quarterly reporting since 2008; EDINET is free and public.
- Similar engineering cost to a corrected UK/EU build, but different geography gives an equally strong reviewer argument.
- **Timeline:** +1-2 weeks (new data pipeline language/tooling: Japanese XBRL).
- **Tradeoff:** New unknowns (Japanese disclosure quirks, yen accounting); loses the UK/EU regime-divergence narrative the original plan leaned on.

### Option 5: Try **Financial Modeling Prep** or **EOD Historical Data** free tier
- Both have free tiers with global fundamentals. Neither was tested in this audit.
- **Effort:** ~1 day to smoke-test coverage the way SimFin was tested today.
- **Tradeoff:** Speculative — may yield the same cap problem.

---

## What I need from you

Please pick the pivot direction before I touch Phase 2. My recommendation ranking:

1. **Option 1 (SimFin+ paid)** — fastest, cleanest, preserves the plan verbatim. $39 one-time.
2. **Option 2 (annual frequency)** — zero cost but reframes the paper. Arguably a *stronger* scientific contribution (frequency × region 2×2) at the price of rewriting §6.5.
3. **Option 5 (smoke-test FMP)** — cheap diagnostic if neither of the above is acceptable.

If you want, I can spend ~1 hour smoke-testing FMP before you commit, to see if it's a free escape hatch.

---

## What already works (salvageable from today's work)

- `scripts/data_pipeline/feasibility_audit_uk_eu.py` — the audit framework itself is reusable; it can be re-pointed at FMP or a SimFin+ key with minimal edits.
- `data/audit/uk_eu_coverage_matrix.csv` — full coverage matrix from today's run, 54 rows.
- Confirmed macro+institutional layer feasibility via FRED + BoE (2-3 real macros ready to pull).
- `UK_EU_PANEL_PLAN.md` §4.5 fallback window to 2014Q1 is **insufficient** — the SimFin-free cap is a hard 2019 start, not a 2012 start. The plan's Pivot B needs revision regardless of which direction you pick.
