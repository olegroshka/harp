# UK/EU Extension — Pre-Registered Analysis Plan

**Committed:** 2026-04-15 (before any UK/EU data were retrieved)
**Referenced by:** paper §6.1 (Panel Construction and Analysis Plan) and Appendix L.

This document captures the pre-registration commitments for the UK/EU
heterogeneous-panel extension reported in Section 6 and Appendix L of the
paper. It is a tight extract of the analysis plan that was committed to the
project repository on 2026-04-15. Hypotheses, test specifications, panel
design, and stop-for-futility rules below were fixed before the panel was
built; they are reproduced here verbatim from the commit-time plan and have
not been edited after the fact.

Post-hoc status (which hypotheses were ultimately supported, partially
supported, or rejected on the realised data) is reported honestly in the
paper itself (§6, Appendix L); it is intentionally **not** included in this
document so that this file remains a clean pre-registration record.

---

## 1. Pre-Registered Hypotheses

The following hypotheses were committed before any experiment was run. Each
statistical decision uses the test specification and α declared here.

| ID | Hypothesis | Test | α | Direction |
|----|-----------|------|---|-----------|
| **H1** (primary) | M2 > G1 on the full UK/EU heterogeneous panel | DM-HAC one-sided, 4-quarter bandwidth | 0.05 | one-sided |
| **H2** | Delta magnitude `(M2 − G1)_UK/EU` lies within ±0.030 of the 93-actor US Delta (+0.047) | CI overlap of the two Deltas | 0.05 | two-sided |
| **H3** | Mixture architecture with US-learned block structure (GICS sectors) transfers to UK/EU panel with positive Delta | DM-HAC on transfer run | 0.05 | one-sided |
| **H4** (scope falsification) | Homogeneous UK/EU firms-only sub-panel shows null or negative Delta (mirrors 146-firm US result) | DM-HAC | 0.05 | two-sided, expected null |
| **H5** (bonus) | Combined US + UK/EU panel shows Delta ≥ +0.040 | DM-HAC | 0.05 | one-sided |

**Stop-for-futility rule.** If H1 yields Δ ≤ −0.010 at the architecture
checkpoint (wrong sign, meaningful magnitude), pause before running H2–H5
and reframe as "US-regime-specific scope narrowing" rather than
"regime-independent replication." This is still publishable, but the paper
structure changes.

---

## 2. Panel Construction Commitments

### 2.1 Target composition

| Layer | Count | Normalisation | Source category |
|-------|-------|---------------|-----------------|
| 0 — global shocks + EU/UK macro | 7 | min-max (trending) | ECB SDW, Eurostat, ONS, BoE IADB, FRED |
| 1 — EU/UK institutional intermediaries | 5 | min-max | ECB, BoE, EIB, ESMA, IMF |
| 2 — UK/EU listed firms (CapEx/Assets) | 80–100 | cross-sectional percentile rank | SimFin + yfinance + SEC 20-F |
| **Total** | **92–112** | — | — |

### 2.2 Universe and window

- **Primary universe:** STOXX 600 constituents as of 2026-01-01 plus any
  UK-listed firms in FTSE 350 not already in STOXX 600.
- **Filter chain:** continuous listing 2010-01-01 to 2025-12-31; CapEx and
  total assets reported for ≥60 of 60 quarters with post-imputation
  tolerance ≤5%; GICS sector classification compatible with the US panel;
  source coverage from SimFin, yfinance, or SEC 20-F.
- **Sample window:** 2010Q1–2025Q4 (60 quarters). Fallback if Gate G1 fails:
  2012Q1–2025Q4 (56 quarters).
- **Test windows:** walk-forward 2015–2024 (40 test quarters). Default
  training window T = 5 years. Matches primary panel.
- **Fundamental variable:** CapEx / Total Assets, then cross-sectional
  percentile rank per quarter (FX-invariant, no currency normalisation).

### 2.3 Block partition

- **Primary:** GICS sectors directly (transfer test — does the US-learned
  block structure work on EU data?).
- **Alternative:** LOWO-selected partition fit afresh on EU data.

Both variants run; the paper reports whichever tells the cleaner story.

### 2.4 Cross-listing filter (specified at panel-build time, not added post-hoc)

EODHD-listed firms on European exchanges include approximately 30%
US/JP/CN cross-listings. The pre-registered build restricts the firm
universe to firms with ISIN prefixes in {GB, DE, FR, ES, IT, NL, BE, AT,
FI, SE, DK, NO, CH, IE, LU, PT, GR, PL, CZ, HU, IS, LI, SK, SI, EE, LV, LT,
MT, CY, RO, BG, HR}.

---

## 3. Pre-Specified Validation Procedures

| # | Procedure | Reuses | Output |
|---|-----------|--------|--------|
| C1 | Table 5-UK — 8 architectures (G0, BA, G1, S1, M1, M2, BA_M2, ENS) | `table5_architectures.py` | `results/metrics/table5_uk_eu.json` |
| C2 | DM-HAC inference on G1 vs M2 with block bootstrap CI | `dm_hac_inference.py` | `results/metrics/dm_hac_uk_eu.json` |
| C3 | 1,000-permutation placebo on random block partitions of identical sizes | `table7_placebo.py` | `results/metrics/placebo_uk_eu.json` |
| C4 | Extended cross-panel comparison (Table 8 with UK/EU column) | `table8_cross_panel.py` | `results/metrics/table8_extended.json` |
| C5 | Scope falsification — UK/EU firms-only sub-panel | embedded in C4 | — |
| C6 | Per-block R² and LOWO selection on UK/EU | `lowo_block_selection.py` | `results/metrics/lowo_uk_eu.json` |

Bonus procedures (run only if C1–C6 are all green):

- **B1 — Transfer test:** project UK/EU data onto the US-learned basis U.
- **B2 — Combined panel:** build the union of the US 93-actor and UK/EU
  panels; re-run Table 5.
- **B3 — Regime stratification:** split test period into pre-Brexit, COVID,
  and post-COVID sub-windows.
- **B4 — Held-out periods:** train pre-2018, test post-2018, and the
  reverse.

---

## 4. Decision Rules

- **H1 decision:** DM-HAC p < 0.05 one-sided ⇒ H1 supported.
- **H2 decision:** compare `(Δ_UK/EU ± CI_UK/EU)` to `(0.047 ± CI_US)`.
  CI overlap ⇒ H2 supported.
- **H3 decision:** run C1 with both fresh LOWO partition and US-sector
  transfer partition. Both Δ > 0 ⇒ H3 supported (architecture + transfer
  both work). Only LOWO works ⇒ H3 rejected but architecture still works.
- **H4 decision:** firms-only Δ ∈ [−0.010, +0.015] ⇒ H4 supported (scope
  condition replicates).
- **H5 decision:** combined-panel Δ ≥ +0.040 with DM-HAC p < 0.05
  one-sided ⇒ H5 supported.

**All decisions are recorded honestly regardless of whether they favour the
paper's narrative.** A negative result on M2 − G1 on the UK/EU panel is an
acceptable outcome and still strengthens the paper (it narrows the scope
condition).

---

## 5. Notes on Implementation Realities (added at archive time)

- The original plan targeted SimFin / yfinance / SEC 20-F as primary firm
  sources. After the Phase 1 data-feasibility audit, the panel was built
  via EODHD, which provided a single coherent multi-exchange feed; this
  source substitution did not change any pre-registered hypothesis or
  decision rule and is documented in the paper's Acknowledgements.
- The "compatible GICS sector labels" requirement was operationalised by
  mapping EODHD's reported sectors to the SMIM six-sector scheme used in
  the US panel; details in Appendix L.
- The full implementation history including milestone-by-milestone
  decisions is maintained outside the replication archive (author's
  working notes); the per-experiment outputs that back every reported
  number are committed to `results/metrics/`.
