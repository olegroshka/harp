# HARP: Replication Package

Replication code and data for:

> **Global Persistence, Local Residual Structure: Forecasting Heterogeneous Investment Panels**


## Key Findings

1. A two-stage architecture (global pooled AR(1) + block-specific local PCA+ridge) improves full-panel out-of-sample R^2 from 0.630 to 0.677 on a 93-actor quarterly investment panel mixing macro indicators, institutional data, and firm-level ratios.

2. The gain (+0.047, CI [+0.036, +0.058], 10/10 windows, placebo z = 7.82) arises from removing cross-block interference in residual dynamics. It requires data-type heterogeneity and does not replicate on homogeneous firm-only panels.

3. Among linear estimators, the gain is architectural rather than methodological. Per-actor gradient boosting with the same block decomposition (R^2 = 0.657) does not close the gap, showing the advantage combines block-specific estimation with low-rank factor extraction.

## Quick Start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd harp
uv sync

# Run the core result (Table 5: 8 architectures, ~4 seconds)
PYTHONIOENCODING=utf-8 uv run python scripts/table5_architectures.py
```

Expected output: G0=0.591, BA=0.611, G1=0.630, S1=0.599, M1=0.669, M2=0.677, BA_M2=0.661, ENS=0.639.

## Full Reproduction

Run all experiments in dependency order (~35 minutes total, dominated by the 1000-permutation placebo test):

```bash
# Phase 1: Core pipeline (independent, ~20s)
PYTHONIOENCODING=utf-8 uv run python scripts/table2_augmentation.py
PYTHONIOENCODING=utf-8 uv run python scripts/table3_method_comparison.py
PYTHONIOENCODING=utf-8 uv run python scripts/geodesic_diagnostics.py

# Phase 2: Main architecture comparison (~4s)
PYTHONIOENCODING=utf-8 uv run python scripts/table5_architectures.py

# Phase 3: Validation (~28 min)
PYTHONIOENCODING=utf-8 uv run python scripts/table7_placebo.py           # ~25 min
PYTHONIOENCODING=utf-8 uv run python scripts/table8_cross_panel.py       # ~8s
PYTHONIOENCODING=utf-8 uv run python scripts/lowo_block_selection.py     # ~2s
PYTHONIOENCODING=utf-8 uv run python scripts/dm_hac_inference.py         # ~3s
PYTHONIOENCODING=utf-8 uv run python scripts/robustness_checks.py        # ~2s
PYTHONIOENCODING=utf-8 uv run python scripts/fred_recursive.py           # ~1s
PYTHONIOENCODING=utf-8 uv run python scripts/filing_lag.py               # ~1s

# Phase 4: Falsification (~13s)
PYTHONIOENCODING=utf-8 uv run python scripts/target_sensitivity.py
PYTHONIOENCODING=utf-8 uv run python scripts/gating_policies.py

# Phase 5: Supplementary (~3 min)
PYTHONIOENCODING=utf-8 uv run python scripts/table9_parsimony.py         # ~7s
PYTHONIOENCODING=utf-8 uv run python scripts/gbm_and_supplementary.py    # ~2 min
PYTHONIOENCODING=utf-8 uv run python scripts/gbm_sector_and_ic.py       # ~46s
PYTHONIOENCODING=utf-8 uv run python scripts/held_out_and_stratified.py  # ~2 min
PYTHONIOENCODING=utf-8 uv run python scripts/local_dmd_vs_pca.py        # ~5s

# Phase 6: Figures
PYTHONIOENCODING=utf-8 uv run python scripts/generate_figures.py
```

## Script-to-Table Mapping

| Script | Paper Table/Section |
|--------|-------------------|
| `table2_augmentation.py` | Table 1 (panel stats), Table 2 (augmentation gains) |
| `table3_method_comparison.py` | Table 3 (12-model method comparison) |
| `table5_architectures.py` | Table 5 (8 architectures), Table 6 (per-block R^2) |
| `table7_placebo.py` | Table 7 (1000 random partitions, z=7.82) |
| `table8_cross_panel.py` | Table 8 (146-firm, 270-actor cross-panel) |
| `table9_parsimony.py` | Table 9 (T x K_b sensitivity grid) |
| `geodesic_diagnostics.py` | Section 4.3 (geodesic), Section 6.2 (rotation), Appendix C/F |
| `lowo_block_selection.py` | Section 5.3 (leave-one-window-out) |
| `dm_hac_inference.py` | Section 4.1 (DM-HAC, block bootstrap) |
| `robustness_checks.py` | Section 5.6-5.8 (boundary, remainder, candidates) |
| `fred_recursive.py` | Appendix H (recursive FRED normalisation) |
| `filing_lag.py` | Appendix I (one-quarter filing lag) |
| `target_sensitivity.py` | Section 6.3, Appendix D (7 target formulations) |
| `gating_policies.py` | Section 6.4, Appendix E (5 gating policies) |
| `gbm_and_supplementary.py` | Table 3 (GBM), Section 5.6 (two-block), Section 5.8 (MAE) |
| `gbm_sector_and_ic.py` | Table 3 (GBM+sector), Section 5.8 (IC) |
| `held_out_and_stratified.py` | Section 5.4-5.5 (held-out decade, stratified placebo) |
| `local_dmd_vs_pca.py` | Section 6.1 (local DMD vs PCA+ridge) |
| `generate_figures.py` | Figures 2, 3, 5 |
| `standalone_diagnostic.py` | Appendix B (standalone DMD R^2: 0.415, 0.483, 0.486) |

## Portfolio Analysis (Table 12)

Scripts in `scripts/portfolio/` reproduce the portfolio backtest in Section 5.8.
Run in order:

```bash
# 1. Extract per-firm predictions from the mixture pipeline (~4s)
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/extract_predictions.py

# 2. Download quarterly returns for 59 US firms (~30s, requires internet)
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/prepare_returns.py

# 3. Run the full backtest (all signals, subsample analysis, TC sensitivity)
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/run_backtest.py

# 4. Sector-level decomposition
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/sector_analysis.py

# 5. Timing analysis (lag structure, leak tests)
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/timing_analysis.py

# 6. Alternative signal constructions (sector rotation, within-block selection)
PYTHONIOENCODING=utf-8 uv run python scripts/portfolio/alternative_uses.py
```

Key result (Table 12): within the tech/health block (25 firms), the local component
signal (M2-G1) produces Sharpe 1.06 vs 0.98 equal-weight benchmark. Active IR = +0.47,
but not statistically significant (t = 1.50, p = 0.14, 40 quarters).

| Script | Purpose |
|--------|---------|
| `extract_predictions.py` | Save per-actor M2/G1/AR(1) predictions |
| `prepare_returns.py` | Download and compute quarterly equity returns |
| `signals.py` | Signal definitions (14 variants) |
| `portfolio.py` | Portfolio construction (quintile sort, long-short, turnover) |
| `metrics.py` | Performance metrics (Sharpe, PSR, IR, drawdown, Calmar) |
| `run_backtest.py` | Main backtest runner |
| `sector_analysis.py` | Sector decomposition of G1-M2 disagreement |
| `timing_analysis.py` | Signal lag analysis and look-ahead tests |
| `alternative_uses.py` | Sector rotation, within-block selection, regime timing |
| `test_backtest.py` | Verification tests (random signal, perfect foresight, leak check) |
| `deep_dive_disagr.py` | Concentrated portfolios, persistence, vol scaling |
| `final_analysis.py` | Long-only analysis with per-year active returns |

## Data

| File | Description |
|------|-------------|
| `data/intensities/experiment_a1_intensities.parquet` | 93-actor primary panel (84 quarters, 2005Q1-2025Q4) |
| `data/registries/experiment_a1_registry.json` | Actor metadata (sector, layer, actor type) |
| `data/processed/edgar_balance_sheet.parquet` | SEC EDGAR data for 146-firm and 270-actor panels |

The primary panel contains quarterly investment intensity values for 7 macro indicators (FRED min-max normalised), 4 institutional actors, and 82 US-listed firms (cross-sectional percentile ranks of CapEx/Assets). The 82-firm sub-panel is a balanced panel of S&P 500 constituents with complete data over the sample period.

## Dependencies

Core: numpy, pandas, scipy, scikit-learn, pyarrow, statsmodels, matplotlib.

GPU acceleration (optional): install with `uv sync --extra gpu` to add PyTorch. All scripts fall back to CPU automatically.

## Library

The `src/harp/` package contains the spectral decomposition, state-space filtering, and evaluation modules used by the experiment scripts:

- `spectral/dmd.py` — Exact Dynamic Mode Decomposition
- `dynamics/kalman.py` — DMD-based Kalman filter with spherical observation noise regularisation (Appendix A). Key features: spherical R for N >> T panels, spectral radius clipping, adaptive Q with quarterly reset
- `validation/metrics.py` — OOS R^2 and Diebold-Mariano test
- `compute/` — SVD with optional CUDA dispatch
- `data/` — Actor registry, intensity mappers, point-in-time store (used by data pipeline scripts)

## Rebuilding the Data from Scratch

The pre-built data files in `data/` are sufficient for all experiment scripts. If you want to rebuild them from primary sources, install the data pipeline dependencies and run the scripts in `scripts/data_pipeline/` in order.

### Setup

```bash
uv sync --extra data-pipeline
```

### Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `FRED_API_KEY` | Yes | [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html) (free registration) |
| `BEA_API_KEY` | Optional | [BEA API](https://apps.bea.gov/API/signup/) (falls back to Excel downloads) |

Set before running:
```bash
export FRED_API_KEY=your_key_here
```

### Pipeline Execution Order

```bash
# 1. Build equity universes and download OHLCV data (~5 min, Yahoo Finance)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/build_universes.py

# 2. Fetch source data (run in parallel, ~10-30 min total)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_edgar.py    # SEC EDGAR (no key)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_fred.py     # FRED (FRED_API_KEY)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_gdelt.py    # GDELT (no key)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_bea.py      # BEA (optional key)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_imf.py      # IMF (no key)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/fetch_oecd.py     # OECD (no key)

# 3. Build actor registries
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/build_registries.py
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/build_mixed_expanded.py

# 4. Compute normalised intensities (builds experiment_a1_intensities.parquet)
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/compute_intensities.py

# 5. Audit data quality
PYTHONIOENCODING=utf-8 uv run python scripts/data_pipeline/data_audit.py
```

### Data Source Summary

| Source | API Key | Rate Limit | Data |
|--------|---------|------------|------|
| SEC EDGAR | None (User-Agent only) | 10 req/s | Firm balance sheets (CapEx, Assets) |
| FRED/ALFRED | `FRED_API_KEY` | 120 req/min | Macro indicators (GDP, rates, VIX) |
| GDELT GKG 2.0 | None | None | Narrative intensity signals |
| BEA | Optional `BEA_API_KEY` | None | Input-output tables |
| IMF DataMapper | None | None | Cross-country macro (GDP, CPI) |
| OECD SDMX 3.0 | None | None | Leading indicators |
| Yahoo Finance | None | Best-effort | Equity OHLCV |

## Paper

The full LaTeX source is in `paper/harp.tex`. Compile with:

```bash
cd paper && pdflatex harp.tex && pdflatex harp.tex
```

## License

MIT
