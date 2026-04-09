"""Portfolio performance metrics.

All functions operate on arrays of periodic (quarterly) returns.
Annualisation assumes 4 quarters per year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PERIODS_PER_YEAR = 4


# ── Return and Risk ──────────────────────────────────────────────

def annualised_return(returns: np.ndarray) -> float:
    """Annualised geometric return from quarterly returns."""
    r = np.asarray(returns, dtype=float)
    cum = np.prod(1 + r)
    n_years = len(r) / PERIODS_PER_YEAR
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)


def annualised_volatility(returns: np.ndarray) -> float:
    """Annualised volatility from quarterly returns."""
    return float(np.std(returns, ddof=1) * np.sqrt(PERIODS_PER_YEAR))


# ── Sharpe Ratio ─────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Quarterly excess returns (or total returns if rf=0).
        rf: Annualised risk-free rate (subtracted per quarter).
    """
    r = np.asarray(returns, dtype=float)
    rf_q = (1 + rf) ** (1 / PERIODS_PER_YEAR) - 1
    excess = r - rf_q
    if np.std(excess, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio_bootstrap_ci(
    returns: np.ndarray,
    rf: float = 0.0,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Returns: (sharpe, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r, rf)

    boot_srs = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        boot_srs.append(sharpe_ratio(sample, rf))

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_srs, alpha * 100))
    hi = float(np.percentile(boot_srs, (1 - alpha) * 100))
    return sr, lo, hi


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    rf: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Tests H0: SR <= sr_benchmark.
    Returns the probability that the true SR exceeds the benchmark.

    PSR = Phi( (SR - SR*) / SE(SR) )
    where SE(SR) = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1) )
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return 0.5

    sr = sharpe_ratio(r, rf)
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))  # excess=False → raw kurtosis

    se_sr_sq = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
    if se_sr_sq <= 0:
        return 1.0 if sr > sr_benchmark else 0.0

    se_sr = np.sqrt(se_sr_sq)
    z = (sr - sr_benchmark) / se_sr
    return float(scipy_stats.norm.cdf(z))


# ── Information Ratio ────────────────────────────────────────────

def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """Annualised information ratio: mean(active) / std(active) * sqrt(4)."""
    active = np.asarray(returns) - np.asarray(benchmark_returns)
    if np.std(active, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(active) / np.std(active, ddof=1) * np.sqrt(PERIODS_PER_YEAR))


# ── Drawdown ─────────────────────────────────────────────────────

def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from peak to trough (as a positive fraction)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def calmar_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd < 1e-12:
        return 0.0
    ann_ret = annualised_return(returns)
    return float(ann_ret / mdd)


# ── Hit Rate ─────────────────────────────────────────────────────

def hit_rate(returns: np.ndarray) -> float:
    """Fraction of positive-return quarters."""
    r = np.asarray(returns, dtype=float)
    return float(np.mean(r > 0))


# ── Summary ──────────────────────────────────────────────────────

def full_metrics(
    returns: np.ndarray,
    rf: float = 0.0,
    benchmark_returns: np.ndarray | None = None,
    label: str = "",
) -> dict[str, float]:
    """Compute all metrics for a return series."""
    r = np.asarray(returns, dtype=float)

    sr, sr_lo, sr_hi = sharpe_ratio_bootstrap_ci(r, rf)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, rf=rf)

    result = {
        "label": label,
        "ann_return": annualised_return(r),
        "ann_vol": annualised_volatility(r),
        "sharpe": sr,
        "sharpe_ci_lo": sr_lo,
        "sharpe_ci_hi": sr_hi,
        "psr": psr,
        "max_dd": max_drawdown(r),
        "calmar": calmar_ratio(r, rf),
        "hit_rate": hit_rate(r),
        "n_quarters": len(r),
    }

    if benchmark_returns is not None:
        result["info_ratio"] = information_ratio(r, benchmark_returns)

    return result
