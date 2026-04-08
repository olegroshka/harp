"""OOS evaluation metrics."""

from __future__ import annotations

import numpy as np
import scipy.stats


def oos_r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """R^2 = 1 - SS_res/SS_tot. Returns 0.0 if SS_tot == 0."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Harvey, Leybourne & Newbold (1997) modified Diebold-Mariano test."""
    errors_1 = np.asarray(errors_1, dtype=float)
    errors_2 = np.asarray(errors_2, dtype=float)
    d = errors_1 ** 2 - errors_2 ** 2
    n = len(d)
    if n < 2:
        return 0.0, 1.0

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0.0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = float(dm_stat * correction)
    p_value = float(2 * scipy.stats.t.sf(abs(dm_stat_corrected), df=n - 1))
    return dm_stat_corrected, p_value
