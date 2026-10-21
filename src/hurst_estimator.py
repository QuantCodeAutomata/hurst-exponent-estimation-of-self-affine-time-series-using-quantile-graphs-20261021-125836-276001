"""
Hurst exponent estimation via power-law scaling of mean jump length.

The method exploits the relationship:

    Delta(k) ~ k^H   (paper §III)

where Delta(k) is the mean jump length of a random walker on the Quantile Graph
at lag k, and H is the Hurst exponent.  H is estimated as the slope of
log(Delta(k)) vs log(k) using linear regression on the scaling region.

References: Campanharo et al., "Hurst exponent estimation of self-affine time
series using quantile graphs".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantile_graph import compute_delta_k


def estimate_hurst_single(
    series: np.ndarray,
    Q: int,
    k_values: np.ndarray,
) -> tuple[float, float]:
    """Estimate the Hurst exponent H from a single time series.

    Fits a power law Delta(k) ~ k^H in log-log space using OLS regression
    over the supplied k range (paper §III).

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Input time series.
    Q : int
        Number of quantile nodes.
    k_values : np.ndarray of int
        Lag values over which to fit the power law.

    Returns
    -------
    H_est : float
        Estimated Hurst exponent (slope of log-log fit).
    std_err : float
        Standard error of the slope estimate.
    """
    deltas = compute_delta_k(series, Q, k_values)

    valid = np.isfinite(deltas) & (deltas > 0)
    if valid.sum() < 2:
        return np.nan, np.nan

    log_k = np.log(k_values[valid])
    log_delta = np.log(deltas[valid])

    slope, intercept, r_value, p_value, std_err = linregress(log_k, log_delta)
    return float(slope), float(std_err)


def estimate_hurst_ensemble(
    ensemble: np.ndarray,
    Q: int,
    k_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate H for each realization in an ensemble and return statistics.

    Parameters
    ----------
    ensemble : np.ndarray, shape (n_realizations, T)
        Array of time series.
    Q : int
        Number of quantile nodes.
    k_values : np.ndarray of int
        Lag values for power-law fitting.

    Returns
    -------
    H_mean : float
        Mean estimated Hurst exponent over realizations.
    H_ci95 : float
        Half-width of the 95% confidence interval (1.96 * std / sqrt(n)).
    """
    n = ensemble.shape[0]
    H_estimates = np.empty(n, dtype=np.float64)

    for r in range(n):
        H_est, _ = estimate_hurst_single(ensemble[r], Q, k_values)
        H_estimates[r] = H_est

    valid = np.isfinite(H_estimates)
    if valid.sum() == 0:
        return np.nan, np.nan

    H_mean = float(np.mean(H_estimates[valid]))
    H_ci95 = float(1.96 * np.std(H_estimates[valid], ddof=1) / np.sqrt(valid.sum()))
    return H_mean, H_ci95


def get_delta_k_ensemble_mean(
    ensemble: np.ndarray,
    Q: int,
    k_values: np.ndarray,
) -> np.ndarray:
    """Compute the ensemble-mean Delta(k) curve.

    Parameters
    ----------
    ensemble : np.ndarray, shape (n_realizations, T)
        Array of time series.
    Q : int
        Number of quantile nodes.
    k_values : np.ndarray of int
        Lag values.

    Returns
    -------
    np.ndarray of float, shape (len(k_values),)
        Mean Delta(k) averaged over realizations.
    """
    n = ensemble.shape[0]
    delta_matrix = np.empty((n, len(k_values)), dtype=np.float64)

    for r in range(n):
        delta_matrix[r] = compute_delta_k(ensemble[r], Q, k_values)

    return np.nanmean(delta_matrix, axis=0)
