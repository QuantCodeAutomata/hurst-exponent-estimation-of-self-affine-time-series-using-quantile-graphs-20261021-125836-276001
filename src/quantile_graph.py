"""
Quantile Graph (QG) construction and analysis.

Maps a time series to a weighted directed graph (Quantile Graph) by:
  1. Coarse-graining the value range into Q quantile nodes (paper §II).
  2. Connecting nodes with weighted arcs proportional to lag-k transition
     frequencies — the Markov transition matrix W_k (paper §II, Eq. before Fig. 1).
  3. Computing the mean jump length Delta(k) from W_k as the key topological
     observable (paper §II).

References: Campanharo et al., "Hurst exponent estimation of self-affine time
series using quantile graphs".
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def coarse_grain(
    series: np.ndarray,
    Q: int,
) -> np.ndarray:
    """Assign each value in the time series to one of Q quantile bins.

    Bins are defined by the empirical Q-quantile boundaries of the series.
    Node indices are 0-based (0 to Q-1).

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Input time series.
    Q : int
        Number of quantile nodes.

    Returns
    -------
    np.ndarray of int, shape (T,)
        Quantile node label for each time step (values in [0, Q-1]).
    """
    T = len(series)
    if T < Q:
        raise ValueError(f"Series length {T} must be >= Q={Q}")

    # Q+1 bin edges at 0%, 100%/Q, 200%/Q, ..., 100%
    percentiles = np.linspace(0.0, 100.0, Q + 1)
    bin_edges = np.percentile(series, percentiles)

    # Assign each value to a bin; np.digitize uses bin_edges[1:-1] as thresholds
    # so result is in [0, Q-1]
    labels = np.digitize(series, bin_edges[1:-1])  # values in {0, 1, ..., Q-1}
    labels = np.clip(labels, 0, Q - 1)
    return labels.astype(np.int32)


def build_transition_matrix(
    labels: np.ndarray,
    k: int,
    Q: int,
) -> np.ndarray:
    """Build the Markov transition matrix W_k for lag k.

    W_k[i, j] = P(x(t+k) in q_j | x(t) in q_i)
              = count(labels[t]=i, labels[t+k]=j) / count(labels[t]=i)

    The matrix is row-stochastic (rows sum to 1, except possibly all-zero rows
    which are left as-is).

    Parameters
    ----------
    labels : np.ndarray of int, shape (T,)
        Quantile node labels from coarse_grain().
    k : int
        Time lag.
    Q : int
        Number of nodes.

    Returns
    -------
    np.ndarray of float, shape (Q, Q)
        Row-normalized transition matrix.
    """
    T = len(labels)
    if k >= T:
        raise ValueError(f"Lag k={k} must be less than T={T}")

    src = labels[: T - k]
    dst = labels[k:]

    # Fast frequency counting using bincount
    flat_idx = src.astype(np.int64) * Q + dst.astype(np.int64)
    counts = np.bincount(flat_idx, minlength=Q * Q).reshape(Q, Q).astype(np.float64)

    # Row-normalize: divide each non-empty row by its own sum
    row_sums = counts.sum(axis=1)          # shape (Q,)
    nonzero_mask = row_sums > 0            # shape (Q,)
    counts[nonzero_mask] /= row_sums[nonzero_mask, np.newaxis]

    return counts  # W_k


def mean_jump_length(W: np.ndarray) -> float:
    """Compute the mean jump length Delta(k) from a transition matrix.

    The mean jump length is the expected absolute difference in node indices
    between successive steps of a random walker on the graph, averaged over
    the uniform stationary distribution of quantile nodes:

        Delta(k) = (1/Q) * sum_{i,j} W[i,j] * |i - j|

    The stationary distribution is uniform (pi_i = 1/Q) because, by
    construction, each quantile node holds an equal fraction of data points.

    Parameters
    ----------
    W : np.ndarray of float, shape (Q, Q)
        Row-stochastic transition matrix.

    Returns
    -------
    float
        Mean jump length.
    """
    Q = W.shape[0]
    idx = np.arange(Q)
    J = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)  # (Q, Q)
    return float(np.sum(W * J) / Q)


def compute_delta_k(
    series: np.ndarray,
    Q: int,
    k_values: np.ndarray,
) -> np.ndarray:
    """Compute the mean jump length Delta(k) for a range of lags k.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Input time series.
    Q : int
        Number of quantile nodes.
    k_values : np.ndarray of int
        Array of lag values to evaluate.

    Returns
    -------
    np.ndarray of float, shape (len(k_values),)
        Mean jump length Delta(k) for each lag in k_values.
    """
    labels = coarse_grain(series, Q)
    T = len(labels)
    deltas = np.empty(len(k_values), dtype=np.float64)

    for idx_k, k in enumerate(k_values):
        if k >= T:
            deltas[idx_k] = np.nan
            continue
        W = build_transition_matrix(labels, int(k), Q)
        deltas[idx_k] = mean_jump_length(W)

    return deltas


def build_w1(
    series: np.ndarray,
    Q: int,
) -> np.ndarray:
    """Build the lag-1 transition matrix W_1 for visualization (exp1).

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Input time series.
    Q : int
        Number of quantile nodes.

    Returns
    -------
    np.ndarray of float, shape (Q, Q)
        Row-normalized transition matrix at lag k=1.
    """
    labels = coarse_grain(series, Q)
    return build_transition_matrix(labels, k=1, Q=Q)
