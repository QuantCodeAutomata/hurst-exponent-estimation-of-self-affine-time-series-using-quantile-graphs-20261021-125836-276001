"""
Synthetic self-affine time series generator.

Generates fractional Gaussian noise (fGn) and fractional Brownian motion (fBm)
using the Davies-Harte method (via the fbm library), which is used as a proxy
for the wavelet-based synthesis described in the paper (paper §II).

Beta-Hurst relationships (paper §II):
  - fGn: beta = 2H - 1,  H = (beta + 1) / 2,  beta in [-1, 1]
  - fBm: beta = 2H + 1,  H = (beta - 1) / 2,  beta in (1, 3]
"""

from __future__ import annotations

import numpy as np
from fbm import FBM


def _hurst_from_beta(beta: float) -> tuple[float, str]:
    """Convert power-spectrum exponent beta to Hurst exponent and process type.

    Parameters
    ----------
    beta : float
        Power-spectrum exponent.  Range [-1, 3].

    Returns
    -------
    H : float
        Hurst exponent in [0, 1].
    process_type : str
        'fgn' for fractional Gaussian noise (beta <= 1),
        'fbm' for fractional Brownian motion (beta > 1).
    """
    if beta < 1.0:
        # fGn: beta = 2H - 1  →  H = (beta + 1) / 2  (beta in [-1, 1))
        H = (beta + 1.0) / 2.0
        process_type = "fgn"
    else:
        # fBm: beta = 2H + 1  →  H = (beta - 1) / 2  (beta in [1, 3])
        H = (beta - 1.0) / 2.0
        process_type = "fbm"

    H = float(np.clip(H, 1e-4, 1.0 - 1e-4))  # avoid numerical edge cases
    return H, process_type


def generate_series(
    T: int,
    beta: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single self-affine time series of length T with exponent beta.

    Parameters
    ----------
    T : int
        Length of the output series.
    beta : float
        Power-spectrum exponent.  Uses fGn for beta in [-1, 1] and
        fBm for beta in (1, 3].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (T,)
        The generated time series.
    """
    H, process_type = _hurst_from_beta(beta)

    # fbm library relies on numpy's global random state; seed it for reproducibility
    rng = np.random.default_rng(seed)
    numpy_seed = int(rng.integers(0, 2**31))
    np.random.seed(numpy_seed)

    if process_type == "fgn":
        f = FBM(n=T, hurst=H, length=1, method="daviesharte")
        series = f.fgn()  # shape (T,)
    else:  # fBm
        f = FBM(n=T, hurst=H, length=1, method="daviesharte")
        series = f.fbm()[1:]  # shape (T,), drop the t=0 point (always 0)

    return series.astype(np.float64)


def generate_ensemble(
    T: int,
    beta: float,
    n_realizations: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Generate an ensemble of self-affine time series.

    Parameters
    ----------
    T : int
        Length of each time series.
    beta : float
        Power-spectrum exponent.
    n_realizations : int
        Number of independent realizations.
    base_seed : int
        Base random seed; each realization uses base_seed + realization_index.

    Returns
    -------
    np.ndarray of shape (n_realizations, T)
        Array of generated time series.
    """
    ensemble = np.empty((n_realizations, T), dtype=np.float64)
    for r in range(n_realizations):
        ensemble[r] = generate_series(T, beta, seed=base_seed + r)
    return ensemble


def hurst_from_beta(beta: float) -> float:
    """Return the Hurst exponent corresponding to power-spectrum exponent beta.

    Parameters
    ----------
    beta : float
        Power-spectrum exponent in [-1, 3].

    Returns
    -------
    float
        Hurst exponent in [0, 1].
    """
    H, _ = _hurst_from_beta(beta)
    return H
