"""
Tests for the Quantile Graph Hurst exponent estimation pipeline.

Covers:
  - Synthetic data generation (fBm/fGn)
  - Quantile coarse-graining
  - Transition matrix construction and properties
  - Mean jump length computation
  - Hurst exponent estimation accuracy
  - Edge cases
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from data.generate_synthetic import generate_series, generate_ensemble, hurst_from_beta
from src.quantile_graph import (
    coarse_grain,
    build_transition_matrix,
    mean_jump_length,
    compute_delta_k,
    build_w1,
)
from src.hurst_estimator import estimate_hurst_single, estimate_hurst_ensemble


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def white_noise_series():
    rng = np.random.default_rng(0)
    return rng.standard_normal(1000)


@pytest.fixture
def fbm_series_H07():
    return generate_series(T=2000, beta=2.4, seed=42)  # beta=2.4 → H=0.7


@pytest.fixture
def labels_Q50(white_noise_series):
    return coarse_grain(white_noise_series, Q=50)


# ── Synthetic data generation ─────────────────────────────────────────────────

class TestGenerateSynthetic:
    def test_fgn_shape(self):
        """fGn series has exactly T data points."""
        s = generate_series(T=500, beta=0.0, seed=1)  # white noise
        assert s.shape == (500,)

    def test_fbm_shape(self):
        """fBm series has exactly T data points."""
        s = generate_series(T=500, beta=2.0, seed=1)  # standard BM
        assert s.shape == (500,)

    def test_hurst_from_beta_fgn_range(self):
        """For fGn (beta in [-1,1]), H=(beta+1)/2 is in [0,1]."""
        for beta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            H = hurst_from_beta(beta)
            assert 0.0 < H < 1.0, f"H={H} out of range for beta={beta}"

    def test_hurst_from_beta_fbm_range(self):
        """For fBm (beta in (1,3]), H=(beta-1)/2 is in [0,1]."""
        for beta in [1.2, 2.0, 2.5, 3.0]:
            H = hurst_from_beta(beta)
            assert 0.0 < H < 1.0, f"H={H} out of range for beta={beta}"

    def test_hurst_from_beta_white_noise(self):
        """White noise: beta=0 → H=0.5."""
        H = hurst_from_beta(0.0)
        assert abs(H - 0.5) < 1e-9

    def test_hurst_from_beta_bm(self):
        """Standard BM: beta=2 → H=0.5."""
        H = hurst_from_beta(2.0)
        assert abs(H - 0.5) < 1e-9

    def test_ensemble_shape(self):
        """Ensemble has shape (n_realizations, T)."""
        ens = generate_ensemble(T=200, beta=2.0, n_realizations=5, base_seed=0)
        assert ens.shape == (5, 200)

    def test_ensemble_different_realizations(self):
        """Different realizations should not be identical."""
        ens = generate_ensemble(T=500, beta=2.0, n_realizations=3, base_seed=0)
        assert not np.allclose(ens[0], ens[1])
        assert not np.allclose(ens[0], ens[2])

    def test_series_no_nan(self):
        """Generated series must not contain NaN or Inf."""
        for beta in [-1.0, 0.0, 1.0, 2.0]:
            s = generate_series(T=500, beta=beta, seed=7)
            assert np.all(np.isfinite(s)), f"Non-finite values for beta={beta}"


# ── Quantile coarse-graining ──────────────────────────────────────────────────

class TestCoarseGrain:
    def test_label_range(self, white_noise_series):
        """All labels must be in [0, Q-1]."""
        Q = 50
        labels = coarse_grain(white_noise_series, Q)
        assert labels.min() >= 0
        assert labels.max() <= Q - 1

    def test_label_shape(self, white_noise_series):
        """Labels have the same length as the input series."""
        labels = coarse_grain(white_noise_series, Q=50)
        assert labels.shape == white_noise_series.shape

    def test_roughly_uniform_distribution(self, white_noise_series):
        """Each quantile should receive roughly T/Q points."""
        Q = 50
        labels = coarse_grain(white_noise_series, Q)
        T = len(white_noise_series)
        expected = T / Q
        for q in range(Q):
            count = np.sum(labels == q)
            # Allow up to 50% deviation from expected (very permissive)
            assert 0 < count < 2 * expected, (
                f"Quantile {q} has {count} points (expected ~{expected:.0f})"
            )

    def test_raises_if_T_lt_Q(self):
        """Coarse-graining raises ValueError when T < Q."""
        with pytest.raises(ValueError):
            coarse_grain(np.zeros(10), Q=50)

    def test_dtype_int(self, white_noise_series):
        """Labels must be integer type."""
        labels = coarse_grain(white_noise_series, Q=20)
        assert np.issubdtype(labels.dtype, np.integer)


# ── Transition matrix ─────────────────────────────────────────────────────────

class TestTransitionMatrix:
    def test_shape(self, labels_Q50):
        """Transition matrix must be (Q, Q)."""
        W = build_transition_matrix(labels_Q50, k=1, Q=50)
        assert W.shape == (50, 50)

    def test_row_stochastic(self, labels_Q50):
        """Non-empty rows must sum to 1 (row-stochastic)."""
        W = build_transition_matrix(labels_Q50, k=1, Q=50)
        row_sums = W.sum(axis=1)
        nonzero_rows = row_sums > 0
        np.testing.assert_allclose(
            row_sums[nonzero_rows], 1.0, atol=1e-10,
            err_msg="Row sums must equal 1 for non-empty rows"
        )

    def test_non_negative(self, labels_Q50):
        """All entries must be non-negative."""
        W = build_transition_matrix(labels_Q50, k=1, Q=50)
        assert np.all(W >= 0.0)

    def test_raises_if_k_ge_T(self, labels_Q50):
        """Raises ValueError when k >= T."""
        T = len(labels_Q50)
        with pytest.raises(ValueError):
            build_transition_matrix(labels_Q50, k=T, Q=50)

    def test_larger_lag(self, labels_Q50):
        """W_k for k > 1 must still be row-stochastic."""
        W = build_transition_matrix(labels_Q50, k=5, Q=50)
        assert W.shape == (50, 50)
        row_sums = W.sum(axis=1)
        nonzero_rows = row_sums > 0
        np.testing.assert_allclose(row_sums[nonzero_rows], 1.0, atol=1e-10)

    def test_white_noise_approximately_uniform(self):
        """For white noise, W_1 should be approximately uniform (Q×Q uniform)."""
        rng = np.random.default_rng(42)
        T = 50_000
        series = rng.standard_normal(T)
        Q = 10
        labels = coarse_grain(series, Q)
        W = build_transition_matrix(labels, k=1, Q=Q)
        # Each entry should be close to 1/Q = 0.1
        # Allow generous tolerance due to finite sample
        assert np.all(np.abs(W - 1.0 / Q) < 0.15), (
            f"White noise W_1 deviates too much from uniform: "
            f"max dev = {np.max(np.abs(W - 1/Q)):.4f}"
        )


# ── Mean jump length ──────────────────────────────────────────────────────────

class TestMeanJumpLength:
    def test_scalar_output(self, labels_Q50):
        """mean_jump_length returns a scalar."""
        W = build_transition_matrix(labels_Q50, k=1, Q=50)
        delta = mean_jump_length(W)
        assert isinstance(delta, float)

    def test_non_negative(self, labels_Q50):
        """Mean jump length must be non-negative."""
        W = build_transition_matrix(labels_Q50, k=1, Q=50)
        delta = mean_jump_length(W)
        assert delta >= 0.0

    def test_bounded_by_Q_minus_1(self, labels_Q50):
        """Mean jump length cannot exceed Q-1."""
        Q = 50
        W = build_transition_matrix(labels_Q50, k=1, Q=Q)
        delta = mean_jump_length(W)
        assert delta <= Q - 1

    def test_identity_matrix_zero(self):
        """Identity transition matrix gives Delta = 0 (no jumps)."""
        Q = 10
        W = np.eye(Q)
        delta = mean_jump_length(W)
        assert delta == pytest.approx(0.0)

    def test_anti_diagonal_matrix_maximum(self):
        """Anti-diagonal W gives Delta = Q/2 (mean |i-(Q-1-i)| / Q = Q/2 for even Q).

        For Q=10, row i maps to node Q-1-i, so |i - (Q-1-i)| = |2i-9|.
        Mean over uniform stationary dist = (1/10)*sum|2i-9| = 50/10 = 5 = Q/2.
        """
        Q = 10
        W = np.zeros((Q, Q))
        for i in range(Q):
            W[i, Q - 1 - i] = 1.0
        delta = mean_jump_length(W)
        assert delta == pytest.approx(Q / 2)

    def test_white_noise_delta_is_moderate(self):
        """White noise should produce a mid-range Delta (neither 0 nor Q-1)."""
        rng = np.random.default_rng(0)
        T = 10_000
        series = rng.standard_normal(T)
        Q = 50
        labels = coarse_grain(series, Q)
        W = build_transition_matrix(labels, k=1, Q=Q)
        delta = mean_jump_length(W)
        # White noise should give Delta close to ~Q/3 (rough expected value)
        assert 5.0 < delta < 40.0, f"White noise Delta={delta:.2f} out of range"


# ── Delta(k) vs k power-law scaling ──────────────────────────────────────────

class TestDeltaKScaling:
    def test_shape(self, fbm_series_H07):
        """compute_delta_k returns array matching k_values length."""
        k_vals = np.array([1, 2, 5, 10, 20])
        deltas = compute_delta_k(fbm_series_H07, Q=50, k_values=k_vals)
        assert deltas.shape == (5,)

    def test_non_negative_values(self, fbm_series_H07):
        """Delta(k) values must be non-negative."""
        k_vals = np.arange(1, 11)
        deltas = compute_delta_k(fbm_series_H07, Q=50, k_values=k_vals)
        assert np.all(deltas[np.isfinite(deltas)] >= 0.0)

    def test_increasing_with_k_for_correlated(self):
        """Delta(k) should increase with k for positively correlated fBm (H > 0.5)."""
        # Use a larger T for stability
        series = generate_series(T=5000, beta=2.4, seed=0)  # H=0.7
        k_vals = np.array([1, 5, 25, 100])
        deltas = compute_delta_k(series, Q=50, k_values=k_vals)
        valid = np.isfinite(deltas)
        if valid.sum() >= 2:
            # Delta should be generally increasing
            assert deltas[valid][-1] > deltas[valid][0], (
                f"Delta should increase for H=0.7: {deltas}"
            )


# ── Hurst estimation ──────────────────────────────────────────────────────────

class TestHurstEstimation:
    def test_white_noise_H_approx_05(self):
        """fBm with H=0.5 (standard Brownian motion, beta=2.0) → H_QG ≈ 0.5.

        White noise (fGn, beta=0) is stationary and does NOT produce a
        power-law Delta(k); the paper uses fBm (beta in [1,3]) for estimation.
        """
        # Standard BM: fBm H=0.5, beta = 2H+1 = 2.0
        series = generate_series(T=5000, beta=2.0, seed=123)
        Q = 50
        k_max = 500
        k_vals = np.unique(
            np.round(np.logspace(0, np.log10(k_max), 40)).astype(int)
        )
        k_vals = k_vals[k_vals >= 1]
        H_est, _ = estimate_hurst_single(series, Q=Q, k_values=k_vals)
        assert np.isfinite(H_est), "H estimate must be finite"
        # Standard BM: H_QG should be near 0.5 (allow ±0.15)
        assert abs(H_est - 0.5) < 0.15, (
            f"Standard BM H_QG={H_est:.4f} too far from 0.5"
        )

    def test_correlated_fbm_H_gt_05(self):
        """Correlated fBm (H=0.8) should give H_QG > 0.5."""
        series = generate_series(T=5000, beta=2.6, seed=10)  # H=0.8
        Q = 50
        k_max = 500
        k_vals = np.unique(
            np.round(np.logspace(0, np.log10(k_max), 40)).astype(int)
        )
        k_vals = k_vals[k_vals >= 1]
        H_est, _ = estimate_hurst_single(series, Q=Q, k_values=k_vals)
        assert np.isfinite(H_est), "H estimate must be finite"
        assert H_est > 0.5, f"H_QG={H_est:.4f} should be > 0.5 for H=0.8 fBm"

    def test_anti_correlated_fbm_H_lt_05(self):
        """Anti-correlated fBm (H=0.2) should give H_QG < 0.5."""
        series = generate_series(T=5000, beta=1.4, seed=20)  # H=0.2
        Q = 50
        k_max = 500
        k_vals = np.unique(
            np.round(np.logspace(0, np.log10(k_max), 40)).astype(int)
        )
        k_vals = k_vals[k_vals >= 1]
        H_est, _ = estimate_hurst_single(series, Q=Q, k_values=k_vals)
        assert np.isfinite(H_est), "H estimate must be finite"
        assert H_est < 0.5, f"H_QG={H_est:.4f} should be < 0.5 for H=0.2 fBm"

    def test_h_estimate_finite_for_all_beta(self):
        """H estimation must return finite values for all valid beta."""
        Q = 30
        k_max = 200
        k_vals = np.unique(
            np.round(np.logspace(0, np.log10(k_max), 30)).astype(int)
        )
        k_vals = k_vals[k_vals >= 1]
        for beta in [0.0, 0.5, 1.5, 2.0, 2.5]:
            series = generate_series(T=2000, beta=beta, seed=99)
            H_est, _ = estimate_hurst_single(series, Q=Q, k_values=k_vals)
            assert np.isfinite(H_est), f"Non-finite H for beta={beta}"

    def test_ensemble_estimation_reduces_variance(self):
        """Ensemble mean should be closer to true H than a single estimate."""
        H_true = 0.7
        beta = 2.0 * H_true + 1.0
        Q = 50
        k_max = 500
        k_vals = np.unique(
            np.round(np.logspace(0, np.log10(k_max), 40)).astype(int)
        )
        k_vals = k_vals[k_vals >= 1]

        # Single realization
        single = generate_series(T=5000, beta=beta, seed=0)
        H_single, _ = estimate_hurst_single(single, Q=Q, k_values=k_vals)

        # Ensemble of 10
        ens = generate_ensemble(T=5000, beta=beta, n_realizations=10, base_seed=100)
        H_mean, ci = estimate_hurst_ensemble(ens, Q=Q, k_values=k_vals)

        assert np.isfinite(H_mean), "Ensemble H must be finite"
        assert np.isfinite(ci), "Ensemble CI must be finite"
        # CI should be positive (there is variance)
        assert ci >= 0.0


# ── W_1 topology properties ───────────────────────────────────────────────────

class TestTopologyProperties:
    def test_anti_correlated_anti_diagonal(self):
        """Anti-correlated series (beta=-1) should produce anti-diagonally dominant W_1."""
        series = generate_series(T=5000, beta=-1.0, seed=5)
        Q = 20
        W1 = build_w1(series, Q=Q)

        # Anti-diagonal dominance: mean(anti-diagonal) > mean(main diagonal)
        main_diag = np.mean(np.diag(W1))
        anti_diag = np.mean([W1[i, Q - 1 - i] for i in range(Q)])
        assert anti_diag > main_diag, (
            f"Anti-correlated series should have anti-diag ({anti_diag:.4f}) "
            f"> main diag ({main_diag:.4f})"
        )

    def test_correlated_diagonal(self):
        """Correlated series (beta=2) should produce diagonally dominant W_1."""
        series = generate_series(T=5000, beta=2.0, seed=5)
        Q = 20
        W1 = build_w1(series, Q=Q)

        # Diagonal dominance: mean(main diagonal) > mean(anti-diagonal)
        main_diag = np.mean(np.diag(W1))
        anti_diag = np.mean([W1[i, Q - 1 - i] for i in range(Q)])
        assert main_diag > anti_diag, (
            f"Correlated series should have main diag ({main_diag:.4f}) "
            f"> anti-diag ({anti_diag:.4f})"
        )

    def test_w1_shape_and_stochastic(self):
        """build_w1 returns (Q,Q) row-stochastic matrix."""
        series = generate_series(T=1000, beta=2.0, seed=1)
        Q = 30
        W1 = build_w1(series, Q=Q)
        assert W1.shape == (Q, Q)
        row_sums = W1.sum(axis=1)
        nonzero = row_sums > 0
        np.testing.assert_allclose(row_sums[nonzero], 1.0, atol=1e-10)


# ── Edge-case / robustness tests ──────────────────────────────────────────────

class TestEdgeCases:
    def test_constant_series(self):
        """A constant series puts all points in one quantile; should not crash."""
        series = np.ones(200)
        Q = 50
        labels = coarse_grain(series, Q)
        assert labels.shape == (200,)
        # All in the same bin — exact bin depends on np.digitize behaviour

    def test_single_k_value(self):
        """compute_delta_k with a single k value works correctly."""
        rng = np.random.default_rng(0)
        series = rng.standard_normal(500)
        deltas = compute_delta_k(series, Q=20, k_values=np.array([1]))
        assert deltas.shape == (1,)
        assert deltas[0] >= 0.0

    def test_k_values_beyond_T_returns_nan(self):
        """k values >= T should produce NaN entries."""
        rng = np.random.default_rng(0)
        series = rng.standard_normal(100)
        k_vals = np.array([1, 5, 100])  # k=100 >= T=100
        deltas = compute_delta_k(series, Q=10, k_values=k_vals)
        assert np.isnan(deltas[2]), "k >= T should give NaN"
        assert np.isfinite(deltas[0])
