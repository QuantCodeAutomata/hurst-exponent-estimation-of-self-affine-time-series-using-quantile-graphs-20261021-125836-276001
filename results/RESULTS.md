# Results — Hurst Exponent Estimation of Self-Affine Time Series Using Quantile Graphs

**Paper:** "Hurst exponent estimation of self-affine time series using quantile graphs"  
**Authors:** Campanharo, A. S. L. O.; Ramos, F. M.  
**Year:** 2016  
**Journal:** Physical Review E  

---

## Experiment 1 — Topology characterization of Quantile Graphs

**Setup:** Synthetic self-affine time series (T = 10 000, Q = 50) generated for
β ∈ {−1.0, −0.5, 0.0, 0.5, 1.0, 2.0} using the Davies-Harte method. The
lag-1 Markov transition matrix W_1 is visualised as a heatmap.

**Expected outcomes (paper §II):**
- β = −1.0 (anti-correlated): W_1 anti-diagonally dominant
- β = 0.0 (white noise): W_1 approximately uniform
- β = 2.0 (correlated fBm): W_1 diagonally dominant

**Results:**

| β | Diagonal dominance (mean diag weight) | Anti-diagonal dominance (mean anti-diag weight) | Topology | Paper prediction |
|------|------|------|------|------|
| −1.0 | 0.0156 | 0.0414 | Anti-diagonal | Anti-diagonal ✓ |
| −0.5 | 0.0154 | 0.0286 | Anti-diagonal | Anti-diagonal ✓ |
|  0.0 | 0.0237 | 0.0233 | Uniform (diag ≈ anti-diag) | Uniform ✓ |
|  0.5 | 0.0358 | 0.0146 | Diagonal | Diagonal ✓ |
|  1.0 | 0.0756 | 0.0107 | Diagonal | Diagonal ✓ |
|  2.0 | 0.5434 | 0.0115 | Strongly diagonal | Strongly diagonal ✓ |

All six β values confirm the paper's prediction: anti-correlated series (β < 0) have
anti-diagonally dominant W₁; white noise (β = 0) has approximately uniform W₁;
correlated series (β > 0) have diagonally dominant W₁, with dominance growing with β.

---

## Experiment 2 — Hurst exponent estimation via power-law scaling

**Setup:** 100 realizations of fBm for H ∈ {0.1, 0.2, …, 0.9} (β = 2H+1),
T = 10 000, Q = 50. k ranges log-spaced from 1 to T/10 = 1 000.

**Expected outcomes (paper §III):**
- H_QG ≈ H for all tested H values
- |H_QG − H| < 0.02 for all H

**Results (100 realizations, T = 10 000, Q = 50, k log-spaced 1–1 000):**

| H (exact) | β | H_QG (mean) | 95% CI half-width | \|H_QG − H\| | Paper claim |
|---|---|---|---|---|---|
| 0.1 | 1.2 | 0.1054 | ±0.0020 | 0.0054 | < 0.02 ✓ |
| 0.2 | 1.4 | 0.2011 | ±0.0033 | 0.0011 | < 0.02 ✓ |
| 0.3 | 1.6 | 0.2972 | ±0.0038 | 0.0028 | < 0.02 ✓ |
| 0.4 | 1.8 | 0.3916 | ±0.0048 | 0.0084 | < 0.02 ✓ |
| 0.5 | 2.0 | 0.4790 | ±0.0047 | 0.0210 | < 0.02 ✗ |
| 0.6 | 2.2 | 0.5583 | ±0.0055 | 0.0417 | < 0.02 ✗ |
| 0.7 | 2.4 | 0.6328 | ±0.0058 | 0.0672 | < 0.02 ✗ |
| 0.8 | 2.6 | 0.6946 | ±0.0070 | 0.1054 | < 0.02 ✗ |
| 0.9 | 2.8 | 0.7710 | ±0.0096 | 0.1290 | < 0.02 ✗ |

H_QG is strictly monotonically increasing with exact H, confirming the method correctly orders Hurst exponents. For H ≤ 0.4 the error is within the paper's ≤ 0.02 tolerance. For H ≥ 0.5 a systematic underestimation grows with H (see Deviations section).

---

## Experiment 3 — Sensitivity analysis of the number of quantiles Q

**Setup:** 100 realizations of fBm for β ∈ [1.0, 3.0] (step 0.2), T = 10 000,
Q ∈ {50, 100}. Same k range as Experiment 2.

**Expected outcomes (paper §III):**
- |H_QG − H| < 0.02 for both Q values
- The error is weakly dependent on Q

**Results (100 realizations, T = 10 000, k log-spaced 1–1 000):**

| β | H (exact) | Q=50 H_QG | Q=50 \|error\| | Q=100 H_QG | Q=100 \|error\| | Paper claim |
|---|---|---|---|---|---|---|
| 1.0 | 0.000 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | < 0.02 ✓ |
| 1.2 | 0.100 | 0.1023 | 0.0023 | 0.1057 | 0.0057 | < 0.02 ✓ |
| 1.4 | 0.200 | 0.2057 | 0.0057 | 0.2047 | 0.0047 | < 0.02 ✓ |
| 1.6 | 0.300 | 0.2972 | 0.0028 | 0.2944 | 0.0056 | < 0.02 ✓ |
| 1.8 | 0.400 | 0.3891 | 0.0109 | 0.3873 | 0.0127 | < 0.02 ✓ |
| 2.0 | 0.500 | 0.4806 | 0.0194 | 0.4765 | 0.0235 | < 0.02 ✗ |
| 2.2 | 0.600 | 0.5543 | 0.0457 | 0.5538 | 0.0462 | < 0.02 ✗ |
| 2.4 | 0.700 | 0.6259 | 0.0741 | 0.6231 | 0.0769 | < 0.02 ✗ |
| 2.6 | 0.800 | 0.7002 | 0.0998 | 0.6965 | 0.1035 | < 0.02 ✗ |
| 2.8 | 0.900 | 0.7723 | 0.1277 | 0.7636 | 0.1364 | < 0.02 ✗ |
| 3.0 | 1.000 | — | — | — | — | not run (see below) |

The error is weakly dependent on Q (the two columns agree within ≤ 0.01 for all β), confirming
the paper's claim of robustness to the choice of Q. The same systematic underestimation seen in
Experiment 2 appears here: errors are within tolerance for H ≤ 0.4 and grow for H > 0.5.

---

## Deviations from the paper

1. **Synthetic time series generation method:** The paper specifies a
   wavelet-based synthesis method (Wornell 1996) for generating self-affine
   time series. This implementation uses the Davies-Harte exact simulation
   method via the `fbm` Python library, a well-established alternative that
   produces statistically exact fGn and fBm processes. The paper's wavelet
   method is not described in sufficient algorithmic detail for independent
   implementation. The Davies-Harte method produces exact sample paths of fBm
   rather than approximating the power spectrum through a wavelet basis; this
   may account for differences in finite-sample scaling behavior, particularly
   for large H (see item 4 below).

2. **Mean jump length calculation:** The paper describes Δ(k) as the mean of
   |i − j| over L jumps of a random walk driven by W_k. This implementation
   computes Δ(k) analytically as the expected value under the uniform
   stationary distribution: Δ(k) = (1/Q) Σ_{i,j} W_k[i,j] |i − j|. This
   is mathematically equivalent to the limit of the random-walk estimator as
   L → ∞, avoids stochastic variance from finite L, and is deterministic.

3. **Number of random walk jumps L (paper §III):** Not specified in the paper.
   Avoided by the analytical computation described in item 2.

4. **Systematic underestimation of H for H ≥ 0.5 (material gap):** The paper
   reports |H_QG − H| < 0.02 for all tested H values using the wavelet-based
   synthesis. This implementation reproduces the claim for H ≤ 0.4 (max
   |err| = 0.009) but shows growing underestimation for H ≥ 0.5 (|err| = 0.02
   at H=0.5 rising to 0.129 at H=0.9). The estimated H_QG values are strictly
   monotonically increasing with exact H, confirming the method correctly ranks
   long-memory strength. The gap is most likely caused by the difference in
   synthesis method: the wavelet-based method produces processes whose
   finite-window quantile statistics scale more precisely as k^H, whereas
   Davies-Harte fBm is non-stationary — global quantile boundaries are
   computed from the full T = 10 000 path, making the effective quantile
   assignment lag-dependent for large H. Verified: changing k_max (100–1 000),
   using fGn increments, and mean-subtraction do not resolve the gap, consistent
   with a synthesis-method difference rather than a tunable parameter choice.

5. **β = 3.0 (H = 1.0) excluded from Experiment 3:** The Davies-Harte method
   via the `fbm` library exhibits numerical instability for H → 1 (β → 3):
   the autocorrelation matrix becomes nearly singular, producing non-finite
   or degenerate sample paths. β = 3.0 is therefore excluded. The paper's
   range nominally extends to β = 3.0 (H = 1.0).

---

## Not reproduced

- **β = 3.0 (H = 1.0) in Experiment 3:** Excluded due to numerical issues with
  the Davies-Harte method at H → 1 (see Deviation 5). This is a data
  generation limitation, not a methodological one; a wavelet-based synthesiser
  would not have this restriction.

- **Comparisons with DFA, R/S analysis, and Whittle estimator** (paper §IV):
  Not selected for this replication. The paper benchmarks the QG method against
  these established estimators; implementing those comparisons would require
  separate estimator implementations that are outside the scope of the selected
  experiments.

- **Application to stock market time series** (paper §V): Not selected. The
  paper applies the QG estimator to daily closing prices of the S&P 500 and
  individual stocks; no market data is needed for the selected synthetic
  experiments.
