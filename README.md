# Hurst Exponent Estimation of Self-Affine Time Series Using Quantile Graphs

**Paper:** "Hurst exponent estimation of self-affine time series using quantile graphs"  
**Authors:** Campanharo, A. S. L. O.; Ramos, F. M.  
**Year:** 2016  
**Journal:** Physical Review E  

---

## Overview

This repository replicates three experiments from the paper that introduces
the **Quantile Graph (QG)** method for estimating the Hurst exponent H of
self-affine time series.

The method maps a time series to a weighted directed graph by coarse-graining
the value range into Q quantile nodes and connecting them with weighted arcs
proportional to the lag-k transition frequencies (Markov transition matrix W_k).
The Hurst exponent is then estimated as the power-law scaling exponent of the
mean jump length Δ(k) of a random walker on the graph:

    Δ(k) ∝ k^H

---

## Experiments

### Experiment 1 — Topology of QG transition matrices
Visualises the lag-1 Markov transition matrix W_1 for self-affine time series
with varying power-spectrum exponents β ∈ {−1.0, −0.5, 0.0, 0.5, 1.0, 2.0}.
Shows that W_1 is anti-diagonally dominant for anti-correlated series (β < 0),
uniform for white noise (β = 0), and diagonally dominant for correlated series
(β > 0).

### Experiment 2 — Hurst exponent estimation via power-law scaling
Estimates H_QG from the slope of log Δ(k) vs log k for fractional Brownian
motion (fBm) with H ∈ {0.1, 0.2, …, 0.9} (β = 2H + 1 ∈ {1.2, …, 2.8}).
Uses T = 10 000, Q = 50, and 100 independent realizations per H value.
Confirms that H_QG ≈ H with |H_QG − H| ≪ 0.02.

### Experiment 3 — Sensitivity to the number of quantiles Q
Measures |H_QG − H| for Q = 50 and Q = 100 across β ∈ [1.0, 3.0].
Shows that the estimation error is < 0.02 and weakly dependent on Q,
demonstrating robustness to the method's only free parameter.

---

## Repository Structure

```
├── data/
│   └── generate_synthetic.py   # fBm/fGn generation (Davies-Harte method)
├── src/
│   ├── quantile_graph.py       # QG construction, W_k, Δ(k)
│   └── hurst_estimator.py      # Power-law fitting, H_QG estimation
├── exp/
│   ├── exp1.py                 # Experiment 1: topology visualization
│   ├── exp2.py                 # Experiment 2: Hurst estimation
│   └── exp3.py                 # Experiment 3: sensitivity to Q
├── tests/
│   └── test_quantile_graph.py  # pytest test suite
├── results/                    # Generated figures, metrics, RESULTS.md
├── reproduce.sh                # Entry point to run all experiments
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run all experiments

```bash
bash reproduce.sh
```

### 3. Run tests only

```bash
pytest tests/ -v
```

---

## Dependencies

- Python ≥ 3.9
- numpy, scipy, matplotlib — core numerics and plotting
- fbm — fractional Brownian motion / Gaussian noise generation

All dependencies are in `requirements.txt`.

---

## Results

Generated figures and metrics are saved in `results/`:

| File | Description |
|------|-------------|
| `exp1_transition_matrices.png` | W_1 heatmaps for 6 β values (Exp 1) |
| `exp1_metrics.json` | Diagonal/anti-diagonal dominance metrics |
| `exp2_hurst_vs_exact.png` | H_QG vs exact H with 95% CI (Exp 2) |
| `exp2_delta_k_scaling.png` | log Δ(k) vs log k for selected H (Exp 2) |
| `exp2_metrics.json` | H_QG estimates, errors, CI values |
| `exp3_sensitivity_Q.png` | |H_QG − H| vs β for Q=50 and Q=100 (Exp 3) |
| `exp3_metrics.json` | Sensitivity analysis metrics |
| `RESULTS.md` | Full comparison with paper reported values |
