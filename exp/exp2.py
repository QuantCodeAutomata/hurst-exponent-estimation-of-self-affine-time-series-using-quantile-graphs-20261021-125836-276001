"""
Experiment 2: Hurst exponent estimation via mean jump length power-law scaling
on Quantile Graphs.

Tests the core hypothesis that H_QG — the slope of log(Delta(k)) vs log(k) —
recovers the exact Hurst exponent H of fractional Brownian motion (paper §III).

Setup (paper §III):
  - T = 10 000,  Q = 50
  - 100 independent fBm realizations per H value
  - H in {0.1, 0.2, …, 0.9}  (beta = 2H+1 in {1.2, …, 2.8})
  - k from 1 to k_max = T // 10 = 1 000 (log-spaced, ~50 values)
  - H_QG estimated as the OLS slope of log Delta(k) vs log k
  - Results reported as mean ± 95 % CI over 100 realizations
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time

from data.generate_synthetic import generate_ensemble, hurst_from_beta
from src.hurst_estimator import (
    estimate_hurst_ensemble,
    get_delta_k_ensemble_mean,
    estimate_hurst_single,
)
from src.quantile_graph import compute_delta_k

# ── Parameters (paper §III) ──────────────────────────────────────────────────
T: int = 10_000
Q: int = 50
N_REALIZATIONS: int = 100
BASE_SEED: int = 0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# H values for fBm (beta = 2H+1 in (1, 3))
H_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Log-spaced k values covering ~3 decades (1 to T//10 = 1000)
K_MAX: int = T // 10
_k_raw = np.unique(np.round(np.logspace(0, np.log10(K_MAX), 50)).astype(int))
K_VALUES: np.ndarray = _k_raw[_k_raw >= 1]


def run_exp2() -> dict:
    """Run experiment 2 and return H_QG estimates vs exact H.

    Returns
    -------
    dict with keys "H_values", "H_qg_mean", "H_qg_ci95", "abs_error".
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    H_qg_mean = np.empty(len(H_VALUES))
    H_qg_ci95 = np.empty(len(H_VALUES))

    t0 = time.time()
    for idx, H in enumerate(H_VALUES):
        beta = 2.0 * H + 1.0
        print(f"[exp2] H={H:.1f} (beta={beta:.1f}) ...", end=" ", flush=True)
        t1 = time.time()

        ensemble = generate_ensemble(T=T, beta=beta, n_realizations=N_REALIZATIONS,
                                     base_seed=BASE_SEED + idx * 1000)
        mean_H, ci = estimate_hurst_ensemble(ensemble, Q=Q, k_values=K_VALUES)
        H_qg_mean[idx] = mean_H
        H_qg_ci95[idx] = ci

        elapsed = time.time() - t1
        print(f"H_QG={mean_H:.4f}  CI95=±{ci:.4f}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"[exp2] Total: {total_elapsed:.1f}s")

    abs_error = np.abs(H_qg_mean - H_VALUES)

    # ── Figures ───────────────────────────────────────────────────────────

    # Figure A: H_QG vs H (paper Fig. 3-like)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(
        H_VALUES, H_qg_mean, yerr=H_qg_ci95,
        fmt="o-", color="steelblue", ecolor="steelblue", capsize=4,
        label="$H_{QG}$ (mean ± 95% CI)",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal $H_{QG} = H$")
    ax.set_xlabel("Exact Hurst exponent H", fontsize=12)
    ax.set_ylabel("Estimated $H_{QG}$", fontsize=12)
    ax.set_title(
        f"QG Hurst estimation — fBm (T={T}, Q={Q}, {N_REALIZATIONS} realizations)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig_path_a = os.path.join(RESULTS_DIR, "exp2_hurst_vs_exact.png")
    fig.savefig(fig_path_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp2] Saved: {fig_path_a}")

    # Figure B: log-log Delta(k) vs k for selected H values (paper Fig. 2-like)
    fig, ax = plt.subplots(figsize=(8, 6))
    selected_H = [0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_H)))
    for H_sel, color in zip(selected_H, colors):
        idx_sel = np.argmin(np.abs(H_VALUES - H_sel))
        beta_sel = 2.0 * H_VALUES[idx_sel] + 1.0
        ens_sel = generate_ensemble(T=T, beta=beta_sel, n_realizations=10,
                                    base_seed=99999 + idx_sel * 100)
        delta_mean = get_delta_k_ensemble_mean(ens_sel, Q=Q, k_values=K_VALUES)
        valid = np.isfinite(delta_mean) & (delta_mean > 0)
        ax.loglog(K_VALUES[valid], delta_mean[valid],
                  color=color, linewidth=1.5,
                  label=f"H = {H_VALUES[idx_sel]:.1f}")

    ax.set_xlabel("Lag k", fontsize=12)
    ax.set_ylabel("Mean jump length Δ(k)", fontsize=12)
    ax.set_title("Power-law scaling of Δ(k) for fBm", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig_path_b = os.path.join(RESULTS_DIR, "exp2_delta_k_scaling.png")
    fig.savefig(fig_path_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp2] Saved: {fig_path_b}")

    # ── Save metrics ──────────────────────────────────────────────────────
    results = {
        "H_values": H_VALUES.tolist(),
        "H_qg_mean": H_qg_mean.tolist(),
        "H_qg_ci95": H_qg_ci95.tolist(),
        "abs_error": abs_error.tolist(),
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "k_values": K_VALUES.tolist(),
        "k_max": K_MAX,
        "T": T,
        "Q": Q,
        "n_realizations": N_REALIZATIONS,
    }
    metrics_path = os.path.join(RESULTS_DIR, "exp2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp2] Saved metrics: {metrics_path}")

    print("\n[exp2] Summary:")
    print(f"  H values:     {H_VALUES.tolist()}")
    print(f"  H_QG means:   {[f'{v:.4f}' for v in H_qg_mean]}")
    print(f"  Abs errors:   {[f'{v:.4f}' for v in abs_error]}")
    print(f"  Max |H_QG-H|: {np.max(abs_error):.4f}  (paper expects < 0.02)")

    return results


if __name__ == "__main__":
    run_exp2()
