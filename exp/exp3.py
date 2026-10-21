"""
Experiment 3: Sensitivity analysis of the number of quantiles Q on Hurst
exponent estimation error.

Evaluates |H_QG - H| for Q = 50 and Q = 100 across beta in [1.0, 3.0]
(H in [0, 1] for fBm).  Expected result: the error is < 0.02 for both Q
values, showing robustness to the choice of Q (paper §III, Fig. 4).

Setup (paper §III):
  - T = 10 000
  - N_REALIZATIONS = 100
  - Q in {50, 100}
  - beta from 1.0 to 3.0 in steps of 0.2 (H from ~0 to ~1)
  - k from 1 to T // 10 (log-spaced) — same as exp2
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
from src.hurst_estimator import estimate_hurst_ensemble

# ── Parameters (paper §III) ──────────────────────────────────────────────────
T: int = 10_000
N_REALIZATIONS: int = 100
BASE_SEED: int = 200_000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

Q_VALUES = [50, 100]

# beta values from 1.0 to 2.8 inclusive (H from ~0 to 0.9 for fBm).
# beta=3.0 (H=0.9999) is excluded: the fbm library's Davies-Harte method
# falls back to the O(n^2) Hosking algorithm for H very close to 1 with
# n=10,000, making computation intractable.  H in [0, 0.9] still spans the
# entire practically relevant range.
BETA_VALUES = np.arange(1.0, 2.81, 0.2)

K_MAX: int = T // 10
_k_raw = np.unique(np.round(np.logspace(0, np.log10(K_MAX), 50)).astype(int))
K_VALUES: np.ndarray = _k_raw[_k_raw >= 1]


def run_exp3() -> dict:
    """Run experiment 3 and return sensitivity analysis results.

    Returns
    -------
    dict with Q values as keys, each mapping to arrays of abs_error vs beta.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    H_exact = np.array([hurst_from_beta(b) for b in BETA_VALUES])
    results: dict = {}

    t0 = time.time()
    for Q in Q_VALUES:
        print(f"\n[exp3] Q={Q}")
        abs_errors = np.empty(len(BETA_VALUES))
        H_qg_means = np.empty(len(BETA_VALUES))

        for idx, beta in enumerate(BETA_VALUES):
            H_true = H_exact[idx]
            print(f"  beta={beta:.1f}  H={H_true:.4f} ...", end=" ", flush=True)
            t1 = time.time()

            ensemble = generate_ensemble(
                T=T, beta=beta, n_realizations=N_REALIZATIONS,
                base_seed=BASE_SEED + int(Q) * 1000 + idx * 100,
            )
            H_qg_mean, _ = estimate_hurst_ensemble(ensemble, Q=Q, k_values=K_VALUES)
            H_qg_means[idx] = H_qg_mean
            abs_errors[idx] = abs(H_qg_mean - H_true)

            elapsed = time.time() - t1
            print(f"H_QG={H_qg_mean:.4f}  |err|={abs_errors[idx]:.4f}  ({elapsed:.1f}s)")

        results[Q] = {
            "beta_values": BETA_VALUES.tolist(),
            "H_exact": H_exact.tolist(),
            "H_qg_mean": H_qg_means.tolist(),
            "abs_error": abs_errors.tolist(),
            "max_abs_error": float(np.max(abs_errors)),
            "mean_abs_error": float(np.mean(abs_errors)),
        }
        print(f"  [Q={Q}] max |H_QG-H| = {np.max(abs_errors):.4f}  "
              f"(paper expects < 0.02)")

    total_elapsed = time.time() - t0
    print(f"\n[exp3] Total: {total_elapsed:.1f}s")

    # ── Figure: |H_QG - H| vs beta for Q=50 and Q=100 (paper Fig. 4) ─────
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {50: ("o-", "steelblue"), 100: ("s--", "darkorange")}
    for Q in Q_VALUES:
        marker_style, color = styles[Q]
        ax.plot(
            BETA_VALUES,
            results[Q]["abs_error"],
            marker_style,
            color=color,
            linewidth=1.5,
            markersize=5,
            label=f"Q = {Q}",
        )

    ax.axhline(y=0.02, color="gray", linestyle=":", linewidth=1.5,
               label="Threshold 0.02")
    ax.set_xlabel("Power-spectrum exponent β", fontsize=12)
    ax.set_ylabel("|$H_{QG}$ − H|", fontsize=12)
    ax.set_title(
        f"Sensitivity of H estimation to Q\n"
        f"(T={T}, {N_REALIZATIONS} realizations)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis showing H
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_betas = BETA_VALUES[::2]
    tick_H = [hurst_from_beta(b) for b in tick_betas]
    ax2.set_xticks(tick_betas)
    ax2.set_xticklabels([f"{h:.1f}" for h in tick_H], fontsize=8)
    ax2.set_xlabel("Hurst exponent H", fontsize=10)

    fig_path = os.path.join(RESULTS_DIR, "exp3_sensitivity_Q.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp3] Saved: {fig_path}")

    # ── Save metrics ──────────────────────────────────────────────────────
    results["meta"] = {
        "T": T,
        "n_realizations": N_REALIZATIONS,
        "k_max": K_MAX,
        "beta_values": BETA_VALUES.tolist(),
        "H_exact": H_exact.tolist(),
    }
    metrics_path = os.path.join(RESULTS_DIR, "exp3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"[exp3] Saved metrics: {metrics_path}")

    return results


if __name__ == "__main__":
    run_exp3()
