"""
Experiment 1: Topology characterization of Quantile Graphs for self-affine
time series with varying correlation.

Generates self-affine time series for beta in {-1.0, -0.5, 0.0, 0.5, 1.0, 2.0},
builds the lag-1 Markov transition matrix W_1, and visualises it as a heatmap.

Expected: anti-correlated series (beta < 0) produce anti-diagonally dominant
W_1; white noise (beta = 0) produces uniform W_1; correlated series (beta > 0)
produce diagonally dominant W_1 (paper §II, Fig. 1).
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

from data.generate_synthetic import generate_series
from src.quantile_graph import build_w1

# ── Parameters (paper §II) ──────────────────────────────────────────────────
T: int = 10_000          # time series length
Q: int = 50              # number of quantile nodes
BETA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # power-spectrum exponents
SEED: int = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_exp1() -> dict:
    """Run experiment 1 and return topology metrics for all beta values.

    Returns
    -------
    dict
        Mapping beta_value → {"diagonal_dominance": float,
                               "anti_diagonal_dominance": float}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics: dict = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, beta in enumerate(BETA_VALUES):
        series = generate_series(T=T, beta=beta, seed=SEED + idx)
        W1 = build_w1(series, Q=Q)

        # ── Topology metrics ──────────────────────────────────────────────
        # Main-diagonal dominance: mean weight on the main diagonal
        diag_dom = float(np.mean(np.diag(W1)))

        # Anti-diagonal dominance: mean weight on the anti-diagonal
        anti_diag = np.array([W1[i, Q - 1 - i] for i in range(Q)])
        anti_dom = float(np.mean(anti_diag))

        metrics[beta] = {
            "diagonal_dominance": diag_dom,
            "anti_diagonal_dominance": anti_dom,
        }

        # ── Plot heatmap ──────────────────────────────────────────────────
        ax = axes[idx]
        im = ax.imshow(
            W1,
            origin="lower",
            aspect="auto",
            cmap="hot_r",
            interpolation="nearest",
        )
        ax.set_title(f"β = {beta}", fontsize=13)
        ax.set_xlabel("Node j (quantile at t+1)", fontsize=9)
        ax.set_ylabel("Node i (quantile at t)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Quantile Graph transition matrices W₁\n"
        "for self-affine time series (T=10 000, Q=50)",
        fontsize=13,
    )
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "exp1_transition_matrices.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1] Saved heatmap: {fig_path}")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_serializable = {str(k): v for k, v in metrics.items()}
    metrics_path = os.path.join(RESULTS_DIR, "exp1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"[exp1] Saved metrics: {metrics_path}")

    for beta, m in metrics.items():
        print(
            f"  β={beta:+.1f}  diag_dom={m['diagonal_dominance']:.4f}"
            f"  anti_diag_dom={m['anti_diagonal_dominance']:.4f}"
        )

    return metrics


if __name__ == "__main__":
    run_exp1()
