#!/usr/bin/env bash
# reproduce.sh — run all three experiments from the paper
# Usage: bash reproduce.sh
set -u

echo "========================================================"
echo " Hurst Exponent Estimation via Quantile Graphs"
echo " Reproducing Campanharo & Ramos (2016)"
echo "========================================================"

# Ensure results directory exists
mkdir -p results

echo ""
echo "--- Experiment 1: Topology of QG transition matrices ---"
python3 exp/exp1.py

echo ""
echo "--- Experiment 2: Hurst exponent estimation (power-law scaling) ---"
python3 exp/exp2.py

echo ""
echo "--- Experiment 3: Sensitivity to number of quantiles Q ---"
python3 exp/exp3.py

echo ""
echo "========================================================"
echo " All experiments complete. Results saved in results/"
echo "========================================================"
