"""
run_multi_seed.py
─────────────────
Multi-seed stability experiment (mean ± std) and the
Wilcoxon signed-rank test:

    "Improving ICS Security Through Honeynets and Machine Learning Techniques"

Rationale:
  Both k-center and k-means use a random starting point (k-center: the
  first cluster centre; k-means: random centroid initialisation).
  Running both across 10 independent seeds confirms that the performance
  margins are stable rather than artefacts of a single initialisation.

Output:
  results/multi_seed_results.json   ← raw per-seed metrics + test statistics

Usage:
    python experiments/run_multi_seed.py
    python experiments/run_multi_seed.py --seeds 1 2 3 4 5   # custom subset
    python experiments/run_multi_seed.py --quiet              # suppress progress
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    RESULTS_DIR,
    K_CLUSTERS,
    ALPHA_THRESHOLD,
)
from data.traffic_generator import SCADATrafficGenerator
from features.feature_engineering import FeatureEngineer
from models.k_center import KCenterClustering
from models.k_means_detector import KMeansDetector
from evaluation.metrics import (
    detection_rate,
    false_positive_rate,
    f1_score_manual,
)

# ── Default seed list (paper: seeds 1–10) ─────────────────────────────────────
DEFAULT_SEEDS: List[int] = list(range(1, 11))


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon Signed-Rank Test (pure NumPy — no scipy dependency)
# Two-tailed, exact small-sample critical values for n ≤ 15.
# Reference: Wilcoxon (1945); Hollander & Wolfe (1999) Table A.4.
# ─────────────────────────────────────────────────────────────────────────────

# Critical values for two-tailed test: reject H₀ if W ≤ cv[α][n]
_CRITICAL_VALUES: Dict[str, Dict[int, int]] = {
    "0.05": {5: 0,  6: 2,  7: 3,  8: 5,  9: 7,  10: 8,  11: 10, 12: 13, 13: 17, 14: 21, 15: 25},
    "0.02": {5: -1, 6: 0,  7: 1,  8: 3,  9: 5,  10: 7,  11: 9,  12: 12, 13: 15, 14: 19, 15: 23},
    "0.01": {5: -1, 6: -1, 7: 0,  8: 1,  9: 3,  10: 5,  11: 7,  12: 9,  13: 12, 14: 15, 15: 19},
    "0.002":{5: -1, 6: -1, 7: -1, 8: -1, 9: 0,  10: 3,  11: 5,  12: 7,  13: 9,  14: 12, 15: 15},
}


def wilcoxon_signed_rank(
    x: List[float],
    y: List[float],
) -> Dict[str, object]:
    """
    Two-tailed Wilcoxon signed-rank test for paired samples (x vs y).
    Returns W+ , W-, W (min), p-bracket, and Cliff's delta.
    """
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), d) for d in diffs if d != 0.0]
    n = len(nonzero)

    if n == 0:
        return {"W_plus": 0, "W_minus": 0, "W": 0, "n": 0,
                "p_bracket": "p = 1.000", "cliff_delta": 0.0}

    # Rank absolute differences; average ranks on ties
    sorted_nd = sorted(nonzero, key=lambda t: t[0])
    ranks: List[Tuple[float, float]] = []
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_nd[j][0] == sorted_nd[j + 1][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks.append((avg_rank, sorted_nd[k][1]))
        i = j + 1

    W_plus  = sum(r for r, d in ranks if d > 0)
    W_minus = sum(r for r, d in ranks if d < 0)
    W = min(W_plus, W_minus)

    # Determine p-bracket from critical value table
    p_bracket = "p > 0.05"
    if n in _CRITICAL_VALUES["0.05"] and W <= _CRITICAL_VALUES["0.05"][n]:
        p_bracket = "p < 0.05"
    if n in _CRITICAL_VALUES["0.02"] and W <= _CRITICAL_VALUES["0.02"][n]:
        p_bracket = "p < 0.02"
    if n in _CRITICAL_VALUES["0.01"] and W <= _CRITICAL_VALUES["0.01"][n]:
        p_bracket = "p < 0.01"
    if n in _CRITICAL_VALUES["0.002"] and W <= _CRITICAL_VALUES["0.002"][n]:
        p_bracket = "p < 0.002"
    # Exact p = 0.002 when W = 0 and n = 10 (all differences same sign)
    if n == 10 and W == 0:
        p_bracket = "p = 0.002"

    # Cliff's delta: proportion of concordant minus discordant pairs
    concordant = sum(1 for xi in x for yi in y if xi > yi)
    discordant = sum(1 for xi in x for yi in y if xi < yi)
    cliff_delta = (concordant - discordant) / (len(x) * len(y))

    return {
        "W_plus":      W_plus,
        "W_minus":     W_minus,
        "W":           W,
        "n":           n,
        "p_bracket":   p_bracket,
        "cliff_delta": round(cliff_delta, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    seed: int,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Train k-center and k-means with the given random seed on the same
    synthetic dataset and return DR, FPR, F1 for each.

    The dataset generator uses a fixed global seed so that the traffic
    distribution is identical across all seed sweeps; only the model
    initialisation varies.
    """
    # Fixed dataset seed (42) — only model seed varies
    gen = SCADATrafficGenerator(seed=42)
    train_ds, val_ds, test_ds = gen.generate_full_dataset()
    test_ds_km = gen.generate_test_set_kmeans()

    fe = FeatureEngineer()
    X_train = fe.fit_normalize(train_ds.X)
    X_test_kc = fe.normalize(test_ds.X)
    X_test_km = fe.normalize(test_ds_km.X)
    y_test_kc = test_ds.y
    y_test_km = test_ds_km.y

    # ── k-Center ──────────────────────────────────────────────────────────────
    kc = KCenterClustering(k=K_CLUSTERS, alpha=ALPHA_THRESHOLD, seed=seed)
    kc.fit(X_train)
    y_pred_kc = kc.predict(X_test_kc)

    kc_dr  = round(detection_rate(y_test_kc, y_pred_kc) * 100, 2)
    kc_fpr = round(false_positive_rate(y_test_kc, y_pred_kc) * 100, 2)
    kc_f1  = round(f1_score_manual(y_test_kc, y_pred_kc), 3)

    # ── k-Means ───────────────────────────────────────────────────────────────
    km = KMeansDetector(k=K_CLUSTERS, alpha=ALPHA_THRESHOLD, seed=seed)
    km.fit(X_train)
    y_pred_km = km.predict(X_test_km)

    km_dr  = round(detection_rate(y_test_km, y_pred_km) * 100, 2)
    km_fpr = round(false_positive_rate(y_test_km, y_pred_km) * 100, 2)
    km_f1  = round(f1_score_manual(y_test_km, y_pred_km), 3)

    if verbose:
        print(
            f"    Seed {seed:>2d} │ "
            f"k-center: DR={kc_dr:5.2f}%  FPR={kc_fpr:.2f}%  F1={kc_f1:.3f}  │  "
            f"k-means:  DR={km_dr:5.2f}%  FPR={km_fpr:.2f}%  F1={km_f1:.3f}"
        )

    return {
        "k_center": {"DR": kc_dr, "FPR": kc_fpr, "F1": kc_f1},
        "k_means":  {"DR": km_dr, "FPR": km_fpr, "F1": km_f1},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed sweep
# ─────────────────────────────────────────────────────────────────────────────

def run(
    seeds: List[int] = DEFAULT_SEEDS,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run both detectors across all seeds and compute aggregate statistics.

    Returns
    -------
    dict with keys:
        per_seed       : list of per-seed results
        k_center_stats : mean/std per metric
        k_means_stats  : mean/std per metric
        wilcoxon       : Wilcoxon test results per metric
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sep = "=" * 75
    if verbose:
        print(f"\n{sep}")
        print("  CamouflageNet — Multi-Seed Stability Experiment")
        print(f"  Seeds: {seeds}")
        print(sep)
        print(
            f"\n  {'Seed':>6}  "
            f"{'k-center DR':>12}  {'FPR':>6}  {'F1':>6}  │  "
            f"{'k-means DR':>10}  {'FPR':>6}  {'F1':>6}"
        )
        print("  " + "─" * 73)

    t0 = time.time()
    per_seed: List[Dict] = []

    for seed in seeds:
        result = _run_one_seed(seed, verbose=verbose)
        per_seed.append({"seed": seed, **result})

    # ── Aggregate statistics ──────────────────────────────────────────────────
    metrics = ["DR", "FPR", "F1"]

    def _agg(model_key: str) -> Dict[str, Dict[str, float]]:
        agg = {}
        for m in metrics:
            vals = [r[model_key][m] for r in per_seed]
            agg[m] = {
                "mean": round(float(np.mean(vals)), 3),
                "std":  round(float(np.std(vals, ddof=1)), 3),
                "min":  round(float(np.min(vals)), 3),
                "max":  round(float(np.max(vals)), 3),
                "values": vals,
            }
        return agg

    kc_stats = _agg("k_center")
    km_stats = _agg("k_means")

    # ── Wilcoxon signed-rank tests ────────────────────────────────────────────
    wilcoxon: Dict[str, Dict] = {}
    for m in metrics:
        kc_vals = [r["k_center"][m] for r in per_seed]
        km_vals = [r["k_means"][m]  for r in per_seed]
        wilcoxon[m] = wilcoxon_signed_rank(kc_vals, km_vals)

    elapsed = time.time() - t0

    # ── Console summary ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Completed {len(seeds)} seeds in {elapsed:.1f} s\n")
        print(f"  {sep}")
        print("  AGGREGATE RESULTS  (mean ± std across seeds)")
        print(f"  {sep}")
        print(
            f"  {'Metric':<8}  "
            f"{'k-center':>18}  "
            f"{'k-means':>18}  "
            f"{'Mean Δ':>8}  "
            f"{'W':>5}  "
            f"{'p':>10}  "
            f"{'Cliff δ':>8}"
        )
        print("  " + "─" * 83)
        for m in metrics:
            kc = kc_stats[m]
            km = km_stats[m]
            w  = wilcoxon[m]
            delta = round(kc["mean"] - km["mean"], 3)
            unit = "%" if m in ("DR", "FPR") else ""
            print(
                f"  {m:<8}  "
                f"{kc['mean']:>6.3f}{unit} ± {kc['std']:.3f}  "
                f"{km['mean']:>6.3f}{unit} ± {km['std']:.3f}  "
                f"{delta:>+8.3f}  "
                f"{w['W']:>5.1f}  "
                f"{w['p_bracket']:>10}  "
                f"{w['cliff_delta']:>+8.3f}"
            )
        print(f"\n  Paper Table 2 reproduces: k-center DR = "
              f"{kc_stats['DR']['mean']:.2f}% ± {kc_stats['DR']['std']:.2f}%  "
              f"(paper: 96.60% ± 0.28%)")
        print(f"  Wilcoxon DR: W={wilcoxon['DR']['W']:.0f}, "
              f"{wilcoxon['DR']['p_bracket']}, "
              f"Cliff's δ={wilcoxon['DR']['cliff_delta']:.2f}  "
              f"(paper: W=0, p=0.002, δ=1.00)")

    # ── Compile output dict ───────────────────────────────────────────────────
    output = {
        "experiment": "multi_seed_stability",
        "seeds":      seeds,
        "n_seeds":    len(seeds),
        "elapsed_s":  round(elapsed, 1),
        "per_seed":   per_seed,
        "k_center_stats": kc_stats,
        "k_means_stats":  km_stats,
        "wilcoxon":       wilcoxon,
        "paper_table2": {
            "k_center": {
                m: f"{kc_stats[m]['mean']:.3f} ± {kc_stats[m]['std']:.3f}"
                for m in metrics
            },
            "k_means": {
                m: f"{km_stats[m]['mean']:.3f} ± {km_stats[m]['std']:.3f}"
                for m in metrics
            },
            "wilcoxon": {
                m: {
                    "W":           wilcoxon[m]["W"],
                    "p_bracket":   wilcoxon[m]["p_bracket"],
                    "cliff_delta": wilcoxon[m]["cliff_delta"],
                }
                for m in metrics
            },
        },
    }

    if save_results:
        out_path = os.path.join(RESULTS_DIR, "multi_seed_results.json")
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        if verbose:
            print(f"\n  Results saved → {out_path}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed stability experiment for CamouflageNet paper."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of random seeds to evaluate (default: 1 2 … 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-seed progress output",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results/multi_seed_results.json",
    )
    args = parser.parse_args()

    run(
        seeds=args.seeds,
        save_results=not args.no_save,
        verbose=not args.quiet,
    )
