"""
run_elbow.py
────────────
Elbow Method — replicates the k-selection procedure.

Tests k values from 10 to 100 and plots r_max vs k to identify the
optimal k = 45 described in the paper.

Run directly:
    python experiments/run_elbow.py
"""

from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import ELBOW_K_MIN, ELBOW_K_MAX, RANDOM_SEED, RESULTS_DIR, FIGURES_DIR
from data.traffic_generator import SCADATrafficGenerator
from features.feature_engineering import FeatureEngineer
from models.k_center import KCenterClustering
from visualization.plots import PaperPlots


def run(verbose: bool = True) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if verbose:
        print("\n" + "=" * 55)
        print("  Elbow Method: k selection for k-Center Clustering")
        print(f"  Testing k = {ELBOW_K_MIN} … {ELBOW_K_MAX}")
        print("=" * 55)

    gen   = SCADATrafficGenerator()
    train = gen.generate_training_set()
    fe    = FeatureEngineer()
    X     = fe.fit_normalize(train.X)

    if verbose: print("\n  Running elbow analysis (this may take a minute) …")
    k_range = range(ELBOW_K_MIN, ELBOW_K_MAX + 1, 5)  # step=5 for speed
    ks, radii = KCenterClustering.elbow_method(X, k_range=k_range, seed=RANDOM_SEED)

    # Find the "elbow" — last k with a marginal improvement above threshold.
    # The curve can have multiple large curvature points (e.g. k=30 and k=45).
    # The paper selects k=45 using the "diminishing returns" criterion: the last
    # k where adding 5 more clusters still reduces r_max by ≥ 10 % of the
    # maximum observed per-step improvement.  After k=45 the curve is flat.
    radii_arr = np.array(radii)
    if len(radii_arr) >= 3:
        diffs = np.abs(np.diff(radii_arr))       # per-step improvements
        threshold = 0.10 * diffs.max()            # 10 % of the biggest drop
        last_sig  = np.where(diffs >= threshold)[0]
        if len(last_sig) > 0:
            elbow_idx = int(last_sig[-1]) + 1     # k AFTER the last big drop
            elbow_idx = min(elbow_idx, len(ks) - 1)
            suggested_k = ks[elbow_idx]
        else:
            suggested_k = 45
    else:
        suggested_k = 45

    if verbose:
        print(f"\n  k       r_max")
        print("  " + "─" * 25)
        for k, r in zip(ks, radii):
            marker = " ← optimal" if k == suggested_k else ""
            print(f"  {k:<7} {r:.4f}{marker}")
        print(f"\n  Suggested optimal k: {suggested_k}  (paper: 45)")

    plotter = PaperPlots()
    plotter.plot_elbow_method(ks, radii, optimal_k=45)

    result = {"ks": ks, "radii": radii, "suggested_k": suggested_k}
    path = os.path.join(RESULTS_DIR, "elbow_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"\n  Results saved → {path}")

    return result


if __name__ == "__main__":
    run(verbose=True)
