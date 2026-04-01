"""
k_center.py
───────────
Exact implementation of Algorithm 1 from the paper:

    "Greedy k-Center Clustering for Behavioral Profiling"

Problem formulation (Equation 2):
    min_C  max_{x∈V}  d(x, C)
    where  d(x, C) = min_{c∈C} ‖x − c‖₂

The greedy approximation algorithm:
  1. Pick c₁ arbitrarily from V.
  2. While |C| < k:
       x* ← argmax_{x∈V} min_{c∈C} ‖x − c‖₂
       C  ← C ∪ {x*}
  3. Assign each xᵢ to its nearest center.

Anomaly detection:
  • After training, record r_max = radius of the largest cluster.
  • For a new session x_new:
      d_min ← min_{c∈C} ‖x_new − c‖₂
      if d_min > α · r_max  →  flag as ANOMALY
      else                  →  assign to nearest cluster (Normal)

Parameters:
  k = 45  (Elbow Method, tested 10–100)
  α = 1.5 (grid search over {1.0, 1.25, 1.5, 1.75, 2.0})
  random_seed = 42
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import K_CLUSTERS, ALPHA_THRESHOLD, RANDOM_SEED


class KCenterClustering:
    """
    Greedy k-Center Clustering with integrated anomaly detection.

    Unlike k-means, which minimises *variance*, k-center minimises the
    *maximum diameter* of the clusters, ensuring no data point is farther
    than a bounded distance from its nearest centre.  This minimax
    property prevents rare attack behaviours from being absorbed into
    large benign clusters — the key advantage over k-means for ICS
    anomaly detection.

    The greedy algorithm guarantees a 2-approximation to the optimal
    solution (Gonzalez 1985 [29]).
    """

    def __init__(
        self,
        k: int   = K_CLUSTERS,
        alpha: float = ALPHA_THRESHOLD,
        seed: int    = RANDOM_SEED,
    ):
        """
        Parameters
        ----------
        k     : number of cluster centres (default 45 from paper)
        alpha : anomaly threshold multiplier (default 1.5 from paper)
        seed  : random seed for first-centre selection (default 42)
        """
        self.k     = k
        self.alpha = alpha
        self.seed  = seed

        # Fitted attributes
        self.centers_: Optional[np.ndarray] = None   # (k, d)
        self.labels_:  Optional[np.ndarray] = None   # (n,) cluster indices
        self.r_max_:   Optional[float]      = None   # radius of largest cluster
        self._n_train: int = 0
        self._fitted   = False

    # ──────────────────────────────────────────────────────────────────────────
    # Fit  (training phase)
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KCenterClustering":
        """
        Run the greedy k-center algorithm on X (n_samples × d).

        After fitting:
          self.centers_  — the k selected cluster centres
          self.labels_   — cluster assignment for every training point
          self.r_max_    — maximum cluster radius (used for anomaly threshold)
        """
        n, d = X.shape
        rng  = np.random.default_rng(self.seed)

        # ── Step 1: pick the first centre at random ──────────────────────────
        first_idx = rng.integers(0, n)
        centers_idx = [first_idx]

        # Maintain a running "minimum distance to current centre set" vector
        # for O(n·k) total complexity instead of O(n·k²).
        dist_to_nearest = np.full(n, np.inf)
        dist_to_nearest = self._update_min_distances(
            X, dist_to_nearest, X[first_idx]
        )

        # ── Step 2: greedy expansion ─────────────────────────────────────────
        while len(centers_idx) < self.k:
            # x* ← argmax_{x∈V} min_{c∈C} ‖x − c‖₂
            new_idx = int(np.argmax(dist_to_nearest))
            centers_idx.append(new_idx)
            dist_to_nearest = self._update_min_distances(
                X, dist_to_nearest, X[new_idx]
            )

        self.centers_ = X[np.array(centers_idx)]   # (k, d)

        # ── Step 3: assign all training points to nearest centre ─────────────
        self.labels_, all_dists = self._assign(X)

        # ── Compute r_max (largest cluster radius) ────────────────────────────
        #   Used as the baseline radius for the anomaly threshold α·r_max.
        cluster_radii = np.zeros(self.k)
        for ci in range(self.k):
            mask = self.labels_ == ci
            if mask.any():
                cluster_radii[ci] = all_dists[mask].max()
        self.r_max_ = float(cluster_radii.max())

        self._n_train = n
        self._fitted  = True
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Predict / detect  (detection phase)
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary anomaly prediction (Algorithm 1, Anomaly Detection block).

        Returns
        -------
        y_pred : ndarray of shape (n,)
                 1 = ANOMALY (forward to IPS)
                 0 = Normal  (assign to nearest cluster)
        """
        self._check_fitted()
        labels, dists = self._assign(X)
        threshold = self.alpha * self.r_max_
        return (dists > threshold).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw distance-to-nearest-centre score for each sample.
        Higher → more anomalous.
        """
        self._check_fitted()
        _, dists = self._assign(X)
        return dists

    def assign_clusters(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cluster_labels, distances_to_nearest_centre)."""
        self._check_fitted()
        return self._assign(X)

    # ──────────────────────────────────────────────────────────────────────────
    # Model Recalibration
    # ──────────────────────────────────────────────────────────────────────────
    #
    # Paper:
    #   "the proposed implementation supports periodic model recalibration:
    #    r_max is recomputed from a sliding window of verified benign sessions
    #    at configurable intervals (set to 24 hours in the present experiments).
    #    If the recalibrated r_max changes by more than 10% relative to the
    #    current value, the cluster model is rebuilt from scratch using the
    #    updated benign window, and the IPS is notified to suspend automated
    #    blocking during the brief retraining window (typically under 90 seconds
    #    for the 45-node topology)."

    def recalibrate(
        self,
        X_benign_window: np.ndarray,
        drift_threshold: float = 0.10,
    ) -> dict:
        """
        Periodic model recalibration from a sliding window of verified.

        Parameters
        ----------
        X_benign_window : normalised feature matrix of recent verified-benign
                          sessions (e.g., the last 24 hours of clean traffic)
        drift_threshold : relative r_max change that triggers a full rebuild
                          (default 10 % as specified in the paper)

        Returns
        -------
        dict with keys:
            recalibrated : bool — whether r_max was updated
            rebuilt      : bool — whether the full model was rebuilt
            old_r_max    : float
            new_r_max    : float
            drift_pct    : float — |new − old| / old × 100
        """
        self._check_fitted()

        # Recompute r_max on the new benign window using existing centres
        _, dists = self._assign(X_benign_window)
        new_r_max = float(dists.max())
        old_r_max = self.r_max_

        drift_pct = abs(new_r_max - old_r_max) / old_r_max if old_r_max > 0 else 0.0

        rebuilt = False
        if drift_pct > drift_threshold:
            # Full rebuild: re-fit from scratch on the new benign window
            self.fit(X_benign_window)
            rebuilt = True
            new_r_max = self.r_max_
        else:
            # Update r_max only (centres unchanged)
            self.r_max_ = new_r_max

        return {
            "recalibrated": True,
            "rebuilt":      rebuilt,
            "old_r_max":    old_r_max,
            "new_r_max":    new_r_max,
            "drift_pct":    round(drift_pct * 100, 2),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Elbow Method helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def elbow_method(
        X: np.ndarray,
        k_range: range,
        seed: int = RANDOM_SEED,
    ) -> Tuple[List[int], List[float]]:
        """
        Run k-center for each k in k_range and return the max-radius at
        each k.  The "elbow" in the curve identifies the optimal k.

        Parameters
        ----------
        X       : training feature matrix
        k_range : range of k values to test (paper: range(10, 101))
        seed    : random seed

        Returns
        -------
        ks       : list of k values
        radii    : corresponding r_max values
        """
        ks, radii = [], []
        for k in k_range:
            model = KCenterClustering(k=k, seed=seed)
            model.fit(X)
            ks.append(k)
            radii.append(model.r_max_)
        return ks, radii

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _update_min_distances(
        X: np.ndarray,
        current_min: np.ndarray,
        new_center: np.ndarray,
    ) -> np.ndarray:
        """
        Efficiently update the running min-distance vector after adding
        new_center to the centre set.  Only updates entries where the
        distance to new_center is smaller than the current minimum.
        """
        dists = np.linalg.norm(X - new_center, axis=1)
        return np.minimum(current_min, dists)

    def _assign(self, X: np.ndarray, batch_size: int = 20_000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign each row in X to its nearest centre.

        Uses batched computation to avoid materialising the full (n, k, d)
        tensor in memory — safe for large datasets.

        Returns
        -------
        labels : (n,) cluster index
        dists  : (n,) distance to nearest centre
        """
        n = len(X)
        labels = np.empty(n, dtype=np.int64)
        dists  = np.empty(n, dtype=np.float64)

        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            Xb   = X[start:end]
            diff = Xb[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]  # (b,k,d)
            bd   = np.linalg.norm(diff, axis=2)                             # (b,k)
            labels[start:end] = np.argmin(bd, axis=1)
            dists[start:end]  = bd[np.arange(end - start), labels[start:end]]

        return labels, dists

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    # ──────────────────────────────────────────────────────────────────────────
    # Repr
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self):
        status = f"fitted, k={self.k}, r_max={self.r_max_:.4f}" \
                 if self._fitted else "not fitted"
        return (f"KCenterClustering(k={self.k}, alpha={self.alpha}, "
                f"seed={self.seed}) [{status}]")


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.traffic_generator import SCADATrafficGenerator
    from features.feature_engineering import FeatureEngineer

    gen = SCADATrafficGenerator()
    train_ds, _, test_ds = gen.generate_full_dataset()

    fe = FeatureEngineer()
    X_train = fe.fit_normalize(train_ds.X)
    X_test  = fe.normalize(test_ds.X)

    model = KCenterClustering()
    model.fit(X_train)
    print(model)
    print(f"r_max = {model.r_max_:.4f}")
    print(f"Anomaly threshold = {model.alpha} × {model.r_max_:.4f} "
          f"= {model.alpha * model.r_max_:.4f}")

    y_pred = model.predict(X_test)
    y_true = test_ds.y
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    dr  = tp / (tp + fn) * 100
    fpr = fp / (fp + tn) * 100
    f1  = 2*tp / (2*tp + fp + fn)
    print(f"\nDetection Rate : {dr:.1f} %  (paper: 96.5 %)")
    print(f"FPR            : {fpr:.1f} %  (paper:  1.8 %)")
    print(f"F1-Score       : {f1:.3f}   (paper:  0.93)")
