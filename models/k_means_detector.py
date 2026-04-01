"""
k_means_detector.py
───────────────────
Standard k-Means anomaly detector — main unsupervised baseline.
Implemented in pure NumPy (Lloyd's algorithm).

Paper result: 91.4 % DR, 3.5 % FPR, F1 = 0.87
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import K_CLUSTERS, ALPHA_THRESHOLD, RANDOM_SEED


class KMeansDetector:
    """
    Anomaly detector built on Lloyd's k-Means algorithm (pure NumPy).
    Uses the same threshold rule as k-center: d_min > alpha * r_max.
    """

    def __init__(
        self,
        k: int       = K_CLUSTERS,
        alpha: float = ALPHA_THRESHOLD,
        seed: int    = RANDOM_SEED,
        max_iter: int = 300,
        tol: float   = 1e-4,
    ):
        self.k        = k
        self.alpha    = alpha
        self.seed     = seed
        self.max_iter = max_iter
        self.tol      = tol

        self.centers_: Optional[np.ndarray] = None
        self.r_max_:   Optional[float]      = None
        self._fitted   = False

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape

        # k-Means++ initialisation
        centers = self._init_plusplus(X, rng)

        for _ in range(self.max_iter):
            # Assignment
            labels, _ = self._assign(X, centers)
            # Update centres
            new_centers = np.array([
                X[labels == ci].mean(axis=0) if (labels == ci).any() else centers[ci]
                for ci in range(self.k)
            ])
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.centers_ = centers
        labels, dists = self._assign(X, centers)

        # r_max
        radii = np.zeros(self.k)
        for ci in range(self.k):
            mask = labels == ci
            if mask.any():
                radii[ci] = dists[mask].max()
        self.r_max_ = float(radii.max())
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _, dists = self._assign(X, self.centers_)
        return (dists > self.alpha * self.r_max_).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _, dists = self._assign(X, self.centers_)
        return dists

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_plusplus(self, X: np.ndarray, rng) -> np.ndarray:
        """k-Means++ initialisation — fully vectorised, no Python loops over rows."""
        n = len(X)
        first = rng.integers(0, n)
        centers = [X[first]]
        # Running minimum squared distance to the current centre set
        min_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for _ in range(1, self.k):
            probs = min_sq / min_sq.sum()
            idx   = rng.choice(n, p=probs)
            centers.append(X[idx])
            new_sq  = np.sum((X - X[idx]) ** 2, axis=1)
            min_sq  = np.minimum(min_sq, new_sq)
        return np.array(centers)

    @staticmethod
    def _assign(X: np.ndarray, centers: np.ndarray,
                batch_size: int = 20_000) -> Tuple[np.ndarray, np.ndarray]:
        """Batched assignment to avoid large (n, k, d) tensors in memory."""
        n = len(X)
        labels = np.empty(n, dtype=np.int64)
        min_d  = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            Xb   = X[start:end]
            diff = Xb[:, np.newaxis, :] - centers[np.newaxis, :, :]
            bd   = np.linalg.norm(diff, axis=2)
            labels[start:end] = np.argmin(bd, axis=1)
            min_d[start:end]  = bd[np.arange(end - start), labels[start:end]]
        return labels, min_d

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def __repr__(self):
        status = f"fitted, k={self.k}, r_max={self.r_max_:.4f}" if self._fitted else "not fitted"
        return f"KMeansDetector(k={self.k}, alpha={self.alpha}) [{status}]"
