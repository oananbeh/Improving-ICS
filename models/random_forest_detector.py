"""
random_forest_detector.py
─────────────────────────
Random Forest classifier — pure NumPy implementation.
Supervised SOTA baseline (Alimi et al. [7]).
Paper result: DR = 98.1 %, FPR = 1.2 %, F1 = 0.96
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RANDOM_SEED


class _DecisionNode:
    """Single node in a decision tree."""
    __slots__ = ["feature", "threshold", "left", "right", "value"]
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # leaf: predicted probability


class _DecisionTree:
    """CART decision tree (Gini impurity) for binary classification."""

    def __init__(self, max_depth=10, min_samples_split=5,
                 max_features: Optional[int] = None, seed=0):
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.max_features     = max_features
        self.seed             = seed
        self._rng             = np.random.default_rng(seed)
        self.root             = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build(X, y, depth=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batch(X, self.root)

    def _predict_batch(self, X: np.ndarray, node) -> np.ndarray:
        """Vectorized batch prediction — all samples in parallel, no Python loop."""
        n = len(X)
        if node.value is not None:
            return np.full(n, node.value)
        mask = X[:, node.feature] <= node.threshold
        result = np.empty(n)
        if mask.any():
            result[mask]  = self._predict_batch(X[mask],  node.left)
        if (~mask).any():
            result[~mask] = self._predict_batch(X[~mask], node.right)
        return result

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p**2 - (1 - p)**2

    def _best_split(self, X, y):
        n, d = X.shape
        n_feat = self.max_features or d
        feat_idx = self._rng.choice(d, size=min(n_feat, d), replace=False)

        best_gain, best_f, best_t = -np.inf, None, None
        g_parent = self._gini(y)

        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(5, 95, 20))
            for t in thresholds:
                left  = y[X[:, f] <= t]
                right = y[X[:, f] >  t]
                if len(left) < 2 or len(right) < 2:
                    continue
                gain = g_parent - (
                    len(left)  / n * self._gini(left) +
                    len(right) / n * self._gini(right)
                )
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t
        return best_f, best_t

    def _build(self, X, y, depth):
        # Stopping criteria
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return _DecisionNode(value=float(y.mean()))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _DecisionNode(value=float(y.mean()))

        mask = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _DecisionNode(feature=feat, threshold=thresh, left=left, right=right)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class RandomForestDetector:
    """
    Ensemble of CART decision trees (Random Forest) for ICS intrusion detection.
    Pure NumPy — no scikit-learn required.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth:    int = 10,
        max_features_ratio: float = 0.6,
        threshold: float = 0.5,
        seed: int = RANDOM_SEED,
    ):
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.max_features_ratio  = max_features_ratio
        self.threshold           = threshold
        self.seed                = seed

        self._trees: List[_DecisionTree] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestDetector":
        n, d = X.shape
        max_f = max(1, int(d * self.max_features_ratio))
        rng   = np.random.default_rng(self.seed)

        self._trees = []
        for i in range(self.n_estimators):
            # Bootstrap sample
            idx  = rng.choice(n, size=n, replace=True)
            tree = _DecisionTree(
                max_depth=self.max_depth,
                max_features=max_f,
                seed=int(rng.integers(0, 2**31)),
            )
            tree.fit(X[idx], y[idx])
            self._trees.append(tree)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_scores(X) > self.threshold).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        probs = np.array([t.predict_proba(X) for t in self._trees])
        return probs.mean(axis=0)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(X, y) before predict().")

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"RandomForestDetector(n_estimators={self.n_estimators}) [{status}]"
