"""
feature_engineering.py
──────────────────────
  "Raw network logs are unstructured and unsuitable for direct algorithmic
   processing. The proposed Data Processing (DP) module transforms these
   logs into feature vectors."

Feature vector (Equation 1):
    x_i = [f_freq, f_vol, f_proto, f_cmd, f_err]

Where f_proto is a 3-dimensional one-hot vector, giving d = 7 in total.

All features are normalised to [0, 1] using min-max scaling:
    x̂ = (x − x_min) / (x_max − x_min)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PROTOCOL_ENCODINGS, CMD_WEIGHTS, TIME_WINDOW, FEATURE_DIM


# ─── Min-Max Normalizer ───────────────────────────────────────────────────────

class MinMaxNormalizer:
    """
    Min-max scaler fitted on training data.
    Prevents magnitude bias in the Euclidean distance computation
    """

    def __init__(self):
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "MinMaxNormalizer":
        """Fit min/max from training matrix X (n_samples × d)."""
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to [0, 1] using training statistics."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        denom = self._max - self._min
        # Avoid division by zero for constant features
        denom = np.where(denom == 0, 1.0, denom)
        return (X - self._min) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        denom = self._max - self._min
        denom = np.where(denom == 0, 1.0, denom)
        return X_scaled * denom + self._min


# ─── Feature Engineer ─────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms raw log-entry dicts into the 7-dimensional feature vector
    described in Equation (1) of the paper.

    Raw log entry schema (dict):
        source_ip   : str   – source IP address used for session aggregation
        timestamp   : float – Unix epoch of the request
        bytes       : int   – payload bytes of this single request
        protocol    : str   – one of {"modbus", "dnp3", "http"}
        function_code: str  – ICS function code label (see CMD_WEIGHTS)
        is_error    : bool  – whether the honeypot returned an error response

    The DP module aggregates all entries with the same source_ip within
    a TIME_WINDOW-second window into a single session event E_i, then
    computes the 7 features.
    """

    FEATURE_NAMES = ["f_freq", "f_vol",
                     "f_proto_modbus", "f_proto_dnp3", "f_proto_http",
                     "f_cmd", "f_err"]

    def __init__(self):
        self.normalizer = MinMaxNormalizer()

    # ── Public interface ──────────────────────────────────────────────────────

    def extract_from_logs(
        self,
        log_entries: List[Dict],
        window: float = TIME_WINDOW,
    ) -> np.ndarray:
        """
        Aggregate a list of raw log entries into session feature vectors.

        Parameters
        ----------
        log_entries : list of dicts, each with keys:
                      source_ip, timestamp, bytes, protocol,
                      function_code, is_error
        window      : session window in seconds (default 60 s)

        Returns
        -------
        X : ndarray of shape (n_sessions, 7)
        """
        # Group by (source_ip, window_id)
        sessions: Dict[Tuple, List] = {}
        for entry in log_entries:
            win_id = int(entry["timestamp"] // window)
            key = (entry["source_ip"], win_id)
            sessions.setdefault(key, []).append(entry)

        rows = []
        for entries in sessions.values():
            rows.append(self._vectorise_session(entries))
        return np.array(rows, dtype=float) if rows else np.empty((0, FEATURE_DIM))

    def extract_from_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Pass-through for pre-built feature matrices (e.g., from the
        SCADATrafficGenerator which already outputs the 7 features).
        Just returns X unchanged — the matrix is already in the right format.
        """
        return X

    def fit_normalizer(self, X: np.ndarray) -> "FeatureEngineer":
        """Fit the min-max normalizer on training data."""
        self.normalizer.fit(X)
        return self

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalise feature matrix to [0, 1]."""
        return self.normalizer.transform(X)

    def fit_normalize(self, X: np.ndarray) -> np.ndarray:
        """Fit normalizer and transform in one step."""
        return self.normalizer.fit_transform(X)

    # ── Session vectorisation ─────────────────────────────────────────────────

    def _vectorise_session(self, entries: List[Dict]) -> np.ndarray:
        """
        Convert a list of raw log entries (same source_ip, same window)
        into the 7-dimensional feature vector of Equation (1).
        """
        n = len(entries)

        # f_freq — number of requests issued within the time window
        f_freq = float(n)

        # f_vol — total bytes transferred during the session
        f_vol = float(sum(e["bytes"] for e in entries))

        # f_proto — majority-vote one-hot encoding of the protocol
        proto_counts = {"modbus": 0, "dnp3": 0, "http": 0}
        for e in entries:
            p = e.get("protocol", "modbus").lower()
            proto_counts[p] = proto_counts.get(p, 0) + 1
        dominant = max(proto_counts, key=proto_counts.get)
        f_proto = np.array(PROTOCOL_ENCODINGS.get(dominant, [0, 0, 1]), dtype=float)

        # f_cmd — mean command severity weight across all requests
        weights = [CMD_WEIGHTS.get(e.get("function_code", "unknown"), 0.5)
                   for e in entries]
        f_cmd = float(np.mean(weights))

        # f_err — fraction of requests that returned an error response
        n_errors = sum(1 for e in entries if e.get("is_error", False))
        f_err = float(n_errors / n) if n > 0 else 0.0

        return np.array([f_freq, f_vol, *f_proto, f_cmd, f_err])

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def feature_summary(X: np.ndarray) -> Dict[str, Dict]:
        """Return per-feature min / mean / max / std statistics."""
        names = FeatureEngineer.FEATURE_NAMES
        stats = {}
        for i, name in enumerate(names):
            col = X[:, i]
            stats[name] = {
                "min":  col.min(),
                "mean": col.mean(),
                "max":  col.max(),
                "std":  col.std(),
            }
        return stats


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.traffic_generator import SCADATrafficGenerator

    gen = SCADATrafficGenerator()
    train, _, test = gen.generate_full_dataset()

    fe = FeatureEngineer()
    X_train_norm = fe.fit_normalize(train.X)
    X_test_norm  = fe.normalize(test.X)

    print("Feature summary (raw training data):")
    for name, s in fe.feature_summary(train.X).items():
        print(f"  {name:<20} min={s['min']:7.3f}  mean={s['mean']:7.3f}  "
              f"max={s['max']:8.3f}  std={s['std']:6.3f}")

    print(f"\nNormalised X_train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(f"Normalised X_test  range: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")
