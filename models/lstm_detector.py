"""
lstm_detector.py
────────────────
LSTM supervised baseline (Tama et al. [14]).
Paper result: DR = 97.3 %, FPR = 1.5 %, F1 = 0.95

Uses PyTorch LSTM if available; falls back to a 3-layer MLP
implemented in pure NumPy otherwise.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RANDOM_SEED

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, input_dim=7, hidden=64, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                batch_first=True, dropout=0.3)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1),      nn.Sigmoid(),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(1)


class _NumpyMLP:
    """
    3-layer MLP trained with mini-batch SGD + sigmoid activations.
    Pure NumPy fallback for LSTM.
    """
    def __init__(self, input_dim=7, hidden=(64, 32), lr=0.01, epochs=30,
                 batch_size=256, seed=RANDOM_SEED):
        rng = np.random.default_rng(seed)
        d   = input_dim
        h1, h2 = hidden
        # Xavier initialisation
        scale1 = np.sqrt(2.0 / (d + h1))
        scale2 = np.sqrt(2.0 / (h1 + h2))
        scale3 = np.sqrt(2.0 / (h2 + 1))
        self.W1 = rng.normal(0, scale1, (d,  h1)); self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, scale2, (h1, h2)); self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, scale3, (h2, 1));  self.b3 = np.zeros(1)
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self._rng = rng

    @staticmethod
    def _sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def _relu(x):     return np.maximum(0, x)

    def fit(self, X, y):
        n = len(X)
        for epoch in range(self.epochs):
            idx = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                sl  = idx[start:start + self.batch_size]
                xb  = X[sl]; yb = y[sl].reshape(-1, 1)
                # Forward
                z1  = xb @ self.W1 + self.b1
                a1  = self._relu(z1)
                z2  = a1 @ self.W2 + self.b2
                a2  = self._relu(z2)
                z3  = a2 @ self.W3 + self.b3
                out = self._sigmoid(z3)
                # Loss gradient (BCE)
                dL  = (out - yb) / len(sl)
                # Back
                dz3 = dL * out * (1 - out)
                dW3 = a2.T @ dz3;   db3 = dz3.sum(axis=0)
                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2;   db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1;   db1 = dz1.sum(axis=0)
                # Update
                for W, dW, b, db in [(self.W1, dW1, self.b1, db1),
                                     (self.W2, dW2, self.b2, db2),
                                     (self.W3, dW3, self.b3, db3)]:
                    W -= self.lr * dW
                    b -= self.lr * db

    def predict_proba(self, X):
        a1 = self._relu(X @ self.W1 + self.b1)
        a2 = self._relu(a1 @ self.W2 + self.b2)
        return self._sigmoid(a2 @ self.W3 + self.b3).ravel()


class LSTMDetector:
    """LSTM (or MLP fallback) intrusion detector."""

    def __init__(
        self,
        seq_len: int = 10, input_dim: int = 7, hidden: int = 64,
        n_layers: int = 2, epochs: int = 30, batch_size: int = 256,
        lr: float = 1e-3, threshold: float = 0.5, seed: int = RANDOM_SEED,
    ):
        self.seq_len    = seq_len
        self.threshold  = threshold
        self.seed       = seed
        self._fitted    = False
        np.random.seed(seed)

        if _TORCH_AVAILABLE:
            torch.manual_seed(seed)
            self._net      = _LSTMNet(input_dim, hidden, n_layers)
            self._backend  = "torch"
            self._epochs   = epochs
            self._bs       = batch_size
            self._lr       = lr
        else:
            self._mlp     = _NumpyMLP(input_dim, (hidden, hidden // 2),
                                       lr=lr, epochs=epochs,
                                       batch_size=batch_size, seed=seed)
            self._backend = "mlp"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMDetector":
        if self._backend == "torch":
            X_seq, y_seq = self._build_sequences(X, y)
            X_t  = torch.tensor(X_seq)
            y_t  = torch.tensor(y_seq)
            dl   = DataLoader(TensorDataset(X_t, y_t),
                              batch_size=self._bs, shuffle=True)
            opt  = torch.optim.Adam(self._net.parameters(), lr=self._lr)
            crit = nn.BCELoss()
            self._net.train()
            for _ in range(self._epochs):
                for xb, yb in dl:
                    loss = crit(self._net(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
        else:
            self._mlp.fit(X.astype(np.float32), y.astype(np.float32))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_scores(X) > self.threshold).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self._backend == "torch":
            X_seq, _ = self._build_sequences(X, np.zeros(len(X)))
            self._net.eval()
            with torch.no_grad():
                return self._net(torch.tensor(X_seq)).numpy()
        return self._mlp.predict_proba(X.astype(np.float32))

    def _build_sequences(self, X, y):
        T = self.seq_len; n = len(X)
        if n < T:
            pad = np.tile(X[0], (T - n, 1))
            X   = np.vstack([pad, X])
            y   = np.concatenate([np.zeros(T - n, dtype=y.dtype), y])
            n   = len(X)
        seqs, labels = [], []
        for i in range(T - 1, n):
            seqs.append(X[i - T + 1: i + 1])
            labels.append(float(y[i]))
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(X, y) before predict().")

    def __repr__(self):
        return f"LSTMDetector(backend={self._backend}) [{'fitted' if self._fitted else 'not fitted'}]"
