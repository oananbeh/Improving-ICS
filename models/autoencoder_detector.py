"""
autoencoder_detector.py
───────────────────────
Autoencoder-based anomaly detector (Choi & Kim [17]).
Paper result: DR = 93.8 %, FPR = 2.8 %, F1 = 0.89

Uses PyTorch if available; falls back to PCA-based reconstruction (pure NumPy).
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
    class _AENet(nn.Module):
        def __init__(self, input_dim=7):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),        nn.ReLU(),
                nn.Linear(16, 8),         nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),          nn.ReLU(),
                nn.Linear(16, 32),         nn.ReLU(),
                nn.Linear(32, input_dim),  nn.Sigmoid(),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))


class _NumpyPCA:
    """PCA-based autoencoder using NumPy SVD as fallback."""
    def __init__(self, n_components=4, seed=RANDOM_SEED):
        self.n_components = n_components
        self._components  = None
        self._mean        = None

    def fit(self, X: np.ndarray):
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self._components = Vt[:self.n_components]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Xc    = X - self._mean
        codes = Xc @ self._components.T
        return codes @ self._components + self._mean

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        return np.mean((X - self.reconstruct(X)) ** 2, axis=1)


class AutoencoderDetector:
    """Autoencoder anomaly detector (PyTorch or PCA fallback)."""

    def __init__(
        self,
        input_dim:       int   = 7,
        latent_dim:      int   = 8,
        epochs:          int   = 50,
        batch_size:      int   = 512,
        lr:              float = 1e-3,
        threshold_sigma: float = 2.0,
        seed: int = RANDOM_SEED,
    ):
        self.input_dim       = input_dim
        self.latent_dim      = latent_dim
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.threshold_sigma = threshold_sigma
        self.seed            = seed
        self._threshold: Optional[float] = None
        self._fitted = False
        np.random.seed(seed)

        if _TORCH_AVAILABLE:
            torch.manual_seed(seed)
            self._net     = _AENet(input_dim)
            self._backend = "torch"
        else:
            self._pca_ae  = _NumpyPCA(n_components=latent_dim, seed=seed)
            self._backend = "pca"

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        if self._backend == "torch":
            self._fit_torch(X)
        else:
            self._pca_ae.fit(X)
        errors = self._reconstruction_error(X)
        self._threshold = float(errors.mean() + self.threshold_sigma * errors.std())
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (self._reconstruction_error(X) > self._threshold).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._reconstruction_error(X)

    def _fit_torch(self, X: np.ndarray):
        X_t  = torch.tensor(X, dtype=torch.float32)
        ds   = TensorDataset(X_t)
        dl   = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt  = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            for (b,) in dl:
                loss = crit(self._net(b), b)
                opt.zero_grad(); loss.backward(); opt.step()

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "torch":
            self._net.eval()
            with torch.no_grad():
                X_hat = self._net(torch.tensor(X, dtype=torch.float32)).numpy()
        else:
            X_hat = self._pca_ae.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def __repr__(self):
        status = f"fitted, backend={self._backend}" if self._fitted else "not fitted"
        return f"AutoencoderDetector(input_dim={self.input_dim}) [{status}]"
