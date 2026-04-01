"""Models package for CamouflageNet."""
from .k_center         import KCenterClustering
from .k_means_detector import KMeansDetector
from .snort_detector   import SnortDetector
from .ips              import IPSProtector, BlockEntry
from .random_forest_detector import RandomForestDetector
from .autoencoder_detector   import AutoencoderDetector

__all__ = [
    "KCenterClustering",
    "KMeansDetector",
    "SnortDetector",
    "IPSProtector",
    "BlockEntry",
    "RandomForestDetector",
    "AutoencoderDetector",
]

# LSTM is optional (requires PyTorch)
try:
    from .lstm_detector import LSTMDetector
    __all__.append("LSTMDetector")
except ImportError:
    pass
