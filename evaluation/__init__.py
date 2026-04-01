"""Evaluation package for CamouflageNet."""
from .metrics import (
    detection_rate, false_positive_rate, f1_score_manual,
    compute_all_metrics, silhouette_score_wrapper,
    evaluate_detector, evaluate_all_detectors,
)

__all__ = [
    "detection_rate",
    "false_positive_rate",
    "f1_score_manual",
    "compute_all_metrics",
    "silhouette_score_wrapper",
    "evaluate_detector",
    "evaluate_all_detectors",
]
