"""
metrics.py
──────────
Performance metrics used throughout the paper.

Equations from the paper:

  Detection Rate (Recall, Eq. 3):
      DR = TP / (TP + FN)

  False Positive Rate (Eq. 4):
      FPR = FP / (FP + TN)

  F1-Score (Eq. 5):
      F1 = 2 · (Precision · Recall) / (Precision + Recall)

Additional metrics:
  Precision  = TP / (TP + FP)
  Silhouette Coefficient — cluster quality validation
  TTI        — Time-to-Identify, measured separately in tti_simulator.py
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
# silhouette_score implemented in pure NumPy below


# ─── Core scalar metrics ──────────────────────────────────────────────────────

def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DR = TP / (TP + FN)  — also called Recall."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN)."""
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def f1_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2 · Precision · Recall / (Precision + Recall)."""
    rec  = detection_rate(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


# ─── Confusion matrix breakdown ───────────────────────────────────────────────

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Return TP, FP, TN, FN counts."""
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


# ─── Aggregate metric computation ────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute all metrics reported in the paper for a single detector.

    Returns
    -------
    dict with keys: name, DR, FPR, Precision, F1, TP, FP, TN, FN
    """
    cm = confusion_matrix(y_true, y_pred)
    dr  = detection_rate(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    prec = precision(y_true, y_pred)
    f1  = f1_score_manual(y_true, y_pred)

    return {
        "name":      name,
        "DR":        round(dr  * 100, 1),   # as percentage
        "FPR":       round(fpr * 100, 1),
        "Precision": round(prec * 100, 1),
        "F1":        round(f1, 3),
        **cm,
        "N_attacks": int(y_true.sum()),
        "N_benign":  int((y_true == 0).sum()),
    }


# ─── Per-attack-category metrics ──────────────────────────────────────────────

def evaluate_by_attack_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: np.ndarray,
) -> Dict[str, Dict]:
    """
    Compute metrics per attack category (Table 6 in paper).

    Parameters
    ----------
    y_true, y_pred : binary label arrays
    attack_types   : string array with one entry per sample
                     (e.g., "benign", "port_scanning", ...)

    Returns
    -------
    dict mapping attack_type → metric dict
    """
    results = {}
    categories = [t for t in np.unique(attack_types) if t != "benign"]

    for cat in categories:
        mask = (attack_types == cat) | (attack_types == "benign")
        yt = y_true[mask]
        yp = y_pred[mask]
        count = int((attack_types == cat).sum())
        m = compute_all_metrics(yt, yp, name=cat)
        m["Count"] = count
        results[cat] = m

    # Overall weighted
    results["Overall (Weighted)"] = compute_all_metrics(y_true, y_pred,
                                                         name="Overall")
    results["Overall (Weighted)"]["Count"] = int(y_true.sum())
    return results


# ─── Silhouette Coefficient (pure NumPy) ─────────────────────────────────────

def _silhouette_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Silhouette Coefficient in pure NumPy.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean intra-cluster distance, b(i) = min inter-cluster mean dist.
    """
    unique = np.unique(labels)
    n = len(X)
    if len(unique) < 2 or n < 4:
        return 0.0

    scores = np.zeros(n)
    for i in range(n):
        own  = labels[i]
        mask_own = labels == own
        mask_own[i] = False
        if mask_own.sum() == 0:
            a = 0.0
        else:
            a = np.linalg.norm(X[i] - X[mask_own], axis=1).mean()

        b = np.inf
        for c in unique:
            if c == own:
                continue
            mask_c = labels == c
            if not mask_c.any():
                continue
            b_c = np.linalg.norm(X[i] - X[mask_c], axis=1).mean()
            b   = min(b, b_c)

        denom = max(a, b)
        scores[i] = (b - a) / denom if denom > 0 else 0.0

    return float(scores.mean())


def silhouette_score_wrapper(
    X: np.ndarray,
    labels: np.ndarray,
    n_runs: int = 10,
    seed: int = 42,
    sample_size: int = 10_000,
) -> Tuple[float, float]:
    """
    Compute Silhouette Coefficient averaged over `n_runs` independent
    subsamples.

    Paper result: 0.78 ± 0.03

    Parameters
    ----------
    X          : feature matrix
    labels     : cluster assignments
    n_runs     : number of independent sample evaluations
    sample_size: samples per run (for speed on large datasets)

    Returns
    -------
    (mean_score, std_score)
    """
    rng    = np.random.default_rng(seed)
    scores = []

    # Remove any "noise" label (-1) if present
    valid = labels >= 0
    X_v, l_v = X[valid], labels[valid]

    n = len(X_v)
    if n < 100:
        return 0.0, 0.0

    for _ in range(n_runs):
        if n > sample_size:
            idx = rng.choice(n, sample_size, replace=False)
            X_s, l_s = X_v[idx], l_v[idx]
        else:
            X_s, l_s = X_v, l_v

        if len(np.unique(l_s)) < 2:
            continue
        try:
            s = _silhouette_numpy(X_s, l_s)
            scores.append(s)
        except Exception:
            pass

    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


# ─── Single-detector evaluator ────────────────────────────────────────────────

def evaluate_detector(
    detector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: Optional[str] = None,
    attack_types: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a detector on test data and return a full metric report.

    Parameters
    ----------
    detector     : any model with a .predict(X) method
    X_test       : normalised test feature matrix
    y_test       : ground-truth binary labels
    name         : optional display name
    attack_types : optional per-sample attack-type labels

    Returns
    -------
    dict with overall metrics and optional per-category breakdown
    """
    name    = name or type(detector).__name__
    y_pred  = detector.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred, name=name)

    if attack_types is not None:
        metrics["by_attack"] = evaluate_by_attack_type(y_test, y_pred, attack_types)

    return metrics


# ─── Multi-detector comparison table ─────────────────────────────────────────

def evaluate_all_detectors(
    detectors: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Evaluate all detectors and return a list of metric dicts suitable for
    printing as Table 3 / Table 4 in the paper.

    Parameters
    ----------
    detectors    : dict mapping display_name → fitted detector
    X_test       : normalised test feature matrix
    y_test       : ground-truth labels
    attack_types : optional per-sample type labels

    Returns
    -------
    list of metric dicts, one per detector
    """
    results = []
    for name, det in detectors.items():
        m = evaluate_detector(det, X_test, y_test, name=name,
                              attack_types=attack_types)
        results.append(m)
    return results


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_results_table(results: List[Dict], title: str = "Results"):
    """Print a formatted comparison table like Table 3 / Table 4 in the paper."""
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")
    print(f"  {'Method':<30} {'DR (%)':>7} {'FPR (%)':>8} {'F1':>6}")
    print(f"{'─'*65}")
    for r in results:
        print(f"  {r['name']:<30} {r['DR']:>7.1f} {r['FPR']:>8.1f} {r['F1']:>6.3f}")
    print(f"{'─'*65}")


def print_ablation_table(results: List[Dict], title: str = "Ablation Study"):
    """Print Table 5 (ablation)."""
    print(f"\n{'─'*75}")
    print(f"  {title}")
    print(f"{'─'*75}")
    print(f"  {'Configuration':<40} {'DR':>5} {'FPR':>5} {'F1':>5} {'TTI(s)':>7}")
    print(f"{'─'*75}")
    for r in results:
        tti = r.get("TTI", "—")
        tti_str = f"{tti:.0f}" if isinstance(tti, (int, float)) else str(tti)
        print(f"  {r['name']:<40} {r['DR']:>5.1f} {r['FPR']:>5.1f} "
              f"{r['F1']:>5.3f} {tti_str:>7}")
    print(f"{'─'*75}")
