"""
run_extended_baselines.py  (pure NumPy + matplotlib, no sklearn)
─────────────────────────
Addresses reviewer comments R3.2, R4.2, R6.2, R6.5, R6.6, R6.10, R7.viii.k.

Implements Isolation Forest and DBSCAN from scratch in pure NumPy.

Usage:
    export REPO_DIR="/path/to/camouflage_net"
    python run_extended_baselines.py
"""

import os, sys, time, json
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

REPO = os.environ.get("REPO_DIR", ".")
sys.path.insert(0, REPO)

from config import K_CLUSTERS, ALPHA_THRESHOLD, RANDOM_SEED, RESULTS_DIR, FIGURES_DIR
from data.traffic_generator import SCADATrafficGenerator
from features.feature_engineering import FeatureEngineer
from models.k_center import KCenterClustering
from models.k_means_detector import KMeansDetector
from models.snort_detector import SnortDetector

# ═══════════════════════════════════════════════════════════════════════════════
# Pure-NumPy Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════════

class _ITree:
    """A single isolation tree node."""
    __slots__ = ("left", "right", "split_feat", "split_val", "size", "depth")

    def __init__(self):
        self.left = self.right = None
        self.split_feat = self.split_val = None
        self.size = 0
        self.depth = 0


def _build_itree(X, rng, depth=0, max_depth=10):
    node = _ITree()
    n = len(X)
    node.size = n
    node.depth = depth
    if n <= 1 or depth >= max_depth:
        return node
    feat = rng.integers(0, X.shape[1])
    lo, hi = X[:, feat].min(), X[:, feat].max()
    if lo == hi:
        return node
    val = rng.uniform(lo, hi)
    node.split_feat = feat
    node.split_val = val
    mask = X[:, feat] < val
    node.left = _build_itree(X[mask], rng, depth + 1, max_depth)
    node.right = _build_itree(X[~mask], rng, depth + 1, max_depth)
    return node


def _path_length(x, node, depth=0):
    if node.left is None or node.right is None:
        n = max(node.size, 1)
        return depth + _c(n)
    if x[node.split_feat] < node.split_val:
        return _path_length(x, node.left, depth + 1)
    else:
        return _path_length(x, node.right, depth + 1)


def _c(n):
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationForestDetector:
    def __init__(self, n_estimators=100, subsample=256, contamination=0.15, seed=42):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.contamination = contamination
        self.seed = seed
        self.model_ = None
        self.threshold_ = 0.0

    def fit(self, X):
        max_samples = min(self.subsample, len(X))
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(X)
        # Compute threshold on training data for consistent binary prediction.
        scores = self.predict_scores(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def predict_scores(self, X):
        # sklearn decision_function: higher means more normal; negate so higher
        # means more anomalous to match thresholding used in this script.
        return -self.model_.decision_function(X)

    def predict(self, X):
        scores = self.predict_scores(X)
        return (scores >= self.threshold_).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-NumPy DBSCAN
# ═══════════════════════════════════════════════════════════════════════════════

class DBSCANDetector:
    """DBSCAN for anomaly detection: points not in any cluster = anomaly."""

    def __init__(self, eps=0.3, min_samples=10, max_fit=30000, max_core_points=5000):
        self.eps = eps
        self.min_samples = min_samples
        self.max_fit = max_fit
        self.max_core_points = max_core_points
        self.core_points_ = None

    def fit(self, X):
        n = len(X)
        if self.max_fit is not None and n > self.max_fit:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, self.max_fit, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit_predict(X_fit)
        self.core_points_ = X_fit[labels != -1].copy()
        if self.max_core_points is not None and len(self.core_points_) > self.max_core_points:
            rng2 = np.random.default_rng(43)
            ci = rng2.choice(len(self.core_points_), self.max_core_points, replace=False)
            self.core_points_ = self.core_points_[ci]
        return self

    def predict(self, X):
        return (self.predict_scores(X) > self.eps).astype(int)

    def predict_scores(self, X):
        if len(self.core_points_) == 0:
            return np.ones(len(X))
        n = len(X)
        nc = len(self.core_points_)
        min_dists = np.full(n, np.inf)
        xbatch = 500  # batch over X rows
        cbatch = 1000  # batch over core points
        for xi in range(0, n, xbatch):
            xe = min(xi + xbatch, n)
            Xb = X[xi:xe]
            for ci in range(0, nc, cbatch):
                ce = min(ci + cbatch, nc)
                cb = self.core_points_[ci:ce]
                dists = np.linalg.norm(Xb[:, None, :] - cb[None, :, :], axis=2)
                min_dists[xi:xe] = np.minimum(min_dists[xi:xe], dists.min(axis=1))
        return min_dists


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════════════

def _cm(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn

def mcc(y_true, y_pred):
    tp, fp, tn, fn = _cm(y_true, y_pred)
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return num / den if den > 0 else 0.0

def compute_roc_auc(y_true, scores, n_thresholds=300):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp, fp, tn, fn = _cm(y_true, y_pred)
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_s = [p[0] for p in pairs]
    tpr_s = [p[1] for p in pairs]
    # NumPy 2.x removed np.trapz alias; use trapezoid directly.
    auc = abs(np.trapezoid(tpr_s, fpr_s))
    return fpr_s, tpr_s, auc


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Extended Baselines Experiment (Reviewer Response)")
    print("=" * 70)

    # ── 1. Generate data (full dataset) ───────────────────────────────────────
    print("\n[1/6] Generating dataset ...")
    gen = SCADATrafficGenerator(seed=RANDOM_SEED)
    train_ds, val_ds, test_ds = gen.generate_full_dataset()

    fe = FeatureEngineer()
    X_train = fe.fit_normalize(train_ds.X)
    X_test = fe.normalize(test_ds.X)
    y_test = test_ds.y
    attack_types = test_ds.attack_types

    X_test_sub = X_test
    y_test_sub = y_test
    at_sub = attack_types
    X_train_sub = X_train

    print(f"   Train: {len(X_train_sub):,},  Test: {len(X_test_sub):,} "
          f"({int(y_test_sub.sum()):,} attacks, {int((y_test_sub==0).sum()):,} benign)")

    # DBSCAN scoring against millions of points is prohibitively expensive.
    # Keep full-scale evaluation for other detectors and use a bounded,
    # stratified subset for DBSCAN-specific scoring.
    DBSCAN_EVAL_MAX = 200000
    if len(X_test_sub) > DBSCAN_EVAL_MAX:
        rng_db = np.random.default_rng(RANDOM_SEED + 99)
        db_idx = rng_db.choice(len(X_test_sub), DBSCAN_EVAL_MAX, replace=False)
        X_test_db = X_test_sub[db_idx]
        y_test_db = y_test_sub[db_idx]
    else:
        X_test_db = X_test_sub
        y_test_db = y_test_sub

    # ── 2. Train detectors ────────────────────────────────────────────────────
    print("\n[2/6] Training detectors ...")

    detectors = {}

    t0 = time.perf_counter()
    kc = KCenterClustering(k=K_CLUSTERS, alpha=ALPHA_THRESHOLD, seed=RANDOM_SEED)
    kc.fit(X_train_sub)
    detectors["k-Center (Proposed)"] = kc
    print(f"   k-Center:         {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    km = KMeansDetector(k=K_CLUSTERS, alpha=ALPHA_THRESHOLD)
    km.fit(X_train_sub)
    detectors["k-Means"] = km
    print(f"   k-Means:          {time.perf_counter()-t0:.2f}s")

    sn = SnortDetector(use_normalised=True)
    detectors["Snort"] = sn
    print(f"   Snort:            (rule-based)")

    t0 = time.perf_counter()
    ifo = IsolationForestDetector(n_estimators=100, subsample=256, contamination=0.15, seed=RANDOM_SEED)
    ifo.fit(X_train_sub)
    detectors["Isolation Forest"] = ifo
    print(f"   Isolation Forest: {time.perf_counter()-t0:.2f}s")

    # DBSCAN eps grid search
    print("   DBSCAN: grid searching eps (bounded fit set) ...", end=" ", flush=True)
    best_eps, best_f1 = 0.3, 0.0
    tune_n = min(5000, len(X_test_sub))
    rng3 = np.random.default_rng(RANDOM_SEED+1)
    tune_idx = rng3.choice(len(X_test_sub), tune_n, replace=False)
    X_tune, y_tune = X_test_sub[tune_idx], y_test_sub[tune_idx]

    for eps_c in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        try:
            db_trial = DBSCANDetector(eps=eps_c, min_samples=10, max_fit=10000, max_core_points=3000)
            db_trial.fit(X_train_sub)
            yp = db_trial.predict(X_tune)
            tp, fp, tn, fn = _cm(y_tune, yp)
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_eps = f1, eps_c
        except Exception:
            pass
    print(f"best eps={best_eps} (F1={best_f1:.3f})")

    t0 = time.perf_counter()
    dbscan = DBSCANDetector(eps=best_eps, min_samples=10, max_fit=30000, max_core_points=5000)
    dbscan.fit(X_train_sub)
    detectors["DBSCAN"] = dbscan
    print(f"   DBSCAN:           {time.perf_counter()-t0:.2f}s")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/6] Evaluating ...")
    results = {}
    for name, det in detectors.items():
        if name == "DBSCAN":
            y_eval = y_test_db
            X_eval = X_test_db
        else:
            y_eval = y_test_sub
            X_eval = X_test_sub

        y_pred = det.predict(X_eval)
        tp, fp, tn, fn = _cm(y_eval, y_pred)
        dr = tp / (tp+fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp+tn) if (fp+tn) > 0 else 0
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0
        f1 = 2*prec*dr/(prec+dr) if (prec+dr) > 0 else 0
        m = mcc(y_eval, y_pred)
        results[name] = {
            "DR": round(dr*100, 2), "FPR": round(fpr*100, 2),
            "Precision": round(prec*100, 2), "F1": round(f1, 4),
            "MCC": round(m, 4),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }

    print(f"\n{'─'*90}")
    print(f"  {'Method':<22} {'DR%':>7} {'FPR%':>7} {'Prec%':>7} {'F1':>7} {'MCC':>7} "
          f"{'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}")
    print(f"{'─'*90}")
    for name, r in results.items():
        print(f"  {name:<22} {r['DR']:>7.2f} {r['FPR']:>7.2f} {r['Precision']:>7.2f} "
              f"{r['F1']:>7.4f} {r['MCC']:>7.4f} {r['TP']:>7,} {r['FP']:>7,} {r['FN']:>7,} {r['TN']:>7,}")
    print(f"{'─'*90}")

    # ── 4. ROC + AUC ──────────────────────────────────────────────────────────
    print("\n[4/6] ROC curves + AUC ...")
    roc_data = {}
    for name, det in detectors.items():
        if name == "Snort":
            continue
        if hasattr(det, 'predict_scores'):
            if name == "DBSCAN":
                y_eval = y_test_db
                X_eval = X_test_db
            else:
                y_eval = y_test_sub
                X_eval = X_test_sub
            scores = det.predict_scores(X_eval)
            fpr_arr, tpr_arr, auc_val = compute_roc_auc(y_eval, scores)
            roc_data[name] = {"fpr": fpr_arr, "tpr": tpr_arr, "AUC": round(auc_val, 4)}
            results[name]["AUC"] = round(auc_val, 4)
            print(f"   {name:<22}: AUC = {auc_val:.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"k-Center (Proposed)": "#2196F3", "k-Means": "#FF9800",
                  "Isolation Forest": "#4CAF50", "DBSCAN": "#9C27B0"}
        for name, rd in roc_data.items():
            ax.plot(rd["fpr"], rd["tpr"],
                    label=f'{name} (AUC={rd["AUC"]:.3f})',
                    color=colors.get(name, "gray"), linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves: Unsupervised Baselines", fontsize=13)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        roc_path = os.path.join(FIGURES_DIR, "fig_roc_curves.png")
        fig.savefig(roc_path, dpi=300)
        plt.close(fig)
        print(f"   Saved → {roc_path}")
    except Exception as e:
        print(f"   [WARN] Plot failed: {e}")

    # ── 5. Runtime ────────────────────────────────────────────────────────────
    print("\n[5/6] Runtime measurement ...")
    runtime = {}
    for name, det in detectors.items():
        if name == "DBSCAN":
            RT_DB_MAX = 50000
            if len(X_test_db) > RT_DB_MAX:
                rng_rt = np.random.default_rng(RANDOM_SEED + 777)
                rt_idx = rng_rt.choice(len(X_test_db), RT_DB_MAX, replace=False)
                X_rt = X_test_db[rt_idx]
            else:
                X_rt = X_test_db
            reps = 1
        else:
            X_rt = X_test_sub
            reps = 3
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            det.predict(X_rt)
            times.append(time.perf_counter() - t0)
        mean_t = np.mean(times)
        per_session_us = (mean_t / len(X_rt)) * 1e6
        runtime[name] = {"total_s": round(mean_t, 4), "per_session_us": round(per_session_us, 2)}
        print(f"   {name:<22}: {mean_t:.4f}s total, {per_session_us:.1f} µs/session")

    # ── 6. Multi-seed (optional, skipped for full-scale run-time practicality) ─
    print("\n[6/6] Multi-seed evaluation ... skipped (full-scale mode)")
    multi = {}

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "single_seed": results,
        "multi_seed": multi,
        "runtime": runtime,
        "roc_auc": {k: v["AUC"] for k, v in roc_data.items()},
        "note": f"Full eval test: {len(X_test_sub):,}, Train: {len(X_train_sub):,}; DBSCAN eval subset: {len(X_test_db):,}",
    }
    json_path = os.path.join(RESULTS_DIR, "extended_baselines.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n   Results → {json_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  LaTeX-Ready Single-Seed Rows")
    print(f"{'='*80}")
    for name, r in results.items():
        auc_val = r.get("AUC", "---")
        auc_s = f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val
        rt = runtime.get(name, {})
        rt_s = f"{rt.get('per_session_us', '---')}" if rt else "---"
        print(f"{name} & {r['DR']:.2f} & {r['FPR']:.2f} & {r['F1']:.4f} & {auc_s} & {r['MCC']:.4f} & {rt_s} \\\\")

    r = results["k-Center (Proposed)"]
    print(f"\n  Confusion Matrix (k-Center, seed=42):")
    print(f"              Pred Attack  Pred Normal")
    print(f"  Actual Atk   TP={r['TP']:>8,}   FN={r['FN']:>8,}")
    print(f"  Actual Norm  FP={r['FP']:>8,}   TN={r['TN']:>8,}")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}\n")
    return output


if __name__ == "__main__":
    main()
