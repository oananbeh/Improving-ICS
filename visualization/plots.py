"""
plots.py
────────
Reproduces all figures from the paper:

  Figure 1 — SCADA architecture diagram (text-based, no external deps)
  Figure 2 — Detection Rate comparison bar chart (Table 3 / Table 4)
  Figure 3 — TTI comparison (static vs. CamouflageNet, 100 Monte Carlo runs)
  Figure 4 — Ablation study bar chart (Table 5)
  Figure 5 — Per-attack-category detection performance (Table 6)
  Figure 6 — Elbow method: k vs. r_max
  Figure 7 — Silhouette coefficient distribution
  Figure 8 — Feature distribution by traffic type (boxplots)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FIGURES_DIR


# ─── Colours matching the paper's style ───────────────────────────────────────
BLUE   = "#2563EB"   # proposed method
ORANGE = "#F97316"   # SOTA / baselines


class PaperPlots:
    """
    Generates and saves all paper figures.
    All figures are saved as high-resolution PNGs to FIGURES_DIR.
    """

    def __init__(self, output_dir: str = FIGURES_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2 — Detection Rate Comparison (Table 3 & 4)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_detection_comparison(
        self,
        results: List[Dict],
        title: str = "Detection Rate Comparison: CamouflageNet (Unsupervised) vs. Methods",
        filename: str = "fig2_detection_comparison.png",
    ):
        """
        Reproduces Figure 2 from the paper.
        results : list of dicts with keys 'name', 'DR', 'FPR', 'F1'
        """
        names = [r["name"] for r in results]
        drs   = [r["DR"]   for r in results]
        fprs  = [r["FPR"]  for r in results]
        f1s   = [r["F1"]   for r in results]

        # Colour: proposed model in blue, everything else in orange
        proposed_idx = next(
            (i for i, n in enumerate(names) if "k-Center" in n or "Proposed" in n), 0
        )
        colors = [BLUE if i == proposed_idx else ORANGE for i in range(len(names))]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

        metrics = [("Detection Rate (%)", drs, "DR (%)"),
                   ("False Positive Rate (%)", fprs, "FPR (%)"),
                   ("F1-Score", f1s, "F1")]

        for ax, (ylabel, vals, label) in zip(axes, metrics):
            bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white",
                          linewidth=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight="bold")

            if ylabel == "F1-Score":
                ax.set_ylim(0.7, 1.05)
            elif ylabel == "False Positive Rate (%)":
                ax.set_ylim(0, max(vals) * 1.4)
            else:
                ax.set_ylim(70, 102)

            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if ylabel != "F1-Score" else 0.003),
                    f"{val:.1f}" if ylabel != "F1-Score" else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        proposed_patch  = mpatches.Patch(color=BLUE,   label="Proposed (k-Center)")
        baseline_patch  = mpatches.Patch(color=ORANGE, label="Baseline / SOTA")
        fig.legend(handles=[proposed_patch, baseline_patch],
                   loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0),
                   fontsize=10, frameon=True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 3 — TTI Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def plot_tti_comparison(
        self,
        static_ttis:    List[float],
        camouflage_ttis: List[float],
        filename: str = "fig3_tti_comparison.png",
    ):
        """
        Reproduces Figure 2 (TTI comparison) caption from.
        Shows histogram + box plot of TTI across 100 Monte Carlo runs.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Time-to-Identify (TTI): Static Network vs. CamouflageNet\n"
            "(100 Monte Carlo Simulation Runs)",
            fontsize=13, fontweight="bold",
        )

        # ── Left: Histogram ────────────────────────────────────────────────
        ax = axes[0]
        bins = np.linspace(
            min(min(static_ttis), min(camouflage_ttis)) - 20,
            max(max(static_ttis), max(camouflage_ttis)) + 20,
            30,
        )
        ax.hist(static_ttis,    bins=bins, alpha=0.65, color=ORANGE,
                label=f"Static  (μ={np.mean(static_ttis):.0f}s)", edgecolor="white")
        ax.hist(camouflage_ttis, bins=bins, alpha=0.65, color=BLUE,
                label=f"CamouflageNet  (μ={np.mean(camouflage_ttis):.0f}s)",
                edgecolor="white")
        ax.axvline(np.mean(static_ttis),    color=ORANGE, linestyle="--", lw=2)
        ax.axvline(np.mean(camouflage_ttis), color=BLUE,   linestyle="--", lw=2)
        ax.set_xlabel("TTI (seconds)", fontsize=11)
        ax.set_ylabel("Frequency",     fontsize=11)
        ax.set_title("Distribution of TTI",  fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ── Right: Box plot ────────────────────────────────────────────────
        ax2 = axes[1]
        bp  = ax2.boxplot(
            [static_ttis, camouflage_ttis],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(ORANGE)
        bp["boxes"][1].set_facecolor(BLUE)
        ax2.set_xticklabels(["Static Network", "CamouflageNet"], fontsize=11)
        ax2.set_ylabel("TTI (seconds)", fontsize=11)
        ax2.set_title("Box Plot of TTI", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Annotate increase
        pct = (np.mean(camouflage_ttis) - np.mean(static_ttis)) / \
              np.mean(static_ttis) * 100
        ax2.text(1.5, max(camouflage_ttis) * 0.95,
                 f"+{pct:.0f} % increase",
                 ha="center", va="top", fontsize=11, color=BLUE, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 4 — Ablation Study
    # ──────────────────────────────────────────────────────────────────────────

    def plot_ablation_study(
        self,
        ablation_results: List[Dict],
        filename: str = "fig4_ablation_study.png",
    ):
        """
        Visualises Table 5 (ablation study) as a grouped bar chart.
        """
        configs = [r["name"] for r in ablation_results]
        drs     = [r["DR"]   for r in ablation_results]
        fprs    = [r["FPR"]  for r in ablation_results]
        f1s     = [r["F1"]   for r in ablation_results]
        ttis    = [r.get("TTI", 0) for r in ablation_results]

        x   = np.arange(len(configs))
        w   = 0.2
        fig, ax = plt.subplots(figsize=(14, 6))

        b1 = ax.bar(x - 1.5*w, drs,  w, label="DR (%)",   color="#2563EB", alpha=0.85)
        b2 = ax.bar(x - 0.5*w, fprs, w, label="FPR (%)",  color="#F97316", alpha=0.85)
        b3 = ax.bar(x + 0.5*w, [f * 100 for f in f1s], w,
                    label="F1 × 100", color="#16A34A", alpha=0.85)
        b4 = ax.bar(x + 1.5*w, [t / 5 for t in ttis], w,
                    label="TTI / 5 (s)", color="#7C3AED", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title("Ablation Study: Impact of Individual Components",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 5 — Per-attack-category performance
    # ──────────────────────────────────────────────────────────────────────────

    def plot_per_attack_performance(
        self,
        attack_results: Dict[str, Dict],
        filename: str = "fig5_per_attack_performance.png",
    ):
        """Visualises Table 6 (per-category detection)."""
        categories = [k for k in attack_results if k != "Overall (Weighted)"]
        drs  = [attack_results[c]["DR"]  for c in categories]
        fprs = [attack_results[c]["FPR"] for c in categories]
        f1s  = [attack_results[c]["F1"]  for c in categories]
        counts = [attack_results[c].get("Count", 0) for c in categories]

        x = np.arange(len(categories))
        w = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Detection Performance by Attack Category (Table 6)",
                     fontsize=13, fontweight="bold")

        # Detection Rate
        ax = axes[0]
        bars = ax.bar(x, drs, color=[BLUE, ORANGE, "#16A34A", "#7C3AED"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(75, 102)
        ax.set_ylabel("Detection Rate (%)", fontsize=11)
        ax.set_title("DR by Attack Category", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, drs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Session counts
        ax2 = axes[1]
        bars2 = ax2.bar(x, counts, color=[BLUE, ORANGE, "#16A34A", "#7C3AED"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Number of Attack Sessions", fontsize=11)
        ax2.set_title("Attack Session Counts", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for bar, val in zip(bars2, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"{val:,}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 6 — Elbow Method
    # ──────────────────────────────────────────────────────────────────────────

    def plot_elbow_method(
        self,
        ks:     List[int],
        radii:  List[float],
        optimal_k: int = 45,
        filename: str = "fig6_elbow_method.png",
    ):
        """
        Plots k vs. r_max (Elbow Method for selecting k=45).
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(ks, radii, "o-", color=BLUE, linewidth=2, markersize=4,
                label="Max cluster radius (r_max)")
        ax.axvline(optimal_k, color="red", linestyle="--", linewidth=1.5,
                   label=f"Optimal k = {optimal_k}")

        # Annotate the elbow point
        if optimal_k in ks:
            idx = ks.index(optimal_k)
            ax.annotate(
                f"k = {optimal_k}\nr_max = {radii[idx]:.3f}",
                xy=(optimal_k, radii[idx]),
                xytext=(optimal_k + 5, radii[idx] + 0.05 * max(radii)),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red",
            )

        ax.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax.set_ylabel("Maximum Cluster Radius (r_max)", fontsize=12)
        ax.set_title("Elbow Method: Selecting Optimal k for k-Center Clustering\n"
                     "(k tested from 10 to 100)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 7 — Silhouette Coefficient
    # ──────────────────────────────────────────────────────────────────────────

    def plot_silhouette_distribution(
        self,
        scores: List[float],
        filename: str = "fig7_silhouette.png",
    ):
        """Histogram of Silhouette Coefficient over independent runs."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=15, color=BLUE, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=2,
                   label=f"Mean = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        ax.set_xlabel("Silhouette Coefficient", fontsize=12)
        ax.set_ylabel("Frequency",              fontsize=12)
        ax.set_title("Cluster Quality: Silhouette Coefficient Distribution\n"
                     "(10 independent runs)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 8 — Feature Distribution by Traffic Type
    # ──────────────────────────────────────────────────────────────────────────

    def plot_feature_distributions(
        self,
        X: np.ndarray,
        attack_types: np.ndarray,
        filename: str = "fig8_feature_distributions.png",
    ):
        """
        Box plots of key features split by traffic type.
        Illustrates the distributional differences that enable detection.
        """
        feature_names = ["f_freq (req/window)", "f_vol (bytes)",
                         "f_cmd (severity)", "f_err (error rate)"]
        feature_idx   = [0, 1, 5, 6]   # indices in the 7-dim vector

        traffic_types = ["benign", "port_scanning", "modbus_fuzzing",
                         "dos", "command_injection"]
        colors_map = {
            "benign":           "#6B7280",
            "port_scanning":    ORANGE,
            "modbus_fuzzing":   "#7C3AED",
            "dos":              "#DC2626",
            "command_injection":"#16A34A",
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Feature Distribution by Traffic Type",
                     fontsize=13, fontweight="bold")

        for ax, fname, fidx in zip(axes.flat, feature_names, feature_idx):
            data  = []
            ticks = []
            clrs  = []
            for tt in traffic_types:
                mask = attack_types == tt
                if mask.any():
                    data.append(X[mask, fidx])
                    ticks.append(tt.replace("_", "\n"))
                    clrs.append(colors_map[tt])

            bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=2))
            for patch, clr in zip(bp["boxes"], clrs):
                patch.set_facecolor(clr)
                patch.set_alpha(0.75)

            ax.set_xticklabels(ticks, fontsize=9)
            ax.set_ylabel(fname.split(" ")[0], fontsize=10)
            ax.set_title(fname, fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 9 — k-Center vs k-Means: cluster radius comparison
    # ──────────────────────────────────────────────────────────────────────────

    def plot_kcenter_vs_kmeans(
        self,
        kcenter_scores: np.ndarray,
        kmeans_scores:  np.ndarray,
        y_true: np.ndarray,
        filename: str = "fig9_kcenter_vs_kmeans.png",
    ):
        """
        Scatter plot of anomaly scores from k-center vs k-means,
        coloured by ground truth, to show why k-center is superior.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Anomaly Score Comparison: k-Center vs. k-Means",
                     fontsize=13, fontweight="bold")

        for ax, scores, name, color in zip(
            axes,
            [kcenter_scores, kmeans_scores],
            ["k-Center (Proposed)", "k-Means (Baseline)"],
            [BLUE, ORANGE],
        ):
            benign_scores = scores[y_true == 0]
            attack_scores = scores[y_true == 1]
            idx = np.random.choice(len(benign_scores),
                                   min(5000, len(benign_scores)), replace=False)
            ax.hist(benign_scores[idx], bins=60, alpha=0.6, color="#6B7280",
                    label="Benign", density=True)
            ax.hist(attack_scores,      bins=60, alpha=0.6, color=color,
                    label="Attack", density=True)
            ax.set_xlabel("Distance to Nearest Cluster Centre", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return path
