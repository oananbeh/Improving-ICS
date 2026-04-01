"""
main.py
───────
CamouflageNet — complete paper replication entry point.

Runs all experiments from:
  "Improving ICS Security Through Honeynets and Machine Learning Techniques"
  Alnozami, Ananbeh, Kim (2025)

Usage:
    python main.py                  # run all experiments
    python main.py --exp main       # main detection comparison
    python main.py --exp ablation   # ablation study
    python main.py --exp tti        # TTI analysis
    python main.py --exp elbow      # Elbow Method
    python main.py --exp all        # all experiments (default)

Output is saved to:
    results/experiment_results.json
    results/ablation_results.json
    results/tti_results.json
    results/elbow_results.json
    results/figures/fig2_detection_comparison.png
    results/figures/fig3_tti_comparison.png
    results/figures/fig4_ablation_study.png
    results/figures/fig5_per_attack_performance.png
    results/figures/fig6_elbow_method.png
    results/figures/fig7_silhouette.png
    results/figures/fig8_feature_distributions.png
    results/figures/fig9_kcenter_vs_kmeans.png
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║              CamouflageNet — Paper Implementation                ║
║  "Improving ICS Security Through Honeynets and ML Techniques"    ║
║  Alnozami, Ananbeh, Kim  |  Oakland University (2025)            ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="CamouflageNet paper implementation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["main", "ablation", "tti", "elbow", "all"],
        default="all",
        help=(
            "Experiment to run:\n"
            "  main     — detection comparison\n"
            "  ablation — ablation study\n"
            "  tti      — TTI analysis \n"
            "  elbow    — Elbow Method k selection\n"
            "  all      — run all experiments (default)"
        ),
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Skip LSTM training (faster; use if PyTorch is not installed)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    print(BANNER)
    t_start = time.time()

    results = {}

    if args.exp in ("all", "elbow"):
        from experiments.run_elbow import run as run_elbow
        print("━━━ Elbow Method ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        results["elbow"] = run_elbow(verbose=not args.quiet)

    if args.exp in ("all", "tti"):
        from experiments.run_tti_analysis import run as run_tti
        print("\n━━━ TTI Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        results["tti"] = run_tti(verbose=not args.quiet)

    if args.exp in ("all", "main"):
        from experiments.run_experiment import run as run_main
        print("\n━━━ Main Experiment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        results["main"] = run_main(
            save_results=True,
            run_lstm=not args.no_lstm,
            verbose=not args.quiet,
        )

    if args.exp in ("all", "ablation"):
        from experiments.run_ablation import run as run_ablation
        print("\n━━━ Ablation Study ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        results["ablation"] = run_ablation(verbose=not args.quiet)

    elapsed = time.time() - t_start
    print(f"\n{'━'*65}")
    print(f"  All experiments complete in {elapsed:.1f} s")
    print(f"  Results → results/")
    print(f"  Figures → results/figures/")
    print(f"{'━'*65}\n")

    return results


if __name__ == "__main__":
    main()
