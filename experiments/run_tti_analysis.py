"""
run_tti_analysis.py
───────────────────
Reconnaissance Delay Analysis.

Runs 100 Monte Carlo TTI simulations for:
  (a) Static network (baseline: avg 121 s)
  (b) CamouflageNet with dynamic IP/MAC rotation (avg 404 s ± 34 s)

Reports the ~234 % increase in adversarial reconnaissance cost.

Run directly:
    python experiments/run_tti_analysis.py
"""

from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    N_MONTE_CARLO, ROTATION_INTERVAL, RESULTS_DIR, FIGURES_DIR,
    STATIC_TTI_EXPECTED, CAMOUFLAGENET_TTI_EXPECTED, TTI_STD_EXPECTED,
)
from simulation.tti_simulator import TTISimulator
from simulation.camouflage_net import CamouflageNet
from visualization.plots import PaperPlots


def run(verbose: bool = True) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    sep = "=" * 65
    if verbose:
        print(f"\n{sep}")
        print(f"  {N_MONTE_CARLO} Monte Carlo runs each")
        print(sep)

    # ── Run simulations ───────────────────────────────────────────────────────
    sim = TTISimulator(n_monte_carlo=N_MONTE_CARLO)

    if verbose: print(f"\n  Running static network simulation …")
    static = sim.simulate_static()

    if verbose: print(f"  Running CamouflageNet simulation …")
    camou  = sim.simulate_camouflage()

    comparison = {
        "static":     static,
        "camouflage": camou,
        "tti_increase_pct": (camou["tti_mean"] - static["tti_mean"]) /
                             static["tti_mean"] * 100,
    }

    # ── Print results ─────────────────────────────────────────────────────────
    if verbose:
        s, c = static, camou
        print(f"\n  {'Configuration':<25} {'TTI Mean':>10} {'Std':>8} "
              f"{'Median':>8} {'Min':>7} {'Max':>7}")
        print("  " + "─" * 65)

        s_arr = np.array(s["tti_all"])
        c_arr = np.array(c["tti_all"])
        print(f"  {'Static Network':<25} {s['tti_mean']:>9.1f}s "
              f"{s['tti_std']:>7.1f}s {s['tti_median']:>7.1f}s "
              f"{s_arr.min():>6.1f}s {s_arr.max():>6.1f}s")
        print(f"  {'CamouflageNet':<25} {c['tti_mean']:>9.1f}s "
              f"{c['tti_std']:>7.1f}s {c['tti_median']:>7.1f}s "
              f"{c_arr.min():>6.1f}s {c_arr.max():>6.1f}s")
        print()
        print(f"  TTI increase: {comparison['tti_increase_pct']:.0f} %  "
              f"(paper: ~234 %)")
        print()
        print(f"  Paper expected:  Static={STATIC_TTI_EXPECTED} s, "
              f"CamouflageNet={CAMOUFLAGENET_TTI_EXPECTED} ± {TTI_STD_EXPECTED} s")
        print()

        # Rotation statistics
        avg_rotations = np.mean([r.n_rotations for r in camou["runs"]])
        avg_hp_hits   = np.mean([r.n_honeypot_hits for r in camou["runs"]])
        print(f"  Avg rotation events per run : {avg_rotations:.1f}")
        print(f"  Avg honeypot hits per run   : {avg_hp_hits:.1f}")
        print(f"    → Forced re-scans: honeypot hits that had to be repeated")

    # ── CamouflageNet live simulation demo ───────────────────────────────────
    if verbose:
        print("\n  Demonstrating CamouflageNet reconfiguration …")
        cn = CamouflageNet()
        cn.build_network()
        n_rot = cn.run_for(duration=3 * ROTATION_INTERVAL)
        print(f"    Simulated {3 * ROTATION_INTERVAL} s → {n_rot} rotation cycles, "
              f"{len(cn.rotation_log)} node re-assignments")
        print(f"    Sample rotation: {cn.rotation_log[0]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_data = {
        "static":    {k: v for k, v in static.items()    if k != "runs"},
        "camouflage":{k: v for k, v in camou.items()     if k != "runs"},
        "tti_increase_pct": comparison["tti_increase_pct"],
    }
    path = os.path.join(RESULTS_DIR, "tti_results.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    if verbose:
        print(f"\n  Results saved → {path}")

    plotter = PaperPlots()
    plotter.plot_tti_comparison(static["tti_all"], camou["tti_all"])

    return comparison


if __name__ == "__main__":
    run(verbose=True)
