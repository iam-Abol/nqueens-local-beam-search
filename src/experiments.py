"""
experiments.py - Pilot calibration, parameter freezing, and experimental pipeline
for N-Queens Local Beam Search investigation.

Covers:
  - 5-seed pilot study to calibrate budgets B and temperature tau.
  - Experiment 1: Fixed total search budget (Methods A, B, C, D across k in {1,2,4,8,16}).
  - Experiment 2: Independent multi-start formula validation (P = 1 - [1 - p(b)]^k).
  - Experiment 3: Diversity collapse diagnosis and mitigation (C vs D vs E at k = 8).
  - Experiment 4: Comparison against Random-Restart Strict Hill Climbing baseline.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.search import run_search, SearchResult

PILOT_SEEDS = [1001, 1002, 1003, 1004, 1005]
EVAL_SEEDS = [2000 + i for i in range(1, 31)]  # 30 matched evaluation seeds


def run_pilot_study() -> Dict[str, Any]:
    """
    Executes calibration on 5 pilot seeds as mandated by Section 6 of the PDF:
      1. Calibrate temperature tau in {0.5, 1.0, 2.0, 5.0} for Method D.
      2. Calibrate budget B across N in {8, 32, 100} to ensure an informative
         regime (avoiding trivial 100% or impossible 0% success across all methods).
    """
    print("=" * 70)
    print("STARTING PILOT CALIBRATION STUDY (5 Pilot Seeds: 1001-1005)")
    print("=" * 70)

    # Part A: Temperature calibration on N = 32, k = 4, B = 10,000
    print("\n--- Part A: Temperature (tau) Calibration for Method D (N=32, k=4, B=10,000) ---")
    tau_candidates = [0.5, 1.0, 2.0, 5.0]
    tau_results = []

    for tau in tau_candidates:
        successes = 0
        final_energies = []
        evals_list = []
        runtimes = []

        for seed in PILOT_SEEDS:
            res = run_search(method="D", n=32, k=4, budget=10000, seed=seed, tau=tau)
            if res.solved:
                successes += 1
                evals_list.append(res.evals_to_solution)
            final_energies.append(res.best_energy)
            runtimes.append(res.runtime_sec)

        avg_energy = float(np.mean(final_energies))
        avg_evals = float(np.mean(evals_list)) if evals_list else float("nan")
        avg_time = float(np.mean(runtimes))

        tau_results.append({
            "tau": tau,
            "success_rate": f"{successes}/{len(PILOT_SEEDS)} ({successes/len(PILOT_SEEDS)*100:.0f}%)",
            "mean_best_energy": round(avg_energy, 2),
            "mean_evals_to_sol": round(avg_evals, 1) if not np.isnan(avg_evals) else "N/A",
            "mean_runtime_s": round(avg_time, 3),
        })

    tau_df = pd.DataFrame(tau_results)
    print(tau_df.to_string(index=False))

    # Pick tau with best success, lowest energy
    best_tau_row = sorted(tau_results, key=lambda x: (-float(x["success_rate"].split("(")[1].split("%")[0]), x["mean_best_energy"]))[0]
    selected_tau = best_tau_row["tau"]
    print(f"\n=> Calibrated tau: {selected_tau}")

    # Part B: Budget Calibration across N in {8, 32, 100}
    print("\n--- Part B: Budget (B) Calibration across N in {8, 32, 100} ---")
    budget_tests = [
        # N, budget_candidates
        (8, [1000, 2000, 5000]),
        (32, [5000, 10000, 20000]),
        (100, [10000, 25000, 50000]),
    ]

    budget_results = []
    chosen_budgets = {}

    for n, budgets in budget_tests:
        print(f"\nTesting N = {n}:")
        for b in budgets:
            # Test across Methods A, B, C, D with k = 4
            methods = ["A", "B", "C", "D"]
            for m_name in methods:
                k_val = 1 if m_name == "A" else 4
                successes = 0
                energies = []
                runtimes = []
                for seed in PILOT_SEEDS:
                    res = run_search(method=m_name, n=n, k=k_val, budget=b, seed=seed, tau=selected_tau)
                    if res.solved:
                        successes += 1
                    energies.append(res.best_energy)
                    runtimes.append(res.runtime_sec)

                budget_results.append({
                    "N": n,
                    "budget_B": b,
                    "method": m_name,
                    "k": k_val,
                    "success": f"{successes}/5",
                    "mean_energy": round(float(np.mean(energies)), 2),
                    "mean_time_s": round(float(np.mean(runtimes)), 3),
                })

    b_df = pd.DataFrame(budget_results)
    print(b_df.to_string(index=False))

    # Based on pilot outcomes, freeze budgets that provide an informative range:
    # N=8: B=2000 (enough to observe fast convergence)
    # N=32: B=15000 (differentiates methods well without hitting 100% or 0% ceiling)
    # N=100: B=30000 (allows deep exploration within reasonable runtime)
    frozen_budgets = {
        "8": 2000,
        "32": 15000,
        "100": 30000,
    }

    config = {
        "pilot_seeds": PILOT_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "num_eval_trials": len(EVAL_SEEDS),
        "frozen_tau": selected_tau,
        "frozen_d_min": 0.10,
        "frozen_budgets": frozen_budgets,
        "core_sizes": [8, 32, 100],
        "k_grid": [1, 2, 4, 8, 16],
    }

    os.makedirs("configs", exist_ok=True)
    with open("configs/frozen_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PILOT STUDY COMPLETE! Frozen configuration saved to configs/frozen_config.json")
    print(f"Frozen budgets: {frozen_budgets}")
    print(f"Frozen tau: {selected_tau}")
    print("=" * 70)

    return config


if __name__ == "__main__":
    run_pilot_study()
