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
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search import run_search, SearchResult, get_default_m

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "frozen_config.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score confidence interval for a proportion.
    Formula from Section 7 of PDF:
      center +- margin, with z = 1.96 for 95% confidence.
    Returns (p_hat, ci_lower, ci_upper).
    """
    if trials == 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / trials
    z = 1.96  # 95% confidence
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / trials) + (z2 / (4.0 * trials * trials)))) / denominator

    ci_lower = max(0.0, float(center - margin))
    ci_upper = min(1.0, float(center + margin))
    return float(p_hat), ci_lower, ci_upper


def run_experiment_1(config: Dict[str, Any], verbose: bool = True):
    """
    Experiment 1: Fixed total search budget comparison across A, B, C, D.
    Evaluates:
      - N in {8, 32, 100}
      - k in {1, 2, 4, 8, 16} (for A, k=1)
      - B fixed per N
      - 75 matched evaluation seeds
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results_csv = os.path.join(DATA_DIR, "exp1_results.csv")
    traces_csv = os.path.join(DATA_DIR, "exp1_traces.csv")

    eval_seeds = config["eval_seeds"]
    frozen_budgets = config["frozen_budgets"]
    tau = config["frozen_tau"]
    k_grid = config["k_grid"]
    core_sizes = config["core_sizes"]

    print("=" * 70)
    print(f"RUNNING EXPERIMENT 1: Fixed Total Search Budget ({len(eval_seeds)} seeds)")
    print("=" * 70)

    results_records = []
    traces_records = []

    total_configs = len(core_sizes) * (1 + 3 * (len(k_grid) - 1))
    current_config = 0

    for n in core_sizes:
        b = frozen_budgets[str(n)]
        m = get_default_m(n)

        # Build list of (method, k) to run
        combos = [("A", 1)]
        for m_name in ["B", "C", "D"]:
            for k in k_grid:
                if k > 1:
                    combos.append((m_name, k))

        for method, k in combos:
            current_config += 1
            t_start = time.perf_counter()
            succ_count = 0

            for seed in eval_seeds:
                res = run_search(method=method, n=n, k=k, budget=b, seed=seed, m=m, tau=tau)
                if res.solved:
                    succ_count += 1

                results_records.append({
                    "experiment": 1,
                    "method": method,
                    "n": n,
                    "k": k,
                    "m": m,
                    "budget": b,
                    "seed": seed,
                    "solved": int(res.solved),
                    "evals_used": res.evals_used,
                    "evals_to_solution": res.evals_to_solution if res.solved else np.nan,
                    "best_energy": res.best_energy,
                    "runtime_sec": round(res.runtime_sec, 4),
                })

                # Append generation traces
                if "evals" in res.trace:
                    for i in range(len(res.trace["evals"])):
                        traces_records.append({
                            "experiment": 1,
                            "method": method,
                            "n": n,
                            "k": k,
                            "seed": seed,
                            "evals": res.trace["evals"][i],
                            "best_energy": res.trace["best_energy"][i],
                            "diversity": res.trace["diversity"][i] if "diversity" in res.trace and i < len(res.trace["diversity"]) else 0.0,
                            "unique_states": res.trace["unique_states"][i] if "unique_states" in res.trace and i < len(res.trace["unique_states"]) else 1,
                            "distinct_ancestors": res.trace["distinct_ancestors"][i] if "distinct_ancestors" in res.trace and i < len(res.trace["distinct_ancestors"]) else 1,
                        })

            elapsed = time.perf_counter() - t_start
            p_hat, ci_l, ci_u = wilson_interval(succ_count, len(eval_seeds))
            if verbose:
                print(f"[{current_config}/{total_configs}] N={n:>3}, B={b:>6}, Method={method}, k={k:>2} | "
                      f"Success={succ_count:>2}/{len(eval_seeds)} ({p_hat*100:5.1f}%, 95% CI: [{ci_l*100:4.1f}%, {ci_u*100:4.1f}%]) | {elapsed:.2f}s")

            # Stream save periodically
            pd.DataFrame(results_records).to_csv(results_csv, index=False)
            pd.DataFrame(traces_records).to_csv(traces_csv, index=False)

    print(f"\n=> Exp 1 complete! Saved results to {results_csv} and traces to {traces_csv}")


def run_experiment_2(config: Dict[str, Any], verbose: bool = True):
    """
    Experiment 2: Check independent-run probability formula P = 1 - [1 - p(b)]^k.
    Setup:
      - Size N = 8, per-trajectory budget b = 500.
      - Estimate p(b) using Method A (k=1) on the evaluation seeds.
      - Run Method B with k in {1, 2, 4, 8, 16} where total allowance is k * b.
      - Compare empirical multi-start success with 1 - [1 - p(b)]^k.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results_csv = os.path.join(DATA_DIR, "exp2_results.csv")

    eval_seeds = config["eval_seeds"]
    n = 8
    b_per_traj = 500
    m = get_default_m(n)
    k_grid = config["k_grid"]

    print("=" * 70)
    print(f"RUNNING EXPERIMENT 2: Independent-Run Probability Formula (N={n}, b={b_per_traj})")
    print("=" * 70)

    # 1. Estimate p(b) with Method A
    a_successes = 0
    for seed in eval_seeds:
        res = run_search(method="A", n=n, k=1, budget=b_per_traj, seed=seed, m=m)
        if res.solved:
            a_successes += 1

    p_b, ci_l_a, ci_u_a = wilson_interval(a_successes, len(eval_seeds))
    print(f"Single-trajectory baseline p(b={b_per_traj}): {a_successes}/{len(eval_seeds)} ({p_b*100:.1f}%, 95% CI: [{ci_l_a*100:.1f}%, {ci_u_a*100:.1f}%])")

    exp2_records = []

    for k in k_grid:
        total_budget = k * b_per_traj
        succ_count = 0
        t_start = time.perf_counter()

        for seed in eval_seeds:
            # Multi-start with total budget kb
            res = run_search(method="B", n=n, k=k, budget=total_budget, seed=seed, m=m)
            if res.solved:
                succ_count += 1

            exp2_records.append({
                "experiment": 2,
                "n": n,
                "k": k,
                "b_per_traj": b_per_traj,
                "total_budget": total_budget,
                "seed": seed,
                "solved": int(res.solved),
                "evals_used": res.evals_used,
                "evals_to_solution": res.evals_to_solution if res.solved else np.nan,
                "best_energy": res.best_energy,
            })

        emp_p, emp_l, emp_u = wilson_interval(succ_count, len(eval_seeds))
        pred_p = 1.0 - ((1.0 - p_b) ** k)
        pred_l = 1.0 - ((1.0 - ci_l_a) ** k)
        pred_u = 1.0 - ((1.0 - ci_u_a) ** k)

        elapsed = time.perf_counter() - t_start
        print(f"k={k:>2} | Total Budget={total_budget:>5} | "
              f"Empirical: {succ_count:>2}/{len(eval_seeds)} ({emp_p*100:5.1f}%, CI: [{emp_l*100:4.1f}%, {emp_u*100:4.1f}%]) | "
              f"Predicted: {pred_p*100:5.1f}% (CI: [{pred_l*100:4.1f}%, {pred_u*100:4.1f}%]) | {elapsed:.2f}s")

    df = pd.DataFrame(exp2_records)
    df.to_csv(results_csv, index=False)
    print(f"\n=> Exp 2 complete! Saved results to {results_csv}")


def run_experiment_3(config: Dict[str, Any], verbose: bool = True):
    """
    Experiment 3: Diversity Collapse Diagnosis & Mitigation.
    Compares Methods C (Greedy), D (Stochastic), and E (Diversity-Aware) at fixed k = 8 on N = 32.
    Logs detailed curves of population diversity D, surviving ancestor count, and relaxed selections.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results_csv = os.path.join(DATA_DIR, "exp3_results.csv")
    traces_csv = os.path.join(DATA_DIR, "exp3_diversity_traces.csv")

    eval_seeds = config["eval_seeds"]
    n = 32
    k = 8
    b = config["frozen_budgets"]["32"]
    tau = config["frozen_tau"]
    d_min = config["frozen_d_min"]
    m = get_default_m(n)

    print("=" * 70)
    print(f"RUNNING EXPERIMENT 3: Diversity Collapse Diagnosis (N={n}, k={k}, B={b}, tau={tau}, d_min={d_min})")
    print("=" * 70)

    methods = ["C", "D", "E"]
    results_records = []
    traces_records = []

    for method in methods:
        succ_count = 0
        t_start = time.perf_counter()

        for seed in eval_seeds:
            res = run_search(method=method, n=n, k=k, budget=b, seed=seed, m=m, tau=tau, d_min=d_min)
            if res.solved:
                succ_count += 1

            results_records.append({
                "experiment": 3,
                "method": method,
                "n": n,
                "k": k,
                "m": m,
                "budget": b,
                "tau": tau,
                "d_min": d_min,
                "seed": seed,
                "solved": int(res.solved),
                "evals_used": res.evals_used,
                "evals_to_solution": res.evals_to_solution if res.solved else np.nan,
                "best_energy": res.best_energy,
                "runtime_sec": round(res.runtime_sec, 4),
            })

            if "evals" in res.trace:
                for i in range(len(res.trace["evals"])):
                    traces_records.append({
                        "experiment": 3,
                        "method": method,
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "evals": res.trace["evals"][i],
                        "best_energy": res.trace["best_energy"][i],
                        "diversity": res.trace["diversity"][i],
                        "unique_states": res.trace["unique_states"][i],
                        "distinct_ancestors": res.trace["distinct_ancestors"][i],
                        "relaxed_selections": res.trace["relaxed_selections"][i] if "relaxed_selections" in res.trace and i < len(res.trace["relaxed_selections"]) else 0,
                    })

        elapsed = time.perf_counter() - t_start
        p_hat, ci_l, ci_u = wilson_interval(succ_count, len(eval_seeds))
        print(f"Method {method} | Success={succ_count:>2}/{len(eval_seeds)} ({p_hat*100:5.1f}%, 95% CI: [{ci_l*100:4.1f}%, {ci_u*100:4.1f}%]) | {elapsed:.2f}s")

    pd.DataFrame(results_records).to_csv(results_csv, index=False)
    pd.DataFrame(traces_records).to_csv(traces_csv, index=False)
    print(f"\n=> Exp 3 complete! Saved results to {results_csv} and traces to {traces_csv}")


def run_experiment_4(config: Dict[str, Any], verbose: bool = True):
    """
    Experiment 4: Comparison against Strong Conventional Baseline.
    Compares best beam search method (Method C, k=4) against Random-Restart
    Strict Hill Climbing (Baseline_RHC) on N = 32, B = 20,000 across 75 seeds.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results_csv = os.path.join(DATA_DIR, "exp4_baseline.csv")

    eval_seeds = config["eval_seeds"]
    n = 32
    b = config["frozen_budgets"]["32"]
    m = get_default_m(n)

    print("=" * 70)
    print(f"RUNNING EXPERIMENT 4: Baseline Comparison (Beam Search vs Random-Restart Hill Climbing, N={n})")
    print("=" * 70)

    comparisons = [
        ("Baseline_RHC", 1),
        ("C", 4),
    ]

    records = []

    for method, k in comparisons:
        succ_count = 0
        energies = []
        t_start = time.perf_counter()

        for seed in eval_seeds:
            res = run_search(method=method, n=n, k=k, budget=b, seed=seed, m=m)
            if res.solved:
                succ_count += 1
            energies.append(res.best_energy)

            records.append({
                "experiment": 4,
                "method": method,
                "n": n,
                "k": k,
                "budget": b,
                "seed": seed,
                "solved": int(res.solved),
                "evals_used": res.evals_used,
                "evals_to_solution": res.evals_to_solution if res.solved else np.nan,
                "best_energy": res.best_energy,
                "runtime_sec": round(res.runtime_sec, 4),
            })

        elapsed = time.perf_counter() - t_start
        p_hat, ci_l, ci_u = wilson_interval(succ_count, len(eval_seeds))
        print(f"Method {method:<12} (k={k}) | Success={succ_count:>2}/{len(eval_seeds)} ({p_hat*100:5.1f}%, 95% CI: [{ci_l*100:4.1f}%, {ci_u*100:4.1f}%]) | "
              f"Mean Energy={np.mean(energies):.2f} | {elapsed:.2f}s")

    df = pd.DataFrame(records)
    df.to_csv(results_csv, index=False)
    print(f"\n=> Exp 4 complete! Saved results to {results_csv}")


def main():
    parser = argparse.ArgumentParser(description="N-Queens Local Beam Search Experiments")
    parser.add_argument("--pilot", action="store_true", help="Run 5-seed pilot calibration")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="Run specific experiment (1-4)")
    parser.add_argument("--all", action="store_true", help="Run all 4 experiments sequentially")

    args = parser.parse_args()
    config = load_config()

    if args.pilot:
        from src.experiments import run_pilot_study
        run_pilot_study()
    elif args.exp == 1:
        run_experiment_1(config)
    elif args.exp == 2:
        run_experiment_2(config)
    elif args.exp == 3:
        run_experiment_3(config)
    elif args.exp == 4:
        run_experiment_4(config)
    elif args.all:
        print("\n>>> Launching Full Benchmark Suite: Experiments 1, 2, 3, and 4 <<<\n")
        run_experiment_1(config)
        run_experiment_2(config)
        run_experiment_3(config)
        run_experiment_4(config)
        print("\n>>> ALL EXPERIMENTS COMPLETED SUCCESSFULLY! <<<\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
