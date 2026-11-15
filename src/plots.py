"""
plots.py - Publication-quality figure generation for N-Queens Local Beam Search study.
Generates:
  - Figure 1: Success fraction vs. k at fixed B with 95% Wilson confidence intervals.
  - Figure 2: Solved fraction vs. cumulative evaluations (survival / CDF curves).
  - Figure 3: Best-so-far energy vs. evaluations with median & interquartile bands.
  - Figure 4: Population diversity D vs. evaluations for Methods C, D, E.
  - Summary Table: Paired statistics, Wilson intervals, and performance metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experiments import wilson_interval

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
})

METHOD_COLORS = {
    "A": "#333333",        # Dark gray / black
    "B": "#1f77b4",        # Blue
    "C": "#2ca02c",        # Green
    "D": "#ff7f0e",        # Orange
    "E": "#9467bd",        # Purple
    "Baseline_RHC": "#d62728", # Red
}

METHOD_NAMES = {
    "A": "Method A (One-State)",
    "B": "Method B (Multi-Start)",
    "C": "Method C (Greedy Beam)",
    "D": "Method D (Stochastic Beam)",
    "E": "Method E (Diversity-Aware Beam)",
    "Baseline_RHC": "Baseline (Random-Restart HC)",
}


def plot_figure_1():
    """
    Figure 1: Success rate versus k at fixed B with 95% Wilson confidence intervals.
    Panel subplots for N = 8, N = 32, N = 100.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "exp1_results.csv"))
    sizes = [8, 32, 100]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, n in enumerate(sizes):
        ax = axes[idx]
        sub_df = df[df["n"] == n]
        budget = sub_df["budget"].iloc[0]
        m = sub_df["m"].iloc[0]
        n_trials = len(sub_df["seed"].unique())

        # Method A reference line
        a_df = sub_df[sub_df["method"] == "A"]
        a_succ = a_df["solved"].sum()
        p_a, ci_l_a, ci_u_a = wilson_interval(a_succ, n_trials)
        ax.axhline(p_a, color=METHOD_COLORS["A"], linestyle="--", alpha=0.7,
                   label=f"Method A (k=1): {p_a*100:.1f}%")
        ax.axhspan(ci_l_a, ci_u_a, color=METHOD_COLORS["A"], alpha=0.1)

        # Plot Methods B, C, D across k
        methods = ["B", "C", "D"]
        for m_name in methods:
            m_df = sub_df[sub_df["method"] == m_name]
            k_vals = sorted(m_df["k"].unique())
            p_vals = []
            err_lower = []
            err_upper = []

            for k in k_vals:
                k_data = m_df[m_df["k"] == k]
                succ = k_data["solved"].sum()
                trials = len(k_data)
                p_hat, l, u = wilson_interval(succ, trials)
                p_vals.append(p_hat)
                err_lower.append(p_hat - l)
                err_upper.append(u - p_hat)

            yerr = [err_lower, err_upper]
            ax.errorbar(
                k_vals, p_vals, yerr=yerr, fmt="-o",
                color=METHOD_COLORS[m_name], label=METHOD_NAMES[m_name],
                capsize=4, capthick=1.5
            )

        ax.set_title(f"N = {n} (Budget B = {budget:,}, m = {m})\nn = {n_trials} trials/config")
        ax.set_xlabel("Beam Width / Trajectories (k)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel("Success Rate (p̂ with 95% Wilson CI)")

    axes[0].legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_success_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"=> Saved Figure 1 to {out_path}")


def plot_figure_2():
    """
    Figure 2: Solved fraction versus cumulative evaluations.
    CDF curves for N = 8 (B=2,000) and N = 32 (B=20,000).
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "exp1_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, n in enumerate([8, 32]):
        ax = axes[idx]
        sub_df = df[df["n"] == n]
        budget = sub_df["budget"].iloc[0]
        n_trials = len(sub_df["seed"].unique())

        eval_grid = np.linspace(0, budget, 300)

        # Plot Method A (k=1) and Methods B, C, D at k=4
        configs = [("A", 1), ("B", 4), ("C", 4), ("D", 4)]

        for m_name, k_val in configs:
            target_df = sub_df[(sub_df["method"] == m_name) & (sub_df["k"] == k_val)]
            solved_evals = target_df["evals_to_solution"].dropna().values

            # Fraction solved by evaluation x
            fracs = [np.sum(solved_evals <= x) / n_trials for x in eval_grid]

            ax.plot(eval_grid, fracs, color=METHOD_COLORS[m_name],
                    label=f"{METHOD_NAMES[m_name]} (k={k_val})")

        ax.set_title(f"Solved Fraction vs. Evaluations (N = {n}, Budget B = {budget:,})")
        ax.set_xlabel("Cumulative Evaluations (x)")
        ax.set_ylabel("Fraction of Trials Solved by x")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.legend(loc="upper left" if n == 8 else "upper left")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_solved_fraction_vs_evals.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"=> Saved Figure 2 to {out_path}")


def plot_figure_3():
    """
    Figure 3: Best-so-far energy versus evaluations with median and interquartile band.
    Focuses on N = 32 and N = 100. Include every trial; after success its best energy remains zero.
    """
    traces_df = pd.read_csv(os.path.join(DATA_DIR, "exp1_traces.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, n in enumerate([32, 100]):
        ax = axes[idx]
        n_traces = traces_df[traces_df["n"] == n]
        budget = 20000 if n == 32 else 100000

        # Discretize eval grid
        eval_points = np.linspace(100, budget, 150)
        configs = [("A", 1), ("B", 4), ("C", 4), ("D", 4)]

        for m_name, k_val in configs:
            m_traces = n_traces[(n_traces["method"] == m_name) & (n_traces["k"] == k_val)]
            seeds = m_traces["seed"].unique()

            # For each seed, interpolate best_energy onto eval_points
            interpolated_matrix = []
            for s in seeds:
                s_data = m_traces[m_traces["seed"] == s].sort_values("evals")
                x_s = s_data["evals"].values
                y_s = s_data["best_energy"].values

                # Step function interpolation: energy stays at current best until next evaluation
                interp_y = []
                curr_idx = 0
                for ep in eval_points:
                    while curr_idx < len(x_s) - 1 and x_s[curr_idx + 1] <= ep:
                        curr_idx += 1
                    interp_y.append(y_s[curr_idx])
                interpolated_matrix.append(interp_y)

            interpolated_matrix = np.array(interpolated_matrix)
            median_curve = np.median(interpolated_matrix, axis=0)
            q25 = np.percentile(interpolated_matrix, 25, axis=0)
            q75 = np.percentile(interpolated_matrix, 75, axis=0)

            ax.plot(eval_points, median_curve, color=METHOD_COLORS[m_name],
                    label=f"{METHOD_NAMES[m_name]} (k={k_val})")
            ax.fill_between(eval_points, q25, q75, color=METHOD_COLORS[m_name], alpha=0.15)

        ax.set_title(f"Best Energy vs. Evaluations (N = {n}, Budget B = {budget:,})")
        ax.set_xlabel("Cumulative Evaluations")
        ax.set_ylabel("Best-So-Far Energy (Conflicts)")
        ax.grid(True)
        ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_energy_vs_evals.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"=> Saved Figure 3 to {out_path}")


def plot_figure_4():
    """
    Figure 4: Population Diversity D and Surviving Ancestor Count vs. Evaluations for C, D, E.
    N = 32, k = 8.
    States how many runs remain active at each step.
    """
    traces_df = pd.read_csv(os.path.join(DATA_DIR, "exp3_diversity_traces.csv"))
    results_df = pd.read_csv(os.path.join(DATA_DIR, "exp3_results.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    budget = 20000
    eval_points = np.linspace(200, budget, 120)
    methods = ["C", "D", "E"]

    for m_name in methods:
        m_traces = traces_df[traces_df["method"] == m_name]
        m_results = results_df[results_df["method"] == m_name]
        seeds = m_traces["seed"].unique()

        mean_div = []
        mean_anc = []
        active_counts = []

        for ep in eval_points:
            # Active runs: runs that did NOT solve before ep
            # (Per Section 7: do not quietly treat successful, terminated populations as live measurements)
            div_at_ep = []
            anc_at_ep = []
            active_seeds_count = 0

            for s in seeds:
                res_row = m_results[m_results["seed"] == s].iloc[0]
                if res_row["solved"] and res_row["evals_to_solution"] < ep:
                    # Run already succeeded and terminated
                    continue

                active_seeds_count += 1
                s_data = m_traces[m_traces["seed"] == s].sort_values("evals")
                x_s = s_data["evals"].values
                d_s = s_data["diversity"].values
                a_s = s_data["distinct_ancestors"].values

                # Find closest evaluated state <= ep
                valid_idx = np.where(x_s <= ep)[0]
                if len(valid_idx) > 0:
                    last_idx = valid_idx[-1]
                    div_at_ep.append(d_s[last_idx])
                    anc_at_ep.append(a_s[last_idx])

            active_counts.append(active_seeds_count)
            mean_div.append(np.mean(div_at_ep) if div_at_ep else 0.0)
            mean_anc.append(np.mean(anc_at_ep) if anc_at_ep else 0.0)

        # Plot Diversity
        ax1.plot(eval_points, mean_div, color=METHOD_COLORS[m_name],
                 label=f"{METHOD_NAMES[m_name]}")
        # Plot Surviving Ancestors
        ax2.plot(eval_points, mean_anc, color=METHOD_COLORS[m_name],
                 label=f"{METHOD_NAMES[m_name]}")

    ax1.set_title("Normalized Hamming Diversity (D) vs. Evaluations\n(N = 32, k = 8, Active Runs Only)")
    ax1.set_xlabel("Cumulative Evaluations")
    ax1.set_ylabel("Population Hamming Diversity (D)")
    ax1.set_ylim(-0.02, 1.0)
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_title("Surviving Initial Ancestor Count vs. Evaluations\n(Initial k = 8 Ancestors)")
    ax2.set_xlabel("Cumulative Evaluations")
    ax2.set_ylabel("Distinct Ancestors Remaining in Beam")
    ax2.set_ylim(0.5, 8.5)
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_diversity_vs_evals.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"=> Saved Figure 4 to {out_path}")


def generate_summary_tables():
    """
    Generates compact performance table and paired win/loss stats against Method A.
    """
    exp1_df = pd.read_csv(os.path.join(DATA_DIR, "exp1_results.csv"))
    exp3_df = pd.read_csv(os.path.join(DATA_DIR, "exp3_results.csv"))
    exp4_df = pd.read_csv(os.path.join(DATA_DIR, "exp4_baseline.csv"))

    table_rows = []

    # Process Exp 1
    for n in [8, 32, 100]:
        n_df = exp1_df[exp1_df["n"] == n]
        a_df = n_df[n_df["method"] == "A"].set_index("seed")

        for m_name in ["A", "B", "C", "D"]:
            k_vals = [1] if m_name == "A" else [2, 4, 8, 16]
            for k in k_vals:
                curr = n_df[(n_df["method"] == m_name) & (n_df["k"] == k)].set_index("seed")
                trials = len(curr)
                succ = int(curr["solved"].sum())
                p_hat, ci_l, ci_u = wilson_interval(succ, trials)

                # Paired wins / losses against Method A (lower energy is better)
                wins, losses, ties = 0, 0, 0
                for s in curr.index:
                    if s in a_df.index:
                        e_curr = curr.loc[s, "best_energy"]
                        e_a = a_df.loc[s, "best_energy"]
                        if e_curr < e_a:
                            wins += 1
                        elif e_curr > e_a:
                            losses += 1
                        else:
                            # If tied on energy, check if solved earlier
                            if curr.loc[s, "solved"] and a_df.loc[s, "solved"]:
                                if curr.loc[s, "evals_to_solution"] < a_df.loc[s, "evals_to_solution"]:
                                    wins += 1
                                elif curr.loc[s, "evals_to_solution"] > a_df.loc[s, "evals_to_solution"]:
                                    losses += 1
                                else:
                                    ties += 1
                            else:
                                ties += 1

                # Median evals to solution conditional on success
                succ_evals = curr[curr["solved"] == 1]["evals_to_solution"]
                med_evals = f"{succ_evals.median():.0f}" if len(succ_evals) > 0 else "N/A"

                table_rows.append({
                    "Experiment": "Exp 1",
                    "N": n,
                    "Method": m_name,
                    "k": k,
                    "Trials": trials,
                    "Successes": succ,
                    "Success_Rate": f"{p_hat*100:.1f}%",
                    "Wilson_95_CI": f"[{ci_l*100:.1f}%, {ci_u*100:.1f}%]",
                    "Mean_Best_Energy": round(curr["best_energy"].mean(), 2),
                    "Median_Evals_To_Sol": med_evals,
                    "Paired_W_L_T_vs_A": f"{wins}W - {losses}L - {ties}T",
                    "Mean_Time_s": round(curr["runtime_sec"].mean(), 3),
                })

    # Add Exp 4 comparison
    baseline_curr = exp4_df[exp4_df["method"] == "Baseline_RHC"].set_index("seed")
    b_succ = int(baseline_curr["solved"].sum())
    p_hat, ci_l, ci_u = wilson_interval(b_succ, len(baseline_curr))
    table_rows.append({
        "Experiment": "Exp 4",
        "N": 32,
        "Method": "Baseline_RHC",
        "k": 1,
        "Trials": len(baseline_curr),
        "Successes": b_succ,
        "Success_Rate": f"{p_hat*100:.1f}%",
        "Wilson_95_CI": f"[{ci_l*100:.1f}%, {ci_u*100:.1f}%]",
        "Mean_Best_Energy": round(baseline_curr["best_energy"].mean(), 2),
        "Median_Evals_To_Sol": "N/A",
        "Paired_W_L_T_vs_A": "Baseline",
        "Mean_Time_s": round(baseline_curr["runtime_sec"].mean(), 3),
    })

    summary_df = pd.DataFrame(table_rows)
    table_csv = os.path.join(FIG_DIR, "summary_table.csv")
    summary_df.to_csv(table_csv, index=False)
    print(f"=> Saved summary table to {table_csv}")
    return summary_df


def main():
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES AND SUMMARY TABLES")
    print("=" * 70)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    summary_df = generate_summary_tables()
    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
