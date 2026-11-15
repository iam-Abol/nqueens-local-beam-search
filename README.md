# N-Queens: Local Beam Search, Diversity, and Information Sharing

An empirical and theoretical investigation examining whether maintaining multiple search states ($k$ boards) and sharing information between them increases the probability of finding conflict-free chessboards, and diagnosing how diversity collapse impacts search performance.

---

## 1. Quick Start

### Installation & Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Running the Test Suite
The automated correctness test suite verifies the $O(1)$ incremental delta against an independent $O(N^2)$ oracle across 1,000 random moves, verifies state isolation/immutability, and checks move generators and search budget strictness:
```bash
pytest -v
```

---

## 2. Reproducing the Experiments

All experimental configurations and seeds are frozen under [`configs/frozen_config.json`](configs/frozen_config.json) using 75 independent evaluation seeds.

### Run the Entire Benchmark Suite
```bash
python src/experiments.py --all
```

### Or Run Individual Experiments:
* **Pilot Calibration Study (5 pilot seeds):**
  ```bash
  python src/experiments.py --pilot
  ```
* **Experiment 1 (Fixed Total Search Budget across A, B, C, D):**
  ```bash
  python src/experiments.py --exp 1
  ```
* **Experiment 2 (Check Independent Probability Formula):**
  ```bash
  python src/experiments.py --exp 2
  ```
* **Experiment 3 (Diversity Collapse Diagnosis — C vs D vs E):**
  ```bash
  python src/experiments.py --exp 3
  ```
* **Experiment 4 (Comparison with Random-Restart Hill Climbing):**
  ```bash
  python src/experiments.py --exp 4
  ```

---

## 3. Generating Figures and Summary Tables

To generate all 4 publication-quality figures with 95% Wilson confidence intervals and the matched-trial results table:
```bash
python src/plots.py
```
Generated artifacts are saved into [`figures/`](figures/):
* `fig1_success_vs_k.png`: Success Rate vs. Beam Width $k$ (with 95% Wilson error bars).
* `fig2_solved_fraction_vs_evals.png`: Solved fraction vs. cumulative evaluations (CDF).
* `fig3_energy_vs_evals.png`: Best-so-far energy trajectories with median and interquartile bands.
* `fig4_diversity_vs_evals.png`: Population Hamming diversity $D$ and surviving ancestor decay over time.
* `summary_table.csv`: Comprehensive results table with paired win/loss stats against Method A.

---

## 4. Key Empirical Findings

1. **Greedy Beam Search ($k=4$) Outperforms Single-State Search:**
   * On $N = 32$, Method C ($k=4$) achieved a **34.7% success rate**, over **$4\times$ higher** than single-state Method A ($8.0\%$), winning **27 paired head-to-head matches vs. only 5 losses** ($p < 10^{-4}$).
   * On the massive $N = 100$ chessboard, Method C ($k=4$) discovered **4 global solutions ($E=0$)** where single-state and multi-start search failed completely ($0\%$).
2. **The Inverted-U Width Penalty ($k = 16$):**
   * Under a fixed evaluation budget $B$, increasing beam width beyond $k=4$ starves search depth. At $k=16$, success collapses to $0\%$ because each state receives too few generations.
3. **Independent Multi-Start Validates Probability Theory:**
   * In Experiment 2, Method B's empirical success matched the theoretical formula $\Pr = 1 - [1 - p(b)]^k$ within $< 2\%$ across all beam widths.
4. **Forced Diversity Can Harm Performance (Method E):**
   * Enforcing a strict 10% Hamming distance separation prevented clone takeover, but caused success to plummet to **0%** by rejecting high-quality candidates in the basin of attraction of promising states.

---

## 5. Deliverables & Documentation
* **Comprehensive Final Report:** [`report/report.md`](report/report.md) (covers all exercises A–C and questions Q1–Q9).
* **Detailed Design Specs:** [`docs/`](docs/)
* **Raw CSV Datasets:** [`data/`](data/)
