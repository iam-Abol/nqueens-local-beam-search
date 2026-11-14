# Experiments Plan & Statistical Methodology

## 1. Experimental Grid
- **Board sizes $N$:** $\{8, 32, 100\}$
- **Beam widths $k$:** $\{1, 2, 4, 8, 16\}$ (with $P = 1$ single CPU process)
- **Evaluation budgets $B$:** $\{10^3, 10^4, 10^5\}$
- **Number of trials:** At least 30 independent trial seeds per configuration (100 preferable where runtime allows). 5 pilot seeds to calibrate and freeze budgets.

---

## 2. The Four Experiments
### Experiment 1: Fixed Total Search Budget Comparison
- Methods: A (One-state, $k=1$), B (Multi-start), C (Greedy Beam), D (Stochastic Beam).
- Setup: Total budget $B$ fixed across all methods.
- Recorded metrics: Success rate $\hat{p}$, 95% Wilson interval, evaluations to first solution, best energy found, full trace.

### Experiment 2: Validation of Independent Probability Formula
- Goal: Test theoretical model $\Pr(\text{at least one succeeds}) = 1 - [1 - p(b)]^k$.
- Setup: Multi-start where each trajectory gets an independent budget $b$, so total budget is $kb$ (extra total work experiment).
- Compare measured multi-start success against the prediction derived from single-trajectory success $p(b)$.

### Experiment 3: Diversity Collapse Diagnosis & Mitigation
- Configuration: Fixed $k = 8$, selected $N$ and budget $B$.
- Methods: Greedy Beam (C), Stochastic Beam (D), Diversity-Aware Beam (E).
- Trace: Evolution of diversity $D$, distinct surviving ancestors, and best-so-far energy across cumulative evaluations.

### Experiment 4: Strong Conventional Baseline
- Compare best beam search against Random-Restart Strict Hill Climbing (restarts after 10 consecutive non-improving steps, charging restarts to $B$).

---

## 3. Statistical Measures
1. **95% Wilson Score Interval for Proportions:**
   $$\hat{p} = \frac{s}{n}, \quad z = 1.96$$
   $$\text{Interval} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
2. **Normalized Population Hamming Diversity:**
   $$D = \frac{2}{k(k-1)} \sum_{i < j} \frac{d_H(q_i, q_j)}{N}$$
   where $d_H(q_i, q_j)$ counts column positions where rows differ.
