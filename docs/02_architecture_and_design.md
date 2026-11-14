# Architecture & Algorithmic Design

## 1. Incremental Energy Evaluation ($O(1)$)
Recomputing $E(q)$ from scratch is $O(N^2)$. To enable fast search across thousands of evaluations, we maintain three attack count arrays:
1. $R[r]$: Number of queens in row $r$ (size $N$).
2. $D^+[d]$: Number of queens in primary diagonal $r + c = d$ (size $2N - 1$, where $0 \le d \le 2N - 2$).
3. $D^-[d]$: Number of queens in secondary diagonal $r - c + N - 1 = d$ (size $2N - 1$, where $0 \le d \le 2N - 2$).

Total energy:
$$E = \sum_{r=0}^{N-1} \binom{R[r]}{2} + \sum_{d=0}^{2N-2} \binom{D^+[d]}{2} + \sum_{d=0}^{2N-2} \binom{D^-[d]}{2}$$

### Move Delta Calculation:
When moving queen in column $c$ from row $a$ to row $r$:
- Old attack contribution:
  $$C_{\text{old}} = (R[a] - 1) + (D^+[a+c] - 1) + (D^-[a-c+N-1] - 1)$$
- New attack contribution:
  $$C_{\text{new}} = R[r] + D^+[r+c] + D^-[r-c+N-1] - 3 \cdot \mathbf{1}[r = a]$$
- Delta and new energy:
  $$\Delta = C_{\text{old}} - C_{\text{new}}$$
  $$E_{\text{child}} = E_{\text{parent}} - \Delta$$

A positive $\Delta$ indicates an improvement (reduction in energy). This computation requires only $O(1)$ arithmetic operations.

---

## 2. Invariant Rules & Population Management
1. **Memory Isolation:**
   - Boards in the beam population **never** share mutable arrays.
   - Materializing a child or its count array takes $O(N)$.
2. **Successor Sampling Rule:**
   - For $N = 8$: enumerate all $N(N-1) = 56$ moves.
   - For $N > 8$: sample $m = \min(2N, N(N-1))$ distinct moves uniformly without replacement.
   - The same rule applies to all methods.
3. **Budget Accounting:**
   - $B$ is the total number of evaluations per run.
   - Generating and scoring an initial board costs $1$ evaluation unit.
   - Evaluating each proposed move costs $1$ evaluation unit.
   - Reading already stored energy or running validation does **not** consume evaluation budget.
4. **Reproducibility:**
   - Independent pseudo-random number generator (RNG) streams for initialization, move sampling, and selection.
   - Track ancestry: for each selected board, maintain its parent index and original starting ancestor ID.
