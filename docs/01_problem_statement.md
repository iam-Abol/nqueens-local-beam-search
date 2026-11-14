# Problem Statement: Local Beam Search & Diversity in N-Queens

## 1. Overview
The **N-Queens problem** requires placing $N$ queens on an $N \times N$ chessboard such that no two queens attack each other (i.e., no two queens share the same row, column, or diagonal).

In this assignment, we examine local search algorithms on N-Queens to address a central empirical and theoretical question:
> **Under a limited search budget, does maintaining several candidate solutions ($k$ boards) and sharing information between them increase the probability of finding a conflict-free board? When does cooperation help, and when does it destroy useful diversity?**

---

## 2. State Representation & Search Formulation
- **State Vector:** Board represented by $q = (q[0], q[1], \dots, q[N-1])$, where $q[c] \in \{0, \dots, N-1\}$ is the row index of the queen in column $c$.
- **Constraint Satisfaction Property:** By definition, each column has exactly one queen.
- **Move Operator:** Changing a single queen's row in column $c$ from row $a$ to row $r$ ($r \neq a$). Column $c$ remains fixed.
- **Objective / Energy Function $E(q)$:** Number of attacking queen pairs:
  $$E(q) = \sum_{0 \le i < j < N} \mathbf{1}[q[i] = q[j] \lor |q[i] - q[j]| = |i - j|]$$
- **Global Minimum:** $E(q) = 0$. A solution has zero conflicts.

---

## 3. The Comparison Paradigms
- **Single-State Walk (Method A):** Tracks one board ($k=1$), taking best one-move neighbor.
- **Independent Multi-Start (Method B):** Tracks $k$ independent trajectories, each running Method A independently without sharing candidates.
- **Greedy Local Beam Search (Method C):** Expands $k$ boards, pools all successors, eliminates duplicates, and retains the globally best $k$ boards.
- **Stochastic Local Beam Search (Method D):** Expands $k$ boards, pools successors, and probabilistically samples $k$ boards using a Boltzmann/softmax distribution with temperature $\tau$.
- **Diversity-Aware Beam Search (Method E):** Enhances Method D by enforcing a minimum Hamming distance $d_{\min} = 0.10$ between selected candidates to counter diversity collapse.
- **Random-Restart Strict Hill Climbing (Exp 4 Baseline):** Single trajectory with strict uphill prevention and random restarts after 10 stagnant steps.
