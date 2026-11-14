# Rubric & Deliverables Checklist

## Grading Rubric (Total: 100 Points)

| Category | Points | Description |
| :--- | :--- | :--- |
| **Correctness & Invariants** | 25 | $O(1)$ move delta matches $O(N^2)$ oracle; immutability of parent/sibling states; proper energy definitions; unit test suite. |
| **Search Policies & Diversity Variant** | 25 | Correct implementations of Methods A, B, C, D, E and Random-Restart baseline. Proper sampling, temperature $\tau$, and Hamming threshold $d_{\min}$. |
| **Experimental Controls & Reproducibility** | 25 | Fixed seed streams, evaluation counter accounting, frozen config, CSV raw logs, matched trials. |
| **Evidence & Interpretation** | 15 | 4 required figures with Wilson confidence intervals, results table with paired win/loss stats, rigorous discussion. |
| **Written Reasoning & Clarity** | 10 | Complete, sound answers to Exercises A–C and Questions Q1–Q9. |

---

## Deliverables Checklist
- [ ] Source code modularized under `src/`
- [ ] README.md with clear instructions to run tests and benchmarks
- [ ] Automated correctness test suite under `tests/`
- [ ] Frozen configuration file under `configs/`
- [ ] Initial boards and seed logs under `data/`
- [ ] Raw CSV results under `data/`
- [ ] 4 required figures under `figures/`
- [ ] 4–6 page final report under `report/`
