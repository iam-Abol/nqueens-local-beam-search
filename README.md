# N-Queens Local Beam Search & Diversity Study

An experimental investigation of Local Beam Search, Multi-Start, Diversity preservation, and Hill Climbing on the N-Queens problem.

## Documentation
Comprehensive technical documentation is maintained under the `docs/` folder:
- [01_problem_statement.md](docs/01_problem_statement.md): Mathematical formulation and search landscape.
- [02_architecture_and_design.md](docs/02_architecture_and_design.md): $O(1)$ move delta calculation, state isolation, and algorithms.
- [03_experiments_plan.md](docs/03_experiments_plan.md): Experimental grid, budget allocation, and statistical metrics.
- [04_rubric_and_deliverables.md](docs/04_rubric_and_deliverables.md): Detailed grading checklist and required artifacts.

## Project Structure
```text
.
├── NQueens Beam Search (3).pdf   # Assignment specification
├── README.md                     # Project overview and instructions
├── requirements.txt              # Python package dependencies
├── docs/                         # Detailed project documentation
├── configs/                      # Experiment configurations and seed parameters
├── src/                          # Core implementation (board, search, policies, metrics)
├── tests/                        # Automated unit tests and invariant verification
├── data/                         # Experimental traces and CSV outputs
├── figures/                      # Generated research plots
└── report/                       # Final written report and theoretical exercises
```

## Setup & Running
```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/ -v
```
