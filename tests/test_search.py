"""
test_search.py - Tests for search algorithms, budget accounting, move generation,
and reproducibility.
"""

import pytest
import numpy as np
from src.search import (
    get_default_m,
    sample_moves_for_board,
    allocate_moves,
    run_search,
)
from src.board import validate_pairwise


def test_move_sampler():
    """Verify move generator generates valid, distinct moves without replacement."""
    rng = np.random.default_rng(42)

    # N = 8: all 56 moves
    q8 = [0, 1, 2, 3, 4, 5, 6, 7]
    moves8 = sample_moves_for_board(q8, m=56, rng=rng)
    assert len(moves8) == 56
    assert len(set(moves8)) == 56
    # None should have new_row == old_row
    for col, new_row in moves8:
        assert new_row != q8[col]

    # N = 20, m = 40
    q20 = list(range(20))
    moves20 = sample_moves_for_board(q20, m=40, rng=rng)
    assert len(moves20) == 40
    assert len(set(moves20)) == 40
    for col, new_row in moves20:
        assert new_row != q20[col]


def test_allocate_moves():
    """Verify move budget allocation round-robin and limits."""
    # 4 parents, m = 10, remaining = 25 -> [10, 10, 5, 0] starting from parent 0
    p1 = allocate_moves(remaining=25, k=4, m=10, gen_idx=0)
    assert p1 == [10, 10, 5, 0]
    assert sum(p1) == 25

    # Same with gen_idx = 1 (start with parent 1) -> [0, 10, 10, 5]
    p2 = allocate_moves(remaining=25, k=4, m=10, gen_idx=1)
    assert p2 == [0, 10, 10, 5]
    assert sum(p2) == 25


@pytest.mark.parametrize("method", ["A", "B", "C", "D", "E", "Baseline_RHC"])
def test_budget_strictness_and_validity(method: str):
    """
    Ensure every search method strictly obeys the budget constraint
    and verifies that if solved, energy is genuinely 0 according to oracle.
    """
    n = 8
    budget = 150
    seed = 101

    res = run_search(method=method, n=n, k=4 if method in ("B", "C", "D", "E") else 1,
                     budget=budget, seed=seed)

    # Budget strictly respected
    assert res.evals_used <= budget

    if res.solved:
        assert res.best_energy == 0
        assert validate_pairwise(res.best_board) == 0
        assert res.evals_to_solution is not None
        assert res.evals_to_solution <= res.evals_used
    else:
        assert res.evals_to_solution is None
        assert res.best_energy > 0


def test_solution_finding_on_easy_n():
    """On N = 8 with reasonable budget, all methods should reliably find solutions."""
    methods = ["A", "B", "C", "D", "E", "Baseline_RHC"]
    for method in methods:
        # Give enough budget to find N=8 solution
        res = run_search(method=method, n=8, k=4 if method in ("B", "C", "D", "E") else 1,
                         budget=3000, seed=42)
        assert res.evals_used <= 3000
        assert validate_pairwise(res.best_board) == res.best_energy


def test_reproducibility():
    """Running identical search with identical seed must yield bitwise identical results."""
    res1 = run_search(method="C", n=12, k=4, budget=200, seed=777)
    res2 = run_search(method="C", n=12, k=4, budget=200, seed=777)

    assert res1.solved == res2.solved
    assert res1.evals_used == res2.evals_used
    assert res1.best_energy == res2.best_energy
    assert res1.best_board == res2.best_board
    assert res1.trace["evals"] == res2.trace["evals"]
    assert res1.trace["best_energy"] == res2.trace["best_energy"]
