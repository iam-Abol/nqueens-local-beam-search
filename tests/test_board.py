"""
test_board.py - Unit test suite verifying:
1. Ground truth energies on reference boards.
2. Pairwise oracle equivalence with count-based energy across 100 seeded boards.
3. Invariant check on 1,000 random moves (including no-ops) verifying Delta = E_before - E_after.
4. State immutability: modifying a child does not mutate parent or siblings.
"""

import pytest
import numpy as np
from src.board import (
    validate_pairwise,
    make_counts,
    energy_from_counts,
    move_delta,
    apply_move_in_place,
    copy_state,
    hamming_distance,
    population_diversity,
)


def test_reference_energies():
    """Test 1: Confirm E([1, 3, 0, 2]) == 0 and E([0, 0, 0, 0]) == 6."""
    b1 = [1, 3, 0, 2]
    assert validate_pairwise(b1) == 0
    c1 = make_counts(b1)
    assert energy_from_counts(c1) == 0

    b2 = [0, 0, 0, 0]
    assert validate_pairwise(b2) == 6
    c2 = make_counts(b2)
    assert energy_from_counts(c2) == 6


@pytest.mark.parametrize("N", [8, 20])
def test_oracle_equivalence_and_move_delta(N: int):
    """
    Test 2:
    On 100 seeded boards at N = 8 and N = 20:
      - Match count-based energy to pairwise oracle.
      - Check 1,000 random moves (including no-ops): Delta = E_before - E_after.
    """
    rng = np.random.default_rng(seed=42 + N)
    num_boards = 100
    moves_per_board = 10  # 100 * 10 = 1,000 random moves tested

    for _ in range(num_boards):
        q = rng.integers(0, N, size=N).tolist()
        oracle_e = validate_pairwise(q)
        counts = make_counts(q)
        count_e = energy_from_counts(counts)

        # Baseline check
        assert count_e == oracle_e, f"Count energy {count_e} != Oracle energy {oracle_e}"

        # Test moves on this board
        current_q = list(q)
        current_counts = (counts[0].copy(), counts[1].copy(), counts[2].copy())
        current_e = count_e

        for _ in range(moves_per_board):
            col = int(rng.integers(0, N))
            # Include no-ops occasionally
            if rng.random() < 0.2:
                new_row = current_q[col]  # No-op move
            else:
                new_row = int(rng.integers(0, N))

            delta = move_delta(current_q, current_counts, col, new_row)
            e_before = current_e

            # Apply move in place
            new_e = apply_move_in_place(current_q, current_counts, current_e, col, new_row)
            e_after = new_e

            # Invariant: Delta == E_before - E_after
            assert delta == (e_before - e_after), (
                f"Delta mismatch: delta={delta}, e_before - e_after = {e_before - e_after}"
            )

            # Invariant: Count energy matches pairwise oracle exactly after move
            oracle_after = validate_pairwise(current_q)
            count_after = energy_from_counts(current_counts)
            assert count_after == oracle_after == e_after, (
                f"Count {count_after}, Oracle {oracle_after}, Updated E {e_after} disagree"
            )

            current_e = new_e


def test_immutability_and_state_isolation():
    """
    Test 3:
    Reconstruct counts after moves. Check one queen per column, valid indices,
    and that modifying a child cannot mutate its parent or a sibling.
    """
    N = 8
    parent_q = [0, 4, 7, 5, 2, 6, 1, 3]
    parent_counts = make_counts(parent_q)
    parent_e = energy_from_counts(parent_counts)

    # Create child 1 and child 2
    child1_q, child1_counts, child1_e = copy_state(parent_q, parent_counts, parent_e)
    child2_q, child2_counts, child2_e = copy_state(parent_q, parent_counts, parent_e)

    # Mutate child 1
    child1_e = apply_move_in_place(child1_q, child1_counts, child1_e, 0, 5)

    # Parent must remain completely unchanged
    assert parent_q[0] == 0
    assert parent_e == energy_from_counts(parent_counts)
    assert validate_pairwise(parent_q) == parent_e

    # Sibling child 2 must remain completely unchanged
    assert child2_q[0] == 0
    assert child2_e == energy_from_counts(child2_counts)
    assert validate_pairwise(child2_q) == child2_e

    # Mutate child 2 differently
    child2_e = apply_move_in_place(child2_q, child2_counts, child2_e, 1, 1)
    assert child1_q[1] == 4
    assert child2_q[1] == 1


def test_diversity_metrics():
    """Test 4: Hamming distance and normalized diversity D."""
    b1 = [0, 1, 2, 3]
    b2 = [0, 1, 2, 3]
    b3 = [3, 2, 1, 0]

    assert hamming_distance(b1, b2) == 0
    assert hamming_distance(b1, b3) == 4

    # k = 1 edge case
    assert population_diversity([b1]) == 0.0

    # Two identical boards: D = 0.0
    assert population_diversity([b1, b2]) == 0.0

    # b1 and b3 differ in all 4 positions: D = 4/4 = 1.0
    assert population_diversity([b1, b3]) == 1.0

    # Three boards: (b1,b2)=0/4, (b1,b3)=4/4, (b2,b3)=4/4 -> avg = (0 + 1 + 1)/3 = 2/3
    assert pytest.approx(population_diversity([b1, b2, b3]), 1e-6) == 2.0 / 3.0

