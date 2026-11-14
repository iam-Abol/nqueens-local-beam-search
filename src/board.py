"""
board.py - Core board representation, O(1) incremental energy calculations,
and O(N^2) ground-truth pairwise conflict validation.
"""

from typing import Tuple, List
import numpy as np


def validate_pairwise(q: List[int]) -> int:
    """
    Independent ground-truth O(N^2) energy oracle.
    Counts the number of unordered pairs of queens that attack each other.
    Two queens in column i and j attack if:
      - They share the same row: q[i] == q[j]
      - They share a diagonal: |q[i] - q[j]| == |i - j|
    """
    n = len(q)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if q[i] == q[j] or abs(q[i] - q[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def make_counts(q: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the three conflict count arrays in O(N) time.
    Returns:
      - R: Row counts, array of size N
      - D_plus: Anti-diagonal counts (r + c), array of size 2N - 1
      - D_minus: Main diagonal counts (r - c + N - 1), array of size 2N - 1
    """
    n = len(q)
    R = np.zeros(n, dtype=np.int32)
    D_plus = np.zeros(2 * n - 1, dtype=np.int32)
    D_minus = np.zeros(2 * n - 1, dtype=np.int32)

    for c, r in enumerate(q):
        R[r] += 1
        D_plus[r + c] += 1
        D_minus[r - c + n - 1] += 1

    return R, D_plus, D_minus


def energy_from_counts(counts: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
    """
    Computes the total energy E(q) from the count arrays in O(N) time.
    For an array where each entry x represents the number of queens on that line,
    the number of attacking pairs is x * (x - 1) // 2.
    """
    R, D_plus, D_minus = counts
    e_row = np.sum(R * (R - 1) // 2)
    e_dplus = np.sum(D_plus * (D_plus - 1) // 2)
    e_dminus = np.sum(D_minus * (D_minus - 1) // 2)
    return int(e_row + e_dplus + e_dminus)


def move_delta(
    q: List[int],
    counts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    column: int,
    new_row: int
) -> int:
    """
    Computes the energy change Delta in strictly O(1) time without modifying the board.
    Delta = E_before - E_after.
    A positive Delta means improvement (lower energy).

    Formula from Section 2 of PDF:
      Cold = (R[a] - 1) + (D+[a + c] - 1) + (D-[a - c + N - 1] - 1)
      Cnew = R[r] + D+[r + c] + D-[r - c + N - 1] - 3 * 1[r == a]
      Delta = Cold - Cnew
    """
    n = len(q)
    old_row = q[column]
    R, D_plus, D_minus = counts

    # Old conflicts contributed by queen at (column, old_row)
    c_old = (
        (R[old_row] - 1)
        + (D_plus[old_row + column] - 1)
        + (D_minus[old_row - column + n - 1] - 1)
    )

    # New conflicts contributed if queen moves to (column, new_row)
    # The -3 * 1[new_row == old_row] handles no-op proposals correctly
    is_same = 1 if new_row == old_row else 0
    c_new = (
        R[new_row]
        + D_plus[new_row + column]
        + D_minus[new_row - column + n - 1]
        - 3 * is_same
    )

    return int(c_old - c_new)


def apply_move_in_place(
    q: List[int],
    counts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    E: int,
    column: int,
    new_row: int
) -> int:
    """
    Updates the board q, count arrays, and energy E in-place in O(1) time.
    Returns the updated energy E_child = E - delta.
    """
    old_row = q[column]
    if old_row == new_row:
        return E  # No-op

    delta = move_delta(q, counts, column, new_row)
    n = len(q)
    R, D_plus, D_minus = counts

    # Decrement counts for old position
    R[old_row] -= 1
    D_plus[old_row + column] -= 1
    D_minus[old_row - column + n - 1] -= 1

    # Increment counts for new position
    R[new_row] += 1
    D_plus[new_row + column] += 1
    D_minus[new_row - column + n - 1] += 1

    # Update board array
    q[column] = new_row

    return E - delta


def copy_state(
    q: List[int],
    counts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    E: int
) -> Tuple[List[int], Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    """
    Creates an independent deep copy of the board and count arrays in O(N).
    Ensures children never mutate parent or sibling states.
    """
    R, D_plus, D_minus = counts
    return list(q), (R.copy(), D_plus.copy(), D_minus.copy()), E
