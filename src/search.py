"""
search.py - Search policies, move generators, budget accounting, and metrics
for N-Queens Local Beam Search study.

Implements:
  - Method A: One-state local search (k = 1)
  - Method B: Independent multi-start (k independent trajectories)
  - Method C: Greedy local beam search (pooled top-k selection)
  - Method D: Stochastic local beam search (Boltzmann/temperature selection)
  - Method E: Diversity-aware beam search (Hamming distance filter + fallback)
  - Baseline: Random-Restart Strict Hill Climbing (Exp 4)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import time
import numpy as np

from src.board import (
    make_counts,
    energy_from_counts,
    move_delta,
    apply_move_in_place,
    copy_state,
    validate_pairwise,
    hamming_distance,
    population_diversity,
)


@dataclass
class State:
    """Represents a live board in the beam population with its own count arrays."""
    q: List[int]
    counts: Tuple[np.ndarray, np.ndarray, np.ndarray]
    energy: int
    ancestor_id: int  # Identifies which initial board this state descended from


@dataclass
class CandidateMove:
    """Represents an unmaterialized candidate move evaluated against a parent."""
    parent_idx: int
    col: int
    new_row: int
    energy: int
    board_tuple: Tuple[int, ...]
    ancestor_id: int


@dataclass
class SearchResult:
    """Complete summary of a single search trial."""
    method: str
    n: int
    k: int
    m: int
    budget: int
    seed: int
    solved: bool
    evals_used: int
    evals_to_solution: Optional[int]
    best_energy: int
    best_board: List[int]
    trace: Dict[str, List] = field(default_factory=dict)
    runtime_sec: float = 0.0


def get_default_m(n: int) -> int:
    """
    Returns m successors to sample per state:
      - For N = 8: all 56 moves.
      - For N > 8: min(2N, N * (N - 1)).
    """
    if n == 8:
        return 56
    return min(2 * n, n * (n - 1))


def sample_moves_for_board(
    q: List[int],
    m: int,
    rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """
    Generates m distinct one-queen moves uniformly without replacement.
    A move changes queen in column c to new_row != q[c].
    Total possible distinct moves is N * (N - 1).
    """
    n = len(q)
    total_possible = n * (n - 1)
    m = min(m, total_possible)

    if m == total_possible:
        # Enumerate all valid moves deterministically then shuffle with rng
        all_moves = []
        for c in range(n):
            current_row = q[c]
            for r in range(n):
                if r != current_row:
                    all_moves.append((c, r))
        rng.shuffle(all_moves)
        return all_moves

    # For m < total_possible, sample m distinct indices using flat index bijection
    sampled_indices = rng.choice(total_possible, size=m, replace=False)
    moves = []
    for idx in sampled_indices:
        col = int(idx // (n - 1))
        offset = int(idx % (n - 1))
        current_row = q[col]
        new_row = offset if offset < current_row else offset + 1
        moves.append((col, new_row))

    return moves


def allocate_moves(remaining: int, k: int, m: int, gen_idx: int) -> List[int]:
    """
    Allocates remaining evaluation budget round-robin across parents,
    rotating the starting parent between generations.
    """
    plan = [0] * k
    start_parent = gen_idx % k
    moves_left = remaining

    for i in range(k):
        parent = (start_parent + i) % k
        take = min(m, moves_left)
        plan[parent] = take
        moves_left -= take
        if moves_left <= 0:
            break

    return plan


def deduplicate_candidates(candidates: List[CandidateMove]) -> List[CandidateMove]:
    """
    Removes exact duplicate board configurations from a pooled candidate list.
    Preserves first seen candidate.
    """
    seen: Set[Tuple[int, ...]] = set()
    unique = []
    for cand in candidates:
        if cand.board_tuple not in seen:
            seen.add(cand.board_tuple)
            unique.append(cand)
    return unique


def select_stochastic_order(
    candidates: List[CandidateMove],
    tau: float,
    rng: np.random.Generator,
    num_to_select: Optional[int] = None
) -> List[CandidateMove]:
    """
    Produces a stochastic ordering of candidates sampled without replacement
    using the Boltzmann distribution:
      Pr(q | U) ~ exp(-(E(q) - E_min) / tau)
    Recomputes E_min at each draw for numerical stability.
    If num_to_select is specified, stops once that many items are drawn.
    """
    pool = list(candidates)
    ordered = []
    limit = len(pool) if num_to_select is None else min(num_to_select, len(pool))

    while pool and len(ordered) < limit:
        energies = np.array([c.energy for c in pool], dtype=np.float64)
        e_min = np.min(energies)
        # Shift and scale by temperature tau
        logits = -(energies - e_min) / max(tau, 1e-6)
        # Numerical stability clip
        logits = np.clip(logits, -500.0, 0.0)
        weights = np.exp(logits)
        prob = weights / np.sum(weights)

        chosen_idx = int(rng.choice(len(pool), p=prob))
        ordered.append(pool.pop(chosen_idx))

    return ordered


def run_search(
    method: str,
    n: int,
    k: int,
    budget: int,
    seed: int,
    m: Optional[int] = None,
    tau: float = 1.0,
    d_min: float = 0.10,
    max_stagnant_steps: int = 10,
) -> SearchResult:
    """
    Universal dispatcher for N-Queens search experiments.
    Methods supported:
      - 'A': One-state search (k=1)
      - 'B': Independent multi-start (k trajectories)
      - 'C': Greedy local beam search
      - 'D': Stochastic local beam search
      - 'E': Diversity-aware local beam search
      - 'Baseline_RHC': Random-Restart Strict Hill Climbing
    """
    method = method.upper()
    if m is None:
        m = get_default_m(n)

    t0 = time.perf_counter()

    if method == "BASELINE_RHC" or method == "RHC":
        res = _run_baseline_rhc(n, budget, seed, m, max_stagnant_steps)
    elif method == "A":
        res = _run_method_a(n, budget, seed, m)
    elif method == "B":
        res = _run_method_b(n, k, budget, seed, m)
    elif method in ("C", "D", "E"):
        res = _run_beam(method, n, k, budget, seed, m, tau, d_min)
    else:
        raise ValueError(f"Unknown method '{method}'")

    res.runtime_sec = time.perf_counter() - t0
    return res


def _run_method_a(n: int, budget: int, seed: int, m: int) -> SearchResult:
    """Method A: One-state local search (k = 1)."""
    # Simply dispatch to beam with k = 1 and Method C (identical logic)
    res = _run_beam("C", n, k=1, budget=budget, seed=seed, m=m, tau=1.0, d_min=0.0)
    res.method = "A"
    return res


def _run_method_b(n: int, k: int, budget: int, seed: int, m: int) -> SearchResult:
    """
    Method B: Independent multi-start.
    Maintains k independent trajectories, each running Method A's rule.
    Total budget B is shared, alternating generation turns between trajectories.
    Multi-start streams are strictly independent across trajectories.
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(k + 1)
    init_rng = np.random.default_rng(child_seeds[0])
    traj_rngs = [np.random.default_rng(child_seeds[i + 1]) for i in range(k)]

    # 1. Initialize and score k boards
    population: List[State] = []
    used_evals = 0
    best_seen_energy = float("inf")
    best_seen_board: List[int] = []

    trace: Dict[str, List] = {
        "evals": [],
        "best_energy": [],
        "diversity": [],
        "unique_states": [],
        "distinct_ancestors": [],
    }

    for i in range(k):
        q = init_rng.integers(0, n, size=n).tolist()
        counts = make_counts(q)
        e = energy_from_counts(counts)
        used_evals += 1
        state = State(q=q, counts=counts, energy=e, ancestor_id=i)
        population.append(state)

        if e < best_seen_energy:
            best_seen_energy = e
            best_seen_board = list(q)

        if e == 0:
            assert validate_pairwise(q) == 0
            trace["evals"].append(used_evals)
            trace["best_energy"].append(0)
            trace["diversity"].append(population_diversity([s.q for s in population]))
            trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
            trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))
            return SearchResult(
                method="B", n=n, k=k, m=m, budget=budget, seed=seed,
                solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                best_energy=0, best_board=list(q), trace=trace
            )

    gen_idx = 0
    while used_evals < budget:
        trace["evals"].append(used_evals)
        trace["best_energy"].append(best_seen_energy)
        trace["diversity"].append(population_diversity([s.q for s in population]))
        trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
        trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))

        plan = allocate_moves(budget - used_evals, k, m, gen_idx)
        if sum(plan) == 0:
            break

        # Each trajectory evaluates and selects independently
        for i in range(k):
            moves_to_eval = plan[i]
            if moves_to_eval == 0:
                continue

            parent = population[i]
            rng = traj_rngs[i]
            sampled_moves = sample_moves_for_board(parent.q, moves_to_eval, rng)

            candidates: List[CandidateMove] = []
            for col, new_row in sampled_moves:
                delta = move_delta(parent.q, parent.counts, col, new_row)
                c_energy = parent.energy - delta
                used_evals += 1

                board_tup = tuple(new_row if idx == col else parent.q[idx] for idx in range(n))
                cand = CandidateMove(
                    parent_idx=i, col=col, new_row=new_row, energy=c_energy,
                    board_tuple=board_tup, ancestor_id=parent.ancestor_id
                )
                candidates.append(cand)

                if c_energy < best_seen_energy:
                    best_seen_energy = c_energy
                    best_seen_board = list(cand.board_tuple)

                if c_energy == 0:
                    child_q = list(cand.board_tuple)
                    assert validate_pairwise(child_q) == 0
                    trace["evals"].append(used_evals)
                    trace["best_energy"].append(0)
                    return SearchResult(
                        method="B", n=n, k=k, m=m, budget=budget, seed=seed,
                        solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                        best_energy=0, best_board=child_q, trace=trace
                    )

                if used_evals >= budget:
                    break

            if candidates:
                # Select best successor for trajectory i (breaking ties with rng)
                min_e = min(c.energy for c in candidates)
                best_cands = [c for c in candidates if c.energy == min_e]
                winner = best_cands[int(rng.integers(0, len(best_cands)))]

                # Apply move to trajectory state in place
                apply_move_in_place(parent.q, parent.counts, parent.energy, winner.col, winner.new_row)
                parent.energy = winner.energy

            if used_evals >= budget:
                break

        gen_idx += 1

    trace["evals"].append(used_evals)
    trace["best_energy"].append(best_seen_energy)
    trace["diversity"].append(population_diversity([s.q for s in population]))
    trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
    trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))

    return SearchResult(
        method="B", n=n, k=k, m=m, budget=budget, seed=seed,
        solved=False, evals_used=used_evals, evals_to_solution=None,
        best_energy=best_seen_energy, best_board=best_seen_board, trace=trace
    )


def _run_beam(
    policy: str,
    n: int,
    k: int,
    budget: int,
    seed: int,
    m: int,
    tau: float,
    d_min: float
) -> SearchResult:
    """
    Local Beam Search engine supporting:
      - 'C': Greedy Local Beam (top-k lowest energy)
      - 'D': Stochastic Local Beam (Boltzmann selection)
      - 'E': Diversity-Aware Beam (Hamming distance threshold d_min)
    """
    seed_seq = np.random.SeedSequence(seed)
    init_seed, move_seed, select_seed = seed_seq.spawn(3)
    init_rng = np.random.default_rng(init_seed)
    move_rng = np.random.default_rng(move_seed)
    select_rng = np.random.default_rng(select_seed)

    population: List[State] = []
    used_evals = 0
    best_seen_energy = float("inf")
    best_seen_board: List[int] = []

    trace: Dict[str, List] = {
        "evals": [],
        "best_energy": [],
        "diversity": [],
        "unique_states": [],
        "distinct_ancestors": [],
        "relaxed_selections": [],
    }

    # 1. Initialize and score k boards
    for i in range(k):
        q = init_rng.integers(0, n, size=n).tolist()
        counts = make_counts(q)
        e = energy_from_counts(counts)
        used_evals += 1
        state = State(q=q, counts=counts, energy=e, ancestor_id=i)
        population.append(state)

        if e < best_seen_energy:
            best_seen_energy = e
            best_seen_board = list(q)

        if e == 0:
            assert validate_pairwise(q) == 0
            trace["evals"].append(used_evals)
            trace["best_energy"].append(0)
            trace["diversity"].append(population_diversity([s.q for s in population]))
            trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
            trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))
            trace["relaxed_selections"].append(0)
            return SearchResult(
                method=policy, n=n, k=k, m=m, budget=budget, seed=seed,
                solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                best_energy=0, best_board=list(q), trace=trace
            )

    gen_idx = 0
    total_relaxed_selections = 0

    while used_evals < budget:
        trace["evals"].append(used_evals)
        trace["best_energy"].append(best_seen_energy)
        trace["diversity"].append(population_diversity([s.q for s in population]))
        trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
        trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))
        trace["relaxed_selections"].append(total_relaxed_selections)

        plan = allocate_moves(budget - used_evals, k, m, gen_idx)
        if sum(plan) == 0:
            break

        # Pool candidates across parents
        candidates: List[CandidateMove] = []
        for p_idx in range(k):
            moves_to_eval = plan[p_idx]
            if moves_to_eval == 0:
                continue

            parent = population[p_idx]
            sampled_moves = sample_moves_for_board(parent.q, moves_to_eval, move_rng)

            for col, new_row in sampled_moves:
                delta = move_delta(parent.q, parent.counts, col, new_row)
                c_energy = parent.energy - delta
                used_evals += 1

                board_tup = tuple(new_row if idx == col else parent.q[idx] for idx in range(n))
                cand = CandidateMove(
                    parent_idx=p_idx, col=col, new_row=new_row, energy=c_energy,
                    board_tuple=board_tup, ancestor_id=parent.ancestor_id
                )
                candidates.append(cand)

                if c_energy < best_seen_energy:
                    best_seen_energy = c_energy
                    best_seen_board = list(cand.board_tuple)

                if c_energy == 0:
                    child_q = list(cand.board_tuple)
                    assert validate_pairwise(child_q) == 0
                    trace["evals"].append(used_evals)
                    trace["best_energy"].append(0)
                    return SearchResult(
                        method=policy, n=n, k=k, m=m, budget=budget, seed=seed,
                        solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                        best_energy=0, best_board=child_q, trace=trace
                    )

                if used_evals >= budget:
                    break

            if used_evals >= budget:
                break

        if not candidates:
            break

        # Deduplicate exact boards across pooled candidates
        unique_cands = deduplicate_candidates(candidates)

        # Select k states according to policy
        selected_cands: List[CandidateMove] = []

        if policy == "C":
            # Greedy top-k lowest energy (break ties with seeded randomness)
            rand_keys = select_rng.random(len(unique_cands))
            paired = list(zip(unique_cands, rand_keys))
            paired.sort(key=lambda item: (item[0].energy, item[1]))
            selected_cands = [item[0] for item in paired[:k]]

        elif policy == "D":
            # Stochastic Boltzmann selection: only need to draw k items
            selected_cands = select_stochastic_order(unique_cands, tau, select_rng, num_to_select=k)

        elif policy == "E":
            # Diversity-Aware Beam: weighted random order + Hamming filter
            stochastic_order = select_stochastic_order(unique_cands, tau, select_rng)
            selected_boards: List[Tuple[int, ...]] = []

            for cand in stochastic_order:
                # Check Hamming distance against all currently accepted
                too_close = False
                for sb in selected_boards:
                    raw_dist = sum(1 for a, b in zip(cand.board_tuple, sb) if a != b)
                    if (raw_dist / n) < d_min:
                        too_close = True
                        break
                if not too_close:
                    selected_cands.append(cand)
                    selected_boards.append(cand.board_tuple)
                    if len(selected_cands) == k:
                        break

            # Fallback if fewer than k survive: fill vacant slots maximizing distance to closest selected
            if len(selected_cands) < k:
                surplus_needed = k - len(selected_cands)
                remaining = [c for c in unique_cands if c.board_tuple not in selected_boards]

                while surplus_needed > 0 and remaining:
                    total_relaxed_selections += 1
                    # Score remaining candidates by min distance to selected set
                    best_cand = None
                    best_dist = -1.0
                    best_energy = float("inf")
                    best_rnd = -1.0

                    for cand in remaining:
                        min_d = min(
                            sum(1 for a, b in zip(cand.board_tuple, sb) if a != b) / n
                            for sb in selected_boards
                        ) if selected_boards else 1.0
                        rnd_val = select_rng.random()

                        # Maximize min_d, minimize energy, maximize rnd
                        if (min_d > best_dist or
                            (min_d == best_dist and cand.energy < best_energy) or
                            (min_d == best_dist and cand.energy == best_energy and rnd_val > best_rnd)):
                            best_dist = min_d
                            best_energy = cand.energy
                            best_rnd = rnd_val
                            best_cand = cand

                    selected_cands.append(best_cand)
                    selected_boards.append(best_cand.board_tuple)
                    remaining.remove(best_cand)
                    surplus_needed -= 1

        # Materialize selected states and their independent count arrays
        new_population: List[State] = []
        for winner in selected_cands:
            parent = population[winner.parent_idx]
            child_q, child_counts, child_e = copy_state(parent.q, parent.counts, parent.energy)
            actual_e = apply_move_in_place(child_q, child_counts, child_e, winner.col, winner.new_row)
            new_population.append(State(
                q=child_q, counts=child_counts, energy=actual_e, ancestor_id=winner.ancestor_id
            ))

        population = new_population
        gen_idx += 1

    trace["evals"].append(used_evals)
    trace["best_energy"].append(best_seen_energy)
    trace["diversity"].append(population_diversity([s.q for s in population]))
    trace["unique_states"].append(len(set(tuple(s.q) for s in population)))
    trace["distinct_ancestors"].append(len(set(s.ancestor_id for s in population)))
    trace["relaxed_selections"].append(total_relaxed_selections)

    return SearchResult(
        method=policy, n=n, k=k, m=m, budget=budget, seed=seed,
        solved=False, evals_used=used_evals, evals_to_solution=None,
        best_energy=best_seen_energy, best_board=best_seen_board, trace=trace
    )


def _run_baseline_rhc(
    n: int,
    budget: int,
    seed: int,
    m: int,
    max_stagnant_steps: int = 10
) -> SearchResult:
    """
    Experiment 4 Baseline: Random-Restart Strict Hill Climbing.
    Single-state walk that only accepts strictly improving moves (Delta > 0).
    Restarts after max_stagnant_steps (default 10) consecutive non-improving expansions.
    Every initial board and restarted board costs 1 evaluation unit charged to B.
    """
    seed_seq = np.random.SeedSequence(seed)
    init_seed, move_seed, select_seed = seed_seq.spawn(3)
    init_rng = np.random.default_rng(init_seed)
    move_rng = np.random.default_rng(move_seed)
    select_rng = np.random.default_rng(select_seed)

    trace: Dict[str, List] = {
        "evals": [],
        "best_energy": [],
        "restarts": [],
    }

    # Initial board
    q = init_rng.integers(0, n, size=n).tolist()
    counts = make_counts(q)
    energy = energy_from_counts(counts)
    used_evals = 1

    best_seen_energy = energy
    best_seen_board = list(q)
    restarts_count = 0
    stagnant_steps = 0

    if energy == 0:
        assert validate_pairwise(q) == 0
        trace["evals"].append(used_evals)
        trace["best_energy"].append(0)
        trace["restarts"].append(0)
        return SearchResult(
            method="Baseline_RHC", n=n, k=1, m=m, budget=budget, seed=seed,
            solved=True, evals_used=used_evals, evals_to_solution=used_evals,
            best_energy=0, best_board=list(q), trace=trace
        )

    while used_evals < budget:
        trace["evals"].append(used_evals)
        trace["best_energy"].append(best_seen_energy)
        trace["restarts"].append(restarts_count)

        moves_allowed = min(m, budget - used_evals)
        if moves_allowed <= 0:
            break

        sampled_moves = sample_moves_for_board(q, moves_allowed, move_rng)

        best_delta = -float("inf")
        best_improving_moves: List[Tuple[int, int, int]] = []

        for col, new_row in sampled_moves:
            delta = move_delta(q, counts, col, new_row)
            used_evals += 1
            child_e = energy - delta

            if child_e < best_seen_energy:
                best_seen_energy = child_e
                child_board = list(q)
                child_board[col] = new_row
                best_seen_board = child_board

            if child_e == 0:
                child_board = list(q)
                child_board[col] = new_row
                assert validate_pairwise(child_board) == 0
                trace["evals"].append(used_evals)
                trace["best_energy"].append(0)
                trace["restarts"].append(restarts_count)
                return SearchResult(
                    method="Baseline_RHC", n=n, k=1, m=m, budget=budget, seed=seed,
                    solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                    best_energy=0, best_board=child_board, trace=trace
                )

            # Strict improvement only: delta > 0
            if delta > 0:
                if delta > best_delta:
                    best_delta = delta
                    best_improving_moves = [(col, new_row, child_e)]
                elif delta == best_delta:
                    best_improving_moves.append((col, new_row, child_e))

            if used_evals >= budget:
                break

        if best_improving_moves:
            # Pick best move (break ties with select_rng)
            chosen = best_improving_moves[int(select_rng.integers(0, len(best_improving_moves)))]
            energy = apply_move_in_place(q, counts, energy, chosen[0], chosen[1])
            stagnant_steps = 0
        else:
            # Stagnant step: no strictly improving neighbor found in sample
            stagnant_steps += 1
            if stagnant_steps >= max_stagnant_steps:
                # Trigger random restart
                if used_evals < budget:
                    q = init_rng.integers(0, n, size=n).tolist()
                    counts = make_counts(q)
                    energy = energy_from_counts(counts)
                    used_evals += 1
                    restarts_count += 1
                    stagnant_steps = 0

                    if energy < best_seen_energy:
                        best_seen_energy = energy
                        best_seen_board = list(q)

                    if energy == 0:
                        assert validate_pairwise(q) == 0
                        trace["evals"].append(used_evals)
                        trace["best_energy"].append(0)
                        trace["restarts"].append(restarts_count)
                        return SearchResult(
                            method="Baseline_RHC", n=n, k=1, m=m, budget=budget, seed=seed,
                            solved=True, evals_used=used_evals, evals_to_solution=used_evals,
                            best_energy=0, best_board=list(q), trace=trace
                        )

    trace["evals"].append(used_evals)
    trace["best_energy"].append(best_seen_energy)
    trace["restarts"].append(restarts_count)

    return SearchResult(
        method="Baseline_RHC", n=n, k=1, m=m, budget=budget, seed=seed,
        solved=False, evals_used=used_evals, evals_to_solution=None,
        best_energy=best_seen_energy, best_board=best_seen_board, trace=trace
    )
