"""Variable-budget fleet minimization for Luna Tomato Advertising.

The algorithm repeatedly removes one spacecraft, then invokes the existing
fixed-budget Adaptive Simulated Annealing to repair/refine the smaller fleet.
A reduction is accepted only when the official ESA evaluator reports MSE <=
its feasibility threshold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from algorithms.adaptive_simulated_annealing import (
    OptimizationResult as FixedBudgetResult,
    _active_slots,
    _remove_slot,
    adaptive_simulated_annealing,
)


@dataclass
class VariableBudgetResult:
    best_solution: np.ndarray
    objective: float
    spacecraft: int
    mse: float
    feasible: bool
    runtime: float
    reductions_accepted: int
    attempts: int


def _evaluate(udp, solution: np.ndarray) -> tuple[float, int, float]:
    objective, spacecraft, mse = udp.fitness(solution, postprocess=True)
    return float(objective[0]), int(spacecraft), float(mse)


def _rank_single_removals(udp, solution: np.ndarray, limit: int):
    """Return promising one-spacecraft removals, ranked by official MSE."""
    candidates = []
    for orbit, slot in _active_slots(solution, udp):
        reduced = _remove_slot(solution, udp, orbit, slot)
        _, count, mse = _evaluate(udp, reduced)
        candidates.append((mse, count, reduced, orbit, slot))

    candidates.sort(key=lambda item: item[0])
    return candidates[:max(1, limit)]


def variable_budget_optimization(
    udp,
    initial_solution: np.ndarray,
    min_spacecraft: int = 1,
    candidates_per_reduction: int = 3,
    restarts_per_candidate: int = 1,
    iterations_per_attempt: int = 3000,
    seed: int = 42,
    refine_every: int = 1000,
    log_every: int = 1000,
) -> VariableBudgetResult:
    """Try to minimize spacecraft count while maintaining official feasibility.

    For each reduction, the algorithm tests the most promising single-satellite
    removals. Each smaller fleet is refined with fixed-budget simulated annealing.
    The current feasible solution is retained if no smaller feasible fleet is found.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    incumbent = np.asarray(initial_solution, dtype=float).copy()
    objective, count, mse = _evaluate(udp, incumbent)

    if mse > udp.mse_thresh:
        raise ValueError(
            f"Initial solution is not feasible: MSE={mse:.12f}, "
            f"threshold={udp.mse_thresh}. Start from a feasible solution."
        )

    if min_spacecraft < 1:
        raise ValueError("min_spacecraft must be at least 1.")

    accepted_reductions = 0
    total_attempts = 0

    print("\nVARIABLE-BUDGET FLEET OPTIMIZATION")
    print("----------------------------------")
    print(f"Starting fleet: {count} spacecraft | MSE: {mse:.12f}")
    print(f"Feasibility threshold: {udp.mse_thresh}")

    while count > min_spacecraft:
        target_count = count - 1
        print(f"\nTrying to reduce fleet: {count} -> {target_count}")

        ranked = _rank_single_removals(
            udp, incumbent, candidates_per_reduction
        )
        best_smaller = None
        best_smaller_mse = float("inf")

        for candidate_index, (initial_mse, _, reduced, orbit, slot) in enumerate(ranked, 1):
            print(
                f"  Candidate {candidate_index}/{len(ranked)} "
                f"(remove orbit {orbit}, slot {slot}) | "
                f"initial MSE: {initial_mse:.12f}"
            )

            for restart in range(restarts_per_candidate):
                total_attempts += 1
                run_seed = int(rng.integers(0, 2**31 - 1))
                refined: FixedBudgetResult = adaptive_simulated_annealing(
                    udp=udp,
                    initial_solution=reduced,
                    iterations=iterations_per_attempt,
                    seed=run_seed,
                    refine_every=refine_every,
                    log_every=log_every,
                )

                official_objective, official_count, official_mse = _evaluate(
                    udp, refined.best_solution
                )

                print(
                    f"    Attempt {restart + 1}/{restarts_per_candidate}: "
                    f"N={official_count}, MSE={official_mse:.12f}, "
                    f"feasible={official_mse <= udp.mse_thresh}"
                )

                if official_count != target_count:
                    print("    Skipped: spacecraft budget changed unexpectedly.")
                    continue

                if official_mse < best_smaller_mse:
                    best_smaller = refined.best_solution.copy()
                    best_smaller_mse = official_mse

                if official_mse <= udp.mse_thresh:
                    break

            if best_smaller is not None and best_smaller_mse <= udp.mse_thresh:
                break

        if best_smaller is None or best_smaller_mse > udp.mse_thresh:
            print(
                f"Could not find a feasible fleet with {target_count} spacecraft."
            )
            print("Keeping the best previously feasible fleet.")
            break

        incumbent = best_smaller
        objective, count, mse = _evaluate(udp, incumbent)
        accepted_reductions += 1
        print(
            f"ACCEPTED: {count} spacecraft | MSE: {mse:.12f} "
            f"| objective: {objective:.8f}"
        )

    objective, count, mse = _evaluate(udp, incumbent)
    return VariableBudgetResult(
        best_solution=incumbent,
        objective=objective,
        spacecraft=count,
        mse=mse,
        feasible=mse <= udp.mse_thresh,
        runtime=time.time() - start,
        reductions_accepted=accepted_reductions,
        attempts=total_attempts,
    )
