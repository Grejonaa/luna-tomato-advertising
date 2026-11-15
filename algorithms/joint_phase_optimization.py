"""Joint optimization of spacecraft phases using Differential Evolution.

The spacecraft selection counts remain fixed. The algorithm optimizes pairs
of active spacecraft phases simultaneously and validates every candidate
with the official ESA UDP fitness function.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution


TAU = 2.0 * np.pi


def _active_phase_indices(udp, selections):
    """Get chromosome indices for all active spacecraft phases."""
    active = []

    for orbit_idx, count in enumerate(selections):
        if orbit_idx < udp.num_dro:
            family_offset = 0
            local_orbit = orbit_idx

        elif orbit_idx < udp.num_dro + udp.num_lyap:
            family_offset = udp.num_dro * udp.max_per_orbit
            local_orbit = orbit_idx - udp.num_dro

        else:
            family_offset = (
                udp.num_dro + udp.num_lyap
            ) * udp.max_per_orbit
            local_orbit = (
                orbit_idx - udp.num_dro - udp.num_lyap
            )

        for slot in range(int(count)):
            phase_index = (
                udp.num_orbits
                + family_offset
                + local_orbit * udp.max_per_orbit
                + slot
            )
            active.append(phase_index)

    return active


def _official_mse(udp, solution):
    """Return the MSE calculated by the official ESA evaluator."""
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def joint_phase_optimization(
    initial_solution,
    udp,
    passes=3,
    blocks_per_pass=5,
    max_iterations=12,
    population_size=6,
    seed=42,
):
    """Optimize pairs of spacecraft phases simultaneously."""
    start_time = time.time()
    rng = np.random.default_rng(seed)

    current = np.asarray(initial_solution, dtype=float).copy()
    best = current.copy()

    selections = np.rint(
        current[:udp.num_orbits]
    ).astype(int)

    if np.any(selections < 0) or np.any(
        selections > udp.max_per_orbit
    ):
        raise ValueError("Invalid spacecraft selection counts.")

    active_indices = _active_phase_indices(udp, selections)

    if len(active_indices) < 2:
        raise ValueError(
            "At least two active spacecraft are required."
        )

    current_mse = _official_mse(udp, current)
    best_mse = current_mse
    spacecraft_count = int(np.sum(selections))

    print("\nJOINT PHASE OPTIMIZATION")
    print("------------------------")
    print(f"Spacecraft: {spacecraft_count}")
    print(f"Initial official MSE: {current_mse:.12f}")
    print(f"Passes: {passes}")
    print(f"Pairs per pass: {blocks_per_pass}")
    print(f"DE max iterations: {max_iterations}")
    print(f"Population size: {population_size}")

    for pass_number in range(1, passes + 1):
        shuffled = np.array(active_indices, dtype=int)
        rng.shuffle(shuffled)

        improvements = 0
        blocks = []

        for i in range(min(blocks_per_pass, len(shuffled) // 2)):
            pair = shuffled[2 * i:2 * i + 2]
            blocks.append(pair)

        for block_number, pair in enumerate(blocks, start=1):
            pair = [int(index) for index in pair]

            starting_phases = np.array(
                [current[index] % TAU for index in pair],
                dtype=float,
            )

            def objective(candidate_phases):
                candidate = current.copy()

                for index, phase in zip(pair, candidate_phases):
                    candidate[index] = phase % TAU

                return _official_mse(udp, candidate)

            result = differential_evolution(
                objective,
                bounds=[(0.0, TAU), (0.0, TAU)],
                maxiter=max_iterations,
                popsize=population_size,
                tol=0.001,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=int(rng.integers(0, 2**31 - 1)),
                x0=starting_phases,
                polish=False,
                updating="immediate",
                workers=1,
            )

            candidate = current.copy()

            for index, phase in zip(pair, result.x):
                candidate[index] = phase % TAU

            candidate_mse = _official_mse(udp, candidate)

            if candidate_mse < current_mse:
                current = candidate
                current_mse = candidate_mse
                improvements += 1

                if current_mse < best_mse:
                    best = current.copy()
                    best_mse = current_mse

            print(
                f"  Pair {block_number}/{len(blocks)} | "
                f"Current MSE: {current_mse:.12f} | "
                f"Best MSE: {best_mse:.12f}"
            )

        print(
            f"Pass {pass_number}/{passes} | "
            f"MSE: {current_mse:.12f} | "
            f"Best: {best_mse:.12f} | "
            f"Improvements: {improvements}"
        )

        if improvements == 0:
            print("No improvement in this pass.")
            break

    objective_value, spacecraft, final_mse = udp.fitness(
        best, postprocess=True
    )

    result_data = {
        "best_solution": best,
        "objective": float(objective_value[0]),
        "spacecraft": int(spacecraft),
        "mse": float(final_mse),
        "feasible": bool(final_mse <= udp.mse_thresh),
        "runtime": time.time() - start_time,
    }

    print("\nFINAL RESULT")
    print("------------")
    print(f"Spacecraft: {result_data['spacecraft']}")
    print(f"MSE: {result_data['mse']:.12f}")
    print(f"Feasible: {result_data['feasible']}")
    print(f"Runtime: {result_data['runtime']:.2f} seconds")

    return result_data