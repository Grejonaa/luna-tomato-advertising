
import numpy as np
import time


def hill_climbing(
    udp,
    initial_solution,
    iterations=100,
    initial_step=np.pi / 4,
    min_step=0.01,
    seed=42
):
    """
    Hill Climbing that optimizes active spacecraft phase angles.

    At each iteration:
    1. Select one active spacecraft phase.
    2. Test a positive and negative phase adjustment.
    3. Keep the best improvement.
    4. Reduce the step size if no improvement is found.
    """

    rng = np.random.default_rng(seed)

    current_solution = initial_solution.copy()

    current_objective, current_count, current_mse = udp.fitness(
        current_solution,
        postprocess=True
    )

    current_objective = float(current_objective[0])
    current_count = int(current_count)
    current_mse = float(current_mse)

    best_solution = current_solution.copy()
    best_objective = current_objective
    best_count = current_count
    best_mse = current_mse

    step = initial_step
    history = [best_objective]

    start_time = time.perf_counter()

    # Identify all active phase positions.
    selections = current_solution[:udp.num_orbits].astype(int)

    active_phase_indices = []

    for orbit_idx, spacecraft_count in enumerate(selections):
        for spacecraft_idx in range(spacecraft_count):
            phase_index = (
                udp.num_orbits
                + orbit_idx * udp.max_per_orbit
                + spacecraft_idx
            )
            active_phase_indices.append(phase_index)

    if not active_phase_indices:
        raise ValueError("Initial solution has no active spacecraft.")

    for iteration in range(iterations):

        # Select one active phase randomly.
        phase_index = int(rng.choice(active_phase_indices))

        best_neighbor = None
        best_neighbor_objective = current_objective
        best_neighbor_count = current_count
        best_neighbor_mse = current_mse

        # Test both directions.
        for direction in (-1, 1):

            candidate = current_solution.copy()

            candidate[phase_index] = (
                candidate[phase_index] + direction * step
            ) % (2 * np.pi)

            objective, count, mse = udp.fitness(
                candidate,
                postprocess=True
            )

            objective = float(objective[0])
            count = int(count)
            mse = float(mse)

            if objective < best_neighbor_objective:
                best_neighbor = candidate
                best_neighbor_objective = objective
                best_neighbor_count = count
                best_neighbor_mse = mse

        # Accept the best improving neighbor.
        if best_neighbor is not None:
            current_solution = best_neighbor
            current_objective = best_neighbor_objective
            current_count = best_neighbor_count
            current_mse = best_neighbor_mse

            if current_objective < best_objective:
                best_solution = current_solution.copy()
                best_objective = current_objective
                best_count = current_count
                best_mse = current_mse
        else:
            # Reduce the step size when no improvement is found.
            step *= 0.8

        history.append(best_objective)

        if (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations} | "
                f"Best objective: {best_objective:.6f} | "
                f"MSE: {best_mse:.6f} | "
                f"Step: {step:.5f}"
            )

        if step < min_step:
            print("Minimum step size reached.")
            break

    elapsed_time = time.perf_counter() - start_time

    return {
        "best_solution": best_solution,
        "objective": best_objective,
        "spacecraft": best_count,
        "mse": best_mse,
        "feasible": best_mse <= udp.mse_thresh,
        "history": history,
        "runtime": elapsed_time
    }
