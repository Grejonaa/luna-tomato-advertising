
import numpy as np
from scipy.optimize import minimize_scalar
import time


def evaluate_mse(solution, udp):
    """
    Evaluates a solution and returns its Mean Squared Error.
    """
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def optimize_phases(
    initial_solution,
    udp,
    max_passes=2,
    grid_points=8,
    max_iterations=20,
    seed=42
):
    """
    Coordinate-based phase optimization.

    The number of spacecraft and selected orbits remain fixed.
    Only the phase of each active spacecraft is optimized.
    """

    start_time = time.time()
    rng = np.random.default_rng(seed)

    best_solution = np.array(initial_solution, dtype=float).copy()

    num_orbits = udp.num_orbits
    max_per_orbit = udp.max_per_orbit

    selections = best_solution[:num_orbits]

    # Find phase indexes belonging to active spacecraft.
    active_phase_indices = []

    for orbit_idx in range(num_orbits):
        number_of_spacecraft = int(selections[orbit_idx])

        for spacecraft_idx in range(number_of_spacecraft):
            phase_index = (
                num_orbits
                + orbit_idx * max_per_orbit
                + spacecraft_idx
            )

            active_phase_indices.append(phase_index)

    best_mse = evaluate_mse(best_solution, udp)

    print("\nPHASE OPTIMIZATION")
    print("------------------------------")
    print(f"Initial MSE: {best_mse:.6f}")
    print(f"Active spacecraft: {len(active_phase_indices)}")

    # Optimize one phase at a time.
    for pass_number in range(max_passes):

        rng.shuffle(active_phase_indices)
        improvements = 0

        for phase_index in active_phase_indices:

            original_phase = best_solution[phase_index]

            # Coarse search across the full phase interval.
            angles = np.linspace(
                0,
                2 * np.pi,
                grid_points,
                endpoint=False
            )

            candidates = []

            for angle in angles:
                candidate = best_solution.copy()
                candidate[phase_index] = angle

                mse = evaluate_mse(candidate, udp)
                candidates.append((mse, angle))

            candidates.sort(key=lambda item: item[0])

            best_grid_mse, best_grid_angle = candidates[0]

            # Refine around the best grid angle.
            step = 2 * np.pi / grid_points

            def objective(angle):
                candidate = best_solution.copy()
                candidate[phase_index] = angle % (2 * np.pi)
                return evaluate_mse(candidate, udp)

            result = minimize_scalar(
                objective,
                bounds=(
                    best_grid_angle - step,
                    best_grid_angle + step
                ),
                method="bounded",
                options={
                    "maxiter": max_iterations,
                    "xatol": 0.02
                }
            )

            refined_phase = result.x % (2 * np.pi)

            # Compare the original, grid and refined phases.
            options = [
                (best_mse, original_phase),
                (best_grid_mse, best_grid_angle),
                (result.fun, refined_phase)
            ]

            local_best_mse, local_best_phase = min(
                options,
                key=lambda item: item[0]
            )

            if local_best_mse < best_mse:
                best_solution[phase_index] = local_best_phase
                best_mse = local_best_mse
                improvements += 1

        print(
            f"Pass {pass_number + 1}/{max_passes} | "
            f"MSE: {best_mse:.6f} | "
            f"Improvements: {improvements}"
        )

        if improvements == 0:
            print("No further improvement. Stopping.")
            break

    objective, spacecraft, final_mse = udp.fitness(
        best_solution,
        postprocess=True
    )

    runtime = time.time() - start_time

    return {
        "best_solution": best_solution,
        "objective": float(objective[0]),
        "spacecraft": int(spacecraft),
        "mse": float(final_mse),
        "feasible": float(final_mse) <= udp.mse_thresh,
        "runtime": runtime
    }
