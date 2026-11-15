
import numpy as np
import time
import math


def generate_neighbor(solution, udp, rng):
    """Generate a neighboring solution using one random operation."""

    candidate = solution.copy()

    num_orbits = udp.num_orbits
    max_per_orbit = udp.max_per_orbit

    selections = candidate[:num_orbits].astype(int)

    operation = rng.choice(["phase", "add", "remove"])

    if operation == "phase":
        active_indices = []

        for orbit_idx, count in enumerate(selections):
            for spacecraft_idx in range(count):
                index = (
                    num_orbits
                    + orbit_idx * max_per_orbit
                    + spacecraft_idx
                )
                active_indices.append(index)

        if active_indices:
            index = int(rng.choice(active_indices))
            candidate[index] = rng.uniform(0, 2 * np.pi)

    elif operation == "add":
        available = np.where(selections < max_per_orbit)[0]

        if len(available) > 0:
            orbit_idx = int(rng.choice(available))
            count = selections[orbit_idx]

            phase_index = (
                num_orbits
                + orbit_idx * max_per_orbit
                + count
            )

            candidate[phase_index] = rng.uniform(0, 2 * np.pi)
            candidate[orbit_idx] += 1

    elif operation == "remove":
        available = np.where(selections > 0)[0]

        if len(available) > 0:
            orbit_idx = int(rng.choice(available))
            count = selections[orbit_idx]

            # Move the last active phase into the removed position.
            last_phase_index = (
                num_orbits
                + orbit_idx * max_per_orbit
                + count - 1
            )

            candidate[orbit_idx] -= 1
            candidate[last_phase_index] = 0.0

    return candidate


def simulated_annealing(
    udp,
    initial_solution,
    iterations=500,
    initial_temperature=10.0,
    cooling_rate=0.995,
    seed=42
):
    """Optimize a solution using Simulated Annealing."""

    rng = np.random.default_rng(seed)

    current_solution = initial_solution.copy()

    objective, count, mse = udp.fitness(
        current_solution,
        postprocess=True
    )

    current_objective = float(objective[0])
    current_count = int(count)
    current_mse = float(mse)

    best_solution = current_solution.copy()
    best_objective = current_objective
    best_count = current_count
    best_mse = current_mse

    temperature = initial_temperature
    history = [best_objective]

    start_time = time.perf_counter()

    for iteration in range(iterations):

        candidate = generate_neighbor(
            current_solution,
            udp,
            rng
        )

        objective, count, mse = udp.fitness(
            candidate,
            postprocess=True
        )

        candidate_objective = float(objective[0])
        candidate_count = int(count)
        candidate_mse = float(mse)

        delta = candidate_objective - current_objective

        # Accept improvements or probabilistically accept worse solutions.
        if delta < 0 or (
            temperature > 0 and
            rng.random() < math.exp(-delta / temperature)
        ):
            current_solution = candidate
            current_objective = candidate_objective
            current_count = candidate_count
            current_mse = candidate_mse

        # Update the best solution found so far.
        if current_objective < best_objective:
            best_solution = current_solution.copy()
            best_objective = current_objective
            best_count = current_count
            best_mse = current_mse

        temperature *= cooling_rate
        history.append(best_objective)

        if (iteration + 1) % 100 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations} | "
                f"Best objective: {best_objective:.6f} | "
                f"MSE: {best_mse:.6f} | "
                f"Spacecraft: {best_count} | "
                f"Temperature: {temperature:.5f}"
            )

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
