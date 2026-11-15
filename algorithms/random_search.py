
import numpy as np
import time

from solution import create_random_solution


def random_search(
    udp,
    iterations=50,
    max_satellites=100,
    seed=42
):
    """
    Random Search for the Luna Tomato Advertising problem.

    Generates random candidate solutions and keeps
    the solution with the lowest official ESA objective.
    """

    rng = np.random.default_rng(seed)

    # Start with the solution containing no spacecraft.
    best_solution = np.zeros(
        udp.num_orbits * (1 + udp.max_per_orbit)
    )

    best_objective, best_count, best_mse = udp.fitness(
        best_solution,
        postprocess=True
    )

    best_objective = float(best_objective[0])
    history = [best_objective]

    start_time = time.perf_counter()

    for iteration in range(iterations):

        # Randomly choose how many orbits will be activated.
        number_of_orbits = int(
            rng.integers(1, max_satellites + 1)
        )

        candidate = create_random_solution(
            udp,
            selected_orbits=number_of_orbits,
            rng=rng
        )

        objective, count, mse = udp.fitness(
            candidate,
            postprocess=True
        )

        objective = float(objective[0])

        # Keep the best solution found so far.
        if objective < best_objective:
            best_objective = objective
            best_solution = candidate.copy()
            best_count = int(count)
            best_mse = float(mse)

        history.append(best_objective)

        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations} | "
                f"Best objective: {best_objective:.6f} | "
                f"MSE: {best_mse:.6f}"
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
