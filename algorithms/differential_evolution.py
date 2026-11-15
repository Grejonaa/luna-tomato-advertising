
import numpy as np
import time
from scipy.optimize import differential_evolution


def evaluate_mse(solution, udp):
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def optimize_with_differential_evolution(
    initial_solution,
    udp,
    group_size=6,
    max_passes=2,
    max_iterations=30,
    population_size=8,
    seed=42
):
    """
    Hybrid grouped Differential Evolution.

    Optimizes active spacecraft phases in small groups.
    Orbit selection and spacecraft count remain unchanged.
    """

    start_time = time.time()
    rng = np.random.default_rng(seed)

    best_solution = np.array(initial_solution, dtype=float).copy()

    num_orbits = udp.num_orbits
    max_per_orbit = udp.max_per_orbit

    # Find phase indexes of all active spacecraft.
    active_indices = []

    for orbit_idx in range(num_orbits):
        count = int(best_solution[orbit_idx])

        for spacecraft_idx in range(count):
            phase_index = (
                num_orbits
                + orbit_idx * max_per_orbit
                + spacecraft_idx
            )
            active_indices.append(phase_index)

    if not active_indices:
        raise ValueError("No active spacecraft to optimize.")

    initial_mse = evaluate_mse(best_solution, udp)
    best_mse = initial_mse

    print("\nDIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("--------------------------------------")
    print(f"Initial MSE: {initial_mse:.6f}")
    print(f"Active phases: {len(active_indices)}")
    print(f"Group size: {group_size}")

    groups = [
        active_indices[i:i + group_size]
        for i in range(0, len(active_indices), group_size)
    ]

    for pass_number in range(max_passes):

        rng.shuffle(groups)
        pass_improvements = 0

        print(f"\nPASS {pass_number + 1}/{max_passes}")

        for group_number, group in enumerate(groups, start=1):

            current_phases = best_solution[group].copy()

            def objective(phases):
                candidate = best_solution.copy()
                candidate[group] = np.mod(phases, 2 * np.pi)
                return evaluate_mse(candidate, udp)

            result = differential_evolution(
                objective,
                bounds=[(0, 2 * np.pi)] * len(group),
                strategy="best1bin",
                maxiter=max_iterations,
                popsize=population_size,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=0.001,
                atol=1e-6,
                init="latinhypercube",
                x0=current_phases,
                polish=False,
                workers=1,
                updating="immediate",
                seed=seed + pass_number * 100 + group_number
            )

            # Accept only an actual improvement.
            if result.fun < best_mse:
                best_solution[group] = np.mod(
                    result.x,
                    2 * np.pi
                )

                best_mse = float(result.fun)
                pass_improvements += 1

            print(
                f"Group {group_number}/{len(groups)} | "
                f"MSE: {best_mse:.6f}"
            )

            if best_mse <= udp.mse_thresh:
                break

        print(
            f"Pass result: MSE={best_mse:.6f}, "
            f"improvements={pass_improvements}"
        )

        if best_mse <= udp.mse_thresh or pass_improvements == 0:
            break

    objective_value, spacecraft, final_mse = udp.fitness(
        best_solution,
        postprocess=True
    )

    runtime = time.time() - start_time

    return {
        "best_solution": best_solution,
        "objective": float(objective_value[0]),
        "spacecraft": int(spacecraft),
        "mse": float(final_mse),
        "feasible": float(final_mse) <= udp.mse_thresh,
        "runtime": runtime,
        "initial_mse": initial_mse
    }
