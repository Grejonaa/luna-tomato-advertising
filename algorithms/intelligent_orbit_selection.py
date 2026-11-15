
import numpy as np
import time


def evaluate_mse(solution, udp):
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def get_candidate_orbits(udp, solution, rng, candidates_per_family):
    """
    Select unused orbits from all three orbit families.
    """

    num_dro = udp.num_dro
    num_lyap = udp.num_lyap
    num_axial = udp.num_axial

    num_orbits = udp.num_orbits
    selections = solution[:num_orbits]

    families = [
        (0, num_dro),
        (num_dro, num_dro + num_lyap),
        (num_dro + num_lyap, num_orbits)
    ]

    candidates = []

    for start, end in families:
        available = np.array([
            i for i in range(start, end)
            if selections[i] < udp.max_per_orbit
        ])

        if len(available) == 0:
            continue

        sample_size = min(candidates_per_family, len(available))

        selected = rng.choice(
            available,
            size=sample_size,
            replace=False
        )

        candidates.extend(selected.tolist())

    return candidates


def intelligent_orbit_selection(
    initial_solution,
    udp,
    max_additions=5,
    candidates_per_family=20,
    phase_grid=8,
    seed=42
):
    """
    Greedy search for useful additional spacecraft.

    Each iteration:
    1. Samples candidate orbits from all three families.
    2. Tests several phases for each candidate.
    3. Adds the spacecraft that gives the largest MSE reduction.
    4. Stops if no candidate improves the current solution.
    """

    start_time = time.time()
    rng = np.random.default_rng(seed)

    best_solution = np.array(initial_solution, dtype=float).copy()

    num_orbits = udp.num_orbits
    max_per_orbit = udp.max_per_orbit

    current_mse = evaluate_mse(best_solution, udp)

    phase_angles = np.linspace(
        0,
        2 * np.pi,
        phase_grid,
        endpoint=False
    )

    print("\nINTELLIGENT ORBIT SELECTION")
    print("--------------------------------")
    print(f"Initial MSE: {current_mse:.6f}")
    print(
        "Initial spacecraft:",
        int(np.sum(best_solution[:num_orbits]))
    )

    for addition in range(max_additions):

        candidates = get_candidate_orbits(
            udp,
            best_solution,
            rng,
            candidates_per_family
        )

        best_candidate_mse = current_mse
        best_candidate_solution = None
        best_candidate_orbit = None
        best_candidate_phase = None

        print(
            f"\nAddition {addition + 1}/{max_additions}: "
            f"Testing {len(candidates)} candidate orbits..."
        )

        for orbit_idx in candidates:

            current_count = int(best_solution[orbit_idx])

            if current_count >= max_per_orbit:
                continue

            # The next unused spacecraft slot in this orbit.
            phase_index = (
                num_orbits
                + orbit_idx * max_per_orbit
                + current_count
            )

            for phase in phase_angles:

                candidate = best_solution.copy()

                candidate[orbit_idx] = current_count + 1
                candidate[phase_index] = phase

                mse = evaluate_mse(candidate, udp)

                if mse < best_candidate_mse:
                    best_candidate_mse = mse
                    best_candidate_solution = candidate
                    best_candidate_orbit = orbit_idx
                    best_candidate_phase = phase

        # Do not add a spacecraft if it does not improve the signal.
        if best_candidate_solution is None:
            print("No improving candidate found. Stopping.")
            break

        best_solution = best_candidate_solution
        current_mse = best_candidate_mse

        spacecraft = int(np.sum(best_solution[:num_orbits]))

        print(
            f"Accepted orbit: {best_candidate_orbit} | "
            f"Phase: {best_candidate_phase:.4f} | "
            f"MSE: {current_mse:.6f} | "
            f"Spacecraft: {spacecraft}"
        )

        if current_mse <= udp.mse_thresh:
            print("Target MSE threshold reached!")
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
