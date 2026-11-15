
import time
import numpy as np


def evaluate_mse(solution, udp):
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def active_phase_indices(solution, udp):
    n = udp.num_orbits
    max_per_orbit = udp.max_per_orbit
    indices = []

    for orbit in range(n):
        count = int(solution[orbit])

        for j in range(count):
            indices.append(
                n + orbit * max_per_orbit + j
            )

    return indices


def remove_spacecraft(solution, orbit, udp, slot):
    """Remove one spacecraft and compact the remaining phases."""
    candidate = solution.copy()

    n = udp.num_orbits
    max_per_orbit = udp.max_per_orbit
    count = int(candidate[orbit])

    if count == 0:
        return None

    start = n + orbit * max_per_orbit
    phases = candidate[start:start + count].copy()

    phases = np.delete(phases, slot)

    candidate[orbit] = count - 1
    candidate[start:start + count - 1] = phases

    return candidate


def add_spacecraft(solution, orbit, phase, udp):
    """Add one spacecraft to the next available slot."""
    candidate = solution.copy()

    n = udp.num_orbits
    count = int(candidate[orbit])

    if count >= udp.max_per_orbit:
        return None

    phase_index = n + orbit * udp.max_per_orbit + count

    candidate[orbit] = count + 1
    candidate[phase_index] = phase % (2 * np.pi)

    return candidate


def sample_available_orbits(solution, udp, rng, per_family):
    """Sample candidate orbits from DRO, Lyapunov and Axial."""
    n_dro = udp.num_dro
    n_lyap = udp.num_lyap
    n_total = udp.num_orbits

    families = [
        (0, n_dro),
        (n_dro, n_dro + n_lyap),
        (n_dro + n_lyap, n_total)
    ]

    candidates = []

    for start, end in families:
        available = [
            i for i in range(start, end)
            if int(solution[i]) < udp.max_per_orbit
        ]

        if available:
            amount = min(per_family, len(available))
            selected = rng.choice(
                available,
                size=amount,
                replace=False
            )
            candidates.extend(selected.tolist())

    return candidates


def hybrid_adaptive_optimization(
    initial_solution,
    udp,
    rounds=3,
    candidates_per_family=6,
    phase_grid=8,
    replacement_sources=6,
    seed=42
):
    """
    Structural search:
    - Add spacecraft
    - Remove spacecraft
    - Replace spacecraft
    - Improve phases of existing spacecraft

    The best solution is preserved throughout the search.
    """

    start_time = time.time()
    rng = np.random.default_rng(seed)

    best = np.array(initial_solution, dtype=float).copy()
    best_mse = evaluate_mse(best, udp)

    angles = np.linspace(
        0, 2 * np.pi, phase_grid, endpoint=False
    )

    print("\nHYBRID ADAPTIVE OPTIMIZATION")
    print("--------------------------------")
    print(f"Initial MSE: {best_mse:.6f}")
    print(f"Initial spacecraft: {int(best[:udp.num_orbits].sum())}")

    for round_number in range(rounds):
        round_best = best.copy()
        round_best_mse = best_mse
        improvements = 0

        print(f"\nROUND {round_number + 1}/{rounds}")

        # -------------------------------------------------
        # 1. ADDITION: test new spacecraft in unused slots.
        # -------------------------------------------------
        destinations = sample_available_orbits(
            best, udp, rng, candidates_per_family
        )

        for orbit in destinations:
            for phase in angles:
                candidate = add_spacecraft(
                    best, orbit, phase, udp
                )

                if candidate is None:
                    continue

                mse = evaluate_mse(candidate, udp)

                if mse < round_best_mse:
                    round_best = candidate
                    round_best_mse = mse

        print(f"After addition search: {round_best_mse:.6f}")

        # -------------------------------------------------
        # 2. REMOVAL: test removing existing spacecraft.
        # -------------------------------------------------
        current_indices = active_phase_indices(best, udp)

        for phase_index in current_indices:
            orbit = (
                phase_index - udp.num_orbits
            ) // udp.max_per_orbit

            slot = (
                phase_index - udp.num_orbits
            ) % udp.max_per_orbit

            candidate = remove_spacecraft(
                best, orbit, udp, slot
            )

            if candidate is None:
                continue

            mse = evaluate_mse(candidate, udp)

            if mse < round_best_mse:
                round_best = candidate
                round_best_mse = mse

        print(f"After removal search: {round_best_mse:.6f}")

        # -------------------------------------------------
        # 3. REPLACEMENT: remove one, add another.
        # -------------------------------------------------
        current_indices = active_phase_indices(best, udp)

        if current_indices and destinations:
            source_count = min(
                replacement_sources, len(current_indices)
            )

            selected_sources = rng.choice(
                current_indices,
                size=source_count,
                replace=False
            )

            for phase_index in selected_sources:
                source_orbit = (
                    phase_index - udp.num_orbits
                ) // udp.max_per_orbit

                source_slot = (
                    phase_index - udp.num_orbits
                ) % udp.max_per_orbit

                reduced = remove_spacecraft(
                    best, source_orbit, udp, source_slot
                )

                if reduced is None:
                    continue

                # Sample a smaller destination subset per source.
                destination_count = min(6, len(destinations))
                selected_destinations = rng.choice(
                    destinations,
                    size=destination_count,
                    replace=False
                )

                for target_orbit in selected_destinations:
                    for phase in angles:
                        candidate = add_spacecraft(
                            reduced, int(target_orbit), phase, udp
                        )

                        if candidate is None:
                            continue

                        mse = evaluate_mse(candidate, udp)

                        if mse < round_best_mse:
                            round_best = candidate
                            round_best_mse = mse

        print(f"After replacement search: {round_best_mse:.6f}")

        # -------------------------------------------------
        # 4. PHASE MUTATION: refine existing phases.
        # -------------------------------------------------
        indices = active_phase_indices(round_best, udp)

        if indices:
            for _ in range(60):
                candidate = round_best.copy()

                amount = min(
                    3, len(indices)
                )

                selected = rng.choice(
                    indices, size=amount, replace=False
                )

                for index in selected:
                    candidate[index] = rng.uniform(
                        0, 2 * np.pi
                    )

                mse = evaluate_mse(candidate, udp)

                if mse < round_best_mse:
                    round_best = candidate
                    round_best_mse = mse

        if round_best_mse < best_mse:
            improvements = 1
            best = round_best
            best_mse = round_best_mse

        spacecraft = int(best[:udp.num_orbits].sum())

        print(
            f"Round result | MSE: {best_mse:.6f} | "
            f"Spacecraft: {spacecraft} | "
            f"Improved: {improvements > 0}"
        )

        if best_mse <= udp.mse_thresh:
            print("Target threshold reached!")
            break

        if improvements == 0:
            print("No improvement in this round. Stopping.")
            break

    objective, spacecraft, final_mse = udp.fitness(
        best, postprocess=True
    )

    return {
        "best_solution": best,
        "objective": float(objective[0]),
        "spacecraft": int(spacecraft),
        "mse": float(final_mse),
        "feasible": float(final_mse) <= udp.mse_thresh,
        "runtime": time.time() - start_time
    }
