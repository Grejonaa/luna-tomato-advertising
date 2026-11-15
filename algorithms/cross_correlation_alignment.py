
"""Cross-correlation-guided phase alignment using the official ESA evaluator.

The method keeps spacecraft selections fixed and searches for better phases.
Candidate phases are ranked by their correlation with the residual signal.
The official UDP fitness function validates every shortlisted candidate.
"""

from __future__ import annotations

import time

import numpy as np


TAU = 2.0 * np.pi


def _phase_index(udp, orbit_idx, slot):
    """Return the chromosome index of a spacecraft phase."""
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
        local_orbit = orbit_idx - udp.num_dro - udp.num_lyap

    return (
        udp.num_orbits
        + family_offset
        + local_orbit * udp.max_per_orbit
        + slot
    )


def _single_spacecraft_signal(udp, orbit_idx, phase):
    """Build one spacecraft contribution using the official UDP model."""
    selections = np.zeros(udp.num_orbits, dtype=int)
    selections[orbit_idx] = 1

    phases = np.zeros(
        udp.num_orbits * udp.max_per_orbit,
        dtype=float
    )

    # The isolated spacecraft is placed in slot zero.
    phase_idx = _phase_index(udp, orbit_idx, 0)
    phases[phase_idx - udp.num_orbits] = phase

    matrix = udp._build_occultation_matrix(selections, phases)

    # DRO has one occultation; Lyapunov and Axial have two.
    columns = 1 if orbit_idx < udp.num_dro else 2

    return matrix[:, :columns].sum(axis=1)


def _official_mse(udp, solution):
    """Evaluate candidates with the official ESA fitness function."""
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def cross_correlation_phase_alignment(
    initial_solution,
    udp,
    passes=3,
    grid_points=48,
    shortlist=4,
    seed=42,
):
    """
    Improve active phases using residual-guided cross-correlation.

    For each spacecraft:
      1. Remove its current contribution from the reconstructed signal.
      2. Calculate the residual against the target signal.
      3. Generate phase-dependent signal templates.
      4. Rank templates by their correlation with the residual.
      5. Validate shortlisted phases with udp.fitness().
    """
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

    current_mse = _official_mse(udp, current)
    best_mse = current_mse

    angles = np.linspace(
        0.0, TAU, grid_points, endpoint=False
    )

    print("\nCROSS-CORRELATION PHASE ALIGNMENT")
    print("--------------------------------")
    print(f"Spacecraft: {int(np.sum(selections))}")
    print(f"Initial official MSE: {current_mse:.12f}")
    print(f"Phase grid: {grid_points}")
    print(f"Passes: {passes}")

    active = [
        (orbit_idx, slot)
        for orbit_idx, count in enumerate(selections)
        for slot in range(int(count))
    ]

    for pass_number in range(1, passes + 1):
        rng.shuffle(active)
        improvements = 0

        for orbit_idx, slot in active:
            phase_idx = _phase_index(
                udp, orbit_idx, slot
            )

            current_phase = current[phase_idx] % TAU

            # Current contribution of this spacecraft.
            current_template = _single_spacecraft_signal(
                udp, orbit_idx, current_phase
            )

            # Reconstruct the complete raw signal using the official model.
            _, phases = udp._decode_chromosome(current)
            matrix = udp._build_occultation_matrix(
                selections, phases
            )
            raw_total = matrix.sum(axis=1)

            # Approximate the signal without this spacecraft.
            signal_without = np.clip(
                raw_total - current_template, 0.0, 1.0
            )

            residual = udp.target_signal - signal_without

            # Rank phase templates by residual correlation.
            ranked = []

            for angle in angles:
                template = _single_spacecraft_signal(
                    udp, orbit_idx, angle
                )

                correlation = float(
                    np.dot(residual, template)
                )

                ranked.append((correlation, angle))

            ranked.sort(
                key=lambda item: item[0],
                reverse=True
            )

            # Always include the current phase as a candidate.
            candidate_angles = [
                angle for _, angle in ranked[:shortlist]
            ]
            candidate_angles.append(current_phase)

            local_best = current.copy()
            local_best_mse = current_mse

            for angle in candidate_angles:
                candidate = current.copy()
                candidate[phase_idx] = angle % TAU

                candidate_mse = _official_mse(
                    udp, candidate
                )

                if candidate_mse < local_best_mse:
                    local_best = candidate
                    local_best_mse = candidate_mse

            if local_best_mse < current_mse:
                current = local_best
                current_mse = local_best_mse
                improvements += 1

                if current_mse < best_mse:
                    best = current.copy()
                    best_mse = current_mse

        print(
            f"Pass {pass_number}/{passes} | "
            f"MSE: {current_mse:.12f} | "
            f"Best: {best_mse:.12f} | "
            f"Improvements: {improvements}"
        )

        if improvements == 0:
            print("No further improvement in this pass.")
            break

    objective, spacecraft, final_mse = udp.fitness(
        best, postprocess=True
    )

    result = {
        "best_solution": best,
        "objective": float(objective[0]),
        "spacecraft": int(spacecraft),
        "mse": float(final_mse),
        "feasible": bool(final_mse <= udp.mse_thresh),
        "runtime": time.time() - start_time,
    }

    print("\nFINAL RESULT")
    print("------------")
    print(f"Spacecraft: {result['spacecraft']}")
    print(f"MSE: {result['mse']:.12f}")
    print(f"Feasible: {result['feasible']}")
    print(f"Runtime: {result['runtime']:.2f} seconds")

    return result
