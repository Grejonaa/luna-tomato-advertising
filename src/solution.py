
import numpy as np


def create_random_solution(udp, selected_orbits=30, rng=None):
    """
    Create a random candidate solution.

    The chromosome contains:
    - First 779 values: spacecraft counts per orbit.
    - Remaining 3895 values: phase angles in radians.
    """

    if rng is None:
        rng = np.random.default_rng()

    num_orbits = udp.num_orbits
    max_per_orbit = udp.max_per_orbit

    if not 1 <= selected_orbits <= num_orbits:
        raise ValueError(
            f"selected_orbits must be between 1 and {num_orbits}"
        )

    # Initialize the selection part.
    selections = np.zeros(num_orbits, dtype=int)

    # Randomly choose which orbits will contain spacecraft.
    chosen_orbits = rng.choice(
        num_orbits,
        size=selected_orbits,
        replace=False
    )

    # Initially place one spacecraft on each chosen orbit.
    selections[chosen_orbits] = 1

    # Generate random phase angles for all possible spacecraft.
    phases = rng.uniform(
        0,
        2 * np.pi,
        size=num_orbits * max_per_orbit
    )

    # Combine selections and phases into one chromosome.
    solution = np.concatenate([selections, phases])

    return solution


def validate_solution(solution, udp):
    """Check whether a solution has the correct structure and bounds."""

    expected_dimension = udp.num_orbits * (
        1 + udp.max_per_orbit
    )

    if len(solution) != expected_dimension:
        raise ValueError("Incorrect solution dimension.")

    selections = solution[:udp.num_orbits]

    if np.any(selections < 0) or np.any(
        selections > udp.max_per_orbit
    ):
        raise ValueError("Invalid spacecraft selection values.")

    if np.any(selections != np.round(selections)):
        raise ValueError("Selection values must be integers.")

    phases = solution[udp.num_orbits:]

    if np.any(phases < 0) or np.any(phases > 2 * np.pi):
        raise ValueError("Phase angles must be between 0 and 2*pi.")

    return True


def get_solution_info(solution, udp):
    """Return useful information about a candidate solution."""

    selections = solution[:udp.num_orbits]

    return {
        "selected_orbits": int(np.count_nonzero(selections)),
        "total_spacecraft": int(np.sum(selections)),
        "dimension": len(solution)
    }
