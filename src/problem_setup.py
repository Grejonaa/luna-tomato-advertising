
"""
Initial setup for the Luna Tomato Advertising optimization problem.
"""

import numpy as np
from celestial_morse_code import udp


def get_problem():
    """Return the official ESA problem instance."""
    return udp


def get_problem_info():
    """Return basic information about the optimization problem."""
    return {
        "num_dro": udp.num_dro,
        "num_lyap": udp.num_lyap,
        "num_axial": udp.num_axial,
        "num_orbits": udp.num_orbits,
        "max_per_orbit": udp.max_per_orbit,
        "mse_threshold": udp.mse_thresh,
        "dimension": udp.num_orbits * (1 + udp.max_per_orbit),
    }


if __name__ == "__main__":
    info = get_problem_info()

    print("LUNA TOMATO ADVERTISING - PROBLEM SETUP")
    print("---------------------------------------")

    for key, value in info.items():
        print(f"{key}: {value}")

    # Test a solution with no selected spacecraft
    x = np.zeros(info["dimension"])

    objective, num_selected, mse = udp.fitness(
        x,
        postprocess=True
    )

    print("\nBASELINE TEST")
    print("---------------------------------------")
    print("Objective:", objective)
    print("Selected spacecraft:", num_selected)
    print("MSE:", mse)
