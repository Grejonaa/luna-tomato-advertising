
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem_setup import get_problem
from algorithms.differential_evolution import (
    optimize_with_differential_evolution
)
from visualization import save_solution_visualization


def main():

    print("LUNA TOMATO ADVERTISING")
    print("=======================")
    print("ADVANCED OPTIMIZATION")

    udp = get_problem()

    solution_path = (
    PROJECT_ROOT / "results" / "plots" / "best_solution.npy"
    )

    if not solution_path.exists():
        raise FileNotFoundError(
            f"Previous solution not found: {solution_path}"
        )

    initial_solution = np.load(solution_path)

    result = optimize_with_differential_evolution(
        initial_solution=initial_solution,
        udp=udp,
        group_size=6,
        max_passes=2,
        max_iterations=30,
        population_size=8,
        seed=42
    )

    print("\nFINAL DIFFERENTIAL EVOLUTION RESULT")
    print("--------------------------------------")
    print(f"Initial MSE: {result['initial_mse']:.6f}")
    print(f"Final MSE: {result['mse']:.6f}")
    print(f"Objective: {result['objective']:.6f}")
    print(f"Spacecraft: {result['spacecraft']}")
    print(f"Feasible: {result['feasible']}")
    print(f"Runtime: {result['runtime']:.2f} seconds")

    # Save the result only if it improves the previous solution.
    if result["mse"] < result["initial_mse"]:

        np.save(
            solution_path,
            result["best_solution"]
        )

        save_solution_visualization(
            udp,
            result["best_solution"],
            PROJECT_ROOT / "results" / "plots"
        )

        print("\nImproved solution saved successfully.")

    else:
        print("\nNo improvement. Previous solution preserved.")


if __name__ == "__main__":
    main()
