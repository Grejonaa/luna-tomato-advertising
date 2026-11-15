
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem_setup import get_problem
from algorithms.hybrid_adaptive_optimization import (
    hybrid_adaptive_optimization
)
from visualization import save_solution_visualization


def main():
    print("LUNA TOMATO ADVERTISING")
    print("=======================")
    print("HYBRID ADAPTIVE OPTIMIZATION")

    udp = get_problem()

    solution_path = (
        PROJECT_ROOT / "results" / "plots" / "best_solution.npy"
    )

    if not solution_path.exists():
        raise FileNotFoundError(
            f"Solution not found: {solution_path}"
        )

    initial_solution = np.load(solution_path)
    initial_mse = float(
        udp.fitness(initial_solution, postprocess=True)[2]
    )

    result = hybrid_adaptive_optimization(
        initial_solution=initial_solution,
        udp=udp,
        rounds=3,
        candidates_per_family=6,
        phase_grid=8,
        replacement_sources=6,
        seed=42
    )

    print("\nFINAL HYBRID RESULT")
    print("===================")
    print(f"Previous MSE: {initial_mse:.6f}")
    print(f"New MSE:      {result['mse']:.6f}")
    print(f"Objective:    {result['objective']:.6f}")
    print(f"Spacecraft:   {result['spacecraft']}")
    print(f"Feasible:     {result['feasible']}")
    print(f"Runtime:      {result['runtime']:.2f} seconds")

    if result["mse"] < initial_mse:
        np.save(solution_path, result["best_solution"])

        save_solution_visualization(
            udp,
            result["best_solution"],
            PROJECT_ROOT / "results" / "plots"
        )

        print("\nNew best solution saved successfully.")
    else:
        print("\nNo improvement. Previous solution preserved.")


if __name__ == "__main__":
    main()
