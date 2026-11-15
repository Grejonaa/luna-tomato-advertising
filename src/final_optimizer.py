"""Minimize the fleet size while satisfying the official ESA MSE constraint.

Run from the project root:
    python src/final_optimizer.py

A shorter smoke run can be started with:
    python src/final_optimizer.py --iterations-per-attempt 300 --candidates 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem_setup import get_problem
from algorithms.variable_budget_optimization import variable_budget_optimization
from visualization import save_solution_visualization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/plots/best_solution.npy",
        help="Starting feasible solution, relative to project root.",
    )
    parser.add_argument("--min-spacecraft", type=int, default=1)
    parser.add_argument("--iterations-per-attempt", type=int, default=3000)
    parser.add_argument("--candidates", type=int, default=3,
                        help="Number of best single-spacecraft removals to test per fleet size.")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refine-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    if args.iterations_per_attempt < 1:
        parser.error("--iterations-per-attempt must be positive")
    if args.candidates < 1 or args.restarts < 1:
        parser.error("--candidates and --restarts must be positive")

    udp = get_problem()
    input_path = (PROJECT_ROOT / args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Starting solution not found: {input_path}")

    initial = np.load(input_path)
    initial_objective, initial_count, initial_mse = udp.fitness(
        initial, postprocess=True
    )

    print("LUNA TOMATO ADVERTISING")
    print("VARIABLE-BUDGET OPTIMIZATION")
    print("============================")
    print(f"Input: {input_path}")
    print(f"Initial spacecraft: {int(initial_count)}")
    print(f"Initial MSE: {float(initial_mse):.12f}")
    print(f"Initial objective: {float(initial_objective[0]):.8f}")

    result = variable_budget_optimization(
        udp=udp,
        initial_solution=initial,
        min_spacecraft=args.min_spacecraft,
        candidates_per_reduction=args.candidates,
        restarts_per_candidate=args.restarts,
        iterations_per_attempt=args.iterations_per_attempt,
        seed=args.seed,
        refine_every=args.refine_every,
        log_every=args.log_every,
    )

    output_dir = PROJECT_ROOT / "results" / "variable_budget"
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_path = output_dir / "best_solution.npy"
    np.save(solution_path, result.best_solution)

    report = {
        "algorithm": "Variable-Budget Greedy Pruning + Adaptive Simulated Annealing",
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "objective": result.objective,
        "spacecraft": result.spacecraft,
        "mse": result.mse,
        "mse_threshold": float(udp.mse_thresh),
        "feasible": result.feasible,
        "runtime_seconds": result.runtime,
        "accepted_reductions": result.reductions_accepted,
        "attempts": result.attempts,
        "iterations_per_attempt": args.iterations_per_attempt,
        "candidates_per_reduction": args.candidates,
        "restarts_per_candidate": args.restarts,
        "seed": args.seed,
    }
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_solution_visualization(udp, result.best_solution, output_dir)

    print("\nFINAL VARIABLE-BUDGET RESULT")
    print("----------------------------")
    print(f"Spacecraft: {result.spacecraft}")
    print(f"MSE: {result.mse:.12f}")
    print(f"Objective: {result.objective:.8f}")
    print(f"Feasible: {result.feasible}")
    print(f"Accepted reductions: {result.reductions_accepted}")
    print(f"Attempts: {result.attempts}")
    print(f"Runtime: {result.runtime:.2f} seconds")
    print(f"Saved solution: {solution_path}")
    print(f"Saved report: {report_path}")
    print("Original reference solution was not overwritten.")


if __name__ == "__main__":
    main()
