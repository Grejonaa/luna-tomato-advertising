
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem_setup import get_problem
from objective import print_evaluation

from visualization import save_solution_visualization
from algorithms.random_search import random_search
from algorithms.hill_climbing import hill_climbing
from algorithms.simulated_annealing import simulated_annealing
from algorithms.phase_optimization import optimize_phases
from algorithms.intelligent_orbit_selection import intelligent_orbit_selection

def display_result(name, result):

    print(f"\n{name} RESULTS")
    print("=" * 35)

    print_evaluation({
        "objective": result["objective"],
        "spacecraft": result["spacecraft"],
        "mse": result["mse"],
        "feasible": result["feasible"]
    })

    print(f"Runtime: {result['runtime']:.2f} seconds")


def main():

    udp = get_problem()

    print("LUNA TOMATO ADVERTISING")
    print("=======================\n")

    # 1. Random Search
    print("1. RUNNING RANDOM SEARCH")

    random_result = random_search(
        udp,
        iterations=50,
        max_satellites=100,
        seed=42
    )

    display_result("RANDOM SEARCH", random_result)

    # 2. Hill Climbing
    print("\n2. RUNNING HILL CLIMBING")

    hill_result = hill_climbing(
        udp,
        initial_solution=random_result["best_solution"],
        iterations=100,
        seed=42
    )

    display_result("HILL CLIMBING", hill_result)

    # 3. Simulated Annealing
    print("\n3. RUNNING SIMULATED ANNEALING")

    annealing_result = simulated_annealing(
        udp,
        initial_solution=random_result["best_solution"],
        iterations=500,
        initial_temperature=10.0,
        cooling_rate=0.995,
        seed=42
    )

    display_result("SIMULATED ANNEALING", annealing_result)

    # PHASE OPTIMIZATION
    phase_result = optimize_phases(
    initial_solution=annealing_result["best_solution"],
    udp=udp,
    max_passes=2,
    grid_points=8,
    max_iterations=20,
    seed=42
    )

    print("\nPHASE OPTIMIZATION RESULT")
    print("------------------------------")
    print(f"Objective: {phase_result['objective']:.6f}")
    print(f"Spacecraft: {phase_result['spacecraft']}")
    print(f"MSE: {phase_result['mse']:.6f}")
    print(f"Feasible: {phase_result['feasible']}")
    print(f"Runtime: {phase_result['runtime']:.2f} seconds")

    # INTELLIGENT ORBIT SELECTION
    orbit_result = intelligent_orbit_selection(
    initial_solution=phase_result["best_solution"],
    udp=udp,
    max_additions=5,
    candidates_per_family=20,
    phase_grid=8,
    seed=42
    )

    print("\nINTELLIGENT ORBIT SELECTION RESULT")
    print("--------------------------------")
    print(f"Objective: {orbit_result['objective']:.6f}")
    print(f"Spacecraft: {orbit_result['spacecraft']}")
    print(f"MSE: {orbit_result['mse']:.6f}")
    print(f"Feasible: {orbit_result['feasible']}")
    print(f"Runtime: {orbit_result['runtime']:.2f} seconds")

    # FINAL PHASE RE-OPTIMIZATION
    final_phase_result = optimize_phases(
    initial_solution=orbit_result["best_solution"],
    udp=udp,
    max_passes=3,
    grid_points=12,
    max_iterations=25,
    seed=42
    )

    print("\nFINAL PHASE RE-OPTIMIZATION")
    print("--------------------------------")
    print(f"Objective: {final_phase_result['objective']:.6f}")
    print(f"Spacecraft: {final_phase_result['spacecraft']}")
    print(f"MSE: {final_phase_result['mse']:.6f}")
    print(f"Feasible: {final_phase_result['feasible']}")
    print(f"Runtime: {final_phase_result['runtime']:.2f} seconds")
    # Final comparison
    print("\nFINAL PERFORMANCE COMPARISON")
    print("=" * 65)

    print(
        f"{'Algorithm':<25}"
        f"{'Objective':<15}"
        f"{'MSE':<12}"
        f"{'Satellites':<12}"
        f"{'Time (s)':<10}"
    )
    

    results = [
        ("Random Search", random_result),
        ("Hill Climbing", hill_result),
        ("Simulated Annealing", annealing_result),
        ("Phase Optimization", phase_result),
        ("Intelligent Orbit Selection", orbit_result),
        ("Final Phase Re-Optimization", final_phase_result)
    ]

    for name, result in results:
        print(
            f"{name:<25}"
            f"{result['objective']:<15.4f}"
            f"{result['mse']:<12.6f}"
            f"{result['spacecraft']:<12}"
            f"{result['runtime']:<10.2f}"
        )

        # Save and visualize the best solution.
    print("\nSAVING BEST SOLUTION")
    print("====================")

    save_solution_visualization(
    udp,
    final_phase_result["best_solution"],
    PROJECT_ROOT / "results" / "plots"
    )
if __name__ == "__main__":
    main()
