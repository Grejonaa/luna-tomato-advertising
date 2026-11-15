import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem_setup import get_problem
from visualization import save_solution_visualization


def load_reference(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = payload[0] if isinstance(payload, list) else payload
    return np.asarray(entry["decisionVector"], dtype=float)


def evaluate(name, vector, udp):
    objective, spacecraft, mse = udp.fitness(vector, postprocess=True)
    return {
        "name": name,
        "vector": vector,
        "objective": float(objective[0]),
        "spacecraft": int(spacecraft),
        "mse": float(mse),
        "feasible": float(mse) <= udp.mse_thresh,
    }


def main():
    udp = get_problem()
    result_dir = PROJECT_ROOT / "results"
    reference_dir = result_dir / "reference"
    plots_dir = result_dir / "plots"
    current_path = plots_dir / "best_solution.npy"

    if not current_path.exists():
        raise FileNotFoundError(f"Current solution not found: {current_path}")

    candidates = [
        evaluate("Current project solution", np.load(current_path), udp)
    ]

    for filename in ("best_N34_reference.json", "lowest_mse_N37_reference.json"):
        path = reference_dir / filename
        if path.exists():
            vector = load_reference(path)
            if vector.size != udp.num_orbits * (udp.max_per_orbit + 1):
                raise ValueError(f"Unexpected vector length in {path}: {vector.size}")
            candidates.append(evaluate(filename, vector, udp))
        else:
            print(f"Reference file not found, skipping: {path}")

    print("\nSOLUTION COMPARISON")
    print("-" * 72)
    for item in candidates:
        print(
            f"{item['name']}: objective={item['objective']:.8f} | "
            f"MSE={item['mse']:.12f} | spacecraft={item['spacecraft']} | "
            f"feasible={item['feasible']}"
        )

    # The official fitness objective prioritizes feasibility, then fleet size.
    winner = min(candidates, key=lambda item: item["objective"])
    print(f"\nBest by official objective: {winner['name']}")

    if winner["name"] == "Current project solution":
        print("Current solution remains the best. No files changed.")
        return

    backup_path = plots_dir / "best_solution_before_reference.npy"
    if not backup_path.exists():
        shutil.copy2(current_path, backup_path)
        print(f"Previous solution backed up to: {backup_path}")

    np.save(current_path, winner["vector"])
    save_solution_visualization(udp, winner["vector"], plots_dir)
    print(f"Updated best solution: {current_path}")


if __name__ == "__main__":
    main()
