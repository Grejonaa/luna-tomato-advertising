
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_solution_visualization(udp, solution, output_dir):
    """Save the target and reconstructed Morse signals."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate the solution.
    objective, spacecraft, mse = udp.fitness(
        solution,
        postprocess=True
    )

    # Save the official ESA signal comparison.
    udp.plot_signal(solution)

    figure = plt.gcf()

    figure.savefig(
        output_dir / "best_solution_signal.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    # Save the chromosome for later use.
    np.save(
        output_dir / "best_solution.npy",
        solution
    )

    print("\nVISUALIZATION SAVED")
    print("-------------------")
    print("Image: results/plots/best_solution_signal.png")
    print("Solution: results/best_solution.npy")
    print("Objective:", float(objective[0]))
    print("Spacecraft:", int(spacecraft))
    print("MSE:", float(mse))
