
from solution import validate_solution


def evaluate_solution(solution, udp):
    """
    Evaluate a candidate solution using the official ESA fitness function.
    """

    validate_solution(solution, udp)

    objective, num_selected, mse = udp.fitness(
        solution,
        postprocess=True
    )

    objective_value = float(objective[0])

    return {
        "objective": objective_value,
        "spacecraft": int(num_selected),
        "mse": float(mse),
        "feasible": float(mse) <= udp.mse_thresh
    }


def print_evaluation(result):
    """Display the evaluation results."""

    print("\nSOLUTION EVALUATION")
    print("---------------------------")
    print(f"Objective: {result['objective']:.6f}")
    print(f"Spacecraft: {result['spacecraft']}")
    print(f"MSE: {result['mse']:.6f}")
    print(f"Feasible: {result['feasible']}")
