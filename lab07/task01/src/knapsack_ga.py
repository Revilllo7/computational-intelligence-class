"""Task 01: Binary-chromosome knapsack optimization with PyGAD."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygad

from common.ga_utils import (
    decode_selected_items,
    summarize_totals,
    write_csv_rows,
    write_json,
)
from task01.data.items import items

# Task constants
CAPACITY = 25.0
TARGET_VALUE = 1630.0
ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"

# PyGAD defaults
NUM_GENERATIONS = 200
SOLUTIONS_PER_POP = 60
NUM_PARENTS_MATING = 20
PARENT_SELECTION_TYPE = "sss"
MUTATION_TYPE = "random"
MUTATION_PERCENT_GENES = 10
CROSSOVER_TYPE = "single_point"
KEEP_PARENTS = 2

# Transform source-of-truth dictionary into aligned arrays for fast fitness.
ITEM_IDS = list(items.keys())
ITEM_NAMES = [items[item_id]["name"] for item_id in ITEM_IDS]
ITEM_VALUES = np.array([float(items[item_id]["value"]) for item_id in ITEM_IDS], dtype=float)
ITEM_WEIGHTS = np.array([float(items[item_id]["weight"]) for item_id in ITEM_IDS], dtype=float)
NUM_GENES = len(ITEM_IDS)


def evaluate_solution(solution: np.ndarray | list[int] | tuple[int, ...]) -> tuple[float, float]:
    """Return value and weight for a binary solution."""
    genes = np.asarray(solution, dtype=float)
    total_value = float(np.dot(genes, ITEM_VALUES))
    total_weight = float(np.dot(genes, ITEM_WEIGHTS))
    return total_value, total_weight


def knapsack_fitness(_: pygad.GA, solution: np.ndarray, __: int) -> float:
    """Feasible solutions are rewarded by value; overweight solutions are strongly penalized."""
    total_value, total_weight = evaluate_solution(solution)
    if total_weight <= CAPACITY:
        return total_value

    overweight = total_weight - CAPACITY
    penalty = 1000.0 * overweight
    return total_value - penalty


def on_generation_stop_if_target(ga_instance: pygad.GA) -> str | None:
    """Stop evolution early once a feasible target solution is found."""
    best_solution, _, _ = ga_instance.best_solution()
    best_value, best_weight = evaluate_solution(best_solution)
    if best_weight <= CAPACITY and best_value >= TARGET_VALUE:
        return "stop"
    return None


def create_ga_instance(random_seed: int | None = None) -> pygad.GA:
    """Return a fresh GA instance for one independent run."""
    return pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        sol_per_pop=SOLUTIONS_PER_POP,
        num_genes=NUM_GENES,
        fitness_func=knapsack_fitness,
        init_range_low=0,
        init_range_high=2,
        gene_type=int,
        gene_space=[0, 1],
        keep_parents=KEEP_PARENTS,
        parent_selection_type=PARENT_SELECTION_TYPE,
        crossover_type=CROSSOVER_TYPE,
        mutation_type=MUTATION_TYPE,
        mutation_percent_genes=MUTATION_PERCENT_GENES,
        random_seed=random_seed,
        on_generation=on_generation_stop_if_target,
        suppress_warnings=True,
    )


def save_fitness_curve(ga_instance: pygad.GA, output_path: Path) -> None:
    """Save optimization curve for the run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = ga_instance.best_solutions_fitness

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(history, color="tab:blue", linewidth=2)
    ax.set_title("Knapsack GA Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_single(random_seed: int | None = None) -> dict[str, Any]:
    """Execute one run and persist single-run artifacts."""
    ga_instance = create_ga_instance(random_seed=random_seed)

    started_at = time.perf_counter()
    ga_instance.run()
    runtime_seconds = time.perf_counter() - started_at

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_value, best_weight = evaluate_solution(best_solution)
    selected_rows = decode_selected_items(
        chromosome=[int(gene) for gene in best_solution],
        item_ids=ITEM_IDS,
        item_names=ITEM_NAMES,
        item_values=ITEM_VALUES.tolist(),
        item_weights=ITEM_WEIGHTS.tolist(),
    )
    totals = summarize_totals(selected_rows)
    success = best_weight <= CAPACITY and best_value >= TARGET_VALUE

    save_fitness_curve(ga_instance, ANALYSIS_DIR / "single_run_fitness.png")

    single_summary = {
        "capacity": CAPACITY,
        "target_value": TARGET_VALUE,
        "runtime_seconds": runtime_seconds,
        "best_fitness": float(best_fitness),
        "best_total_value": float(best_value),
        "best_total_weight": float(best_weight),
        "success": bool(success),
        "selected_item_ids": [row["id"] for row in selected_rows],
    }
    write_json(ANALYSIS_DIR / "single_run_summary.json", single_summary)
    write_csv_rows(
        ANALYSIS_DIR / "single_run_selected_items.csv",
        selected_rows,
        fieldnames=["id", "name", "value", "weight"],
    )

    print("Single run best solution:")
    for row in selected_rows:
        print(
            f"- {row['id']} ({row['name']}): value={row['value']:.0f}, weight={row['weight']:.1f}"
        )
    print(f"Total value: {totals['total_value']:.0f}")
    print(f"Total weight: {totals['total_weight']:.1f}")
    print(f"Runtime (s): {runtime_seconds:.6f}")

    return {
        "best_value": float(best_value),
        "best_weight": float(best_weight),
        "runtime_seconds": runtime_seconds,
        "success": bool(success),
    }


def run_experiments(num_runs: int = 10) -> dict[str, Any]:
    """Run exactly 10 independent GA runs and report success statistics."""
    run_rows: list[dict[str, Any]] = []

    for run_idx in range(1, num_runs + 1):
        run_seed = random.randint(1, 10_000_000)
        ga_instance = create_ga_instance(random_seed=run_seed)

        started_at = time.perf_counter()
        ga_instance.run()
        runtime_seconds = time.perf_counter() - started_at

        best_solution, _, _ = ga_instance.best_solution()
        best_value, best_weight = evaluate_solution(best_solution)
        success = best_weight <= CAPACITY and best_value >= TARGET_VALUE

        run_rows.append(
            {
                "run_index": run_idx,
                "best_value": float(best_value),
                "best_weight": float(best_weight),
                "success": bool(success),
                "runtime_seconds": runtime_seconds,
            }
        )

    write_csv_rows(
        ANALYSIS_DIR / "ten_run_results.csv",
        run_rows,
        fieldnames=[
            "run_index",
            "best_value",
            "best_weight",
            "success",
            "runtime_seconds",
        ],
    )

    successes = [row for row in run_rows if row["success"]]
    success_count = len(successes)
    success_percentage = (success_count / num_runs) * 100.0
    avg_success_runtime = (
        float(sum(row["runtime_seconds"] for row in successes) / success_count)
        if success_count > 0
        else 0.0
    )

    print("\n10-run experiment summary:")
    print(f"Runs reaching value {TARGET_VALUE:.0f}: {success_count}/{num_runs}")
    print(f"Success percentage: {success_percentage:.2f}%")
    print(f"Average successful runtime (s): {avg_success_runtime:.6f}")

    return {
        "success_count": success_count,
        "num_runs": num_runs,
        "success_percentage": success_percentage,
        "avg_success_runtime": avg_success_runtime,
    }


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_single()
    run_experiments(num_runs=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
