"""Task 01: Binary-chromosome knapsack optimization with PyGAD."""

from __future__ import annotations

import argparse
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
from common.gif_utils import figure_to_rgb_array, save_gif_from_arrays
from task01.data.items import items
from task01.src.config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

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


def penalized_fitness(
    solution: np.ndarray | list[int] | tuple[int, ...],
    *,
    capacity: float,
) -> float:
    """Return value-based fitness with a strong linear penalty for overweight solutions."""
    total_value, total_weight = evaluate_solution(solution)
    if total_weight <= capacity:
        return total_value

    overweight = total_weight - capacity
    penalty = 1000.0 * overweight
    return total_value - penalty


def make_knapsack_fitness(capacity: float):
    """Create PyGAD-compatible fitness callback bound to configured capacity."""

    def _fitness(_: pygad.GA, solution: np.ndarray, __: int) -> float:
        return penalized_fitness(solution, capacity=capacity)

    return _fitness


def _make_generation_callback(
    config: LoadedExperimentConfig,
    generation_fitness_history: list[float] | None,
    generation_population_snapshots: list[np.ndarray] | None,
):
    """Capture optional per-generation state and stop when target is reached."""

    def _on_generation(ga_instance: pygad.GA) -> str | None:
        if generation_fitness_history is not None and ga_instance.best_solutions_fitness:
            generation_fitness_history.append(float(ga_instance.best_solutions_fitness[-1]))

        if generation_population_snapshots is not None:
            generation_population_snapshots.append(
                np.array(ga_instance.population, dtype=int, copy=True)
            )

        best_solution, _, _ = ga_instance.best_solution()
        best_value, best_weight = evaluate_solution(best_solution)
        if (
            best_weight <= config.experiment.problem.capacity
            and best_value >= config.experiment.problem.target_value
        ):
            return "stop"
        return None

    return _on_generation


def create_ga_instance(
    config: LoadedExperimentConfig,
    random_seed: int | None = None,
    *,
    generation_fitness_history: list[float] | None = None,
    generation_population_snapshots: list[np.ndarray] | None = None,
) -> pygad.GA:
    """Return a fresh GA instance for one independent run."""
    mutation_num_genes = config.resolve_mutation_num_genes(NUM_GENES)
    ga = pygad.GA(
        num_generations=config.experiment.ga.num_generations,
        num_parents_mating=config.experiment.ga.num_parents_mating,
        sol_per_pop=config.experiment.ga.solutions_per_population,
        num_genes=NUM_GENES,
        fitness_func=make_knapsack_fitness(config.experiment.problem.capacity),
        init_range_low=0,
        init_range_high=2,
        gene_type=int,
        gene_space=[0, 1],
        keep_parents=config.experiment.ga.keep_parents,
        parent_selection_type=config.experiment.ga.parent_selection_type,
        crossover_type=config.experiment.ga.crossover_type,
        mutation_type=config.experiment.ga.mutation_type,
        mutation_num_genes=mutation_num_genes,
        random_seed=random_seed,
        on_generation=_make_generation_callback(
            config,
            generation_fitness_history,
            generation_population_snapshots,
        ),
        suppress_warnings=True,
    )

    if generation_population_snapshots is not None:
        generation_population_snapshots.append(np.array(ga.population, dtype=int, copy=True))

    return ga


def save_single_run_fitness_curve(
    history: list[float],
    output_path: Path,
    figure_dpi: int,
    target_value: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=figure_dpi)
    ax.plot(
        generations,
        history,
        color="tab:blue",
        linewidth=2,
        marker="o",
        markersize=7,
    )

    ax.axhline(y=target_value, color="red", linestyle=":", linewidth=2)
    ax.set_xlim(0.5, max(2, len(history)) + 0.5)
    ax.set_ylim(
        max(0, min(history) - 50),
        max(max(history), target_value) + 50,
    )
    ax.set_title("Knapsack GA Convergence (Single Run)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_multi_run_overlay(
    run_histories: list[list[float]],
    output_path: Path,
    figure_dpi: int,
    target_value: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=figure_dpi)

    for run_idx, history in enumerate(run_histories, start=1):
        generations = np.arange(1, len(history) + 1)
        ax.plot(generations, history, linewidth=1.2, alpha=0.8, label=f"run {run_idx}")

    ax.axhline(y=target_value, color="red", linestyle=":", linewidth=2)

    ax.set_title("Best Fitness by Generation (All Runs)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_aggregate_trend(
    run_histories: list[list[float]],
    output_path: Path,
    figure_dpi: int,
    target_value: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_generations = max(len(history) for history in run_histories)
    fitness_matrix = np.full((len(run_histories), max_generations), np.nan, dtype=float)

    for row_idx, history in enumerate(run_histories):
        fitness_matrix[row_idx, : len(history)] = history

    generations = np.arange(1, max_generations + 1)
    mean_fitness = np.nanmean(fitness_matrix, axis=0)
    min_fitness = np.nanmin(fitness_matrix, axis=0)
    max_fitness = np.nanmax(fitness_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=figure_dpi)

    ax.fill_between(generations, min_fitness, max_fitness, color="tab:cyan", alpha=0.25)
    ax.plot(generations, mean_fitness, color="tab:blue", linewidth=2, label="mean")
    ax.plot(generations, min_fitness, color="tab:green", linewidth=1, alpha=0.8, label="min")
    ax.plot(generations, max_fitness, color="tab:red", linewidth=1, alpha=0.8, label="max")

    ax.axhline(y=target_value, color="red", linestyle=":", linewidth=2)

    ax.set_title("Aggregate Best Fitness Trend (Mean with Min/Max Band)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _build_fitness_animation_frames(
    history: list[float], figure_dpi: int, target_value: float
) -> list[np.ndarray]:
    """Build animation frames of convergence line growth over generations."""
    frames: list[np.ndarray] = []
    full_generations = np.arange(1, len(history) + 1)
    if history:
        y_min = min(history)
        y_max = max(history)
    else:
        y_min = 0
        y_max = target_value

    # ensure useful scale using YAML target
    y_max = max(y_max, target_value)
    span = max(target_value * 0.1, y_max - y_min, 50)

    for frame_end in range(1, len(history) + 1):
        shown_generations = full_generations[:frame_end]
        shown_fitness = history[:frame_end]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=figure_dpi)
        ax.plot(shown_generations, shown_fitness, color="tab:blue", linewidth=2)
        ax.scatter(shown_generations[-1], shown_fitness[-1], color="tab:red", s=18)
        ax.set_xlim(1, max(2, len(history)))
        ax.set_ylim(max(0, y_min - 0.1 * span), y_max + 0.1 * span)

        ax.set_title("Fitness Over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()

        frames.append(figure_to_rgb_array(fig))
        plt.close(fig)

    return frames


def _build_population_animation_frames(
    population_snapshots: list[np.ndarray],
    figure_dpi: int,
) -> list[np.ndarray]:
    """Build animation frames for per-generation chromosome heatmaps."""
    frames: list[np.ndarray] = []
    total_generations = max(1, len(population_snapshots) - 1)

    for snapshot_idx, snapshot in enumerate(population_snapshots):
        generation_label = max(0, min(total_generations, snapshot_idx))

        fig, ax = plt.subplots(figsize=(8, 4), dpi=figure_dpi)
        image = ax.imshow(snapshot, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Population Heatmap (Generation {generation_label})")
        ax.set_xlabel("Gene Index")
        ax.set_ylabel("Population Member")
        fig.colorbar(image, ax=ax, ticks=[0, 1], label="Gene Value")
        fig.tight_layout()

        frames.append(figure_to_rgb_array(fig))
        plt.close(fig)

    return frames


def run_single(config: LoadedExperimentConfig) -> dict[str, Any]:
    """Execute one run and persist single-run artifacts."""
    output_dir = config.output_dir
    plots_dir = output_dir / "plots"
    gifs_dir = output_dir / "gifs"

    generation_fitness_history: list[float] = []
    generation_population_snapshots: list[np.ndarray] = []
    ga_instance = create_ga_instance(
        config=config,
        random_seed=config.experiment.runs.single_run_seed,
        generation_fitness_history=generation_fitness_history,
        generation_population_snapshots=generation_population_snapshots,
    )

    started_at = time.perf_counter()
    ga_instance.run()
    runtime_seconds = time.perf_counter() - started_at

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_value, best_weight = evaluate_solution(best_solution)
    success = (
        best_weight <= config.experiment.problem.capacity
        and best_value >= config.experiment.problem.target_value
    )

    final_history = [float(value) for value in ga_instance.best_solutions_fitness]
    if not generation_fitness_history:
        generation_fitness_history = final_history
    if not generation_fitness_history:
        generation_fitness_history = [float(best_fitness)]

    selected_rows = decode_selected_items(
        chromosome=[int(gene) for gene in best_solution],
        item_ids=ITEM_IDS,
        item_names=ITEM_NAMES,
        item_values=ITEM_VALUES.tolist(),
        item_weights=ITEM_WEIGHTS.tolist(),
    )
    totals = summarize_totals(selected_rows)
    mutation_num_genes = config.resolve_mutation_num_genes(NUM_GENES)

    if config.experiment.plotting.enabled and config.experiment.plotting.single_run_curve:
        save_single_run_fitness_curve(
            generation_fitness_history,
            plots_dir / "single_run_fitness.png",
            config.experiment.plotting.figure_dpi,
            config.experiment.problem.target_value,
        )

    if config.experiment.gif.enabled and generation_fitness_history:
        if config.experiment.gif.fitness_animation:
            fitness_frames = _build_fitness_animation_frames(
                generation_fitness_history,
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_value,
            )
            save_gif_from_arrays(
                fitness_frames,
                gifs_dir / "fitness_over_generations.gif",
                duration_ms=config.experiment.gif.frame_duration_ms,
            )

        if config.experiment.gif.population_animation and generation_population_snapshots:
            population_frames = _build_population_animation_frames(
                generation_population_snapshots,
                config.experiment.plotting.figure_dpi,
            )
            save_gif_from_arrays(
                population_frames,
                gifs_dir / "population_heatmap.gif",
                duration_ms=config.experiment.gif.frame_duration_ms,
            )

    single_summary = {
        "config_name": config.config_name,
        "config_path": str(config.config_path),
        "resolved_output_dir": str(output_dir),
        "problem": {
            "capacity": config.experiment.problem.capacity,
            "target_value": config.experiment.problem.target_value,
        },
        "ga_parameters": {
            "num_generations": config.experiment.ga.num_generations,
            "solutions_per_population": config.experiment.ga.solutions_per_population,
            "num_parents_mating": config.experiment.ga.num_parents_mating,
            "parent_selection_type": config.experiment.ga.parent_selection_type,
            "mutation_type": config.experiment.ga.mutation_type,
            "mutation_percent_genes": config.experiment.ga.mutation_percent_genes,
            "mutation_num_genes": mutation_num_genes,
            "crossover_type": config.experiment.ga.crossover_type,
            "keep_parents": config.experiment.ga.keep_parents,
        },
        "seed_details": {
            "single_run_seed": config.experiment.runs.single_run_seed,
            "seed_strategy": config.experiment.runs.seed_strategy,
            "base_seed": config.experiment.runs.base_seed,
            "fixed_seeds": config.experiment.runs.fixed_seeds,
        },
        "runtime_seconds": runtime_seconds,
        "best_fitness": float(best_fitness),
        "best_total_value": float(best_value),
        "best_total_weight": float(best_weight),
        "success": bool(success),
        "generations_completed": len(generation_fitness_history),
        "selected_item_ids": [row["id"] for row in selected_rows],
    }
    write_json(output_dir / "single_run_summary.json", single_summary)
    write_csv_rows(
        output_dir / "single_run_selected_items.csv",
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
    print(f"Output directory: {output_dir}")

    return {
        "config_name": config.config_name,
        "best_value": float(best_value),
        "best_weight": float(best_weight),
        "runtime_seconds": runtime_seconds,
        "success": bool(success),
        "history": generation_fitness_history,
    }


def run_experiments(config: LoadedExperimentConfig) -> dict[str, Any]:
    """Run configured independent GA runs and report success statistics."""
    output_dir = config.output_dir
    plots_dir = output_dir / "plots"

    seeds = list(config.run_seeds())
    num_runs = config.experiment.runs.num_runs
    run_rows: list[dict[str, Any]] = []
    run_histories: list[list[float]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        ga_instance = create_ga_instance(config=config, random_seed=run_seed)

        started_at = time.perf_counter()
        ga_instance.run()
        runtime_seconds = time.perf_counter() - started_at

        best_solution, best_fitness, _ = ga_instance.best_solution()
        best_value, best_weight = evaluate_solution(best_solution)
        success = (
            best_weight <= config.experiment.problem.capacity
            and best_value >= config.experiment.problem.target_value
        )
        history = [float(value) for value in ga_instance.best_solutions_fitness]
        run_histories.append(history)

        run_rows.append(
            {
                "run_index": run_idx,
                "seed": run_seed,
                "best_fitness": float(best_fitness),
                "best_value": float(best_value),
                "best_weight": float(best_weight),
                "success": bool(success),
                "runtime_seconds": runtime_seconds,
                "generations_completed": len(history),
            }
        )

    write_csv_rows(
        output_dir / "multi_run_results.csv",
        run_rows,
        fieldnames=[
            "run_index",
            "seed",
            "best_fitness",
            "best_value",
            "best_weight",
            "success",
            "runtime_seconds",
            "generations_completed",
        ],
    )

    successes = [row for row in run_rows if row["success"]]
    success_count = len(successes)
    success_percentage = (success_count / num_runs) * 100.0

    target_value = config.experiment.problem.target_value
    capacity = config.experiment.problem.capacity
    avg_target_completion_percentage = float(
        sum(min((row["best_value"] / target_value) * 100.0, 100.0) for row in run_rows) / num_runs
    )
    avg_feasible_target_completion_percentage = float(
        sum(
            min((row["best_value"] / target_value) * 100.0, 100.0)
            if row["best_weight"] <= capacity
            else 0.0
            for row in run_rows
        )
        / num_runs
    )

    avg_success_runtime = (
        float(sum(row["runtime_seconds"] for row in successes) / success_count)
        if success_count > 0
        else 0.0
    )

    if config.experiment.plotting.enabled and run_histories:
        if config.experiment.plotting.multi_run_overlay:
            save_multi_run_overlay(
                run_histories,
                plots_dir / "multi_run_overlay.png",
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_value,
            )
        if config.experiment.plotting.aggregate_trend:
            save_aggregate_trend(
                run_histories,
                plots_dir / "multi_run_aggregate_trend.png",
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_value,
            )

    summary_payload = {
        "config_name": config.config_name,
        "config_path": str(config.config_path),
        "resolved_output_dir": str(output_dir),
        "problem": {
            "capacity": config.experiment.problem.capacity,
            "target_value": config.experiment.problem.target_value,
        },
        "ga_parameters": {
            "num_generations": config.experiment.ga.num_generations,
            "solutions_per_population": config.experiment.ga.solutions_per_population,
            "num_parents_mating": config.experiment.ga.num_parents_mating,
            "parent_selection_type": config.experiment.ga.parent_selection_type,
            "mutation_type": config.experiment.ga.mutation_type,
            "mutation_percent_genes": config.experiment.ga.mutation_percent_genes,
            "mutation_num_genes": config.resolve_mutation_num_genes(NUM_GENES),
            "crossover_type": config.experiment.ga.crossover_type,
            "keep_parents": config.experiment.ga.keep_parents,
        },
        "seed_details": {
            "seed_strategy": config.experiment.runs.seed_strategy,
            "base_seed": config.experiment.runs.base_seed,
            "fixed_seeds": config.experiment.runs.fixed_seeds,
            "resolved_run_seeds": seeds,
        },
        "success_metrics": {
            "success_count": success_count,
            "num_runs": num_runs,
            "success_percentage": success_percentage,
            "avg_target_completion_percentage": avg_target_completion_percentage,
            "avg_feasible_target_completion_percentage": avg_feasible_target_completion_percentage,
            "avg_success_runtime": avg_success_runtime,
        },
    }
    write_json(output_dir / "multi_run_summary.json", summary_payload)

    print("\n10-run experiment summary:")
    print(
        "Runs reaching value "
        f"{config.experiment.problem.target_value:.0f}: {success_count}/{num_runs}"
    )
    print(f"Success percentage: {success_percentage:.2f}%")
    print(f"Avg target completion (%): {avg_target_completion_percentage:.2f}%")
    print(f"Avg feasible target completion (%): {avg_feasible_target_completion_percentage:.2f}%")
    print(f"Average successful runtime (s): {avg_success_runtime:.6f}")

    return {
        "config_name": config.config_name,
        "success_count": success_count,
        "num_runs": num_runs,
        "success_percentage": success_percentage,
        "avg_target_completion_percentage": avg_target_completion_percentage,
        "avg_feasible_target_completion_percentage": avg_feasible_target_completion_percentage,
        "avg_success_runtime": avg_success_runtime,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Return CLI parser for task01 experiment execution."""
    parser = argparse.ArgumentParser(description="Run config-driven knapsack GA experiments")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file (default: task01/configs/knapsack_default.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_single(config)
    run_experiments(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
