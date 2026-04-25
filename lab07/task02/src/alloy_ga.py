"""Task 02: Alloy durability optimization with a continuous PyGAD chromosome."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygad

from common.ga_utils import write_csv_rows, write_json
from common.gif_utils import figure_to_rgb_array, save_gif_from_arrays
from task02.src.config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

GENE_ORDER: tuple[str, ...] = ("x", "y", "z", "v", "u", "w")
NUM_GENES = len(GENE_ORDER)
GENE_SPACE_LOW = 0.01
GENE_SPACE_HIGH = 0.99
DEFAULT_ACTIVE_METAL_THRESHOLD = 0.05
DEFAULT_TARGET_DURABILITY = 2.8
CLIP_GENE_VALUES = True


def sanitize_solution(
    solution: np.ndarray | Sequence[float], *, clip: bool = CLIP_GENE_VALUES
) -> np.ndarray:
    """Convert a raw chromosome into a finite float vector."""
    genes = np.asarray(solution, dtype=float)
    genes = np.nan_to_num(genes, nan=0.5, posinf=GENE_SPACE_HIGH, neginf=GENE_SPACE_LOW)
    if clip:
        genes = np.clip(genes, GENE_SPACE_LOW, GENE_SPACE_HIGH)
    return genes


def evaluate_endurance(solution: np.ndarray | Sequence[float]) -> float:
    """Evaluate the alloy endurance objective for one chromosome."""
    x, y, z, v, u, w = sanitize_solution(solution, clip=True)
    value = np.exp(-2.0 * (y - np.sin(x)) ** 2) + np.sin(z * u) + np.cos(v * w)
    return float(np.nan_to_num(value, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf))


def make_fitness_function():
    """Create a PyGAD-compatible fitness callback for maximization."""

    def _fitness(_: pygad.GA, solution: np.ndarray, __: int) -> float:
        return evaluate_endurance(solution)

    return _fitness


def active_metal_count(solution: Sequence[float], threshold: float) -> int:
    """Count how many genes are materially used in the best chromosome."""
    genes = sanitize_solution(solution, clip=True)
    return int(np.count_nonzero(genes > threshold))


def summarize_solution(
    solution: np.ndarray | Sequence[float],
    fitness: float,
    threshold: float,
) -> dict[str, Any]:
    genes = sanitize_solution(solution, clip=True)
    gene_values = {name: float(value) for name, value in zip(GENE_ORDER, genes, strict=True)}
    active_genes = [name for name, value in gene_values.items() if value > threshold]
    return {
        "gene_order": list(GENE_ORDER),
        "best_chromosome": [float(value) for value in genes],
        "gene_values": gene_values,
        "active_gene_threshold": float(threshold),
        "active_gene_names": active_genes,
        "active_metal_count": len(active_genes),
        "metals_to_mix": len(active_genes),
        "best_durability": float(fitness),
    }


def _make_generation_callback(
    config: LoadedExperimentConfig,
    generation_fitness_history: list[float] | None,
    generation_population_snapshots: list[np.ndarray] | None,
):
    """Capture optional per-generation state and stop when the target is reached."""

    def _on_generation(ga_instance: pygad.GA) -> str | None:
        if generation_fitness_history is not None and ga_instance.best_solutions_fitness:
            generation_fitness_history.append(float(ga_instance.best_solutions_fitness[-1]))

        if generation_population_snapshots is not None:
            generation_population_snapshots.append(
                np.array(ga_instance.population, dtype=float, copy=True)
            )

        _, best_fitness, _ = ga_instance.best_solution()
        if best_fitness >= config.experiment.problem.target_durability:
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
    gene_space = [{"low": GENE_SPACE_LOW, "high": GENE_SPACE_HIGH} for _ in range(NUM_GENES)]
    ga = pygad.GA(
        num_generations=config.experiment.ga.num_generations,
        num_parents_mating=config.experiment.ga.num_parents_mating,
        sol_per_pop=config.experiment.ga.solutions_per_population,
        num_genes=NUM_GENES,
        fitness_func=make_fitness_function(),
        init_range_low=0,
        init_range_high=1,
        gene_type=float,
        gene_space=gene_space,
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
        generation_population_snapshots.append(np.array(ga.population, dtype=float, copy=True))

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
    ax.plot(generations, history, color="tab:blue", linewidth=2, marker="o", markersize=6)
    ax.axhline(y=target_value, color="red", linestyle=":", linewidth=2)
    ax.set_xlim(0.5, max(2, len(history)) + 0.5)
    ax.set_ylim(max(0.0, min(history) - 0.1), max(max(history), target_value) + 0.1)
    ax.set_title("Alloy GA Convergence (Single Run)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Durability")
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
    ax.set_title("Best Durability by Generation (All Runs)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Durability")
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
    ax.set_title("Aggregate Best Durability Trend (Mean with Min/Max Band)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Durability")
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
    y_min = min(history) if history else 0.0
    y_max = max(max(history, default=target_value), target_value)
    span = max(target_value * 0.1, y_max - y_min, 0.1)

    for frame_end in range(1, len(history) + 1):
        shown_generations = full_generations[:frame_end]
        shown_fitness = history[:frame_end]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=figure_dpi)
        ax.plot(shown_generations, shown_fitness, color="tab:blue", linewidth=2)
        ax.scatter(shown_generations[-1], shown_fitness[-1], color="tab:red", s=18)
        ax.set_xlim(1, max(2, len(history)))
        ax.set_ylim(y_min - 0.1 * span, y_max + 0.1 * span)
        ax.set_title("Best Durability Over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Durability")
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
        image = ax.imshow(
            snapshot,
            aspect="auto",
            cmap="viridis",
            vmin=GENE_SPACE_LOW,
            vmax=GENE_SPACE_HIGH,
        )
        ax.set_title(f"Population Heatmap (Generation {generation_label})")
        ax.set_xlabel("Gene Index")
        ax.set_ylabel("Population Member")
        fig.colorbar(image, ax=ax, label="Gene Value")
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
    summary = summarize_solution(
        best_solution,
        float(best_fitness),
        config.experiment.problem.active_metal_threshold,
    )
    summary.update(
        {
            "config_name": config.config_name,
            "seed": config.experiment.runs.single_run_seed,
            "runtime_seconds": runtime_seconds,
            "target_durability": config.experiment.problem.target_durability,
            "reached_target": float(best_fitness) >= config.experiment.problem.target_durability,
        }
    )

    final_history = [float(value) for value in ga_instance.best_solutions_fitness]
    if not generation_fitness_history:
        generation_fitness_history = final_history
    if not generation_fitness_history:
        generation_fitness_history = [float(best_fitness)]

    if config.experiment.plotting.enabled and config.experiment.plotting.single_run_curve:
        save_single_run_fitness_curve(
            generation_fitness_history,
            plots_dir / "single_run_fitness.png",
            config.experiment.plotting.figure_dpi,
            config.experiment.problem.target_durability,
        )

    if config.experiment.gif.enabled and generation_fitness_history:
        if config.experiment.gif.fitness_animation:
            fitness_frames = _build_fitness_animation_frames(
                generation_fitness_history,
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_durability,
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

    write_json(output_dir / "single_run_summary.json", summary)
    return summary


def run_experiments(config: LoadedExperimentConfig) -> dict[str, Any]:
    """Execute repeated runs and persist aggregate artifacts."""
    output_dir = config.output_dir
    plots_dir = output_dir / "plots"
    run_histories: list[list[float]] = []
    results_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    run_seeds = list(config.run_seeds())

    for run_index, seed in enumerate(run_seeds, start=1):
        generation_history: list[float] = []
        ga_instance = create_ga_instance(
            config=config,
            random_seed=seed,
            generation_fitness_history=generation_history,
        )

        started_at = time.perf_counter()
        ga_instance.run()
        runtime_seconds = time.perf_counter() - started_at

        best_solution, best_fitness, _ = ga_instance.best_solution()
        summary = summarize_solution(
            best_solution,
            float(best_fitness),
            config.experiment.problem.active_metal_threshold,
        )
        summary.update(
            {
                "run_index": run_index,
                "seed": seed,
                "runtime_seconds": runtime_seconds,
                "target_durability": config.experiment.problem.target_durability,
                "reached_target": float(best_fitness)
                >= config.experiment.problem.target_durability,
            }
        )
        run_summaries.append(summary)
        run_histories.append(generation_history or [float(best_fitness)])
        results_rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                "best_durability": float(best_fitness),
                "active_metal_count": summary["active_metal_count"],
                "runtime_seconds": runtime_seconds,
            }
        )

    best_run = max(run_summaries, key=lambda row: float(row["best_durability"]))
    best_values = np.array([float(row["best_durability"]) for row in run_summaries], dtype=float)
    summary_payload = {
        "config_name": config.config_name,
        "run_count": len(run_summaries),
        "gene_order": list(GENE_ORDER),
        "active_gene_threshold": config.experiment.problem.active_metal_threshold,
        "target_durability": config.experiment.problem.target_durability,
        "best_run_index": best_run["run_index"],
        "best_run_seed": best_run["seed"],
        "best_durability": best_run["best_durability"],
        "best_chromosome": best_run["best_chromosome"],
        "best_gene_values": best_run["gene_values"],
        "best_active_metal_count": best_run["active_metal_count"],
        "variability": {
            "mean": float(np.mean(best_values)),
            "std": float(np.std(best_values, ddof=0)),
            "min": float(np.min(best_values)),
            "max": float(np.max(best_values)),
            "median": float(np.median(best_values)),
        },
        "runs": run_summaries,
    }

    if config.experiment.plotting.enabled:
        if config.experiment.plotting.multi_run_overlay:
            save_multi_run_overlay(
                run_histories,
                plots_dir / "multi_run_overlay.png",
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_durability,
            )
        if config.experiment.plotting.aggregate_trend:
            save_aggregate_trend(
                run_histories,
                plots_dir / "multi_run_aggregate_trend.png",
                config.experiment.plotting.figure_dpi,
                config.experiment.problem.target_durability,
            )

    write_csv_rows(
        output_dir / "multi_run_results.csv",
        results_rows,
        ["run_index", "seed", "best_durability", "active_metal_count", "runtime_seconds"],
    )
    write_json(output_dir / "multi_run_summary.json", summary_payload)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 02 alloy durability GA")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)

    single_run_summary = run_single(config)
    multi_run_summary = run_experiments(config)

    print(
        "Single run best durability="
        f"{single_run_summary['best_durability']:.6f}, active metals="
        f"{single_run_summary['active_metal_count']}"
    )
    print(
        "Multi-run best durability="
        f"{multi_run_summary['best_durability']:.6f}, active metals="
        f"{multi_run_summary['best_active_metal_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
