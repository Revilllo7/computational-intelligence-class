"""Tests for task01 knapsack GA workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from task01.src.config import load_experiment_config
from task01.src.knapsack_ga import (
    NUM_GENES,
    create_ga_instance,
    evaluate_solution,
    penalized_fitness,
    run_experiments,
    run_single,
)


def test_overweight_penalty_is_strongly_negative() -> None:
    """Overweight solutions should be punished below zero in this setup."""
    overweight_solution = np.ones(11, dtype=int)
    fitness = penalized_fitness(overweight_solution, capacity=25.0)
    assert fitness < 0


def test_known_optimal_feasible_set_hits_target() -> None:
    """Known best feasible set: item2 + item3 + item5 + item7 + item8 + item10."""
    chromosome = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
    total_value, total_weight = evaluate_solution(chromosome)

    assert total_weight == 25.0
    assert total_value == 1630.0


def test_ga_factory_returns_independent_instances() -> None:
    """Factory must return fresh GA instances for repeated experiments."""
    config = load_experiment_config(Path("task01/configs/knapsack_default.yaml"))
    ga_a = create_ga_instance(config=config, random_seed=101)
    ga_b = create_ga_instance(config=config, random_seed=202)

    assert ga_a is not ga_b
    assert ga_a.random_seed != ga_b.random_seed
    assert ga_a.mutation_num_genes == config.resolve_mutation_num_genes(NUM_GENES)


def test_artifact_generation_smoke(tmp_path: Path) -> None:
    """A tiny config should generate expected JSON/CSV/PNG/GIF artifacts."""
    config_file = tmp_path / "knapsack_pytest_smoke.yaml"
    config_payload = {
        "experiment_name": "pytest_smoke_artifacts",
        "output_root": "task01/experiments",
        "problem": {"capacity": 25.0, "target_value": 1630.0},
        "ga": {
            "num_generations": 8,
            "solutions_per_population": 18,
            "num_parents_mating": 8,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 10,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
        "runs": {
            "num_runs": 3,
            "single_run_seed": 123,
            "seed_strategy": "incremental",
            "base_seed": 1000,
        },
        "plotting": {
            "enabled": True,
            "single_run_curve": True,
            "multi_run_overlay": True,
            "aggregate_trend": True,
            "figure_dpi": 80,
        },
        "gif": {
            "enabled": True,
            "fitness_animation": True,
            "population_animation": True,
            "frame_duration_ms": 80,
        },
    }
    config_file.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    config = load_experiment_config(config_file)
    output_dir = config.output_dir

    try:
        run_single(config)
        run_experiments(config)

        expected_artifacts = [
            output_dir / "single_run_summary.json",
            output_dir / "single_run_selected_items.csv",
            output_dir / "multi_run_results.csv",
            output_dir / "multi_run_summary.json",
            output_dir / "plots" / "single_run_fitness.png",
            output_dir / "plots" / "multi_run_overlay.png",
            output_dir / "plots" / "multi_run_aggregate_trend.png",
            output_dir / "gifs" / "fitness_over_generations.gif",
            output_dir / "gifs" / "population_heatmap.gif",
        ]

        for artifact in expected_artifacts:
            assert artifact.exists()
            assert artifact.stat().st_size > 0
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
