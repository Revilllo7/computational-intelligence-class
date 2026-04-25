"""Tests for task02 alloys GA workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from task02.src.alloy_ga import (
    NUM_GENES,
    active_metal_count,
    create_ga_instance,
    evaluate_endurance,
    run_experiments,
    run_single,
    summarize_solution,
)
from task02.src.config import load_experiment_config


def test_endurance_matches_known_values() -> None:
    solution = np.array([0.25, np.sin(0.25), 0.5, 0.25, 0.75, 0.25], dtype=float)

    fitness = evaluate_endurance(solution)

    assert np.isclose(fitness, 1.0 + np.sin(0.5 * 0.75) + np.cos(0.25 * 0.25))


def test_active_metal_threshold_counts_expected_genes() -> None:
    chromosome = [0.02, 0.5, 0.0, 0.06, 0.8, 0.05]

    assert active_metal_count(chromosome, threshold=0.05) == 3


def test_summary_includes_gene_vector_and_active_count() -> None:
    summary = summarize_solution([0.1, 0.2, 0.3, 0.04, 0.9, 0.7], 2.5, 0.05)

    assert summary["best_durability"] == 2.5
    assert summary["best_chromosome"] == [0.1, 0.2, 0.3, 0.04, 0.9, 0.7]
    assert summary["active_metal_count"] == 5
    assert summary["metals_to_mix"] == 5


def test_ga_factory_returns_independent_instances() -> None:
    config = load_experiment_config(Path("task02/configs/alloy_default.yaml"))
    ga_a = create_ga_instance(config=config, random_seed=101)
    ga_b = create_ga_instance(config=config, random_seed=202)

    assert ga_a is not ga_b
    assert ga_a.random_seed != ga_b.random_seed
    assert ga_a.mutation_num_genes == config.resolve_mutation_num_genes(NUM_GENES)


def test_artifact_generation_smoke(tmp_path: Path) -> None:
    config_file = tmp_path / "alloy_pytest_smoke.yaml"
    config_payload = {
        "experiment_name": "pytest_smoke_artifacts",
        "output_root": "task02/experiments",
        "problem": {"target_durability": 2.8, "active_metal_threshold": 0.05},
        "ga": {
            "num_generations": 16,
            "solutions_per_population": 20,
            "num_parents_mating": 8,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 20,
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
        single_summary = run_single(config)
        multi_summary = run_experiments(config)

        assert single_summary["best_durability"] >= 0.0
        assert multi_summary["best_durability"] >= 0.0

        expected_artifacts = [
            output_dir / "single_run_summary.json",
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
