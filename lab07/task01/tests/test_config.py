"""Tests for task01 YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from task01.src.config import DEFAULT_OUTPUT_ROOT, load_experiment_config


def test_load_default_config() -> None:
    config = load_experiment_config(Path("task01/configs/knapsack_default.yaml"))

    assert config.config_name == "knapsack_default"
    assert config.output_root == DEFAULT_OUTPUT_ROOT.resolve()
    assert config.output_dir == (DEFAULT_OUTPUT_ROOT.resolve() / "knapsack_default")
    assert config.experiment.problem.capacity == 25.0
    assert config.experiment.ga.mutation_percent_genes == 10


def test_output_directory_uses_explicit_experiment_name(tmp_path: Path) -> None:
    config_file = tmp_path / "knapsack_unused_stem.yaml"
    payload = {
        "experiment_name": "manual_experiment_name",
        "output_root": "task01/experiments",
        "problem": {"capacity": 25.0, "target_value": 1630.0},
        "ga": {
            "num_generations": 20,
            "solutions_per_population": 20,
            "num_parents_mating": 8,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 12,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_experiment_config(config_file)
    assert config.config_name == "manual_experiment_name"
    assert config.output_dir == (DEFAULT_OUTPUT_ROOT.resolve() / "manual_experiment_name")


def test_output_directory_routing_by_config_name_stem() -> None:
    default_config = load_experiment_config(Path("task01/configs/knapsack_default.yaml"))
    variant_config = load_experiment_config(Path("task01/configs/knapsack_high_mutation.yaml"))

    assert default_config.config_name == "knapsack_default"
    assert variant_config.config_name == "knapsack_high_mutation"
    assert default_config.output_dir != variant_config.output_dir


def test_rejects_non_standard_output_root(tmp_path: Path) -> None:
    config_file = tmp_path / "knapsack_invalid_output.yaml"
    payload = {
        "output_root": "reports/experiments",
        "problem": {"capacity": 25.0, "target_value": 1630.0},
        "ga": {
            "num_generations": 20,
            "solutions_per_population": 20,
            "num_parents_mating": 8,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 12,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(config_file)


def test_raises_for_missing_config_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_config(Path("task01/configs/not_existing.yaml"))
