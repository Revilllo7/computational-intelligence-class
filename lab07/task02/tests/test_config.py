"""Tests for task02 YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from task02.src.config import DEFAULT_OUTPUT_ROOT, load_experiment_config


def test_load_default_config() -> None:
    config = load_experiment_config(Path("task02/configs/alloy_default.yaml"))

    assert config.config_name == "alloy_default"
    assert config.output_root == DEFAULT_OUTPUT_ROOT.resolve()
    assert config.output_dir == (DEFAULT_OUTPUT_ROOT.resolve() / "alloy_default")
    assert config.experiment.problem.target_durability == 2.8
    assert config.experiment.problem.active_metal_threshold == 0.05
    assert config.experiment.ga.mutation_percent_genes == 20


def test_output_directory_uses_explicit_experiment_name(tmp_path: Path) -> None:
    config_file = tmp_path / "alloy_unused_stem.yaml"
    payload = {
        "experiment_name": "manual_experiment_name",
        "output_root": "task02/experiments",
        "problem": {"target_durability": 2.8, "active_metal_threshold": 0.05},
        "ga": {
            "num_generations": 12,
            "solutions_per_population": 16,
            "num_parents_mating": 6,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 20,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_experiment_config(config_file)
    assert config.config_name == "manual_experiment_name"
    assert config.output_dir == (DEFAULT_OUTPUT_ROOT.resolve() / "manual_experiment_name")


def test_output_directory_routing_by_config_name_stem() -> None:
    default_config = load_experiment_config(Path("task02/configs/alloy_default.yaml"))

    assert default_config.config_name == "alloy_default"
    assert default_config.output_dir == (DEFAULT_OUTPUT_ROOT.resolve() / "alloy_default")


def test_rejects_non_standard_output_root(tmp_path: Path) -> None:
    config_file = tmp_path / "alloy_invalid_output.yaml"
    payload = {
        "output_root": "reports/experiments",
        "problem": {"target_durability": 2.8, "active_metal_threshold": 0.05},
        "ga": {
            "num_generations": 12,
            "solutions_per_population": 16,
            "num_parents_mating": 6,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 20,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(config_file)


def test_rejects_invalid_threshold(tmp_path: Path) -> None:
    config_file = tmp_path / "alloy_invalid_threshold.yaml"
    payload = {
        "output_root": "task02/experiments",
        "problem": {"target_durability": 2.8, "active_metal_threshold": 1.2},
        "ga": {
            "num_generations": 12,
            "solutions_per_population": 16,
            "num_parents_mating": 6,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_percent_genes": 20,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_experiment_config(config_file)


def test_raises_for_missing_config_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_config(Path("task02/configs/not_existing.yaml"))
