"""Tests for task02 YAML configuration loading and validation."""

from pathlib import Path

import pytest

from task02.src.config import DEFAULT_CONFIG_PATH, load_experiment_config


def test_load_default_config_reads_aco_values() -> None:
    """Default YAML values should be applied to the loaded ACO config."""
    cfg = load_experiment_config(DEFAULT_CONFIG_PATH)

    assert cfg.config_path.exists()
    assert cfg.config_name == "default_config"
    assert cfg.output_dir.name == cfg.config_name

    assert cfg.experiment.aco.ant_count == 300
    assert cfg.experiment.aco.alpha == 0.5
    assert cfg.experiment.aco.beta == 1.2
    assert cfg.experiment.aco.pheromone_evaporation_rate == 0.4
    assert cfg.experiment.aco.pheromone_constant == 1000.0
    assert cfg.experiment.aco.iterations == 300


def test_load_config_missing_file_raises() -> None:
    """Loading a non-existing YAML file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_experiment_config(Path("task02/configs/does_not_exist.yaml"))


def test_load_flat_yaml_config(tmp_path: Path) -> None:
    """Flat YAML format should be supported and mapped into the nested ACO schema."""
    cfg_file = tmp_path / "flat.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "ant_count: 123",
                "alpha: 0.9",
                "beta: 1.1",
                "pheromone_evaporation_rate: 0.25",
                "pheromone_constant: 777.0",
                "iterations: 42",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_file)

    assert cfg.experiment.aco.ant_count == 123
    assert cfg.experiment.aco.alpha == 0.9
    assert cfg.experiment.aco.beta == 1.1
    assert cfg.experiment.aco.pheromone_evaporation_rate == 0.25
    assert cfg.experiment.aco.pheromone_constant == 777.0
    assert cfg.experiment.aco.iterations == 42
