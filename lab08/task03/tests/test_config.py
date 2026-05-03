"""Tests for task03 YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from task03.src.config import DEFAULT_OUTPUT_ROOT, ExperimentConfig, load_experiment_config


def test_load_experiment_config_reads_nested_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: tuned",
                "output_root: task03/experiments",
                "aco:",
                "  ant_count: 11",
                "  alpha: 0.7",
                "  beta: 1.1",
                "  pheromone_evaporation_rate: 0.3",
                "  pheromone_constant: 500.0",
                "  iterations: 12",
                "pso:",
                "  particle_count: 9",
                "  iterations: 10",
                "  sequence_length: 8",
                "  lower_bound: -3.0",
                "  upper_bound: 3.0",
                "  velocity_clamp: 1.5",
                "  c1: 0.6",
                "  c2: 0.4",
                "  w: 0.5",
                "  step_weight: 1.0",
                "  revisit_weight: 2.0",
                "  goal_distance_weight: 5.0",
                "  failure_penalty: 2000.0",
                "plotting:",
                "  enabled: false",
                "  save_cost_plot: false",
                "  figure_dpi: 80",
                "gif:",
                "  enabled: false",
                "  aco_evolution: false",
                "  pso_evolution: false",
                "  astar_solving: false",
                "  sample_every_n_generations: 2",
                "  astar_sample_every_n_steps: 1",
                "  frame_duration_ms: 120",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_config(config_path)

    assert loaded.config_name == "tuned"
    assert loaded.output_root == DEFAULT_OUTPUT_ROOT.resolve()
    assert loaded.output_dir == DEFAULT_OUTPUT_ROOT.resolve()
    assert loaded.experiment.aco.ant_count == 11
    assert loaded.experiment.pso.sequence_length == 8
    assert loaded.experiment.plotting.enabled is False
    assert loaded.experiment.gif.enabled is False


def test_load_experiment_config_rejects_wrong_output_root(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: bad",
                "output_root: /tmp/outside-task03",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_root must resolve"):
        load_experiment_config(config_path)


def test_experiment_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(unexpected_field=True)  # type: ignore
