"""Tests for task03 Maze PSO implementation."""

from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from task03.data.maze import GOAL, START, is_passable
from task03.src.config import (
    ExperimentConfig,
    GifConfig,
    LoadedExperimentConfig,
    PlotConfig,
    PSOConfig,
)
from task03.src.maze_pso import ACTIONS, _decode_particle, _evaluate_actions, run_experiments


def _shortest_path() -> list[tuple[int, int]]:
    queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque([(START, [START])])
    visited = {START}
    while queue:
        current, path = queue.popleft()
        if current == GOAL:
            return path
        for dx, dy in ACTIONS:
            candidate = (current[0] + dx, current[1] + dy)
            if is_passable(*candidate) and candidate not in visited:
                visited.add(cast(Any, candidate))
                queue.append((candidate, [*path, candidate]))
    raise AssertionError("maze has no path from start to goal")


def _path_to_actions(path: list[tuple[int, int]]) -> list[int]:
    action_lookup = {action: index for index, action in enumerate(ACTIONS)}
    action_indices: list[int] = []
    for start, end in itertools.pairwise(path):
        delta = (end[0] - start[0], end[1] - start[1])
        action_indices.append(action_lookup[delta])
    return action_indices


def test_decode_particle_chooses_segment_argmax() -> None:
    particle = np.asarray([0.1, 2.0, 0.3, -1.0, 5.0, 1.0, 0.0, 4.0], dtype=float)

    assert _decode_particle(particle) == [1, 0]


def test_evaluate_actions_on_shortest_path() -> None:
    path = _shortest_path()
    action_indices = _path_to_actions(path)
    pso_config = SimpleNamespace(
        step_weight=1.0,
        revisit_weight=0.0,
        goal_distance_weight=4.0,
        failure_penalty=1000.0,
    )

    result = _evaluate_actions(action_indices, pso_config)

    assert result.reached_goal is True
    assert result.steps_taken == len(path) - 1
    assert result.revisits == 0
    assert result.goal_distance == 0
    assert result.cost == float(len(path) - 1)


def test_run_experiments_with_tuned_config_reaches_goal(tmp_path: Path) -> None:
    experiment = ExperimentConfig(
        pso=PSOConfig(
            particle_count=90,
            iterations=160,
            sequence_length=30,
            lower_bound=-4.0,
            upper_bound=4.0,
            velocity_clamp=2.0,
            c1=0.5,
            c2=0.3,
            w=0.5,
            step_weight=1.0,
            revisit_weight=2.0,
            goal_distance_weight=5.0,
            failure_penalty=2000.0,
        ),
        plotting=PlotConfig(enabled=False, save_cost_plot=False, figure_dpi=80),
        gif=GifConfig(enabled=False, aco_evolution=False, pso_evolution=False, astar_solving=False),
    )
    config = LoadedExperimentConfig(
        config_path=tmp_path / "config.yaml",
        config_name="pso-test",
        project_root=tmp_path,
        task_root=tmp_path,
        output_root=tmp_path,
        output_dir=tmp_path,
        experiment=experiment,
    )

    summary = run_experiments(config)

    summary_path = tmp_path / "pso-test" / "pso" / "summary.json"
    assert summary_path.exists()
    assert summary["solver"] == "pso"
    assert summary["seed"] == 42
    assert summary["reached_goal"] is True
    assert summary["best_path"][-1] == list(GOAL)
