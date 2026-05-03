"""Tests for task03 Maze ACO implementation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, cast

from task03.data.maze import GOAL, START, is_passable
from task03.src.config import (
    ACOConfig,
    ExperimentConfig,
    GifConfig,
    LoadedExperimentConfig,
    PlotConfig,
)
from task03.src.maze_aco import ACTIONS, _evaluate_path, legal_neighbors, run_experiments


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


def test_legal_neighbors_from_start() -> None:
    assert legal_neighbors(START) == [(2, 1)]


def test_evaluate_path_uses_path_length_and_revisits() -> None:
    path = _shortest_path()

    result = _evaluate_path(path)

    assert result.reached_goal is True
    assert result.steps_taken == len(path) - 1
    assert result.revisits == 0
    assert result.goal_distance == 0
    assert result.cost == float(len(path) - 1)


def test_run_experiments_writes_summary(tmp_path: Path) -> None:
    experiment = ExperimentConfig(
        aco=ACOConfig(
            ant_count=8,
            alpha=0.6,
            beta=1.0,
            pheromone_evaporation_rate=0.35,
            pheromone_constant=250.0,
            iterations=10,
        ),
        plotting=PlotConfig(enabled=False, save_cost_plot=False, figure_dpi=80),
        gif=GifConfig(enabled=False, aco_evolution=False, pso_evolution=False, astar_solving=False),
    )
    config = LoadedExperimentConfig(
        config_path=tmp_path / "config.yaml",
        config_name="aco-test",
        project_root=tmp_path,
        task_root=tmp_path,
        output_root=tmp_path,
        output_dir=tmp_path,
        experiment=experiment,
    )

    summary = run_experiments(config)

    summary_path = tmp_path / "aco-test" / "aco" / "summary.json"
    assert summary_path.exists()
    assert summary["solver"] == "aco"
    assert summary["seed"] == 42
    assert summary["best_path"][0] == list(START)
