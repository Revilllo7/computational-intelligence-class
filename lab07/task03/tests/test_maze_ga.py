"""Tests for task03 maze GA + A* workflow."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from task03.data.maze import GOAL, START
from task03.src.config import load_experiment_config
from task03.src.maze_ga import (
    a_star_solve,
    create_ga_instance,
    decode_chromosome,
    make_maze_fitness,
    simulate_route,
)


def _coords_to_actions(path: list[tuple[int, int]]) -> list[int]:
    deltas_to_action = {
        (0, -1): 0,
        (1, 0): 1,
        (0, 1): 2,
        (-1, 0): 3,
    }
    actions: list[int] = []
    for current, nxt in pairwise(path):
        dx = nxt[0] - current[0]
        dy = nxt[1] - current[1]
        actions.append(deltas_to_action[(dx, dy)])
    return actions


def test_decode_chromosome_normalizes_to_fixed_length() -> None:
    decoded = decode_chromosome([0, 1, 2, 3, 7])

    assert len(decoded) == 30
    assert decoded[:5] == [0, 1, 2, 3, 3]


def test_simulate_route_stays_on_wall_collision() -> None:
    # From START (1,1), moving up enters the border wall.
    route = simulate_route([0])

    assert route.trajectory[0] == START
    assert route.trajectory[-1] == START
    assert route.invalid_move_attempts == 1


def test_solved_route_scores_above_unsolved_route() -> None:
    solved_path = a_star_solve().path
    assert solved_path
    solved_actions = _coords_to_actions(solved_path)

    solved_route = simulate_route(solved_actions)
    unsolved_route = simulate_route([0] * 30)

    config = load_experiment_config(Path("task03/configs/maze_baseline.yaml"))
    fitness_fn = make_maze_fitness(config)
    ga = create_ga_instance(config, random_seed=7)

    solved_score = fitness_fn(ga, np.array(solved_actions + [0] * 30, dtype=int), 0)
    unsolved_score = fitness_fn(ga, np.array([0] * 30, dtype=int), 0)

    assert solved_route.reached_goal
    assert not unsolved_route.reached_goal
    assert solved_score > unsolved_score


def test_shorter_solved_route_scores_above_longer_solved_route() -> None:
    solved_path = a_star_solve().path
    assert solved_path
    solved_actions = _coords_to_actions(solved_path)

    # A valid detour near the start (right-left repeats) keeps solvability but adds inefficiency.
    longer_solved_actions = [1, 3, 1, 3, 1, 3, *solved_actions]

    config = load_experiment_config(Path("task03/configs/maze_baseline.yaml"))
    fitness_fn = make_maze_fitness(config)
    ga = create_ga_instance(config, random_seed=9)

    short_score = fitness_fn(ga, np.array(solved_actions + [0] * 30, dtype=int), 0)
    long_score = fitness_fn(ga, np.array(longer_solved_actions + [0] * 30, dtype=int), 0)

    assert short_score > long_score


def test_generation_callback_stops_when_solved() -> None:
    config = load_experiment_config(Path("task03/configs/maze_baseline.yaml"))
    ga = create_ga_instance(config, random_seed=123)

    ga.run()

    best_solution, _, _ = ga.best_solution()
    best_route = simulate_route(decode_chromosome(best_solution))

    assert best_route.effective_steps <= 30


def test_astar_path_is_valid_and_reaches_goal() -> None:
    result = a_star_solve()

    assert result.path
    assert result.path[0] == START
    assert result.path[-1] == GOAL

    for current, nxt in pairwise(result.path):
        dx = abs(current[0] - nxt[0])
        dy = abs(current[1] - nxt[1])
        assert dx + dy == 1


def test_artifact_generation_smoke(tmp_path: Path) -> None:
    config_file = tmp_path / "maze_pytest_smoke.yaml"
    config_payload = {
        "experiment_name": "pytest_smoke_artifacts",
        "output_root": "task03/experiments",
        "ga": {
            "num_generations": 30,
            "solutions_per_population": 60,
            "num_parents_mating": 20,
            "parent_selection_type": "sss",
            "mutation_type": "random",
            "mutation_num_genes": 1,
            "crossover_type": "single_point",
            "keep_parents": 2,
        },
        "runs": {
            "num_runs": 3,
            "timing_runs": 3,
            "single_run_seed": 11,
            "seed_strategy": "incremental",
            "base_seed": 90,
            "sweep_configs": ["task03/configs/maze_small.yaml", "task03/configs/maze_large.yaml"],
        },
        "gif": {
            "enabled": True,
            "ga_evolution": True,
            "astar_solving": True,
            "frame_duration_ms": 80,
            "ga_sample_every_n_generations": 2,
            "astar_sample_every_n_steps": 2,
        },
    }
    config_file.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    from task03.src.maze_ga import run_parameter_sweep, run_single, run_timing_benchmark

    config = load_experiment_config(config_file)
    single = run_single(config)
    sweep = run_parameter_sweep(config)
    benchmark = run_timing_benchmark(config)

    assert single["ga_single_run"]["effective_steps"] <= 30
    assert sweep["ranking"]
    assert benchmark["ga"]["run_count"] == 3

    expected_artifacts = [
        config.output_dir / "single_run_summary.json",
        config.output_dir / "sweep_comparison.csv",
        config.output_dir / "sweep_comparison.json",
        config.output_dir / "benchmark_summary.json",
        config.output_dir / "benchmark_ga_runs.csv",
        config.output_dir / "benchmark_astar_runs.csv",
        config.output_dir / "gifs" / "ga_evolution.gif",
        config.output_dir / "gifs" / "astar_solving.gif",
        config.output_dir / "plots" / "ga_best_path.png",
        config.output_dir / "plots" / "astar_path.png",
    ]

    for artifact in expected_artifacts:
        assert artifact.exists()
        assert artifact.stat().st_size > 0
