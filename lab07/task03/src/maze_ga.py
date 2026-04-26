"""Task 03: Maze solving with a GA baseline and deterministic A* comparison."""

from __future__ import annotations

import argparse
import heapq
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygad

from common.ga_utils import write_csv_rows, write_json
from common.gif_utils import figure_to_rgb_array, save_gif_from_arrays
from task03.data.maze import GOAL, MAX_STEPS, START, maze
from task03.src.config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

Coord = tuple[int, int]


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIRECTION_TO_DELTA: dict[Direction, Coord] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}
DIRECTION_SPACE = [
    int(Direction.UP),
    int(Direction.RIGHT),
    int(Direction.DOWN),
    int(Direction.LEFT),
]
NUM_GENES = MAX_STEPS


def to_coord(value: Sequence[int]) -> Coord:
    """Convert a two-item integer sequence to a coordinate tuple."""
    if len(value) != 2:
        raise ValueError(f"Expected coordinate of length 2, got {len(value)}")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class RouteStats:
    trajectory: list[Coord]
    action_codes: list[int]
    effective_path: list[Coord]
    effective_action_codes: list[int]
    reached_goal: bool
    invalid_move_attempts: int
    revisited_cells: int
    effective_steps: int
    unique_passable_cells: int
    final_position: Coord


@dataclass
class StopState:
    reason: str = "generation_limit"
    generation: int | None = None
    best_fitness: float | None = None
    best_steps: int | None = None


@dataclass
class AStarResult:
    path: list[Coord]
    explored_order: list[Coord]
    visited_count: int
    runtime_seconds: float


def to_direction(value: int) -> Direction:
    """Map any integer into one of the 4 canonical direction values."""
    normalized = int(value) % 4
    return Direction(normalized)


def decode_chromosome(solution: np.ndarray | Sequence[int]) -> list[int]:
    """Decode a chromosome to MAX_STEPS direction-gene integers in [0, 3]."""
    genes = np.asarray(solution, dtype=int).tolist()
    decoded = [int(gene) % 4 for gene in genes[:MAX_STEPS]]
    if len(decoded) < MAX_STEPS:
        decoded.extend([0] * (MAX_STEPS - len(decoded)))
    return decoded


def in_bounds(coord: Coord) -> bool:
    x, y = coord
    return 0 <= y < len(maze) and 0 <= x < len(maze[0])


def is_wall(coord: Coord) -> bool:
    x, y = coord
    return maze[y][x] == 1


def is_passable(coord: Coord) -> bool:
    return in_bounds(coord) and not is_wall(coord)


def move_coordinate(coord: Coord, direction_code: int) -> Coord:
    direction = to_direction(direction_code)
    dx, dy = DIRECTION_TO_DELTA[direction]
    x, y = coord
    return x + dx, y + dy


def manhattan_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def truncate_path_at_goal(path: Sequence[Coord], goal: Coord = GOAL) -> list[Coord]:
    for index, coord in enumerate(path):
        if coord == goal:
            return list(path[: index + 1])
    return list(path)


def simulate_route(
    action_codes: Sequence[int],
    *,
    start: Coord = START,
    goal: Coord = GOAL,
    max_steps: int = MAX_STEPS,
) -> RouteStats:
    """Simulate a route and return trajectory plus quality/status metrics."""
    current = start
    visited = {start}
    trajectory: list[Coord] = [start]
    effective_path: list[Coord] = [start]
    effective_actions: list[int] = []
    bounded_actions = [int(code) % 4 for code in action_codes[:max_steps]]

    invalid_moves = 0
    revisits = 0

    for action_code in bounded_actions:
        candidate = move_coordinate(current, action_code)
        if is_passable(candidate):
            if candidate in visited:
                revisits += 1
            current = candidate
            visited.add(current)
            effective_path.append(current)
            effective_actions.append(action_code)
        else:
            invalid_moves += 1

        trajectory.append(current)

        if current == goal:
            break

    reached_goal = current == goal
    if reached_goal:
        effective_path = truncate_path_at_goal(effective_path, goal)

    final_position = effective_path[-1] if effective_path else current

    return RouteStats(
        trajectory=trajectory,
        action_codes=bounded_actions,
        effective_path=effective_path,
        effective_action_codes=effective_actions,
        reached_goal=reached_goal,
        invalid_move_attempts=invalid_moves,
        revisited_cells=revisits,
        effective_steps=max(0, len(effective_path) - 1),
        unique_passable_cells=len(set(effective_path)),
        final_position=final_position,
    )


def score_route(
    stats: RouteStats,
    *,
    success_bonus: float,
    distance_weight: float,
    progress_weight: float,
    efficiency_weight: float,
    solved_step_penalty: float,
    invalid_move_penalty: float,
    revisit_penalty: float,
    stagnation_penalty: float,
    exploration_reward: float,
    target_steps: int,
    goal: Coord = GOAL,
) -> float:
    """Reward solved, short, valid routes and penalize invalid/repetitive behavior."""
    start_distance = manhattan_distance(stats.trajectory[0], goal)
    distance = manhattan_distance(stats.final_position, goal)
    best_distance = min(manhattan_distance(coord, goal) for coord in stats.trajectory)
    progress = max(0, start_distance - best_distance)
    wasted_moves = max(0, stats.effective_steps - progress)

    score = 0.0
    score -= distance_weight * distance
    score += progress_weight * progress
    score -= invalid_move_penalty * stats.invalid_move_attempts
    score -= revisit_penalty * stats.revisited_cells
    score -= stagnation_penalty * wasted_moves
    score += exploration_reward * stats.unique_passable_cells
    score += efficiency_weight * max(0, target_steps - stats.effective_steps)

    if stats.reached_goal:
        score += success_bonus
        score -= solved_step_penalty * max(0, stats.effective_steps - target_steps)

    return float(score)


def make_maze_fitness(config: LoadedExperimentConfig):
    """Create a PyGAD-compatible fitness callback for the maze problem."""
    fitness_cfg = config.experiment.fitness
    start = to_coord(config.experiment.problem.start)
    goal = to_coord(config.experiment.problem.goal)

    def _fitness(_: pygad.GA, solution: np.ndarray, __: int) -> float:
        stats = simulate_route(
            decode_chromosome(solution),
            start=start,
            goal=goal,
            max_steps=config.experiment.problem.max_steps,
        )
        return score_route(
            stats,
            success_bonus=fitness_cfg.success_bonus,
            distance_weight=fitness_cfg.distance_weight,
            progress_weight=fitness_cfg.progress_weight,
            efficiency_weight=fitness_cfg.efficiency_weight,
            solved_step_penalty=fitness_cfg.solved_step_penalty,
            invalid_move_penalty=fitness_cfg.invalid_move_penalty,
            revisit_penalty=fitness_cfg.revisit_penalty,
            stagnation_penalty=fitness_cfg.stagnation_penalty,
            exploration_reward=fitness_cfg.exploration_reward,
            target_steps=config.experiment.problem.target_steps,
            goal=goal,
        )

    return _fitness


def _make_generation_callback(
    config: LoadedExperimentConfig,
    stop_state: StopState,
    generation_fitness_history: list[float] | None,
    best_route_history: list[RouteStats] | None,
):
    """Capture generation-level telemetry and early-stop when a valid route is found."""

    def _on_generation(ga_instance: pygad.GA) -> str | None:
        if ga_instance.best_solutions_fitness:
            best_fitness = float(ga_instance.best_solutions_fitness[-1])
        else:
            _, best_fitness_raw, _ = ga_instance.best_solution()
            best_fitness = float(best_fitness_raw)

        if generation_fitness_history is not None:
            generation_fitness_history.append(best_fitness)

        best_solution, _, _ = ga_instance.best_solution()
        best_stats = simulate_route(
            decode_chromosome(best_solution),
            start=to_coord(config.experiment.problem.start),
            goal=to_coord(config.experiment.problem.goal),
            max_steps=config.experiment.problem.max_steps,
        )

        if best_route_history is not None:
            best_route_history.append(best_stats)

        if (
            best_stats.reached_goal
            and best_stats.effective_steps <= config.experiment.problem.max_steps
        ):
            stop_state.reason = "goal_reached"
            stop_state.generation = int(ga_instance.generations_completed)
            stop_state.best_steps = int(best_stats.effective_steps)
            stop_state.best_fitness = best_fitness
            return "stop"

        stop_state.reason = "generation_limit"
        stop_state.generation = int(ga_instance.generations_completed)
        stop_state.best_steps = int(best_stats.effective_steps)
        stop_state.best_fitness = best_fitness
        return None

    return _on_generation


def create_ga_instance(
    config: LoadedExperimentConfig,
    random_seed: int | None = None,
    *,
    stop_state: StopState | None = None,
    generation_fitness_history: list[float] | None = None,
    best_route_history: list[RouteStats] | None = None,
) -> pygad.GA:
    """Create a fresh GA instance for one independent run."""
    state = stop_state or StopState()
    ga = pygad.GA(
        num_generations=config.experiment.ga.num_generations,
        num_parents_mating=config.experiment.ga.num_parents_mating,
        sol_per_pop=config.experiment.ga.solutions_per_population,
        num_genes=NUM_GENES,
        fitness_func=make_maze_fitness(config),
        init_range_low=0,
        init_range_high=4,
        gene_type=int,
        gene_space=DIRECTION_SPACE,
        keep_parents=config.experiment.ga.keep_parents,
        parent_selection_type=config.experiment.ga.parent_selection_type,
        crossover_type=config.experiment.ga.crossover_type,
        mutation_type=config.experiment.ga.mutation_type,
        mutation_num_genes=config.experiment.ga.mutation_num_genes,
        random_seed=random_seed,
        suppress_warnings=True,
        on_generation=_make_generation_callback(
            config,
            state,
            generation_fitness_history,
            best_route_history,
        ),
    )
    return ga


def a_star_solve(
    *,
    start: Coord = START,
    goal: Coord = GOAL,
) -> AStarResult:
    """Solve the maze with deterministic A* using Manhattan distance."""
    started_at = time.perf_counter()
    frontier: list[tuple[int, int, Coord]] = []
    counter = 0
    heapq.heappush(frontier, (manhattan_distance(start, goal), counter, start))

    came_from: dict[Coord, Coord | None] = {start: None}
    g_score: dict[Coord, int] = {start: 0}
    explored_order: list[Coord] = []

    while frontier:
        _, _, current = heapq.heappop(frontier)
        explored_order.append(current)

        if current == goal:
            break

        for direction in (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT):
            neighbor = move_coordinate(current, int(direction))
            if not is_passable(neighbor):
                continue

            tentative = g_score[current] + 1
            best_known = g_score.get(neighbor)
            if best_known is not None and tentative >= best_known:
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            priority = tentative + manhattan_distance(neighbor, goal)
            heapq.heappush(frontier, (priority, counter, neighbor))

    path: list[Coord] = []
    if goal in came_from:
        cursor: Coord | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = came_from[cursor]
        path.reverse()

    runtime_seconds = time.perf_counter() - started_at
    return AStarResult(
        path=path,
        explored_order=explored_order,
        visited_count=len(explored_order),
        runtime_seconds=runtime_seconds,
    )


def _render_maze_frame(
    *,
    path: Sequence[Coord] | None = None,
    explored: Sequence[Coord] | None = None,
    title: str,
) -> np.ndarray:
    grid = np.asarray(maze, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
    ax.imshow(grid, cmap="binary", vmin=0, vmax=1)

    if explored:
        xs = [coord[0] for coord in explored]
        ys = [coord[1] for coord in explored]
        ax.scatter(xs, ys, c="#2a9d8f", s=22, alpha=0.7, label="explored")

    if path:
        path_x = [coord[0] for coord in path]
        path_y = [coord[1] for coord in path]
        ax.plot(path_x, path_y, c="#e76f51", linewidth=2.4, label="path")
        ax.scatter(path_x[-1], path_y[-1], c="#e63946", s=45)

    ax.scatter(START[0], START[1], c="#1d3557", marker="o", s=75, label="start")
    ax.scatter(GOAL[0], GOAL[1], c="#ffb703", marker="*", s=120, label="goal")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout()

    frame = figure_to_rgb_array(fig)
    plt.close(fig)
    return frame


def _save_png_snapshot(path: Path, *, route: Sequence[Coord], title: str) -> None:
    frame = _render_maze_frame(path=route, title=title)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
    ax.imshow(frame)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _sample_best_routes(
    best_route_history: Sequence[RouteStats], sample_every: int
) -> list[RouteStats]:
    if not best_route_history:
        return []
    sampled = [
        best_route_history[index] for index in range(0, len(best_route_history), sample_every)
    ]
    if sampled[-1] is not best_route_history[-1]:
        sampled.append(best_route_history[-1])
    return sampled


def create_ga_evolution_gif(
    best_route_history: Sequence[RouteStats],
    output_gif_path: Path,
    output_png_path: Path,
    *,
    sample_every_generations: int,
    duration_ms: int,
) -> None:
    sampled_routes = _sample_best_routes(best_route_history, sample_every_generations)
    if not sampled_routes:
        return

    frames: list[np.ndarray] = []
    for idx, route in enumerate(sampled_routes, start=1):
        frame = _render_maze_frame(
            path=route.effective_path,
            title=(
                "GA Best-So-Far"
                f" | frame={idx} | steps={route.effective_steps} | solved={route.reached_goal}"
            ),
        )
        frames.append(frame)

    save_gif_from_arrays(frames, output_gif_path, duration_ms=duration_ms)
    _save_png_snapshot(
        output_png_path,
        route=sampled_routes[-1].effective_path,
        title="GA final best path",
    )


def create_astar_gif(
    astar_result: AStarResult,
    output_gif_path: Path,
    output_png_path: Path,
    *,
    sample_every_steps: int,
    duration_ms: int,
) -> None:
    if not astar_result.path:
        return

    sampled_indices = list(range(1, len(astar_result.explored_order) + 1, sample_every_steps))
    if sampled_indices[-1] != len(astar_result.explored_order):
        sampled_indices.append(len(astar_result.explored_order))

    frames: list[np.ndarray] = []
    for count in sampled_indices:
        explored = astar_result.explored_order[:count]
        partial_path = astar_result.path if count == len(astar_result.explored_order) else []
        frame = _render_maze_frame(
            path=partial_path,
            explored=explored,
            title=f"A* search | explored={count}",
        )
        frames.append(frame)

    save_gif_from_arrays(frames, output_gif_path, duration_ms=duration_ms)
    _save_png_snapshot(
        output_png_path,
        route=astar_result.path,
        title="A* shortest path",
    )


def run_ga_once(config: LoadedExperimentConfig, seed: int | None) -> dict[str, Any]:
    stop_state = StopState()
    fitness_history: list[float] = []
    route_history: list[RouteStats] = []

    ga_instance = create_ga_instance(
        config,
        random_seed=seed,
        stop_state=stop_state,
        generation_fitness_history=fitness_history,
        best_route_history=route_history,
    )

    started_at = time.perf_counter()
    ga_instance.run()
    runtime_seconds = time.perf_counter() - started_at

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_stats = simulate_route(
        decode_chromosome(best_solution),
        start=to_coord(config.experiment.problem.start),
        goal=to_coord(config.experiment.problem.goal),
        max_steps=config.experiment.problem.max_steps,
    )

    return {
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "best_fitness": float(best_fitness),
        "generations_completed": int(ga_instance.generations_completed),
        "stop_reason": stop_state.reason,
        "route": best_stats,
        "fitness_history": fitness_history,
        "best_route_history": route_history,
        "chromosome": decode_chromosome(best_solution),
    }


def _serialize_route(route: RouteStats) -> dict[str, Any]:
    payload = asdict(route)
    payload["trajectory"] = [list(coord) for coord in route.trajectory]
    payload["effective_path"] = [list(coord) for coord in route.effective_path]
    payload["final_position"] = list(route.final_position)
    return payload


def run_single(config: LoadedExperimentConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    gifs_dir = output_dir / "gifs"
    plots_dir = output_dir / "plots"

    one_run = run_ga_once(config, seed=config.experiment.runs.single_run_seed)
    best_route: RouteStats = one_run["route"]

    astar_result = a_star_solve(
        start=to_coord(config.experiment.problem.start),
        goal=to_coord(config.experiment.problem.goal),
    )

    if config.experiment.gif.enabled:
        if config.experiment.gif.ga_evolution:
            create_ga_evolution_gif(
                one_run["best_route_history"],
                gifs_dir / "ga_evolution.gif",
                plots_dir / "ga_best_path.png",
                sample_every_generations=config.experiment.gif.ga_sample_every_n_generations,
                duration_ms=config.experiment.gif.frame_duration_ms,
            )

        if config.experiment.gif.astar_solving:
            create_astar_gif(
                astar_result,
                gifs_dir / "astar_solving.gif",
                plots_dir / "astar_path.png",
                sample_every_steps=config.experiment.gif.astar_sample_every_n_steps,
                duration_ms=config.experiment.gif.frame_duration_ms,
            )

    summary = {
        "config_name": config.config_name,
        "config_path": str(config.config_path),
        "resolved_output_dir": str(output_dir),
        "ga_single_run": {
            "seed": one_run["seed"],
            "runtime_seconds": one_run["runtime_seconds"],
            "best_fitness": one_run["best_fitness"],
            "generations_completed": one_run["generations_completed"],
            "stop_reason": one_run["stop_reason"],
            "reached_goal": best_route.reached_goal,
            "effective_steps": best_route.effective_steps,
            "invalid_move_attempts": best_route.invalid_move_attempts,
            "revisited_cells": best_route.revisited_cells,
            "best_chromosome": one_run["chromosome"],
            "route": _serialize_route(best_route),
        },
        "astar_baseline": {
            "runtime_seconds": astar_result.runtime_seconds,
            "path_length_steps": max(0, len(astar_result.path) - 1),
            "visited_count": astar_result.visited_count,
            "path": [list(coord) for coord in astar_result.path],
        },
    }
    write_json(output_dir / "single_run_summary.json", summary)
    return summary


def run_repeated_ga(config: LoadedExperimentConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    for run_index, seed in enumerate(config.run_seeds(), start=1):
        one_run = run_ga_once(config, seed=seed)
        route: RouteStats = one_run["route"]
        rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                "success": int(route.reached_goal),
                "best_fitness": one_run["best_fitness"],
                "effective_steps": route.effective_steps,
                "generations_completed": one_run["generations_completed"],
                "runtime_seconds": one_run["runtime_seconds"],
                "stop_reason": one_run["stop_reason"],
            }
        )
        runs.append(
            {
                "run_index": run_index,
                "seed": seed,
                "success": route.reached_goal,
                "best_fitness": one_run["best_fitness"],
                "effective_steps": route.effective_steps,
                "runtime_seconds": one_run["runtime_seconds"],
                "generations_completed": one_run["generations_completed"],
                "stop_reason": one_run["stop_reason"],
            }
        )

    success_values = np.array([int(row["success"]) for row in rows], dtype=float)
    runtime_values = np.array([float(row["runtime_seconds"]) for row in rows], dtype=float)
    step_values = np.array([float(row["effective_steps"]) for row in rows], dtype=float)
    fitness_values = np.array([float(row["best_fitness"]) for row in rows], dtype=float)

    summary = {
        "config_name": config.config_name,
        "run_count": len(rows),
        "success_rate": float(np.mean(success_values)),
        "success_count": int(np.sum(success_values)),
        "best_fitness": float(np.max(fitness_values)),
        "mean_best_fitness": float(np.mean(fitness_values)),
        "mean_runtime_seconds": float(np.mean(runtime_values)),
        "runtime_std_seconds": float(np.std(runtime_values, ddof=0)),
        "median_steps": float(np.median(step_values)),
        "mean_steps": float(np.mean(step_values)),
        "runs": runs,
    }

    write_csv_rows(
        config.output_dir / "multi_run_results.csv",
        rows,
        [
            "run_index",
            "seed",
            "success",
            "best_fitness",
            "effective_steps",
            "generations_completed",
            "runtime_seconds",
            "stop_reason",
        ],
    )
    write_json(config.output_dir / "multi_run_summary.json", summary)
    return summary


def run_parameter_sweep(config: LoadedExperimentConfig) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []
    per_config_summaries: list[dict[str, Any]] = []

    for cfg_path in config.resolve_sweep_config_paths():
        sweep_config = load_experiment_config(cfg_path)
        # GIF artifacts are created by run_single, so invoke it for each sweep config.
        if sweep_config.experiment.gif.enabled and (
            sweep_config.experiment.gif.ga_evolution or sweep_config.experiment.gif.astar_solving
        ):
            run_single(sweep_config)

        summary = run_repeated_ga(sweep_config)
        per_config_summaries.append(summary)
        comparison_rows.append(
            {
                "config_name": sweep_config.config_name,
                "config_path": str(sweep_config.config_path),
                "success_rate": summary["success_rate"],
                "mean_runtime_seconds": summary["mean_runtime_seconds"],
                "median_steps": summary["median_steps"],
                "best_fitness": summary["best_fitness"],
            }
        )

    ranking = sorted(
        comparison_rows,
        key=lambda row: (
            -float(row["success_rate"]),
            float(row["mean_runtime_seconds"]),
            float(row["median_steps"]),
        ),
    )

    write_csv_rows(
        config.output_dir / "sweep_comparison.csv",
        ranking,
        [
            "config_name",
            "config_path",
            "success_rate",
            "mean_runtime_seconds",
            "median_steps",
            "best_fitness",
        ],
    )
    write_json(
        config.output_dir / "sweep_comparison.json",
        {
            "ranking": ranking,
            "configs": per_config_summaries,
        },
    )

    return {
        "ranking": ranking,
        "configs": per_config_summaries,
    }


def run_timing_benchmark(config: LoadedExperimentConfig) -> dict[str, Any]:
    ga_rows: list[dict[str, Any]] = []
    ga_runtimes: list[float] = []
    ga_steps: list[float] = []
    ga_success: list[int] = []

    for run_index, seed in enumerate(config.timing_seeds(), start=1):
        one_run = run_ga_once(config, seed=seed)
        route: RouteStats = one_run["route"]
        ga_rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                "success": int(route.reached_goal),
                "runtime_seconds": one_run["runtime_seconds"],
                "effective_steps": route.effective_steps,
            }
        )
        ga_runtimes.append(float(one_run["runtime_seconds"]))
        ga_steps.append(float(route.effective_steps))
        ga_success.append(int(route.reached_goal))

    astar_rows: list[dict[str, Any]] = []
    astar_runtimes: list[float] = []
    astar_steps: list[float] = []
    timing_runs = config.experiment.runs.timing_runs

    for run_index in range(1, timing_runs + 1):
        result = a_star_solve(
            start=to_coord(config.experiment.problem.start),
            goal=to_coord(config.experiment.problem.goal),
        )
        astar_rows.append(
            {
                "run_index": run_index,
                "runtime_seconds": result.runtime_seconds,
                "path_length_steps": max(0, len(result.path) - 1),
                "visited_count": result.visited_count,
            }
        )
        astar_runtimes.append(float(result.runtime_seconds))
        astar_steps.append(float(max(0, len(result.path) - 1)))

    summary = {
        "ga": {
            "run_count": timing_runs,
            "success_count": int(np.sum(ga_success)),
            "success_rate": float(np.mean(ga_success)),
            "runtime_mean_seconds": float(np.mean(ga_runtimes)),
            "runtime_std_seconds": float(np.std(ga_runtimes, ddof=0)),
            "steps_mean": float(np.mean(ga_steps)),
            "steps_std": float(np.std(ga_steps, ddof=0)),
        },
        "astar": {
            "run_count": timing_runs,
            "runtime_mean_seconds": float(np.mean(astar_runtimes)),
            "runtime_std_seconds": float(np.std(astar_runtimes, ddof=0)),
            "steps_mean": float(np.mean(astar_steps)),
            "steps_std": float(np.std(astar_steps, ddof=0)),
        },
    }

    write_csv_rows(
        config.output_dir / "benchmark_ga_runs.csv",
        ga_rows,
        ["run_index", "seed", "success", "runtime_seconds", "effective_steps"],
    )
    write_csv_rows(
        config.output_dir / "benchmark_astar_runs.csv",
        astar_rows,
        ["run_index", "runtime_seconds", "path_length_steps", "visited_count"],
    )
    write_json(config.output_dir / "benchmark_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 03 maze GA + A* benchmark")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "single", "sweep", "benchmark"],
        default="all",
        help="Run scope",
    )
    return parser


def _load_best_ranked_config(
    base_config: LoadedExperimentConfig, sweep: dict[str, Any]
) -> LoadedExperimentConfig:
    ranking = sweep.get("ranking", [])
    if not ranking:
        return base_config

    best_path = ranking[0]["config_path"]
    return load_experiment_config(best_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)

    if args.mode in {"single", "all"}:
        single_summary = run_single(config)
        print(
            "Single run: reached_goal="
            f"{single_summary['ga_single_run']['reached_goal']}"
            f", steps={single_summary['ga_single_run']['effective_steps']}"
            f", stop_reason={single_summary['ga_single_run']['stop_reason']}"
        )

    sweep_summary: dict[str, Any] = {"ranking": []}
    if args.mode in {"sweep", "all"}:
        sweep_summary = run_parameter_sweep(config)
        if sweep_summary["ranking"]:
            top = sweep_summary["ranking"][0]
            print(
                "Sweep winner: "
                f"{top['config_name']}"
                f" (success_rate={top['success_rate']:.3f}, "
                f"mean_runtime={top['mean_runtime_seconds']:.6f}s)"
            )

    if args.mode in {"benchmark", "all"}:
        benchmark_config = _load_best_ranked_config(config, sweep_summary)
        benchmark_summary = run_timing_benchmark(benchmark_config)
        print(
            "Timing benchmark: "
            f"GA mean={benchmark_summary['ga']['runtime_mean_seconds']:.6f}s, "
            f"A* mean={benchmark_summary['astar']['runtime_mean_seconds']:.6f}s"
        )

    print(f"Artifacts generated under: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
