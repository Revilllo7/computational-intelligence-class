"""Task 03 maze solving with Particle Swarm Optimization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from task03.data.maze import GOAL, START, is_passable, maze

from .config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

DEFAULT_SEED = 42

ACTIONS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left")
WALL_COLOR = "#1f2933"
PATH_COLOR = "#7c3aed"
START_COLOR = "#16a34a"
GOAL_COLOR = "#dc2626"


@dataclass(frozen=True)
class MazePathResult:
    path: list[tuple[int, int]]
    cost: float
    reached_goal: bool
    steps_taken: int
    revisits: int
    goal_distance: int


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def legal_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    neighbors: list[tuple[int, int]] = []
    for dx, dy in ACTIONS:
        candidate = (x + dx, y + dy)
        if is_passable(*candidate):
            neighbors.append(candidate)
    return neighbors


def _decode_particle(particle: np.ndarray) -> list[int]:
    action_count = len(ACTIONS)
    expected_dimensions = particle.shape[0]
    if expected_dimensions % action_count != 0:
        raise ValueError("particle dimension must be divisible by the number of actions")
    sequence_length = expected_dimensions // action_count
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if particle.size != expected_dimensions:
        raise ValueError(
            f"particle must have {expected_dimensions} dimensions, got {particle.size}"
        )
    reshaped = particle.reshape(sequence_length, action_count)
    return [int(np.argmax(segment)) for segment in reshaped]


def _evaluate_actions(action_indices: Sequence[int], pso_config: Any) -> MazePathResult:
    current = START
    path = [current]
    visited_counts: dict[tuple[int, int], int] = {current: 1}

    for action_index in action_indices:
        ordered_actions = list(
            np.argsort(
                np.asarray([action_index == idx for idx in range(len(ACTIONS))], dtype=float)
            )
        )[::-1]
        candidate = current
        for idx in ordered_actions:
            dx, dy = ACTIONS[int(idx)]
            potential = (current[0] + dx, current[1] + dy)
            if is_passable(*potential):
                candidate = potential
                break
        current = candidate
        path.append(cast(Any, current))
        visited_counts[current] = visited_counts.get(current, 0) + 1
        if current == GOAL:
            break

    steps_taken = max(0, len(path) - 1)
    revisits = len(path) - len(set(path))
    goal_distance = manhattan_distance(path[-1], GOAL)
    reached_goal = path[-1] == GOAL
    if reached_goal:
        cost = pso_config.step_weight * float(steps_taken) + pso_config.revisit_weight * revisits
    else:
        cost = (
            pso_config.failure_penalty
            + pso_config.step_weight * float(steps_taken)
            + pso_config.goal_distance_weight * goal_distance
            + pso_config.revisit_weight * revisits
        )
    return MazePathResult(
        path=cast(Any, path),
        cost=cost,
        reached_goal=reached_goal,
        steps_taken=steps_taken,
        revisits=revisits,
        goal_distance=goal_distance,
    )


def _evaluate_particle(particle: np.ndarray, pso_config: Any) -> MazePathResult:
    action_indices = _decode_particle(np.asarray(particle, dtype=float))
    return _evaluate_actions(action_indices, pso_config)


def _evaluate_swarm(swarm: np.ndarray, pso_config: Any) -> tuple[np.ndarray, list[MazePathResult]]:
    results = [_evaluate_particle(particle, pso_config) for particle in swarm]
    costs = np.asarray([result.cost for result in results], dtype=float)
    return costs, results


# multi-run support removed; single-run uses DEFAULT_SEED


def _draw_maze(ax: Any, path: Sequence[tuple[int, int]] | None = None, *, title: str = "") -> None:
    grid = np.asarray(maze, dtype=int)
    cmap = ListedColormap(["#f8fafc", WALL_COLOR])
    ax.imshow(grid, cmap=cmap, origin="upper")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if path:
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.5, marker="o", markersize=3)
    ax.scatter([START[0]], [START[1]], color=START_COLOR, s=70, label="start", zorder=3)
    ax.scatter([GOAL[0]], [GOAL[1]], color=GOAL_COLOR, s=70, label="goal", zorder=3)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.set_title(title)


def _save_cost_plot(cost_history: Sequence[float], output_path: Path, *, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(list(cost_history), color=PATH_COLOR, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("PSO Maze Cost History")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_path_gif(
    path_history: Sequence[Sequence[tuple[int, int]]],
    output_path: Path,
    *,
    dpi: int,
    frame_duration_ms: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    def update(frame_index: int):
        ax.clear()
        _draw_maze(ax, path_history[frame_index], title=f"PSO Iteration {frame_index + 1}")
        return ()

    animation = FuncAnimation(
        fig, update, frames=range(len(path_history)), interval=frame_duration_ms, blit=False
    )
    writer = PillowWriter(fps=max(1, (round(1000 / frame_duration_ms))))
    animation.save(output_path, writer=writer)
    plt.close(fig)


def _run_pso(
    config: LoadedExperimentConfig,
) -> dict[str, Any]:
    pso_config = config.experiment.pso
    rng = np.random.default_rng(DEFAULT_SEED)

    dimensions = pso_config.sequence_length * len(ACTIONS)
    particle_count = pso_config.particle_count
    iterations = pso_config.iterations
    lower_bound = pso_config.lower_bound
    upper_bound = pso_config.upper_bound
    velocity_clamp = pso_config.velocity_clamp

    swarm = rng.uniform(lower_bound, upper_bound, size=(particle_count, dimensions))
    velocity = np.zeros_like(swarm)

    costs, results = _evaluate_swarm(swarm, pso_config)
    personal_best_positions = swarm.copy()
    personal_best_costs = costs.copy()
    personal_best_results = list(results)

    global_best_index = int(np.argmin(personal_best_costs))
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = float(personal_best_costs[global_best_index])
    global_best_result = personal_best_results[global_best_index]

    cost_history: list[float] = []
    path_history: list[list[tuple[int, int]]] = []

    for _iteration in range(iterations):
        costs, results = _evaluate_swarm(swarm, pso_config)
        improved = costs < personal_best_costs
        if np.any(improved):
            personal_best_positions[improved] = swarm[improved]
            personal_best_costs[improved] = costs[improved]
            for index in np.flatnonzero(improved):
                personal_best_results[index] = results[index]

        global_best_index = int(np.argmin(personal_best_costs))
        if float(personal_best_costs[global_best_index]) < global_best_cost:
            global_best_cost = float(personal_best_costs[global_best_index])
            global_best_position = personal_best_positions[global_best_index].copy()
            global_best_result = personal_best_results[global_best_index]

        cost_history.append(global_best_cost)
        path_history.append(list(global_best_result.path))

        random_1 = rng.random(size=swarm.shape)
        random_2 = rng.random(size=swarm.shape)
        velocity = (
            pso_config.w * velocity
            + pso_config.c1 * random_1 * (personal_best_positions - swarm)
            + pso_config.c2 * random_2 * (global_best_position - swarm)
        )
        velocity = np.clip(velocity, -velocity_clamp, velocity_clamp)
        swarm = np.clip(swarm + velocity, lower_bound, upper_bound)

    run_dir = config.output_dir / config.config_name / "pso"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "solver": "pso",
        "seed": DEFAULT_SEED,
        "reached_goal": global_best_result.reached_goal,
        "best_cost": float(global_best_result.cost),
        "best_path": [list(point) for point in global_best_result.path],
        "steps_taken": global_best_result.steps_taken,
        "revisits": global_best_result.revisits,
        "goal_distance": global_best_result.goal_distance,
        "cost_history": cost_history,
        "parameters": config.experiment.pso.model_dump(),
        "optimizer": {
            "particle_count": particle_count,
            "iterations": iterations,
            "dimensions": dimensions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "velocity_clamp": velocity_clamp,
        },
    }

    if config.experiment.plotting.enabled and config.experiment.plotting.save_cost_plot:
        _save_cost_plot(
            cost_history,
            run_dir / "cost_history.png",
            dpi=config.experiment.plotting.figure_dpi,
        )

    if config.experiment.gif.enabled and config.experiment.gif.pso_evolution:
        _save_path_gif(
            path_history,
            run_dir / "evolution.gif",
            dpi=config.experiment.plotting.figure_dpi,
            frame_duration_ms=config.experiment.gif.frame_duration_ms,
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def run_experiments(config: LoadedExperimentConfig) -> dict[str, Any]:
    return _run_pso(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run task03 maze PSO experiments.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML configuration file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    summary = run_experiments(config)
    print(
        f"PSO best cost={summary['best_cost']:.3f}, reached_goal={summary['reached_goal']}, "
        f"output={config.output_dir / 'pso'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
