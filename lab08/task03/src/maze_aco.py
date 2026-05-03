"""Task 03 maze solving with Ant Colony Optimization."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
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

from task03.data.maze import GOAL, MAX_STEPS, START, is_passable, maze

from .config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

DEFAULT_SEED = 42

ACTIONS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left")
WALL_COLOR = "#101010"
PATH_COLOR = "#4400ff"
BEST_COLOR = "#eeff00"
START_COLOR = "#00ff37"
GOAL_COLOR = "#ee0808"


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


def _softmax_choice(rng: np.random.Generator, weights: np.ndarray) -> int:
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(weights)):
        weights = np.ones_like(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        probabilities = np.full(weights.shape, 1.0 / weights.size, dtype=float)
    else:
        probabilities = weights / total
    return int(rng.choice(weights.size, p=probabilities))


def _evaluate_path(path: list[tuple[int, int]]) -> MazePathResult:
    steps_taken = max(0, len(path) - 1)
    revisits = len(path) - len(set(path))
    goal_distance = manhattan_distance(path[-1], GOAL)
    reached_goal = path[-1] == GOAL
    if reached_goal:
        cost = float(steps_taken) + revisits * 0.25
    else:
        cost = 100.0 + float(steps_taken) + goal_distance * 4.0 + revisits * 0.5
    return MazePathResult(
        path=path,
        cost=cost,
        reached_goal=reached_goal,
        steps_taken=steps_taken,
        revisits=revisits,
        goal_distance=goal_distance,
    )


def _construct_ant_path(
    rng: np.random.Generator,
    pheromone: dict[tuple[tuple[int, int], tuple[int, int]], float],
    *,
    alpha: float,
    beta: float,
    max_steps: int,
) -> MazePathResult:
    current = START
    path = [current]
    visited_counts: dict[tuple[int, int], int] = defaultdict(int)
    visited_counts[current] = 1

    for _step in range(max_steps):
        if current == GOAL:
            break
        neighbors = legal_neighbors(current)
        if not neighbors:
            break

        weights = []
        for neighbor in neighbors:
            pheromone_level = pheromone.get((current, neighbor), 1.0)
            heuristic = 1.0 / (1.0 + manhattan_distance(neighbor, GOAL))
            revisit_penalty = 1.0 / (1.0 + visited_counts[neighbor])
            weights.append((pheromone_level**alpha) * (heuristic**beta) * revisit_penalty)

        chosen_index = _softmax_choice(rng, np.asarray(weights, dtype=float))
        current = neighbors[chosen_index]
        path.append(cast(Any, current))
        visited_counts[current] += 1
        if current == GOAL:
            break

    return _evaluate_path(cast(Any, path))


def _evaporate_pheromone(
    pheromone: dict[tuple[tuple[int, int], tuple[int, int]], float],
    evaporation_rate: float,
) -> None:
    retention = 1.0 - evaporation_rate
    for edge in list(pheromone):
        pheromone[edge] = max(1e-6, pheromone[edge] * retention)


def _deposit_path(
    pheromone: dict[tuple[tuple[int, int], tuple[int, int]], float],
    path: Sequence[tuple[int, int]],
    amount: float,
) -> None:
    if amount <= 0.0:
        return
    for start, end in itertools.pairwise(path):
        pheromone[(start, end)] = pheromone.get((start, end), 1.0) + amount
        pheromone[(end, start)] = pheromone.get((end, start), 1.0) + amount


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
    ax.plot(list(cost_history), color=BEST_COLOR, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("ACO Maze Cost History")
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
    sample_every_n_generations: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = list(range(0, len(path_history), max(1, sample_every_n_generations)))
    if not frame_indices or frame_indices[-1] != len(path_history) - 1:
        frame_indices.append(len(path_history) - 1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    def update(frame_index: int):
        ax.clear()
        _draw_maze(ax, path_history[frame_index], title=f"ACO Iteration {frame_index + 1}")
        return ()

    animation = FuncAnimation(
        fig, update, frames=frame_indices, interval=frame_duration_ms, blit=False
    )
    writer = PillowWriter(fps=max(1, (round(1000 / frame_duration_ms))))
    animation.save(output_path, writer=writer)
    plt.close(fig)


def _run_single_experiment(
    config: LoadedExperimentConfig,
    *,
    seed: int,
    run_index: int,
) -> dict[str, Any]:
    aco_config = config.experiment.aco
    rng = np.random.default_rng(seed)
    pheromone: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    best_result: MazePathResult | None = None
    best_path_history: list[list[tuple[int, int]]] = []
    best_cost_history: list[float] = []

    for _iteration in range(aco_config.iterations):
        ant_results = [
            _construct_ant_path(
                rng,
                pheromone,
                alpha=aco_config.alpha,
                beta=aco_config.beta,
                max_steps=MAX_STEPS,
            )
            for _ant in range(aco_config.ant_count)
        ]
        iteration_best = min(ant_results, key=lambda result: result.cost)
        if best_result is None or iteration_best.cost < best_result.cost:
            best_result = iteration_best

        assert best_result is not None
        best_path_history.append(list(best_result.path))
        best_cost_history.append(float(best_result.cost))

        _evaporate_pheromone(pheromone, aco_config.pheromone_evaporation_rate)
        for result in ant_results:
            deposit_amount = aco_config.pheromone_constant / max(result.cost, 1.0)
            _deposit_path(pheromone, result.path, deposit_amount)
        _deposit_path(
            pheromone,
            best_result.path,
            aco_config.pheromone_constant / max(best_result.cost * 0.5, 1.0),
        )

    assert best_result is not None

    run_dir = config.output_dir / config.config_name / "aco"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "solver": "aco",
        "run_index": run_index,
        "seed": seed,
        "reached_goal": best_result.reached_goal,
        "best_cost": float(best_result.cost),
        "best_path": [list(point) for point in best_result.path],
        "steps_taken": best_result.steps_taken,
        "revisits": best_result.revisits,
        "goal_distance": best_result.goal_distance,
        "cost_history": best_cost_history,
        "parameters": config.experiment.aco.model_dump(),
    }

    if config.experiment.plotting.enabled and config.experiment.plotting.save_cost_plot:
        _save_cost_plot(
            best_cost_history,
            run_dir / "cost_history.png",
            dpi=config.experiment.plotting.figure_dpi,
        )

    if config.experiment.gif.enabled and config.experiment.gif.aco_evolution:
        _save_path_gif(
            best_path_history,
            run_dir / "evolution.gif",
            dpi=config.experiment.plotting.figure_dpi,
            frame_duration_ms=config.experiment.gif.frame_duration_ms,
            sample_every_n_generations=config.experiment.gif.sample_every_n_generations,
        )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def run_experiments(config: LoadedExperimentConfig) -> dict[str, Any]:
    return _run_single_experiment(config, seed=DEFAULT_SEED, run_index=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run task03 maze ACO experiments.")
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
        f"ACO best cost={summary['best_cost']:.3f}, reached_goal={summary['reached_goal']}, "
        f"output={config.output_dir / 'aco'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
