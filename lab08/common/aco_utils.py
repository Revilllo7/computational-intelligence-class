"""Reusable ACO (Ant Colony Optimization) utilities shared across lab08 tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the process working directory."""
    original_cwd = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def calculate_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two coordinates."""
    return float(np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2))


def calculate_path_distance(
    path: Sequence[tuple[float, float]],
) -> float:
    """
    Calculate total distance of a TSP path (includes return to start).

    Args:
        path: Sequence of (x, y) coordinates representing the tour.

    Returns:
        Total distance traveling all edges in the closed tour.
    """
    if len(path) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(len(path)):
        current = path[i]
        next_node = path[(i + 1) % len(path)]  # Loop back to start at end
        total_distance += calculate_distance(current, next_node)

    return total_distance


def nearest_neighbor_tsp(
    coords: Sequence[tuple[float, float]],
    start_idx: int | None = None,
) -> tuple[list[tuple[float, float]], float]:
    """
    Greedy nearest-neighbor heuristic for TSP.

    Starts at a given node (or tries all starting nodes if None) and always
    visits the nearest unvisited city. Returns the best path found.

    Args:
        coords: Sequence of (x, y) coordinates.
        start_idx: Starting city index. If None, tries all possible starting points.

    Returns:
        Tuple of (best_path, best_distance).
    """
    coords_list = list(coords)
    n = len(coords_list)

    if n < 2:
        return coords_list, 0.0

    best_path: list[tuple[float, float]] = []
    best_distance = float("inf")

    # Determine which starting indices to try
    start_indices = [start_idx] if start_idx is not None else range(n)

    for start in start_indices:
        unvisited = set(range(n))
        current = start
        path = [coords_list[current]]
        unvisited.remove(current)

        while unvisited:
            nearest = min(
                unvisited, key=lambda i: calculate_distance(coords_list[current], coords_list[i])
            )
            path.append(coords_list[nearest])
            unvisited.remove(nearest)
            current = nearest

        distance = calculate_path_distance(path)
        if distance < best_distance:
            best_distance = distance
            best_path = path

    return best_path, best_distance


def save_tsp_convergence_plot(
    convergence_data: dict[str, Any],
    output_dir: Path,
    filename: str = "convergence_comparison.png",
) -> None:
    """
    Save a plot comparing convergence curves from multiple ACO runs or methods.

    Args:
        convergence_data: Dict mapping method names to lists of best distances per iteration.
                         E.g., {"baseline": [500, 480, 470, ...], "high_exploration": [...]}
        output_dir: Directory to save plot.
        filename: Output filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.style.use("default")

    for method_name, distances in convergence_data.items():
        iterations = range(len(distances))
        plt.plot(
            iterations,
            distances,
            marker="o",
            label=method_name,
            linewidth=2,
            markersize=3,
            alpha=0.7,
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Path Distance", fontsize=12)
    plt.title("ACO Convergence Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def save_tsp_visualization(
    coords: Sequence[tuple[float, float]],
    path: Sequence[tuple[float, float]],
    output_dir: Path,
    title: str = "TSP Solution",
    filename: str = "tsp_solution.png",
) -> None:
    """
    Save a visualization of the TSP solution.

    Args:
        coords: All coordinate points.
        path: The solution path (should form a cycle).
        output_dir: Directory to save plot.
        title: Plot title.
        filename: Output filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.style.use("default")

    # Plot all nodes
    coords_array = np.array(coords)
    plt.scatter(coords_array[:, 0], coords_array[:, 1], c="blue", s=100, zorder=3, label="Cities")

    # Plot solution path
    path_array = np.array(path)
    for i in range(len(path_array)):
        next_idx = (i + 1) % len(path_array)
        plt.plot(
            [path_array[i, 0], path_array[next_idx, 0]],
            [path_array[i, 1], path_array[next_idx, 1]],
            "r-",
            linewidth=2,
            alpha=0.7,
        )

    # Mark start node
    plt.scatter(
        [path_array[0, 0]],
        [path_array[0, 1]],
        c="green",
        s=150,
        marker="s",
        zorder=4,
        label="Start",
    )

    distance = calculate_path_distance(path)
    plt.xlabel("X", fontsize=11)
    plt.ylabel("Y", fontsize=11)
    plt.title(f"{title}\nDistance: {distance:.2f}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def save_tsp_comparison_plot(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
    filename: str = "tsp_comparison.png",
) -> None:
    """
    Save a comparison plot of multiple TSP solutions side-by-side.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    num_methods = len(results)
    _fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 6))

    # Handle single subplot case
    if num_methods == 1:
        axes = [axes]

    plt.style.use("default")

    for ax, (method_name, result) in zip(axes, results.items(), strict=False):
        path = result["path"]
        distance = result["distance"]
        execution_time = result.get("time", 0.0)

        path_array = np.array(path)

        # Plot nodes
        ax.scatter(path_array[:, 0], path_array[:, 1], c="blue", s=100, zorder=3)

        # Plot edges
        for i in range(len(path_array)):
            next_idx = (i + 1) % len(path_array)
            ax.plot(
                [path_array[i, 0], path_array[next_idx, 0]],
                [path_array[i, 1], path_array[next_idx, 1]],
                "r-",
                linewidth=2,
                alpha=0.7,
            )

        # Mark start
        ax.scatter([path_array[0, 0]], [path_array[0, 1]], c="green", s=150, marker="s", zorder=4)

        time_str = f"\nTime: {execution_time:.3f}s" if execution_time > 0 else ""
        ax.set_title(f"{method_name}\nDistance: {distance:.2f}{time_str}", fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

    plt.suptitle("TSP Solution Comparison", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def save_json_summary(
    data: dict[str, Any],
    output_dir: Path,
    filename: str = "summary.json",
) -> None:
    """
    Save experiment results as JSON.

    Args:
        data: Dictionary of results to save.
        output_dir: Directory to save JSON file.
        filename: Output filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with Path.open(filepath, "w") as f:
        json.dump(data, f, indent=2)
