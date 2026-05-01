"""Reusable PSO helpers shared across the lab08 tasks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


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


def make_bounds(
    lower_bound: float, upper_bound: float, dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create lower and upper PSO bounds arrays."""
    lower = np.full(dimensions, lower_bound, dtype=float)
    upper = np.full(dimensions, upper_bound, dtype=float)
    return lower, upper


def make_global_best_pso(
    *,
    n_particles: int,
    dimensions: int,
    options: dict[str, float],
    lower_bound: float,
    upper_bound: float,
) -> Any:
    """Build a bounded GlobalBestPSO instance."""
    import pyswarms as ps

    return ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=make_bounds(lower_bound, upper_bound, dimensions),
    )


def save_cost_history_plot(
    cost_history: Sequence[float],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    """Persist a cost-history plot to disk."""
    from pyswarms.utils.plotters import plot_cost_history

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(output_path.parent)
    try:
        ax = plot_cost_history(cost_history, title=title)
        fig = ax.get_figure()
        if not isinstance(fig, Figure):
            raise TypeError("plot_cost_history() did not return a Figure-backed axes")
        fig.tight_layout()
        fig.savefig(output_path.name, dpi=dpi)
        plt.close(fig)
    finally:
        os.chdir(original_cwd)


def save_contour_animation(
    pos_history: np.ndarray,
    objective_function: Callable[[np.ndarray], np.ndarray],
    output_path: Path,
    *,
    mark: tuple[float, float] | None = None,
    fps: int = 10,
    writer: str = "imagemagick",
    title: str = "Trajectory",
) -> None:
    """Create and save a 2D contour animation from a swarm history."""
    from pyswarms.utils.plotters.formatters import Mesher
    from pyswarms.utils.plotters.plotters import plot_contour

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(output_path.parent)
    try:
        mesher = Mesher(objective_function)  # pyright: ignore[reportCallIssue]
        animation = plot_contour(pos_history=pos_history, mesher=mesher, mark=mark, title=title)
        animation.save(output_path.name, writer=writer, fps=fps)
    finally:
        os.chdir(original_cwd)


def remove_report_log(path: Path | None = None) -> None:
    """Remove a generated report.log file if it exists."""
    report_path = (path or Path.cwd()) / "report.log"
    if report_path.exists():
        report_path.unlink()
