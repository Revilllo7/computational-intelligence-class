"""Reusable PSO helpers shared across the lab08 tasks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (register projection)


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


def save_3d_surface_and_animation(
    pos_history: np.ndarray,
    objective_function: Callable[[np.ndarray], np.ndarray],
    output_png: Path,
    output_gif: Path | None = None,
    *,
    dpi: int = 150,
    grid_size: int = 80,
    fps: int = 10,
):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_gif is not None:
        output_gif.parent.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(output_png.parent)
    try:
        # Normalize input
        arr = np.asarray(pos_history, dtype=float)
        if arr.ndim != 3:
            raise ValueError("pos_history must be shape (n_iters, n_particles, dims)")
        n_iters, _n_particles, _dims = arr.shape

        # Project to first two dimensions for surface plotting
        proj = arr[..., :2].reshape(-1, 2)
        x_min, y_min = proj.min(axis=0)
        x_max, y_max = proj.max(axis=0)
        # Add small margins
        x_margin = max(1e-6, 0.1 * (x_max - x_min) if x_max > x_min else 0.1)
        y_margin = max(1e-6, 0.1 * (y_max - y_min) if y_max > y_min else 0.1)
        xs = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
        ys = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
        X, Y = np.meshgrid(xs, ys)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        # Evaluate objective on grid (expects vectorized callable)
        try:
            Z = objective_function(grid_points)
        except Exception:
            # Fallback: evaluate in a loop
            Z = np.array([float(objective_function(p)) for p in grid_points], dtype=float)
        Z = Z.reshape(X.shape)

        # Create static 3D surface with min/max markers
        fig = plt.figure(figsize=(6, 5), dpi=dpi)
        ax = cast(Any, fig.add_subplot(111, projection="3d"))
        surf = cast(
            Any,
            ax.plot_surface(
                X, Y, Z, cmap=plt.colormaps["viridis"], linewidth=0, antialiased=True, alpha=0.9
            ),
        )
        cbar = cast(Any, fig.colorbar(surf, shrink=0.5, aspect=10))
        cbar.ax.set_ylabel("Objective")

        # Mark global min and max from grid
        flat_idx_min = np.nanargmin(Z)
        flat_idx_max = np.nanargmax(Z)
        xm_min, ym_min = X.ravel()[flat_idx_min], Y.ravel()[flat_idx_min]
        zm_min = Z.ravel()[flat_idx_min]
        xm_max, ym_max = X.ravel()[flat_idx_max], Y.ravel()[flat_idx_max]
        zm_max = Z.ravel()[flat_idx_max]
        scatter3d = cast(Any, ax.scatter)
        scatter3d([xm_min], [ym_min], zs=[zm_min], color="red", s=50, label="min")
        scatter3d([xm_max], [ym_max], zs=[zm_max], color="cyan", s=50, label="max")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        ax.view_init(elev=30, azim=-60)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_png.name, dpi=dpi)
        plt.close(fig)

        # Optionally create animation of particle trajectories
        if output_gif is not None:
            fig = plt.figure(figsize=(6, 5), dpi=dpi)
            ax = cast(Any, fig.add_subplot(111, projection="3d"))
            cast(Any, ax.plot_surface)(
                X, Y, Z, cmap=plt.colormaps["viridis"], linewidth=0, antialiased=True, alpha=0.6
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x,y)")

            # Initial scatter at t=0
            p0 = arr[0]
            try:
                z0 = objective_function(p0[..., :2])
            except Exception:
                z0 = np.array([float(objective_function(p[:2])) for p in p0], dtype=float)
            scat = cast(Any, ax.scatter)(p0[:, 0], p0[:, 1], zs=z0, color="k", s=20)

            def _update(frame):
                pts = arr[frame]
                try:
                    zs = objective_function(pts[..., :2])
                except Exception:
                    zs = np.array([float(objective_function(p[:2])) for p in pts], dtype=float)
                scat._offsets3d = (pts[:, 0], pts[:, 1], zs)  # type: ignore[attr-defined]
                ax.set_title(f"Iteration {frame + 1}/{n_iters}")
                return (scat,)

            anim = animation.FuncAnimation(
                fig, _update, frames=n_iters, interval=int(1000 / fps), blit=False
            )
            writer = PillowWriter(fps=fps)
            anim.save(output_gif.name, writer=writer)
            plt.close(fig)

    finally:
        os.chdir(original_cwd)
