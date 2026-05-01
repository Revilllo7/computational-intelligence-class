"""Task 01: Alloy durability optimization with Particle Swarm Optimization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from common.pso_utils import (
    make_global_best_pso,
    remove_report_log,
    save_3d_surface_and_animation,
    save_cost_history_plot,
    working_directory,
)
from task01.src.config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config

GENE_ORDER: tuple[str, ...] = ("x", "y", "z", "v", "u", "w")
NUM_GENES = len(GENE_ORDER)


def sanitize_solution(solution: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert a raw chromosome into a finite float vector."""
    genes = np.asarray(solution, dtype=float)
    genes = np.nan_to_num(genes, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(genes, 0.0, 1.0)


def evaluate_endurance(solution: np.ndarray | Sequence[float]) -> float:
    """Evaluate the alloy endurance objective for one chromosome."""
    x, y, z, v, u, w = sanitize_solution(solution)
    value = np.exp(-2.0 * (y - np.sin(x)) ** 2) + np.sin(z * u) + np.cos(v * w)
    return float(np.nan_to_num(value, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf))


def evaluate_endurance_swarm(swarm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate the alloy endurance for an entire swarm."""
    particles = np.asarray(swarm, dtype=float)
    if particles.ndim == 1:
        particles = particles.reshape(1, -1)
    return np.array([evaluate_endurance(particle) for particle in particles], dtype=float)


def minimize_sphere_swarm(swarm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return sphere costs for a swarm, shaped for pyswarms."""
    particles = np.asarray(swarm, dtype=float)
    return np.sum(np.square(particles), axis=1)


def maximize_endurance_swarm(swarm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return negative endurance so pyswarms can minimize the objective."""
    return -evaluate_endurance_swarm(swarm)


def run_sphere_demo(config: LoadedExperimentConfig) -> dict[str, Any]:
    """Run the teaching sphere minimization example from the task statement."""
    sphere_config = config.experiment.sphere
    options = config.experiment.pso_options.model_dump()
    with working_directory(config.output_dir):
        optimizer = make_global_best_pso(
            n_particles=sphere_config.n_particles,
            dimensions=sphere_config.dimensions,
            options=options,
            lower_bound=sphere_config.lower_bound,
            upper_bound=sphere_config.upper_bound,
        )
        best_cost, best_pos = optimizer.optimize(minimize_sphere_swarm, iters=sphere_config.iters)
    summary = {
        "problem": "sphere",
        "best_cost": float(best_cost),
        "best_position": [float(value) for value in best_pos],
        "cost_history": [float(value) for value in optimizer.cost_history],
    }

    if config.experiment.plotting.enabled and config.experiment.plotting.save_cost_plot:
        save_cost_history_plot(
            optimizer.cost_history,
            config.output_dir / "plots" / "sphere_cost_history.png",
            title="Sphere PSO Cost History",
            dpi=config.experiment.plotting.figure_dpi,
        )

    # 3D surface + animation (project to first two dims)
    if config.experiment.plotting.enabled and config.experiment.plotting.save_3d_plot:
        pos_history = np.asarray(getattr(optimizer, "pos_history", []), dtype=float)
        if pos_history.size:
            save_3d_surface_and_animation(
                pos_history,
                minimize_sphere_swarm,
                config.output_dir / "plots" / "sphere_surface.png",
                config.output_dir / "plots" / "sphere_trajectory.gif",
                dpi=config.experiment.plotting.figure_dpi,
                fps=10,
            )

    return summary


def _active_gene_names(solution: np.ndarray | Sequence[float], threshold: float) -> list[str]:
    genes = sanitize_solution(solution)
    return [name for name, value in zip(GENE_ORDER, genes, strict=True) if value > threshold]


def run_alloy_optimization(config: LoadedExperimentConfig) -> dict[str, Any]:
    """Optimize the alloy endurance objective by minimizing its negative value."""
    alloy_config = config.experiment.alloy
    options = config.experiment.pso_options.model_dump()
    with working_directory(config.output_dir):
        optimizer = make_global_best_pso(
            n_particles=alloy_config.n_particles,
            dimensions=alloy_config.dimensions,
            options=options,
            lower_bound=alloy_config.lower_bound,
            upper_bound=alloy_config.upper_bound,
        )
        best_cost, best_pos = optimizer.optimize(
            maximize_endurance_swarm,
            iters=alloy_config.iters,
        )

    best_durability = -float(best_cost)
    best_position = sanitize_solution(best_pos)
    best_gene_values = {
        name: float(value) for name, value in zip(GENE_ORDER, best_position, strict=True)
    }
    active_gene_names = _active_gene_names(best_position, alloy_config.active_metal_threshold)

    summary = {
        "problem": "alloy",
        "gene_order": list(GENE_ORDER),
        "best_cost": float(best_cost),
        "best_durability": best_durability,
        "best_position": [float(value) for value in best_position],
        "best_gene_values": best_gene_values,
        "active_gene_threshold": float(alloy_config.active_metal_threshold),
        "active_gene_names": active_gene_names,
        "active_metal_count": len(active_gene_names),
        "cost_history": [float(value) for value in optimizer.cost_history],
    }

    if config.experiment.plotting.enabled and config.experiment.plotting.save_cost_plot:
        save_cost_history_plot(
            optimizer.cost_history,
            config.output_dir / "plots" / "alloy_cost_history.png",
            title="Alloy PSO Cost History (Minimizing Negative Endurance)",
            dpi=config.experiment.plotting.figure_dpi,
        )

    # 3D surface + animation for alloy (project to first two genes)
    if config.experiment.plotting.enabled and config.experiment.plotting.save_3d_plot:
        pos_history = np.asarray(getattr(optimizer, "pos_history", []), dtype=float)
        if pos_history.size:

            def _alloy_proj(arr: np.ndarray) -> np.ndarray:
                a = np.asarray(arr, dtype=float)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                if a.shape[1] == 2:
                    # pad remaining dimensions with 0.5 (midpoint)
                    pad = np.full((a.shape[0], alloy_config.dimensions - 2), 0.5, dtype=float)
                    a = np.hstack((a, pad))
                return evaluate_endurance_swarm(a)

            save_3d_surface_and_animation(
                pos_history,
                _alloy_proj,
                config.output_dir / "plots" / "alloy_surface.png",
                config.output_dir / "plots" / "alloy_trajectory.gif",
                dpi=config.experiment.plotting.figure_dpi,
                fps=10,
            )

    return summary


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 01 alloy durability PSO")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)

    sphere_summary = run_sphere_demo(config)
    alloy_summary = run_alloy_optimization(config)

    output_dir = config.output_dir
    _write_json(output_dir / "sphere_summary.json", sphere_summary)
    _write_json(output_dir / "alloy_summary.json", alloy_summary)
    remove_report_log()

    print(
        "Sphere best cost="
        f"{sphere_summary['best_cost']:.6f}, best pos={sphere_summary['best_position']}"
    )
    print(
        "Alloy best cost="
        f"{alloy_summary['best_cost']:.6f}, best pos={alloy_summary['best_position']}"
    )
    print(f"Alloy best durability={alloy_summary['best_durability']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
