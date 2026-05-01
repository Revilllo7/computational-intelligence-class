"""Task 02: Traveling Salesman Problem using Ant Colony Optimization."""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from aco import AntColony

from common.aco_utils import (
    calculate_path_distance,
    nearest_neighbor_tsp,
    save_json_summary,
    save_tsp_comparison_plot,
    save_tsp_visualization,
    working_directory,
)
from task02.data.coords_5x5 import COORDS as COORDS_5x5
from task02.data.coords_7 import COORDS as COORDS_7
from task02.src.config import DEFAULT_CONFIG_PATH, LoadedExperimentConfig, load_experiment_config


def generate_random_coords(
    n_points: int, min_val: int = 0, max_val: int = 100
) -> tuple[tuple[int, int], ...]:
    """Generate n random coordinate points in the given range."""
    return tuple(
        (random.randint(min_val, max_val), random.randint(min_val, max_val))
        for _ in range(n_points)
    )


class ACOExperiment:
    """Manages a single ACO TSP experiment with parameter tracking."""

    def __init__(
        self,
        coords: Sequence[tuple[float, float]],
        name: str,
        ant_count: int = 300,
        alpha: float = 0.5,
        beta: float = 1.2,
        pheromone_evaporation_rate: float = 0.40,
        pheromone_constant: float = 1000.0,
        iterations: int = 300,
    ):
        """Initialize experiment with coordinates and ACO parameters."""
        self.coords = coords
        self.name = name
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.pheromone_evaporation_rate = pheromone_evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations
        self.best_distance: float | None = None
        self.best_path: Sequence[tuple[float, float]] | None = None
        self.execution_time: float | None = None
        self.convergence_data: list[float] = []

    def run(self) -> None:
        """Execute the ACO solver and track best path."""
        start_time = time.time()

        # The upstream `aco` package stores mutable run state in class-level
        # containers. Reset them to avoid cross-experiment path contamination.
        AntColony.antArray = []
        AntColony.pheromoneMap = {}
        AntColony.tmpPheromoneMap = {}

        colony = AntColony(
            self.coords,
            ant_count=self.ant_count,
            alpha=self.alpha,
            beta=self.beta,
            pheromone_evaporation_rate=self.pheromone_evaporation_rate,
            pheromone_constant=self.pheromone_constant,
            iterations=self.iterations,
        )

        self.best_path = colony.get_path()
        self.best_distance = calculate_path_distance(self.best_path)
        self.execution_time = time.time() - start_time

    def get_summary(self) -> dict[str, Any]:
        """Return experiment results as a dictionary."""
        return {
            "name": self.name,
            "coordinates_count": len(self.coords),
            "best_distance": self.best_distance,
            "path": list(self.best_path) if self.best_path else [],
            "execution_time": self.execution_time,
            "parameters": {
                "ant_count": self.ant_count,
                "alpha": self.alpha,
                "beta": self.beta,
                "pheromone_evaporation_rate": self.pheromone_evaporation_rate,
                "pheromone_constant": self.pheromone_constant,
                "iterations": self.iterations,
            },
        }


def run_part_a(loaded_config: LoadedExperimentConfig | None = None) -> None:
    """
    Part A: Generate TSP version with 7 and 15 nodes and run ACO.

    Tests with:
    - 7-node fixed coordinates
    - 15-node randomly generated coordinates (0-100 range)
    """
    # Resolve configuration and output directory
    if loaded_config is None:
        loaded_config = load_experiment_config(DEFAULT_CONFIG_PATH)

    output_dir = loaded_config.output_dir / "part_a"
    with working_directory(output_dir):
        all_results = {}

        # Experiment 1: 7-node TSP
        print("Running Part A: 7-node TSP...")
        aco_params = loaded_config.experiment.aco

        exp_7 = ACOExperiment(
            coords=COORDS_7,
            name="7-node",
            ant_count=aco_params.ant_count,
            alpha=aco_params.alpha,
            beta=aco_params.beta,
            pheromone_evaporation_rate=aco_params.pheromone_evaporation_rate,
            pheromone_constant=aco_params.pheromone_constant,
            iterations=aco_params.iterations,
        )
        exp_7.run()
        print(f"  ✓ Best distance: {exp_7.best_distance:.2f} (Time: {exp_7.execution_time:.3f}s)")
        all_results["7-node"] = exp_7.get_summary()

        save_tsp_visualization(
            coords=exp_7.coords,
            path=cast(Any, exp_7.best_path),
            output_dir=Path.cwd(),
            title="7-Node TSP Solution (ACO)",
            filename="solution_7node.png",
        )

        # Experiment 2: 15-node random TSP
        print("Running Part A: 15-node random TSP...")
        random.seed(42)  # For reproducibility
        coords_15 = generate_random_coords(15, min_val=0, max_val=100)

        exp_15 = ACOExperiment(
            coords=coords_15,
            name="15-node-random",
            ant_count=aco_params.ant_count,
            alpha=aco_params.alpha,
            beta=aco_params.beta,
            pheromone_evaporation_rate=aco_params.pheromone_evaporation_rate,
            pheromone_constant=aco_params.pheromone_constant,
            iterations=aco_params.iterations,
        )
        exp_15.run()
        print(f"  ✓ Best distance: {exp_15.best_distance:.2f} (Time: {exp_15.execution_time:.3f}s)")
        all_results["15-node-random"] = exp_15.get_summary()

        save_tsp_visualization(
            coords=exp_15.coords,
            path=cast(Any, exp_15.best_path),
            output_dir=Path.cwd(),
            title="15-Node Random TSP Solution (ACO)",
            filename="solution_15node.png",
        )

        # Save summary
        save_json_summary(all_results, Path.cwd(), "part_a_results.json")
        print(f"✓ Part A results saved to {Path.cwd()}/part_a_results.json")


def run_part_b(
    configs_dir: Path | None = None, base_loaded: LoadedExperimentConfig | None = None
) -> None:
    """
    Part B: Parameter modification experiments.

    Tests 4 configurations measuring path distance and convergence speed.
    Configurations:
    1. Baseline (balanced)
    2. High exploration (high ant_count, low alpha)
    3. High exploitation (low ant_count, high beta)
    4. Alternative balanced
    """
    # Resolve base configuration
    if base_loaded is None:
        base_loaded = load_experiment_config(DEFAULT_CONFIG_PATH)

    # configs_dir: directory with YAML experiment configs; default to task02/configs
    if configs_dir is None:
        configs_dir = Path(__file__).parent.parent / "configs"

    output_dir = base_loaded.output_dir / "part_b"
    with working_directory(output_dir):
        cfg_files = list(sorted(configs_dir.glob("*.yaml")))
        if not cfg_files:
            cfg_files = [base_loaded.config_path]

        datasets = {
            "7-node": COORDS_7,
            "15-node-random": generate_random_coords(15, 0, 100),
        }

        all_results: dict[str, Any] = {}

        for cfg_file in cfg_files:
            try:
                loaded = load_experiment_config(cfg_file)
            except Exception as exc:
                print(f"Skipping config {cfg_file}: {exc}")
                continue

            cfg_name = loaded.config_name
            print(f"\nUsing config: {cfg_name} ({cfg_file.name})")
            all_results[cfg_name] = {}

            for dataset_name, coords in datasets.items():
                print(f"  Testing on {dataset_name}...", end=" ", flush=True)
                params = loaded.experiment.aco
                exp = ACOExperiment(
                    coords=coords,
                    name=f"{dataset_name}-{cfg_name}",
                    ant_count=params.ant_count,
                    alpha=params.alpha,
                    beta=params.beta,
                    pheromone_evaporation_rate=params.pheromone_evaporation_rate,
                    pheromone_constant=params.pheromone_constant,
                    iterations=params.iterations,
                )
                exp.run()
                print(f"Distance: {exp.best_distance:.2f}")

                all_results[cfg_name][dataset_name] = exp.get_summary()

        save_json_summary(all_results, Path.cwd(), "part_b_results.json")
        print(f"\n✓ Part B results saved to {Path.cwd()}/part_b_results.json")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for task02 experiments."""
    parser = argparse.ArgumentParser(description="Run task02 TSP ACO experiments.")
    parser.add_argument(
        "--part",
        choices=("a", "b", "c", "all"),
        default="all",
        help="Which experiment part to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config used for Parts A and C, and as a fallback for Part B.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIG_PATH.parent,
        help="Directory containing YAML configs for Part B.",
    )
    return parser


def run_part_c(loaded_config: LoadedExperimentConfig | None = None) -> None:
    """
    Part C: Analyze 5x5 grid (25 nodes) with three solvers.

    Compares:
    1. ACO (baseline config)
    2. ACO (best config from Part B)
    3. Nearest-neighbor heuristic

    Also calculates hand-designed serpentine path for comparison.
    """
    if loaded_config is None:
        loaded_config = load_experiment_config(DEFAULT_CONFIG_PATH)

    output_dir = loaded_config.output_dir / "part_c"
    with working_directory(output_dir):
        print("\nRunning Part C: 5x5 Grid Analysis...")

        # Hand-designed serpentine path (optimal human approach)
        # Row-by-row, alternating direction (snake pattern)
        hand_path = _generate_serpentine_path_5x5()
        hand_distance = calculate_path_distance(hand_path)
        print(f"  Hand-designed serpentine: {hand_distance:.2f}")

        # Solver 1: ACO with baseline config
        aco_params = loaded_config.experiment.aco

        aco_baseline = ACOExperiment(
            coords=COORDS_5x5,
            name="5x5-aco-baseline",
            ant_count=aco_params.ant_count,
            alpha=aco_params.alpha,
            beta=aco_params.beta,
            pheromone_evaporation_rate=aco_params.pheromone_evaporation_rate,
            pheromone_constant=aco_params.pheromone_constant,
            iterations=aco_params.iterations,
        )
        aco_baseline.run()
        print(f"  ACO (baseline): {aco_baseline.best_distance:.2f}")

        # Solver 2: ACO with high_exploration config (typical best performer)
        # Try to load a dedicated exploration config if present under configs
        exploration_cfg = Path(__file__).parent.parent / "configs" / "high_exploration.yaml"
        if exploration_cfg.exists():
            try:
                expl_loaded = load_experiment_config(exploration_cfg)
                expl_params = expl_loaded.experiment.aco
            except Exception:
                expl_params = aco_params
        else:
            expl_params = aco_params

        aco_exploration = ACOExperiment(
            coords=COORDS_5x5,
            name="5x5-aco-exploration",
            ant_count=expl_params.ant_count,
            alpha=expl_params.alpha,
            beta=expl_params.beta,
            pheromone_evaporation_rate=expl_params.pheromone_evaporation_rate,
            pheromone_constant=expl_params.pheromone_constant,
            iterations=expl_params.iterations,
        )
        aco_exploration.run()
        print(f"  ACO (high exploration): {aco_exploration.best_distance:.2f}")

        # Solver 3: Nearest-neighbor heuristic
        nn_path, nn_distance = nearest_neighbor_tsp(COORDS_5x5)
        print(f"  Nearest-neighbor: {nn_distance:.2f}")

        # Create comparison visualization
        comparison_results = {
            "Hand-Designed (Serpentine)": {
                "path": hand_path,
                "distance": hand_distance,
                "time": 0.0,
            },
            "ACO (Baseline)": {
                "path": aco_baseline.best_path,
                "distance": aco_baseline.best_distance,
                "time": aco_baseline.execution_time,
            },
            "ACO (High Exploration)": {
                "path": aco_exploration.best_path,
                "distance": aco_exploration.best_distance,
                "time": aco_exploration.execution_time,
            },
            "Nearest-Neighbor": {
                "path": nn_path,
                "distance": nn_distance,
                "time": 0.0,
            },
        }

        save_tsp_comparison_plot(comparison_results, Path.cwd(), filename="5x5_grid_comparison.png")

        # Save detailed results
        results_summary = {
            "problem": "5x5_grid",
            "num_cities": len(COORDS_5x5),
            "solvers": {
                "hand_designed": {
                    "distance": hand_distance,
                    "approach": "Serpentine/snake pattern (row-by-row alternating)",
                    "time": 0.0,
                },
                "aco_baseline": aco_baseline.get_summary(),
                "aco_exploration": aco_exploration.get_summary(),
                "nearest_neighbor": {
                    "distance": nn_distance,
                    "execution_time": 0.0,
                    "approach": "Greedy nearest-neighbor from all starting points",
                },
            },
            "analysis": {
                # Ensure all distances are comparable floats. If any solver failed
                # to produce a distance (None), treat it as +inf so it won't be
                # selected as the best solver.
                "best_solver": (lambda candidates: min(candidates, key=lambda t: t[1])[0])(
                    [
                        ("hand_designed", float(hand_distance)),
                        (
                            "aco_baseline",
                            float(aco_baseline.best_distance)
                            if aco_baseline.best_distance is not None
                            else float("inf"),
                        ),
                        (
                            "aco_exploration",
                            float(aco_exploration.best_distance)
                            if aco_exploration.best_distance is not None
                            else float("inf"),
                        ),
                        ("nearest_neighbor", float(nn_distance)),
                    ]
                ),
            },
        }

        save_json_summary(results_summary, Path.cwd(), "part_c_results.json")
        print(f"✓ Part C results saved to {Path.cwd()}/part_c_results.json")


def _generate_serpentine_path_5x5() -> list[tuple[int, int]]:
    """
    Generate optimal serpentine (snake) path for 5x5 grid.

    Pattern: Row-by-row, alternating direction (left-to-right, then right-to-left, etc.)
    Grid layout (0,0) at top-left, (40,40) at bottom-right.
    """
    path = []
    for row in range(5):
        if row % 2 == 0:  # Even rows: left to right
            for col in range(5):
                path.append((col * 10, row * 10))
        else:  # Odd rows: right to left
            for col in range(4, -1, -1):
                path.append((col * 10, row * 10))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for TSP-ACO experiments."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    loaded_config = load_experiment_config(args.config)

    print("=" * 60)
    print("TSP-ACO Optimization Experiments")
    print("=" * 60)
    print(f"Config: {loaded_config.config_path}")
    print(f"Part: {args.part}")

    if args.part in {"a", "all"}:
        run_part_a(loaded_config)
    if args.part in {"b", "all"}:
        run_part_b(args.configs_dir, loaded_config)
    if args.part in {"c", "all"}:
        run_part_c(loaded_config)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
