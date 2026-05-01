"""Tests for task02 TSP ACO implementation."""

from __future__ import annotations

import math
from pathlib import Path

from common.aco_utils import (
    calculate_distance,
    calculate_path_distance,
    nearest_neighbor_tsp,
    save_json_summary,
    save_tsp_comparison_plot,
    save_tsp_convergence_plot,
    save_tsp_visualization,
)
from task02.data.coords_7 import COORDS as COORDS_7
from task02.src.tsp_aco import (
    ACOExperiment,
    _generate_serpentine_path_5x5,
    generate_random_coords,
)


class TestDistanceCalculations:
    """Test distance calculation functions."""

    def test_calculate_distance_simple(self) -> None:
        """Test distance between two points."""
        # (0,0) to (3,4) should be 5
        dist = calculate_distance((0, 0), (3, 4))
        assert abs(dist - 5.0) < 1e-10

    def test_calculate_distance_same_point(self) -> None:
        """Test distance from point to itself is zero."""
        dist = calculate_distance((5, 5), (5, 5))
        assert dist == 0.0

    def test_calculate_distance_negative_coords(self) -> None:
        """Test distance works with negative coordinates."""
        dist = calculate_distance((-1, -1), (2, 3))
        expected = math.sqrt((2 - (-1)) ** 2 + (3 - (-1)) ** 2)
        assert abs(dist - expected) < 1e-10

    def test_calculate_path_distance_simple_triangle(self) -> None:
        """Test path distance for simple 3-node path."""
        path = [(0, 0), (3, 0), (3, 4)]
        # Distance: (0,0)->(3,0) = 3, (3,0)->(3,4) = 4, (3,4)->(0,0) = 5
        dist = calculate_path_distance(path)
        assert abs(dist - 12.0) < 1e-10

    def test_calculate_path_distance_single_node(self) -> None:
        """Test path distance with single node."""
        path = [(0, 0)]
        dist = calculate_path_distance(path)
        assert dist == 0.0

    def test_calculate_path_distance_two_nodes(self) -> None:
        """Test path distance with two nodes."""
        path = [(0, 0), (1, 1)]
        # Distance: (0,0)->(1,1) = sqrt(2), (1,1)->(0,0) = sqrt(2)
        dist = calculate_path_distance(path)
        expected = 2 * math.sqrt(2)
        assert abs(dist - expected) < 1e-10


class TestNearestNeighbor:
    """Test nearest-neighbor heuristic."""

    def test_nearest_neighbor_valid_path(self) -> None:
        """Test that nearest-neighbor returns a valid complete path."""
        coords = [(0, 0), (1, 1), (2, 0), (1, 2)]
        path, distance = nearest_neighbor_tsp(coords)

        # Check all nodes are visited exactly once
        assert len(path) == len(coords)
        assert set(path) == set(coords)
        assert distance > 0

    def test_nearest_neighbor_with_start_index(self) -> None:
        """Test nearest-neighbor from specific starting point."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        path, distance = nearest_neighbor_tsp(coords, start_idx=0)

        assert path[0] == (0, 0)
        assert len(path) == 4
        assert distance > 0

    def test_nearest_neighbor_single_node(self) -> None:
        """Test nearest-neighbor with single node."""
        coords = [(5, 5)]
        path, distance = nearest_neighbor_tsp(coords)

        assert path == [(5, 5)]
        assert distance == 0.0

    def test_nearest_neighbor_two_nodes(self) -> None:
        """Test nearest-neighbor with two nodes."""
        coords = [(0, 0), (1, 0)]
        path, distance = nearest_neighbor_tsp(coords)

        assert len(path) == 2
        assert set(path) == set(coords)
        expected_distance = 2.0  # Go to (1,0) and back to (0,0)
        assert abs(distance - expected_distance) < 1e-10

    def test_nearest_neighbor_all_starting_points_considered(self) -> None:
        """Test that when start_idx=None, best path is found."""
        coords = [(0, 0), (10, 0), (10, 10)]
        _path_from_none, dist_from_none = nearest_neighbor_tsp(coords, start_idx=None)
        _path_from_zero, dist_from_zero = nearest_neighbor_tsp(coords, start_idx=0)

        # When trying all starting points, we should find at least as good a solution
        # as starting from any single point
        assert dist_from_none <= dist_from_zero


class TestCoordinateGeneration:
    """Test random coordinate generation."""

    def test_generate_random_coords_count(self) -> None:
        """Test that correct number of coordinates are generated."""
        n = 15
        coords = generate_random_coords(n)
        assert len(coords) == n

    def test_generate_random_coords_range(self) -> None:
        """Test that coordinates are within specified range."""
        coords = generate_random_coords(100, min_val=0, max_val=100)
        for x, y in coords:
            assert 0 <= x <= 100
            assert 0 <= y <= 100

    def test_generate_random_coords_reproducibility(self) -> None:
        """Test that same seed produces same coordinates."""
        import random

        random.seed(42)
        coords1 = generate_random_coords(10)

        random.seed(42)
        coords2 = generate_random_coords(10)

        assert coords1 == coords2


class TestSerpentinePath:
    """Test serpentine path generation for 5x5 grid."""

    def test_serpentine_path_length(self) -> None:
        """Test that serpentine path visits all 25 nodes."""
        path = _generate_serpentine_path_5x5()
        assert len(path) == 25

    def test_serpentine_path_all_nodes(self) -> None:
        """Test that all grid nodes are visited."""
        path = _generate_serpentine_path_5x5()
        grid_nodes = {(x, y) for x in range(0, 50, 10) for y in range(0, 50, 10)}
        visited = set(path)
        assert visited == grid_nodes

    def test_serpentine_path_is_valid_cycle(self) -> None:
        """Test that serpentine path forms a valid cycle."""
        path = _generate_serpentine_path_5x5()
        # No duplicate visits (except start=end implicit)
        assert len(set(path)) == len(path)

    def test_serpentine_path_distance(self) -> None:
        """Test that serpentine path distance is reasonable."""
        path = _generate_serpentine_path_5x5()
        distance = calculate_path_distance(path)
        # Grid is 40x40, serpentine should be roughly 4*40 + 24*10 = 320
        # Actual optimal serpentine for 5x5 is ~297
        assert 290 < distance < 310


class TestACOExperiment:
    """Test ACO experiment class."""

    def test_aco_experiment_creation(self) -> None:
        """Test creating an ACO experiment."""
        exp = ACOExperiment(
            coords=COORDS_7,
            name="test",
            ant_count=100,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=1000.0,
            iterations=50,
        )
        assert exp.name == "test"
        assert exp.ant_count == 100
        assert len(exp.coords) == 7

    def test_aco_experiment_run(self) -> None:
        """Test running an ACO experiment (with small parameters for speed)."""
        exp = ACOExperiment(
            coords=[(0, 0), (1, 1), (2, 0), (1, 2)],
            name="small_test",
            ant_count=50,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=100.0,
            iterations=10,
        )
        exp.run()

        # Verify results
        assert exp.best_distance is not None
        assert exp.best_distance > 0
        assert exp.best_path is not None
        # AntColony returns path with start node repeated at end (cycle closure)
        assert len(exp.best_path) == 5 or len(exp.best_path) == 4
        assert exp.execution_time is not None
        assert exp.execution_time > 0

    def test_aco_experiment_summary(self) -> None:
        """Test getting experiment summary."""
        exp = ACOExperiment(
            coords=COORDS_7,
            name="test",
            ant_count=100,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=1000.0,
            iterations=50,
        )
        exp.run()

        summary = exp.get_summary()
        assert "name" in summary
        assert "best_distance" in summary
        assert "path" in summary
        assert "execution_time" in summary
        assert "parameters" in summary
        assert summary["coordinates_count"] == 7

    def test_sequential_experiments_do_not_reuse_previous_path(self) -> None:
        """Sequential runs should not leak path state across datasets."""
        exp_small = ACOExperiment(
            coords=[(0, 0), (1, 0), (1, 1), (0, 1)],
            name="small",
            ant_count=30,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=100.0,
            iterations=8,
        )
        exp_small.run()

        exp_large = ACOExperiment(
            coords=[(0, 0), (2, 0), (4, 1), (6, 2), (8, 0), (10, 1)],
            name="large",
            ant_count=30,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=100.0,
            iterations=8,
        )
        exp_large.run()

        assert exp_large.best_path is not None
        large_path_nodes = set(exp_large.best_path)
        assert large_path_nodes.issubset(set(exp_large.coords))
        assert len(large_path_nodes) == len(exp_large.coords)


class TestVisualizationAndIO:
    """Test visualization and file I/O functions."""

    def test_save_json_summary(self, tmp_path: Path) -> None:
        """Test saving JSON summary."""
        data = {"test": "data", "value": 42}
        save_json_summary(data, tmp_path, "test.json")

        output_file = tmp_path / "test.json"
        assert output_file.exists()

        import json

        with Path.open(output_file) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_tsp_visualization(self, tmp_path: Path) -> None:
        """Test saving TSP visualization plot."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        path = coords
        save_tsp_visualization(coords, path, tmp_path, filename="test.png")

        output_file = tmp_path / "test.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_save_tsp_comparison_plot(self, tmp_path: Path) -> None:
        """Test saving TSP comparison plot."""
        coords = [(0, 0), (1, 0), (1, 1)]
        results = {
            "method1": {"path": coords, "distance": 4.0, "time": 0.1},
            "method2": {"path": list(reversed(coords)), "distance": 4.0, "time": 0.2},
        }
        save_tsp_comparison_plot(results, tmp_path, "comparison.png")

        output_file = tmp_path / "comparison.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_save_convergence_plot(self, tmp_path: Path) -> None:
        """Test saving convergence comparison plot."""
        convergence_data = {
            "method1": [100, 90, 85, 82, 80],
            "method2": [100, 95, 92, 91, 90],
        }
        save_tsp_convergence_plot(convergence_data, tmp_path, "convergence.png")

        output_file = tmp_path / "convergence.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0
