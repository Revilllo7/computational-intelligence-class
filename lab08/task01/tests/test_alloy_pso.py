"""Tests for task01 Alloy PSO implementation.

Uses a fake optimizer to exercise the alloy optimization routine quickly.
"""

from __future__ import annotations

import numpy as np

from task01.src.alloy_pso import run_alloy_optimization
from task01.src.config import DEFAULT_CONFIG_PATH, load_experiment_config


class FakeOptimizer:
    def __init__(self, n_particles: int, dimensions: int, **kwargs):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.cost_history = []
        self.pos_history = []

    def optimize(self, func, iters: int = 1):
        # Return a pretend negative cost (since alloy routine minimizes -endurance)
        self.cost_history = [float(-i) for i in range(iters)]
        self.pos_history = np.zeros((iters, self.n_particles, self.dimensions), dtype=float)
        best_pos = np.full(self.dimensions, 0.5, dtype=float)
        best_cost = -1.2345
        return best_cost, best_pos


def test_run_alloy_optimization_monkeypatched(monkeypatch):
    cfg_path = DEFAULT_CONFIG_PATH.parent / "default_config.yaml"
    cfg = load_experiment_config(cfg_path)
    cfg.experiment.alloy.n_particles = 3
    cfg.experiment.alloy.iters = 2
    cfg.experiment.plotting.enabled = False

    def _fake_factory(*, n_particles, dimensions, **kwargs):
        return FakeOptimizer(n_particles=n_particles, dimensions=dimensions)

    monkeypatch.setattr("common.pso_utils.make_global_best_pso", _fake_factory)

    summary = run_alloy_optimization(cfg)

    assert summary["problem"] == "alloy"
    assert isinstance(summary["best_cost"], float)
    assert isinstance(summary["best_position"], list)
    assert "best_durability" in summary
    assert len(summary["cost_history"]) == cfg.experiment.alloy.iters
