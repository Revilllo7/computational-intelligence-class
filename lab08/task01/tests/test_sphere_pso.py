"""Tests for task01 Sphere PSO implementation.

Uses a fake optimizer to exercise the alloy optimization routine quickly.
"""

from __future__ import annotations

import numpy as np

from task01.src.alloy_pso import run_sphere_demo
from task01.src.config import DEFAULT_CONFIG_PATH, load_experiment_config


class FakeOptimizer:
    def __init__(self, n_particles: int, dimensions: int, **kwargs):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.cost_history = []
        self.pos_history = []

    def optimize(self, func, iters: int = 1):
        # produce a predictable cost_history and positions
        self.cost_history = [float(i) for i in range(iters)]
        self.pos_history = np.zeros((iters, self.n_particles, self.dimensions), dtype=float)
        best_pos = np.zeros(self.dimensions, dtype=float)
        best_cost = 0.0
        return best_cost, best_pos


def test_run_sphere_demo_monkeypatched(monkeypatch, tmp_path):
    # Load config and make small for fast test
    cfg_path = DEFAULT_CONFIG_PATH.parent / "default_config.yaml"
    cfg = load_experiment_config(cfg_path)
    cfg.experiment.sphere.n_particles = 3
    cfg.experiment.sphere.iters = 2
    cfg.experiment.plotting.enabled = False
    # ensure output directory is a temporary location
    cfg = (
        cfg.__class__(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
        if hasattr(cfg, "__dataclass_fields__")
        else cfg
    )

    # Monkeypatch the PSO factory to return our FakeOptimizer
    def _fake_factory(*, n_particles, dimensions, **kwargs):
        return FakeOptimizer(n_particles=n_particles, dimensions=dimensions)

    monkeypatch.setattr("common.pso_utils.make_global_best_pso", _fake_factory)

    summary = run_sphere_demo(cfg)

    assert summary["problem"] == "sphere"
    assert isinstance(summary["best_cost"], float)
    assert isinstance(summary["best_position"], list)
    assert isinstance(summary["cost_history"], list)
    assert len(summary["cost_history"]) == cfg.experiment.sphere.iters
