"""Tests for task01 knapsack GA workflow."""

from __future__ import annotations

import numpy as np

from task01.src.knapsack_ga import (
    CAPACITY,
    TARGET_VALUE,
    create_ga_instance,
    evaluate_solution,
    knapsack_fitness,
)


def test_overweight_penalty_is_strongly_negative() -> None:
    """Overweight solutions should be punished below zero in this setup."""
    overweight_solution = np.ones(11, dtype=int)
    ga = create_ga_instance(random_seed=0)
    fitness = knapsack_fitness(ga, overweight_solution, 0)
    assert fitness < 0


def test_known_optimal_feasible_set_hits_target() -> None:
    """Known best feasible set: item2 + item3 + item5 + item7 + item8 + item10."""
    chromosome = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
    total_value, total_weight = evaluate_solution(chromosome)

    assert total_weight == CAPACITY
    assert total_value == TARGET_VALUE


def test_ga_factory_returns_independent_instances() -> None:
    """Factory must return fresh GA instances for repeated experiments."""
    ga_a = create_ga_instance(random_seed=101)
    ga_b = create_ga_instance(random_seed=202)

    assert ga_a is not ga_b
    assert ga_a.random_seed != ga_b.random_seed
