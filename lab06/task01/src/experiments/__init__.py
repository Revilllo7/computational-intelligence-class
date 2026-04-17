"""Experiment registry and experiment implementations."""

from src.experiments.linear_experiment import LinearClassifierExperiment
from src.experiments.mlp_experiment import MLPClassifierExperiment
from src.experiments.registry import ExperimentFactory, register_experiment
from src.experiments.shallow_experiment import ShallowMLPClassifierExperiment

__all__ = [
    "ExperimentFactory",
    "LinearClassifierExperiment",
    "MLPClassifierExperiment",
    "ShallowMLPClassifierExperiment",
    "register_experiment",
]
