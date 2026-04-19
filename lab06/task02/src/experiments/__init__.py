"""Experiment registry and experiment implementations."""

from src.experiments.cnn_experiment import CNNClassifierExperiment
from src.experiments.registry import ExperimentFactory, register_experiment

__all__ = [
    "CNNClassifierExperiment",
    "ExperimentFactory",
    "register_experiment",
]
