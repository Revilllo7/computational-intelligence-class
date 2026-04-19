"""Experiment registry and experiment implementations."""

from src.experiments.cnn_experiment import CNNClassifierExperiment
from src.experiments.registry import ExperimentFactory, register_experiment
from src.experiments.transfer_experiment import TransferLearningExperiment

__all__ = [
    "CNNClassifierExperiment",
    "ExperimentFactory",
    "TransferLearningExperiment",
    "register_experiment",
]
