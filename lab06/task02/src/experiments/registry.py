from __future__ import annotations

from typing import TypeVar

from src.experiments.base import BaseExperiment

ExperimentType = TypeVar("ExperimentType", bound=type[BaseExperiment])

_REGISTRY: dict[str, type[BaseExperiment]] = {}


def register_experiment(name: str):
    def decorator(cls: type[BaseExperiment]) -> type[BaseExperiment]:
        _REGISTRY[name] = cls
        return cls

    return decorator


class ExperimentFactory:
    @staticmethod
    def build(name: str, config) -> BaseExperiment:
        if name not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY))
            raise ValueError(f"Unknown experiment: {name}. Available: {available}")
        return _REGISTRY[name](config)

    @staticmethod
    def list() -> list[str]:
        return sorted(_REGISTRY)
