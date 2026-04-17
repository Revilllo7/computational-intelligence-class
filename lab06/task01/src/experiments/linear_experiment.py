from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.linear import IrisLinearClassifier


@register_experiment("linear_classifier")
class LinearClassifierExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "linear_classifier"

    def build_model(self) -> IrisLinearClassifier:
        return IrisLinearClassifier(
            input_dim=len(self.config.data.feature_columns),
            num_classes=len(self.config.data.class_names),
        )
