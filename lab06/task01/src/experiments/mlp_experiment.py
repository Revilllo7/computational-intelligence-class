from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.mlp import IrisMLP


@register_experiment("mlp_classifier")
class MLPClassifierExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "mlp_classifier"

    def build_model(self) -> IrisMLP:
        return IrisMLP(
            input_dim=len(self.config.data.feature_columns),
            hidden_dims=self.config.training.hidden_dims,
            num_classes=len(self.config.data.class_names),
        )
