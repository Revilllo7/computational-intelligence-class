from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.shallow_mlp import IrisShallowMLP


@register_experiment("shallow_mlp_classifier")
class ShallowMLPClassifierExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "shallow_mlp_classifier"

    def build_model(self) -> IrisShallowMLP:
        hidden_dim = self.config.training.hidden_dims[0] if self.config.training.hidden_dims else 8
        return IrisShallowMLP(
            input_dim=len(self.config.data.feature_columns),
            hidden_dim=hidden_dim,
            num_classes=len(self.config.data.class_names),
        )
