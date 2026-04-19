from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.cnn import CatsDogsCNN


@register_experiment("cnn_classifier")
class CNNClassifierExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "cnn_classifier"

    def build_model(self) -> CatsDogsCNN:
        return CatsDogsCNN(
            image_size=self.config.preprocessing.image_size,
            conv_channels=self.config.model.conv_channels,
            hidden_dim=self.config.model.hidden_dim,
            num_classes=len(self.config.data.class_names),
            dropout=self.config.model.dropout,
            activation=self.config.model.activation,
        )
