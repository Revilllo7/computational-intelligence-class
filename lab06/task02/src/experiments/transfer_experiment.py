from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.transfer import TransferLearningClassifier


@register_experiment("transfer_classifier")
class TransferLearningExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "transfer_classifier"

    def build_model(self) -> TransferLearningClassifier:
        backbone = self.config.model.backbone
        if backbone is None:
            raise ValueError("Transfer experiment requires model.backbone in config.")

        return TransferLearningClassifier(
            backbone=backbone,
            num_classes=len(self.config.data.class_names),
            pretrained=self.config.model.pretrained,
            freeze_backbone=self.config.model.freeze_backbone,
            head_dropout=self.config.model.head_dropout,
        )
