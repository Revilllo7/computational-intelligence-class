from __future__ import annotations

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402


class TransferLearningClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        freeze_backbone: bool,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.model = self._build_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            head_dropout=head_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    @staticmethod
    def _build_model(
        backbone: str,
        num_classes: int,
        pretrained: bool,
        freeze_backbone: bool,
        head_dropout: float,
    ) -> nn.Module:
        head_parameters: list[nn.Parameter]

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            head_parameters = list(model.fc.parameters())

        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            classifier_last = model.classifier[-1]
            if not isinstance(classifier_last, nn.Linear):
                raise ValueError("Unexpected MobileNet classifier layout.")
            model.classifier[-1] = nn.Linear(classifier_last.in_features, num_classes)
            dropout_layer = model.classifier[-2]
            if isinstance(dropout_layer, nn.Dropout):
                dropout_layer.p = head_dropout
            head_parameters = list(model.classifier.parameters())

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            classifier_last = model.classifier[-1]
            if not isinstance(classifier_last, nn.Linear):
                raise ValueError("Unexpected EfficientNet classifier layout.")
            model.classifier = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(classifier_last.in_features, num_classes),
            )
            head_parameters = list(model.classifier.parameters())

        else:
            raise ValueError(f"Unsupported transfer-learning backbone: {backbone}")

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in head_parameters:
                parameter.requires_grad = True

        return model
