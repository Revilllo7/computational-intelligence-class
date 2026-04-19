from __future__ import annotations

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402


class CatsDogsCNN(nn.Module):
    def __init__(
        self,
        image_size: int,
        conv_channels: list[int],
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    self._build_activation(activation),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature_dim = int(torch.flatten(self.features(dummy), start_dim=1).shape[1])

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            self._build_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        if name == "elu":
            return nn.ELU(inplace=True)
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation function: {name}")
