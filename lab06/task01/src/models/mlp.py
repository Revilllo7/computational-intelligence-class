from __future__ import annotations

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402


class IrisMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
