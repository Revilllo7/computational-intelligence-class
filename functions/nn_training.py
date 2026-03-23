from __future__ import annotations

import torch


def loader_average_loss(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    # Evaluate average loss over a dataloader without gradient updates.
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            total_loss += criterion(outputs, targets).item()

    return total_loss / max(1, len(dataloader))