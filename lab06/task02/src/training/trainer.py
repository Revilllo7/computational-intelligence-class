from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.utils.io import ensure_parent  # noqa: E402


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_validation_accuracy: float


@dataclass(frozen=True)
class PredictionOutput:
    true_indices: list[int]
    pred_indices: list[int]
    probabilities: list[list[float]]
    image_paths: list[str]
    loss: float
    accuracy: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    checkpoint_path: Path,
    device: torch.device,
    optimizer_name: str = "adam",
    sgd_momentum: float = 0.9,
) -> TrainingResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=sgd_momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    best_state: dict[str, torch.Tensor] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_validation_accuracy = 0.0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_targets: list[int] = []
        train_predictions: list[int] = []

        for images, labels, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_targets.extend(labels.detach().cpu().tolist())
            train_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

        validation_output = predict_with_model(model, validation_loader, device)
        train_accuracy = _accuracy(train_targets, train_predictions)
        validation_accuracy = validation_output.accuracy

        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "validation_loss": validation_output.loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    ensure_parent(checkpoint_path)
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)

    return TrainingResult(
        history=pd.DataFrame(rows),
        best_validation_accuracy=best_validation_accuracy,
    )


def predict_with_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> PredictionOutput:
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    true_indices: list[int] = []
    pred_indices: list[int] = []
    probabilities: list[list[float]] = []
    image_paths: list[str] = []

    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            losses.append(float(loss.item()))
            true_indices.extend(labels.detach().cpu().tolist())
            pred_indices.extend(preds.detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())
            image_paths.extend([str(path) for path in paths])

    return PredictionOutput(
        true_indices=true_indices,
        pred_indices=pred_indices,
        probabilities=probabilities,
        image_paths=image_paths,
        loss=float(np.mean(losses)) if losses else 0.0,
        accuracy=_accuracy(true_indices, pred_indices),
    )


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    matches = sum(int(t == p) for t, p in zip(y_true, y_pred, strict=True))
    return matches / len(y_true)
