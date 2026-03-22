from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .classification import build_confusion_matrix


def fit_standard_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fit column-wise mean and std on training features."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def transform_standard_scaler(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    # Apply pre-fitted standardization parameters to features.
    return (features - mean) / std


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    # Train one epoch and return average loss.
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(dataloader))


def predict_from_loader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    # Return concatenated true and predicted labels from data loader.
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_true.append(targets.detach().cpu().numpy())
            all_pred.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return y_true, y_pred


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    positive_label: Any | None = None,
) -> dict[str, Any]:
    # Compute reusable classification metrics from predictions.
    result: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "correct": int((y_true == y_pred).sum()),
        "wrong": int((y_true != y_pred).sum()),
        "confusion_df": build_confusion_matrix(pd.Series(y_true), pd.Series(y_pred), labels),
    }

    if len(labels) == 2:
        pos_label = positive_label if positive_label is not None else labels[-1]
        result["precision_binary"] = precision_score(
            y_true,
            y_pred,
            average="binary",
            pos_label=pos_label,
            zero_division=0,
        )
        result["recall_binary"] = recall_score(
            y_true,
            y_pred,
            average="binary",
            pos_label=pos_label,
            zero_division=0,
        )

    result["precision_weighted"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    result["recall_weighted"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return result
