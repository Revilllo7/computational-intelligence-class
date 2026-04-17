from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.utils.io import ensure_parent  # noqa: E402


@dataclass(frozen=True)
class TrainingBundle:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    feature_columns: list[str]
    class_names: list[str]


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    best_validation_accuracy: float


def dataframe_to_loader(
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    mapping = {name: index for index, name in enumerate(class_names)}
    features = torch.tensor(frame[feature_columns].to_numpy(dtype=np.float32))
    labels = torch.tensor(frame["species"].map(mapping).to_numpy(dtype=np.int64))
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    bundle: TrainingBundle,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    checkpoint_path: Path,
) -> TrainingResult:
    train_loader = dataframe_to_loader(
        bundle.train_frame,
        bundle.feature_columns,
        bundle.class_names,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = dataframe_to_loader(
        bundle.validation_frame,
        bundle.feature_columns,
        bundle.class_names,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    ensure_parent(checkpoint_path)
    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_predictions: list[int] = []
        train_targets: list[int] = []

        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_predictions.extend(torch.argmax(logits, dim=1).tolist())
            train_targets.extend(labels.tolist())

        model.eval()
        validation_losses: list[float] = []
        validation_predictions: list[int] = []
        validation_targets: list[int] = []
        with torch.no_grad():
            for features, labels in validation_loader:
                logits = model(features)
                loss = criterion(logits, labels)
                validation_losses.append(float(loss.item()))
                validation_predictions.extend(torch.argmax(logits, dim=1).tolist())
                validation_targets.extend(labels.tolist())

        train_accuracy = accuracy_score(train_targets, train_predictions)
        validation_accuracy = accuracy_score(validation_targets, validation_predictions)
        mean_train_loss = float(np.mean(train_losses))
        mean_validation_loss = float(np.mean(validation_losses))

        rows.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "validation_loss": mean_validation_loss,
                "train_accuracy": float(train_accuracy),
                "validation_accuracy": float(validation_accuracy),
            }
        )

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = float(validation_accuracy)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_validation_accuracy=best_validation_accuracy,
    )


def predict_with_model(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
) -> tuple[list[str], list[list[float]]]:
    features = torch.tensor(frame[feature_columns].to_numpy(dtype=np.float32))
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probabilities_tensor = torch.softmax(logits, dim=1)
        prediction_indices = cast(list[int], torch.argmax(logits, dim=1).tolist())
        probabilities = cast(list[list[float]], probabilities_tensor.tolist())
    predictions = [class_names[index] for index in prediction_indices]
    return predictions, probabilities
