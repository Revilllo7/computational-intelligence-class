from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict[Any, int], dict[int, Any]]:
    # Encode arbitrary labels into contiguous integer IDs.
    classes = sorted(labels.unique().tolist())
    to_index = {label: idx for idx, label in enumerate(classes)}
    to_label = {idx: label for label, idx in to_index.items()}
    encoded = labels.map(to_index).to_numpy(dtype=np.int64)
    return encoded, to_index, to_label


def make_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    # Create TensorDataset-backed DataLoader from numpy arrays.
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(feature_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
