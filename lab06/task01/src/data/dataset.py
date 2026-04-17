from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.datasets import load_iris

IRIS_RENAME_MAP = {
    "sepal length (cm)": "sepal_length_cm",
    "sepal width (cm)": "sepal_width_cm",
    "petal length (cm)": "petal_length_cm",
    "petal width (cm)": "petal_width_cm",
}


@dataclass(frozen=True)
class DatasetArtifacts:
    frame: pd.DataFrame
    metadata: dict[str, object]


def build_iris_dataframe() -> DatasetArtifacts:
    dataset = cast(Any, load_iris(as_frame=True))
    frame = cast(pd.DataFrame, dataset.frame).rename(columns=IRIS_RENAME_MAP)
    frame["species"] = frame["target"].map(
        {index: name for index, name in enumerate(dataset.target_names)}
    )
    frame = frame.drop(columns=["target"])
    metadata: dict[str, object] = {
        "source": "sklearn.datasets.load_iris",
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
        "target_names": list(dataset.target_names),
        "feature_names": [IRIS_RENAME_MAP[name] for name in dataset.feature_names],
        "description": "Klasyczny zbiór Iris przygotowany jako plik CSV.",
    }
    return DatasetArtifacts(frame=frame, metadata=metadata)


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
