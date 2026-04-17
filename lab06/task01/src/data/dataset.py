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


def build_csv_metadata(
    frame: pd.DataFrame,
    source: str,
    target_column: str,
    feature_columns: list[str],
) -> dict[str, object]:
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the provided dataset.")

    return {
        "source": source,
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
        "target_column": target_column,
        "target_names": sorted(frame[target_column].astype(str).unique().tolist()),
        "feature_names": feature_columns,
        "description": "Dataset loaded from CSV file.",
    }


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
