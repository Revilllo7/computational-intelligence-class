from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


PreprocessorStats = dict[str, dict[str, float]]


def split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float,
    validation_size: float,
    random_seed: int,
) -> SplitResult:
    split_one = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[target_column],
        random_state=random_seed,
    )
    train_validation = cast(pd.DataFrame, split_one[0])
    test = cast(pd.DataFrame, split_one[1])
    validation_fraction = validation_size / (1.0 - test_size)
    split_two = train_test_split(
        train_validation,
        test_size=validation_fraction,
        stratify=train_validation[target_column],
        random_state=random_seed,
    )
    train = cast(pd.DataFrame, split_two[0])
    validation = cast(pd.DataFrame, split_two[1])
    return SplitResult(train=train, validation=validation, test=test)


def fit_standardizer(frame: pd.DataFrame, feature_columns: list[str]) -> PreprocessorStats:
    stats: PreprocessorStats = {}
    for column in feature_columns:
        mean = float(frame[column].mean())
        std = float(frame[column].std())
        stats[column] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def fit_minmax_scaler(frame: pd.DataFrame, feature_columns: list[str]) -> PreprocessorStats:
    stats: PreprocessorStats = {}
    for column in feature_columns:
        minimum = float(frame[column].min())
        maximum = float(frame[column].max())
        if minimum == maximum:
            maximum = minimum + 1.0
        stats[column] = {"min": minimum, "max": maximum}
    return stats


def apply_standardizer(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: PreprocessorStats,
) -> pd.DataFrame:
    transformed = frame.copy()
    for column in feature_columns:
        transformed[column] = (transformed[column] - stats[column]["mean"]) / stats[column]["std"]
    return transformed


def apply_minmax_scaler(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: PreprocessorStats,
) -> pd.DataFrame:
    transformed = frame.copy()
    for column in feature_columns:
        denominator = stats[column]["max"] - stats[column]["min"]
        transformed[column] = (transformed[column] - stats[column]["min"]) / denominator
    return transformed


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    strategy: Literal["zscore", "minmax"],
) -> PreprocessorStats:
    if strategy == "zscore":
        return fit_standardizer(frame, feature_columns)
    return fit_minmax_scaler(frame, feature_columns)


def apply_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: PreprocessorStats,
    strategy: Literal["zscore", "minmax"],
) -> pd.DataFrame:
    if strategy == "zscore":
        return apply_standardizer(frame, feature_columns, stats)
    return apply_minmax_scaler(frame, feature_columns, stats)
