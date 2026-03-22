from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 292583,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Split data into train/test sets and return both full and feature/target splits.
    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    x_train = train_set[feature_columns].copy()
    x_test = test_set[feature_columns].copy()
    y_train = train_set[target_column].copy()
    y_test = test_set[target_column].copy()
    return train_set, test_set, x_train, x_test, y_train, y_test


def build_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[Any],
) -> pd.DataFrame:
    # Build confusion matrix as a labeled DataFrame.
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def evaluate_classifier(
    classifier: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    # Fit classifier and return standard metrics plus confusion matrix.
    classifier.fit(x_train, y_train)
    y_pred = pd.Series(classifier.predict(x_test), index=y_test.index, name="predicted")

    if labels is None:
        labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))

    result: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "correct": int((y_pred == y_test).sum()),
        "wrong": int((y_pred != y_test).sum()),
        "confusion_df": build_confusion_matrix(y_test, y_pred, labels),
        "y_pred": y_pred,
    }

    if len(labels) == 2:
        binary_label = positive_label if positive_label is not None else labels[-1]
        result["precision_binary"] = precision_score(
            y_test,
            y_pred,
            average="binary",
            pos_label=binary_label,
            zero_division=0,
        )
        result["recall_binary"] = recall_score(
            y_test,
            y_pred,
            average="binary",
            pos_label=binary_label,
            zero_division=0,
        )

    result["precision_weighted"] = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    result["recall_weighted"] = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return result
