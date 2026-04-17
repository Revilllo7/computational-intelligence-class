import pandas as pd
import pytest

from src.training.trainer import dataframe_to_loader


def test_dataframe_to_loader_uses_target_column_with_numeric_labels() -> None:
    frame = pd.DataFrame(
        {
            "param1": [0.1, 0.2, 0.3],
            "param2": [1.1, 1.2, 1.3],
            "param3": [2.1, 2.2, 2.3],
            "diagnosis": [0, 1, 0],
        }
    )

    loader = dataframe_to_loader(
        frame=frame,
        feature_columns=["param1", "param2", "param3"],
        class_names=["0", "1"],
        target_column="diagnosis",
        batch_size=3,
        shuffle=False,
    )

    _, labels = next(iter(loader))
    assert labels.tolist() == [0, 1, 0]


def test_dataframe_to_loader_raises_on_unknown_label() -> None:
    frame = pd.DataFrame(
        {
            "param1": [0.1, 0.2],
            "param2": [1.1, 1.2],
            "param3": [2.1, 2.2],
            "diagnosis": [0, 3],
        }
    )

    with pytest.raises(ValueError, match="Unknown class labels"):
        dataframe_to_loader(
            frame=frame,
            feature_columns=["param1", "param2", "param3"],
            class_names=["0", "1"],
            target_column="diagnosis",
            batch_size=2,
            shuffle=False,
        )
