import pandas as pd

from src.data.preprocessing import (
    apply_minmax_scaler,
    apply_standardizer,
    fit_minmax_scaler,
    fit_standardizer,
    split_dataset,
)


def test_split_dataset_preserves_all_rows() -> None:
    frame = pd.DataFrame(
        {
            "sepal_length_cm": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                5.0,
                5.1,
                5.2,
                5.3,
                5.4,
                7.0,
                7.1,
                7.2,
                7.3,
                7.4,
            ],
            "sepal_width_cm": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                3.0,
                3.1,
                3.2,
                3.3,
                3.4,
            ],
            "petal_length_cm": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                4.0,
                4.1,
                4.2,
                4.3,
                4.4,
                6.0,
                6.1,
                6.2,
                6.3,
                6.4,
            ],
            "petal_width_cm": [
                0.1,
                0.1,
                0.2,
                0.2,
                0.3,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ],
            "species": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
                "virginica",
            ],
        }
    )
    split = split_dataset(
        frame=frame,
        target_column="species",
        test_size=0.2,
        validation_size=0.2,
        random_seed=42,
    )
    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)


def test_standardizer_changes_feature_scale() -> None:
    frame = pd.DataFrame(
        {
            "sepal_length_cm": [1.0, 2.0, 3.0],
            "sepal_width_cm": [2.0, 3.0, 4.0],
            "petal_length_cm": [3.0, 4.0, 5.0],
            "petal_width_cm": [4.0, 5.0, 6.0],
            "species": ["setosa", "versicolor", "virginica"],
        }
    )
    columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    stats = fit_standardizer(frame, columns)
    transformed = apply_standardizer(frame, columns, stats)
    assert abs(float(transformed["sepal_length_cm"].mean())) < 1e-8


def test_minmax_scaler_maps_values_to_zero_one_range() -> None:
    frame = pd.DataFrame(
        {
            "sepal_length_cm": [1.0, 2.0, 3.0],
            "sepal_width_cm": [2.0, 3.0, 4.0],
            "petal_length_cm": [3.0, 4.0, 5.0],
            "petal_width_cm": [4.0, 5.0, 6.0],
            "species": ["setosa", "versicolor", "virginica"],
        }
    )
    columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    stats = fit_minmax_scaler(frame, columns)
    transformed = apply_minmax_scaler(frame, columns, stats)
    assert float(transformed["sepal_length_cm"].min()) == 0.0
    assert float(transformed["sepal_length_cm"].max()) == 1.0
