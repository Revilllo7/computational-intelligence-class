from pathlib import Path

import pytest
from pydantic import ValidationError
from src.utils.config import ProjectConfig


def test_project_config_loads_yaml() -> None:
    config = ProjectConfig.from_yaml(Path("configs/iris_mlp_zscore.yaml"))
    assert config.project_name == "iris-classification-template"
    assert config.experiment_name == "iris_mlp_zscore"
    assert config.paths.raw_csv == Path("data/raw/iris.csv")
    assert len(config.data.feature_columns) == 4


def test_project_config_rejects_invalid_preprocessing_strategy() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(Path("configs/faulty_linear_maxabs.yaml"))


def test_project_config_rejects_invalid_model_name() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(Path("configs/faulty_additive_minmax.yaml"))


def test_project_config_raises_for_missing_config_file() -> None:
    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml(Path("configs/not_existing.yaml"))
