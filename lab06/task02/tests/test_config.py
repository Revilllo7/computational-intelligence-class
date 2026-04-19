from pathlib import Path

import pytest

from src.utils.config import ComparisonConfig, ProjectConfig


def test_project_config_loads_quick_yaml() -> None:
    config = ProjectConfig.from_yaml(Path("configs/cats_dogs_quick.yaml"))
    assert config.project_name == "cats-dogs-cnn-template"
    assert config.experiment_name == "cats_dogs_quick"
    assert config.paths.raw_zip == Path("data/raw/dogs-and-cats.zip")
    assert config.preprocessing.image_size == 128
    assert config.data.class_names == ["cat", "dog"]
    assert config.model.activation == "relu"
    assert config.training.optimizer == "adam"


def test_comparison_config_loads_yaml() -> None:
    config = ComparisonConfig.from_yaml(Path("configs/compare_experiments_matrix.yaml"))
    assert config.project_name == "cats-dogs-cnn-template"
    assert len(config.experiments) == 7
    assert config.comparison_csv == Path("reports/comparisons/experiment_matrix.csv")


def test_project_config_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml(Path("configs/not_existing.yaml"))
