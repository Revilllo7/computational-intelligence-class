from pathlib import Path

import pytest

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig


def test_experiment_factory_lists_expected_experiment() -> None:
    assert "cnn_classifier" in ExperimentFactory.list()
    assert "transfer_classifier" in ExperimentFactory.list()


def test_experiment_factory_raises_for_unknown_experiment() -> None:
    config = ProjectConfig.from_yaml(Path("configs/cats_dogs_quick.yaml"))
    with pytest.raises(ValueError, match="Unknown experiment"):
        ExperimentFactory.build("unknown_classifier", config)
