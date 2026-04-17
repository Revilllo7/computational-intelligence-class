from pathlib import Path

import pytest
from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig


def test_experiment_factory_lists_expected_experiments() -> None:
    available = ExperimentFactory.list()
    assert "mlp_classifier" in available
    assert "linear_classifier" in available
    assert "shallow_mlp_classifier" in available


def test_experiment_factory_raises_for_unknown_experiment() -> None:
    config = ProjectConfig.from_yaml(Path("configs/iris_mlp_zscore.yaml"))
    with pytest.raises(ValueError, match="Nieznany eksperyment"):
        ExperimentFactory.build("unknown_classifier", config)
