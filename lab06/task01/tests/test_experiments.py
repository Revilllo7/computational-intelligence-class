from src.experiments import ExperimentFactory


def test_experiment_factory_lists_expected_experiments() -> None:
    available = ExperimentFactory.list()
    assert "mlp_classifier" in available
    assert "linear_classifier" in available
