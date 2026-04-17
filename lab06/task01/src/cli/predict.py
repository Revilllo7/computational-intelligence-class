from __future__ import annotations

import argparse

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wykonaj pojedyncza predykcje dla probki Iris.")
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("predict")

    raw_values = {
        "sepal_length_cm": args.sepal_length,
        "sepal_width_cm": args.sepal_width,
        "petal_length_cm": args.petal_length,
        "petal_width_cm": args.petal_width,
    }
    experiment = ExperimentFactory.build(config.model.name, config)
    label, probabilities = experiment.predict_one(raw_values)

    logger.info("Predykcja dla eksperymentu %s: %s", config.experiment_name, label)
    for class_name, probability in zip(config.data.class_names, probabilities, strict=True):
        logger.info("%s -> %.4f", class_name, probability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
