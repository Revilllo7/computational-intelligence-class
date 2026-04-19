from __future__ import annotations

import argparse

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN model for selected experiment config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("train_model")

    experiment = ExperimentFactory.build(config.model.name, config)
    experiment.train()

    logger.info("Training completed for %s.", config.experiment_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
