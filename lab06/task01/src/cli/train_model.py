from __future__ import annotations

import argparse

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wytrenuj model dla wybranego eksperymentu.")
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("train_model")

    experiment = ExperimentFactory.build(config.model.name, config)
    experiment.train()
    logger.info("Trening wykonany dla eksperymentu %s.", config.experiment_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
