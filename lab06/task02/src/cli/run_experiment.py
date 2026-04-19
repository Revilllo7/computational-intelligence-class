from __future__ import annotations

import argparse

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full end-to-end CNN experiment pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("run_experiment")

    experiment = ExperimentFactory.build(config.model.name, config)
    experiment.run()

    logger.info("Full run completed for %s.", config.experiment_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
