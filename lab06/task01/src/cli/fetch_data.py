from __future__ import annotations

import argparse

from src.data.dataset import build_iris_dataframe
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, write_json
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pobierz i zapisz dane Iris do CSV.")
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("fetch_data")

    artifacts = build_iris_dataframe()
    ensure_parent(config.paths.raw_csv)
    artifacts.frame.to_csv(config.paths.raw_csv, index=False)
    write_json(config.paths.dataset_metadata, artifacts.metadata)

    logger.info("Zapisano dane surowe do %s", config.paths.raw_csv)
    logger.info("Zapisano metadane do %s", config.paths.dataset_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
