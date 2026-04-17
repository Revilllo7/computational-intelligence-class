from __future__ import annotations

import argparse

from src.data.dataset import build_csv_metadata, build_iris_dataframe, read_dataset
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, write_json
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pobierz lub odczytaj dane surowe wskazane w konfiguracji."
    )
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("fetch_data")

    if config.paths.raw_csv.exists():
        frame = read_dataset(config.paths.raw_csv)
        metadata = build_csv_metadata(
            frame=frame,
            source=str(config.paths.raw_csv),
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
        )
    elif config.paths.raw_csv.name == "iris.csv":
        artifacts = build_iris_dataframe()
        frame = artifacts.frame
        metadata = artifacts.metadata
    else:
        raise FileNotFoundError(
            "Configured raw CSV does not exist: "
            f"{config.paths.raw_csv}. Provide the file or use iris.csv for auto-fetch."
        )

    ensure_parent(config.paths.raw_csv)
    frame.to_csv(config.paths.raw_csv, index=False)
    write_json(config.paths.dataset_metadata, metadata)

    logger.info("Zapisano dane surowe do %s", config.paths.raw_csv)
    logger.info("Zapisano metadane do %s", config.paths.dataset_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
