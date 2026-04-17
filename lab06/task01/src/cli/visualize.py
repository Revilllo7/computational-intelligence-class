from __future__ import annotations

import argparse

from src.data.dataset import read_dataset
from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger
from src.visualization.plots import save_feature_histograms, save_pairplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wygeneruj wykresy dla wybranego zestawu danych.")
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    parser.add_argument(
        "--kind",
        choices=["raw", "training", "all"],
        default="all",
        help="Rodzaj wykresow do wygenerowania.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("visualize")

    if args.kind in {"raw", "all"}:
        raw_frame = read_dataset(config.paths.raw_csv)
        save_feature_histograms(
            raw_frame,
            config.data.feature_columns,
            config.data.target_column,
            config.paths.feature_hist_png,
            dpi=config.visualization.figure_dpi,
        )
        save_pairplot(
            raw_frame,
            config.data.feature_columns,
            config.data.target_column,
            config.paths.pairplot_png,
            dpi=config.visualization.figure_dpi,
        )
        logger.info("Wygenerowano wykresy danych surowych.")

    if args.kind in {"training", "all"} and config.paths.training_history_csv.exists():
        experiment = ExperimentFactory.build(config.model.name, config)
        experiment.visualize_training()
        logger.info("Wygenerowano wykresy treningu.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
