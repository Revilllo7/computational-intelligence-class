from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_json
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokaz podsumowanie wynikow eksperymentow.")
    parser.add_argument(
        "--config",
        required=True,
        help="Sciezka do pliku YAML z konfiguracja porownania.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Liczba najlepszych eksperymentow do wyswietlenia.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_config = ComparisonConfig.from_yaml(args.config)
    logger = get_logger("report_summary")

    rows: list[dict[str, str | float]] = []
    for experiment_config_path in compare_config.experiments:
        config = ProjectConfig.from_yaml(experiment_config_path)
        training_summary = read_json(config.paths.training_summary_json)
        evaluation = read_json(config.paths.evaluation_json)
        macro_f1 = float(
            evaluation.get(
                "macro_f1",
                evaluation["classification_report"]["macro avg"]["f1-score"],
            )
        )
        weighted_f1 = float(
            evaluation.get(
                "weighted_f1",
                evaluation["classification_report"]["weighted avg"]["f1-score"],
            )
        )
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "classifier": config.model.name,
                "preprocessing": config.preprocessing.strategy,
                "best_validation_accuracy": float(training_summary["best_validation_accuracy"]),
                "test_accuracy": float(evaluation["accuracy"]),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["test_accuracy", "macro_f1", "best_validation_accuracy"],
        ascending=False,
    )

    print(results.head(args.top).to_string(index=False))

    summary_csv = compare_config.comparison_csv.with_name("experiment_summary.csv")
    ensure_parent(summary_csv)
    results.to_csv(summary_csv, index=False)
    logger.info("Zapisano podsumowanie eksperymentow do %s", summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
