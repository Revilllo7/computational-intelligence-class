from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Porownaj wyniki wielu eksperymentow.")
    parser.add_argument(
        "--config", required=True, help="Sciezka do pliku YAML z konfiguracja porownania."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_config = ComparisonConfig.from_yaml(args.config)
    logger = get_logger("compare_experiments")

    rows: list[dict[str, str | float]] = []
    for experiment_config_path in compare_config.experiments:
        config = ProjectConfig.from_yaml(experiment_config_path)
        training_summary = read_json(config.paths.training_summary_json)
        evaluation = read_json(config.paths.evaluation_json)
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "classifier": config.model.name,
                "preprocessing": config.preprocessing.strategy,
                "best_validation_accuracy": float(training_summary["best_validation_accuracy"]),
                "test_accuracy": float(evaluation["accuracy"]),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["test_accuracy", "best_validation_accuracy"],
        ascending=False,
    )
    ensure_parent(compare_config.comparison_csv)
    results.to_csv(compare_config.comparison_csv, index=False)
    write_json(
        compare_config.comparison_json,
        {"experiments": rows},
    )

    ensure_parent(compare_config.comparison_plot_png)
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.bar(results["experiment_name"], results["test_accuracy"])
    axis.set_ylabel("Test accuracy")
    axis.set_title("Porownanie eksperymentow")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(compare_config.comparison_plot_png)
    plt.close(fig)

    logger.info("Zapisano porownanie eksperymentow do %s", compare_config.comparison_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
