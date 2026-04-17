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
    ensure_parent(compare_config.comparison_csv)
    results.to_csv(compare_config.comparison_csv, index=False)
    write_json(
        compare_config.comparison_json,
        {"experiments": results.to_dict(orient="records")},
    )

    ensure_parent(compare_config.comparison_plot_png)
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    x_positions = list(range(len(results)))
    bar_width = 0.4
    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        results["test_accuracy"],
        width=bar_width,
        label="test_accuracy",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        results["macro_f1"],
        width=bar_width,
        label="macro_f1",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(results["experiment_name"], rotation=20)
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.set_title("Porownanie eksperymentow")
    fig.tight_layout()
    fig.savefig(compare_config.comparison_plot_png)
    plt.close(fig)

    logger.info("Zapisano porownanie eksperymentow do %s", compare_config.comparison_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
