from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger
from src.visualization.plots import save_accuracy_comparison, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare results from multiple experiment configs."
    )
    parser.add_argument("--config", required=True, help="Path to comparison YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_config = ComparisonConfig.from_yaml(args.config)
    logger = get_logger("compare_experiments")

    rows: list[dict[str, Any]] = []

    for experiment_config_path in compare_config.experiments:
        config = ProjectConfig.from_yaml(experiment_config_path)
        training_summary = read_json(config.paths.training_summary_json)
        evaluation = read_json(config.paths.evaluation_json)

        if not config.paths.training_curves_png.exists():
            raise FileNotFoundError(f"Missing training curves: {config.paths.training_curves_png}")
        if not config.paths.confusion_matrix_png.exists():
            raise FileNotFoundError(
                f"Missing confusion matrix: {config.paths.confusion_matrix_png}"
            )

        report = cast(dict[str, Any], evaluation["classification_report"])
        macro_f1 = float(evaluation.get("macro_f1", report["macro avg"]["f1-score"]))
        weighted_f1 = float(evaluation.get("weighted_f1", report["weighted avg"]["f1-score"]))

        rows.append(
            {
                "experiment_name": config.experiment_name,
                "config_path": str(experiment_config_path),
                "optimizer": config.training.optimizer,
                "activation": config.model.activation,
                "conv_channels": "-".join(str(channel) for channel in config.model.conv_channels),
                "num_conv_layers": len(config.model.conv_channels),
                "dropout": config.model.dropout,
                "augment_train": config.preprocessing.augment_train,
                "random_horizontal_flip_p": config.preprocessing.random_horizontal_flip_p,
                "random_rotation_degrees": config.preprocessing.random_rotation_degrees,
                "color_jitter_brightness": config.preprocessing.color_jitter_brightness,
                "color_jitter_contrast": config.preprocessing.color_jitter_contrast,
                "best_validation_accuracy": float(training_summary["best_validation_accuracy"]),
                "test_accuracy": float(evaluation["accuracy"]),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "training_curves_png": str(config.paths.training_curves_png),
                "confusion_matrix_png": str(config.paths.confusion_matrix_png),
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
        {
            "project_name": compare_config.project_name,
            "experiments": cast(list[dict[str, Any]], results.to_dict(orient="records")),
        },
    )

    save_accuracy_comparison(
        results, compare_config.accuracy_plot_png, dpi=compare_config.figure_dpi
    )

    experiment_names = [str(name) for name in results["experiment_name"].tolist()]
    training_curve_paths = [Path(path) for path in results["training_curves_png"].tolist()]
    confusion_matrix_paths = [Path(path) for path in results["confusion_matrix_png"].tolist()]

    save_image_grid(
        image_paths=training_curve_paths,
        titles=experiment_names,
        output_path=compare_config.learning_curves_grid_png,
        dpi=compare_config.figure_dpi,
        title="Learning curves by experiment",
    )
    save_image_grid(
        image_paths=confusion_matrix_paths,
        titles=experiment_names,
        output_path=compare_config.confusion_matrices_grid_png,
        dpi=compare_config.figure_dpi,
        title="Confusion matrices by experiment",
    )

    ranking = results[
        [
            "experiment_name",
            "test_accuracy",
            "macro_f1",
            "optimizer",
            "activation",
            "num_conv_layers",
            "dropout",
        ]
    ]
    print(ranking.to_string(index=False))

    logger.info("Saved comparison CSV to %s", compare_config.comparison_csv)
    logger.info("Saved comparison JSON to %s", compare_config.comparison_json)
    logger.info("Saved accuracy chart to %s", compare_config.accuracy_plot_png)
    logger.info("Saved learning curves grid to %s", compare_config.learning_curves_grid_png)
    logger.info("Saved confusion matrix grid to %s", compare_config.confusion_matrices_grid_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
