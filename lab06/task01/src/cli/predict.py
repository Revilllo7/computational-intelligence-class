from __future__ import annotations

import argparse

from src.experiments import ExperimentFactory
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger


def parse_feature_argument(raw_value: str) -> tuple[str, float]:
    if "=" not in raw_value:
        raise argparse.ArgumentTypeError(
            "Invalid --feature format. Use --feature <name>=<float_value>."
        )
    name, value = raw_value.split("=", maxsplit=1)
    if not name:
        raise argparse.ArgumentTypeError("Feature name cannot be empty.")
    try:
        return name, float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Feature '{name}' must have a numeric value.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wykonaj pojedyncza predykcje dla probki wskazanej konfiguracja."
    )
    parser.add_argument("--config", required=True, help="Sciezka do pliku YAML z konfiguracja.")
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        type=parse_feature_argument,
        metavar="name=value",
        help=(
            "Wartosc cechy zgodna z konfiguracja, np. --feature param1=0.9. "
            "Mozesz podac wiele razy."
        ),
    )
    parser.add_argument("--sepal-length", type=float)
    parser.add_argument("--sepal-width", type=float)
    parser.add_argument("--petal-length", type=float)
    parser.add_argument("--petal-width", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = get_logger("predict")

    if args.feature:
        raw_feature_pairs = dict(args.feature)
        if len(raw_feature_pairs) != len(args.feature):
            raise ValueError("Each feature name can be provided only once.")

        configured_features = config.data.feature_columns
        missing_features = [
            feature for feature in configured_features if feature not in raw_feature_pairs
        ]
        extra_features = sorted(set(raw_feature_pairs) - set(configured_features))

        if missing_features or extra_features:
            raise ValueError(
                "Provided --feature values do not match configuration. "
                f"Missing: {missing_features}; extra: {extra_features}."
            )

        raw_values = {
            feature: raw_feature_pairs[feature] for feature in config.data.feature_columns
        }
    else:
        legacy_values = {
            "sepal_length_cm": args.sepal_length,
            "sepal_width_cm": args.sepal_width,
            "petal_length_cm": args.petal_length,
            "petal_width_cm": args.petal_width,
        }
        if any(value is not None for value in legacy_values.values()):
            missing_legacy = [name for name, value in legacy_values.items() if value is None]
            if missing_legacy:
                raise ValueError(
                    f"Legacy Iris arguments require all four values. Missing: {missing_legacy}."
                )
            raw_values = {key: float(value) for key, value in legacy_values.items()}
        else:
            raise ValueError(
                "Provide feature values using --feature name=value (recommended) "
                "or the legacy Iris arguments."
            )

    experiment = ExperimentFactory.build(config.model.name, config)
    label, probabilities = experiment.predict_one(raw_values)

    logger.info("Predykcja dla eksperymentu %s: %s", config.experiment_name, label)
    for class_name, probability in zip(config.data.class_names, probabilities, strict=True):
        logger.info("%s -> %.4f", class_name, probability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
