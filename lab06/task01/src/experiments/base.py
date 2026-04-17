from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.data.preprocessing import apply_preprocessor, fit_preprocessor, split_dataset
from src.training.trainer import TrainingBundle, predict_with_model, train_model
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.utils.torch_runtime import prepare_torch_import
from src.visualization.plots import save_confusion_matrix, save_training_curves


class BaseExperiment(ABC):
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = get_logger(self.config.experiment_name)

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> Any:
        raise NotImplementedError

    def preprocess(self, frame: pd.DataFrame) -> None:
        split = split_dataset(
            frame=frame,
            target_column=self.config.data.target_column,
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            random_seed=self.config.random_seed,
        )
        stats = fit_preprocessor(
            split.train,
            self.config.data.feature_columns,
            self.config.preprocessing.strategy,
        )
        processed_train = apply_preprocessor(
            split.train,
            self.config.data.feature_columns,
            stats,
            self.config.preprocessing.strategy,
        )
        processed_validation = apply_preprocessor(
            split.validation,
            self.config.data.feature_columns,
            stats,
            self.config.preprocessing.strategy,
        )
        processed_test = apply_preprocessor(
            split.test,
            self.config.data.feature_columns,
            stats,
            self.config.preprocessing.strategy,
        )

        self.config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        processed_train.to_csv(self.config.paths.processed_dir / "train.csv", index=False)
        processed_validation.to_csv(self.config.paths.processed_dir / "validation.csv", index=False)
        processed_test.to_csv(self.config.paths.processed_dir / "test.csv", index=False)

        write_json(
            self.config.paths.preprocessor_artifact,
            {
                "feature_columns": self.config.data.feature_columns,
                "class_names": self.config.data.class_names,
                "target_column": self.config.data.target_column,
                "strategy": self.config.preprocessing.strategy,
                "parameters": stats,
            },
        )
        self.logger.info(
            "Preprocessing zakonczony dla eksperymentu %s (%s).",
            self.config.experiment_name,
            self.config.preprocessing.strategy,
        )

    def train(self) -> None:
        set_global_seed(self.config.random_seed)
        train_frame = pd.read_csv(self.config.paths.processed_dir / "train.csv")
        validation_frame = pd.read_csv(self.config.paths.processed_dir / "validation.csv")
        model = self.build_model()
        result = train_model(
            model=model,
            bundle=TrainingBundle(
                train_frame=train_frame,
                validation_frame=validation_frame,
                feature_columns=self.config.data.feature_columns,
                class_names=self.config.data.class_names,
                target_column=self.config.data.target_column,
            ),
            learning_rate=self.config.training.learning_rate,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            weight_decay=self.config.training.weight_decay,
            checkpoint_path=self.config.paths.model_checkpoint,
        )
        ensure_parent(self.config.paths.training_history_csv)
        result.history.to_csv(self.config.paths.training_history_csv, index=False)
        write_json(
            self.config.paths.training_summary_json,
            {
                "experiment_name": self.config.experiment_name,
                "classifier": self.config.model.name,
                "preprocessing": self.config.preprocessing.strategy,
                "best_validation_accuracy": result.best_validation_accuracy,
                "epochs": self.config.training.epochs,
                "hidden_dims": self.config.training.hidden_dims,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
            },
        )
        self.logger.info("Trening zakonczony dla eksperymentu %s.", self.config.experiment_name)

    def evaluate(self) -> None:
        test_frame = pd.read_csv(self.config.paths.processed_dir / "test.csv")
        model = self.build_model()
        self.load_checkpoint(model)
        y_true = list(test_frame[self.config.data.target_column].astype(str))
        y_pred, probabilities = predict_with_model(
            model,
            test_frame,
            self.config.data.feature_columns,
            self.config.data.class_names,
        )
        accuracy = float(accuracy_score(y_true, y_pred))
        report = classification_report(y_true, y_pred, output_dict=True)
        report_dict = cast(dict[str, Any], report)
        macro_f1 = float(report_dict["macro avg"]["f1-score"])
        weighted_f1 = float(report_dict["weighted avg"]["f1-score"])
        predictions = test_frame.copy()
        predictions[f"predicted_{self.config.data.target_column}"] = y_pred
        predictions["confidence"] = [max(row) for row in probabilities]
        predictions.to_csv(self.config.paths.predictions_csv, index=False)

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.config.data.class_names,
            output_path=self.config.paths.confusion_matrix_png,
            dpi=self.config.visualization.figure_dpi,
        )
        write_json(
            self.config.paths.evaluation_json,
            {
                "experiment_name": self.config.experiment_name,
                "classifier": self.config.model.name,
                "preprocessing": self.config.preprocessing.strategy,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report_dict,
            },
        )
        self.logger.info(
            "Ewaluacja zakonczona dla eksperymentu %s z accuracy %.4f.",
            self.config.experiment_name,
            accuracy,
        )

    def visualize_training(self) -> None:
        history = pd.read_csv(self.config.paths.training_history_csv)
        save_training_curves(
            history,
            self.config.paths.training_curves_png,
            dpi=self.config.visualization.figure_dpi,
        )

    def predict_one(self, raw_values: dict[str, float]) -> tuple[str, list[float]]:
        artifact = read_json(self.config.paths.preprocessor_artifact)
        parameters = artifact["parameters"]
        strategy = artifact["strategy"]
        normalized: list[float] = []
        for column in self.config.data.feature_columns:
            if strategy == "zscore":
                normalized.append(
                    (raw_values[column] - parameters[column]["mean"]) / parameters[column]["std"]
                )
            elif strategy == "minmax":
                normalized.append(
                    (raw_values[column] - parameters[column]["min"])
                    / (parameters[column]["max"] - parameters[column]["min"])
                )
            elif strategy == "robust":
                normalized.append(
                    (raw_values[column] - parameters[column]["median"]) / parameters[column]["iqr"]
                )
            else:
                raise ValueError(f"Unsupported preprocessing strategy: {strategy}")

        frame = pd.DataFrame([raw_values])
        for index, column in enumerate(self.config.data.feature_columns):
            frame[column] = normalized[index]

        model = self.build_model()
        self.load_checkpoint(model)
        labels, probabilities = predict_with_model(
            model,
            frame,
            self.config.data.feature_columns,
            self.config.data.class_names,
        )
        return labels[0], probabilities[0]

    def run(self, frame: pd.DataFrame) -> None:
        self.preprocess(frame)
        self.train()
        self.evaluate()
        self.visualize_training()

    def load_checkpoint(self, model: Any) -> None:
        prepare_torch_import()
        import torch

        state_dict = torch.load(self.config.paths.model_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
