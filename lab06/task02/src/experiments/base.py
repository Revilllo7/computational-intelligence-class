from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.metrics import classification_report

from src.data.dataset import (
    build_dataset_metadata,
    build_manifest,
    discover_image_paths,
    split_manifest,
    unpack_zip_dataset,
    write_manifest,
)
from src.data.preprocessing import build_dataloader, write_preprocessing_artifact
from src.training.trainer import predict_with_model, train_model
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, write_json
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

    def fetch(self) -> None:
        unpack_zip_dataset(self.config.paths.raw_zip, self.config.paths.extracted_dir)
        image_paths = discover_image_paths(self.config.paths.extracted_dir)
        if not image_paths:
            raise ValueError(
                "No images found after zip extraction. Check dataset archive structure and paths."
            )

        manifest = build_manifest(image_paths, self.config.data.class_names)
        metadata = build_dataset_metadata(
            manifest=manifest,
            raw_zip=self.config.paths.raw_zip,
            extracted_dir=self.config.paths.extracted_dir,
            class_names=self.config.data.class_names,
            max_samples=self.config.data.max_samples,
        )
        write_json(self.config.paths.dataset_metadata, metadata)
        self.logger.info(
            "Fetched dataset metadata with %d images from %s",
            len(manifest),
            self.config.paths.raw_zip,
        )

    def preprocess(self) -> None:
        self.fetch()

        image_paths = discover_image_paths(self.config.paths.extracted_dir)
        manifest = build_manifest(image_paths, self.config.data.class_names)
        split = split_manifest(
            manifest=manifest,
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            random_seed=self.config.random_seed,
            max_samples=self.config.data.max_samples,
        )

        self.config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(self.config.paths.train_manifest_csv, split.train)
        write_manifest(self.config.paths.validation_manifest_csv, split.validation)
        write_manifest(self.config.paths.test_manifest_csv, split.test)

        write_preprocessing_artifact(
            output_path=self.config.paths.preprocessor_artifact,
            class_names=self.config.data.class_names,
            preprocessing_config=self.config.preprocessing,
            train_size=len(split.train),
            validation_size=len(split.validation),
            test_size=len(split.test),
        )

        metadata = build_dataset_metadata(
            manifest=manifest,
            raw_zip=self.config.paths.raw_zip,
            extracted_dir=self.config.paths.extracted_dir,
            class_names=self.config.data.class_names,
            max_samples=self.config.data.max_samples,
        )
        metadata["split_sizes"] = {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        }
        write_json(self.config.paths.dataset_metadata, metadata)

        self.logger.info(
            "Preprocessed dataset: train=%d validation=%d test=%d",
            len(split.train),
            len(split.validation),
            len(split.test),
        )

    def train(self) -> None:
        set_global_seed(self.config.random_seed)

        train_loader = build_dataloader(
            manifest_path=self.config.paths.train_manifest_csv,
            class_names=self.config.data.class_names,
            preprocessing_config=self.config.preprocessing,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=True,
            is_train=True,
        )
        validation_loader = build_dataloader(
            manifest_path=self.config.paths.validation_manifest_csv,
            class_names=self.config.data.class_names,
            preprocessing_config=self.config.preprocessing,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=False,
            is_train=False,
        )

        model = self.build_model()
        device = self.resolve_device()

        result = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            learning_rate=self.config.training.learning_rate,
            epochs=self.config.training.epochs,
            weight_decay=self.config.training.weight_decay,
            optimizer_name=self.config.training.optimizer,
            sgd_momentum=self.config.training.sgd_momentum,
            checkpoint_path=self.config.paths.model_checkpoint,
            device=device,
        )

        ensure_parent(self.config.paths.training_history_csv)
        result.history.to_csv(self.config.paths.training_history_csv, index=False)
        train_manifest = pd.read_csv(self.config.paths.train_manifest_csv)
        validation_manifest = pd.read_csv(self.config.paths.validation_manifest_csv)
        write_json(
            self.config.paths.training_summary_json,
            {
                "experiment_name": self.config.experiment_name,
                "classifier": self.config.model.name,
                "best_validation_accuracy": result.best_validation_accuracy,
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "optimizer": self.config.training.optimizer,
                "sgd_momentum": self.config.training.sgd_momentum,
                "activation": self.config.model.activation,
                "conv_channels": self.config.model.conv_channels,
                "num_conv_layers": len(self.config.model.conv_channels),
                "dropout": self.config.model.dropout,
                "augment_train": self.config.preprocessing.augment_train,
                "random_horizontal_flip_p": self.config.preprocessing.random_horizontal_flip_p,
                "random_rotation_degrees": self.config.preprocessing.random_rotation_degrees,
                "color_jitter_brightness": self.config.preprocessing.color_jitter_brightness,
                "color_jitter_contrast": self.config.preprocessing.color_jitter_contrast,
                "train_samples": len(train_manifest),
                "validation_samples": len(validation_manifest),
            },
        )
        self.logger.info(
            "Training completed for %s with best validation accuracy %.4f",
            self.config.experiment_name,
            result.best_validation_accuracy,
        )

    def evaluate(self) -> None:
        test_loader = build_dataloader(
            manifest_path=self.config.paths.test_manifest_csv,
            class_names=self.config.data.class_names,
            preprocessing_config=self.config.preprocessing,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=False,
            is_train=False,
        )

        model = self.build_model()
        self.load_checkpoint(model)
        device = self.resolve_device()

        output = predict_with_model(model=model, data_loader=test_loader, device=device)
        y_true = [self.config.data.class_names[index] for index in output.true_indices]
        y_pred = [self.config.data.class_names[index] for index in output.pred_indices]

        predictions = pd.DataFrame(
            {
                "image_path": output.image_paths,
                "true_label": y_true,
                "predicted_label": y_pred,
                "confidence": [max(row) for row in output.probabilities],
            }
        )
        for index, class_name in enumerate(self.config.data.class_names):
            predictions[f"prob_{class_name}"] = [row[index] for row in output.probabilities]

        ensure_parent(self.config.paths.predictions_csv)
        predictions.to_csv(self.config.paths.predictions_csv, index=False)

        report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=self.config.data.class_names,
            output_dict=True,
        )
        report_dict = cast(dict[str, Any], report)
        macro_f1 = float(report_dict["macro avg"]["f1-score"])
        weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.config.data.class_names,
            output_path=self.config.paths.confusion_matrix_png,
            dpi=self.config.visualization.figure_dpi,
        )

        misclassified = predictions[
            predictions["true_label"] != predictions["predicted_label"]
        ].copy()
        misclassified_summary = self._export_misclassified_images(misclassified)
        write_json(self.config.paths.misclassified_summary_json, misclassified_summary)

        write_json(
            self.config.paths.evaluation_json,
            {
                "experiment_name": self.config.experiment_name,
                "classifier": self.config.model.name,
                "accuracy": output.accuracy,
                "loss": output.loss,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report_dict,
                "misclassification_analysis": misclassified_summary,
            },
        )

        self.logger.info(
            "Evaluation completed for %s with test accuracy %.4f",
            self.config.experiment_name,
            output.accuracy,
        )

    def visualize_training(self) -> None:
        history = pd.read_csv(self.config.paths.training_history_csv)
        save_training_curves(
            history=history,
            output_path=self.config.paths.training_curves_png,
            dpi=self.config.visualization.figure_dpi,
        )
        self.logger.info("Training curves saved to %s", self.config.paths.training_curves_png)

    def run(self) -> None:
        self.preprocess()
        self.train()
        self.evaluate()
        self.visualize_training()

    def resolve_device(self) -> Any:
        prepare_torch_import()
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_checkpoint(self, model: Any) -> None:
        prepare_torch_import()
        import torch

        state_dict = torch.load(self.config.paths.model_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    def _export_misclassified_images(self, frame: pd.DataFrame) -> dict[str, object]:
        output_dir = self.config.paths.misclassified_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        counts_by_direction: dict[str, int] = {}
        copied_files = 0

        for _, row in frame.iterrows():
            image_path = Path(str(row["image_path"]))
            true_label = str(row["true_label"])
            predicted_label = str(row["predicted_label"])
            direction = f"{true_label}_as_{predicted_label}"

            direction_dir = output_dir / direction
            direction_dir.mkdir(parents=True, exist_ok=True)

            destination = self._unique_destination_path(
                directory=direction_dir,
                stem=image_path.stem,
                suffix=image_path.suffix,
                predicted_label=predicted_label,
            )
            shutil.copy2(image_path, destination)

            counts_by_direction[direction] = counts_by_direction.get(direction, 0) + 1
            copied_files += 1

        return {
            "total_misclassified": len(frame),
            "copied_files": copied_files,
            "counts_by_direction": counts_by_direction,
            "output_dir": str(output_dir),
        }

    @staticmethod
    def _unique_destination_path(
        directory: Path,
        stem: str,
        suffix: str,
        predicted_label: str,
    ) -> Path:
        candidate = directory / f"{stem}__pred_{predicted_label}{suffix}"
        index = 1
        while candidate.exists():
            candidate = directory / f"{stem}__pred_{predicted_label}_{index}{suffix}"
            index += 1
        return candidate
