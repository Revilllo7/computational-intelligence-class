from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PathConfig(BaseModel):
    raw_zip: Path
    extracted_dir: Path
    dataset_metadata: Path
    processed_dir: Path
    train_manifest_csv: Path
    validation_manifest_csv: Path
    test_manifest_csv: Path
    preprocessor_artifact: Path
    model_checkpoint: Path
    training_history_csv: Path
    training_summary_json: Path
    evaluation_json: Path
    predictions_csv: Path
    confusion_matrix_png: Path
    training_curves_png: Path
    misclassified_dir: Path
    misclassified_summary_json: Path


class DataConfig(BaseModel):
    class_names: list[str] = Field(min_length=2)
    test_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)
    max_samples: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_split_sizes(self) -> DataConfig:
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be smaller than 1.0")
        return self


class PreprocessingConfig(BaseModel):
    image_size: int = Field(gt=0)
    normalize_mean: list[float] = Field(min_length=3, max_length=3)
    normalize_std: list[float] = Field(min_length=3, max_length=3)
    augment_train: bool = True
    random_horizontal_flip_p: float = Field(default=0.5, ge=0.0, le=1.0)
    random_rotation_degrees: float = Field(default=10.0, ge=0.0)
    color_jitter_brightness: float = Field(default=0.15, ge=0.0)
    color_jitter_contrast: float = Field(default=0.15, ge=0.0)

    @model_validator(mode="after")
    def _validate_normalize_std(self) -> PreprocessingConfig:
        if any(value <= 0.0 for value in self.normalize_std):
            raise ValueError("All normalize_std values must be > 0")
        return self


class ModelConfig(BaseModel):
    name: Literal["cnn_classifier"]
    activation: Literal["relu", "leaky_relu", "elu", "tanh"] = "relu"
    conv_channels: list[int] = Field(min_length=1)
    hidden_dim: int = Field(gt=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)
    optimizer: Literal["adam", "sgd"] = "adam"
    sgd_momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    num_workers: int = Field(default=0, ge=0)


class VisualizationConfig(BaseModel):
    figure_dpi: int = Field(default=160, gt=0)


class ComparisonConfig(BaseModel):
    project_name: str
    experiments: list[Path]
    comparison_csv: Path
    comparison_json: Path
    accuracy_plot_png: Path
    learning_curves_grid_png: Path
    confusion_matrices_grid_png: Path
    figure_dpi: int = Field(default=160, gt=0)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ComparisonConfig:
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @field_validator("experiments", mode="before")
    @classmethod
    def _convert_experiment_paths(cls, value: list[str]) -> list[Path]:
        return [Path(raw) for raw in value]

    @field_validator(
        "comparison_csv",
        "comparison_json",
        "accuracy_plot_png",
        "learning_curves_grid_png",
        "confusion_matrices_grid_png",
        mode="before",
    )
    @classmethod
    def _convert_single_path(cls, value: str) -> Path:
        return Path(value)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    project_name: str
    experiment_name: str
    random_seed: int = 42
    paths: PathConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ProjectConfig:
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @field_validator("paths", mode="before")
    @classmethod
    def _convert_paths(cls, value: dict[str, Any]) -> dict[str, Path]:
        return {key: Path(raw) for key, raw in value.items()}
