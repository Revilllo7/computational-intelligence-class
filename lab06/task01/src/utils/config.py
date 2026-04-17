from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathConfig(BaseModel):
    raw_csv: Path
    dataset_metadata: Path
    processed_dir: Path
    preprocessor_artifact: Path
    model_checkpoint: Path
    training_history_csv: Path
    training_summary_json: Path
    evaluation_json: Path
    predictions_csv: Path
    confusion_matrix_png: Path
    pairplot_png: Path
    feature_hist_png: Path
    training_curves_png: Path


class DataConfig(BaseModel):
    target_column: str
    feature_columns: list[str]
    class_names: list[str]
    test_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    hidden_dims: list[int]
    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)


class VisualizationConfig(BaseModel):
    figure_dpi: int = Field(default=160, gt=0)


class PreprocessingConfig(BaseModel):
    strategy: Literal["zscore", "minmax", "robust"]


class ModelConfig(BaseModel):
    name: Literal["mlp_classifier", "linear_classifier", "shallow_mlp_classifier"]


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


class ComparisonConfig(BaseModel):
    project_name: str
    experiments: list[Path]
    comparison_csv: Path
    comparison_json: Path
    comparison_plot_png: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ComparisonConfig:
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @field_validator("experiments", mode="before")
    @classmethod
    def _convert_experiments(cls, value: list[str]) -> list[Path]:
        return [Path(raw) for raw in value]
