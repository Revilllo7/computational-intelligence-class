"""Task 01 configuration models and YAML loading helpers for the PSO alloy run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

TASK01_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK01_ROOT.parent
DEFAULT_OUTPUT_ROOT = TASK01_ROOT / "experiments"
DEFAULT_CONFIG_PATH = TASK01_ROOT / "configs" / "default.yaml"


class SphereConfig(BaseModel):
    dimensions: int = Field(default=2, gt=0)
    lower_bound: float = Field(default=1.0)
    upper_bound: float = Field(default=2.0)
    n_particles: int = Field(default=10, gt=0)
    iters: int = Field(default=200, gt=0)


class AlloyConfig(BaseModel):
    dimensions: int = Field(default=6, gt=0)
    lower_bound: float = Field(default=0.0)
    upper_bound: float = Field(default=1.0)
    active_metal_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    n_particles: int = Field(default=10, gt=0)
    iters: int = Field(default=50, gt=0)
    target_durability: float = Field(default=2.833945350484618, gt=0.0)


class PSOOptions(BaseModel):
    c1: float = Field(default=0.5, ge=0.0)
    c2: float = Field(default=0.3, ge=0.0)
    w: float = Field(default=0.5, ge=0.0)


class PlotConfig(BaseModel):
    enabled: bool = True
    save_cost_plot: bool = True
    save_3d_plot: bool = True
    save_animation: bool = True
    figure_dpi: int = Field(default=120, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    experiment_name: str | None = None
    output_root: Path = Path("task01/experiments")
    sphere: SphereConfig = Field(default_factory=SphereConfig)
    alloy: AlloyConfig = Field(default_factory=AlloyConfig)
    pso_options: PSOOptions = Field(default_factory=PSOOptions)
    plotting: PlotConfig = Field(default_factory=PlotConfig)

    @field_validator("output_root", mode="before")
    @classmethod
    def _convert_output_root(cls, value: str | Path) -> Path:
        return Path(value)


@dataclass(frozen=True)
class LoadedExperimentConfig:
    config_path: Path
    config_name: str
    project_root: Path
    task_root: Path
    output_root: Path
    output_dir: Path
    experiment: ExperimentConfig


def _resolve_path(raw_path: Path, base_dir: Path) -> Path:
    return raw_path if raw_path.is_absolute() else (base_dir / raw_path)


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> LoadedExperimentConfig:
    """Load and validate the task 01 configuration from a YAML file."""
    raw_path = Path(config_path)
    resolved_config_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    resolved_config_path = resolved_config_path.resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    experiment = ExperimentConfig(**payload)

    explicit_name = (experiment.experiment_name or "").strip()
    config_name = explicit_name if explicit_name else resolved_config_path.stem

    resolved_output_root = _resolve_path(experiment.output_root, PROJECT_ROOT).resolve()
    expected_output_root = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved_output_root != expected_output_root:
        raise ValueError(
            f"output_root must resolve to '{expected_output_root}', got '{resolved_output_root}'"
        )

    output_dir = resolved_output_root / config_name

    return LoadedExperimentConfig(
        config_path=resolved_config_path,
        config_name=config_name,
        project_root=PROJECT_ROOT,
        task_root=TASK01_ROOT,
        output_root=resolved_output_root,
        output_dir=output_dir,
        experiment=experiment,
    )
