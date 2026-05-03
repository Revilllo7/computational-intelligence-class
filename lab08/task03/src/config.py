"""Task 03 configuration models and YAML loading helpers for the maze solvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

TASK03_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK03_ROOT.parent
DEFAULT_OUTPUT_ROOT = TASK03_ROOT / "experiments"
DEFAULT_CONFIG_PATH = TASK03_ROOT / "configs" / "default_config.yaml"


class ACOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ant_count: int = Field(default=300, gt=0)
    alpha: float = Field(default=0.5, ge=0.0)
    beta: float = Field(default=1.2, ge=0.0)
    pheromone_evaporation_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    pheromone_constant: float = Field(default=1000.0, gt=0.0)
    iterations: int = Field(default=300, gt=0)


class PSOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    particle_count: int = Field(default=30, gt=0)
    iterations: int = Field(default=120, gt=0)
    sequence_length: int = Field(default=30, gt=0)
    lower_bound: float = Field(default=-4.0)
    upper_bound: float = Field(default=4.0)
    velocity_clamp: float = Field(default=2.0, gt=0.0)
    c1: float = Field(default=0.5, ge=0.0)
    c2: float = Field(default=0.3, ge=0.0)
    w: float = Field(default=0.5, ge=0.0, le=1.0)
    step_weight: float = Field(default=1.0, gt=0.0)
    revisit_weight: float = Field(default=0.5, ge=0.0)
    goal_distance_weight: float = Field(default=4.0, gt=0.0)
    failure_penalty: float = Field(default=100.0, ge=0.0)


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    save_cost_plot: bool = True
    figure_dpi: int = Field(default=120, gt=0)


class GifConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    aco_evolution: bool = True
    pso_evolution: bool = True
    astar_solving: bool = True
    sample_every_n_generations: int = Field(default=4, gt=0)
    astar_sample_every_n_steps: int = Field(default=1, gt=0)
    frame_duration_ms: int = Field(default=300, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    experiment_name: str | None = None
    output_root: Path = Path("task03/experiments")
    aco: ACOConfig = Field(default_factory=ACOConfig)
    pso: PSOConfig = Field(default_factory=PSOConfig)
    plotting: PlotConfig = Field(default_factory=PlotConfig)
    gif: GifConfig = Field(default_factory=GifConfig)

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


def _nest_legacy_sections(payload: dict[str, Any]) -> dict[str, Any]:
    nested = dict(payload)
    section_fields = {
        "aco": set(ACOConfig.model_fields),
        "pso": set(PSOConfig.model_fields),
        "plotting": set(PlotConfig.model_fields),
        "gif": set(GifConfig.model_fields),
        # runs intentionally omitted — single-run mode
    }
    for section_name, field_names in section_fields.items():
        if section_name in nested:
            continue
        section_payload = {
            field: nested.pop(field) for field in list(field_names) if field in nested
        }
        if section_payload:
            nested[section_name] = section_payload
    return nested


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> LoadedExperimentConfig:
    """Load and validate a task 03 maze experiment configuration."""

    raw_path = Path(config_path)
    resolved_config_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    resolved_config_path = resolved_config_path.resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise TypeError("Configuration file must contain a YAML mapping at the top level")

    experiment = ExperimentConfig(**_nest_legacy_sections(payload))

    # Use the resolved output root directly (single-run mode)
    resolved_output_root = _resolve_path(experiment.output_root, PROJECT_ROOT).resolve()
    expected_output_root = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved_output_root != expected_output_root:
        raise ValueError(
            f"output_root must resolve to '{expected_output_root}', got '{resolved_output_root}'"
        )

    output_dir = resolved_output_root

    # Preserve a stable config name for metadata, but do not create a per-config subfolder
    explicit_name = (experiment.experiment_name or "").strip()
    config_name = explicit_name if explicit_name else resolved_config_path.stem

    return LoadedExperimentConfig(
        config_path=resolved_config_path,
        config_name=config_name,
        project_root=PROJECT_ROOT,
        task_root=TASK03_ROOT,
        output_root=resolved_output_root,
        output_dir=output_dir,
        experiment=experiment,
    )
