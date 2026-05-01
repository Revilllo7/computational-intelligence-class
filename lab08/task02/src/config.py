"""Task 02 configuration models and YAML loading helpers for the ACO TSP run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

TASK02_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK02_ROOT.parent
DEFAULT_OUTPUT_ROOT = TASK02_ROOT / "experiments"
DEFAULT_CONFIG_PATH = TASK02_ROOT / "configs" / "default_config.yaml"


class ACOConfig(BaseModel):
    ant_count: int = Field(default=300)
    alpha: float = Field(default=0.5)
    beta: float = Field(default=1.2)
    pheromone_evaporation_rate: float = Field(default=0.4)
    pheromone_constant: float = Field(default=1000.0)
    iterations: int = Field(default=300)


class ExperimentConfig(BaseModel):
    # Keep a small, explicit schema for task02 experiments
    aco: ACOConfig = Field(default_factory=ACOConfig)
    output_root: Path = Path("task02/experiments")


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
    """Load the task02 configuration from a YAML file.

    Returns a LoadedExperimentConfig with resolved output paths.
    """
    raw_path = Path(config_path)
    resolved_config_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    resolved_config_path = resolved_config_path.resolve()

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    # Support both config styles:
    # 1) nested: {aco: {...}, output_root: ...}
    # 2) flat: {ant_count: ..., alpha: ..., ...}
    if "aco" not in payload and any(field in payload for field in ACOConfig.model_fields):
        aco_payload: dict[str, Any] = {}
        for field_name in ACOConfig.model_fields:
            if field_name in payload:
                aco_payload[field_name] = payload.pop(field_name)
        payload["aco"] = aco_payload

    experiment = ExperimentConfig(**payload)

    explicit_name = (resolved_config_path.stem or "").strip()
    config_name = explicit_name if explicit_name else "default"

    resolved_output_root = _resolve_path(experiment.output_root, PROJECT_ROOT).resolve()
    output_dir = resolved_output_root / config_name

    return LoadedExperimentConfig(
        config_path=resolved_config_path,
        config_name=config_name,
        project_root=PROJECT_ROOT,
        task_root=TASK02_ROOT,
        output_root=resolved_output_root,
        output_dir=output_dir,
        experiment=experiment,
    )
