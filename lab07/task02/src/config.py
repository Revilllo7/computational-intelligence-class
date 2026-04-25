"""Task 02 configuration models and YAML loading helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TASK02_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK02_ROOT.parent
DEFAULT_OUTPUT_ROOT = TASK02_ROOT / "experiments"
DEFAULT_CONFIG_PATH = TASK02_ROOT / "configs" / "alloy_default.yaml"


class ProblemConfig(BaseModel):
    target_durability: float = Field(default=2.8, gt=0.0)
    active_metal_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)


class GAConfig(BaseModel):
    num_generations: int = Field(gt=0)
    solutions_per_population: int = Field(gt=0)
    num_parents_mating: int = Field(gt=0)
    parent_selection_type: str
    mutation_type: str
    mutation_percent_genes: float = Field(gt=0.0, le=100.0)
    crossover_type: str
    keep_parents: int = Field(ge=0)


class RunConfig(BaseModel):
    num_runs: int = Field(default=10, gt=0)
    single_run_seed: int | None = 42
    seed_strategy: Literal["random", "incremental", "fixed"] = "random"
    base_seed: int | None = 42
    fixed_seeds: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_seed_settings(self) -> RunConfig:
        if self.seed_strategy == "incremental" and self.base_seed is None:
            raise ValueError("base_seed is required when seed_strategy='incremental'")
        if self.seed_strategy == "fixed" and len(self.fixed_seeds) < self.num_runs:
            raise ValueError("fixed_seeds must provide at least num_runs entries")
        return self


class PlotConfig(BaseModel):
    enabled: bool = True
    single_run_curve: bool = True
    multi_run_overlay: bool = True
    aggregate_trend: bool = True
    figure_dpi: int = Field(default=120, gt=0)


class GifConfig(BaseModel):
    enabled: bool = False
    fitness_animation: bool = True
    population_animation: bool = True
    frame_duration_ms: int = Field(default=120, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    experiment_name: str | None = None
    output_root: Path = Path("task02/experiments")
    problem: ProblemConfig
    ga: GAConfig
    runs: RunConfig = Field(default_factory=RunConfig)
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

    def resolve_mutation_num_genes(self, num_genes: int) -> int:
        """Convert configured mutation percentage to an integer number of genes."""
        raw = (self.experiment.ga.mutation_percent_genes / 100.0) * num_genes
        rounded = round(raw)
        return max(1, min(num_genes, rounded))

    def run_seeds(self) -> Sequence[int | None]:
        """Return one seed per experimental run based on configured strategy."""
        run_config = self.experiment.runs

        if run_config.seed_strategy == "fixed":
            return run_config.fixed_seeds[: run_config.num_runs]

        if run_config.seed_strategy == "incremental":
            base_seed = run_config.base_seed
            assert base_seed is not None
            return [base_seed + idx for idx in range(run_config.num_runs)]

        if run_config.base_seed is None:
            return [None for _ in range(run_config.num_runs)]

        generator = random.Random(run_config.base_seed)
        return [generator.randint(1, 10_000_000) for _ in range(run_config.num_runs)]


def _resolve_path(raw_path: Path, base_dir: Path) -> Path:
    return raw_path if raw_path.is_absolute() else (base_dir / raw_path)


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> LoadedExperimentConfig:
    """Load and validate the task02 experiment config from a YAML file."""
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
        task_root=TASK02_ROOT,
        output_root=resolved_output_root,
        output_dir=output_dir,
        experiment=experiment,
    )
