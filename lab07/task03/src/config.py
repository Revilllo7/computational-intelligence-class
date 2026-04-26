"""Task 03 configuration models and YAML loading helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from task03.data.maze import MAX_STEPS

TASK03_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK03_ROOT.parent
DEFAULT_OUTPUT_ROOT = TASK03_ROOT / "experiments"
DEFAULT_CONFIG_PATH = TASK03_ROOT / "configs" / "maze_baseline.yaml"


class ProblemConfig(BaseModel):
    start: tuple[int, int] = (1, 1)
    goal: tuple[int, int] = (10, 10)
    max_steps: int = Field(default=MAX_STEPS, gt=0)
    target_steps: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def _validate_max_steps(self) -> ProblemConfig:
        if self.max_steps != MAX_STEPS:
            raise ValueError(f"max_steps must be exactly {MAX_STEPS}")
        if self.target_steps > self.max_steps:
            raise ValueError("target_steps must be <= max_steps")
        return self


class FitnessConfig(BaseModel):
    success_bonus: float = 10_000.0
    distance_weight: float = 25.0
    progress_weight: float = 40.0
    efficiency_weight: float = 8.0
    solved_step_penalty: float = 120.0
    invalid_move_penalty: float = 20.0
    revisit_penalty: float = 6.0
    stagnation_penalty: float = 8.0
    exploration_reward: float = 0.2


class GAConfig(BaseModel):
    num_generations: int = Field(default=300, gt=0)
    solutions_per_population: int = Field(default=120, gt=0)
    num_parents_mating: int = Field(default=40, gt=0)
    parent_selection_type: str = "sss"
    mutation_type: str = "random"
    mutation_num_genes: int = Field(default=1, gt=0)
    crossover_type: str = "single_point"
    keep_parents: int = Field(default=2, ge=0)


class RunConfig(BaseModel):
    num_runs: int = Field(default=10, gt=0)
    timing_runs: int = Field(default=10, gt=0)
    single_run_seed: int | None = 42
    seed_strategy: Literal["random", "incremental", "fixed"] = "incremental"
    base_seed: int | None = 42
    fixed_seeds: list[int] = Field(default_factory=list)
    sweep_configs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_seed_settings(self) -> RunConfig:
        if self.seed_strategy == "incremental" and self.base_seed is None:
            raise ValueError("base_seed is required when seed_strategy='incremental'")
        if self.seed_strategy == "fixed" and len(self.fixed_seeds) < self.num_runs:
            raise ValueError("fixed_seeds must provide at least num_runs entries")
        return self


class GifConfig(BaseModel):
    enabled: bool = True
    ga_evolution: bool = True
    astar_solving: bool = True
    frame_duration_ms: int = Field(default=150, gt=0)
    ga_sample_every_n_generations: int = Field(default=4, gt=0)
    astar_sample_every_n_steps: int = Field(default=1, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    experiment_name: str | None = None
    output_root: Path = Path("task03/experiments")
    problem: ProblemConfig = Field(default_factory=ProblemConfig)
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    runs: RunConfig = Field(default_factory=RunConfig)
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

    def run_seeds(self) -> Sequence[int | None]:
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

    def timing_seeds(self) -> Sequence[int | None]:
        seeds = list(self.run_seeds())
        if len(seeds) >= self.experiment.runs.timing_runs:
            return seeds[: self.experiment.runs.timing_runs]
        return seeds + [None] * (self.experiment.runs.timing_runs - len(seeds))

    def resolve_sweep_config_paths(self) -> list[Path]:
        raw = self.experiment.runs.sweep_configs
        if not raw:
            return [self.config_path]

        resolved_paths: list[Path] = []
        for item in raw:
            item_path = Path(item)
            if not item_path.is_absolute():
                item_path = (self.project_root / item_path).resolve()
            if not item_path.exists():
                raise FileNotFoundError(f"Sweep config not found: {item_path}")
            resolved_paths.append(item_path)
        return resolved_paths


def _resolve_path(raw_path: Path, base_dir: Path) -> Path:
    return raw_path if raw_path.is_absolute() else (base_dir / raw_path)


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> LoadedExperimentConfig:
    """Load and validate the task03 experiment config from a YAML file."""
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
        task_root=TASK03_ROOT,
        output_root=resolved_output_root,
        output_dir=output_dir,
        experiment=experiment,
    )
