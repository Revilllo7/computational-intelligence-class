"""Tests for task01 YAML configuration loading and validation."""

from task01.src.config import DEFAULT_CONFIG_PATH, load_experiment_config


def test_load_default_config():
    cfg_path = DEFAULT_CONFIG_PATH.parent / "default_config.yaml"
    cfg = load_experiment_config(cfg_path)
    assert cfg.config_path.exists()
    assert cfg.output_dir.name == cfg.config_name
    # basic experiment fields present
    assert hasattr(cfg.experiment, "sphere")
    assert hasattr(cfg.experiment, "alloy")
