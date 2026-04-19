setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

fetch:
	uv run -m src.cli.fetch_data --config configs/cats_dogs_quick.yaml

preprocess:
	uv run -m src.cli.preprocess_data --config configs/cats_dogs_quick.yaml

train:
	uv run -m src.cli.train_model --config configs/cats_dogs_quick.yaml

evaluate:
	uv run -m src.cli.evaluate_model --config configs/cats_dogs_quick.yaml

visualize:
	uv run -m src.cli.visualize --config configs/cats_dogs_quick.yaml

run-quick:
	uv run -m src.cli.run_experiment --config configs/cats_dogs_quick.yaml

run-full:
	uv run -m src.cli.run_experiment --config configs/cats_dogs_full.yaml

run-experiment: run-quick

run-matrix:
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_baseline_adam_relu.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_optimizer_sgd.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_activation_tanh.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_layers_2.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_filters_high.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_dropout_off.yaml
	uv run -m src.cli.run_experiment --config configs/cats_dogs_exp_augment_strong.yaml

compare:
	uv run -m src.cli.compare_experiments --config configs/compare_experiments_matrix.yaml

matrix: run-matrix compare

test:
	uv run pytest

ruff:
	uv run ruff check .
	uv run ruff format --check .

ruff-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

pyright:
	uv run pyright
