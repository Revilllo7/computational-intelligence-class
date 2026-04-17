setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

fetch:
	uv run -m src.cli.fetch_data --config configs/iris_mlp_zscore.yaml

preprocess:
	uv run -m src.cli.preprocess_data --config configs/iris_mlp_zscore.yaml

train:
	uv run -m src.cli.train_model --config configs/iris_mlp_zscore.yaml

evaluate:
	uv run -m src.cli.evaluate_model --config configs/iris_mlp_zscore.yaml

visualize:
	uv run -m src.cli.visualize --config configs/iris_mlp_zscore.yaml --kind all

run-mlp-zscore:
	uv run -m src.cli.run_experiment --config configs/iris_mlp_zscore.yaml

run-experiment: run-mlp-zscore

run-mlp-minmax:
	uv run -m src.cli.run_experiment --config configs/iris_mlp_minmax.yaml

run-linear-zscore:
	uv run -m src.cli.run_experiment --config configs/iris_linear_zscore.yaml

run-linear-minmax:
	uv run -m src.cli.run_experiment --config configs/iris_linear_minmax.yaml

compare:
	uv run -m src.cli.compare_experiments --config configs/compare_experiments.yaml

compare-diagnosis:
	uv run -m src.cli.compare_experiments --config configs/compare_experiments_diagnosis.yaml

report-summary:
	uv run -m src.cli.report_summary --config configs/compare_experiments.yaml --top 10

report-summary-diagnosis:
	uv run -m src.cli.report_summary --config configs/compare_experiments_diagnosis.yaml --top 10

predict-example:
	uv run -m src.cli.predict --config configs/iris_mlp_zscore.yaml --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2

predict-example-diagnosis:
	uv run -m src.cli.predict --config configs/diagnosis_mlp_zscore.yaml --feature param1=0.9654818714 --feature param2=-0.5412880001 --feature param3=-0.8462698572

pipeline:
	uv run -m src.cli.fetch_data --config configs/iris_mlp_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/iris_mlp_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/iris_mlp_minmax.yaml
	uv run -m src.cli.run_experiment --config configs/iris_linear_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/iris_linear_minmax.yaml
	uv run -m src.cli.visualize --config configs/iris_mlp_zscore.yaml --kind raw
	uv run -m src.cli.compare_experiments --config configs/compare_experiments.yaml

fetch-diagnosis:
	uv run -m src.cli.fetch_data --config configs/diagnosis_mlp_zscore.yaml

run-diagnosis-mlp-zscore:
	uv run -m src.cli.run_experiment --config configs/diagnosis_mlp_zscore.yaml

run-diagnosis-mlp-minmax:
	uv run -m src.cli.run_experiment --config configs/diagnosis_mlp_minmax.yaml

run-diagnosis-linear-zscore:
	uv run -m src.cli.run_experiment --config configs/diagnosis_linear_zscore.yaml

run-diagnosis-linear-minmax:
	uv run -m src.cli.run_experiment --config configs/diagnosis_linear_minmax.yaml

run-diagnosis-shallow-zscore:
	uv run -m src.cli.run_experiment --config configs/diagnosis_shallow_zscore.yaml

pipeline-diagnosis:
	uv run -m src.cli.fetch_data --config configs/diagnosis_mlp_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/diagnosis_mlp_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/diagnosis_mlp_minmax.yaml
	uv run -m src.cli.run_experiment --config configs/diagnosis_linear_zscore.yaml
	uv run -m src.cli.run_experiment --config configs/diagnosis_linear_minmax.yaml
	uv run -m src.cli.run_experiment --config configs/diagnosis_shallow_zscore.yaml
	uv run -m src.cli.visualize --config configs/diagnosis_mlp_zscore.yaml --kind raw
	uv run -m src.cli.compare_experiments --config configs/compare_experiments_diagnosis.yaml

test:
	uv run pytest

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

pyright:
	uv run pyright
