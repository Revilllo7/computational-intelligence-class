setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

install: setup

run-task01:
	uv run -m task01.src.breakout
	uv run -m task01.src.cart_pole
	uv run -m task01.src.frozen_lake
	uv run -m task01.src.half_cheetah
	uv run -m task01.src.lunar_lander
	uv run -m task01.src.pendulum
	uv run -m task01.src.smart_cart_pole
	uv run -m task01.src.smart_frozen_lake

run-task02:
	uv run -m task02.src.fuzzy_tips

run-task03:
	uv run -m task03.src.main

run-task04:
	uv run -m task04.src.main

test:
	uv run pytest

pytest: test

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

pyright:
	uv run pyright

checks:
	uv run ruff format . --check
	uv run ruff check .
	uv run pyright
	uv run pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
