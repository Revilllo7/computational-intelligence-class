.PHONY: help setup install hooks run-all run-task01 run-task02 run-task03 test clean

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

help:
	@echo "Lab07 - Genetic Algorithms Tasks"
	@echo "================================"
	@echo "Available commands:"
	@echo "  make setup        - Setup uv environment and install dependencies"
	@echo "  make install      - Alias for setup"
	@echo "  make hooks        - Install pre-commit hooks (requires pre-commit)"
	@echo "  make run-all      - Run all tasks"
	@echo "  make run-task01   - Run Task 01 (Knapsack Problem)"
	@echo "  make run-task02   - Run Task 02 (Alloy Durability)"
	@echo "  make run-task03   - Run Task 03 (Maze Solving)"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up generated files"

setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

install: setup

run-all:
	@echo "Running Task 01..."
	uv run -m task01.src.knapsack_ga
	@echo "\nRunning Task 02..."
	uv run -m task02.src.alloy_ga
	@echo "\nRunning Task 03..."
	uv run -m task03.src.maze_ga

run-task01:
	uv run -m task01.src.knapsack_ga

run-task02:
	uv run -m task02.src.alloy_ga

run-task03:
	uv run -m task03.src.maze_ga

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
