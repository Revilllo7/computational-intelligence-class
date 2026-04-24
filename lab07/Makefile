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
	cd $(ROOT_DIR) && uv sync --group dev
	cd $(ROOT_DIR) && uv run pre-commit install
	cd $(ROOT_DIR) && uv run pre-commit install --hook-type pre-push
	cd $(ROOT_DIR) && uv run pre-commit install --hook-type commit-msg

install: setup

run-all:
	@echo "Running Task 01..."
	cd $(ROOT_DIR) && uv run -m task01.src.knapsack_ga
	@echo "\nRunning Task 02..."
	cd $(ROOT_DIR) && uv run -m task02.src.alloy_ga
	@echo "\nRunning Task 03..."
	cd $(ROOT_DIR) && uv run -m task03.src.maze_ga

run-task01:
	cd $(ROOT_DIR) && uv run -m task01.src.knapsack_ga

run-task02:
	cd $(ROOT_DIR) && uv run -m task02.src.alloy_ga

run-task03:
	cd $(ROOT_DIR) && uv run -m task03.src.maze_ga

test:
	cd $(ROOT_DIR) && uv run pytest -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
