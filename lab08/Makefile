.PHONY: help setup install hooks run-all run-task01 run-task01-config run-task02 run-task03 test clean

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TASK01_DEFAULT_CONFIG := task01/configs/default.yaml
TASK01_CONFIG ?= $(TASK01_DEFAULT_CONFIG)

help:
	@echo "Lab08 - Particle Swarm Optimization and Ant Colony Optimization Tasks"
	@echo "================================"
	@echo "Available commands:"
	@echo "  make setup             - Setup uv environment and install dependencies"
	@echo "  make install           - Alias for setup"
	@echo "  make hooks             - Install pre-commit hooks (requires pre-commit)"
	@echo "  make run-all           - Run all tasks"
	@echo "  make run-task01        - Run Task 01 (Alloy durability - swarm) with default config"
	@echo "  make run-task01-all    - Run Task 01 with all configs in task01/configs/"
	@echo "  make run-task01-config - TASK01_CONFIG=task01/configs/<name>.yaml - Run Task 01 with selected config"
	@echo "  make run-task02        - Run Task 02 (Traveling Salesman - ants)"
	@echo "  make run-task03        - Run Task 03 (Maze solving - swarm and ants)"
	@echo "  make test              - Run tests"
	@echo "  make clean             - Clean up generated files"

setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

install: setup

run-all:
	@echo "Running Task 01..."
	uv run -m task01.src.alloy_pso --config $(TASK01_DEFAULT_CONFIG)
	@echo "\nRunning Task 02..."
	uv run -m task02.src.tsp_aco
	@echo "\nRunning Task 03..."
	uv run -m task03.src.maze_aco
	uv run -m task03.src.maze_pso

run-task00:
	uv run -m task00.aco_tsp
	uv run -m task00.pso_animation
	uv run -m task00.pso_sphere

run-task01:
	uv run -m task01.src.alloy_pso --config $(TASK01_DEFAULT_CONFIG)

run-task01-all:
	@for config in $(ROOT_DIR)task01/configs/*.yaml; do \
		echo "Running Task 01 with config: $$config"; \
		uv run -m task01.src.alloy_pso --config "$$config"; \
		echo ""; \
	done

run-task01-config:
	uv run -m task01.src.alloy_pso --config $(TASK01_CONFIG)

run-task02:
	uv run -m task02.src.tsp_aco

run-task03:
	uv run -m task03.src.maze_aco
	uv run -m task03.src.maze_pso

run-task03-all:
	@for config in $(ROOT_DIR)task03/configs/*.yaml; do \
		echo "Running Task 03 with config: $$config"; \
		uv run -m task03.src.maze_aco --config "$$config"; \
		echo ""; \
	done

	@for config in $(ROOT_DIR)task03/configs/*.yaml; do \
		echo "Running Task 03 with config: $$config"; \
		uv run -m task03.src.maze_pso --config "$$config"; \
		echo ""; \
	done

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
