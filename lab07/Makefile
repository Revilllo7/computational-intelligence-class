.PHONY: help install run-all run-task01 run-task02 run-task03 test clean

help:
	@echo "Lab07 - Genetic Algorithms Tasks"
	@echo "================================"
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make run-all      - Run all tasks"
	@echo "  make run-task01   - Run Task 01 (Knapsack Problem)"
	@echo "  make run-task02   - Run Task 02 (Alloy Durability)"
	@echo "  make run-task03   - Run Task 03 (Maze Solving)"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up generated files"

install:
	pip install -e .
	pip install -e ".[dev]"

run-all:
	@echo "Running Task 01..."
	python -m task01.src.knapsack_ga
	@echo "\nRunning Task 02..."
	python -m task02.src.alloy_ga
	@echo "\nRunning Task 03..."
	python -m task03.src.maze_ga

run-task01:
	python -m task01.src.knapsack_ga

run-task02:
	python -m task02.src.alloy_ga

run-task03:
	python -m task03.src.maze_ga

test:
	pytest -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
