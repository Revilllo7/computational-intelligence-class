# Task 01: Knapsack Problem (Thief Problem)

## Problem setup

- Capacity: 25
- Target value: 1630
- Chromosome encoding: binary vector (0/1), one gene per item
- Source data: `data/items.py` dictionary, transformed into aligned arrays (`ids`, `names`, `values`, `weights`) for efficient vectorized fitness

## Method

The implementation uses PyGAD with a factory function that creates a fresh `pygad.GA` instance for every run.
Fitness is based on value maximization with a hard feasibility pressure:

- If total weight `<= 25`: fitness is the total value.
- If total weight `> 25`: apply strong linear penalty `fitness = value - 1000 * overweight`.

An early-stop callback terminates evolution when a feasible solution with value `>= 1630` is found.

## Known optimal feasible set

The validated optimal combination for the constraint is:

- item2 + item3 + item5 + item7 + item8 + item10

Totals:

- Total value = 1630
- Total weight = 25

## Artifacts

Generated in `analysis/`:

- `single_run_fitness.png` (convergence / optimization curve)
- `single_run_summary.json` (single-run structured summary)
- `single_run_selected_items.csv` (selected items from best single run)
- `ten_run_results.csv` (10 independent runs with runtime and success flag)

## Single-run outcome

Best chromosome and selected items are printed to the console together with:

- Total value
- Total weight
- Runtime

## 10-run statistics

The script runs exactly 10 independent experiments (new GA instance each time) and prints:

- Number of runs that reached value 1630
- Success percentage over 10 runs
- Average runtime over successful runs only

Observed result from the current run:

- Success count: 10/10
- Success percentage: 100.00%
- Average successful runtime: 0.014621 s

Values are reproducible from the generated `analysis/ten_run_results.csv` file.

## Convergence discussion

The optimization curve in `analysis/single_run_fitness.png` should show a fast early increase in best fitness and then stabilization near the constrained optimum. When the curve reaches the target feasible value, the early-stop callback ends the run to avoid unnecessary generations.
