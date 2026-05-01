"""Sphere optimization demo for task00."""

from pathlib import Path

from matplotlib import pyplot as plt
from pyswarms.utils.functions import single_obj as fx

from common.pso_utils import (
    make_global_best_pso,
    remove_report_log,
    save_cost_history_plot,
    working_directory,
)

# Set-up hyperparameters
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Call instance of PSO
with working_directory(OUTPUT_DIR):
    optimizer = make_global_best_pso(
        n_particles=10,
        dimensions=2,
        options=options,
        lower_bound=1.0,
        upper_bound=2.0,
    )

# Perform optimization
with working_directory(OUTPUT_DIR):
    cost, pos = optimizer.optimize(fx.sphere, iters=200)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
save_cost_history_plot(
    cost_history, OUTPUT_DIR / "sphere_cost_history.png", title="Cost History", dpi=120
)
remove_report_log()
plt.show()
