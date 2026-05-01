"""Contour animation demo for task00."""

from pathlib import Path

import numpy as np
from pyswarms.utils.functions import single_obj as fx

from common.pso_utils import (
    make_global_best_pso,
    remove_report_log,
    save_contour_animation,
    working_directory,
)

options = {"c1": 0.5, "c2": 0.3, "w": 0.5}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

with working_directory(OUTPUT_DIR):
    optimizer = make_global_best_pso(
        n_particles=10,
        dimensions=2,
        options=options,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    optimizer.optimize(fx.sphere, iters=50)

save_contour_animation(
    np.asarray(optimizer.pos_history, dtype=float),
    fx.sphere,
    OUTPUT_DIR / "plot0.gif",
    mark=(0, 0),
    fps=10,
)
remove_report_log()
