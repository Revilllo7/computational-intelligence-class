"""Reusable linguistic variable plotting functions."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Headless backend — safe on Windows without a display server
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_linguistic_variable(
    lv,
    var_name: str,
    save_path: str,
    title: str | None = None,
    figsize: tuple = (9, 4),
    colors: list | None = None,
) -> str:
    """
    Plot all fuzzy sets of a Simpful LinguisticVariable and save to *save_path*.
    Parameters
    ----------
    lv : simpful.LinguisticVariable
        The linguistic variable to plot.
    var_name : str
        Human-readable name shown on the x-axis label.
    save_path : str
        Absolute or relative path where the PNG will be saved.
    title : str, optional
        Custom plot title.  Defaults to "Linguistic variable: <var_name>".
    figsize : tuple
        Matplotlib figure size.
    colors : list, optional
        List of colours for each fuzzy set.  If None a pleasant palette is used.
    Returns
    -------
    str
        Path where the figure was saved (same as *save_path*).
    """
    default_colors = [
        "#2483ef",  # blue
        "#de9904",  # amber
        "#19b996",  # teal
        "#e04b5f",  # coral
        "#c372e3",  # purple
        "#1abc3d",  # emerald
    ]
    if colors is None:
        colors = default_colors
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    # Determine universe of discourse bounds
    universe = lv._universe_of_discourse  # [min, max]
    min_val, max_val = universe[0], universe[1]
    x = np.linspace(min_val, max_val, 500)
    legend_patches = []
    for idx, fuzzy_set in enumerate(lv._FSlist):
        color = colors[idx % len(colors)]
        membership_values = [fuzzy_set.get_value(xi) for xi in x]
        ax.plot(x, membership_values, color=color, linewidth=2.5)
        ax.fill_between(x, membership_values, alpha=0.25, color=color)
        legend_patches.append(mpatches.Patch(color=color, label=fuzzy_set._term))
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        framealpha=0.4,
        facecolor="#0f3460",
        edgecolor="#4e9af1",
        labelcolor="white",
    )
    ax.set_xlabel(var_name, color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Membership", color="#c9d1d9", fontsize=12)
    ax.set_title(
        title if title else f"Linguistic variable: {var_name}",
        color="#e6edf3",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fuzzy_plots] Saved plot → {save_path}")
    return save_path
