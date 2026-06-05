"""Publication plotting style for MLSysBook figures.

This module owns book-level matplotlib policy: palette, font stack, grid style,
and the small setup helper used by generated figure blocks. Simulator-aware
plots remain in ``mlsysim.viz``; book production style belongs here.
"""

from __future__ import annotations

import os

try:
    import matplotlib

    if "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    _MATPLOTLIB_AVAILABLE = False


COLORS = {
    "crimson": "#A51C30",  # Harvard Crimson
    "primary": "#333333",  # Dark gray text
    "grid": "#CCCCCC",  # Light gray
    "BlueLine": "#006395",
    "BlueL": "#D1E6F3",
    "BlueFill": "#D6EAF8",
    "RedLine": "#CB202D",
    "RedL": "#F5D2D5",
    "RedFill": "#F2D7D5",
    "GreenLine": "#008F45",
    "GreenL": "#D4EFDF",
    "GreenFill": "#D5F5E3",
    "OrangeLine": "#E67817",
    "OrangeL": "#FCE4CC",
    "VioletLine": "#7E317B",
    "VioletL": "#E6D4E5",
    "BrownLine": "#78492A",
    "BrownL": "#E3D3C8",
    "YellowFill": "#FEF9E0",
}

WEB_FIG_DPI = 120


def set_book_style() -> None:
    """Apply the global MLSysBook matplotlib style."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plot generation.")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "text.color": COLORS["primary"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def finalize_web_figure(fig):
    """Match Quarto web defaults while publication figures keep print DPI."""
    fig.set_dpi(WEB_FIG_DPI)
    return fig


def setup_plot(figsize=(8, 5)):
    """Create a matplotlib figure/axis using the MLSysBook style."""
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt


def bar_compare(labels, values, *, title=None, ylabel=None, figsize=(8, 4), color=None):
    """Render a compact bar comparison with the MLSysBook style."""
    fig, ax, colors, _ = setup_plot(figsize=figsize)
    bar_color = color or colors["BlueLine"]
    bars = ax.bar(labels, values, color=bar_color, alpha=0.8)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:g}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["primary"],
        )
    return finalize_web_figure(fig), ax
