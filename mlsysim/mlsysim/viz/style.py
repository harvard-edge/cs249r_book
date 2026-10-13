"""Publication plotting style: the single source for every matplotlib figure.

Palette, font stack, font sizes, grid, and figure size live here and nowhere
else. ``mlsysim.viz`` plotters and ``binder.tools.figures.style`` (which
re-exports this module) both draw from it, so a figure cell inherits the same
text sizes whichever import it uses.

Printed size. Body figures are scaled to the text block, so the printed size of
plot text is its font size times the print scale, ``PRINT_TEXT_WIDTH_IN`` over
the width of the tightly cropped image. ``BODY_FIGSIZE`` lands a single panel in
``PRINT_SCALE_RANGE``; ``MIN_FONT_PT`` is the smallest size any cell may set.
``binder check figures --scope print-scale`` reads these constants and measures
every rendered figure against them.
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
    "secondary": "#666666",  # Medium gray secondary text / lines
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

# Figure sizing (see the module docstring). Text width is measured from the
# printed book's text block.
PRINT_TEXT_WIDTH_IN = 5.38
PRINT_SCALE_RANGE = (0.80, 1.05)
BODY_FIGSIZE = (6.2, 3.8)
MIN_FONT_PT = 9

# Bundled Helvetica-metric font, registered at runtime so figures render in a
# consistent Helvetica clone WITH A REAL BOLD on every platform (local macOS,
# local Linux, CI container) regardless of installed system fonts. Without this,
# matplotlib falls back per-platform (macOS Helvetica.ttc has no reachable bold
# face; a bare Linux/CI box drops all the way to DejaVu Sans), so bold weight
# and text metrics drift between local previews and the published build.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_BUNDLED_FONTS_REGISTERED = False


def _register_bundled_fonts() -> None:
    """Register the bundled TeX Gyre Heros faces with matplotlib (idempotent)."""
    global _BUNDLED_FONTS_REGISTERED
    if _BUNDLED_FONTS_REGISTERED or not _MATPLOTLIB_AVAILABLE:
        return
    try:
        import matplotlib.font_manager as fm

        if os.path.isdir(_FONT_DIR):
            known = {getattr(f, "fname", None) for f in fm.fontManager.ttflist}
            for name in sorted(os.listdir(_FONT_DIR)):
                if name.lower().endswith((".otf", ".ttf")):
                    path = os.path.join(_FONT_DIR, name)
                    if path not in known:
                        fm.fontManager.addfont(path)
    except Exception:
        # Never let a font-registration hiccup break figure generation;
        # matplotlib will fall through the sans-serif stack below.
        pass
    _BUNDLED_FONTS_REGISTERED = True


def set_book_style() -> None:
    """Apply the global MLSysBook matplotlib style."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plot generation.")
    _register_bundled_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "TeX Gyre Heros",  # bundled Helvetica-metric clone (has bold)
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


def setup_plot(figsize=BODY_FIGSIZE):
    """Create a matplotlib figure/axis using the MLSysBook style."""
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt


def bar_compare(labels, values, *, title=None, ylabel=None, figsize=(BODY_FIGSIZE[0], 3.4), color=None):
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
            fontsize=MIN_FONT_PT,
            color=colors["primary"],
        )
    return finalize_web_figure(fig), ax
