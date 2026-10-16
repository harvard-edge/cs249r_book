"""Slide-size data plots in the book's color contract.

Every lecture's plot generator imports this, so data slides look the same
across the course: large type, direct labels instead of legends, the value
printed on each bar, and the book's semantic colors (memory blue, compute
orange, data green, interconnect violet, danger red, neutral gray).

Plots are written as SVG with text kept as text; the slides Makefile turns
them into PDF. A plot revealed in steps is written as name_step1.svg ...
name_stepN.svg, every step with the same axes so nothing moves.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {
    "memory": "#006395", "memory_l": "#D1E6F3",
    "compute": "#E67817", "compute_l": "#FCE4CC",
    "data": "#008F45", "data_l": "#D4EFDF",
    "network": "#7E317B", "network_l": "#E6D4E5",
    "danger": "#CB202D", "danger_l": "#F5D2D5",
    "neutral": "#666666", "neutral_l": "#DDDDDD",
    "text": "#1A1A2E", "grid": "#CCCCCC",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": C["text"],
    "xtick.color": C["text"], "ytick.color": C["text"],
    "svg.fonttype": "none",
})

# Drawn at the size it is shown, so type stays legible: the slide body is
# about 15 x 6.5 cm (5.9 x 2.6 in). A full-width plot is 5.8 x 2.4 in;
# a short two-bar plot 5.8 x 1.7 in. Text is 11 to 13 pt at that size.
SIZE = (5.8, 2.4)
SHORT = (5.8, 1.7)


def save(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15,
                transparent=False, facecolor="white")
    plt.close(fig)


def hbars(path, rows, unit, log=True, xlabel=None, step=None, note=None,
          value_fmt="{:,.0f}", size=SIZE, vlines=None):
    """Horizontal bars, one per row, top to bottom.

    rows: list of (label, value, color_key) or (label, value, color_key,
    value_label) when the printed value needs words ("3 MW").
    vlines: optional reference lines, list of (x, label, color_key), such
    as a 1 ms deadline or a budget; drawn dashed with the label at the top.
    step: show only the first
    `step` rows (the rest keep their slot so the axes do not move).
    """
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    vals = [r[1] for r in rows]
    for i, r in enumerate(rows):
        label, v, ck = r[:3]
        shown_text = r[3] if len(r) > 3 else f"{value_fmt.format(v)} {unit}"
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, v if shown else 0, color=C[ck], height=0.62,
                alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13, color=C["text"],
                fontweight="bold")
        # Every step draws every value label (hidden ones invisible), so the
        # tight bounding box, and the figure size, are the same in every step.
        ax.text(v * (1.15 if log else 1.01) if log else v + max(vals) * 0.01,
                y, shown_text, va="center", fontsize=13, color=C[ck],
                fontweight="bold", alpha=1.0 if shown else 0.0)
    ax.set_yticks([])
    for x, label, ck in (vlines or []):
        ax.axvline(x, color=C[ck], lw=1.5, ls="--", zorder=1)
        ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=10,
                fontweight="bold", va="bottom", ha="left")
    if log:
        ax.set_xscale("log")
        lo = min(vals) / 3
        hi = max(vals) * 40
        ax.set_xlim(lo, hi)
    else:
        ax.set_xlim(0, max(vals) * 1.3)
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if note:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9,
                 color=C["neutral"])
    save(fig, path)


def hbars_steps(stem, rows, unit, **kw):
    """Write stem_step1.svg ... stem_stepN.svg, one more bar per step."""
    for k in range(1, len(rows) + 1):
        hbars(f"{stem}_step{k}.svg", rows, unit, step=k, **kw)


def roofline_steps(stem, points, size=SIZE, ridge=100.0, peak=1.0,
                   xlabel="Arithmetic intensity (operations per byte moved)",
                   ylabel="Achieved performance", note="Schematic",
                   xticks=None, ridge_label=None):
    """A schematic roofline (memory slope blue, compute ceiling orange) with
    workload points revealed one per step. points: list of (label, intensity,
    color_key). Axes are log-log and unnumbered unless the caller adds ticks;
    the shape, not the numbers, carries the point."""
    import numpy as np
    xs = np.logspace(-2, 4.3, 300)
    roof = np.minimum(peak, xs / ridge * peak)
    for k in range(0, len(points) + 1):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(xs[xs <= ridge], roof[xs <= ridge], color=C["memory"], lw=3)
        ax.plot(xs[xs >= ridge], roof[xs >= ridge], color=C["compute"], lw=3)
        ax.fill_between(xs[xs <= ridge], 1e-4, roof[xs <= ridge], color=C["memory_l"], alpha=0.6)
        ax.fill_between(xs[xs >= ridge], 1e-4, roof[xs >= ridge], color=C["compute_l"], alpha=0.6)
        # regime names sit low in each shaded region, away from the points
        ax.text(ridge / 15, 1.3e-3, "Memory bound", color=C["memory"], fontsize=12,
                fontweight="bold", ha="center")
        ax.text(ridge * 2.0, 1.3e-3, "Compute bound", color=C["compute"],
                fontsize=12, fontweight="bold")
        for label, x, ck in points[:k]:
            y = min(peak, x / ridge * peak)
            ax.plot([x], [y], "o", ms=11, color=C[ck], mec="white", mew=1.5, zorder=5)
            # label below and right of the point, inside the shaded region
            below = x >= ridge
            below = x >= ridge
            ax.annotate(label, (x, y), xytext=(0, -22) if below else (-6, 12),
                        textcoords="offset points", ha="center" if below else "right",
                        fontsize=12, fontweight="bold", color=C[ck])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 2e4)
        ax.set_ylim(5e-4, 3)
        ax.minorticks_off()
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:g}" for v in xticks])
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        if ridge_label:
            ax.axvline(ridge, color=C["neutral"], lw=1, ls="--")
            ax.text(ridge * 1.08, 0.02, ridge_label, color=C["neutral"], fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if note:
            ax.text(0.01, 0.97, note, transform=ax.transAxes, ha="left",
                    va="top", fontsize=9, color=C["neutral"])
        save(fig, f"{stem}_step{k}.svg")
