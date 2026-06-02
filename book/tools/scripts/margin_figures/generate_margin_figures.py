#!/usr/bin/env python3
"""Generate the committed MLSysBook margin-figure SVG assets.

The output is intentionally SVG. The figures are authored at the native
margin-column scale, use the book Helvetica stack through ``mlsysim.viz``, and
reuse the canonical margin-device vocabulary documented in
``.claude/rules/margin-figures.md``.

LLM/editor notes:
    * Treat this file plus ``margin_devices.py`` as the source of truth. Do not
      hand-edit generated SVGs.
    * A margin figure must serve the paragraph beside it. The chapter placement
      lives in QMD and, for curated/generic figures, in
      ``book/tools/audit/margin_figure_opportunities.yml`` and
      ``margin_figure_decisions.yml``. The stable asset name is the candidate id
      with hyphens changed to underscores.
    * Quantitative bars must be honest. If a rectangle length encodes a value,
      it must be proportional on a declared linear/log/normalized scale. Do not
      add visual minimum widths to numeric bars. If a relationship is symbolic
      rather than numeric, draw it as labels, formulas, dots, or arrows instead
      of a bar chart.
    * Text is rendered through ``new_fig()`` with ``svg.fonttype='path'`` so
      labels become vector outlines. QA should still check the SVGs for live
      ``<text>`` or ``font-family`` residue before publication.

Usage:
    MPLCONFIGDIR=/tmp/mplconfig python3 book/tools/scripts/margin_figures/generate_margin_figures.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mlsysim"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from margin_devices import (  # noqa: E402
    C,
    COMP,
    DATA,
    GRID,
    INK,
    MEM,
    NET,
    RED,
    REDFILL,
    SEL,
    TIME,
    blast,
    dam,
    ironbar,
    knee,
    ladder,
    new_fig,
    roofline,
    save,
    sparkline,
    taxonomy,
)

CONTENTS = ROOT / "book/quarto/contents"
AUDIT_DIR = ROOT / "book/tools/audit"
OPPORTUNITIES = AUDIT_DIR / "margin_figure_opportunities.yml"
DECISIONS = AUDIT_DIR / "margin_figure_decisions.yml"


def target(chapter: str, name: str) -> str:
    """Return the logical PNG path consumed by margin_devices.save()."""
    return str(CONTENTS / chapter / "images/png" / f"{name}.png")


def write(fig, chapter: str, name: str) -> None:
    save(fig, target(chapter, name))


def curated_asset_name(candidate_id: str) -> str:
    return candidate_id.replace("-", "_")


def rect(ax, x, y, w, h, color, ec="white", lw=0.5, alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=ec, lw=lw, alpha=alpha))


def clean(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def make_ladder(chapter, name, tiers, *, domain="memory", wall=False, style="bars"):
    fig, ax = new_fig("hierarchy-ladder")
    ladder(ax, tiers, domain=domain, wall=wall, style=style)
    write(fig, chapter, name)


def make_knee(chapter, name, *, knee_frac=0.72, style="shaded", pct_label=None):
    fig, ax = new_fig("scale-anchor")
    knee(ax, knee_frac=knee_frac, style=style, pct_label=pct_label)
    write(fig, chapter, name)


def make_labeled_knee(
    chapter,
    name,
    *,
    knee_frac=0.72,
    style="shaded",
    pct_label=None,
    word_label="threshold",
):
    fig, ax = new_fig("scale-anchor")
    knee(ax, knee_frac=knee_frac, style=style, pct_label=pct_label)
    if style == "twotone":
        ax.text(18, 4.0, "safe", ha="center", va="center", color=DATA, fontsize=5.2)
        ax.text(
            82,
            19.5,
            word_label,
            ha="center",
            va="center",
            color=RED,
            fontsize=5.2,
            fontweight="bold",
        )
    elif style == "dashed":
        ax.text(knee_frac * 100 + 4, 23.5, word_label, ha="left", va="center", color=RED, fontsize=5.0)
    else:
        ax.text(
            min(knee_frac * 100 + 14, 90),
            23.5,
            word_label,
            ha="center",
            va="center",
            color=RED,
            fontsize=5.1,
            fontweight="bold",
        )
    write(fig, chapter, name)


def make_sparkline(chapter, name, *, threat=True, style="gap", steep=1.8, saturating=False, endpoints=None):
    fig, ax = new_fig("sparkline-trend")
    sparkline(ax, threat=threat, style=style, steep=steep, saturating=saturating, endpoints=endpoints)
    write(fig, chapter, name)


def make_roofline(chapter, name, *, ridge=60.0, dot_ai=6.0):
    fig, ax = new_fig("thumbnail-roofline")
    roofline(ax, ridge=ridge, dot_ai=dot_ai)
    write(fig, chapter, name)


def make_roofline_points(chapter, name, *, ridge=60.0, points=None, arrow=False):
    """Draw a margin roofline with one or more labeled operating points."""
    points = points or [("workload", 6.0, INK)]
    fig, ax = new_fig("thumbnail-roofline")
    ais = [p[1] for p in points]
    lo, hi = min([ridge] + ais), max([ridge] + ais)
    xmin, xmax = lo / 5.0, hi * 5.0
    x = np.logspace(np.log10(xmin), np.log10(xmax), 240)
    y = np.minimum(x / ridge, 1.0)
    m = x < ridge
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x[m], y[m], color=MEM, lw=1.45)
    ax.plot(x[~m], y[~m], color=COMP, lw=1.45)
    ax.axvline(ridge, color=GRID, ls="--", lw=0.55)
    yvals = [min(ai / ridge, 1.0) for ai in ais]
    if arrow and len(points) >= 2:
        ax.annotate(
            "",
            xy=(ais[-1], yvals[-1]),
            xytext=(ais[0], yvals[0]),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.6),
        )
    for label, ai, color in points:
        yy = min(ai / ridge, 1.0)
        ax.plot(ai, yy, "o", color=color, ms=3.4, zorder=4)
        if yy < 0.78:
            ax.text(
                ai,
                yy * 1.38,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=5.0,
                fontweight="bold",
            )
        else:
            ax.annotate(
                label,
                xy=(ai, yy),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                color=color,
                fontsize=5.0,
                fontweight="bold",
            )
    ymin = max(min(yvals + [xmin / ridge]) / 3.0, 1e-4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, 2.0)
    clean(ax)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(axis="both", which="both", length=0)
    write(fig, chapter, name)


def make_ironbar(chapter, name, segs, *, dom=1, style="stacked"):
    fig, ax = new_fig("iron-law-bar")
    ironbar(ax, segs=segs, dom=dom, style=style)
    write(fig, chapter, name)


def make_dam(chapter, name, *, focus="all", vol="vol1", style="triangle"):
    fig, ax = new_fig("dam-locator")
    dam(ax, focus=focus, vol=vol, style=style)
    write(fig, chapter, name)


def make_blast(chapter, name, *, n=5, style="fan"):
    fig, ax = new_fig("blast-radius")
    blast(ax, n=n, style=style)
    write(fig, chapter, name)


def margin_axes(device="other-new", figsize=None):
    fig, ax = new_fig(device)
    if figsize is not None:
        fig.set_size_inches(*figsize, forward=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    clean(ax)
    return fig, ax


def taxonomy_quadrant(chapter, name, *, selected=(0, 1), xlabel="", ylabel="", labels=None):
    fig, ax = margin_axes("taxonomy-mini")
    x0, y0, c, gap = 0.25, 0.18, 0.28, 0.035
    labels = labels or {}
    for col in range(2):
        for row in range(2):
            on = (col, row) == selected
            x = x0 + col * (c + gap)
            y = y0 + row * (c + gap)
            rect(ax, x, y, c, c, SEL if on else "#EEEEEE", ec="white", lw=0.7)
            text = labels.get((col, row), "")
            if text:
                ax.text(x + c / 2, y + c / 2, text, ha="center", va="center",
                        color="white" if on else "#777777", fontsize=5.0, fontweight="bold")
    if xlabel:
        ax.text(x0 + c + gap / 2, 0.07, xlabel, ha="center", va="center", color=INK, fontsize=5.0)
    if ylabel:
        ax.text(0.08, y0 + c + gap / 2, ylabel, ha="center", va="center", color=INK, fontsize=5.0, rotation=90)
    write(fig, chapter, name)


def latency_budget():
    fig, ax = margin_axes("iron-law-bar", figsize=(1.25, 0.35))
    x, y, w, h = 0.04, 0.35, 0.92, 0.28
    segs = [("pre", 0.38, GRID, "#666666"), ("infer", 0.42, COMP, "white"), ("post", 0.20, GRID, "#666666")]
    cur = x
    for label, frac, color, tc in segs:
        sw = w * frac
        rect(ax, cur, y, sw, h, color)
        ax.text(cur + sw / 2, y + h / 2, label, ha="center", va="center", color=tc, fontsize=5.2)
        cur += sw
    write(fig, "vol1/model_serving", "model_serving_latency_budget_bar")


def escalation_curve():
    fig, ax = margin_axes("scale-anchor", figsize=(1.18, 0.82))
    costs = [1, 2, 4, 8, 16, 32]
    xs = [0.12 + i * 0.15 for i in range(6)]
    ys = [0.16 + 0.56 * (math.log2(c) / 5.0) ** 1.08 for c in costs]
    ax.plot([0.08, 0.92], [0.12, 0.12], color=GRID, lw=0.7)
    ax.plot(xs, ys, color=INK, lw=1.7)
    ax.scatter(xs[:-1], ys[:-1], s=12, color=COMP, zorder=3)
    ax.scatter(xs[-1:], ys[-1:], s=18, color=RED, zorder=4)
    ax.fill_between(xs[-2:], [0.12, 0.12], ys[-2:], color=RED, alpha=0.10)
    ax.text(xs[0] - 0.01, 0.035, "define\n1x", ha="center", va="bottom", color=INK, fontsize=5.0)
    ax.text(xs[-1] - 0.02, ys[-1] + 0.045, "monitor\n32x", ha="center", va="bottom", color=RED, fontsize=4.8)
    ax.text(0.86, 0.18, "late", ha="center", va="bottom", color=RED, fontsize=5.0)
    write(fig, "vol1/ml_workflow", "ml_workflow_constraint_cost_escalation")


def list_dots(chapter, name, items):
    fig, ax = margin_axes("taxonomy-mini")
    for i, (label, color) in enumerate(items[::-1]):
        ax.plot(0.18, i * 0.28 + 0.16, "o", color=color, ms=6)
        ax.text(0.32, i * 0.28 + 0.16, label, fontsize=5.5, va="center", color=INK)
    write(fig, chapter, name)


def before_after_quant():
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    x0, x1 = 0.26, 0.76
    ax.text(x0, 0.88, "FP32", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(x1, 0.88, "INT8", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(0.08, 0.66, "size", ha="left", va="center", color=INK, fontsize=5.1)
    ax.text(0.08, 0.30, "acc", ha="left", va="center", color=INK, fontsize=5.1)
    ax.plot([x0, x1], [0.70, 0.43], color=COMP, lw=1.8)
    ax.scatter([x0, x1], [0.70, 0.43], s=13, color=COMP, zorder=3)
    ax.text(0.62, 0.69, "4x smaller", ha="center", va="center", color=COMP, fontsize=4.8)
    ax.plot([x0, x1], [0.30, 0.29], color=MEM, lw=1.8)
    ax.scatter([x0, x1], [0.30, 0.29], s=13, color=MEM, zorder=3)
    ax.text(0.62, 0.22, "~same", ha="center", va="center", color=MEM, fontsize=4.8)
    write(fig, "vol1/model_compression", "model_compression_int8_beforeafter")


def labeled_memory_bars(chapter, name, rows, *, title=None):
    """Compatibility wrapper for memory comparisons.

    Older versions added a fixed visual width so tiny rows stayed visible, but
    that made the bars mathematically dishonest. Route through the canonical
    ladder renderer instead: it is linear for small spans and log-scaled for
    large spans, with the scale choice documented in ``margin_devices.py``.
    """
    _ = title
    make_ladder(chapter, name, rows, domain="memory", wall=False)


def simple_bar(chapter, name, segments, *, height=0.20, y=0.48):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.20, 0.42))
    x, w = 0.06, 0.88
    cur = x
    for label, frac, color, tc in segments:
        sw = w * frac
        rect(ax, cur, y, sw, height, color, ec="white")
        if label and sw > 0.16:
            ax.text(cur + sw / 2, y + height / 2, label, ha="center", va="center", color=tc, fontsize=5.3)
        elif label:
            ax.text(cur + sw / 2, y + height + 0.07, label, ha="center", va="bottom", color=INK, fontsize=4.0)
        cur += sw
    write(fig, chapter, name)


def pipeline_rows(chapter, name, rows, *, title=None):
    """Two compact stacked bars for Amdahl-style before/after comparisons.

    Both rows use the same ``max_total`` denominator, so segment widths and the
    total row lengths stay on the same millisecond scale.
    """
    fig, ax = margin_axes("iron-law-bar", figsize=(1.22, 0.70))
    max_total = max(sum(value for _, value, _ in segs) for _, segs, _ in rows)
    x, w, h = 0.18, 0.70, 0.14
    if title:
        ax.text(0.54, 0.91, title, ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold")
    for label, segs, y in rows:
        ax.text(0.07, y + h / 2, label, ha="left", va="center", color=INK, fontsize=4.9)
        cur = x
        for seg_label, value, color in segs:
            sw = w * value / max_total
            rect(ax, cur, y, sw, h, color, ec="white", lw=0.35)
            if sw > 0.18:
                ax.text(cur + sw / 2, y + h / 2, seg_label, ha="center", va="center", color="white", fontsize=4.6, fontweight="bold")
            cur += sw
        ax.text(x + w + 0.035, y + h / 2, f"{sum(value for _, value, _ in segs):.0f}ms", ha="left", va="center", color="#555555", fontsize=4.7)
    write(fig, chapter, name)


def ratio_annotation_ladder(chapter, name, tiers, *, ratio_label, domain="memory"):
    """Two or more measured tiers, with the ratio rendered as annotation text.

    Use this when the prose names both concrete quantities and a derived ratio.
    The ratio is not a tier and must not become a third bar.
    """
    fig, ax = new_fig("hierarchy-ladder")
    ladder(ax, tiers, domain=domain, wall=False)
    ratio_x, ratio_y = (0.60, 0.50) if len(tiers) == 2 else (0.64, 0.58)
    ax.text(
        ratio_x,
        ratio_y,
        ratio_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=5.0,
        fontweight="bold",
    )
    write(fig, chapter, name)


def formula_rows(chapter, name, rows, *, title=None):
    """Compact symbolic rows for formulas or ratios that should not become bars."""
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.28 + 0.24 * len(rows)))
    if title:
        ax.text(0.50, 0.91, title, ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold")
    for idx, (label, value, color) in enumerate(rows):
        y = 0.18 + (len(rows) - 1 - idx) * 0.24
        ax.plot(0.18, y, "o", color=color, ms=5.6)
        ax.text(0.30, y, label, ha="left", va="center", color=INK, fontsize=4.9)
        ax.text(0.88, y, value, ha="right", va="center", color=color, fontsize=5.1, fontweight="bold")
    write(fig, chapter, name)


def normalized_rows(chapter, name, rows, *, title=None, color=MEM, max_value=None, suffix=""):
    """Horizontal rows on one shared linear denominator."""
    max_value = max_value or max(value for _, value in rows)
    fig, ax = margin_axes("hierarchy-ladder", figsize=(1.24, 0.34 + 0.24 * len(rows)))
    x, w, h = 0.36, 0.50, 0.13
    if title:
        ax.text(0.54, 0.92, title, ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold")
    for idx, (label, value) in enumerate(rows):
        y = 0.18 + (len(rows) - 1 - idx) * 0.24
        ax.text(0.06, y + h / 2, label, ha="left", va="center", color=INK, fontsize=4.8)
        rect(ax, x, y, w * value / max_value, h, color, ec="white", lw=0.35)
        ax.text(
            x + min(w * value / max_value, w) + 0.025,
            y + h / 2,
            f"{value:g}{suffix}",
            ha="left",
            va="center",
            color="#555555",
            fontsize=4.7,
        )
    write(fig, chapter, name)


def benchmarking_tail_latency_gap():
    ratio_annotation_ladder(
        "vol1/benchmarking",
        "benchmarking_tail_latency_gap",
        [("prod p99 200ms", 200), ("bench mean 15ms", 15)],
        ratio_label="10-13.3x",
        domain="time",
    )


def vol1_conclusion_fleet_mtbf_ladder():
    from mlsysim import Systems, ureg
    from mlsysim.physics import calc_mtbf_cluster

    gpu_count = Systems.Clusters.Training_1K.total_accelerators
    gpu_mtbf_h = float(Systems.Reliability.Gpu.mttf_hours)
    cluster_mtbf_h = calc_mtbf_cluster(gpu_mtbf_h * ureg.hour, gpu_count).m_as("hour")
    gpu_mtbf_years = gpu_mtbf_h / (24 * 365)
    make_ladder(
        "vol1/conclusion",
        "vol1_conclusion_fleet_mtbf_ladder",
        [(f"1 GPU {gpu_mtbf_years:.1f}y", gpu_mtbf_h), (f"1024 GPUs {cluster_mtbf_h:.1f}h", cluster_mtbf_h)],
        domain="time",
        wall=False,
    )


def benchmarking_component_speedup_bars(candidate=None):
    model_latency = 10.0
    total_latency = 50.0
    component_speedup = 3.0
    other = total_latency - model_latency
    optimized_model = model_latency / component_speedup
    pipeline_rows(
        "vol1/benchmarking",
        "vol1_benchmarking_margin_001",
        [
            ("before", [("other", other, GRID), ("model", model_latency, COMP)], 0.58),
            ("after", [("other", other, GRID), ("model", optimized_model, COMP)], 0.28),
        ],
        title="3x -> 1.2x",
    )


def introduction_amdahl_pipeline(candidate=None):
    pipeline_rows(
        "vol1/introduction",
        "vol1_introduction_margin_004",
        [
            ("before", [("pre", 60, GRID), ("infer", 45, COMP), ("post", 25, GRID)], 0.58),
            ("after", [("pre", 60, GRID), ("infer", 15, COMP), ("post", 25, GRID)], 0.28),
        ],
        title="local speedup",
    )


def ml_systems_camera_pipeline_amdahl(candidate=None):
    pipeline_rows(
        "vol1/ml_systems",
        "vol1_ml_systems_margin_004",
        [
            ("before", [("ISP", 100, GRID), ("ML", 60, COMP), ("post", 40, GRID)], 0.58),
            ("after", [("ISP", 100, GRID), ("ML", 6, COMP), ("post", 40, GRID)], 0.28),
        ],
        title="10x -> 1.37x",
    )


def ml_systems_thermal_throttling(candidate=None):
    """Burst performance falls to a lower sustained thermal envelope."""
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.66))
    ax.plot([0.12, 0.44], [0.72, 0.72], color=RED, lw=1.55)
    ax.plot([0.44, 0.54], [0.72, 0.36], color=RED, lw=1.1)
    ax.plot([0.54, 0.90], [0.36, 0.36], color=MEM, lw=1.55)
    ax.plot([0.44], [0.72], "o", color=RED, ms=3.2)
    ax.plot([0.90], [0.36], "o", color=MEM, ms=3.2)
    ax.text(0.26, 0.83, "burst", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(0.73, 0.24, "sustain", ha="center", va="center", color=MEM, fontsize=5.0, fontweight="bold")
    ax.text(0.57, 0.57, "throttle", ha="left", va="center", color=INK, fontsize=4.7)
    write(fig, "vol1/ml_systems", "vol1_ml_systems_margin_003")


def ml_systems_edge_bandwidth_ladder(candidate=None):
    """Camera ingest demand from the chapter's BandwidthBottleneck example."""
    make_ladder(
        "vol1/ml_systems",
        "vol1_ml_systems_margin_002",
        [("raw 18.7 GB/s", 18.7), ("10G 1.25 GB/s", 1.25)],
        domain="bandwidth",
        wall=False,
    )


def training_activation_memory_ladder(candidate=None):
    from mlsysim import Hardware, Models
    from mlsysim.core.constants import GB
    from mlsysim.physics import calc_activation_memory

    gpt2 = Models.Language.GPT2
    total_act_gb = calc_activation_memory(
        n_layers=gpt2.layers,
        seq_len=1024,
        batch_size=8,
        hidden_dim=gpt2.hidden_dim,
        precision_bytes=2,
        strategy="none",
    ).m_as(GB)
    v100_gib = Hardware.Cloud.V100.memory.capacity.m_as("GiB")
    make_ladder(
        "vol1/training",
        "vol1_training_margin_001",
        [("GPT-2 acts %.1f GB" % total_act_gb, total_act_gb), ("V100 %d GiB" % round(v100_gib), v100_gib), ("MNIST 438 KB", 0.000438)],
        domain="memory",
        wall=True,
    )


def training_bandwidth_path_ladder(candidate=None):
    from mlsysim import Hardware

    v100_bw = Hardware.Cloud.V100.memory.bandwidth.m_as("GB/s")
    make_ladder(
        "vol1/training",
        "vol1_training_margin_002",
        [("HBM %.0f GB/s" % v100_bw, v100_bw), ("DRAM 75 GB/s", 75), ("Storage 1.5 GB/s", 1.5)],
        domain="bandwidth",
        wall=False,
    )


def training_flash_attention_tile_ladder(candidate=None):
    make_ladder(
        "vol1/training",
        "vol1_training_margin_003",
        [("Full 64 MB", 64), ("Tile 64 KB", 0.064)],
        domain="memory",
        wall=False,
    )


def compute_infrastructure_mtbf_ladder(candidate=None):
    from mlsysim.systems.reliability import Reliability

    gpu_mttf = float(Reliability.Gpu.mttf_hours)
    make_ladder(
        "vol2/compute_infrastructure",
        "vol2_compute_infrastructure_margin_003",
        [("1 GPU %.0fh" % gpu_mttf, gpu_mttf), ("1K GPUs %.0fh" % (gpu_mttf / 1_000), gpu_mttf / 1_000), ("10K GPUs %.0fh" % (gpu_mttf / 10_000), gpu_mttf / 10_000)],
        domain="time",
        wall=False,
    )


def compute_infrastructure_rack_power_envelope(candidate=None):
    """Linear rack-power scale: legacy air envelope versus DGX H100 rack."""
    fig, ax = margin_axes("scale-anchor", figsize=(1.22, 0.58))
    x0, w, y = 0.10, 0.78, 0.48
    max_kw = 40.0
    legacy_hi = 10.0
    dgx_kw = 33.5
    ax.plot([x0, x0 + w], [y, y], color=GRID, lw=0.75)
    ax.axvspan(x0 + w * legacy_hi / max_kw, x0 + w, color=REDFILL, alpha=0.32)
    ax.axvline(x0 + w * legacy_hi / max_kw, color=RED, lw=0.7, ls="--")
    ax.plot(x0 + w * dgx_kw / max_kw, y, "o", color=RED, ms=3.8)
    ax.text(x0 + w * legacy_hi / max_kw + 0.035, y + 0.23, "10kW\nair", ha="left", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(x0 + w * dgx_kw / max_kw - 0.02, y - 0.20, "DGX\n33kW", ha="right", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(0.68, 0.88, "rack power", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/compute_infrastructure", "vol2_compute_infrastructure_margin_002")


def distributed_training_ratio_threshold(candidate=None):
    fig, ax = margin_axes("scale-anchor", figsize=(1.18, 0.68))
    x, y, w, h = 0.12, 0.40, 0.76, 0.16
    rect(ax, x, y, w / 2, h, COMP, ec="white", lw=0.45)
    rect(ax, x + w / 2, y, w / 2, h, NET, ec="white", lw=0.45)
    ax.axvline(x + w / 2, ymin=0.22, ymax=0.78, color=RED, lw=0.75, ls="--")
    ax.text(x + w * 0.25, y + h / 2, "compute", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(x + w * 0.75, y + h / 2, "comm", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(x + w / 2 + 0.035, 0.78, "rho=1", ha="left", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(x + w * 0.22, 0.24, "ideal", ha="center", va="center", color=COMP, fontsize=4.8)
    ax.text(x + w * 0.76, 0.24, "wait", ha="center", va="center", color=RED, fontsize=4.8)
    write(fig, "vol2/distributed_training", "vol2_distributed_training_margin_001")


def distributed_training_barrier_block(candidate=None):
    fig, ax = margin_axes("blast-radius", figsize=(1.20, 0.84))
    worker_y = np.linspace(0.20, 0.80, 5)
    for idx, yy in enumerate(worker_y):
        color = RED if idx == 2 else MEM
        ax.plot(0.20, yy, "o", color=color, ms=5.6)
        ax.plot([0.30, 0.74], [yy, yy], color=GRID, lw=0.65)
    ax.plot([0.74, 0.74], [0.14, 0.86], color=RED, lw=1.0)
    ax.text(0.74, 0.91, "barrier", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(0.18, 0.07, "missing", ha="center", va="center", color=RED, fontsize=4.5)
    ax.text(0.58, 0.07, "peers wait", ha="center", va="center", color=INK, fontsize=4.5)
    write(fig, "vol2/distributed_training", "vol2_distributed_training_margin_002")


def distributed_training_energy_tax(candidate=None):
    # Representative midpoints from the adjacent prose: HBM 1-2, NVLink 5-10, IB 20-50 pJ/bit.
    make_ladder(
        "vol2/distributed_training",
        "vol2_distributed_training_margin_003",
        [("IB 35 pJ/bit", 35.0), ("NVLink 7.5", 7.5), ("HBM 1.5", 1.5)],
        domain="energy",
        wall=False,
    )


def distributed_training_bandwidth_gap(candidate=None):
    ratio_annotation_ladder(
        "vol2/distributed_training",
        "vol2_distributed_training_margin_004",
        [("NVLink 900 GB/s", 900), ("NDR 50 GB/s", 50)],
        ratio_label="18x",
        domain="bandwidth",
    )


def distributed_training_pipeline_bubble_tax():
    """Pipeline bubble bars share one 100% denominator."""
    from mlsysim.physics import calc_pipeline_bubble

    rows = [
        ("p8 m32", calc_pipeline_bubble(8, 32), 0.58),
        ("p16 m16", calc_pipeline_bubble(16, 16), 0.28),
    ]
    fig, ax = margin_axes("iron-law-bar", figsize=(1.22, 0.70))
    x, w, h = 0.34, 0.48, 0.14
    ax.text(0.54, 0.91, "bubble tax", ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold")
    for label, bubble, y in rows:
        useful = 1.0 - bubble
        ax.text(0.06, y + h / 2, label, ha="left", va="center", color=INK, fontsize=4.8)
        rect(ax, x, y, w * useful, h, COMP, ec="white", lw=0.35)
        rect(ax, x + w * useful, y, w * bubble, h, RED, ec="white", lw=0.35)
        if useful > 0.42:
            ax.text(x + w * useful / 2, y + h / 2, "work", ha="center", va="center", color="white", fontsize=4.6, fontweight="bold")
        ax.text(x + w + 0.025, y + h / 2, f"{bubble * 100:.0f}% idle", ha="left", va="center", color=RED, fontsize=4.7, fontweight="bold")
    write(fig, "vol2/distributed_training", "distributed_training_pipeline_bubble_tax")


def distributed_training_young_daly_checkpoint_curve():
    """Young-Daly checkpoint interval trade-off from the chapter worked example."""
    from mlsysim import Systems, ureg
    from mlsysim.physics import calc_young_daly_interval

    t_write_min = 5.0
    num_gpus = Systems.Clusters.Training_1K.total_accelerators
    cluster_mtbf_hr = float(Systems.Reliability.Gpu.mttf_hours) / num_gpus
    t_opt_hr = calc_young_daly_interval(t_write_min * 60 * ureg.second, cluster_mtbf_hr * 3600 * ureg.second).m_as("hour")
    safe_hr = 0.25
    sparse_hr = 8.0

    xs = np.linspace(0.18, 8.0, 160)
    overhead = (t_write_min / 60) / xs + xs / (2 * cluster_mtbf_hr)
    y = 0.14 + 0.66 * (overhead - overhead.min()) / (overhead.max() - overhead.min())
    fig, ax = margin_axes("scale-anchor", figsize=(1.22, 0.74))
    ax.plot(xs, y, color=INK, lw=1.35)
    for hour_value, label, color, text_x, yoff, ha in [
        (safe_hr, "15m", RED, 0.62, 0.13, "left"),
        (t_opt_hr, "2.9h opt", DATA, t_opt_hr, 0.16, "center"),
        (sparse_hr, "8h", GRID, sparse_hr, -0.16, "center"),
    ]:
        yy = np.interp(hour_value, xs, y)
        ax.plot(hour_value, yy, "o", color=color, ms=3.5, zorder=4)
        ax.text(text_x, yy + yoff, label, ha=ha, va="center", color=color if color != GRID else "#555555", fontsize=4.8, fontweight="bold")
    ax.text(4.0, 0.90, "checkpoint interval", ha="center", va="center", color=INK, fontsize=5.0)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 1)
    write(fig, "vol2/distributed_training", "distributed_training_young_daly_optimum")


def collective_communication_payload_shrink(candidate=None):
    ratio_annotation_ladder(
        "vol2/collective_communication",
        "vol2_collective_communication_margin_003",
        [("M", 1.0), ("M/8", 0.125), ("M/32", 0.03125)],
        ratio_label="32x",
        domain="memory",
    )


def fleet_orchestration_priority_inversion():
    fig, ax = margin_axes("other-new", figsize=(1.18, 0.86))
    nodes = [
        ("High", 0.20, 0.70, MEM),
        ("Low", 0.72, 0.70, NET),
        ("Med", 0.20, 0.28, GRID),
    ]
    for label, x, y, color in nodes:
        rect(ax, x - 0.14, y - 0.08, 0.28, 0.16, color, ec="white", lw=0.55)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=4.9, fontweight="bold")
    ax.annotate("", xy=(0.56, 0.70), xytext=(0.34, 0.70), arrowprops=dict(arrowstyle="->", color=GRID, lw=0.75))
    ax.text(0.45, 0.83, "waits", ha="center", va="center", color=INK, fontsize=4.8)
    ax.annotate("", xy=(0.59, 0.62), xytext=(0.31, 0.36), arrowprops=dict(arrowstyle="->", color=GRID, lw=0.75))
    ax.text(0.50, 0.43, "exit\nblocked", ha="center", va="center", color=INK, fontsize=4.5)
    ax.plot(0.08, 0.78, "o", color=RED, ms=4.2, zorder=5)
    write(fig, "vol2/fleet_orchestration", "fleet_orchestration_dependency_cascade")


def fleet_orchestration_failure_rate(candidate=None):
    """Failure cadence comparison without tiny-label clipping on a huge log span."""
    formula_rows(
        "vol2/fleet_orchestration",
        "vol2_fleet_orchestration_margin_001",
        [("10K GPUs", "daily", RED), ("1 GPU", "rare", GRID)],
        title="failure cadence",
    )


def fleet_orchestration_capacity_lag(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 120)
    demand = 0.18 + 0.68 * t
    capacity = 0.18 + 0.48 * (1 - np.exp(-2.2 * t))
    ax.plot(t, demand, color=RED, lw=1.45)
    ax.plot(t, capacity, color=MEM, lw=1.45)
    ax.fill_between(t, capacity, demand, where=demand > capacity, color=REDFILL, alpha=0.36)
    ax.text(0.77, 0.80, "demand", ha="center", va="center", color=RED, fontsize=4.9, fontweight="bold")
    ax.text(0.74, 0.49, "capacity", ha="center", va="center", color=MEM, fontsize=4.9, fontweight="bold")
    ax.text(0.50, 0.26, "SLO gap", ha="center", va="center", color=RED, fontsize=4.8)
    write(fig, "vol2/fleet_orchestration", "vol2_fleet_orchestration_margin_003")


def fleet_orchestration_scheduler_comparison(candidate=None):
    normalized_rows(
        "vol2/fleet_orchestration",
        "vol2_fleet_orchestration_margin_004",
        [("allocated", 500), ("active", 350), ("productive", 280)],
        title="utilization",
        color=COMP,
        suffix="",
    )


def ops_scale_embedding_update_blast():
    fig, ax = margin_axes("blast-radius", figsize=(1.18, 0.86))
    ax.plot(0.10, 0.50, "s", color=RED, ms=10)
    ax.text(0.10, 0.30, "emb", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    for i, yy in enumerate(np.linspace(0.12, 0.88, 5), 1):
        ax.annotate("", xy=(0.90, yy), xytext=(0.17, 0.50), arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.65))
        ax.plot(0.90, yy, "o", color=MEM, ms=5)
        ax.text(0.98, yy, f"M{i}", ha="left", va="center", color=INK, fontsize=4.7)
    write(fig, "vol2/ops_scale", "ops_scale_cross_model_blast")


def sustainable_ai_pue_overhead(candidate=None):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.20, 0.66))
    rows = [("1.58", 0.58, 0.56), ("1.10", 0.10, 0.26)]
    x, w, h = 0.24, 0.58, 0.14
    for label, overhead, y in rows:
        ax.text(0.07, y + h / 2, label, ha="left", va="center", color=INK, fontsize=5.0, fontweight="bold")
        rect(ax, x, y, w * (1.0 / 1.58), h, COMP, ec="white", lw=0.35)
        rect(ax, x + w * (1.0 / 1.58), y, w * (overhead / 1.58), h, GRID, ec="white", lw=0.35)
        ax.text(x + 0.03, y + h / 2, "IT", ha="left", va="center", color="white", fontsize=4.8, fontweight="bold")
        ax.text(x + w + 0.035, y + h / 2, f"+{int(round(overhead * 100))}%", ha="left", va="center", color="#555555", fontsize=4.8)
    ax.text(0.53, 0.88, "PUE overhead", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/sustainable_ai", "vol2_sustainable_ai_margin_002")


def sustainable_ai_carbon_frontier(candidate=None):
    ratio_annotation_ladder(
        "vol2/sustainable_ai",
        "vol2_sustainable_ai_margin_001",
        [("Poland 80 tCO2", 80), ("Quebec 2 tCO2", 2)],
        ratio_label="40x",
        domain="energy",
    )


def sustainable_ai_radio_energy(candidate=None):
    ratio_annotation_ladder(
        "vol2/sustainable_ai",
        "vol2_sustainable_ai_margin_004",
        [("radio bit\n250K pJ", 250_000), ("FP32 mult\n4 pJ", 4), ("INT32 add\n0.1 pJ", 0.1)],
        ratio_label="25K-125Kx",
        domain="energy",
    )


def conclusion_gain_stack(candidate=None):
    make_ladder(
        "vol2/conclusion",
        "vol2_conclusion_margin_003",
        [("Orch 10x", 10), ("Hardware 4x", 4), ("Algo 2.5x", 2.5)],
        domain="compute",
        wall=False,
    )


def conclusion_tail_latency_fanout():
    fig, ax = margin_axes("scale-anchor", figsize=(1.22, 0.58))
    x0, w, y = 0.10, 0.78, 0.48
    servers = 100
    hit_probability = 1 - 0.99**servers
    marker_x = x0 + w * hit_probability
    ax.plot([x0, x0 + w], [y, y], color=GRID, lw=0.75)
    ax.axvspan(marker_x, x0 + w, color=REDFILL, alpha=0.30)
    ax.axvline(marker_x, color=RED, lw=0.75, ls="--")
    ax.plot(x0 + w, y, "o", color=RED, ms=3.7)
    ax.text(marker_x, y + 0.21, "63%", ha="center", va="center", color=RED, fontsize=5.1, fontweight="bold")
    ax.text(x0 + w, y - 0.18, "100\nservers", ha="center", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(0.48, 0.88, "tail hit", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/conclusion", "conclusion_tail_latency_rise")


def responsible_ai_representation_tax_ladder():
    make_ladder(
        "vol2/responsible_ai",
        "responsible_ai_representation_tax_ladder",
        [("10 groups $125M", 125), ("1 group $12.5M", 12.5)],
        domain="compute",
        wall=False,
    )


def nn_computation_paradigm_ops_ladder(candidate=None):
    make_ladder(
        "vol1/nn_computation",
        "vol1_nn_computation_margin_001",
        [("NN 109K MACs", 109_184), ("HOG 8K ops", 8_000), ("rules 100", 100)],
        domain="compute",
        wall=False,
    )


def nn_computation_training_energy_ladder(candidate=None):
    from mlsysim import Hardware, Infrastructure, Models
    from mlsysim.core.constants import HOURS_PER_DAY, THOUSAND, watt

    lenet_kwh = 3 * HOURS_PER_DAY * 0.75
    a100_kw = Hardware.Cloud.A100.tdp.m_as(watt) / THOUSAND
    gpt4_mwh = Models.Language.GPT4.training_gpu_days * HOURS_PER_DAY * a100_kw / THOUSAND
    gpt4_mwh *= Infrastructure.FacilityCooling.Typical.pue
    make_ladder(
        "vol1/nn_computation",
        "vol1_nn_computation_margin_002",
        [("GPT-4 %.0f MWh" % gpt4_mwh, gpt4_mwh * THOUSAND), ("LeNet %.0f kWh" % lenet_kwh, lenet_kwh)],
        domain="energy",
        wall=False,
    )


def nn_computation_activation_logic_ladder(candidate=None):
    make_ladder(
        "vol1/nn_computation",
        "vol1_nn_computation_margin_003",
        [("sigmoid 2500", 2500), ("ReLU 50", 50)],
        domain="compute",
        wall=False,
    )


def nn_computation_mnist_roofline(candidate=None):
    from mlsysim import Hardware
    from mlsysim.core.constants import KB, THOUSAND, byte, flop

    mnist_dims = [784, 128, 64, 10]
    batch_size = 32
    bytes_per_value = 4
    weights = [mnist_dims[0] * mnist_dims[1], mnist_dims[1] * mnist_dims[2], mnist_dims[2] * mnist_dims[3]]
    biases = mnist_dims[1:]
    total_params = sum(w + b for w, b in zip(weights, biases))
    total_flops = (2 * sum(weights) * batch_size) + (sum(biases) * batch_size)
    param_kb = (total_params * bytes_per_value * byte).m_as(KB)
    act_kb = (sum(mnist_dims) * bytes_per_value * byte).m_as(KB)
    mnist_ai = (total_flops / batch_size) / ((param_kb + act_kb) * THOUSAND)
    ridge = (Hardware.Cloud.A100.compute.peak_flops / Hardware.Cloud.A100.memory.bandwidth).m_as(flop / byte)
    make_roofline_points(
        "vol1/nn_computation",
        "vol1_nn_computation_margin_004",
        ridge=ridge,
        points=[("MNIST", mnist_ai, MEM)],
    )


def frameworks_training_memory_ladder(candidate=None):
    from mlsysim import Models
    from mlsysim.core.constants import BYTES_FP32, GB

    infer_gb = Models.Vision.ResNet50.size_in_bytes(BYTES_FP32).m_as(GB)
    training_mid_gb = 12.5
    make_ladder(
        "vol1/frameworks",
        "vol1_frameworks_margin_001",
        [("train 10-15 GB", training_mid_gb), ("infer %.0f MB" % (infer_gb * 1000), infer_gb)],
        domain="memory",
        wall=False,
    )


def frameworks_stream_overlap(candidate=None):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.22, 0.66))
    x, w, h = 0.23, 0.58, 0.14
    ax.text(0.08, 0.65, "sum", ha="left", va="center", color=INK, fontsize=5.1, fontweight="bold")
    rect(ax, x, 0.58, w * 0.5, h, NET, ec="white", lw=0.35)
    rect(ax, x + w * 0.5, 0.58, w * 0.5, h, COMP, ec="white", lw=0.35)
    ax.text(x + w * 0.25, 0.65, "copy", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(x + w * 0.75, 0.65, "compute", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(0.08, 0.33, "max", ha="left", va="center", color=INK, fontsize=5.1, fontweight="bold")
    rect(ax, x, 0.26, w * 0.5, h, COMP, ec="white", lw=0.35)
    ax.text(x + w * 0.25, 0.33, "longer", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.plot([x, x], [0.22, 0.76], color=GRID, lw=0.55)
    ax.plot([x + w * 0.5, x + w * 0.5], [0.22, 0.76], color=GRID, lw=0.55, ls="--")
    ax.text(0.54, 0.88, "overlap hides copy", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol1/frameworks", "vol1_frameworks_margin_002")


def data_selection_compute_data_gap():
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 120)
    compute = 0.12 + 0.74 * (t ** 1.18)
    data = 0.12 + 0.28 * (t ** 0.95)
    ax.plot(t, compute, color=COMP, lw=1.45)
    ax.plot(t, data, color=DATA, lw=1.45)
    ax.fill_between(t, data, compute, color=REDFILL, alpha=0.28)
    ax.text(0.80, 0.82, "compute", ha="center", va="center", color=COMP, fontsize=4.9, fontweight="bold")
    ax.text(0.80, 0.39, "data", ha="center", va="center", color=DATA, fontsize=4.9, fontweight="bold")
    ax.text(0.46, 0.55, "gap", ha="center", va="center", color=RED, fontsize=4.8)
    write(fig, "vol1/data_selection", "data_selection_scaling_saturation")


def data_selection_quality_multiplier(candidate=None):
    make_ladder(
        "vol1/data_selection",
        "vol1_data_selection_margin_001",
        [("noisy 10K", 10_000), ("clean 100", 100)],
        domain="data",
        wall=False,
    )


def data_selection_echo_threshold(candidate=None):
    fig, ax = new_fig("scale-anchor")
    knee(ax, knee_frac=0.63, style="dashed", pct_label="e=R")
    ax.text(82, 23.5, "over", ha="center", va="center", color=RED, fontsize=4.8, fontweight="bold")
    write(fig, "vol1/data_selection", "vol1_data_selection_margin_003")


def model_serving_model_load_slo(candidate=None):
    from mlsysim import Hardware
    from mlsysim.core.constants import GB, ms, second

    slo_ms = 50.0
    load_ms = (10.0 / Hardware.Cloud.A100.interconnect.bandwidth.m_as(GB / second) * second).m_as(ms)
    xmax = max(320.0, load_ms * 1.04)
    fig, ax = margin_axes("scale-anchor", figsize=(1.20, 0.58))
    x0, w, y = 0.10, 0.78, 0.48
    slo_x = x0 + w * slo_ms / xmax
    load_x = x0 + w * load_ms / xmax
    ax.plot([x0, x0 + w], [y, y], color=GRID, lw=0.75)
    ax.axvspan(slo_x, x0 + w, color=REDFILL, alpha=0.36)
    ax.axvline(slo_x, color=RED, lw=0.7, ls="--")
    ax.plot(load_x, y, "o", color=RED, ms=3.8)
    ax.text(slo_x - 0.035, y + 0.21, "50ms\nSLO", ha="right", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(load_x - 0.02, y - 0.20, "load\n312ms", ha="right", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(0.50, 0.88, "model swap", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol1/model_serving", "vol1_model_serving_margin_001")


def model_serving_paged_attention_waste(candidate=None):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.20, 0.66))
    x, w, h = 0.25, 0.60, 0.15
    rows = [("contig", 0.55, 0.58), ("paged", 0.96, 0.28)]
    for label, used, y in rows:
        ax.text(0.07, y + h / 2, label, ha="left", va="center", color=INK, fontsize=4.9)
        rect(ax, x, y, w * used, h, MEM, ec="white", lw=0.35)
        rect(ax, x + w * used, y, w * (1 - used), h, RED, ec="white", lw=0.35)
        ax.text(x + w * used / 2, y + h / 2, f"{int(round(used * 100))}%", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(0.54, 0.88, "usable KV", ha="center", va="center", color=INK, fontsize=5.0)
    ax.text(0.91, 0.18, "waste", ha="right", va="center", color=RED, fontsize=4.8)
    write(fig, "vol1/model_serving", "vol1_model_serving_margin_002")


def model_serving_traffic_adaptive(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 80)
    window = 0.78 - 0.46 * (t ** 0.55)
    batch = 0.18 + 0.62 * (t ** 0.82)
    ax.plot(t, window, color=TIME, lw=1.45)
    ax.plot(t, batch, color=COMP, lw=1.45)
    ax.plot([1], [window[-1]], "o", color=TIME, ms=3.2)
    ax.plot([1], [batch[-1]], "o", color=COMP, ms=3.2)
    ax.text(0.20, 0.77, "window", ha="center", va="center", color=TIME, fontsize=4.9, fontweight="bold")
    ax.text(0.76, 0.76, "batch", ha="center", va="center", color=COMP, fontsize=4.9, fontweight="bold")
    ax.text(0.82, 0.13, "QPS", ha="center", va="center", color=INK, fontsize=4.9)
    write(fig, "vol1/model_serving", "vol1_model_serving_margin_003")


def data_engineering_locality_ladder(candidate=None):
    make_ladder(
        "vol1/data_engineering",
        "vol1_data_engineering_margin_003",
        [("gather 120s", 120.0), ("local 0.2s", 0.2)],
        domain="time",
        wall=False,
    )


def data_engineering_segmentation_ladder(candidate=None):
    make_ladder(
        "vol1/data_engineering",
        "vol1_data_engineering_margin_004",
        [("mask 2.1M", 2_073_600), ("boxes 40", 40)],
        domain="count",
        wall=False,
    )


def edge_intelligence_memory_share(candidate=None):
    ratio_annotation_ladder(
        "vol2/edge_intelligence",
        "vol2_edge_intelligence_margin_001",
        [("app 300 MB", 300), ("update 75 MB", 75)],
        ratio_label="25%",
        domain="memory",
    )


def edge_intelligence_adapter_storage(candidate=None):
    ratio_annotation_ladder(
        "vol2/edge_intelligence",
        "vol2_edge_intelligence_margin_002",
        [("full 40 MB", 40), ("adapter 0.2 MB", 0.2)],
        ratio_label="200x",
        domain="memory",
    )


def edge_intelligence_forgetting(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 80)
    new_task = 0.24 + 0.54 * (1 - np.exp(-2.3 * t))
    old_task = 0.78 - 0.46 * (t ** 0.8)
    ax.plot(t, new_task, color=DATA, lw=1.45)
    ax.plot(t, old_task, color=RED, lw=1.45)
    ax.plot([1], [new_task[-1]], "o", color=DATA, ms=3.2)
    ax.plot([1], [old_task[-1]], "o", color=RED, ms=3.2)
    ax.text(0.30, 0.73, "old", ha="center", va="center", color=RED, fontsize=4.9, fontweight="bold")
    ax.text(0.72, 0.71, "new", ha="center", va="center", color=DATA, fontsize=4.9, fontweight="bold")
    ax.text(0.50, 0.20, "forget", ha="center", va="center", color=RED, fontsize=4.8)
    write(fig, "vol2/edge_intelligence", "vol2_edge_intelligence_margin_003")


def edge_intelligence_federated_savings(candidate=None):
    ratio_annotation_ladder(
        "vol2/edge_intelligence",
        "vol2_edge_intelligence_margin_004",
        [("raw 200 MB", 200), ("update 2.5 MB", 2.5)],
        ratio_label="80x",
        domain="network",
    )


def fault_tolerance_checkpoint_payload(candidate=None):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.22, 0.56))
    x, y, w, h = 0.08, 0.40, 0.84, 0.18
    weights, adam = 700, 2100
    total = weights + adam
    rect(ax, x, y, w * weights / total, h, GRID, ec="white", lw=0.35)
    rect(ax, x + w * weights / total, y, w * adam / total, h, MEM, ec="white", lw=0.35)
    ax.text(x + w * weights / total / 2, y - 0.10, "weights", ha="center", va="center", color="#555555", fontsize=4.5)
    ax.text(x + w * weights / total + w * adam / total / 2, y + h / 2, "Adam 75%", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(0.52, 0.78, "checkpoint", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/fault_tolerance", "vol2_fault_tolerance_margin_002")


def fault_tolerance_replica_downtime(candidate=None):
    make_ladder(
        "vol2/fault_tolerance",
        "vol2_fault_tolerance_margin_004",
        [("1 repl 3.65d", 87.6), ("2 repl 52.6m", 0.8767), ("3 repl 31.5s", 0.00875)],
        domain="time",
        wall=False,
    )


def data_storage_prefetch_windows(candidate=None):
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.70))
    x, y, w, h = 0.10, 0.40, 0.24, 0.16
    for idx in range(3):
        rect(ax, x + idx * w, y, w - 0.012, h, GRID, ec="white", lw=0.35)
        ax.text(x + idx * w + (w - 0.012) / 2, y + h / 2, "200", ha="center", va="center", color="#555555", fontsize=4.7)
    ax.plot([x, x + 2.5 * w], [0.70, 0.70], color=RED, lw=1.35)
    ax.plot([x, x], [0.66, 0.74], color=RED, lw=0.8)
    ax.plot([x + 2.5 * w, x + 2.5 * w], [0.66, 0.74], color=RED, lw=0.8)
    ax.text(x + 1.25 * w, 0.84, "P99 500ms", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(0.80, 0.27, "Q=3", ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold")
    write(fig, "vol2/data_storage", "vol2_data_storage_margin_002")


def data_storage_egress_cost(candidate=None):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.22, 0.50))
    x, y, w, h = 0.06, 0.42, 0.88, 0.18
    storage, egress = 24, 90
    total = storage + egress
    rect(ax, x, y, w * storage / total, h, GRID, ec="white", lw=0.35)
    rect(ax, x + w * storage / total, y, w * egress / total, h, NET, ec="white", lw=0.35)
    ax.text(x + w * storage / total / 2, y + h + 0.10, "$24K", ha="center", va="center", color="#555555", fontsize=4.5, fontweight="bold")
    ax.text(x + w * storage / total / 2, y - 0.09, "store", ha="center", va="center", color="#555555", fontsize=4.2)
    ax.text(x + w * storage / total + w * egress / total / 2, y + h / 2, "$90K egress", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(0.52, 0.78, "10 epochs", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/data_storage", "vol2_data_storage_margin_003")


def data_storage_checkpoint_storm_write_time():
    from mlsysim import Models, Systems, Bparam, BYTES_FP16, BYTES_FP32, GB, THOUSAND, byte, second

    params_b = Models.Language.Llama2_70B.parameters.m_as(Bparam)
    nodes = Systems.Clusters.Training_1K.total_accelerators
    fabric_bw_gbs = Systems.Fabrics.InfiniBand_XDR.bandwidth.m_as(GB / second)
    weights_gb = params_b * BYTES_FP16.m_as(byte)
    gradients_gb = weights_gb
    optimizer_state_gb = params_b * BYTES_FP32.m_as(byte) * 3
    zero3_total_gb = weights_gb + gradients_gb + optimizer_state_gb
    zero3_write_s = zero3_total_gb / fabric_bw_gbs
    naive_write_s = weights_gb * nodes / fabric_bw_gbs
    ratio_annotation_ladder(
        "vol2/data_storage",
        "data_storage_checkpoint_storm_write_time",
        [(f"naive {naive_write_s / 60:.1f}m", naive_write_s), (f"ZeRO-3 {zero3_write_s:.1f}s", zero3_write_s)],
        ratio_label=f"{naive_write_s / zero3_write_s:.0f}x",
        domain="time",
    )


def ops_scale_sample_size_curve(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0.08, 1.0, 120)
    y = 0.16 + 0.74 * (1 / t**2 - 1) / (1 / 0.08**2 - 1)
    ax.plot(t, y, color=INK, lw=1.45)
    ax.plot(t[8], y[8], "o", color=RED, ms=3.3)
    ax.plot(t[-1], y[-1], "o", color=DATA, ms=3.3)
    ax.text(0.25, 0.80, "small\neffect", ha="center", va="center", color=RED, fontsize=4.8, fontweight="bold")
    ax.text(0.78, 0.34, "large\neffect", ha="center", va="center", color=DATA, fontsize=4.8)
    ax.text(0.55, 0.53, "n ~ 1/d^2", ha="center", va="center", color=INK, fontsize=4.8)
    write(fig, "vol2/ops_scale", "vol2_ops_scale_margin_002")


def ops_scale_false_alert_saturation(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    tests = np.linspace(1, 100, 120)
    alpha = 0.05
    p = 1 - (1 - alpha) ** tests
    y = 0.12 + 0.78 * p
    x = np.linspace(0, 1, len(tests))
    ax.plot(x, y, color=RED, lw=1.45)
    ax.axhline(0.90, color=GRID, lw=0.6)
    ax.text(0.22, 0.82, "near 1", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(0.76, 0.21, "tests", ha="center", va="center", color=INK, fontsize=4.9)
    ax.text(0.44, 0.52, "1-(1-a)^N", ha="center", va="center", color=INK, fontsize=4.7)
    write(fig, "vol2/ops_scale", "vol2_ops_scale_margin_003")


def ops_scale_detection_window(candidate=None):
    ratio_annotation_ladder(
        "vol2/ops_scale",
        "vol2_ops_scale_margin_004",
        [("late 5d", 120), ("alert 4h", 4)],
        ratio_label="30x",
        domain="time",
    )


def responsible_ai_shap_subset_explosion(candidate=None):
    ratio_annotation_ladder(
        "vol2/responsible_ai",
        "vol2_responsible_ai_margin_001",
        [("20 feat 1M", 2**20), ("3 feat 8", 2**3)],
        ratio_label="2^n",
        domain="compute",
    )


def responsible_ai_monitoring_scale():
    formula_rows(
        "vol2/responsible_ai",
        "responsible_ai_monitoring_scale",
        [
            ("metrics", "150", DATA),
            ("events", "8.64M", NET),
            ("false", "7.5", RED),
        ],
        title="monitoring/day",
    )


def responsible_ai_override_trend(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 80)
    accuracy = 0.22 + 0.55 * (1 - np.exp(-2.4 * t))
    override = 0.78 - 0.48 * (t ** 0.75)
    ax.plot(t, accuracy, color=DATA, lw=1.45)
    ax.plot(t, override, color=RED, lw=1.45)
    ax.plot([1], [accuracy[-1]], "o", color=DATA, ms=3.2)
    ax.plot([1], [override[-1]], "o", color=RED, ms=3.2)
    ax.text(0.74, 0.72, "accuracy", ha="center", va="center", color=DATA, fontsize=4.8, fontweight="bold")
    ax.text(0.30, 0.72, "overrides", ha="center", va="center", color=RED, fontsize=4.8, fontweight="bold")
    ax.text(0.55, 0.21, "vigilance falls", ha="center", va="center", color=RED, fontsize=4.6)
    write(fig, "vol2/responsible_ai", "vol2_responsible_ai_margin_002")


def responsible_ai_governance_stack(candidate=None):
    fig, ax = margin_axes("taxonomy-mini", figsize=(1.12, 1.08))
    rows = [("provenance", 0.80), ("explain", 0.60), ("appeal", 0.40), ("outcome", 0.20)]
    ax.plot([0.20, 0.20], [0.14, 0.86], color=GRID, lw=0.7)
    for idx, (label, y) in enumerate(rows):
        color = SEL if idx == 0 else "#BBBBBB"
        ax.plot(0.20, y, "o", color=color, ms=5.7)
        ax.text(0.33, y, label, ha="left", va="center", color=INK, fontsize=4.9)
    write(fig, "vol2/responsible_ai", "vol2_responsible_ai_margin_003")


def responsible_ai_fleet_risk(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    t = np.linspace(0, 1, 100)
    p = 1 - np.exp(-5.0 * t)
    y = 0.12 + 0.78 * p
    ax.plot(t, y, color=RED, lw=1.45)
    ax.axhline(0.90, color=GRID, lw=0.6)
    ax.text(0.27, 0.78, "rare", ha="center", va="center", color=INK, fontsize=4.9)
    ax.text(0.72, 0.70, "certain", ha="center", va="center", color=RED, fontsize=5.0, fontweight="bold")
    ax.text(0.58, 0.24, "fleet scale", ha="center", va="center", color=INK, fontsize=4.8)
    write(fig, "vol2/responsible_ai", "vol2_responsible_ai_margin_004")


def inference_moe_memory_ladder(candidate=None):
    make_ladder(
        "vol2/inference",
        "vol2_inference_margin_002",
        [("MoE resident 1342 GB", 1342), ("dense resident 800 GB", 800), ("MoE active 74 GB", 74)],
        domain="memory",
        wall=False,
    )


def performance_engineering_memory_energy(candidate=None):
    make_ladder(
        "vol2/performance_engineering",
        "vol2_performance_engineering_margin_001",
        [("HBM 640 pJ", 640), ("SRAM 0.5 pJ", 0.5), ("register 0.01 pJ", 0.01)],
        domain="energy",
        wall=False,
    )


def performance_engineering_batch_roofline(candidate=None):
    make_roofline_points(
        "vol2/performance_engineering",
        "vol2_performance_engineering_margin_002",
        ridge=295.0,
        points=[("B=1", 1.0, MEM), ("B=256", 256.0, COMP)],
        arrow=True,
    )


def performance_engineering_kv_precision(candidate=None):
    ratio_annotation_ladder(
        "vol2/performance_engineering",
        "vol2_performance_engineering_margin_003",
        [("FP16 80 GB", 80), ("INT4 20 GB", 20)],
        ratio_label="4x",
        domain="memory",
    )


def performance_engineering_fleet_mfu(candidate=None):
    fig, ax = margin_axes("sparkline-trend", figsize=(1.18, 0.72))
    x0, x1 = 0.24, 0.76
    ax.text(x0, 0.86, "kernel", ha="center", va="center", color="#555555", fontsize=4.9)
    ax.text(x1, 0.86, "fleet", ha="center", va="center", color="#555555", fontsize=4.9)
    ax.plot([x0, x1], [0.68, 0.34], color=RED, lw=1.6)
    ax.scatter([x0, x1], [0.68, 0.34], s=13, color=RED, zorder=3)
    ax.text(0.50, 0.42, "sync tax", ha="center", va="center", color=RED, fontsize=4.8, fontweight="bold")
    ax.text(x0, 0.20, "high", ha="center", va="center", color=INK, fontsize=4.8)
    ax.text(x1, 0.20, "lower", ha="center", va="center", color=INK, fontsize=4.8)
    write(fig, "vol2/performance_engineering", "vol2_performance_engineering_margin_004")


def security_privacy_sgx_memory(candidate=None):
    make_ladder(
        "vol2/security_privacy",
        "vol2_security_privacy_margin_001",
        [("EPC 128 MB", 128), ("ResNet-50 102 MB", 102), ("ResNet-18 12 MB", 12)],
        domain="memory",
        wall=True,
    )


def security_privacy_dp_dataset_threshold():
    fig, ax = margin_axes("scale-anchor", figsize=(1.22, 0.58))
    x0, w, y = 0.10, 0.78, 0.48
    max_samples = 100_000
    small = 5_000
    threshold = 50_000
    small_x = x0 + w * small / max_samples
    thresh_x = x0 + w * threshold / max_samples
    ax.plot([x0, x0 + w], [y, y], color=GRID, lw=0.75)
    ax.axvspan(x0, thresh_x, color=REDFILL, alpha=0.34)
    ax.axvline(thresh_x, color=RED, lw=0.7, ls="--")
    ax.plot(small_x, y, "o", color=RED, ms=3.8)
    ax.text(small_x + 0.04, y - 0.19, "5K", ha="left", va="center", color=RED, fontsize=4.9, fontweight="bold")
    ax.text(thresh_x + 0.05, y + 0.19, "50K\nthreshold", ha="left", va="center", color=RED, fontsize=4.6, fontweight="bold")
    write(fig, "vol2/security_privacy", "security_privacy_dp_dataset_threshold")


def sharing_fill():
    fig, ax = margin_axes("iron-law-bar", figsize=(1.25, 0.72))
    for y, used, label in [(0.62, 52, "shared"), (0.30, 26, "exclusive")]:
        x, w, h = 0.10, 0.72, 0.16
        rect(ax, x, y, w, h, "#DDDDDD", ec="none")
        rect(ax, x, y, w * used / 80.0, h, MEM, ec="none")
        ax.text(x + w * used / 80.0 - 0.03, y + h / 2, f"{used}G", ha="right", va="center", color="white", fontsize=5.0)
        ax.text(x + w + 0.03, y + h / 2, label, ha="left", va="center", color="#555555", fontsize=4.8)
    write(fig, "vol2/fleet_orchestration", "fleet_orchestration_sharing_fill")


def fairness_tax(chapter, name, left_label, left_pct, right_label, right_pct):
    fig, ax = margin_axes("iron-law-bar", figsize=(1.12, 0.70))
    base_y, max_h = 0.18, 0.56
    for label, val, color, x in [(left_label, left_pct, GRID, 0.30), (right_label, right_pct, COMP, 0.62)]:
        h = max_h * val / max(left_pct, right_pct, 0.01)
        rect(ax, x, base_y, 0.20, h, color, ec="none")
        ax.text(x + 0.10, base_y + h + 0.07, f"{int(round(val * 100))}%", ha="center", va="center", color=INK, fontsize=5.1)
        ax.text(x + 0.10, 0.07, label, ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, chapter, name)


def intersectional_quadrant():
    fig, ax = margin_axes("taxonomy-mini", figsize=(1.12, 0.95))
    x0, y0, c, gap = 0.30, 0.18, 0.24, 0.035
    ax.text(x0 + c / 2, 0.83, "men", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(x0 + c + gap + c / 2, 0.83, "women", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(0.13, y0 + c + gap + c / 2, "light", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(0.13, y0 + c / 2, "dark", ha="center", va="center", color="#555555", fontsize=5.0)
    cells = [(0, 1, "#EEEEEE", "99%", "#777777"), (1, 1, "#EEEEEE", "99%", "#777777"),
             (0, 0, "#EEEEEE", "99%", "#777777"), (1, 0, RED, "65%", "white")]
    for col, row, color, label, tc in cells:
        x = x0 + col * (c + gap)
        y = y0 + row * (c + gap)
        rect(ax, x, y, c, c, color, ec="white", lw=0.7)
        ax.text(x + c / 2, y + c / 2, label, ha="center", va="center", color=tc, fontsize=5.4, fontweight="bold")
    write(fig, "vol2/responsible_ai", "responsible_ai_intersectional_quadrant")


def network_fabrics_physical_reach_ladder():
    """Schematic, evenly spaced reach ladder for the Level-1 physical medium discussion.

    The numeric labels identify typical operating ranges, but the vertical spacing
    is categorical rather than proportional distance.
    """
    fig, ax = margin_axes("other-new", figsize=(1.16, 1.42))
    spine_x = 0.19
    rows = [
        ("package", "mm", 0.84),
        ("DAC", "1-3 m", 0.62),
        ("AOC", "3-30 m", 0.40),
        ("fiber", "100 m", 0.18),
    ]
    ax.annotate(
        "",
        xy=(spine_x, 0.10),
        xytext=(spine_x, 0.91),
        arrowprops=dict(arrowstyle="-|>", color=NET, lw=1.05, alpha=0.45),
    )
    for label, value, y in rows:
        ax.plot(
            spine_x,
            y,
            "o",
            ms=7.8,
            mfc="white",
            mec=NET,
            mew=1.0,
            zorder=3,
        )
        ax.text(0.34, y + 0.030, label, ha="left", va="center", color=INK, fontsize=5.5, fontweight="bold")
        ax.text(0.34, y - 0.045, value, ha="left", va="center", color="#666666", fontsize=5.1)
    ax.text(0.19, 0.965, "reach", ha="center", va="center", color=NET, fontsize=5.1, fontweight="bold")
    write(fig, "vol2/network_fabrics", "network_fabrics_physical_reach_ladder")


def benchmarking_confidence_detectability():
    """Sample-size detectability marker for the statistical confidence trap."""
    fig, ax = margin_axes("scale-anchor", figsize=(1.18, 0.72))
    x0, x1, y = 0.12, 0.88, 0.48
    test_x, need_x = 0.40, 0.70
    ax.plot([x0, x1], [y, y], color=GRID, lw=0.65)
    ax.axvspan(x0, need_x, color=REDFILL, alpha=0.40)
    ax.axvline(need_x, color=RED, lw=0.65, ls="--")
    ax.plot(test_x, y, "o", color=TIME, ms=3.6, zorder=3)
    ax.plot(need_x, y, "o", color=RED, ms=3.8, zorder=4)
    ax.text(test_x, y + 0.18, "1K", ha="center", va="center", color=TIME, fontsize=5.3, fontweight="bold")
    ax.text(test_x, y - 0.17, "noisy", ha="center", va="center", color=TIME, fontsize=4.8)
    ax.text(need_x + 0.10, y + 0.18, "~2K", ha="center", va="center", color=RED, fontsize=5.3, fontweight="bold")
    ax.text(need_x + 0.10, y - 0.17, "+/-1 pp", ha="center", va="center", color=RED, fontsize=4.8)
    ax.text(0.50, 0.88, "detect", ha="center", va="center", color=INK, fontsize=5.1, fontweight="bold")
    write(fig, "vol1/benchmarking", "benchmarking_confidence_detectability")


def collective_communication_fsdp_collective_count():
    """Replace the FSDP process loop with the symbolic count relationship.

    ``2NL`` is a formula, not a numeric value on a shared scale. Keep it out of
    a bar chart so the geometry does not imply an arbitrary ratio.
    """
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.58))
    rows = [("DP", "1", 0.62, GRID, "#555555"), ("FSDP", "2N_L", 0.30, NET, "white")]
    for label, value, y, color, tc in rows:
        ax.plot(0.20, y + 0.075, "o", color=color, ms=5.5)
        ax.text(0.32, y + 0.075, label, ha="left", va="center", color=INK, fontsize=5.0)
        rect(ax, 0.58, y, 0.30, 0.15, color, ec="white", lw=0.35)
        ax.text(0.73, y + 0.075, value, ha="center", va="center", color=tc, fontsize=5.3, fontweight="bold")
    ax.text(0.61, 0.88, "collectives/step", ha="center", va="center", color=INK, fontsize=5.0)
    write(fig, "vol2/collective_communication", "vol2_collective_communication_margin_002")


def _short_label(text: str, max_len: int = 18) -> str:
    text = str(text)
    replacements = {
        "approximately": "~",
        "about ": "~",
        "communication": "comm",
        "Communication": "Comm",
        "computation": "comp",
        "Computation": "Comp",
        "infrastructure": "infra",
        "Infrastructure": "Infra",
        "orchestration": "orch",
        "Orchestration": "Orch",
        "optimization": "opt",
        "Optimization": "Opt",
        "throughput": "tput",
        "Throughput": "Tput",
        "sensitivity": "sens",
        "Sensitivity": "Sens",
        "acceptance": "accept",
        "Acceptance": "Accept",
        "training": "train",
        "Training": "Train",
        "inference": "infer",
        "Inference": "Infer",
        "gradient": "grad",
        "Gradient": "Grad",
        "optimizer": "opt",
        "Optimizer": "Opt",
        "bandwidth": "BW",
        "Bandwidth": "BW",
        "latency": "lat",
        "Latency": "Lat",
        "memory": "mem",
        "Memory": "Mem",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    parts = text.split()
    out = ""
    for part in parts:
        trial = (out + " " + part).strip()
        if len(trial) > max_len:
            break
        out = trial
    return out or text[: max_len - 1]


# Unit -> scale toward a per-dimension base (one ladder shares a dimension, so
# relative ORDERING stays honest). Case-sensitive power (mW != MW) is handled first.
_POWER_SCALE = {"uW": 1e-3, "\u00b5W": 1e-3, "mW": 1.0, "W": 1e3, "kW": 1e6, "MW": 1e9, "GW": 1e12}
_UNIT_SCALE = {
    "kb/s": 1e-3, "mb/s": 1.0, "gb/s": 1e3, "tb/s": 1e6,                  # bandwidth (base MB/s)
    "ns": 1e-6, "us": 1e-3, "\u00b5s": 1e-3, "ms": 1.0, "s": 1e3, "sec": 1e3,  # time (base ms)
    "min": 6e4, "h": 3.6e6, "hr": 3.6e6, "hour": 3.6e6, "day": 8.64e7, "week": 6.048e8,
    "fj": 1e-3, "pj": 1.0, "nj": 1e3, "uj": 1e6, "\u00b5j": 1e6, "mj": 1e9, "j": 1e12,  # energy (base pJ)
    "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "pb": 1e15,             # capacity (base B)
}


def _parse_number(text: str, fallback: float) -> float:
    """Magnitude a label encodes, normalized by its UNIT so bar length reads honestly.
    Unit-aware (ms vs s, MB/s vs GB/s, pJ vs nJ): e.g. 'P99 2s' -> 2000 ms correctly
    outranks 'mean 50ms' -> 50. A bare k/M/B/T suffix still scales raw counts. PREFER an
    explicit SSOT value over parsing a label; this is the heuristic fallback."""
    s = str(text).replace(",", "")
    match = re.search(r"([-+]?\d*\.?\d+)\s*([A-Za-z\u00b5/]*)", s)
    if not match:
        return fallback
    value = float(match.group(1))
    if value <= 0:
        return fallback
    unit = match.group(2)
    if unit in _POWER_SCALE:                       # case-sensitive: mW vs MW
        return value * _POWER_SCALE[unit]
    u = unit.lower()
    if u in _UNIT_SCALE:
        return value * _UNIT_SCALE[u]
    if len(u) == 1:                                # bare count multiplier (175B, 2k, 3M)
        return value * {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(u, 1.0)
    return value


def _domain(text: str) -> str:
    low = text.lower()
    if any(tok in low for tok in ("gb/s", "tb/s", "bandwidth", "link", "nvlink", "pcie", "infiniband", "network", "10g")):
        return "bandwidth"
    if any(tok in low for tok in ("pj", "watt", " watts", "kw", "mw", "power", "energy", "carbon", "co2", "emission", "flop", "mac", "transistor")):
        return "energy"
    if any(tok in low for tok in ("ms", "us", "ns", "second", "minute", "hour", "day", "week", "latency", "p99", "ttft", "tpot", "freshness")):
        return "time"
    if any(tok in low for tok in ("gb", "mb", "kb", "hbm", "dram", "sram", "ram", "cache", "weights", "state", "model", "token", "epc", "storage")):
        return "memory"
    return "memory"


def _color_for_label(label: str):
    domain = _domain(label)
    if domain == "bandwidth":
        return NET
    if domain == "energy":
        return COMP
    if domain == "time":
        return TIME
    if any(tok in label.lower() for tok in ("compute", "comp", "infer", "train", "model", "algorithm", "active")):
        return COMP
    if any(tok in label.lower() for tok in ("data", "raw", "clean", "stream")):
        return DATA
    return MEM


def _load_curated_candidates():
    opportunities = yaml.safe_load(OPPORTUNITIES.read_text(encoding="utf-8"))["recommendations"]
    decisions = yaml.safe_load(DECISIONS.read_text(encoding="utf-8"))["decisions"]
    opp_by_id = {row["id"]: row for row in opportunities}
    for decision in decisions:
        if decision["decision"] not in {"must_add", "should_add", "revise_then_add"}:
            continue
        opp = opp_by_id[decision["id"]]
        yield {**opp, **decision, "opportunity": opp}


def _labels(candidate):
    labels = list(candidate.get("opportunity", {}).get("labels") or [])
    if labels:
        return labels
    purpose = candidate.get("purpose", "")
    chunks = re.split(r":|;|,| and | versus | vs\.? ", purpose)
    return [chunk.strip() for chunk in chunks if chunk.strip()][:4] or [candidate["id"]]


def _generic_ladder(candidate):
    labels = _labels(candidate)[:6]
    values = []
    for i, label in enumerate(labels):
        values.append((_short_label(label), _parse_number(label, 10 ** (len(labels) - i - 1))))
    if len(values) == 1:
        values.append(("baseline", max(values[0][1] / 10, 1)))
    domain = _domain(" ".join(labels + [candidate.get("purpose", "")]))
    make_ladder(candidate["chapter"], curated_asset_name(candidate["id"]), values, domain=domain, wall=False)


def _generic_knee(candidate):
    labels = _labels(candidate)
    text = " ".join(labels + [candidate.get("purpose", "")])
    lower = text.lower()
    if "throttle" in lower:
        word_label = "throttle"
    elif "slo" in lower or "latency" in lower:
        word_label = "SLO"
    elif "accept" in lower:
        word_label = "accept"
    elif "rho" in lower or "communication" in lower:
        word_label = "rho=1"
    elif "capacity" in lower or "memory" in lower:
        word_label = "capacity"
    elif "spars" in lower:
        word_label = "payoff"
    elif "risk" in lower or "false" in lower:
        word_label = "failure"
    elif "tile" in lower or "fringe" in lower:
        word_label = "fringe"
    else:
        word_label = "threshold"
    pct = re.search(r"(\d{1,3})\s*%", text)
    if pct:
        value = max(0.15, min(float(pct.group(1)) / 100.0, 0.9))
        make_labeled_knee(
            candidate["chapter"],
            curated_asset_name(candidate["id"]),
            knee_frac=value,
            style="dashed",
            pct_label=f"{pct.group(1)}%",
            word_label=word_label,
        )
        return
    if any(tok in lower for tok in ("safe", "danger", "throttle", "wall", "cliff", "limit")):
        make_labeled_knee(
            candidate["chapter"],
            curated_asset_name(candidate["id"]),
            knee_frac=0.70,
            style="twotone",
            word_label=word_label,
        )
    else:
        make_labeled_knee(
            candidate["chapter"],
            curated_asset_name(candidate["id"]),
            knee_frac=0.70,
            word_label=word_label,
        )


def _generic_sparkline(candidate):
    text = (candidate.get("purpose", "") + " " + " ".join(_labels(candidate))).lower()
    falling = any(tok in text for tok in ("decay", "drop", "drops", "fall", "falls", "degrad", "collapse", "lower"))
    saturating = any(tok in text for tok in ("saturat", "plateau", "diminishing"))
    positive = any(tok in text for tok in ("feedback", "throughput", "iteration", "payback", "accuracy rises", "capacity", "streaming"))
    if "rises while" in text or "paired with" in text or "gain" in text and ("decay" in text or "falls" in text):
        make_sparkline(candidate["chapter"], curated_asset_name(candidate["id"]), style="enddots", threat=True, endpoints=[(0.18, 0.82), (0.72, 0.30)])
    elif falling:
        make_sparkline(candidate["chapter"], curated_asset_name(candidate["id"]), style="enddots", threat=True, endpoints=[(0.84, 0.32), (0.28, 0.28)])
    elif saturating:
        make_sparkline(candidate["chapter"], curated_asset_name(candidate["id"]), style="inflection", threat=False, saturating=True)
    else:
        make_sparkline(candidate["chapter"], curated_asset_name(candidate["id"]), threat=not positive, steep=2.0)


def _generic_roofline(candidate):
    text = " ".join(_labels(candidate) + [candidate.get("purpose", "")]).lower()
    name = curated_asset_name(candidate["id"])
    if "ttft" in text and "tpot" in text:
        make_roofline_points(
            candidate["chapter"],
            name,
            ridge=60.0,
            points=[("TPOT", 3.0, MEM), ("TTFT", 110.0, COMP)],
            arrow=False,
        )
    elif "batch" in text and ("compute-bound" in text or "ridge" in text or "plateau" in text):
        make_roofline_points(
            candidate["chapter"],
            name,
            ridge=8.0,
            points=[("B=1", 2.0, MEM), ("B=32", 24.0, COMP)],
            arrow=True,
        )
    elif "speculative" in text or "acceptance" in text:
        make_roofline_points(
            candidate["chapter"],
            name,
            ridge=8.0,
            points=[("draft", 14.0, COMP)],
        )
    else:
        label = "decode" if "decode" in text or "tpot" in text else "work"
        dot = (
            2.0
            if any(tok in text for tok in ("batch=1", "decode", "tpot", "memory-bound", "mnist"))
            else 16.0
        )
        ridge = 80.0 if any(tok in text for tok in ("h100", "bert", "b=256")) else 60.0
        make_roofline_points(
            candidate["chapter"],
            name,
            ridge=ridge,
            points=[(label, dot, MEM if dot < ridge else COMP)],
        )


def _generic_ironbar(candidate):
    labels = _labels(candidate)[:4]
    parsed = [_parse_number(label, 0.0) for label in labels]
    if sum(parsed) <= 0:
        parsed = [1.0 for _ in labels]
    total = sum(parsed)
    segs = []
    for label, value in zip(labels, parsed):
        # keep labels whole: a 5-char cap chopped "serial"->"seri", "overlap"->"over"
        segs.append((_short_label(label, 16), value / total, _color_for_label(label)))
    dom = max(range(len(segs)), key=lambda i: segs[i][1])
    make_ironbar(candidate["chapter"], curated_asset_name(candidate["id"]), segs, dom=dom)


def _generic_dam(candidate):
    text = " ".join(_labels(candidate) + [candidate.get("purpose", "")]).lower()
    focus = "all"
    if "data" in text:
        focus = 0
    elif "algorithm" in text or "model" in text:
        focus = 1
    elif "machine" in text or "infra" in text:
        focus = 2
    vol = "vol2" if candidate["chapter"].startswith("vol2/") else "vol1"
    make_dam(candidate["chapter"], curated_asset_name(candidate["id"]), focus=focus, vol=vol)


def _generic_taxonomy(candidate):
    labels = [_short_label(label, 13) for label in _labels(candidate)[:5]]
    colors = [DATA, COMP, NET, SEL, GRID]
    list_dots(candidate["chapter"], curated_asset_name(candidate["id"]), list(zip(labels, colors)))


def _generic_blast(candidate):
    text = (candidate.get("purpose", "") + " " + candidate.get("idea", "")).lower()
    style = "tree" if any(tok in text for tok in ("cascade", "chain", "barrier", "dependency", "pipeline")) else "fan"
    make_blast(candidate["chapter"], curated_asset_name(candidate["id"]), n=5, style=style)


def _nested_ml_system(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.92))
    rect(ax, 0.08, 0.14, 0.84, 0.66, "#E8ECEF", ec=GRID, lw=0.8)
    rect(ax, 0.38, 0.39, 0.24, 0.16, COMP, ec="white", lw=0.8)
    ax.text(0.50, 0.47, "ML", ha="center", va="center", color="white", fontsize=6.0, fontweight="bold")
    ax.text(0.50, 0.72, "support 95%", ha="center", va="center", color=INK, fontsize=5.2)
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _all_to_all(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.15, 0.98))
    pts = [(0.25, 0.72), (0.75, 0.72), (0.25, 0.28), (0.75, 0.28)]
    for i, (x0, y0) in enumerate(pts):
        for j, (x1, y1) in enumerate(pts):
            if i < j:
                ax.plot([x0, x1], [y0, y1], color=NET, lw=0.7, alpha=0.55)
    for i, (x, y) in enumerate(pts, 1):
        ax.plot(x, y, "s", color=MEM, ms=10)
        ax.text(x, y, str(i), ha="center", va="center", color="white", fontsize=5.0, fontweight="bold")
    ax.text(0.50, 0.08, "all-to-all", ha="center", va="center", color=INK, fontsize=5.4)
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _pareto(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.15, 0.92))
    x = np.array([0.12, 0.33, 0.55, 0.82])
    y = np.array([0.22, 0.46, 0.65, 0.76])
    ax.plot(x, y, color=DATA, lw=1.4)
    ax.scatter(x, y, s=12, color=DATA)
    ax.scatter([0.50], [0.33], s=20, color=RED)
    ax.text(0.50, 0.22, "dominated", ha="center", va="center", color=RED, fontsize=4.7)
    ax.plot([0.10, 0.10, 0.90], [0.15, 0.82, 0.15], color=GRID, lw=0.8)
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _error_feedback(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.15, 0.95))
    nodes = [("g+e", 0.22, 0.64), ("compress", 0.70, 0.64), ("e next", 0.46, 0.25)]
    for label, x, y in nodes:
        rect(ax, x - 0.16, y - 0.08, 0.32, 0.16, "#EEEEEE", ec=GRID, lw=0.8)
        ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=4.8, fontweight="bold")
    arrows = [((0.38, 0.64), (0.54, 0.64)), ((0.70, 0.55), (0.53, 0.33)), ((0.38, 0.30), (0.22, 0.55))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=NET, lw=1.0))
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _epsilon_budget(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.52))
    x, y, w, h = 0.08, 0.42, 0.84, 0.18
    for i in range(10):
        rect(ax, x + i * w / 10, y, w / 10 - 0.004, h, RED, ec="white", lw=0.2)
    ax.text(0.08, 0.23, "10 x eps=1", ha="left", va="center", color=INK, fontsize=5.1)
    ax.text(0.92, 0.23, "eps~10", ha="right", va="center", color=RED, fontsize=5.1, fontweight="bold")
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _causal_chain(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.55))
    labels = ["arch", "INT8", "P99", "drift"]
    xs = np.linspace(0.13, 0.87, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.plot(x, 0.55, "o", color=COMP if i < 2 else RED, ms=8)
        ax.text(x, 0.30, label, ha="center", va="center", color=INK, fontsize=4.9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.06, 0.55), xytext=(x + 0.06, 0.55),
                        arrowprops=dict(arrowstyle="->", color=GRID, lw=0.8))
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _codesign(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.18, 0.62))
    rect(ax, 0.14, 0.58, 0.72, 0.12, NET, ec="white")
    rect(ax, 0.14, 0.30, 0.72, 0.12, MEM, ec="white")
    ax.text(0.50, 0.64, "comm cap", ha="center", va="center", color="white", fontsize=5.0)
    ax.text(0.50, 0.36, "storage BW", ha="center", va="center", color="white", fontsize=5.0)
    ax.text(0.50, 0.12, "matched rates", ha="center", va="center", color=DATA, fontsize=5.0, fontweight="bold")
    write(fig, candidate["chapter"], curated_asset_name(candidate["id"]))


def _other_new(candidate):
    cid = candidate["id"]
    if cid == "vol1-introduction-margin-001":
        _nested_ml_system(candidate)
    elif cid == "vol1-introduction-margin-004":
        introduction_amdahl_pipeline(candidate)
    elif cid == "vol1-ml-systems-margin-002":
        ml_systems_edge_bandwidth_ladder(candidate)
    elif cid == "vol1-ml-systems-margin-003":
        ml_systems_thermal_throttling(candidate)
    elif cid == "vol1-ml-systems-margin-004":
        ml_systems_camera_pipeline_amdahl(candidate)
    elif cid == "vol1-data-engineering-margin-003":
        data_engineering_locality_ladder(candidate)
    elif cid == "vol1-data-engineering-margin-004":
        data_engineering_segmentation_ladder(candidate)
    elif cid == "vol1-data-selection-margin-001":
        data_selection_quality_multiplier(candidate)
    elif cid == "vol1-data-selection-margin-003":
        data_selection_echo_threshold(candidate)
    elif cid == "vol1-nn-computation-margin-001":
        nn_computation_paradigm_ops_ladder(candidate)
    elif cid == "vol1-nn-computation-margin-002":
        nn_computation_training_energy_ladder(candidate)
    elif cid == "vol1-nn-computation-margin-003":
        nn_computation_activation_logic_ladder(candidate)
    elif cid == "vol1-nn-computation-margin-004":
        nn_computation_mnist_roofline(candidate)
    elif cid == "vol1-frameworks-margin-001":
        frameworks_training_memory_ladder(candidate)
    elif cid == "vol1-frameworks-margin-002":
        frameworks_stream_overlap(candidate)
    elif cid == "vol1-model-serving-margin-001":
        model_serving_model_load_slo(candidate)
    elif cid == "vol1-model-serving-margin-002":
        model_serving_paged_attention_waste(candidate)
    elif cid == "vol1-model-serving-margin-003":
        model_serving_traffic_adaptive(candidate)
    elif cid == "vol1-training-margin-001":
        training_activation_memory_ladder(candidate)
    elif cid == "vol1-training-margin-002":
        training_bandwidth_path_ladder(candidate)
    elif cid == "vol1-training-margin-003":
        training_flash_attention_tile_ladder(candidate)
    elif cid == "vol1-benchmarking-margin-001":
        benchmarking_component_speedup_bars(candidate)
    elif cid == "vol2-compute-infrastructure-margin-002":
        compute_infrastructure_rack_power_envelope(candidate)
    elif cid == "vol2-compute-infrastructure-margin-003":
        compute_infrastructure_mtbf_ladder(candidate)
    elif cid == "vol2-collective-communication-margin-002":
        collective_communication_fsdp_collective_count()
    elif cid == "vol2-collective-communication-margin-003":
        collective_communication_payload_shrink(candidate)
    elif cid == "vol2-data-storage-margin-002":
        data_storage_prefetch_windows(candidate)
    elif cid == "vol2-data-storage-margin-003":
        data_storage_egress_cost(candidate)
    elif cid == "vol2-distributed-training-margin-001":
        distributed_training_ratio_threshold(candidate)
    elif cid == "vol2-distributed-training-margin-002":
        distributed_training_barrier_block(candidate)
    elif cid == "vol2-distributed-training-margin-003":
        distributed_training_energy_tax(candidate)
    elif cid == "vol2-distributed-training-margin-004":
        distributed_training_bandwidth_gap(candidate)
    elif cid == "vol2-edge-intelligence-margin-001":
        edge_intelligence_memory_share(candidate)
    elif cid == "vol2-edge-intelligence-margin-002":
        edge_intelligence_adapter_storage(candidate)
    elif cid == "vol2-edge-intelligence-margin-003":
        edge_intelligence_forgetting(candidate)
    elif cid == "vol2-edge-intelligence-margin-004":
        edge_intelligence_federated_savings(candidate)
    elif cid == "vol2-fault-tolerance-margin-002":
        fault_tolerance_checkpoint_payload(candidate)
    elif cid == "vol2-fault-tolerance-margin-004":
        fault_tolerance_replica_downtime(candidate)
    elif cid == "vol2-fleet-orchestration-margin-001":
        fleet_orchestration_failure_rate(candidate)
    elif cid == "vol2-fleet-orchestration-margin-003":
        fleet_orchestration_capacity_lag(candidate)
    elif cid == "vol2-fleet-orchestration-margin-004":
        fleet_orchestration_scheduler_comparison(candidate)
    elif cid == "vol2-inference-margin-002":
        inference_moe_memory_ladder(candidate)
    elif cid == "vol2-ops-scale-margin-002":
        ops_scale_sample_size_curve(candidate)
    elif cid == "vol2-ops-scale-margin-003":
        ops_scale_false_alert_saturation(candidate)
    elif cid == "vol2-ops-scale-margin-004":
        ops_scale_detection_window(candidate)
    elif cid == "vol2-performance-engineering-margin-001":
        performance_engineering_memory_energy(candidate)
    elif cid == "vol2-performance-engineering-margin-002":
        performance_engineering_batch_roofline(candidate)
    elif cid == "vol2-performance-engineering-margin-003":
        performance_engineering_kv_precision(candidate)
    elif cid == "vol2-performance-engineering-margin-004":
        performance_engineering_fleet_mfu(candidate)
    elif cid == "vol2-responsible-ai-margin-001":
        responsible_ai_shap_subset_explosion(candidate)
    elif cid == "vol2-responsible-ai-margin-002":
        responsible_ai_override_trend(candidate)
    elif cid == "vol2-responsible-ai-margin-003":
        responsible_ai_governance_stack(candidate)
    elif cid == "vol2-responsible-ai-margin-004":
        responsible_ai_fleet_risk(candidate)
    elif cid == "vol2-security-privacy-margin-001":
        security_privacy_sgx_memory(candidate)
    elif cid == "vol1-nn-architectures-margin-003":
        _all_to_all(candidate)
    elif cid == "vol1-benchmarking-margin-004":
        _pareto(candidate)
    elif cid == "vol2-collective-communication-margin-004":
        _error_feedback(candidate)
    elif cid == "vol2-security-privacy-margin-003":
        _epsilon_budget(candidate)
    elif cid == "vol1-conclusion-margin-001":
        _causal_chain(candidate)
    elif cid == "vol2-sustainable-ai-margin-001":
        sustainable_ai_carbon_frontier(candidate)
    elif cid == "vol2-sustainable-ai-margin-002":
        sustainable_ai_pue_overhead(candidate)
    elif cid == "vol2-sustainable-ai-margin-004":
        sustainable_ai_radio_energy(candidate)
    elif cid == "vol2-conclusion-margin-002":
        _codesign(candidate)
    elif cid == "vol2-conclusion-margin-003":
        conclusion_gain_stack(candidate)
    else:
        _generic_taxonomy(candidate)


def generate_curated_margin_figures() -> None:
    """Generate one SVG for every curated margin opportunity.

    These are intentionally simple first-pass visuals driven by the pedagogical
    decision file. Each image can be refined later without touching the QMD
    placement because the asset name is stable. Generic figures get their
    chapter/output directory from the candidate's ``chapter`` field; the exact
    paragraph anchor is still an editorial placement decision in the QMD.
    """
    for candidate in _load_curated_candidates():
        if candidate["id"] in {
            "vol1-introduction-margin-004",
            "vol1-ml-systems-margin-002",
            "vol1-ml-systems-margin-003",
            "vol1-ml-systems-margin-004",
            "vol1-data-engineering-margin-003",
            "vol1-data-engineering-margin-004",
            "vol1-data-selection-margin-001",
            "vol1-data-selection-margin-003",
            "vol1-nn-computation-margin-001",
            "vol1-nn-computation-margin-002",
            "vol1-nn-computation-margin-003",
            "vol1-nn-computation-margin-004",
            "vol1-frameworks-margin-001",
            "vol1-frameworks-margin-002",
            "vol1-model-serving-margin-001",
            "vol1-model-serving-margin-002",
            "vol1-model-serving-margin-003",
            "vol1-training-margin-001",
            "vol1-training-margin-002",
            "vol1-training-margin-003",
            "vol1-benchmarking-margin-001",
            "vol2-compute-infrastructure-margin-002",
            "vol2-compute-infrastructure-margin-003",
            "vol2-collective-communication-margin-002",
            "vol2-collective-communication-margin-003",
            "vol2-data-storage-margin-002",
            "vol2-data-storage-margin-003",
            "vol2-distributed-training-margin-001",
            "vol2-distributed-training-margin-002",
            "vol2-distributed-training-margin-003",
            "vol2-distributed-training-margin-004",
            "vol2-edge-intelligence-margin-001",
            "vol2-edge-intelligence-margin-002",
            "vol2-edge-intelligence-margin-003",
            "vol2-edge-intelligence-margin-004",
            "vol2-fault-tolerance-margin-002",
            "vol2-fault-tolerance-margin-004",
            "vol2-fleet-orchestration-margin-001",
            "vol2-fleet-orchestration-margin-003",
            "vol2-fleet-orchestration-margin-004",
            "vol2-inference-margin-002",
            "vol2-ops-scale-margin-002",
            "vol2-ops-scale-margin-003",
            "vol2-ops-scale-margin-004",
            "vol2-performance-engineering-margin-001",
            "vol2-performance-engineering-margin-002",
            "vol2-performance-engineering-margin-003",
            "vol2-performance-engineering-margin-004",
            "vol2-responsible-ai-margin-001",
            "vol2-responsible-ai-margin-002",
            "vol2-responsible-ai-margin-003",
            "vol2-responsible-ai-margin-004",
            "vol2-security-privacy-margin-001",
            "vol2-sustainable-ai-margin-001",
            "vol2-sustainable-ai-margin-002",
            "vol2-sustainable-ai-margin-004",
            "vol2-conclusion-margin-002",
            "vol2-conclusion-margin-003",
        }:
            _other_new(candidate)
            continue
        device = candidate.get("device") or candidate.get("opportunity", {}).get("device")
        if device == "new-matplotlib":
            device = "other-new"
        if device == "hierarchy-ladder":
            _generic_ladder(candidate)
        elif device == "scale-anchor":
            _generic_knee(candidate)
        elif device == "sparkline-trend":
            _generic_sparkline(candidate)
        elif device == "thumbnail-roofline":
            _generic_roofline(candidate)
        elif device == "iron-law-bar":
            _generic_ironbar(candidate)
        elif device == "dam-locator":
            _generic_dam(candidate)
        elif device == "taxonomy-mini":
            _generic_taxonomy(candidate)
        elif device == "blast-radius":
            _generic_blast(candidate)
        else:
            _other_new(candidate)


def coordination_tax():
    fig, ax = margin_axes("iron-law-bar", figsize=(1.25, 0.34))
    x, y, w, h = 0.05, 0.38, 0.90, 0.26
    compute = 0.04
    rect(ax, x, y, w * compute, h, GRID, ec="white")
    rect(ax, x + w * compute, y, w * (1 - compute), h, NET, ec="white")
    ax.text(x + w * compute + w * (1 - compute) / 2, y + h / 2, "sync 96%", ha="center", va="center", color="white", fontsize=5.4)
    ax.text(x + w * compute + 0.01, 0.19, "compute 4%", ha="left", va="center", color="#555555", fontsize=5.0)
    write(fig, "vol2/introduction", "vol2_introduction_coordination_tax")


def kv_cache_ladder():
    make_ladder(
        "vol2/inference",
        "inference_kv_cache_ladder",
        [("KV budget 480 GB", 480.0), ("128K req 43 GB", 43.0), ("token 0.33 MB", 0.00033)],
        domain="memory",
        wall=True,
    )


def energy_per_byte():
    make_ladder(
        "vol2/sustainable_ai",
        "sustainable_ai_energy_per_byte_ladder",
        [("Net 10k", 10000), ("NVMe 1k", 1000), ("DRAM 160", 160), ("L2 5", 5), ("L1 1", 1), ("Reg 0.1", 0.1)],
        domain="energy",
        wall=False,
        style="staircase",
    )


def alpha_beta():
    fig, ax = margin_axes("iron-law-bar", figsize=(1.25, 0.64))
    for y, left_label, right_label, frac, color in [
        (0.62, "n/beta", "large\nmsg", 0.82, NET),
        (0.30, "alpha", "small\nmsg", 0.70, TIME),
    ]:
        x, w, h = 0.10, 0.76, 0.18
        rect(ax, x, y, w, h, GRID, ec="white")
        rect(ax, x, y, w * frac, h, color, ec="white")
        ax.text(x + w * frac / 2, y + h / 2, left_label, ha="center", va="center", color="white", fontsize=5.1)
        ax.text(x + w + 0.03, y + h / 2, right_label, ha="left", va="center", color="#555555", fontsize=4.6)
    write(fig, "vol2/collective_communication", "collective_communication_alpha_beta_dominance")


def generate() -> None:
    # Volume I
    make_ladder("vol1/benchmarking", "benchmarking_power_ladder", [("rack 10 kW", 10000), ("node 400 W", 400), ("edge 80 W", 80), ("RPi4 3.5 W", 3.5), ("MCU 25 mW", 0.025), ("NDP 150 uW", 0.00015)], domain="power")
    benchmarking_confidence_detectability()
    benchmarking_tail_latency_gap()
    taxonomy_quadrant("vol1/data_engineering", "data_engineering_data_gravity_entropy", selected=(0, 1), xlabel="data gravity", ylabel="info entropy", labels={(0, 1): "high\ngain"})
    make_ladder("vol1/data_engineering", "data_engineering_storage_latency_hierarchy", [("internet 100ms", 0.1), ("network 500us", 5e-4), ("SSD 100us", 1e-4), ("DRAM 100ns", 1e-7), ("L1 0.5ns", 5e-10)], domain="time", wall=True)
    data_selection_compute_data_gap()
    make_knee("vol1/data_selection", "data_selection_icr_frontier", knee_frac=0.72)
    make_ladder("vol1/frameworks", "frameworks_bandwidth_hierarchy", [("HBM 2039", 2039), ("NVLink 600", 600), ("PCIe 32", 32)], domain="bandwidth")
    make_sparkline("vol1/frameworks", "frameworks_dispatch_tax_divergence", threat=False, steep=1.9)
    make_dam("vol1/hw_acceleration", "hw_acceleration_dam_locator", focus=2, vol="vol1")
    make_ladder("vol1/hw_acceleration", "hw_acceleration_energy_ladder", [("DRAM 640 pJ", 640), ("MAC 3.7 pJ", 3.7), ("SRAM 0.5 pJ", 0.5)], domain="energy")
    make_roofline("vol1/hw_acceleration", "hw_acceleration_roofline_elbow")
    make_ladder("vol1/introduction", "introduction_energy_hierarchy", [("DRAM 160 pJ", 160), ("FP16 1.1 pJ", 1.1), ("INT8 0.2 pJ", 0.2)], domain="energy")
    make_ironbar("vol1/introduction", "introduction_iron_law_bars", [("D", 0.58, MEM), ("C", 0.20, COMP), ("L", 0.22, NET)], dom=0)
    make_ladder("vol1/ml_ops", "ml_ops_drift_threshold_knee", [("low traffic 10d", 14_400), ("1 QPS 17m", 17)], domain="time", wall=False)
    make_ladder("vol1/ml_systems", "ml_systems_deployment_span", [("Cloud 3 MW", 3_000_000), ("Edge 200 W", 200), ("Mobile 5 W", 5), ("Tiny 50 mW", 0.05)], domain="power")
    make_sparkline("vol1/ml_systems", "ml_systems_memory_wall_divergence", threat=True, steep=1.9)
    make_dam("vol1/ml_systems", "ml_systems_dam_locator", focus="all", vol="vol1")
    escalation_curve()
    make_ladder("vol1/ml_workflow", "ml_workflow_feedback_timescales", [("quarter", 90), ("month", 30), ("week", 7), ("day", 1), ("hour", 1 / 24), ("minute", 1 / 1440)], domain="time")
    make_dam("vol1/model_compression", "model_compression_dam_locator", focus=2, vol="vol1", style="boxes")
    before_after_quant()
    make_blast("vol1/model_serving", "model_serving_blast_radius", n=4)
    latency_budget()
    list_dots("vol1/nn_architectures", "nn_architectures_inductive_bias", [("CNN", INK), ("Transformer", "#888888"), ("MLP", GRID)])
    make_dam("vol1/nn_architectures", "nn_architectures_algorithm_axis", focus=1, vol="vol1", style="boxes")
    make_ladder("vol1/nn_architectures", "nn_architectures_arithmetic_intensity", [("ResNet 40", 40), ("MobileNet 21", 21), ("GPT-2 0.5", 0.5)], domain="compute", style="lollipop")
    make_knee("vol1/nn_architectures", "nn_architectures_attention_memory_wall", knee_frac=0.72)
    labeled_memory_bars("vol1/nn_architectures", "nn_architectures_capacity_wall", [("Item+User 102", 102), ("A100 80", 80), ("Item 51", 51)])
    labeled_memory_bars("vol1/nn_computation", "nn_computation_memory_explosion", [("GPT-2 6 GB", 6000), ("MNIST 438 KB", 0.438)], title="model memory")
    simple_bar("vol1/nn_computation", "nn_computation_matmul_dominance", [("MatMul", 0.92, COMP, "white"), ("", 0.08, GRID, INK)])
    make_dam("vol1/responsible_engr", "responsible_engr_dam_locator_data", focus=0, vol="vol1")
    make_blast("vol1/responsible_engr", "responsible_engr_blast_radius_sepsis", n=5)
    make_knee("vol1/responsible_engr", "responsible_engr_scale_anchor_goodhart", knee_frac=0.72)
    simple_bar("vol1/responsible_engr", "responsible_engr_tco_bar", [("train", 0.04, GRID, INK), ("inf", 0.72, COMP, "white"), ("ops", 0.24, GRID, "white")])
    make_knee("vol1/training", "training_cost_asymmetry", knee_frac=0.72)
    make_ironbar("vol1/training", "training_iron_law_bars", [("D", 0.16, MEM), ("C", 0.66, COMP), ("L", 0.18, NET)], dom=1)
    make_ironbar("vol1/training", "training_optimizer_memory", [("P", 0.25, GRID), ("G", 0.25, GRID), ("Adam", 0.50, MEM)], dom=2)
    vol1_conclusion_fleet_mtbf_ladder()

    # Volume II
    make_ironbar("vol2/collective_communication", "collective_communication_comm_dominance", [("compute", 0.30, GRID), ("comm", 0.70, NET)], dom=1, style="trio")
    alpha_beta()
    make_sparkline("vol2/collective_communication", "collective_communication_ring_tree_divergence", threat=True, steep=2.0)
    make_roofline("vol2/compute_infrastructure", "compute_infrastructure_decode_roofline", ridge=295, dot_ai=1)
    conclusion_tail_latency_fanout()
    make_dam("vol2/data_storage", "data_storage_dai_locator", focus=2, vol="vol2", style="pills")
    make_ladder("vol2/data_storage", "data_storage_checkpoint_dominance", [("Ckpts 7.56 PB", 7560), ("Data 6 TB", 6)], domain="memory")
    make_ladder("vol2/data_storage", "data_storage_bandwidth_cliff", [("HBM 3.35 TB/s", 3350), ("DRAM 200", 200), ("NVMe 7", 7)], domain="bandwidth")
    data_storage_checkpoint_storm_write_time()
    make_ladder("vol2/distributed_training", "distributed_training_memory_budget", [("Optimizer 2100 GB", 2100), ("Gradients 350", 350), ("Weights 350", 350)], domain="memory")
    distributed_training_pipeline_bubble_tax()
    distributed_training_young_daly_checkpoint_curve()
    make_ladder("vol2/edge_intelligence", "edge_intelligence_bandwidth_ladder", [("HBM3 3350", 3350), ("Mobile 100", 100)], domain="bandwidth", wall=True)
    make_ladder("vol2/edge_intelligence", "edge_intelligence_device_memory_ladder", [("Phone 8 GB", 8000), ("IoT 1 GB", 1000), ("MCU 4 MB", 4), ("SRAM 520 KB", 0.52)], domain="memory")
    make_ladder("vol2/fault_tolerance", "fault_tolerance_mtbf_ladder", [("1 GPU 50K h", 50000), ("1K 50 h", 50), ("10K 5 h", 5)], domain="time")
    make_blast("vol2/fault_tolerance", "fault_tolerance_blast", n=5)
    make_ladder("vol2/fault_tolerance", "fault_tolerance_detection_ladder", [("SDC ~2h", 7200), ("partition 180s", 180), ("GPU hang 120s", 120), ("crash 30s", 30)], domain="time")
    make_knee("vol2/fleet_orchestration", "fleet_orchestration_util_knee", knee_frac=0.70)
    fleet_orchestration_priority_inversion()
    make_ladder("vol2/fleet_orchestration", "fleet_orchestration_bw_hierarchy", [("NVLink 900 GB/s", 900), ("IB 50 GB/s", 50), ("spine 12 GB/s", 12)], domain="bandwidth")
    sharing_fill()
    make_blast("vol2/fleet_orchestration", "fleet_orchestration_preempt_cascade", n=5)
    make_ironbar("vol2/inference", "inference_serving_cost_dominance", [("CapEx", 0.15, GRID), ("OpEx", 0.85, COMP)], dom=1, style="trio")
    make_knee("vol2/inference", "inference_batching_knee", knee_frac=0.68)
    make_ladder("vol2/inference", "inference_logic_wall_ladder", [("reasoning 12.8 s", 12.8), ("fast 0.1 s", 0.1)], domain="time")
    kv_cache_ladder()
    make_roofline_points(
        "vol2/inference",
        "inference_decode_roofline",
        ridge=60,
        points=[("decode", 6.0, MEM), ("verify", 55.0, COMP)],
        arrow=True,
    )
    make_knee("vol2/introduction", "vol2_introduction_reliability_knee", knee_frac=0.70)
    make_knee("vol2/introduction", "vol2_introduction_ci_knee", knee_frac=0.70, style="dashed", pct_label="CI")
    coordination_tax()
    make_blast("vol2/network_fabrics", "network_fabrics_gpu_fanout", n=6)
    network_fabrics_physical_reach_ladder()
    ops_scale_embedding_update_blast()
    make_ironbar("vol2/ops_scale", "ops_scale_tco_dominance", [("Tr", 0.10, GRID), ("Inf", 0.50, COMP), ("Da", 0.25, GRID), ("It", 0.15, GRID)], dom=1)
    make_ironbar("vol2/performance_engineering", "performance_engineering_iron_law_bars", [("D", 0.58, MEM), ("C", 0.22, COMP), ("L", 0.20, NET)], dom=0)
    make_ladder("vol2/performance_engineering", "performance_engineering_flash_ladder", [("naive 35 GB", 35), ("Flash 537 MB", 0.537)], domain="memory")
    make_roofline("vol2/performance_engineering", "performance_engineering_specdec_roofline", ridge=60, dot_ai=20)
    fairness_tax("vol2/responsible_ai", "responsible_ai_fairness_tax", "Base", 0.85, "Parity", 0.81)
    intersectional_quadrant()
    make_ladder("vol2/responsible_ai", "responsible_ai_unlearning_cost_ladder", [("Full $4.6M", 4_600_000), ("SISA $46k", 46_000)], domain="compute")
    responsible_ai_representation_tax_ladder()
    responsible_ai_monitoring_scale()
    fairness_tax("vol2/robust_ai", "robust_ai_robustness_tax", "Std", 0.76, "Robust", 0.50)
    make_knee("vol2/robust_ai", "robust_ai_psi_drift_knee", knee_frac=0.70)
    make_dam("vol2/security_privacy", "security_privacy_dai_attack_surface", focus="all", vol="vol2")
    security_privacy_dp_dataset_threshold()
    energy_per_byte()
    make_sparkline("vol2/sustainable_ai", "sustainable_ai_inference_crossover", threat=True, steep=1.9)
    make_knee("vol2/sustainable_ai", "sustainable_ai_thermal_throttle_knee", knee_frac=0.70, style="twotone")
    generate_curated_margin_figures()


if __name__ == "__main__":
    generate()
