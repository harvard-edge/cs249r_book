#!/usr/bin/env python3
"""Generate the committed MLSysBook margin-figure SVG assets.

The output is intentionally SVG. The figures are authored at the native
margin-column scale, use the book Helvetica stack through ``mlsysim.viz``, and
reuse the canonical margin-device vocabulary documented in
``.claude/rules/margin-figures.md``.

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


def make_sparkline(chapter, name, *, threat=True, style="gap", steep=1.8, saturating=False, endpoints=None):
    fig, ax = new_fig("sparkline-trend")
    sparkline(ax, threat=threat, style=style, steep=steep, saturating=saturating, endpoints=endpoints)
    write(fig, chapter, name)


def make_roofline(chapter, name, *, ridge=60.0, dot_ai=6.0):
    fig, ax = new_fig("thumbnail-roofline")
    roofline(ax, ridge=ridge, dot_ai=dot_ai)
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
            rect(ax, x, y, c, c, RED if on else "#EEEEEE", ec="white", lw=0.8)
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
    ax.text(0.62, 0.62, "4x smaller", ha="center", va="center", color=COMP, fontsize=4.8)
    ax.plot([x0, x1], [0.30, 0.29], color=MEM, lw=1.8)
    ax.scatter([x0, x1], [0.30, 0.29], s=13, color=MEM, zorder=3)
    ax.text(0.62, 0.22, "~same", ha="center", va="center", color=MEM, fontsize=4.8)
    write(fig, "vol1/model_compression", "model_compression_int8_beforeafter")


def labeled_memory_bars(chapter, name, rows, *, title=None):
    fig, ax = margin_axes("hierarchy-ladder", figsize=(1.20, 0.74 + 0.10 * max(0, len(rows) - 2)))
    maxv = max(v for _, v in rows)
    y0 = 0.64
    step = 0.28 if len(rows) <= 2 else 0.20
    if title:
        ax.text(0.08, 0.92, title, ha="left", va="center", color=INK, fontsize=5.0)
    for idx, (label, value) in enumerate(rows):
        y = y0 - idx * step
        width = 0.12 + 0.76 * (value / maxv)
        rect(ax, 0.08, y, width, 0.15, MEM, ec="none")
        if width > 0.48:
            ax.text(0.08 + width - 0.025, y + 0.075, label, ha="right", va="center", color="white", fontsize=4.8)
        else:
            ax.text(0.08 + width + 0.03, y + 0.075, label, ha="left", va="center", color=INK, fontsize=5.0)
    write(fig, chapter, name)


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
        rect(ax, x, y, c, c, color, ec="white", lw=0.8)
        ax.text(x + c / 2, y + c / 2, label, ha="center", va="center", color=tc, fontsize=5.4, fontweight="bold")
    write(fig, "vol2/responsible_ai", "responsible_ai_intersectional_quadrant")


def precision_dotcells():
    fig, ax = margin_axes("taxonomy-mini", figsize=(1.10, 0.95))
    x0, y0, c, gap = 0.30, 0.18, 0.26, 0.035
    ax.text(x0 + c / 2, 0.08, "Ampere", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(x0 + c + gap + c / 2, 0.08, "Hopper", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(0.13, y0 + c + gap + c / 2, "FP8", ha="center", va="center", color="#555555", fontsize=5.0)
    ax.text(0.13, y0 + c / 2, "INT8", ha="center", va="center", color="#555555", fontsize=5.0)
    for col in range(2):
        for row in range(2):
            x = x0 + col * (c + gap)
            y = y0 + row * (c + gap)
            rect(ax, x, y, c, c, "none", ec=GRID, lw=1.0)
            ax.plot(x + c / 2, y + c / 2, "o", color=RED if (col, row) == (1, 1) else GRID, ms=8)
    write(fig, "vol2/compute_infrastructure", "compute_infrastructure_precision_dotcells")


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


def _parse_number(text: str, fallback: float) -> float:
    s = str(text).replace(",", "")
    match = re.search(r"([-+]?\d*\.?\d+)\s*([kKmMbBtT]?)", s)
    if not match:
        return fallback
    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000
    elif suffix == "t":
        value *= 1_000_000_000_000
    if value <= 0:
        return fallback
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
    pct = re.search(r"(\d{1,3})\s*%", text)
    if pct:
        value = max(0.15, min(float(pct.group(1)) / 100.0, 0.9))
        make_knee(candidate["chapter"], curated_asset_name(candidate["id"]), knee_frac=value, style="dashed", pct_label=f"{pct.group(1)}%")
        return
    if any(tok in text.lower() for tok in ("safe", "danger", "throttle", "wall", "cliff", "limit")):
        make_knee(candidate["chapter"], curated_asset_name(candidate["id"]), knee_frac=0.70, style="twotone")
    else:
        make_knee(candidate["chapter"], curated_asset_name(candidate["id"]), knee_frac=0.70)


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
    dot = 2.0 if any(tok in text for tok in ("batch=1", "decode", "tpot", "memory-bound", "mnist")) else 16.0
    ridge = 80.0 if any(tok in text for tok in ("h100", "bert", "batch", "b=256")) else 60.0
    make_roofline(candidate["chapter"], curated_asset_name(candidate["id"]), ridge=ridge, dot_ai=dot)


def _generic_ironbar(candidate):
    labels = _labels(candidate)[:4]
    parsed = [_parse_number(label, 0.0) for label in labels]
    if sum(parsed) <= 0:
        parsed = [1.0 for _ in labels]
    total = sum(parsed)
    segs = []
    for label, value in zip(labels, parsed):
        segs.append((_short_label(label, 5), max(value / total, 0.04), _color_for_label(label)))
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
    colors = [DATA, COMP, NET, RED, GRID]
    list_dots(candidate["chapter"], curated_asset_name(candidate["id"]), list(zip(labels, colors)))


def _generic_blast(candidate):
    text = (candidate.get("purpose", "") + " " + candidate.get("idea", "")).lower()
    style = "tree" if any(tok in text for tok in ("cascade", "chain", "barrier", "dependency", "pipeline")) else "fan"
    make_blast(candidate["chapter"], curated_asset_name(candidate["id"]), n=5, style=style)


def _nested_ml_system(candidate):
    fig, ax = margin_axes("other-new", figsize=(1.20, 0.92))
    rect(ax, 0.08, 0.14, 0.84, 0.66, "#E8ECEF", ec=GRID, lw=0.8)
    rect(ax, 0.39, 0.39, 0.22, 0.16, COMP, ec="white", lw=0.8)
    ax.text(0.50, 0.47, "ML\ncode", ha="center", va="center", color="white", fontsize=5.1, fontweight="bold")
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
        rect(ax, x + i * w / 10, y, w / 10 - 0.004, h, RED if i < 7 else "#E5E5E5", ec="white", lw=0.2)
    ax.text(0.08, 0.23, "epsilon budget", ha="left", va="center", color=INK, fontsize=5.1)
    ax.text(0.92, 0.23, "spent", ha="right", va="center", color=RED, fontsize=5.1)
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
    elif cid == "vol2-conclusion-margin-002":
        _codesign(candidate)
    else:
        _generic_taxonomy(candidate)


def generate_curated_margin_figures() -> None:
    """Generate one SVG for every curated margin opportunity.

    These are intentionally simple first-pass visuals driven by the pedagogical
    decision file. Each image can be refined later without touching the QMD
    placement because the asset name is stable.
    """
    for candidate in _load_curated_candidates():
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
    fig, ax = margin_axes("hierarchy-ladder", figsize=(1.15, 0.86))
    vals = [("HBM\n480 GB", 480.0, 0.56), ("128K req\n43 GB", 43.0, 0.32), ("token\n0.33 MB", 0.00033, 0.08)]
    lo, hi = math.log10(0.00033), math.log10(480.0)
    for label, value, y in vals:
        width = 0.10 + 0.78 * ((math.log10(value) - lo) / (hi - lo))
        width = max(0.12, min(width, 0.88))
        rect(ax, 0.08, y, width, 0.18, MEM, ec="none")
        if width > 0.42:
            ax.text(0.08 + width - 0.03, y + 0.09, label, ha="right", va="center", color="white", fontsize=5.0)
        else:
            ax.text(0.08 + width + 0.03, y + 0.09, label, ha="left", va="center", color=INK, fontsize=5.0)
    ax.plot([0.08, 0.96], [0.80, 0.80], color=RED, lw=1.0)
    write(fig, "vol2/inference", "inference_kv_cache_ladder")


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
        (0.62, "eta/B", "large\nmsg", 0.82, MEM),
        (0.30, "alpha", "small\nmsg", 0.70, NET),
    ]:
        x, w, h = 0.10, 0.76, 0.18
        rect(ax, x, y, w, h, GRID, ec="white")
        rect(ax, x, y, w * frac, h, color, ec="white")
        ax.text(x + w * frac / 2, y + h / 2, left_label, ha="center", va="center", color="white", fontsize=5.1)
        ax.text(x + w + 0.03, y + h / 2, right_label, ha="left", va="center", color="#555555", fontsize=4.6)
    write(fig, "vol2/collective_communication", "collective_communication_alpha_beta_dominance")


def generate() -> None:
    # Volume I
    make_sparkline("vol1/benchmarking", "benchmarking_mlperf_speedup_divergence", threat=False, steep=2.0)
    make_ladder("vol1/benchmarking", "benchmarking_power_ladder", [("rack 10 kW", 10000), ("node 400 W", 400), ("edge 80 W", 80), ("RPi4 3.5 W", 3.5), ("MCU 25 mW", 0.025), ("NDP 150 uW", 0.00015)], domain="power")
    taxonomy_quadrant("vol1/data_engineering", "data_engineering_data_gravity_entropy", selected=(0, 1), xlabel="data gravity", ylabel="info entropy", labels={(0, 1): "high\ngain"})
    make_ladder("vol1/data_engineering", "data_engineering_storage_latency_hierarchy", [("internet 100ms", 0.1), ("network 500us", 5e-4), ("SSD 100us", 1e-4), ("DRAM 100ns", 1e-7), ("L1 0.5ns", 5e-10)], domain="time", wall=True)
    make_sparkline("vol1/data_selection", "data_selection_scaling_saturation", style="inflection", threat=False, saturating=True)
    make_knee("vol1/data_selection", "data_selection_icr_frontier", knee_frac=0.72)
    make_ladder("vol1/frameworks", "frameworks_bandwidth_hierarchy", [("HBM 2039", 2039), ("NVLink 600", 600), ("PCIe 32", 32)], domain="bandwidth")
    make_sparkline("vol1/frameworks", "frameworks_dispatch_tax_divergence", threat=False, steep=1.9)
    make_dam("vol1/hw_acceleration", "hw_acceleration_dam_locator", focus=2, vol="vol1")
    make_ladder("vol1/hw_acceleration", "hw_acceleration_energy_ladder", [("DRAM 640 pJ", 640), ("MAC 3.7 pJ", 3.7), ("SRAM 0.5 pJ", 0.5)], domain="energy")
    make_roofline("vol1/hw_acceleration", "hw_acceleration_roofline_elbow")
    make_ladder("vol1/introduction", "introduction_energy_hierarchy", [("DRAM 160 pJ", 160), ("FP16 1.1 pJ", 1.1), ("INT8 0.2 pJ", 0.2)], domain="energy")
    make_ironbar("vol1/introduction", "introduction_iron_law_bars", [("D", 0.58, MEM), ("C", 0.20, COMP), ("L", 0.22, NET)], dom=0)
    make_knee("vol1/ml_ops", "ml_ops_drift_threshold_knee", knee_frac=0.70)
    make_ladder("vol1/ml_systems", "ml_systems_deployment_span", [("Cloud 3 MW", 3_000_000), ("Edge 200 W", 200), ("Mobile 5 W", 5), ("Tiny 50 mW", 0.05)], domain="power")
    make_sparkline("vol1/ml_systems", "ml_systems_memory_wall_divergence", threat=True, steep=1.9)
    make_dam("vol1/ml_systems", "ml_systems_dam_locator", focus="all", vol="vol1")
    escalation_curve()
    make_ladder("vol1/ml_workflow", "ml_workflow_feedback_timescales", [("quarter", 90), ("month", 30), ("week", 7), ("day", 1), ("hour", 1 / 24), ("minute", 1 / 1440)], domain="time")
    make_dam("vol1/model_compression", "model_compression_dam_locator", focus="all", vol="vol1")
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
    make_ironbar("vol1/training", "training_optimizer_memory", [("P", 0.20, GRID), ("G", 0.20, GRID), ("Opt", 0.60, MEM)], dom=2)

    # Volume II
    make_ironbar("vol2/collective_communication", "collective_communication_comm_dominance", [("compute", 0.30, GRID), ("comm", 0.70, NET)], dom=1, style="trio")
    alpha_beta()
    make_sparkline("vol2/collective_communication", "collective_communication_ring_tree_divergence", threat=True, steep=2.0)
    make_roofline("vol2/compute_infrastructure", "compute_infrastructure_decode_roofline", ridge=60, dot_ai=6)
    precision_dotcells()
    make_sparkline("vol2/conclusion", "conclusion_tail_latency_rise", threat=True, steep=1.8)
    make_dam("vol2/data_storage", "data_storage_dai_locator", focus=2, vol="vol2", style="pills")
    make_ladder("vol2/data_storage", "data_storage_checkpoint_dominance", [("Ckpts 7.56 PB", 7560), ("Data 6 TB", 6)], domain="memory")
    make_ladder("vol2/data_storage", "data_storage_bandwidth_cliff", [("HBM 3.35 TB/s", 3350), ("DRAM 200", 200), ("NVMe 7", 7)], domain="bandwidth")
    make_ladder("vol2/distributed_training", "distributed_training_memory_budget", [("Optimizer 2100 GB", 2100), ("Gradients 350", 350), ("Weights 350", 350)], domain="memory")
    make_ladder("vol2/edge_intelligence", "edge_intelligence_bandwidth_ladder", [("HBM3 3350", 3350), ("Mobile 100", 100)], domain="bandwidth", wall=True)
    make_ladder("vol2/edge_intelligence", "edge_intelligence_device_memory_ladder", [("Phone 8 GB", 8000), ("IoT 1 GB", 1000), ("MCU 4 MB", 4), ("SRAM 520 KB", 0.52)], domain="memory")
    make_ladder("vol2/fault_tolerance", "fault_tolerance_mtbf_ladder", [("1 GPU 50,000h", 50000), ("1K 50h", 50), ("10K 5h", 5)], domain="time")
    make_blast("vol2/fault_tolerance", "fault_tolerance_blast", n=5)
    make_ladder("vol2/fault_tolerance", "fault_tolerance_detection_ladder", [("SDC ~2h", 7200), ("partition 180s", 180), ("GPU hang 120s", 120), ("crash 30s", 30)], domain="time")
    make_knee("vol2/fleet_orchestration", "fleet_orchestration_util_knee", knee_frac=0.70)
    make_blast("vol2/fleet_orchestration", "fleet_orchestration_dependency_cascade", n=6, style="tree")
    make_ladder("vol2/fleet_orchestration", "fleet_orchestration_bw_hierarchy", [("NVLink 900 GB/s", 900), ("IB 50 GB/s", 50), ("spine 12 GB/s", 12)], domain="bandwidth")
    sharing_fill()
    make_blast("vol2/fleet_orchestration", "fleet_orchestration_preempt_cascade", n=5)
    make_ironbar("vol2/inference", "inference_serving_cost_dominance", [("CapEx", 0.15, GRID), ("OpEx", 0.85, COMP)], dom=1, style="trio")
    make_knee("vol2/inference", "inference_batching_knee", knee_frac=0.68)
    make_ladder("vol2/inference", "inference_logic_wall_ladder", [("reasoning 12.8 s", 12.8), ("fast 0.1 s", 0.1)], domain="time")
    kv_cache_ladder()
    make_roofline("vol2/inference", "inference_decode_roofline", ridge=60, dot_ai=6)
    make_knee("vol2/introduction", "vol2_introduction_reliability_knee", knee_frac=0.70)
    make_knee("vol2/introduction", "vol2_introduction_ci_knee", knee_frac=0.70, style="dashed", pct_label="CI")
    coordination_tax()
    make_blast("vol2/network_fabrics", "network_fabrics_gpu_fanout", n=6)
    make_blast("vol2/ops_scale", "ops_scale_cross_model_blast", n=5)
    make_ironbar("vol2/ops_scale", "ops_scale_tco_dominance", [("Tr", 0.10, GRID), ("Inf", 0.50, COMP), ("Da", 0.25, GRID), ("It", 0.15, GRID)], dom=1)
    make_ironbar("vol2/performance_engineering", "performance_engineering_iron_law_bars", [("D", 0.58, MEM), ("C", 0.22, COMP), ("L", 0.20, NET)], dom=0)
    make_ladder("vol2/performance_engineering", "performance_engineering_flash_ladder", [("naive 35 GB", 35), ("Flash 537 MB", 0.537)], domain="memory")
    make_roofline("vol2/performance_engineering", "performance_engineering_specdec_roofline", ridge=60, dot_ai=20)
    fairness_tax("vol2/responsible_ai", "responsible_ai_fairness_tax", "Base", 0.85, "Parity", 0.81)
    intersectional_quadrant()
    make_ladder("vol2/responsible_ai", "responsible_ai_unlearning_cost_ladder", [("Full $4.6M", 4_600_000), ("SISA $46k", 46_000)], domain="compute")
    fairness_tax("vol2/robust_ai", "robust_ai_robustness_tax", "Std", 0.76, "Robust", 0.50)
    make_knee("vol2/robust_ai", "robust_ai_psi_drift_knee", knee_frac=0.70)
    make_dam("vol2/security_privacy", "security_privacy_dai_attack_surface", focus="all", vol="vol2")
    energy_per_byte()
    make_sparkline("vol2/sustainable_ai", "sustainable_ai_inference_crossover", threat=True, steep=1.9)
    make_knee("vol2/sustainable_ai", "sustainable_ai_thermal_throttle_knee", knee_frac=0.70, style="twotone")
    generate_curated_margin_figures()


if __name__ == "__main__":
    generate()
