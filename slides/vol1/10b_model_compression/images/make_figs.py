#!/usr/bin/env python3
"""Figures for Lecture 10b (Model Compression, Part 2: How Few Bits?).

Every value is sourced. "Chapter" means Model Compression in Volume I.
Outside papers were read on 2026-10-17.

  - Krishnamoorthi 2018 (arXiv 1806.08342). Table 3, post-training INT8
    weights and activations: MobileNetV2 1.0 224 asymmetric per-layer 0.001,
    symmetric per-channel 0.698, float 0.719; ResNet-v1-50 0.75, 0.751, 0.752;
    MobileNet-v1 symmetric per-channel 0.591, asymmetric per-channel 0.703,
    float 0.709. Table 4 (QAT, symmetric per-channel): MobileNetV2 0.711,
    ResNet-v1-50 0.75. Table 5 (4-bit weights): ResNet-v1-50 per-layer PTQ
    0.002, per-channel PTQ 0.54, QAT 0.732; MobileNetV2 0.001, 0.001, 0.62.
    Table 7 (Pixel 2): MobileNetV2 105 ms float CPU, 63 ms fixed point CPU,
    15.5 ms fixed point on the Hexagon DSP (NN-API).
  - Nagel et al. 2019 (arXiv 1906.04721), Figure 2: per-output-channel weight
    ranges of the first depthwise-separable layer of MobileNetV2, 32 channels,
    "strong differences between channel weight ranges". The redraw below is a
    schematic of that finding; no channel value is read from the plot. The
    paper's own example (section 3.1): one channel in [-128, 128], another in
    (-0.5, 0.5); at 8 bits the second rounds entirely to 0.
  - Dettmers et al. 2022 (arXiv 2208.07339), Table 1: C4 perplexity of fairseq
    dense transformers, 125M to 13B. 32-bit float 25.65 15.91 14.43 13.30
    12.45; INT8 absmax 87.76 16.55 15.11 14.59 19.08; absmax LLM.int8() 25.83
    15.93 14.44 13.24 12.45.
  - Frantar et al. 2023, GPTQ (arXiv 2210.17323). Table 5, OPT-175B WikiText2:
    FP16 8.34, RTN 4-bit 10.54, GPTQ 4-bit 8.37, RTN 3-bit 7.3e3, GPTQ 3-bit
    8.68. Table 6: per-token latency, FP16 on five A100s 230 ms, 3-bit on one
    A100 71 ms (3.24x).
  - Lin et al. 2024, AWQ (arXiv 2306.00978), Table 1: OPT-6.7B INT3 group 128,
    WikiText perplexity: FP16 10.86, RTN 23.54, 1 percent kept in FP16 chosen
    by activation 11.39, by weight 22.37.
  - Micikevicius et al. 2022 (arXiv 2209.05433), Table 1: E4M3 max normal 448,
    min subnormal 2^-9; E5M2 max 57,344. The E4M3 value grid below is derived
    from that definition (one NaN pattern, 127 non-negative finite values).
  - Chapter: FP16 max 65,504 (Sec. 10.3.1); storage 2x / 4x / 8x (table 7);
    Horowitz energy (table 8); Napkin Math 10.1 and 10.4; listing 2 weights;
    blur waterfall (fig-compression-waterfall, illustrative); GPT-2 XL
    streaming times (table 1).
  - Every value now comes from MLSysIM through calcs.py (imported; the
    printed strings through deckvals.val and num), so figures, slides, and
    the chapter share one computation. The citations above record where the
    MLSysIM entries come from.
  - Published figures shown as images (not generated here): Wu et al. 2020
    Fig. 2, Nagel et al. 2019 Fig. 2, Dettmers et al. 2022 Fig. 3(a); see
    SOURCES.yml.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402  every value below comes from calcs.py (MLSysIM)
from deckvals import val, num  # noqa: E402
from mlsysim.core.units import FLOAT_FORMAT_BITS, FP8_FORMAT_BITS, FP32_BITS, INT4_BITS, INT8_BITS  # noqa: E402
from mlsysim.physics import calc_dequantize, calc_quantize  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Helpers (copied from Lecture 10's generator so this folder stands alone)
# ---------------------------------------------------------------------------
def bars(path, rows, shown, log=False, xlim=None, xlabel=None, size=sp.SIZE,
         vlines=(), xticks=None, note=None, label_size=13, boxed=False,
         text_size=12, axis_size=None):
    """axis_size: tick, axis-label, and reference-line label size; raise it
    with label_size and text_size when the plot is drawn for a column slot."""
    """Horizontal bars with a show mask; hidden rows keep their slot."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        vis = shown[i]
        ax.barh(y, r["value"], color=C[r["color"]], height=0.62,
                alpha=1.0 if vis else 0.0)
        ax.text(-0.01, y, r["label"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=label_size, color=C["text"],
                fontweight="bold", alpha=1.0 if vis else 0.0)
        tx = r["value"] * 1.25 if log else r["value"] + (xlim[1] if xlim else 1) * 0.012
        bb = dict(facecolor="white", edgecolor="none", pad=1.0,
                  alpha=1.0 if vis else 0.0) if boxed else None
        ax.text(r.get("tx", tx), y, r["text"], va="center", fontsize=text_size, bbox=bb,
                zorder=4, color=C[r.get("tcolor", r["color"])],
                fontweight="bold", alpha=1.0 if vis else 0.0)
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls="--", zorder=1)
        ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=axis_size or 10,
                fontweight="bold", va="bottom", ha="left")
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axis_size)
    if axis_size:
        ax.tick_params(labelsize=axis_size)
    if note:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9,
                 color=C["neutral"])
    return fig, ax


def bars_save(path, *a, **k):
    fig, _ = bars(path, *a, **k)
    sp.save(fig, path)


def svg(path, w, h, body):
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        f'<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')


def text(x, y, s, size=22, color=C["text"], weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n')


def box(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def vbars(ax, groups, series, shown, width=0.2, fmt="{:.1f}", ymax=85,
          inside=None):
    """Grouped vertical bars. groups: names; series: list of (name, color,
    values per group). shown[s][g] masks a bar (hidden bars keep their slot).
    """
    ns = len(series)
    for si, (sname, ck, vals) in enumerate(series):
        for gi, v in enumerate(vals):
            x = gi + (si - (ns - 1) / 2) * width
            vis = shown[si][gi]
            if v is None:
                continue
            ax.bar(x, v, width=width * 0.92, color=C[ck], alpha=1.0 if vis else 0.0)
            ax.text(x, v + ymax * 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=C[ck],
                    alpha=1.0 if vis else 0.0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=12, fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Hook: one INT8 recipe, two networks (Krishnamoorthi 2018, Table 3)
# ---------------------------------------------------------------------------
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    groups = ["ResNet-50", "MobileNetV2"]
    flt = [num("r50_float"), num("mbv2_float")]
    i8 = [num("r50_tensor"), num("mbv2_tensor")]
    w = 0.36
    for gi in range(2):
        show_f = gi == 0 or k >= 2
        show_q = gi == 0 or k >= 3
        a = 1.0 if show_f else 0.0
        ax.bar(gi - w / 2, flt[gi], width=w * 0.92, color=C["neutral"], alpha=a)
        ax.text(gi - w / 2, flt[gi] + 1.5, f"{flt[gi]:.1f}", ha="center",
                fontsize=12, fontweight="bold", color=C["neutral"], alpha=a)
        aq = 1.0 if show_q else 0.0
        ax.bar(gi + w / 2, max(i8[gi], 0.6), width=w * 0.92, color=C["compute"], alpha=aq)
        ax.text(gi + w / 2, i8[gi] + 1.5, f"{i8[gi]:.1f}", ha="center", fontsize=12,
                fontweight="bold", color=C["danger"] if i8[gi] < 1 else C["compute"],
                alpha=aq)
        if gi == 1 and k == 2:
            ax.add_patch(plt.Rectangle((gi + w / 2 - w * 0.46, 0), w * 0.92, 40,
                                       fill=False, ls="--", lw=1.5, ec=C["compute"]))
            ax.text(gi + w / 2, 18, "?", ha="center", va="center", fontsize=26,
                    fontweight="bold", color=C["compute"])
    ax.text(0.02, 0.98, "float", transform=ax.transAxes, color=C["neutral"],
            fontsize=12, fontweight="bold", va="top")
    ax.text(0.22, 0.98, "INT8, one scale per tensor", transform=ax.transAxes,
            color=C["compute"], fontsize=12, fontweight="bold", va="top")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"hook_step{k}.svg")

# ---------------------------------------------------------------------------
# 2. Bit fields to scale: what each format spends its bits on (3 steps)
# ---------------------------------------------------------------------------
def smaller(e, f):
    return f"{FP32_BITS // (1 + e + f)}× smaller"


F = {**FLOAT_FORMAT_BITS, **FP8_FORMAT_BITS}
formats = [
    # name, sign, exponent, mantissa, integer bits, note, step
    ("FP32", 1, *F["fp32"], 0, "baseline", 1),
    ("BF16", 1, *F["bf16"], 0, f"FP32's range, {smaller(*F['bf16'])}", 1),
    ("FP16", 1, *F["fp16"], 0, f"max {val('fp16_max')}, {smaller(*F['fp16'])}", 1),
    ("FP8 E4M3", 1, *F["e4m3"], 0, f"max {val('e4m3_max')}, {smaller(*F['e4m3'])}", 2),
    ("FP8 E5M2", 1, *F["e5m2"], 0, f"max {val('e5m2_max')}, {smaller(*F['e5m2'])}", 2),
    ("INT8", 0, 0, 0, INT8_BITS, f"{val('int8_levels')} levels, {val('int8_smaller')}× smaller", 3),
    ("INT4", 0, 0, 0, INT4_BITS, f"{val('int4_levels')} levels, {val('int4_smaller')}× smaller", 3),
]
BW = 15  # px per bit
X0 = 150
for k in (1, 2, 3):
    body = ""
    # legend
    lx = X0
    for name, fill, stroke in (("sign", C["neutral_l"], C["neutral"]),
                               ("exponent: range", C["memory_l"], C["memory"]),
                               ("mantissa: precision", C["data_l"], C["data"]),
                               ("integer levels", C["compute_l"], C["compute"])):
        body += box(lx, 8, 22, 22, fill, stroke, 2, 2)
        body += text(lx + 30, 26, name, 20, C["text"])
        lx += 46 + len(name) * 10.5
    for i, (name, s, e, m, b, note, step) in enumerate(formats):
        y = 38 + i * 44
        op = 1.0 if step <= k else 0.0
        if step == 1 and k > 1:
            op = 0.55  # recall rows recede once new formats arrive
        body += f'<g opacity="{op}">\n'
        body += text(X0 - 12, y + 27, name, 22, C["text"], "bold", "end")
        x = X0
        for nbits, fill, stroke in ((s, C["neutral_l"], C["neutral"]),
                                    (e, C["memory_l"], C["memory"]),
                                    (m, C["data_l"], C["data"]),
                                    (b, C["compute_l"], C["compute"])):
            if nbits:
                body += box(x, y + 6, nbits * BW, 30, fill, stroke, 2, 2)
                if nbits * BW >= 30:
                    body += text(x + nbits * BW / 2, y + 28, str(nbits), 20, stroke,
                                 "bold", "middle")
                x += nbits * BW
        body += text(X0 + 32 * BW + 18, y + 28, note, 21, C["text"])
        body += "</g>\n"
    body += text(X0 + 32 * BW + 18, 30, "", 10)
    svg(out / f"formats_step{k}.svg", 900, 352, body)

# ---------------------------------------------------------------------------
# 3. Float grid against integer grid on one log axis (derived)
# ---------------------------------------------------------------------------
pos = np.array(calcs.E4M3_POS)
s_int = calcs.INT8_448.scale
int_levels = s_int * np.arange(1, calcs.int8_pos + 1)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.vlines(pos, 1.0, 1.6, color=C["memory"], lw=1.1)
    ax.text(0.0013, 1.95, f"FP8 E4M3: {val('e4m3_pos')} positive values, {val('e4m3_below1')} of them below 1",
            fontsize=12, fontweight="bold", color=C["memory"])
    a = 1.0 if k == 2 else 0.0
    ax.vlines(int_levels, 0.0, 0.6, color=C["compute"], lw=1.1, alpha=a)
    ax.axvspan(0.001, s_int / 2, ymin=0.233, ymax=0.433, color=C["danger_l"], alpha=a)
    ax.text(0.0013, -0.5, f"INT8 scaled to {val('e4m3_max')}: one level every {val('int8_step448')};"
            f" all below {val('int8_zero_below')} round to 0", fontsize=12,
            fontweight="bold", color=C["compute"], alpha=a)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 1000)
    ax.set_ylim(-0.7, 2.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Magnitude of a stored value (log scale)")
    sp.save(fig, out / f"grids_step{k}.svg")

# ---------------------------------------------------------------------------
# 4. Energy per operation (copied from Lecture 10; Horowitz 2014 via table 8)
# ---------------------------------------------------------------------------
E = calcs.ENERGY
ops = [(n, calcs.m(f32, "pJ/flop"), calcs.m(i8, "pJ/flop"), f"{val(k)}×")
       for n, f32, i8, k in (("Add", E.wide_add, E.narrow_add, "e_add_ratio"),
                             ("Multiply", E.wide_mul, E.narrow_mul, "e_mul_ratio"),
                             ("Multiply-accumulate", E.wide_mac, E.narrow_mac, "e_mac_ratio"))]
DRAM_32 = num("e_dram")
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ylabels = []
    for i, (name, f32, i8, ratio) in enumerate(ops):
        y = 3 - i
        ax.barh(y + 0.19, f32, height=0.34, color=C["compute"])
        ax.barh(y - 0.19, i8, height=0.34, color=C["neutral"])
        ax.text(f32 * 1.2, y + 0.19, f"FP32 {f32:.2f} pJ", va="center", fontsize=11,
                color=C["compute"], fontweight="bold")
        ax.text(i8 * 1.2, y - 0.19, f"INT8 {i8:.2f} pJ: {ratio} less", va="center",
                fontsize=11, color=C["neutral"], fontweight="bold")
        ylabels.append((y, name))
    vis = 1.0 if k == 2 else 0.0
    ax.barh(-0.05, DRAM_32, height=0.42, color=C["memory"], alpha=vis)
    ax.text(DRAM_32 * 1.2, -0.05, f"{val('e_dram')} pJ", va="center", fontsize=11, color=C["memory"],
            fontweight="bold", alpha=vis)
    ylabels.append((-0.05, "32-bit DRAM read"))
    for y, name in ylabels:
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"],
                alpha=vis if name.startswith("32") else 1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 20000)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Energy (picojoules, log scale; 45 nm)")
    sp.save(fig, out / f"energy_step{k}.svg")

# ---------------------------------------------------------------------------
# 5. The integer grid laid over a real range: a 3-bit toy (derived)
#    [alpha, beta] = [-1, 2.5], b = 3: s = 3.5 / 7 = 0.5, z = round(1/0.5) = 2
# ---------------------------------------------------------------------------
TOY = calcs.TOY
ALPHA, BETA = TOY.lo, TOY.hi
S_TOY, Z_TOY = TOY.scale, TOY.zero_point
levels = S_TOY * (np.arange(TOY.qmin, TOY.qmax + 1) - Z_TOY)
samples = TOY.values[:-1]  # the last value is the slide's question


def quant_toy(x):
    q = calc_quantize(x, S_TOY, Z_TOY, TOY.qmin, TOY.qmax)
    return q, calc_dequantize(q, S_TOY, Z_TOY)


for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.plot([-1.75, 3.25], [1, 1], color=C["text"], lw=1.5)
    ax.axvspan(ALPHA, BETA, ymin=0.52, ymax=0.70, color=C["memory_l"])
    ax.text(ALPHA, 1.58, f"α = {val('toy_alpha')}".replace("-", "−"), ha="center", fontsize=13, fontweight="bold",
            color=C["memory"])
    ax.text(BETA, 1.58, f"β = {val('toy_beta')}", ha="center", fontsize=13, fontweight="bold",
            color=C["memory"])
    ax.text(-1.9, 1, "real", ha="right", va="center", fontsize=13, fontweight="bold")
    for x in samples:
        inside = ALPHA <= x <= BETA
        ax.plot([x], [1], "o", ms=9, color=C["data"] if inside else C["danger"], zorder=5)
        ax.text(x, 1.18, f"{x:g}", ha="center", fontsize=12, fontweight="bold",
                color=C["data"] if inside else C["danger"])
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot([-1.75, 3.25], [0, 0], color=C["neutral"], lw=1.5, alpha=a2)
    ax.text(-1.9, 0, "3-bit\ncode", ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["neutral"], alpha=a2)
    for q, lv in enumerate(levels):
        hl = q == Z_TOY
        ax.plot([lv, lv], [-0.12, 0.12], color=C["compute"], lw=3 if hl else 2, alpha=a2)
        ax.plot([lv, lv], [0.12, 0.95], color=C["compute"], lw=0.8, ls=":", alpha=a2)
        ax.text(lv, -0.42, str(q), ha="center", fontsize=13, fontweight="bold",
                color=C["danger"] if hl else C["compute"], alpha=a2)
    ax.annotate("", (levels[5], -0.66), (levels[4], -0.66),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.3, alpha=a2))
    ax.text(levels[4] + S_TOY / 2, -1.02, f"s = {val('toy_s')}", ha="center", fontsize=12,
            fontweight="bold", alpha=a2)
    ax.text(levels[Z_TOY], -1.02, f"z = {val('toy_z')}", ha="center", fontsize=12,
            fontweight="bold", color=C["danger"], alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    for x in samples:
        q, xr = quant_toy(x)
        inside = ALPHA <= x <= BETA
        ck = C["data"] if inside else C["danger"]
        ax.annotate("", (xr, 0.14), (x, 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=ck, lw=1.6, alpha=a3))
    ax.text(3.25, 0.5, "outside:\nclamped", ha="left", va="center", fontsize=11,
            fontweight="bold", color=C["danger"], alpha=a3)
    ax.set_xlim(-2.6, 3.9)
    ax.set_ylim(-1.2, 1.85)
    ax.axis("off")
    sp.save(fig, out / f"affgrid_step{k}.svg")

# ---------------------------------------------------------------------------
# 6. Napkin Math 10.1 on a number line (copied from Lecture 10)
# ---------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 1.7))
    ax.plot([-1.25, 3.25], [1, 1], color=C["text"], lw=1.5)
    ax.plot([-1.25, 3.25], [0, 0], color=C["neutral"], lw=1.5)
    ax.text(-1.4, 1, "real", ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(-1.4, 0, "UINT8", ha="right", va="center", fontsize=15, fontweight="bold",
            color=C["neutral"])
    for real, q in zip(calcs.NAPKIN.values, calcs.NAPKIN.codes):
        xq = calcs.NAPKIN.lo + q * calcs.NAPKIN.scale
        ax.plot([real], [1], "o", ms=9, color=C["memory"])
        ax.text(real, 1.25, f"{real:g}", ha="center", fontsize=15, fontweight="bold",
                color=C["memory"])
        a = 1.0 if k == 2 else 0.0
        ax.annotate("", (xq, 0.06), (real, 0.92), alpha=a,
                    arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=1.6, alpha=a))
        ax.plot([xq], [0], "s", ms=8, color=C["compute"], alpha=a)
        ax.text(xq, -0.5, f"{q}", ha="center", fontsize=15, fontweight="bold",
                color=C["compute"], alpha=a)
    ax.set_xlim(-2.2, 3.4)
    ax.set_ylim(-0.6, 1.5)
    ax.axis("off")
    sp.save(fig, out / f"numline_step{k}.svg")

# ---------------------------------------------------------------------------
# 7. Rounding error against the real value, same 3-bit toy (derived)
# ---------------------------------------------------------------------------
xs = np.linspace(-1.75, 3.25, 4001)
err = np.array([quant_toy(x)[1] - x for x in xs])
inside = (xs >= ALPHA) & (xs <= BETA)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhspan(-S_TOY / 2, S_TOY / 2, color=C["data_l"], zorder=0)
    ax.plot(xs[inside], err[inside], color=C["data"], lw=2)
    ax.text(1.0, 0.33, f"inside: at most s/2 = {val('toy_half')}", ha="center", fontsize=12,
            fontweight="bold", color=C["data"])
    a = 1.0 if k == 2 else 0.0
    ax.plot(xs[xs < ALPHA], err[xs < ALPHA], color=C["danger"], lw=2.4, alpha=a)
    ax.plot(xs[xs > BETA], err[xs > BETA], color=C["danger"], lw=2.4, alpha=a)
    ax.annotate("clipped: grows", (2.95, -0.47), (0.55, -0.66), fontsize=12,
                fontweight="bold", color=C["danger"], alpha=a, va="center",
                arrowprops=dict(arrowstyle="-|>", color=C["danger"], lw=1.4, alpha=a))
    for x in (ALPHA, BETA):
        ax.axvline(x, color=C["memory"], lw=1.2, ls="--")
    ax.text(ALPHA, 0.72, "α", ha="center", fontsize=13, fontweight="bold", color=C["memory"])
    ax.text(BETA, 0.72, "β", ha="center", fontsize=13, fontweight="bold", color=C["memory"])
    ax.axhline(0, color=C["neutral"], lw=0.8)
    ax.set_xlim(-1.75, 3.25)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel(f"Real value x ({val('toy_bits')}-bit grid, s = {val('toy_s')})")
    ax.set_ylabel("Error after round trip")
    sp.save(fig, out / f"rounderr_step{k}.svg")

# ---------------------------------------------------------------------------
# 8. Symmetric against asymmetric, MobileNet-v1 per-channel PTQ (Table 3)
# ---------------------------------------------------------------------------
zp_rows = [
    dict(label="Symmetric weights (z = 0)", value=num("mbv1_sym"), color="danger", text=val("mbv1_sym")),
    dict(label="Asymmetric weights (z ≠ 0)", value=num("mbv1_asym"), color="data", text=val("mbv1_asym")),
    dict(label="Float", value=num("mbv1_float"), color="neutral", text=val("mbv1_float")),
]
bars_save(out / "zeropoint.svg", zp_rows, [1, 1, 1], xlim=(0, 82), size=(6.6, 1.25),
          label_size=15, text_size=15, axis_size=14,
          xlabel="MobileNet-v1 top-1 (%), INT8 post-training, per-channel weights")

# ---------------------------------------------------------------------------
# 8b. Symmetric and asymmetric mapping at slide size (redraw after the
#     chapter's figure 22). Symmetric INT8: [-1, 1] onto -127..127, z = 0.
#     Asymmetric INT8: [-0.5, 1.5] onto -128..127; s = 2 / 255, real zero at
#     code -128 + round(0.5 / s) = -128 + 64 = -64 (derived).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(5.9, 1.75))
for ax, (lo, hi, qlo, qhi, zq, title, ck) in zip(axes, (
        (calcs.SYM_UNIT.lo, calcs.SYM_UNIT.hi, calcs.SYM_UNIT.qmin, calcs.SYM_UNIT.qmax,
         calcs.SYM_UNIT.zero_point, f"Symmetric: z = {calcs.SYM_UNIT.zero_point}", "memory"),
        (calcs.ASYM.lo, calcs.ASYM.hi, calcs.ASYM.qmin, calcs.ASYM.qmax, calcs.ASYM.zero_point,
         f"Asymmetric: z = {val('zp_asym_z')}".replace("-", "−"), "data"))):
    # real line on top (y = 1), integer line below (y = 0), drawn on 0..1
    ax.plot([0, 1], [1, 1], color=C["text"], lw=1.5)
    ax.plot([0, 1], [0, 0], color=C["neutral"], lw=1.5)
    xr0 = (0 - lo) / (hi - lo)
    xq0 = (zq - qlo) / (qhi - qlo)
    for xr, xq, lab, qlab, hl in ((0, 0, f"α = {lo:g}", f"{qlo}", False),
                                  (xr0, xq0, "0", f"{zq}", True),
                                  (1, 1, f"β = {hi:g}", f"{qhi}", False)):
        col = C["danger"] if hl else C[ck]
        ax.annotate("", (xq, 0.06), (xr, 0.94),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.plot([xr], [1], "o", ms=7, color=col)
        ax.plot([xq], [0], "s", ms=6, color=col)
        ax.text(xr, 1.18, lab.replace("-", "−"), ha="center", fontsize=12,
                fontweight="bold", color=col)
        ax.text(xq, -0.42, qlab.replace("-", "−"), ha="center", fontsize=12,
                fontweight="bold", color=col)
    ax.text(0.5, 1.62, title, ha="center", fontsize=13, fontweight="bold", color=C[ck])
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.55, 1.85)
    ax.axis("off")
axes[0].text(-0.16, 1, "real", ha="right", va="center", fontsize=12, fontweight="bold")
axes[0].text(-0.16, 0, "INT8", ha="right", va="center", fontsize=12, fontweight="bold",
             color=C["neutral"])
fig.tight_layout(w_pad=2.0)
sp.save(fig, out / "calranges.svg")

# ---------------------------------------------------------------------------
# 10b. Activation histogram at slide size, redrawn from the chapter's figure
#      21 (illustrative bars, after Wu et al. 2020; bins of 0.1 read from the
#      book's SVG source). Step 2 marks the three calibration methods; the
#      percentile and entropy positions are illustrative (the chapter assigns
#      no percentile rank to its 1.1 boundary).
# ---------------------------------------------------------------------------
hist_px = [238, 228, 218, 200, 175, 145, 115, 90, 70, 52, 40, 30, 23, 18, 14,
           11, 9, 7, 6, 4, 3]
hist = np.array(hist_px) / 240
for k in (1,):  # the measured calibration ranges are Wu et al.'s own figure (next frame)
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    for i, h in enumerate(hist):
        tail = 0.1 * i >= num("clip_cut") - 1e-9
        ax.bar(0.1 * i + 0.05, h, width=0.092,
               color=C["danger_l"] if tail else C["memory_l"],
               edgecolor=C["danger"] if tail else C["memory"], lw=0.8)
    if k == 1:
        cut, full = num("clip_cut"), num("clip_full")
        ax.axvline(cut, color=C["data"], lw=1.8, ls="--")
        ax.text(cut + 0.03, 0.9, f"clip at {val('clip_cut')}?", fontsize=12, fontweight="bold", color=C["data"])
        ax.axvline(full, color=C["neutral"], lw=1.8, ls="--")
        ax.text(full - 0.03, 0.9, f"cover all: {val('clip_full')}", fontsize=12, fontweight="bold",
                color=C["neutral"], ha="right")
        ax.annotate("rare tail", (1.65, 0.07), (1.75, 0.45), fontsize=12,
                    fontweight="bold", color=C["danger"],
                    arrowprops=dict(arrowstyle="-|>", color=C["danger"], lw=1.4))
    ax.set_xlim(0, 2.55)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Activation value, one ResNet-50 layer (illustrative)")
    sp.save(fig, out / f"calhist_step{k}.svg")

# ---------------------------------------------------------------------------
# 9. Per-channel ranges: the slide now shows Nagel et al. 2019, Fig. 2 as
#    published (nagel2019-fig2.png); the schematic redraw was retired
#    2026-10-18.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10. Per-channel scales restore MobileNetV2 (Krishnamoorthi 2018, Tables 3, 4)
# ---------------------------------------------------------------------------
pc_series = [
    ("Float", "neutral", [num("mbv2_float"), num("r50_float")]),
    ("INT8 per-tensor", "danger", [num("mbv2_tensor"), num("r50_tensor")]),
    ("INT8 per-channel", "data", [num("mbv2_channel"), num("r50_channel")]),
    ("INT8 per-channel + QAT", "memory", [num("mbv2_qat"), num("r50_qat")]),
]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    shown = [[1, 1], [1, 1], [1, 1], [k >= 2, k >= 2]]
    vbars(ax, ["MobileNetV2", "ResNet-50"], pc_series, shown, width=0.19, ymax=92)
    for si, (sname, ck, _) in enumerate(pc_series):
        vis = all(shown[si])
        ax.text(1.02, 0.92 - si * 0.16, sname, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=C[ck], alpha=1.0 if vis else 0.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("ImageNet top-1 (%)")
    sp.save(fig, out / f"perchannel_step{k}.svg")

# Synthesis: MobileNetV2 alone, large enough to read in half a slide
syn_rows = [
    dict(label="Float", value=num("mbv2_float"), color="neutral", text=val("mbv2_float")),
    dict(label="INT8 per-tensor", value=num("mbv2_tensor"), color="danger", text=val("mbv2_tensor")),
    dict(label="INT8 per-channel", value=num("mbv2_channel"), color="data", text=val("mbv2_channel")),
    dict(label="+ QAT", value=num("mbv2_qat"), color="memory", text=val("mbv2_qat")),
]
bars_save(out / "synth_acc.svg", syn_rows, [1, 1, 1, 1], xlim=(0, 85), size=(4.0, 2.3),
          xlabel="MobileNetV2 top-1 (%)")

# ---------------------------------------------------------------------------
# 11. Outliers in large language models (Dettmers et al. 2022, Table 1)
# ---------------------------------------------------------------------------
size_b = [r["params_b"] for r in calcs.DET]
f32 = [r["fp32"] for r in calcs.DET]
absmax = [r["absmax"] for r in calcs.DET]
llmint8 = [r["llm_int8"] for r in calcs.DET]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.axvspan(6.0, 15, color=C["danger_l"], alpha=0.6 if k == 2 else 0.0, zorder=0)
    ax.plot(size_b, f32, "o-", color=C["neutral"], lw=2.2, ms=7)
    ax.plot(size_b, absmax, "o-", color=C["danger"], lw=2.2, ms=7)
    a = 1.0 if k == 2 else 0.0
    ax.plot(size_b, llmint8, "s--", color=C["data"], lw=2.0, ms=6, alpha=a)
    ax.text(19, absmax[-1], f"INT8, one scale\nper tensor: {val('det_absmax_large')}", fontsize=11,
            fontweight="bold", color=C["danger"], va="center")
    ax.text(19, 12.2, f"32-bit float: {val('det_fp32_large')}" if k == 1 else
            f"32-bit float and\nLLM.int8(): {val('det_fp32_large')}", fontsize=11, fontweight="bold",
            color=C["neutral"] if k == 1 else C["data"], va="center")
    ax.text(6.3, 60, "outlier features\nin every layer", fontsize=10.5,
            fontweight="bold", color=C["danger"], alpha=a)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.09, 15)
    ax.set_ylim(10, 100)
    ax.set_xticks(size_b)
    ax.set_xticklabels([f"{x * 10 ** 3:g}M" if x < 1 else f"{x:g}B" for x in size_b])
    ax.set_yticks([10, 15, 20, 30, 50, 100])
    ax.set_yticklabels(["10", "15", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Parameters")
    ax.set_ylabel("C4 perplexity (lower\nis better, log scale)")
    ax.grid(color=C["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"llmint8_step{k}.svg")

# ---------------------------------------------------------------------------
# 12. The cliff: 8-bit against 4-bit weights (Krishnamoorthi 2018, Tables 3-5)
# ---------------------------------------------------------------------------
cliff_series = [
    ("Float", "neutral", [num("r50_float"), num("mbv2_float")]),
    ("INT8 per-channel PTQ", "data", [num("r50_channel"), num("mbv2_channel")]),
    ("INT4 per-tensor PTQ", "danger", [num("r50_w4_tensor"), num("mbv2_w4_tensor")]),
    ("INT4 per-channel PTQ", "compute", [num("r50_w4_channel"), num("mbv2_w4_channel")]),
    ("INT4 per-channel QAT", "memory", [num("r50_w4_qat"), num("mbv2_w4_qat")]),
]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    vis = [k >= 1, k >= 1, k >= 2, k >= 2, k >= 3]
    shown = [[v, v] for v in vis]
    vbars(ax, ["ResNet-50", "MobileNetV2"], cliff_series, shown, width=0.155, ymax=92)
    for si, (sname, ck, _) in enumerate(cliff_series):
        ax.text(1.02, 0.95 - si * 0.15, sname, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", color=C[ck], alpha=1.0 if vis[si] else 0.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("ImageNet top-1 (%)")
    sp.save(fig, out / f"cliff_step{k}.svg")

# ---------------------------------------------------------------------------
# 13. GPTQ on OPT-175B and AWQ's salient channels on OPT-6.7B
# ---------------------------------------------------------------------------
gptq_rows = [
    dict(label="FP16", value=num("gptq_fp16"), color="neutral", text=val("gptq_fp16")),
    dict(label="4-bit, round to nearest", value=num("gptq_rtn4"), color="danger", text=val("gptq_rtn4")),
    dict(label="4-bit, GPTQ", value=num("gptq_4"), color="data", text=val("gptq_4")),
    dict(label="3-bit, round to nearest", value=16, color="danger",  # bar cut at the axis end
         text="", tx=10.6),
    dict(label="3-bit, GPTQ", value=num("gptq_3"), color="data", text=val("gptq_3")),
]
fig, ax = bars(None, gptq_rows, [1] * 5, xlim=(0, 16), size=(5.8, 2.5),
               xlabel="OPT-175B WikiText2 perplexity (lower is better)", boxed=True)
for dx in (13.6, 14.0):  # break marks on the cut bar
    ax.plot([dx - 0.15, dx + 0.15], [0.69, 1.31], color="white", lw=3, zorder=5)
ax.text(0.3, 1, f"{val('gptq_rtn3_k')} thousand: bar cut off", va="center", fontsize=12,
        fontweight="bold", color="white", zorder=6)
sp.save(fig, out / "gptq.svg")

awq_rows = [
    dict(label="FP16", value=num("awq_fp16"), color="neutral", text=val("awq_fp16")),
    dict(label="INT3, round to nearest", value=num("awq_rtn"), color="danger", text=val("awq_rtn")),
    dict(label=f"Keep {val('awq_kept_pct')}% by weight size", value=num("awq_by_weight"), color="compute",
         text=val("awq_by_weight")),
    dict(label=f"Keep {val('awq_kept_pct')}% by activation size", value=num("awq_by_act"), color="data",
         text=val("awq_by_act")),
]
for k in (1, 2):
    bars_save(out / f"awq_step{k}.svg", awq_rows, [1, 1, k >= 2, k >= 2], xlim=(0, 30),
              size=(5.8, 2.1), xlabel="OPT-6.7B WikiText perplexity (lower is better)")

# ---------------------------------------------------------------------------
# 14. Two mechanisms on a schematic roofline. Positions are illustrative; the
#     ratios are the chapter's: decode 3.3 -> 1.06 GB per token, 65 -> 21 ms
#     (Napkin Math 10.4, 3.1x); H100 dense INT8 1,979 against FP16 989
#     tera-ops/s (Napkin Math 10.2). One shape stands for both machines.
#     Step 1: FP16 points. Step 2: decode moves along the slope (fewer bytes
#     per operation). Step 3: a second, INT8 roof at twice the height; the
#     MLP point rises to it (INT8 also halves its bytes, so it moves right);
#     the decode point does not move.
# ---------------------------------------------------------------------------
RIDGE, ROOF = 100.0, 1.0
xs = np.logspace(-1.5, 4.5, 300)
MS16, MS4 = val("dec_t16_ms"), val("dec_t4_ms")
SPEEDUP = num("dec_speedup")
INT8_X = val("h100_int8_x")
dec16, dec4 = 6.0, 6.0 * SPEEDUP   # position illustrative; the ratio is computed
mlp16, mlp8 = 3000.0, 6000.0
FS = 10.5
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    roof = np.minimum(ROOF, xs / RIDGE * ROOF)
    ax.plot(xs[xs <= RIDGE], roof[xs <= RIDGE], color=C["memory"], lw=3)
    ax.plot(xs[xs >= RIDGE], roof[xs >= RIDGE], color=C["compute"], lw=3)
    ax.text(0.75, 0.028, "memory slope", color=C["memory"], fontsize=FS,
            fontweight="bold", ha="center", va="center", rotation=44)
    # decode points
    ax.plot([dec16], [dec16 / RIDGE], "o", ms=10, color=C["memory"], mec="white",
            mew=1.5, zorder=5, alpha=0.45 if k >= 2 else 1.0)
    ax.annotate(f"Decode, FP16:\n{MS16} ms per token", (dec16, dec16 / RIDGE), (8, -6),
                textcoords="offset points", ha="left", va="top", fontsize=FS,
                fontweight="bold", color=C["memory"])
    a2 = 1.0 if k >= 2 else 0.0
    if True:
        ax.annotate("", (dec4, dec4 / RIDGE), (dec16 * 1.08, dec16 / RIDGE * 1.08),
                    arrowprops=dict(arrowstyle="-|>", color=C["data"], lw=2.2, alpha=a2))
        ax.plot([dec4], [dec4 / RIDGE], "o", ms=10, color=C["data"], mec="white",
                mew=1.5, zorder=5, alpha=a2)
        ax.annotate(f"INT4 weights: {MS4} ms\n({val('dec_speedup')}× fewer bytes)", (dec4, dec4 / RIDGE),
                    (-8, 6), textcoords="offset points", ha="right", va="bottom",
                    fontsize=FS, fontweight="bold", color=C["data"], alpha=a2)
    # MLP points
    ax.plot([mlp16], [ROOF], "o", ms=10, color=C["compute"], mec="white", mew=1.5,
            zorder=5, alpha=0.45 if k >= 3 else 1.0)
    ax.annotate(f"MLP, large batch:\nFP16 roof, {val('h100_fp16')}", (mlp16, ROOF), (0, -12),
                textcoords="offset points", ha="center", va="top", fontsize=FS,
                fontweight="bold", color=C["compute"])
    a3 = 1.0 if k >= 3 else 0.0
    if True:
        xi = xs[(xs >= RIDGE) & (xs <= 2 * RIDGE)]
        ax.plot(xi, xi / RIDGE * ROOF, color=C["memory"], lw=2, ls="--", alpha=a3)
        xr = xs[xs >= 2 * RIDGE]
        ax.plot(xr, np.full_like(xr, 2 * ROOF), color=C["danger"], lw=2.4, ls="--", alpha=a3)
        ax.text(230, 2 * ROOF * 1.25, f"INT8 roof: {val('h100_int8')} ({INT8_X}×)", color=C["danger"],
                fontsize=FS, fontweight="bold", va="bottom", alpha=a3)
        ax.annotate("", (mlp8, 2 * ROOF * 0.95), (mlp16 * 1.06, ROOF * 1.04),
                    arrowprops=dict(arrowstyle="-|>", color=C["danger"], lw=2.0, alpha=a3))
        ax.plot([mlp8], [2 * ROOF], "o", ms=10, color=C["danger"], mec="white", mew=1.5,
                zorder=5, alpha=a3)
        ax.text(500, 0.008, "Decode does not move:\nits math stays 16-bit",
                fontsize=FS - 0.5, color=C["neutral"], fontweight="bold", va="center",
                alpha=a3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.035, 3e4)
    ax.set_ylim(3e-3, 6)
    ax.minorticks_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Operations per byte moved")
    ax.set_ylabel("Operations per second")
    ax.text(0.01, 0.99, "Schematic; positions illustrative", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"roofint_step{k}.svg")

# ---------------------------------------------------------------------------
# 15. Bytes per token for GPT-2 XL on the phone: fused against unfused
#     dequantization (Napkin Math 10.4; unfused traffic derived from Sec. 10.4.1)
# ---------------------------------------------------------------------------
W16, W4, KV = num("dec_w16_gb"), num("dec_w4_gb"), num("dec_kv_gb")


def tot(gb, ms, approx=False):
    return f"{val(gb)} GB, {'≈' if approx else ''}{val(ms)} ms"


segs = [
    ("FP16", [(W16, "memory", f"FP16 weights, {val('dec_w16_gb')} GB"), (KV, "neutral", "")],
     tot("dec_t16_gb", "dec_t16_ms")),
    ("INT4, fused", [(W4, "data", ""), (KV, "neutral", "")], tot("dec_t4_gb", "dec_t4_ms")),
    ("INT4, separate\ndequantize pass",
     [(W4, "data", ""), (W16, "danger", f"write FP16, {val('dec_w16_gb')} GB"),
      (W16, "memory", f"read FP16, {val('dec_w16_gb')} GB"), (KV, "neutral", "")],
     tot("dq_sep_gb", "dq_sep_ms", True)),
]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    for i, (name, parts, tot) in enumerate(segs):
        y = 2 - i
        vis = 1.0 if (i < 2 or k == 2) else 0.0
        x = 0
        for v, ck, lab in parts:
            ax.barh(y, v, left=x, color=C[ck], height=0.6, alpha=vis, edgecolor="white")
            if lab:
                ax.text(x + v / 2, y, lab, ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold", alpha=vis)
            x += v
        ax.text(x + 0.1, y, tot, va="center", fontsize=12, fontweight="bold",
                color=C["danger"] if i == 2 else C["text"], alpha=vis)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"], alpha=vis)
    ax.text(0.0, -0.75, f"green: packed INT4 weights, {val('dec_w4_gb')} GB;  gray: FP16 KV cache, {val('dec_kv_gb')} GB",
            fontsize=10, color=C["neutral"])
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"GB read and written per token at {val('phone_bw')} GB/s")
    sp.save(fig, out / f"dequant_step{k}.svg")

# ---------------------------------------------------------------------------
# 16. MobileNetV2 on a Pixel 2 (Krishnamoorthi 2018, Table 7)
# ---------------------------------------------------------------------------
FRAME_MS = calcs.m(calcs.BLUR.frame_budget, "ms")
FRAME_LABEL = f"{val('frame_ms')} ms frame"
px_rows = [
    dict(label="FP32, CPU", value=num("px_fp32"), color="neutral", text=f"{val('px_fp32')} ms"),
    dict(label="INT8, CPU", value=num("px_cpu"), color="compute", text=f"{val('px_cpu')} ms ({val('px_cpu_x')}×)"),
    dict(label="INT8, DSP", value=num("px_dsp"), color="data", text=f"{val('px_dsp')} ms ({val('px_dsp_x')}×)"),
]
bars_save(out / "pixel2.svg", px_rows, [1, 1, 1], xlim=(0, 150), size=(4.0, 1.5),
          label_size=15, text_size=15, axis_size=14,
          vlines=[(FRAME_MS, FRAME_LABEL, "danger")], boxed=True,
          xlabel="MobileNetV2 on a Pixel 2, one image (ms)")

# ---------------------------------------------------------------------------
# 17. Part 1 recall: weight streaming and the blur waterfall (copied code)
# ---------------------------------------------------------------------------
stream_rows = [
    dict(label="MobileNetV2, FP32", value=num("mnv2_read_ms"), color="memory",
         text=f"{val('mnv2_read_ms')} ms: {val('mnv2_read_share_pct')}% of a frame"),
    dict(label="ResNet-50, FP32", value=calcs.m(calcs.R50_READ.read_time, "ms"), color="memory",
         text=f"{val('r50_read_ms')} ms"),
    dict(label="GPT-2 XL, FP16", value=num("gpt2_read_ms"), color="danger", text=f"{val('gpt2_read_ms')} ms per token"),
]
bars_save(out / "stream.svg", stream_rows, [1, 1, 1], xlim=(0, 80), size=(3.6, 1.6),
          label_size=15, text_size=15, axis_size=14,
          vlines=[(FRAME_MS, FRAME_LABEL, "compute")], boxed=True,
          xlabel=f"Read every weight once at {val('phone_bw')} GB/s (ms)")

stages = ["FP32, CPU", "Pruned, CPU", "Pruned INT8,\naccelerator"]
memv = [num("blur_fp32_mb"), num("blur_pruned_mb"), num("blur_int8_mb")]
timev = [num("blur_cpu_ms"), num("blur_pruned_ms"), num("blur_int8_ms")]
# Drawn for a half-slide slot: stages are rows, shared by both panels, so
# every label stays at slide-legible size.
for k in (2,):  # the deck shows only step 2 of the Part 1 waterfall
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.4, 1.55), sharey=True,
                                 gridspec_kw={"wspace": 0.12})
    for ax, vals, ck, unit, ttl in ((a1, memv, "memory", "MB", "Weight memory (MB)"),
                                    (a2, timev, "compute", "ms", "Frame time (ms)")):
        for i, v in enumerate(vals):
            a = 1.0 if i < k else 0.0
            y = 2 - i
            ax.barh(y, v, color=C[ck], height=0.62, alpha=a)
            ax.text(v + max(vals) * 0.04, y, f"{v:g} {unit}", ha="left", va="center",
                    fontsize=13, fontweight="bold", color=C[ck], alpha=a)
        ax.set_xlim(0, max(vals) * 1.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color(C["neutral"])
        ax.set_title(ttl, fontsize=13, fontweight="bold")
    a1.set_yticks(range(3))
    a1.set_yticklabels(stages[::-1], fontsize=13)
    a1.tick_params(axis="y", length=0)
    a2.tick_params(axis="y", length=0)
    a1.set_ylim(-0.5, 2.5)
    a2.axvline(FRAME_MS, color=C["danger"], lw=1.5, ls="--")
    a2.text(FRAME_MS + 3, -0.45, f"{val('frame_ms')} ms budget", color=C["danger"], fontsize=12.5,
            fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, -0.08, "Illustrative scenario (Example 10.1)", fontsize=12.5,
             color=C["neutral"])
    sp.save(fig, out / f"waterfall_step{k}.svg")

# ---------------------------------------------------------------------------
# 15b. llama-bench on the M2 Ultra, redrawn at slide size (visual review
# 2026-10-21): generation and prompt throughput per weight format, from
# Literature.Series.LlamaBenchAppleSilicon through calcs.py.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(5.8, 1.55))
for ax, key, title, ck in ((axes[0], "tg", "Generate (tg 128), tokens/s", "memory"),
                           (axes[1], "pp", "Prompt (pp 512), tokens/s", "compute")):
    rows = [("F16", f"lb_{key}16"), ("Q8_0", f"lb_{key}8"), ("Q4_0", f"lb_{key}4")]
    top = max(num(k) for _, k in rows)
    for i, (lab, k) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, num(k), color=C[ck] if lab == "Q4_0" else C["neutral_l"], height=0.62)
        ax.text(num(k) + top * 0.02, y, val(k), va="center", fontsize=13, fontweight="bold",
                color=C[ck] if lab == "Q4_0" else C["text"])
        ax.text(-top * 0.03, y, lab, va="center", ha="right", fontsize=13, fontweight="bold",
                color=C["text"])
    ax.set_xlim(0, top * 1.32)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", color=C[ck])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.6)
sp.save(fig, out / "llamabench.svg")

# The frame-time-only waterfall figure was retired 2026-10-18:
# the Pixel 2 frame now shows the phone itself.

print("figures written")
