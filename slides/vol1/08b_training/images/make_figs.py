#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8b (Model Training, Part 2).

Every number is read from MLSysIM through calcs.py (val() for a printed
label, num() for a plotted value, or the scenario itself for a curve):
the scoreboard and the batch line from the GPT-2 XL step scenarios; the
precision ladder from the hardware registry; the number formats and the
swamped update from FLOAT_FORMAT_BITS and Fp16WeightSmallUpdates; the
loss-scaling histogram from SsdGradientLossScaling (Micikevicius et al. 2018,
Figure 3); the per-layer bytes from Gpt2XlBatch32Activations; the tiles and
stored tensors from AttentionScores4096TokensFp32; checkpointing from
Gpt2XlBatch32SegmentCheckpointed; selective recomputation from
Literature.TrainingMemory (Korthikanti et al. 2023, Table 4). The GradScaler
trace is our run (scaler_run.py, data/scaler_trace.csv). Coordinates and
axis limits are layout, not data.

Run from this folder: python3 make_figs.py
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
KIT = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol1"
CAP = num("h100_gb")  # H100 HBM, 80 GiB in decimal GB
STATIC = num("static_gb")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci_label(key):
    """A value recorded as key_m, key_e, printed as m×10ᵉ for a plot label."""
    return f"{val(key + '_m')}×10{val(key + '_e').translate(SUP)}"


def pow2(key):
    """2 raised to a recorded power, for an axis label: 2⁻²⁴."""
    return "2" + val(key).translate(SUP)


def label(ax, x, y, text, color, shown=True, **kw):
    """A bold label drawn in every step (invisible until shown), so every
    step has the same bounding box and nothing moves on the slide."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


# 1. Running scoreboard (copied from Part 1's generator) --------------------
names = ["FP32", "Mixed\nprecision", "Flash-\nAttention", "2 micro-\nbatches"]
TOTAL_KEYS = ["fp32_step_gb", "mp_step_gb", "flash_step_gb", "accum_step_gb"]
totals = [num(k) for k in TOTAL_KEYS]


def scoreboard(path, k, costs=None, alt=False):
    """alt adds option B of the vote, segment checkpointing at 35.5 GB, as a
    hollow bar beside the chosen option (Napkin Math 8.8)."""
    # Type sized for the column slots this figure also fills (about 3.5 in).
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    for i, t in enumerate(totals):
        shown = i < k
        a = 1.0 if shown else 0.0
        xpos = i - 0.17 if (alt and i == 3) else i
        width = 0.32 if (alt and i == 3) else 0.58
        ax.bar(xpos, STATIC, width, color=C["neutral"], alpha=a)
        ax.bar(xpos, t - STATIC, width, bottom=STATIC, color=C["data"], alpha=a)
        label(ax, xpos, max(t, CAP) + 18, f"{val(TOTAL_KEYS[i])} GB", C["data"] if t < CAP else C["danger"],
              shown, ha="center", fontsize=15 if not (alt and i == 3) else 14)
        if costs:
            label(ax, i, 395, costs[i], C["compute"], i > 0,
                  ha="center", va="bottom", fontsize=11.5, linespacing=1.1)
    if alt:
        ax.bar(3.19, num("ckpt_step_gb"), 0.32, color="white", edgecolor=C["compute"], lw=1.8, ls="--")
        ax.text(3.19, 150, f"B:\n{val('ckpt_step_gb')}", color=C["compute"], fontsize=14,
                fontweight="bold", ha="center", va="bottom", linespacing=1.0)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
    ax.text(3.55, CAP, f"H100\n{val('h100_gb')} GB\n({val('h100_gib')} GiB)", color=C["memory"], fontsize=14,
            fontweight="bold", ha="left", va="center")
    if costs:
        ax.text(2.0, 540, "What each step cost", color=C["compute"], fontsize=13,
                fontweight="bold", ha="center")
    else:
        if alt:
            ax.text(3.45, 425, "dashed: B, checkpoint", color=C["compute"], fontsize=14,
                    fontweight="bold", ha="right")
        ax.text(3.45, 575, f"gray: static {val('static_gb')} GB", color=C["neutral"], fontsize=14,
                fontweight="bold", ha="right")
        ax.text(3.45, 500, "green: activations", color=C["data"], fontsize=14,
                fontweight="bold", ha="right")
    ax.set_xticks(range(4), names, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 660)
    ax.set_ylabel("GB per step", fontsize=14)
    sp.save(fig, path)


for k in (1, 2):
    scoreboard(out / f"scoreboard_step{k}.svg", k)
scoreboard(out / "scoreboard_step4.svg", 4, alt=True)
scoreboard(out / "scoreboard_costs.svg", 4,
           costs=["", "FP32 master,\nloss scaling", "recomputed\ntiles",
                  f"update every\n{val('accum_k_word')} passes"])

# 2. Bytes per parameter (copied from Part 1's generator) ---------------------
fig, ax = plt.subplots(figsize=(5.8, 1.5))
x = 0
for w, text, ck in [(num("mp_w_b"), "W", "memory"), (num("mp_w_b"), "G", "network"),
                    (num("mp_master_b"), "master", "neutral"),
                    (num("mp_moments_b"), "Adam m, v (FP32)", "compute")]:
    ax.barh(0, w, left=x, color=C[ck], height=0.6, edgecolor="white")
    ax.text(x + w / 2, 0, text, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    x += w
ax.text(x + 0.3, 0, f"{val('mp_total_b')} B per parameter", va="center", fontsize=13, fontweight="bold",
        color=C["danger"])
ax.set_xlim(0, 24)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
sp.save(fig, out / "static_bytes.svg")

# 3. Peak dense throughput by format, four generations (book figure 10) -------
fmts = [("FP32", "#333333"), ("TF32", C["memory"]), ("FP16/BF16", C["data"]),
        ("FP8", C["compute"]), ("FP4", C["danger"])]
gens = [(f"{hw.name.replace('NVIDIA ', '')}\n({hw.release_year})", rates)
        for hw, rates in zip(calcs.LADDER, calcs.LADDER_RATES)]
show_label = {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), (3, 4)}
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    w = 0.17
    for g, (gname, vals) in enumerate(gens):
        shown = g == 0 or k >= 2
        dim = k == 3 and g != 2
        for j, v in enumerate(vals):
            if v is None:
                continue
            xpos = g + (j - 2) * w
            a = (0.35 if dim else 1.0) if shown else 0.0
            ax.bar(xpos, v, w * 0.92, color=fmts[j][1], alpha=a)
            if (g, j) in show_label:
                txt = f"{v:,.4g}" if v < 100 else f"{v:,.0f}"
                label(ax, xpos, v * 1.18, txt, fmts[j][1], shown and not dim,
                      ha="center", va="bottom", fontsize=10)
    for (fname, col), xk in zip(fmts, [-0.45, 0.15, 0.8, 1.9, 2.45]):
        ax.text(xk, 3.2e4, fname, color=col, fontsize=11, fontweight="bold", va="bottom")
    label(ax, 2.0, 5000, f"{val('h100_fp32')} against {val('h100_peak')}: {val('fp16_over_fp32')}×", C["danger"], k == 3, ha="center",
          fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(5, 6e4)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(range(4), [g[0] for g in gens], fontsize=11)
    ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylabel("Peak dense TFLOP/s")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"precision_ladder_step{k}.svg")

# 4. Number formats: bits to scale, and the range each one covers ------------
def float_range(fmt):
    return mlsysim_physics.calc_float_range(*calcs.BITS[fmt])


mlsysim_physics = calcs.mlsysim.physics
f32, f16, b16 = (float_range(f) for f in ("fp32", "fp16", "bf16"))
digits = {f: calcs.digits(f) for f in ("fp32", "fp16", "bf16")}
rows = [("FP32", *calcs.BITS["fp32"], f"about {round(digits['fp32'])} digits", (f32[1], f32[0]), None),
        ("FP16", *calcs.BITS["fp16"], f"about {round(digits['fp16'])} digits", (f16[1], f16[0]), (f16[2], f16[1])),
        ("BF16", *calcs.BITS["bf16"], f"{np.floor(digits['bf16']):.0f} to {np.ceil(digits['bf16']):.0f} digits",
         (b16[1], b16[0]), None)]
for k in range(1, 5):
    fig = plt.figure(figsize=(5.8, 2.75))
    axb = fig.add_axes([0.13, 0.12, 0.36, 0.72])
    axr = fig.add_axes([0.55, 0.12, 0.43, 0.72])
    for i, (name, e, m, digits, (lo, hi), sub) in enumerate(rows):
        y = 2 - i
        shown = i < k
        a = 1.0 if shown else 0.0
        axb.barh(y, 1, left=0, color=C["neutral_l"], height=0.55, alpha=a, edgecolor="white")
        axb.barh(y, e, left=1, color=C["memory"], height=0.55, alpha=a, edgecolor="white")
        axb.barh(y, m, left=1 + e, color=C["compute"], height=0.55, alpha=a, edgecolor="white")
        label(axb, 1 + e / 2, y, str(e), "white", shown, ha="center", va="center", fontsize=11)
        label(axb, 1 + e + m / 2, y, str(m), "white", shown, ha="center", va="center",
              fontsize=11)
        label(axb, -0.6, y + 0.1, name, C["text"], shown, ha="right", va="center", fontsize=13)
        label(axb, -0.6, y - 0.27, digits, C["neutral"], shown, ha="right", va="center",
              fontsize=9.5, fontweight="normal")
        axr.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=C["data"],
                 height=0.45, alpha=a)
        if sub:
            axr.barh(y, np.log10(sub[1]) - np.log10(sub[0]), left=np.log10(sub[0]),
                     color=C["data_l"], height=0.45, alpha=a)
    label(axr, 3, 1.38, f"{sci_label('fp16_min')} to {val('fp16_max')}", C["data"], k >= 2, ha="left",
          va="bottom", fontsize=11)
    e38 = val("fp32_min_e").translate(SUP)
    label(axr, 3, 2.38, f"about 10{e38} to 10{e38.replace('⁻', '')}", C["data"], k >= 1, ha="left",
          va="bottom", fontsize=11)
    show_grad = k == 4
    axr.axvline(num("grad_e"), color=C["danger"], lw=2, ls="--", alpha=1.0 if show_grad else 0.0)
    label(axr, -7.5, 0.5, f"gradient 10{val('grad_e').translate(SUP)}", C["danger"], show_grad, ha="right",
          va="center", fontsize=11)
    axb.text(0, 2.62, "sign", color=C["neutral"], fontsize=10, fontweight="bold")
    axb.text(5.5, 2.62, "exponent", color=C["memory"], fontsize=10, fontweight="bold")
    axb.text(16, 2.62, "mantissa bits", color=C["compute"], fontsize=10, fontweight="bold")
    axb.set_xlim(0, 32.5)
    axb.set_ylim(-0.6, 2.95)
    axb.axis("off")
    axr.set_xlim(-42, 42)
    axr.set_ylim(-0.6, 2.95)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.set_xticks([-38, -20, 0, 20, 38], ["10⁻³⁸", "10⁻²⁰", "1", "10²⁰", "10³⁸"])
    axr.set_xlabel("Normal range (log scale)", fontsize=11)
    sp.save(fig, out / f"fp_formats_step{k}.svg")

# 5. Swamping: an update below half the FP16 spacing rounds away -------------
SW = calcs.SWAMP
W0, UPD, N_UPD = SW.weight, SW.update, SW.repeats
spacing = SW.spacing("fp16")
hist16, hist32 = SW.trace("fp16"), SW.trace("fp32")
for k in (1, 2):
    fig = plt.figure(figsize=(5.8, 2.5))
    axg = fig.add_axes([0.02, 0.2, 0.44, 0.62])
    axl = fig.add_axes([0.6, 0.2, 0.38, 0.62])
    # left: the FP16 grid near 0.5
    axg.axhline(0, color=C["neutral"], lw=1.5)
    for j in range(-1, 3):
        xv = W0 + j * spacing
        axg.plot([xv, xv], [-0.18, 0.18], color=C["memory"], lw=2.5)
    axg.text(W0, -0.3, val("sw_weight"), ha="center", va="top", fontsize=12, fontweight="bold",
             color=C["text"])
    axg.text(W0 + spacing, -0.3, val("sw_next"), ha="center", va="top", fontsize=11,
             color=C["memory"])
    axg.text(W0, -0.78, f"{val('sw_weight')} + {val('sw_update')} is stored as {val('sw_fp16')}", ha="left", va="top",
             fontsize=11, color=C["danger"], fontweight="bold")
    axg.annotate("", xy=(W0 + UPD, 0.42), xytext=(W0, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2.2))
    axg.text(W0 + UPD / 2, 0.55, f"+{val('sw_update')}", ha="center", va="bottom", fontsize=12,
             fontweight="bold", color=C["compute"])
    axg.plot([W0 + UPD], [0], "o", ms=8, mfc="white", mec=C["compute"], mew=2)
    axg.annotate("", xy=(W0, 1.0), xytext=(W0 + spacing, 1.0),
                 arrowprops=dict(arrowstyle="<->", color=C["memory"], lw=1.5))
    axg.text(W0 + spacing / 2, 1.12, f"FP16 spacing {sci_label('sw_fp16_sp')}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=C["memory"])
    axg.set_xlim(W0 - 1.3 * spacing, W0 + 2.4 * spacing)
    axg.set_ylim(-1.2, 1.45)
    axg.axis("off")
    # right: 10,000 updates
    a = 1.0 if k == 2 else 0.0
    steps = np.arange(len(hist16))
    axl.plot(steps, hist32, color=C["memory"], lw=2.5, alpha=a)
    axl.plot(steps, hist16, color=C["danger"], lw=2.5, alpha=a)
    label(axl, N_UPD / 2, 1.12, f"FP32: {val('sw_fp32')}", C["memory"], k == 2, ha="right", fontsize=12)
    label(axl, 0.98 * N_UPD, 0.58, f"FP16: {val('sw_fp16')}", C["danger"], k == 2, ha="right", fontsize=12)
    axl.set_xlim(0, N_UPD)
    axl.set_ylim(0.4, 1.6)
    axl.set_xticks([0, N_UPD / 2, N_UPD], ["0", f"{N_UPD / 2:,.0f}", val("sw_repeats")])
    axl.set_yticks([0.5, 1.0, 1.5])
    axl.set_xlabel(f"Updates of +{val('sw_update')}", fontsize=11)
    axl.set_ylabel("Stored weight", fontsize=11)
    for sname in ("bottom", "left"):
        axl.spines[sname].set_alpha(a)
    axl.tick_params(colors=C["text"] if k == 2 else "white")
    axl.xaxis.label.set_alpha(a)
    axl.yaxis.label.set_alpha(a)
    sp.save(fig, out / f"swamp_step{k}.svg")

# 6. Loss scaling on measured gradients: Micikevicius et al. 2018, Figure 3
# (SSD activation gradients, FP32 training), bins read from the figure in
# calcs.py; the zero bin (67 percent, caption) is a label, not a bar. Step 2
# adds the same histogram scaled by 8, the factor the paper found sufficient.
bins = [(b["low"], b["high"], b["percent"]) for b in calcs.LS.bins if b["percent"]]
SHIFT = calcs.LS.shift
SUBF, NORMF = calcs.LS.subnormal_floor_log2, calcs.LS.normal_floor_log2
OVF = np.log2(calcs.LS.largest)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 2.25))
    ax.axvspan(SUBF, NORMF, color=C["data_l"], alpha=0.6, lw=0)
    ax.axvspan(NORMF, OVF, color=C["data_l"], alpha=1.0, lw=0)
    ax.axvspan(-62, SUBF, color=C["danger_l"], alpha=0.6, lw=0)
    for lo, hi, p in bins:
        ax.bar(lo, p, hi - lo, align="edge", color=C["neutral"], edgecolor="white", lw=0.6,
               alpha=1.0 if k == 1 else 0.3, bottom=0)
        ax.bar(lo + SHIFT, p, hi - lo, align="edge", color="none", edgecolor=C["compute"],
               lw=1.8, alpha=1.0 if k == 2 else 0.0)
    ax.axvline(OVF, color=C["danger"], lw=2.5)
    ax.text(15.3, 4.5, f"overflow\n{val('fp16_max')}", color=C["danger"], fontsize=11,
            fontweight="bold", ha="right", va="top", linespacing=1.0)
    ax.text(-2, 250, "FP16 normal", color=C["data"], fontsize=12, fontweight="bold",
            ha="center", va="top")
    ax.annotate("", xy=(14.5, 0.011), xytext=(-5.5, 0.011),
                arrowprops=dict(arrowstyle="<->", color=C["data"], lw=1.5))
    ax.text(4.5, 0.014, "unused", color=C["data"], fontsize=11, fontweight="bold",
            ha="center", va="bottom")
    ax.text(-19, 250, "sub-\nnormal", color=C["data"], fontsize=11, ha="center", va="top",
            linespacing=1.0)
    ax.text(-43, 250, "becomes zero in FP16", color=C["danger"], fontsize=12,
            fontweight="bold", ha="center", va="top")
    ax.text(-60.5, 1.6, f"+ {val('ls_zero_pct')}% exactly\nzero (own bin)", color=C["neutral"], fontsize=10.5,
            ha="left", va="center", linespacing=1.0)
    label(ax, -7.5, 0.045, f"orange: × {val('ls_scale')}, shift by {val('ls_shift')}:\nSSD then matched FP32", C["compute"],
          k == 2, ha="left", va="bottom", fontsize=11, linespacing=1.05)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 300)
    ax.set_yticks([0.01, 0.1, 1, 10], ["0.01%", "0.1%", "1%", "10%"])
    ax.minorticks_off()
    ax.set_xlim(-62, 18)
    ax.set_xticks([-60, -45, SUBF, NORMF, 0, OVF],
                  ["2⁻⁶⁰", "2⁻⁴⁵", pow2("fp16_sub_log2"), pow2("fp16_min_log2"), "1", "2¹⁶"], fontsize=11)
    ax.set_xlabel("Activation-gradient magnitude, SSD (log scale)", fontsize=11)
    ax.set_ylabel("Share of values", fontsize=11)
    sp.save(fig, out / f"loss_scale_step{k}.svg")

# 6b. A real GradScaler trace (scaler_run.py, data/scaler_trace.csv) ---------
rows = calcs.TRACE
steps = np.array([r[0] for r in rows])
logs = np.log2([r[1] for r in rows])
skips = [(r[0], np.log2(r[1])) for r in rows if r[2]]
fig, ax = plt.subplots(figsize=(3.4, 2.55))
ax.step(steps / 1000, logs, where="post", color=C["memory"], lw=2.2)
ax.plot([s / 1000 for s, _ in skips], [v + 1 for _, v in skips], "o", ms=6,
        color=C["danger"], mec="white", mew=1)
ax.text(1, 23.2, f"× 2 every\n{val('gs_interval')} clean\nsteps", color=C["memory"], fontsize=11,
        fontweight="bold", va="center", linespacing=1.05)
ax.text(27.5, 26.1, "overflow: skip, ÷ 2", color=C["danger"], fontsize=11,
        fontweight="bold", ha="right", va="bottom")
ax.set_xlim(0, 28)
ax.set_ylim(15, 27.4)
ax.set_yticks([16, 20, 24], ["2¹⁶", "2²⁰", "2²⁴"])
ax.set_xticks([0, 10, 20], ["0", "10k", "20k"])
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Loss scale S", fontsize=11)
ax.grid(color=C["grid"], lw=0.5)
ax.set_axisbelow(True)
sp.save(fig, out / "scaler_trace.svg")

# 7. Run time by format (after Part 1's precision_days, narrowed for a column)
sp.hbars(out / "precision_days_col.svg", [
    ("FP32", num("fp32_days"), "neutral", f"{val('fp32_days')} days"),
    ("TF32", num("tf32_days"), "memory", f"{val('tf32_days')} days"),
    ("BF16", num("run_days"), "compute", f"{val('run_days')} days, ${val('run_usd')}"),
], "days", log=False, size=(3.9, 2.2),
    xlabel=f"Days on one H100, {val('eta_pct')}% utilization")

# 8. One layer's activation bytes at batch 32 (recall from Part 1), then the
# three saved copies of the scores that FlashAttention stops storing --------
ACT = calcs.ACT
mem = calcs.mlsysim.physics.memory
PROJ = ACT.projection_activations_per_layer.to("MB").magnitude
FFN = ACT.ffn_activations_per_layer.to("MB").magnitude
SCORES = ACT.score_activations_per_layer.to("MB").magnitude
NORM = ACT.normalization_per_layer.to("MB").magnitude
parts = [(PROJ, "memory"), (FFN, "compute"), (SCORES, "danger"), (NORM, "neutral")]
copies = [(mem.SCORE_SOFTMAX_BYTES, f"P\n{mem.SCORE_SOFTMAX_BYTES} B"),
          (mem.SCORE_DROPOUT_MASK_BYTES, f"mask\n{mem.SCORE_DROPOUT_MASK_BYTES} B"),
          (mem.SCORE_DROPPED_BYTES, f"dropped P\n{mem.SCORE_DROPPED_BYTES} B")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    x = 0
    for w, ck in parts:
        ax.barh(1.45, w, left=x, color=C[ck], height=0.5, edgecolor="white")
        x += w
    ax.text(x + 150, 1.45, f"{x:,.0f} MB", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    for xt, t, ck in [(0, f"proj. {val('layer_proj_mb')}", "memory"),
                      (2300, f"scores {val('layer_scores_mb')}: {val('layer_scores_pct')}%", "danger")]:
        ax.text(xt, 1.0, t, color=C[ck], fontsize=12, fontweight="bold", va="center")
    ax.text(8100, 2.05, f"× {val('gpt2_layers')} layers: scores are {val('scores_gb')} of {val('act_gb')} GB", color=C["danger"],
            fontsize=12, fontweight="bold", va="center", ha="right")
    shown = k == 2
    a = 1.0 if shown else 0.0
    x0 = PROJ + FFN
    for nbytes, t in copies:
        w = SCORES * nbytes / mem.KORTHIKANTI_SCORE_BYTES
        ax.barh(0, w, left=x0, color=C["danger_l"], height=0.62, edgecolor=C["danger"],
                lw=1.5, hatch="//", alpha=a)
        # each copy is named under its box: the 1-byte mask box is too
        # narrow to hold its label
        label(ax, x0 + w / 2, -0.4, t, C["danger"], shown, ha="center", va="top",
              fontsize=12, linespacing=1.0)
        x0 += w
    for xe in (PROJ + FFN, PROJ + FFN + SCORES):
        ax.plot([xe, xe], [1.2, 0.31], color=C["danger"], lw=1.2, ls=":", alpha=a)
    label(ax, 5250, 0, "FlashAttention\nstores none; it keeps\na max and a sum per row", C["memory"],
          shown, va="center", fontsize=12, linespacing=1.05)
    ax.set_xlim(0, 8200)
    ax.set_ylim(-1.3, 2.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 2000, 4000], ["0", "2,000", "4,000"])
    ax.set_xlabel(f"MB saved per layer for backward, batch {val('batch')}, FP16", fontsize=12)
    sp.save(fig, out / f"layer_bytes_step{k}.svg")

# 9. Standard attention on the kit's accelerator node -----------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def kit_body(name):
    """The kit node without the host side, shifted left so the accelerator
    starts at x = 10; the right of the canvas is free for annotation."""
    keep = []
    for line in (KIT / f"{name}.svg").read_text().splitlines():
        if not line.startswith("<") or line.startswith(("<svg", "</svg", "<title")):
            continue
        xs = re.findall(r'\bx1?="([\d.]+)"', line)
        if xs and float(xs[0]) < 320:
            continue
        keep.append(line)
    return '<g transform="translate(-320,0)">\n' + "\n".join(keep) + "\n</g>"


def text(x, y, s, color, size=20, weight=700, anchor="start"):
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


for k in (1, 2, 3):
    body = kit_body("machine_hl_hbm" if k != 2 else "machine_hl_sram")
    extra = []
    blue, red, gray = C["memory"], "#A51C30", "#888888"
    extra.append(text(575, 368, "HBM", blue, 27))
    extra.append(text(575, 402, f"{val('h100_gib')} GB, {val('h100_bw')} TB/s", blue, 25, 400))
    if k >= 2:
        side = blue if k == 2 else gray
        extra.append(text(575, 50, "On-chip SRAM", side, 27))
        extra.append(text(575, 84, f"{val('sram_lo_mb')} to {val('sram_hi_mb')} MB,", side, 25, 400))
        extra.append(text(575, 116, f"{val('sram_bw_lo')} to {val('sram_bw_hi')} TB/s", side, 25, 400))
        extra.append(text(575, 148, f"{val('h100_smem_kb')} KB per SM", side, 25, 400))
    if k == 3:
        # SRAM bypassed: gray it out
        extra.append('<rect x="76" y="106" width="138" height="218" rx="10" fill="white" '
                     'fill-opacity="0.9" stroke="none"/>')
        extra.append(text(145, 270, "bypassed", gray, 22, 700, "middle"))
        # four numbered transfers between the compute units and HBM
        for j, (xa, down) in enumerate([(290, True), (350, False), (410, True), (470, False)]):
            y1, y2 = (318, 380) if down else (380, 318)
            head = y2 - 10 if down else y2 + 10
            extra.append(f'<line x1="{xa}" y1="{y1}" x2="{xa}" y2="{head}" stroke="{red}" '
                         'stroke-width="5"/>')
            tip = (f"{xa - 11},{head} {xa + 11},{head} {xa},{y2 + 6}" if down else
                   f"{xa - 11},{head} {xa + 11},{head} {xa},{y2 - 6}")
            extra.append(f'<polygon points="{tip}" fill="{red}"/>')
            extra.append(f'<circle cx="{xa}" cy="347" r="15" fill="white" stroke="{red}" '
                         'stroke-width="2.5"/>')
            extra.append(text(xa, 348, str(j + 1), red, 20, 700, "middle"))
        extra.append(text(575, 200, "Standard attention:", red, 25))
        extra.append(text(575, 234, "1 write S,  2 read S,", red, 25, 400))
        extra.append(text(575, 268, "3 write P,  4 read P", red, 25, 400))
        extra.append(text(10, 485, f"Scores only: d_head / 2 = {val('dhead')} / 2 = {val('attn_intensity')} FLOP per byte; ridge {val('h100_ridge')}",
                          red, 23))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" '
           'viewBox="0 0 900 500">\n<rect width="900" height="500" fill="white"/>\n'
           + body + "\n" + "\n".join(extra) + "\n</svg>\n")
    (out / f"attention_io_step{k}.svg").write_text(svg)

# 10. Tiling: one query block streams key and value blocks -------------------
for k in (1, 2):
    fig = plt.figure(figsize=(5.8, 2.7))
    axm = fig.add_axes([0.0, 0.06, 0.38, 0.86])
    axb = fig.add_axes([0.52, 0.24, 0.46, 0.56])
    n = calcs.TILE.tiles_per_side
    axm.add_patch(plt.Rectangle((0, 0), n, n, fc=C["danger_l"], ec=C["danger"], lw=2))
    for t in range(1, n):
        axm.plot([t, t], [0, n], color="white", lw=0.5)
        axm.plot([0, n], [t, t], color="white", lw=0.5)
    axm.add_patch(plt.Rectangle((0, n - 1), n, 1, fc=C["memory_l"], ec=C["memory"], lw=1.5))
    axm.add_patch(plt.Rectangle((5, n - 1), 1, 1, fc=C["memory"], ec=C["memory"], lw=1.5))
    axm.annotate("", xy=(14, n - 2.6), xytext=(6.5, n - 2.6),
                 arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=2))
    axm.text(6.5, n - 3.6, "K, V blocks stream by", color=C["memory"], fontsize=11,
             fontweight="bold", va="top")
    axm.text(n / 2, n / 2 - 3, f"{val('tile_seq')} × {val('tile_seq')} scores\n= {val('tiles_per_side')} × {val('tiles_per_side')} tiles of {val('tile_side')}",
             ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["danger"])
    axm.text(n / 2, n + 0.6, f"one query block ({val('tile_side')} rows)", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=C["memory"])
    axm.set_xlim(-0.5, n + 0.5)
    axm.set_ylim(-0.5, n + 3)
    axm.set_aspect("equal")
    axm.axis("off")
    shown = k == 2
    vals = [(calcs.TILE.score_matrix.to("byte").magnitude, "danger", f"Full matrix {val('tile_full_mb')} MB"),
            (calcs.TILE.tile_bytes.to("byte").magnitude, "memory", f"One tile {val('tile_kb')} KB")]
    for i, (v, ck, t) in enumerate(vals):
        axb.barh(1 - i, v, color=C[ck], height=0.55, alpha=1.0 if shown else 0.0)
        label(axb, v * 1.4 if i == 0 else 3.2e5, 1 - i, t, C[ck], shown, va="center",
              fontsize=12)
    SMEM = num("h100_smem_kb") * 1e3
    axb.axvline(SMEM, color=C["data"], lw=2, ls="--", alpha=1.0 if shown else 0.0)
    label(axb, SMEM, 1.55, f"{val('h100_smem_kb')} KB shared\nmemory per SM", C["data"], shown,
          ha="center", va="bottom", fontsize=10.5)
    axb.set_xscale("log")
    axb.set_xlim(1e4, 3e10)
    axb.set_ylim(-0.6, 1.5)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xticks([1e4, 1e6, 1e8], ["10 KB", "1 MB", "100 MB"])
    axb.minorticks_off()
    axb.set_xlabel("Bytes per head (FP32, log scale)", fontsize=11)
    if not shown:
        axb.tick_params(colors="white")
        axb.spines["bottom"].set_alpha(0)
        axb.xaxis.label.set_alpha(0)
    sp.save(fig, out / f"tile_grid_step{k}.svg")

# 11. What each kernel keeps for backward, per head at 4,096 tokens ----------
sp.hbars(out / "flash_store.svg", [
    ("Standard attention", num("store_std_mb"), "danger", f"{val('store_std_mb')} MB"),
    ("FlashAttention", num("store_fa_mb"), "memory", f"{val('store_fa_mb')} MB: Q, K, V, O, and m, l per row"),
], "MB", xlabel=f"Stored for backward, per head, {val('tile_seq')} tokens (MB, log scale)",
    size=sp.SHORT)

# 12. Step memory against batch size with FlashAttention ----------------------
bs = np.arange(0, 41)
PER_SEQ = calcs.ACT.per_sequence_activations.to("GB").magnitude
FIT = calcs.FLASH.resident_batch_limit
mem = STATIC + PER_SEQ * bs
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(bs, mem, color=C["data"], lw=2.8)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--")
    ax.text(0.5, CAP + 4, f"H100: {val('h100_gb')} GB ({val('h100_gib')} GiB)", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(39.5, 8, f"{val('static_gb')} GB + {val('per_seq_gb')} GB per sequence", ha="right",
            va="bottom", color=C["data"], fontsize=11, fontweight="bold")
    shown = k == 2
    for b, key, col in [(int(num("micro")), "accum_step_gb", C["data"]),
                        (int(num("batch")), "flash_step_gb", C["danger"])]:
        m = STATIC + PER_SEQ * b
        ax.plot([b], [m], "o", ms=10, color=col, mec="white", mew=1.5,
                alpha=1.0 if shown else 0.0)
        label(ax, b - 0.8, m + 5, f"batch {b}: {val(key)} GB", col, shown, ha="right",
              fontsize=11.5)
    ax.plot([FIT], [CAP], "o", ms=9, color=C["memory"], mec="white", mew=1.5)
    ax.annotate(f"about {val('fit_batch')} fit", xy=(FIT, CAP), xytext=(26.5, 52), fontsize=11.5,
                fontweight="bold", color=C["memory"],
                arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=1.5))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 135)
    ax.set_xlabel(f"Sequences resident at once ({val('seq')} tokens each)", fontsize=11)
    ax.set_ylabel("GB per step")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"batch_fit_step{k}.svg")

# 13. Segment checkpointing (copied from Part 1's generator) -----------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    for s in range(calcs.SEG.checkpoint_segments):
        x0 = s * 1.0
        ax.add_patch(plt.Rectangle((x0 + 0.02, 0.0), 0.12, 1.0, fc=C["data"], ec=C["data"]))
        rebuilt = k == 2 and s == 5
        ax.add_patch(plt.Rectangle((x0 + 0.17, 0.0), 0.78, 1.0,
                                   fc=C["compute_l"] if rebuilt else "white",
                                   ec=C["compute"] if rebuilt else C["neutral_l"],
                                   lw=2 if rebuilt else 1.2, ls="-" if rebuilt else "--"))
    label(ax, 0.0, 1.12, f"kept: {val('segments')} segment inputs, {val('seg_inputs_gb')} GB", C["data"], True,
          fontsize=15, va="bottom")
    label(ax, 5.56, -0.12, f"rebuilt in backward,\none at a time: {val('live_seg_gb')} GB", C["compute"],
          k == 2, fontsize=15, va="top", ha="center", linespacing=1.05)
    label(ax, 8.05, 1.12, "dropped", C["neutral"], True, fontsize=15, va="bottom",
          ha="right")
    label(ax, 0.56, -0.12, f"{val('segment_layers')} layers", C["neutral"], True, fontsize=15, va="top",
          ha="center")
    ax.set_xlim(-0.05, 8.1)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")
    sp.save(fig, out / f"ckpt_segments_step{k}.svg")

# 15. Activation memory against checkpoint count, 48 layers (table 13) -------
L_ = calcs.GPT2.layers
ks = np.arange(1, L_ + 1)
units = np.array([calcs.SEG.checkpoint_memory_units(int(k)) for k in ks])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(ks, units, color=C["compute"], lw=2.8)
    ax.axhline(L_, color=C["neutral"], lw=1.5, ls=":")
    ax.text(25, 51.5, f"no checkpoints: {val('gpt2_layers')}", ha="center", color=C["neutral"], fontsize=11,
            fontweight="bold")
    kb = int(num("k_best"))
    for kk, txt, xt, yt, ha in [(1, f"k = 1: {val('k_one_units')}", 2.5, 53, "left"),
                                (kb, f"k = {val('k_best')}: about {val('k_best_units')}", 9, 9.5, "left"),
                                (L_, f"k = {val('gpt2_layers')}: {val('k_one_units')}", 47.5, 53, "right")]:
        u = calcs.SEG.checkpoint_memory_units(kk)
        ax.plot([kk], [u], "o", ms=9, color=C["compute"], mec="white", mew=1.5)
        ax.text(xt, yt, txt, ha=ha, va="center", fontsize=11.5, fontweight="bold",
                color=C["compute"])
    ax.text(40, 24, f"memory ≈ k + {val('gpt2_layers')}/k\n(L = {val('gpt2_layers')} layers)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=C["compute"], linespacing=1.05)
    label(ax, 9, 3.8, f"{val('sqrt_saving_pct')}% less memory, about {val('recompute_pct')}% more arithmetic", C["data"], k == 2,
          va="center", fontsize=11.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 58)
    ax.set_xlabel("Checkpoints k (curve derived from table 13)", fontsize=11)
    ax.set_ylabel("Layers of activations")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"sqrt_ckpt_step{k}.svg")

# 16. Selective recomputation, measured (Korthikanti et al. 2023, Table 4) ---
krows = [("No recompute", num("k_none_ms"), "neutral", f"{val('k_none_ms')} ms"),
         ("Full recompute", num("k_full_ms"), "danger", f"{val('k_full_ms')} ms (+{val('k_full_pct')}%)"),
         ("Selective recompute", num("k_sel_ms"), "data", f"{val('k_sel_ms')} ms (+{val('k_sel_pct')}%)")]
sp.hbars(out / "selective_recompute.svg", krows, "ms", log=False, size=(5.8, 1.9),
         xlabel=f"One layer of a {val('k_model_b')}B GPT, forward + backward, A100 (ms)")

print("figures written")
