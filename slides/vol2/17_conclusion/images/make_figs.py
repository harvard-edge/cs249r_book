#!/usr/bin/env python3
"""Figures for Lecture 17 (Conclusion), Volume II.

Plots use the shared slideplot helper; diagrams are SVG written by the small
builder below, one file per build step on one canvas so nothing moves. Every
number comes from MLSysIM through calcs.py in this folder (imported, never
retyped): printed labels use val(key), plotted points num(key) or the same
Scenarios.* and Literature series calcs.py reads. Axis ranges and tick
positions are layout.

Diagrams follow the Volume II C3 roles: compute blue #1F407A, communication
amber #d97706, coordination violet #7c3aed, storage green, risk red.

Run from this folder: python3 make_figs.py
"""

import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import slideplot as sp  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = HERE

ETH = "#1F407A"
ETH_L = "#EAF1F8"
AMBER, AMBER_L = "#d97706", "#fef3c7"
VIOLET, VIOLET_L = "#7c3aed", "#f3e8ff"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
RED, RED_L = "#dc2626", "#fee2e2"
SLATE, SLATE_L = "#64748b", "#f1f5f9"
TEXT = "#1A1A2E"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


class Svg:
    """A tiny SVG builder: every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n with hidden elements kept transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, last=99):
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', last)

    def text(self, x, y, s, size=22, color=TEXT, weight="normal", anchor="middle",
             first=1, last=99):
        lines = s.replace("&", "&amp;").split("\n")
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = "".join(f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>'
                        for i, ln in enumerate(lines))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" '
                        f'font-weight="{weight}" text-anchor="{anchor}" '
                        f'dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, last=99):
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                        f'stroke-width="{sw}"{m}/>', last)

    def write(self, stem, steps):
        colors = [ETH, AMBER, VIOLET, GREEN, RED, SLATE, TEXT]
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for c in colors)
        for k in range(1, steps + 1):
            body = [s if first <= k <= last else '<g opacity="0">' + s + "</g>"
                    for first, last, s in self.items]
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                   f'width="{self.w}" height="{self.h}"><defs>{defs}</defs>'
                   f'<rect width="100%" height="100%" fill="white"/>' + "".join(body) + "</svg>")
            name = f"{stem}_step{k}.svg" if steps > 1 else f"{stem}.svg"
            (out / name).write_text(svg)


# =============================================================================
# 1. The fleet stack with one number per row (retrieval reveal)
#    Canvas 900 x 372 for a full-width slot (about 14.5 cm): 19 px = 8.7 pt.
# =============================================================================
rows = [  # top to bottom: layer, number, owning lecture
    ("Governance & Sustainability",
     f"Same energy: Poland grid {val('geo_ratio')}× Quebec's carbon", "L15"),
    ("Assurance", f"DP noise: ${val('sal_err')} error on a mean of {val('sal_n')} salaries", "L13"),
    ("Operations", f"{val('al_daily')} false alarms a day from {val('al_monitors')} monitors", "L12"),
    ("Runtime & Serving",
     f"{val('kv_long_n')} requests of {val('kv_context_k')}K tokens fill one {val('node_gpus')}-GPU node", "L10"),
    ("Reliability & Control", f"Checkpoint every {val('yd_tau_min')} min at {val('yd_gpus')} GPUs", "L7"),
    ("Distributed Execution",
     f"{val('params_b')}B, pure data parallel, {val('ref_gpus')} GPUs: {val('dp_eff_pct')}% efficient", "L6"),
    ("Fabric & Storage", f"{val('ckpt_gb')} GB checkpoint: {val('pause_s')} s to local NVMe", "L4"),
    ("Facility & Accelerators",
     f"{val('stair_gb')} GB: HBM {val('stair_hbm_ms')} ms, NVLink {val('stair_nvlink_ms')}, "
     f"InfiniBand {val('stair_ib_ms')}", "L2"),
]
g = Svg(900, 372)
for i, (name, number, lec) in enumerate(rows):
    y = 4 + i * 46
    g.rect(4, y, 892, 42, ETH_L, ETH, sw=1.5, rx=6)
    g.text(16, y + 21, name, 19, ETH, "bold", "start")
    g.text(300, y + 21, number, 19, TEXT, "normal", "start")
    g.text(884, y + 21, lec, 19, ETH, "bold", "end")
g.write("stack_numbers", 1)

# =============================================================================
# 2. Six principles placed on the four bands, each with its Fleet Law term
#    Canvas 900 x 380 for a full-width slot: 18 px = 8.2 pt.
# =============================================================================
g = Svg(900, 380)
bands = [  # (y, height, band, principles, color, step), top to bottom
    (4, 96, "Governance", [("5  Responsibility constrains design", "spends compute; approval gates add coordination"),
                           ("6  Sustainability is a first-order cost", "a step's energy = average power × T_step")],
     GREEN, 3),
    (106, 60, "Serving", [("4  Serving is bounded by its contract", "the same terms, counted per request")],
     VIOLET, 2),
    (172, 126, "Distribution", [("2  Communication can dominate", "T_comm(N) and T_sync(N) need not shrink"),
                                ("3  Failure is routine", "recovery adds to T_sync(N); rollback adds work")],
     AMBER, 1),
    (304, 72, "Infrastructure", [("1  Infrastructure determines capability",
                                  "sets the rates behind T_compute / N and T_comm(N)")], ETH, 1),
]
for y, h, band, prin, color, step in bands:
    g.rect(64, y, 832, h, "#ffffff", color, sw=2.5, first=step, rx=6)
    g.text(78, y + h / 2, band, 19, color, "bold", "start", first=step)
    n = len(prin)
    for j, (pr, term) in enumerate(prin):
        yy = y + h * (j + 0.5) / n
        g.text(250, yy - 11, pr, 19, TEXT, "bold", "start", first=step)
        g.text(250, yy + 12, term, 18, SLATE, "normal", "start", first=step)
g.line(30, 372, 30, 12, RED, sw=4, first=3)
g.add(3, f'<text {FONT} x="14" y="190" font-size="18" font-weight="700" fill="{RED}" '
         f'text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 14 190)">'
         f'the scale moment</text>')
g.write("principles", 3)

# =============================================================================
# 4. Tail at scale: Dean and Barroso 2013, Table 1, p99 (measured)
# =============================================================================
tail_rows = [
    ("One random leaf", num("db_one_p99"), "memory", f"{val('db_one_p99')} ms"),
    ("95% of all leaves", num("db_95_p99"), "compute", f"{val('db_95_p99')} ms"),
    ("All leaves", num("db_all_p99"), "danger", f"{val('db_all_p99')} ms"),
]
sp.hbars(out / "tail_measured.svg", tail_rows, "ms", log=False, size=(5.8, 1.9))

# =============================================================================
# 5. Interruption interval against job size: model MTTF / N, and measured
# =============================================================================
import numpy as np  # noqa: E402

mttf_h = calcs.EXP.component_mtbf.to("hour").magnitude
fig, ax = plt.subplots(figsize=(5.8, 2.75))
ns = np.logspace(2, 4.6, 100)
ax.plot(ns, mttf_h / ns, color=C["memory"], lw=2.5)
ax.text(150, mttf_h / 150 * 1.5, f"model: {val('gpu_mttf_h')} h / N", color=C["memory"],
        fontsize=12, fontweight="bold")
pts = [(num("ref_gpus"), calcs.m(calcs.EXP.mtbf, "hour"), f"{val('ref_gpus')}: {val('mtbi_ref_h')} h"),
       (num("exp_gpus"), calcs.m(calcs.EXP.expanded_mtbf, "hour"), f"{val('exp_gpus')}: {val('mtbi_exp_h')} h")]
for (x, y, lab), off, ha, va in zip(pts, [(-8, -6), (8, 6)], ["right", "left"], ["top", "bottom"]):
    ax.plot([x], [y], "o", ms=9, color=C["memory"], mec="white", mew=1.5, zorder=5)
    ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                va=va, fontsize=13, color=C["memory"], fontweight="bold")
x = num("llama_gpus")
llama_h = calcs.m(calcs.Q_(calcs.llama_days, "day"), "hour") / calcs.llama_n
ax.plot([x], [llama_h], "D", ms=9, color=C["danger"], zorder=6)
ax.annotate(f"Llama 3, measured: {val('llama_mttf_h')} h", (x, num("llama_mttf_h")), xytext=(-14, -16),
            textcoords="offset points", ha="right", va="top", fontsize=12, color=C["danger"],
            fontweight="bold")
ax.plot([x], [num("kok_16k_h")], "o", ms=8, mfc="white", mec=C["danger"], mew=2, zorder=6)
ax.annotate(f"projected: {val('kok_16k_h')} h", (x, num("kok_16k_h")), xytext=(-14, -22),
            textcoords="offset points", ha="right", va="top", fontsize=12, color=C["danger"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(100, 4e4)
ax.set_ylim(0.25, 800)
ax.set_xticks([100, 1000, 10000])
ax.set_xticklabels(["100", "1,000", "10,000"])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1 h", "10 h", "100 h"])
ax.minorticks_off()
ax.set_xlabel("GPUs the job needs at once")
ax.set_ylabel("Hours between interruptions")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "mtbi.svg")

# =============================================================================
# 5b. Doubling at an ideal 2x: run time halves, accelerator-hours do not fall
# =============================================================================
E = calcs.EXP
fig, ax = plt.subplots(figsize=(2.9, 1.35))
bars = [("run time", 1 + E.run_time_change(E.factor), f"−{val('ideal_sooner_pct')}%"),
        ("accel.-hours", 1 + E.accelerator_hours_change(E.factor), f"+{val('ideal_hours_pct')}%")]
for i, (lab, r, txt) in enumerate(bars):
    y = 1 - i
    ax.barh(y + 0.19, 1, height=0.34, color=C["neutral_l"], edgecolor=C["neutral"])
    ax.barh(y - 0.19, r, height=0.34, color=C["memory"])
    ax.text(-0.04, y, lab, ha="right", va="center", fontsize=14, fontweight="bold",
            color=C["text"])
    ax.text(r + 0.03, y - 0.19, txt, va="center", fontsize=14, fontweight="bold",
            color=C["memory"])
ax.text(1.03, 1.19, val("ref_gpus"), va="center", fontsize=13, color=C["neutral"])
ax.text(0.03, 0.81, val("exp_gpus"), va="center", fontsize=13, color="white", fontweight="bold")
ax.set_xlim(0, 1.45)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xticks([])
for s in ("left", "top", "right", "bottom"):
    ax.spines[s].set_visible(False)
sp.save(fig, out / "expand_bars.svg")

# =============================================================================
# 6. Young-Daly overhead against cluster size (derived, GPU failures only)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.6))
ns = np.logspace(np.log10(128), np.log10(16384), 100)
ov = [100 * calcs.EXP.at(n).result.waste for n in ns]
ax.plot(ns, ov, color=C["network"], lw=2.5)
for gpus, pct_key, ov_n, off, ha in [("ref_gpus", "ov_ref_pct", calcs.EXP.overhead, (-10, 10), "right"),
                                     ("exp_gpus", "ov_exp_pct", calcs.EXP.expanded_overhead, (10, -18), "left")]:
    n, v = num(gpus), 100 * ov_n
    ax.plot([n], [v], "o", ms=9, color=C["network"], mec="white", mew=1.5, zorder=5)
    ax.annotate(f"{val(gpus)} GPUs: {val(pct_key)}%", (n, v), xytext=off, textcoords="offset points",
                ha=ha, fontsize=12, color=C["network"], fontweight="bold")
ax.text(4200, 4.6, f"×{val('ov_growth')} per doubling\n(grows as √N)",
        fontsize=12, color=C["text"], ha="center")
ax.set_xscale("log")
ax.set_xlim(128, 16384)
ax.set_ylim(0, 7.5)
ax.set_xticks([128, num("ref_gpus"), num("exp_gpus"), 16384])
ax.set_xticklabels(["128", val("ref_gpus"), val("exp_gpus"), "16,384"])
ax.minorticks_off()
ax.set_xlabel("GPUs (independent GPU failures only)")
ax.set_ylabel("Checkpoint + rework (%)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "overhead.svg")

# =============================================================================
# 7. MegaScale strong scaling, 175B (measured, Jiang et al. 2024, Table 2)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.7))
gs = [int(r["gpus"]) for r in calcs.MEGA]
tp = [r["megascale_tokens_s"] / 1e3 for r in calcs.MEGA]
mid = calcs.mega(num("ms_mid_gpus"))
ideal = [mid["megascale_tokens_s"] / 1e3 * g_ / mid["gpus"] for g_ in gs]
ax.plot(gs, ideal, ls="--", color=C["neutral"], lw=1.8)
ax.text(1090, 12, f"dashed: linear from {val('ms_mid_gpus')}", color=C["neutral"], fontsize=11, ha="right",
        va="bottom")
ax.plot(gs, tp, "o-", color=C["compute"], lw=2.5, ms=9, mec="white", mew=1.5)
for g_, t in zip(gs, tp):
    ax.annotate(f"{t:.1f}k", (g_, t), xytext=(10, -14), textcoords="offset points",
                fontsize=11, color=C["compute"], fontweight="bold")
ax.annotate(f"{val('ms_mid_gpus')} to {val('ms_hi_gpus')} GPUs: {val('ms_speedup')}× "
            f"(linear: {val('exp_factor')}×)",
            (1024, tp[-1]), xytext=(256, 175), textcoords="data", fontsize=12,
            color=C["text"], fontweight="bold")
ax.set_xlim(200, 1100)
ax.set_ylim(0, 215)
ax.set_xticks(gs)
ax.set_xticklabels([f"{g_:,}" for g_ in gs])
ax.set_xlabel("GPUs (batch 768, 175B model)")
ax.set_ylabel("Tokens/s (thousands)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "megascale.svg")

# =============================================================================
# 8. Each compute doubling buys the same fraction, a smaller amount (derived)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 2.6))
mult, step = calcs.K2.multiple_factor, calcs.K2.resource_multiple
xs = [step ** k for k in range(calcs.doublings + 1)]
ls = [mult ** k for k in range(calcs.doublings + 1)]
ax.plot(xs, ls, "o-", color=C["memory"], lw=2.5, ms=9, mec="white", mew=1.5)
for k in range(1, calcs.doublings + 1):
    ax.annotate(f"−{val(f'loss_drop{k}')}", (xs[k], ls[k]), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=11, color=C["memory"], fontweight="bold")
ax.text(xs[-1], 0.995, f"each doubling: ×{val('loss_factor')} (−{val('loss_cut_pct')}%)",
        ha="right", fontsize=12, color=C["text"], fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels([f"{x:.0f}×" for x in xs])
ax.minorticks_off()
ax.set_ylim(0.85, 1.02)
ax.set_xlabel("Training compute (accelerator-hours), multiple of today")
ax.set_ylabel("Loss (today = 1)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "loss_doublings.svg")

# =============================================================================
# 9. The six-step diagnostic procedure, applied to the lagging expanded run
#    Canvas 900 x 380 for a full-width slot: 18 px = 8.2 pt.
# =============================================================================
g = Svg(900, 285)
steps = [
    ("1 Symptom", "run falls\nbehind\nschedule", SLATE, SLATE_L, 1),
    ("2 Metric", "MFU per\nstep has\ndropped", SLATE, SLATE_L, 1),
    ("3 C³ term", "per-rank\ntraces: more\nexposed\ntransfer", AMBER, AMBER_L, 2),
    ("4 Row", "Distributed\nExecution", AMBER, AMBER_L, 2),
    ("5 Cost\nmoved", "compression\nmoves risk\nto quality", RED, RED_L, 3),
    ("6 Evidence", "convergence\nand eval\nrecord", GREEN, GREEN_L, 3),
]
for i, (head, body, stroke, fill, step) in enumerate(steps):
    x = 4 + i * 150
    g.rect(x, 4, 140, 58, stroke, stroke, first=step, sw=1, rx=6)
    g.text(x + 70, 33, head, 19, "#ffffff", "bold", first=step)
    g.rect(x, 70, 140, 150, fill, stroke, first=step, sw=2, rx=6)
    g.text(x + 70, 145, body, 20, TEXT, "normal", first=step)
    if i < 5:
        g.line(x + 141, 33, x + 149, 33, SLATE, sw=2.5, first=steps[i + 1][4])
g.text(450, 262, "Barrier wait alone cannot tell slow transfer from late arrival.", 21,
       AMBER, "bold", first=2)
g.write("diagnose", 3)

# =============================================================================
# 9b. The next question: an agent's loop (closing slide, no numbers)
#     Canvas 520 x 420 for a right-hand column: 22 px = about 10 pt.
# =============================================================================
g = Svg(520, 420)
nodes = [(260, 55, "plan"), (430, 210, "call a tool"), (260, 365, "observe"), (90, 210, "act")]
for x, y, lab in nodes:
    g.rect(x - 88, y - 38, 176, 76, ETH_L, ETH, sw=2.5, rx=10)
    g.text(x, y + 11, lab, 30, ETH, "bold")
for (x1, y1), (x2, y2) in [((350, 70), (425, 166)), ((425, 254), (350, 350)),
                           ((170, 350), (95, 254)), ((95, 166), (170, 70))]:
    g.line(x1, y1, x2, y2, ETH, sw=3)
g.text(260, 200, "one task,", 28, SLATE)
g.text(260, 238, "many steps", 28, SLATE)
g.write("agent_loop", 1)

# =============================================================================
# 10. Where next: MoE resident against active bytes (Chapter 10, Sec. 10.10.4)
# =============================================================================
moe_rows = [
    ("MoE, all experts resident", num("moe_resident_gb"), "memory", f"{val('moe_resident_gb')} GB"),
    (f"Dense {val('dense_b')}B, read per step", num("dense_gb"), "neutral", f"{val('dense_gb')} GB"),
    ("MoE, active per token", num("moe_active_gb"), "compute", f"{val('moe_active_gb')} GB"),
]
sp.hbars(out / "moe_bytes.svg", moe_rows, "", log=False, size=(5.8, 1.9))

print("figures written")
