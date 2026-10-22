#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 3 (Network Fabrics), Volume II.

Every number a figure prints or plots comes from calcs.py, which reads it from
MLSysIM (registry values and Scenarios.*); the list below names the source of
each figure. Derived and illustrative values say so here and on the slide.
  - Staircase recall: 10 GB at HBM 3,350 GB/s, NVLink 450 GB/s, InfiniBand
    50 GB/s: 3.0, 22.2, 200 ms (Compute Infrastructure, Napkin Math 2.3).
  - Interconnect generations: MLSysIM vol2/03_network_fabrics/data/interconnect_generations.csv
    (IEEE 802.3, IBTA, NVIDIA specs), per direction; NVLink is the sum of one
    GPU's links. H100 NVLink 3,600 Gb/s = 450 GB/s; NDR 400 Gb/s = 50 GB/s.
  - Step timeline: illustrative 100 + 100 ms compute with a 1 ms or 30 ms
    AllReduce (Network Fabrics, figure 2); 30 / 230 = 13 percent.
  - Flow timelines (web against training): illustrative shapes, labeled.
  - ECMP: 8 flows on 4 equal-cost paths; even 2 per path, hashed 4/2/2/0
    (Network Fabrics, Sec. 3.6.4). Drain time in units of one flow alone.
  - Alpha-beta crossover: NDR alpha 1.5 us, beta 50 GB/s (n* = 75 KB); RoCE
    100G alpha 5 us, beta 12.5 GB/s (n* = 62.5 KB) (Napkin Math 3.1, fig. 7).
  - NCCL sweep: MLSysIM vol2/03_network_fabrics/data/nccl_allreduce_h100_node.csv (NVIDIA
    exemplar-performance all_reduce_perf output, 8x H100 over NVLink). Model:
    floor = median time for sizes up to 64 KB (17.0 us); bandwidth = algbw at
    16 GB (274 GB/s); knee 4.7 MB; worst ratio 1.74 at 8 MB (8 to 128 MB).
  - Bisection: 512 links x 50 GB/s = 25.6 TB/s; 4:1 gives 6.4 TB/s (Def. 3.4).
  - Fat-tree: radix-4, three tiers, 16 hosts, 4 equal-cost paths (figure 9).
  - Oversubscription: 5,115 GB crossing; 0.200 s against 0.799 s; throughput
    1 / (0.70 + 0.30 x 4) = 0.53 (Napkin Math 3.2; 30 percent is a scenario
    input).
  - Rails: 8 rails of 128 GPUs on 128 nodes (Sec. 3.5.3, figure 10).
  - Torus against fat-tree at 4,096 endpoints: 512 against 2,048 cut links,
    25.6 against 102.4 TB/s, 24 hops against 5 switch traversals (Sec. 3.5.4).
  - PFC spreading and the deadlock cycle: Sec. 3.6.1 and 3.6.2, figures 12.
  - Incast: 256 senders x 50 GB/s = 12.8 TB/s into one 50 GB/s egress
    (Def. 3.10); 32 MB fills in 2.5 us, drains in 640 us (derived).
  - Silent waste: 1,024 GPUs x 720 h = 737,280 GPU-hours; 2 and 3 percent
    give 14,746 and 22,118 GPU-hours (Sec. 3.9, scenario; derived).
  - Diagnosis order: Sec. 3.9.3.
  - SR-IOV: 8 VFs x 50 Gb/s on a 400 Gb/s NIC; 100 GB at 50 GB/s = 2 s
    (Sec. 3.8).

Run from this folder: python3 make_figs.py
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402  (records every value; figures read them)
from mlsysim.core.series_loader import load_series  # noqa: E402
from deckvals import val, num  # noqa: E402
from mlsysim import Literature  # noqa: E402
from mlsysim.core.units import Q_  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Volume II roles (palette.md): blue compute, amber communication, violet
# coordination, green healthy data flow, red faults, slate frame.
V = {
    "comp": "#1F407A", "comp_l": "#EAF1F8", "comp_t": "#14294E",
    "comm": "#d97706", "comm_l": "#fef3c7", "comm_t": "#b45309",
    "coord": "#7c3aed", "coord_l": "#f3e8ff", "coord_t": "#6d28d9",
    "ok": "#16a34a", "ok_l": "#dcfce7", "ok_t": "#15803d",
    "bad": "#dc2626", "bad_l": "#fee2e2", "bad_t": "#b91c1c",
    "frame": "#94a3b8", "frame_l": "#f8fafc", "ink": "#0f172a",
    "gray": "#64748b", "light": "#cbd5e1",
}


def label(ax, x, y, text, color, shown=True, **kw):
    """A label drawn in every step (invisible until shown), so every step has
    the same bounding box and nothing moves on the slide."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


# --- SVG helpers -------------------------------------------------------------
FONT = "Helvetica, Arial, sans-serif"
FS = 1.3  # every SVG label is drawn 30 percent larger than its nominal size


def svg_doc(width, height, body):
    defs = "".join(
        f'<marker id="arr-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{V[k]}"/></marker>'
        for k in ("comp", "comm", "coord", "ok", "bad", "gray", "ink"))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="{FONT}">'
            f'<defs>{defs}</defs><rect width="{width}" height="{height}" fill="white"/>'
            f'{body}</svg>')


def text(x, y, s, size=22, color=None, anchor="middle", weight="normal", opacity=1.0, italic=False):
    color = color or V["ink"]
    style = ' font-style="italic"' if italic else ""
    size = round(size * FS, 1)
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" opacity="{opacity}"{style} dominant-baseline="middle">{s}</text>')


def box(x, y, w, h, fill, stroke, width=2, radius=6, opacity=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{d}/>')


def line(x1, y1, x2, y2, color, width=2.5, arrow=None, dash=None, opacity=1.0, both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#arr-{arrow})"' if arrow else ""
    if arrow and both:
        m += f' marker-start="url(#arr-{arrow})"'
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
            f'opacity="{opacity}"{d}{m}/>')


def path(d, color, width=2.5, arrow=None, dash=None, opacity=1.0):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#arr-{arrow})"' if arrow else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'opacity="{opacity}"{ds}{m}/>')


def write(name, width, height, body):
    (out / name).write_text(svg_doc(width, height, body))


# =============================================================================
# 1. Recall: the staircase from Lecture 2
# =============================================================================
sp.hbars(out / "staircase_recall.svg", [
    ("HBM", num("stair_hbm_ms"), "memory", f"{val('stair_hbm_ms')} ms"),
    ("NVLink", num("stair_nvlink_ms"), "network", f"{val('stair_nvlink_ms')} ms"),
    ("InfiniBand", num("stair_ib_ms"), "compute", f"{val('stair_ib_ms')} ms"),
], "ms", log=True, xlabel=f"Time to move {val('buffer_gb')} GB one hop (log scale)", size=(4.8, 1.5))

# =============================================================================
# 2. Interconnect generations (MLSysIM, per direction, GB/s)
# =============================================================================
rows = load_series("vol2/03_network_fabrics/data/interconnect_generations.csv").entries
best = {}
for r in rows:
    if r["first_product_year"] is None or r["family"] == "PCIe":
        continue
    key = (r["family"], int(r["first_product_year"]))
    best[key] = max(best.get(key, 0.0), Q_(r["line_rate_gbps"], "Gbps").to("GB/s").magnitude)
fig, ax = plt.subplots(figsize=(4.2, 2.9))
styles = {"NVLink": (V["coord"], "-", "o"), "InfiniBand": (V["comm"], "--", "o"),
          "Ethernet": (V["gray"], ":", "D")}
for fam, (col, ls, mk) in styles.items():
    pts = sorted((y, v) for (f, y), v in best.items() if f == fam)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, ls=ls, marker=mk, lw=2.6, ms=8,
            mfc="white" if fam == "InfiniBand" else col, mew=2)
ax.text(2009.8, 2600, "NVLink (one GPU)", color=V["coord"], fontsize=15, fontweight="bold")
ax.text(2016.3, 6, "InfiniBand port", color=V["comm"], fontsize=15, fontweight="bold")
ax.text(2009.8, 140, "Ethernet port", color=V["gray"], fontsize=15, fontweight="bold")
ax.annotate("", xy=(2022, num("ib_gbs")), xytext=(2022, num("nvlink_gbs")),
            arrowprops=dict(arrowstyle="<->", color=V["bad"], lw=2.2))
ax.text(2022.35, 150, f"{val('nvlink_ib_ratio')}×", color=V["bad"], fontsize=18, fontweight="bold", va="center")
ax.text(2021.6, 1300, f"H100: {val('nvlink_gbs')}", color=V["coord"], fontsize=14, fontweight="bold", ha="right")
ax.text(2022.3, 18, f"NDR: {val('ib_gbs')}", color=V["comm"], fontsize=14, fontweight="bold", ha="left")
ax.set_yscale("log")
ax.set_ylim(3, 5000)
ax.set_xlim(2009.5, 2025.8)
ax.set_xticks([2010, 2014, 2018, 2022])
ax.set_yticks([5, 50, 500])
ax.set_yticklabels(["5", "50", "500"])
ax.minorticks_off()
ax.tick_params(labelsize=14)
ax.set_ylabel("GB/s per direction", fontsize=15)
ax.grid(axis="y", color=C["grid"], lw=0.6)
sp.save(fig, out / "bw_generations.svg")

# =============================================================================
# 3. Exposed AllReduce in an illustrative step (book figure 2), two steps
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.8, 1.65))
    fwd, bwd = num("step_fwd_ms"), num("step_bwd_ms")
    for row, (name, key, shown) in enumerate([("Inter-node\n64 GPUs", "ar_inter_ms", k >= 2),
                                              ("Intra-node\n8 GPUs", "ar_intra_ms", True)]):
        a = 1.0 if shown else 0.0
        ar = num(key)
        ax.barh(row, fwd, left=0, color=V["comp_l"], edgecolor=V["comp"], lw=1.5, alpha=a, height=0.6)
        ax.barh(row, bwd, left=fwd, color=V["comp_l"], edgecolor=V["comp"], lw=1.5, alpha=a, height=0.6)
        ax.barh(row, ar, left=fwd + bwd, color=V["bad"], alpha=a, height=0.6)
        label(ax, fwd / 2, row, "forward", V["comp_t"], shown, ha="center", va="center", fontsize=12)
        label(ax, fwd + bwd / 2, row, "backward", V["comp_t"], shown, ha="center", va="center", fontsize=12)
        label(ax, fwd + bwd + 3 + ar, row, f"AllReduce {val(key)} ms", V["bad"], shown, va="center", fontsize=12)
        label(ax, -4, row, name, V["ink"], shown, ha="right", va="center", fontsize=12)
    label(ax, 115, -0.95, f"{val('ar_inter_ms')} of {val('step_inter_ms')} ms exposed: "
          f"{val('exposed_pct')}% of every step", V["bad"], k >= 2, ha="center", fontsize=13)
    ax.set_xlim(0, 300)
    ax.set_ylim(-1.3, 1.5)
    ax.set_yticks([])
    ax.set_xticks([0, fwd, fwd + bwd, num("step_inter_ms")])
    ax.set_xlabel("Time in one step (ms), illustrative")
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / f"step_timeline_step{k}.svg")

# =============================================================================
# 4. Flow timelines: web service against synchronous training (illustrative)
# =============================================================================
rng = random.Random(7)
web = []
for lane in range(6):
    t = rng.uniform(0, 4)
    while t < 96:
        dur = rng.uniform(1.5, 4.5)
        web.append((lane, t, dur))
        t += dur + rng.uniform(3, 14)
train_ends = [38, 40, 41, 42, 43, 44, 45, 58]
for k in (1, 2, 3):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(4.8, 2.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.45))
    for lane, t, dur in web:
        top.barh(lane, dur, left=t, color=V["gray"], height=0.6)
    top.set_title("Web service: many short flows, each on its own schedule",
                  fontsize=13, loc="left", color=V["ink"], fontweight="bold")
    shown = k >= 2
    a = 1.0 if shown else 0.0
    for start in (0, 62):
        for lane, end in enumerate(train_ends):
            bot.barh(lane, end - 6 if start == 0 else end - 8, left=start + 2,
                     color=V["comm"] if lane < 7 else V["bad"], height=0.62, alpha=a)
    bot.set_title("Synchronous training: a few long flows, released together",
                  fontsize=13, loc="left", color=V["ink"], fontweight="bold", alpha=a)
    bshow = k >= 3
    bot.axvline(54, color=V["coord"], lw=2.5, ls="--", alpha=1.0 if bshow else 0.0)
    for ax in (top, bot):
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 100)
    top.set_ylim(-0.7, 5.7)
    bot.set_ylim(-0.7, 7.7)
    bot.set_xticks([])
    bot.set_xlabel("Time (illustrative)")
    sp.save(fig, out / f"inversion_step{k}.svg")

# =============================================================================
# 5. ECMP: 8 flows on 4 equal-cost paths
# =============================================================================
flow_colors = ["#1F407A", "#d97706", "#7c3aed", "#16a34a", "#0e7490", "#be185d", "#a16207", "#475569"]
even = [round(calcs.ECMP.even_load)] * calcs.ECMP.paths
layouts = {1: (even, "Even split", V["ok_t"]),
           2: (list(calcs.ECMP.path_loads), "Static hash", V["bad_t"]),
           3: (even, "Adaptive routing", V["ok_t"])}
for k, (loads, caption, ccol) in layouts.items():
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    fid = 0
    for p, n in enumerate(loads):
        y = 3 - p
        for j in range(n):
            ax.barh(y, 1, left=j, color=flow_colors[fid % 8], edgecolor="white", lw=2, height=0.62)
            fid += 1
        ax.text(-0.12, y, f"path {p + 1}", ha="right", va="center", fontsize=13,
                fontweight="bold", color=V["ink"])
        if n == 0:
            ax.text(0.1, y, "idle", va="center", fontsize=13, color=V["gray"], fontweight="bold")
    ax.axvline(max(loads), color=V["bad"] if max(loads) > max(even) else V["ok"], lw=2, ls="--")
    ax.text(5.0, 1.5, caption + ":", ha="center", va="center",
            fontsize=13, fontweight="bold", color=ccol)
    ax.text(5.0, 0.9, "done at " + str(max(loads)), ha="center", va="center",
            fontsize=13, fontweight="bold", color=ccol)
    ax.set_xlim(0, 5.9)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Drain time (one flow alone on a path = 1)")
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / f"ecmp_step{k}.svg")

# =============================================================================
# 6. Kernel TCP path against RDMA verbs with GPUDirect (book figure 6, redrawn)
# =============================================================================
W, H = 900, 400


def rdma_path(k):
    b = []
    for y, name in ((69, "user space"), (178, "kernel"), (289, "hardware")):
        b.append(text(10, y, name, 17, V["gray"], "start", italic=True))
    b.append(line(130, 118, 890, 118, V["light"], 1.5, dash="6 5"))
    b.append(line(130, 238, 890, 238, V["light"], 1.5, dash="6 5"))
    b.append(text(290, 20, "Kernel TCP sockets", 21, V["ink"], weight="bold"))
    b.append(box(150, 42, 280, 54, V["comp_l"], V["comp"]))
    b.append(text(290, 69, "training process", 19, V["comp_t"]))
    b.append(box(150, 136, 280, 84, V["frame_l"], V["frame"]))
    b.append(text(290, 160, "socket + TCP/IP", 19))
    b.append(text(290, 195, "CPU runs transport", 17, V["gray"]))
    b.append(box(150, 262, 280, 54, V["comm_l"], V["comm"]))
    b.append(text(290, 289, "Ethernet NIC", 19, V["comm_t"]))
    b.append(line(290, 96, 290, 134, V["ink"], 2.5, "ink"))
    b.append(line(290, 220, 290, 260, V["ink"], 2.5, "ink"))
    b.append(text(305, 116, "copy 1", 17, V["bad_t"], "start", "bold"))
    b.append(text(305, 240, "copy 2", 17, V["bad_t"], "start", "bold"))
    b.append(text(290, 360, "two copies via host memory", 19, V["bad_t"], weight="bold"))
    a = 1.0 if k >= 2 else 0.0
    r = []
    r.append(text(690, 20, "RDMA verbs + GPUDirect", 21, V["ink"], weight="bold"))
    r.append(box(560, 42, 280, 54, V["comp_l"], V["comp"]))
    r.append(text(700, 69, "NCCL + verbs API", 19, V["comp_t"]))
    r.append(text(760, 178, "kernel not on data path", 17, V["gray"], "end", italic=True))
    r.append(box(480, 262, 180, 54, V["comp_l"], V["comp"]))
    r.append(text(570, 289, "GPU memory", 19, V["comp_t"]))
    r.append(box(700, 252, 195, 74, V["comm_l"], V["comm"]))
    r.append(text(797, 276, "RDMA NIC", 19, V["comm_t"], weight="bold"))
    r.append(text(797, 304, "hardware transport", 14, V["comm_t"]))
    r.append(line(800, 96, 800, 250, V["comm"], 3.5, "comm"))
    r.append(line(662, 289, 698, 289, V["ok"], 3.5, "ok"))
    r.append(text(690, 360, "NIC reads GPU memory", 19, V["ok_t"], weight="bold"))
    b.append(f'<g opacity="{a}">' + "".join(r) + "</g>")
    write(f"rdma_path_step{k}.svg", 900, 385, "".join(b))


for k in (1, 2):
    rdma_path(k)

# =============================================================================
# 7. Where InfiniBand and RoCE keep buffers from overflowing (column)
# =============================================================================
b = []
# left panel: InfiniBand credits
b.append(text(10, 22, "InfiniBand: credits", 22, V["comm_t"], "start", "bold"))
b.append(box(10, 70, 120, 60, V["comp_l"], V["comp"]))
b.append(text(70, 100, "sender", 19, V["comp_t"]))
b.append(line(132, 86, 300, 86, V["comm"], 3, "comm"))
b.append(text(216, 66, "data", 16, V["comm_t"]))
for i in range(4):
    b.append(box(304 + i * 40, 70, 36, 60, V["ok_l"] if i < 3 else V["comm_l"],
                 V["ok"] if i < 3 else V["comm"], 2, 3))
b.append(text(382, 150, "receiver buffer", 16, V["gray"]))
b.append(line(300, 116, 134, 116, V["ok"], 3, "ok", dash="7 5"))
b.append(text(216, 150, "3 credits free", 16, V["ok_t"], weight="bold"))
b.append(text(10, 205, "The sender waits for credits;", 18, V["ink"], "start"))
b.append(text(10, 240, "no buffer overflows.", 18, V["ink"], "start"))
b.append(line(500, 10, 500, 300, V["light"], 1.5))
# right panel: RoCE pause + ECN
x0 = 530
b.append(text(x0, 22, "RoCE: PFC pause + ECN", 22, V["comm_t"], "start", "bold"))
b.append(box(x0, 70, 120, 60, V["comp_l"], V["comp"]))
b.append(text(x0 + 60, 100, "sender", 19, V["comp_t"]))
b.append(line(x0 + 122, 86, x0 + 274, 86, V["comm"], 3, "comm"))
for i in range(5):
    b.append(box(x0 + 278 + i * 34, 70, 30, 60, V["bad_l"] if i < 4 else "white",
                 V["bad"] if i < 4 else V["frame"], 2, 3))
b.append(line(x0 + 398, 62, x0 + 398, 138, V["bad"], 2.5, dash="5 4"))
b.append(text(x0 + 330, 150, "switch queue", 16, V["gray"]))
b.append(line(x0 + 274, 116, x0 + 124, 116, V["bad"], 3, "bad", dash="7 5"))
b.append(text(x0 + 200, 150, "PAUSE", 16, V["bad_t"], weight="bold"))
b.append(text(x0 + 398, 52, "threshold", 15, V["bad_t"]))
b.append(text(x0, 205, "ECN marks slow senders first;", 18, V["ink"], "start"))
b.append(text(x0, 240, "PAUSE stops the upstream port.", 18, V["ink"], "start"))
write("ib_roce.svg", 1000, 265, "".join(b))

# =============================================================================
# 8. Alpha-beta crossover, two modeled paths (book figure 7, redrawn)
# =============================================================================
sizes = np.logspace(1, 9.3, 400)  # bytes, decimal
fig, ax = plt.subplots(figsize=(4.8, 2.25))
AB = calcs.AB
paths = [(AB.alphas[p].to("microsecond").magnitude, AB.bandwidths[p].to("GB/s").magnitude, col)
         for p, col in (("NDR", V["comm"]), ("RoCE 100G", V["gray"]))]
for alpha_us, beta_gb, col in paths:
    t_us = alpha_us + sizes / (beta_gb * 1e9) * 1e6
    ax.plot(sizes, t_us, color=col, lw=2.8)
    knee = alpha_us * 1e-6 * beta_gb * 1e9
    ax.plot([knee], [2 * alpha_us], "o", ms=9, color=col, mec="white", mew=1.5, zorder=5)
ax.text(15, 1.0e5, f"RoCE, {val('roce_gbps')} Gb/s: α {val('roce_alpha_us')} μs, β {val('roce_beta_gbs')} GB/s",
        color=V["gray"], fontsize=11.5, fontweight="bold")
ax.text(15, 2.0e4, f"NDR: α {val('ndr_alpha_us')} μs, β {val('ib_gbs')} GB/s",
        color=V["comm"], fontsize=11.5, fontweight="bold")
ctrl_b, shard_b = (AB.messages[k].to("byte").magnitude for k in ("control", "shard"))
ctrl_t, shard_t = num("ctrl_ndr_us"), num("shard_ndr_ms") * 10 ** 3
ax.plot([ctrl_b, shard_b], [ctrl_t, shard_t], "s", ms=8, color=V["coord"], zorder=6)
ax.annotate(f"{val('ctrl_kb')} KB control: {val('ctrl_ndr_us')} μs", (ctrl_b, ctrl_t), xytext=(60, 0.35), color=V["coord"],
            fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-", color=V["coord"], lw=1.2))
ax.annotate(f"{val('shard_mb')} MB shard: {val('shard_ndr_ms')} ms", (shard_b, shard_t), xytext=(2e6, 2.5), color=V["coord"],
            fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-", color=V["coord"], lw=1.2))
ax.text(15, 150, "latency bound", color=V["ink"], fontsize=12, fontstyle="italic")
ax.text(1.5e9, 0.25, "bandwidth bound", color=V["ink"], fontsize=12, fontstyle="italic", ha="right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 2e9)
ax.set_ylim(0.15, 6e5)
ax.set_xticks([1e2, 1e4, 1e6, 1e8])
ax.set_xticklabels(["100 B", "10 KB", "1 MB", "100 MB"])
ax.set_yticks([1, 100, 1e4])
ax.set_yticklabels(["1 μs", "100 μs", "10 ms"])
ax.minorticks_off()
ax.set_xlabel("Message size $n$ (modeled)")
ax.set_ylabel("$T(n)$")
sp.save(fig, out / "alphabeta_crossover.svg")

# =============================================================================
# 9. Measured NCCL AllReduce against the two-parameter model (column)
# =============================================================================
nccl = calcs.NCCL.rows
fit = calcs.NCCL.result
floor_us = fit.floor_time.to("microsecond").magnitude
mid_lo, mid_hi = (q.to("byte").magnitude for q in (calcs.NCCL.mid_low, calcs.NCCL.mid_high))
xs = np.array([r["size_bytes"] for r in nccl])
ts = np.array([r["oop_time_us"] for r in nccl])
grid = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 300)
fig, ax = plt.subplots(figsize=(4.2, 2.9))
ax.axvspan(mid_lo, mid_hi, color=V["bad_l"], alpha=0.8, lw=0)
ax.plot(grid, [fit.model_time(Q_(g, "byte")).magnitude for g in grid], color=V["comp"], lw=2.6, zorder=2)
ax.scatter(xs, ts, s=40, color=V["bad"], zorder=3)
ax.text(30, 2.8, f"floor ≈ {val('nccl_floor_us')} μs", color=V["comp"], fontsize=15, fontweight="bold")
ax.text(16, 4e4, f"model: floor + n / {val('nccl_bw_gbs')} GB/s", color=V["comp"], fontsize=13,
        fontweight="bold", ha="left")
ax.text(16, 250, f"shaded, {val('nccl_mid_lo_mb')} to {val('nccl_mid_hi_mb')} MB:\nup to {val('nccl_mid_ratio')}× the model", color=V["bad_t"], fontsize=14,
        fontweight="bold", ha="left")
ax.text(16, 1.0e4, "red: measured", color=V["bad"], fontsize=15, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([8, 2**10, 2**20, 2**30])
ax.set_xticklabels(["8 B", "1 KB", "1 MB", "1 GB"])
ax.set_yticks([10, 100, 1e3, 1e4])
ax.set_yticklabels(["10", "100", "1,000", "10,000"])
ax.minorticks_off()
ax.set_ylim(1.5, 1.5e5)
ax.tick_params(labelsize=14)
ax.set_xlabel("Message size", fontsize=15)
ax.set_ylabel("AllReduce time (μs)", fontsize=15)
sp.save(fig, out / "nccl_measured.svg")

# =============================================================================
# 10. Bisection: 1:1 against 4:1
# =============================================================================


def bisection(k):
    b = []
    b.append(box(40, 70, 290, 230, V["comp_l"], V["comp"], 2.5, 10))
    b.append(box(570, 70, 290, 230, V["comp_l"], V["comp"], 2.5, 10))
    b.append(text(185, 165, f"{val('cut_half')} GPUs", 28, V["comp_t"], weight="bold"))
    b.append(text(185, 205, "half A", 22, V["comp_t"]))
    b.append(text(715, 165, f"{val('cut_half')} GPUs", 28, V["comp_t"], weight="bold"))
    b.append(text(715, 205, "half B", 22, V["comp_t"]))
    b.append(line(450, 40, 450, 330, V["bad"], 3, dash="10 7"))
    b.append(text(450, 22, "the cut", 22, V["bad_t"], weight="bold"))
    for i in range(8):
        y = 92 + i * 26
        live = k == 1 or i % 4 == 0
        b.append(line(332, y, 568, y, V["comm"], 4 if live else 2.5, opacity=1.0 if live else 0.22))
    if k == 1:
        b.append(text(450, 362, f"1:1 · {val('cut_half')} links × {val('ib_gbs')} GB/s = {val('cut_full_tbs')} TB/s each way",
                      24, V["comm_t"], weight="bold"))
    else:
        b.append(text(450, 362, f"{val('oversub')}:1 spine · {val('cut_full_tbs')} ÷ {val('oversub')} = "
                      f"{val('cut_over_tbs')} TB/s each way", 24, V["bad_t"], weight="bold"))
    write(f"bisection_step{k}.svg", 900, 385, "".join(b))


for k in (1, 2):
    bisection(k)

# =============================================================================
# 11. Fat-tree, radix 4, three tiers (book figure 9, redrawn)
# =============================================================================


def fattree(k):
    b = []
    core_x = [255, 385, 515, 645]
    pods = [(60 + p * 210) for p in range(4)]
    spine_y, leaf_y, core_y, gpu_y = 140, 235, 45, 320
    # pod outlines
    for px in pods:
        b.append(box(px - 8, 110, 176, 250, "none", V["frame"], 1.5, 8, dash="6 5"))
    # links leaf-spine, spine-core, gpu-leaf
    links = []
    for p, px in enumerate(pods):
        sx = [px + 38, px + 122]
        lx = [px + 38, px + 122]
        for s in sx:
            for leaf in lx:
                links.append((s, spine_y + 16, leaf, leaf_y - 16))
        for i, s in enumerate(sx):
            for c in core_x[2 * i:2 * i + 2]:
                links.append((s, spine_y - 16, c, core_y + 16))
        for i, leaf in enumerate(lx):
            for g in (leaf - 20, leaf + 20):
                links.append((leaf, leaf_y + 16, g, gpu_y - 14))
    for x1, y1, x2, y2 in links:
        b.append(line(x1, y1, x2, y2, V["light"], 1.6))
    # highlighted equal-cost paths: GPU at pod 1 leaf 0 -> pod 4 leaf 1
    if k >= 2:
        src = (pods[0] + 18, gpu_y - 14)
        dst = (pods[3] + 142, gpu_y - 14)
        l1 = pods[0] + 38
        l4 = pods[3] + 122
        for si, s1 in enumerate([pods[0] + 38, pods[0] + 122]):
            s4 = [pods[3] + 38, pods[3] + 122][si]
            for c in core_x[2 * si:2 * si + 2]:
                pts = [src, (l1, leaf_y + 16), (l1, leaf_y - 16), (s1, spine_y + 16), (s1, spine_y - 16),
                       (c, core_y + 16), (c, core_y + 16), (s4, spine_y - 16), (s4, spine_y + 16),
                       (l4, leaf_y - 16), (l4, leaf_y + 16), dst]
                d = "M" + " L".join(f"{x},{y}" for x, y in pts)
                main = (si == 0 and c == core_x[0])
                b.append(path(d, V["comm"], 4.5 if main else 2.6, opacity=1.0 if main else 0.6))
    # switches
    for c in core_x:
        b.append(box(c - 34, core_y - 16, 68, 32, V["comm_l"], V["comm"], 2, 4))
    for px in pods:
        for s in (px + 38, px + 122):
            b.append(box(s - 34, spine_y - 16, 68, 32, V["comm_l"], V["comm"], 2, 4))
            b.append(box(s - 34, leaf_y - 16, 68, 32, V["comm_l"], V["comm"], 2, 4))
        for g in (px + 18, px + 58, px + 102, px + 142):
            hot = k >= 2 and (g in (pods[0] + 18, pods[3] + 142))
            b.append(box(g - 15, gpu_y - 14, 30, 28, V["comp"] if hot else V["comp_l"], V["comp"], 2, 3))
    for i, px in enumerate(pods):
        b.append(text(px + 80, 380, f"pod {i + 1}", 24, V["gray"]))
    b.append(text(880, core_y, "core", 26, V["comm_t"], "start", "bold"))
    b.append(text(880, spine_y, "spine", 26, V["comm_t"], "start", "bold"))
    b.append(text(880, leaf_y, "leaf", 26, V["comm_t"], "start", "bold"))
    b.append(text(880, gpu_y, "GPUs", 26, V["comp_t"], "start", "bold"))
    write(f"fattree_step{k}.svg", 1010, 395, "".join(b))


for k in (1, 2):
    fattree(k)

# =============================================================================
# 12. Oversubscription: AllReduce bound and relative throughput (column)
# =============================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.2, 2.8), gridspec_kw=dict(wspace=0.55))
times = (("os_t_full_s", num("os_t_full_s")), ("os_t_over_s", num("os_t_over_s")))
a1.bar([0, 1], [t for _, t in times], color=[V["comm"], V["bad"]], width=0.6)
for x, (key, v) in enumerate(times):
    a1.text(x, v + 0.03, f"{val(key)} s", ha="center", fontsize=15, fontweight="bold",
            color=V["comm_t"] if x == 0 else V["bad_t"])
a1.set_ylim(0, 1.0)
a1.set_title("Crossing time", fontsize=15, fontweight="bold")
a2.bar([0, 1], [1.0, num("os_rel")], color=[V["comm"], V["bad"]], width=0.6)
for x, (v, s) in enumerate(((1.0, "1.00"), (num("os_rel"), val("os_rel")))):
    a2.text(x, v + 0.03, s, ha="center", fontsize=15, fontweight="bold",
            color=V["comm_t"] if x == 0 else V["bad_t"])
a2.set_ylim(0, 1.25)
a2.set_title("Throughput", fontsize=15, fontweight="bold")
for ax in (a1, a2):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1:1", f"{val('oversub')}:1"], fontsize=15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
sp.save(fig, out / "oversub_bars.svg")

# =============================================================================
# 13. Rail-optimized wiring (book figure 10, redrawn), three steps
# =============================================================================


def rails(k):
    b = []
    rail_x = [180, 360, 540, 720]
    node_x = [60, 330, 600]
    sw_y, node_y = 70, 230
    if k >= 3:
        b.append(box(180, 2, 540, 36, V["coord_l"], V["coord"], 2, 4, dash="6 4"))
        b.append(text(450, 20, "a spine bridges rails for cross-rank traffic", 17, V["coord_t"], weight="bold"))
        for x in rail_x:
            b.append(line(x, sw_y - 18, x if x in (360, 540) else (x + 90 if x < 450 else x - 90), 38,
                          V["coord"], 2, dash="5 4"))
    for n, nx in enumerate(node_x):
        for g in range(4):
            gx = nx + 30 + g * 55
            col = V["comm"] if g == 0 and k >= 2 else V["light"]
            width = 4 if g == 0 and k >= 2 else 2
            b.append(line(gx + 20, node_y, rail_x[g], sw_y + 18, col, width))
    for r, x in enumerate(rail_x):
        hot = r == 0 and k >= 2
        b.append(box(x - 82, sw_y - 18, 164, 36, V["comm"] if hot else V["comm_l"], V["comm"], 2, 4))
        b.append(text(x, sw_y, f"rail {r} switch", 18, "white" if hot else V["comm_t"], weight="bold"))
    for n, nx in enumerate(node_x):
        b.append(box(nx, node_y, 250, 120, V["frame_l"], V["frame"], 2, 8))
        for g in range(4):
            gx = nx + 30 + g * 55
            hot = g == 0 and k >= 2
            b.append(box(gx, node_y + 16, 42, 40, V["comp"] if hot else V["comp_l"], V["comp"], 2, 3))
            b.append(text(gx + 21, node_y + 36, f"{g}", 20, "white" if hot else V["comp_t"], weight="bold"))
        b.append(box(nx + 20, node_y + 70, 210, 22, V["coord_l"], V["coord"], 1.5, 3))
        b.append(text(nx + 125, node_y + 81, "NVSwitch", 16, V["coord_t"], weight="bold"))
        b.append(text(nx + 125, node_y + 108, f"node {n}", 18, V["gray"]))
    b.append(text(450, 378, "GPU 0 of every node shares rail 0; GPU 1 shares rail 1; ..." if k >= 2
                  else "Each GPU's port is cabled by its slot number", 20, V["ink"], weight="bold"))
    write(f"rails_step{k}.svg", 900, 395, "".join(b))


for k in (1, 2, 3):
    rails(k)

# =============================================================================
# 14. Torus against fat-tree at 4,096 endpoints (column)
# =============================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.2, 2.8), gridspec_kw=dict(wspace=0.6))
a1.bar([0, 1], [num("ft_cut_tbs"), num("torus_cut_tbs")], color=[V["comm"], V["coord"]], width=0.6)
a1.text(0, num("ft_cut_tbs") + 3.6, val("ft_cut_tbs"), ha="center", fontsize=15, fontweight="bold", color=V["comm_t"])
a1.text(1, num("torus_cut_tbs") + 4.4, val("torus_cut_tbs"), ha="center", fontsize=15, fontweight="bold", color=V["coord_t"])
a1.set_ylim(0, 125)
a1.set_title("Cut, TB/s", fontsize=14, fontweight="bold")
a2.bar([0, 1], [num("hops_core"), num("torus_hops")], color=[V["comm"], V["coord"]], width=0.6)
a2.text(0, num("hops_core") + 0.8, val("hops_core"), ha="center", fontsize=15, fontweight="bold", color=V["comm_t"])
a2.text(1, num("torus_hops") + 0.8, val("torus_hops"), ha="center", fontsize=15, fontweight="bold", color=V["coord_t"])
a2.set_ylim(0, 29)
a2.set_title("Worst path, hops", fontsize=14, fontweight="bold")
for ax in (a1, a2):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fat-tree", "torus"], fontsize=14)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
sp.save(fig, out / "torus_fattree.svg")

# =============================================================================
# 15. PFC pause spreading (book margin figure, redrawn), three steps
# =============================================================================


def pfc(k):
    b = []
    senders = [(80, 55 + i * 58) for i in range(6)]
    switches = [(430, 85), (430, 200), (430, 315)]
    full = (790, 200)
    other = (790, 345)
    b.append(text(80, 22, "senders", 21, V["gray"], weight="bold"))
    b.append(text(430, 22, "switches", 21, V["gray"], weight="bold"))
    b.append(text(790, 22, "receivers", 21, V["gray"], weight="bold"))
    # data paths (gray)
    for i, (sx, sy) in enumerate(senders):
        tx, ty = switches[i // 2]
        b.append(line(sx + 22, sy, tx - 34, ty, V["light"], 2))
    for tx, ty in switches:
        b.append(line(tx + 34, ty, full[0] - 50, full[1], V["light"], 2))
    # victim flow: sender 6 -> switch 3 -> other receiver
    vic = V["bad"] if k >= 3 else V["ok"]
    b.append(line(senders[5][0] + 22, senders[5][1], switches[2][0] - 34, switches[2][1] + 8, vic, 3.5))
    b.append(line(switches[2][0] + 34, switches[2][1] + 8, other[0] - 50, other[1], vic, 3.5, "bad" if k >= 3 else "ok"))
    # pause arrows
    if k >= 1:
        for tx, ty in switches:
            b.append(line(full[0] - 52, full[1] - 6, tx + 38, ty - 6, V["bad"], 3, "bad", dash="8 5"))
        b.append(text(640, 100, "PAUSE", 22, V["bad_t"], weight="bold"))
    if k >= 2:
        for i, (sx, sy) in enumerate(senders):
            tx, ty = switches[i // 2]
            b.append(line(tx - 38, ty - 6, sx + 26, sy - 6, V["bad"], 2.5, "bad", dash="8 5", opacity=0.75))
        b.append(text(250, 398, "pauses climb tier by tier", 19, V["bad_t"], weight="bold"))
    if k >= 3:
        b.append(text(700, 398, "victim flow stops too", 19, V["bad_t"], weight="bold"))
    for sx, sy in senders:
        b.append(f'<circle cx="{sx}" cy="{sy}" r="20" fill="{V["comp"]}"/>')
    for tx, ty in switches:
        b.append(box(tx - 34, ty - 26, 68, 52, V["coord_l"], V["coord"], 2.5, 5))
    b.append(box(full[0] - 50, full[1] - 34, 100, 68, V["bad"], V["bad"], 2, 5))
    b.append(text(full[0], full[1], "full port", 18, "white", weight="bold"))
    b.append(box(other[0] - 50, other[1] - 24, 100, 48, V["ok_l"], V["ok"], 2, 5))
    b.append(text(other[0], other[1], "other", 18, V["ok_t"], weight="bold"))
    write(f"pfc_spread_step{k}.svg", 900, 415, "".join(b))


for k in (1, 2, 3):
    pfc(k)

# =============================================================================
# 16. Cyclic buffer dependency (book figure 12, redrawn), column
# =============================================================================
b = []
pos = {1: (280, 70), 2: (470, 230), 3: (280, 390), 4: (90, 230)}
for i in (1, 2, 3, 4):
    x, y = pos[i]
    b.append(box(x - 80, y - 42, 160, 84, V["comm_l"], V["comm"], 2.5, 6))
    b.append(text(x, y - 20, f"switch {i}", 19, V["comm_t"], weight="bold"))
    for s in range(4):
        b.append(box(x - 60 + s * 30, y - 2, 26, 30, V["bad_l"], V["bad"], 2, 2))
# forwarding clockwise on the outside, PAUSE counterclockwise on the inside
b.append(path("M370,58 Q480,70 510,182", V["ink"], 3, "ink"))
b.append(path("M500,280 Q480,390 370,402", V["ink"], 3, "ink"))
b.append(path("M190,402 Q80,390 60,280", V["ink"], 3, "ink"))
b.append(path("M60,182 Q80,70 190,58", V["ink"], 3, "ink"))
b.append(path("M400,190 Q380,135 330,118", V["bad"], 3, "bad", dash="8 5"))
b.append(path("M330,345 Q380,330 400,272", V["bad"], 3, "bad", dash="8 5"))
b.append(path("M160,272 Q180,330 230,345", V["bad"], 3, "bad", dash="8 5"))
b.append(path("M230,118 Q180,135 160,190", V["bad"], 3, "bad", dash="8 5"))
b.append(text(280, 230, "deadlock", 24, V["bad_t"], weight="bold"))
b.append(text(520, 460, "solid: packets wait for the next queue", 18, V["ink"], "end"))
b.append(text(520, 486, "dashed: PAUSE to the previous switch", 18, V["bad_t"], "end"))
write("deadlock.svg", 560, 500, "".join(b))

# =============================================================================
# 17. DCQCN feedback loop (column)
# =============================================================================
b = []
b.append(box(5, 120, 160, 74, V["comp_l"], V["comp"], 2.5, 6))
b.append(text(85, 144, "sender", 20, V["comp_t"], weight="bold"))
b.append(text(85, 172, "cuts its rate", 15, V["comp_t"]))
b.append(box(215, 110, 150, 90, V["comm_l"], V["comm"], 2.5, 6))
b.append(text(290, 132, "switch", 20, V["comm_t"], weight="bold"))
for s in range(4):
    b.append(box(230 + s * 31, 150, 27, 36, V["bad_l"] if s < 3 else "white", V["bad"] if s < 3 else V["frame"], 2, 2))
b.append(box(415, 120, 140, 74, V["comp_l"], V["comp"], 2.5, 6))
b.append(text(485, 157, "receiver", 20, V["comp_t"], weight="bold"))
b.append(line(167, 157, 213, 157, V["comm"], 3, "comm"))
b.append(line(367, 157, 413, 157, V["comm"], 3, "comm"))
b.append(text(400, 96, "ECN mark", 17, V["bad_t"], weight="bold"))
b.append(path("M485,196 Q485,290 285,290 Q85,290 85,198", V["coord"], 3, "coord", dash="8 5"))
b.append(text(285, 314, "congestion notification", 17, V["coord_t"], weight="bold"))
b.append(text(280, 26, "DCQCN: a queue past its threshold", 18, V["ink"], weight="bold"))
b.append(text(280, 56, "marks packets; senders slow down", 18, V["ink"], weight="bold"))
write("dcqcn.svg", 560, 330, "".join(b))

# =============================================================================
# 18. Incast: offered load against one egress (column)
# =============================================================================
fig, ax = plt.subplots(figsize=(4.2, 1.9))
ax.barh([1, 0], [num("incast_offered_gbs"), num("incast_egress_gbs")], color=[V["bad"], V["comm"]], height=0.55)
ax.text(num("incast_offered_gbs") * 1.15, 1, f"{val('incast_offered_tbs')} TB/s", va="center", fontsize=15, fontweight="bold", color=V["bad_t"])
ax.text(num("incast_egress_gbs") * 1.15, 0, f"{val('incast_egress_gbs')} GB/s", va="center", fontsize=15, fontweight="bold", color=V["comm_t"])
ax.text(-0.02, 1, f"{val('incast_senders')} senders", transform=ax.get_yaxis_transform(), ha="right", va="center",
        fontsize=15, fontweight="bold")
ax.text(-0.02, 0, "one egress", transform=ax.get_yaxis_transform(), ha="right", va="center",
        fontsize=15, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(10, 2e5)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.6)
ax.spines["left"].set_visible(False)
ax.minorticks_off()
ax.set_xticks([10, 100, 1000, 10000])
ax.set_xticklabels(["10", "100", "1,000", "10,000"])
ax.tick_params(labelsize=13)
ax.set_xlabel("GB/s (log scale)", fontsize=14)
sp.save(fig, out / "incast_bars.svg")

# =============================================================================
# 19. Silent waste: cumulative GPU-hours lost over 30 days (scenario)
# =============================================================================
SW = calcs.SILENT
run_days = SW.run_duration.to("day").magnitude
days = np.linspace(0, run_days, 200)
hours_per_day = Q_(1, "day").to("hour").magnitude
fig, ax = plt.subplots(figsize=(4.8, 1.95))
for case, col in (("high", V["bad"]), ("low", V["comm"])):
    lost = SW.fleet.total_accelerators * hours_per_day * days * SW.slowdowns[case]
    ax.plot(days, lost, color=col, lw=3)
    ax.text(run_days + 0.6, lost[-1], f"{val('sw_' + case + '_pct')}% slower:\n{val('sw_lost_' + case)} GPU-h", color=col,
            fontsize=12, fontweight="bold", va="center")
ax.text(1, 26500, f"{val('ref_gpus')} GPUs, {val('sw_days')} days; no alarm fires", color=V["ink"], fontsize=12, fontweight="bold")
ax.set_xlim(0, 39)
ax.set_ylim(0, 29000)
ax.set_xticks([0, 10, 20, 30])
ax.set_yticks([0, 10000, 20000])
ax.set_yticklabels(["0", "10,000", "20,000"])
ax.set_xlabel("Days of training")
ax.set_ylabel("GPU-hours lost")
ax.grid(axis="y", color=C["grid"], lw=0.6)
sp.save(fig, out / "silent_waste.svg")

# =============================================================================
# 20. Healthy bus bandwidth by message size (measured, NCCL)
# =============================================================================
bus = np.array([r["oop_busbw_GBps"] for r in nccl])
fig, ax = plt.subplots(figsize=(4.8, 2.0))
ax.axvspan(mid_lo, mid_hi, color=V["comm_l"], lw=0)
ax.plot(xs[xs >= 1024], bus[xs >= 1024], "o-", color=V["comp"], lw=2.4, ms=6)
ax.text(3e5, 470, f"{val('nccl_mid_lo_mb')} to {val('nccl_mid_hi_mb')} MB:\n{val('nccl_mid_busbw_lo')} to {val('nccl_mid_busbw_hi')} GB/s", color=V["comm_t"], fontsize=12,
        fontweight="bold", ha="right", va="top")
ax.text(xs[-1], 560, f"{val('nccl_top_busbw')} GB/s at {val('nccl_top_gb')} GB", color=V["comp"], fontsize=12, fontweight="bold", ha="right")
ax.set_xscale("log", base=2)
ax.set_xticks([2**10, 2**20, 2**30, 16 * 2**30])
ax.set_xticklabels(["1 KB", "1 MB", "1 GB", "16 GB"])
ax.minorticks_off()
ax.set_ylim(0, 650)
ax.set_yticks([0, 200, 400])
ax.set_xlabel("Message size")
ax.set_ylabel("Bus bandwidth (GB/s)")
ax.grid(axis="y", color=C["grid"], lw=0.6)
sp.save(fig, out / "nccl_busbw.svg")

# =============================================================================
# 21. Diagnosis order, application to wire (column)
# =============================================================================
steps = [
    ("1  Utilization vs. timeline", "nvidia-smi", V["comp"], V["comp_l"]),
    ("2  Collective log", "NCCL_DEBUG=INFO", V["comm"], V["comm_l"]),
    ("3  Pairwise RDMA test", "ib_write_bw", V["comm"], V["comm_l"]),
    ("4  Pause, ECN counters", "switch", V["coord"], V["coord_l"]),
    ("5  Symbol errors, CRC", "port", V["bad"], V["bad_l"]),
]
b = []
for i, (title, tool, stroke, fill) in enumerate(steps):
    y = 4 + i * 64
    b.append(box(6, y, 588, 54, fill, stroke, 2.5, 6))
    b.append(text(20, y + 27, title, 21, V["ink"], "start", "bold"))
    b.append(text(580, y + 27, tool, 18, V["gray"], "end"))
    if i < 4:
        b.append(line(300, y + 54, 300, y + 64, V["gray"], 2.5, "gray"))
write("diag_ladder.svg", 600, 330, "".join(b))

# =============================================================================
# 22. SR-IOV slices and a checkpoint burst (column, optional slide)
# =============================================================================
fig, ax = plt.subplots(figsize=(4.4, 1.7))
vf, nic = num("vf_gbps"), num("nic_gbps")
for i in range(round(num("vf_count"))):
    ax.barh(1, vf, left=i * vf, color=V["comp"] if i == 0 else V["comp_l"],
            edgecolor=V["comp"], lw=1.5, height=0.55)
ax.text(vf / 2, 1, val("vf_gbps"), ha="center", va="center", color="white", fontsize=13, fontweight="bold")
ax.text(nic / 2, 1.55, f"one {val('nic_gbps')} Gb/s NIC as {val('vf_count')} virtual functions", ha="center", fontsize=13,
        fontweight="bold", color=V["comp_t"])
ax.barh(0, nic, color=V["bad_l"], edgecolor=V["bad"], lw=1.5, height=0.55)
ax.text(nic / 2, 0, f"{val('burst_gb')} GB checkpoint: port full for {val('burst_s')} s", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=V["bad_t"])
ax.set_xlim(0, 400)
ax.set_ylim(-0.5, 1.9)
ax.set_yticks([])
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_xlabel("Gb/s")
ax.spines["left"].set_visible(False)
sp.save(fig, out / "sriov_slices.svg")

# =============================================================================
# 23. Five levels with today's numbers (synthesis)
# =============================================================================
levels = [
    ("L5", "Cluster design", f"{val('su_count')} units of {val('su_gpus')} GPUs = the {val('ref_gpus')}-GPU job"),
    ("L4", "Fabric behavior", "PAUSE loop froze Microsoft's lossless class"),
    ("L3", "Switch, topology", f"{val('cut_full_tbs')} TB/s cut at 1:1; {val('oversub')}:1 gives {val('os_rel')}"),
    ("L2", "Transport", f"α {val('ndr_alpha_us')} μs, β {val('ib_gbs')} GB/s, crossover {val('cross_ndr_kb')} KB"),
    ("L1", "Wire and link", "PAM4 + FEC; copper to 3 m, then optics"),
]
b = []
for i, (tag, name, note) in enumerate(levels):
    y = 8 + i * 74
    hot = tag == "L4"
    b.append(box(10, y, 980, 62, V["bad_l"] if hot else V["comm_l"], V["bad"] if hot else V["comm"], 2.5, 6))
    b.append(text(46, y + 31, tag, 22, V["bad_t"] if hot else V["comm_t"], weight="bold"))
    b.append(text(88, y + 31, name, 20, V["ink"], "start", "bold"))
    b.append(text(390, y + 31, note, 18, V["ink"], "start"))
write("five_levels.svg", 1000, 380, "".join(b))

print("figures written to", out)
