#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 5 (Collective Communication), Volume II.

Every value is from the rendered chapter (Collective Communication) unless
noted. "Derived" values are arithmetic on chapter inputs, shown in the notes.

  - Horovod hook: images/s at 128 NVIDIA Pascal GPUs over 25GbE TCP, read
    from Sergeev and Del Balso 2018, Figure 6 (to about +/-250 images/s):
    Inception V3 ideal 17,000, distributed TensorFlow 9,000, Horovod 15,000;
    ResNet-101 ideal 15,000, TensorFlow 6,200, Horovod 13,300. The paper's
    text: "about half of our resources" lost at 128 GPUs; Horovod "88 percent
    efficiency mark". Efficiencies printed here are those readings divided by
    ideal (derived).
  - Traffic map: table 1 (Sec. 5.3).
  - Contracts: Sec. 5.4 and figure 2.
  - Critical size alpha*beta per interconnect: table 4 (Sec. 5.5). Chunks of
    a 25 MB bucket: 390.6 KB on 64 ranks, 24.4 KB on 1,024 (Napkin Math 5.1).
  - Scoreboard, 1 GB reference buffer on 64 GPUs at 50 GB/s:
    parameter server 64 x 1 GB / 50 GB/s = 1,280 ms per direction (derived
    from Sec. 5.6.1, server time N x M / beta); simple tree 240 ms and flat
    ring byte term 39.4 ms (Sec. 5.6.4, 5.6.2); hierarchical 8.3 ms
    (Napkin Math 5.4).
  - Rounds: ring 2(N-1), tree 2 log2 N; 2,046 against 20 at N = 1,024
    (Sec. 5.6.4).
  - Crossover: N = 64, alpha = 10 us, beta = 10 GB/s; 1 MB gives ring
    1,456.9 us and tree 1,320 us (Napkin Math 5.3). Curves are the chapter's
    two cost models; exact crossover 1.14 MB derived (chapter estimate
    1.1 MB).
  - Bandwidth per GPU, DGX H100: NVLink 450 GB/s per direction, InfiniBand
    NDR 50 GB/s (Sec. 5.7.1).
  - Hierarchical phases: 1.94 + 4.40 + 1.94 = 8.3 ms against a flat ring of
    about 40 ms (Napkin Math 5.4); phase times are printed on the diagram.
  - MoE dispatch: 8 GPUs, 512 tokens each, 64 per expert-rank pair,
    4.096 KB per token, 1.84 MB per GPU; receive bound 36.7 us uniform,
    55.1 us with a hot receiver at 672 remote tokens (Napkin Math 5.5, 5.6).
  - Compression payback: 40 ms -> 5 ms + 2 ms (Napkin Math 5.7). Run level:
    50,000 s -> 38,500 s; 33,000 s -> 33,825 s (Sec. 5.9.5).
  - 1-bit Adam: up to 5x communication volume, 3.3x BERT-Large and 2.9x
    SQuAD throughput (Tang et al. 2021, via Sec. 5.9.4).
  - Overlap: 32 layers, 15 ms backward, 27 ms of reduction per layer:
    1,344 ms sequential, 879 ms overlapped (Napkin Math 5.9); at 0.85 ms per
    100 MB bucket, 9 buckets = 7.6 ms per layer, so 480 + 7.6 = 487.6 ms
    (derived from Sec. 5.10.4).

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = dict(sp.C)
# Volume II roles (palette.md): compute blue, communication amber,
# coordination violet; green and red keep their base meanings.
C.update({
    "c1": "#1F407A", "c1_l": "#EAF1F8",
    "c2": "#d97706", "c2_l": "#fef3c7", "c2_t": "#b45309",
    "c3": "#7c3aed", "c3_l": "#f3e8ff",
    "ok": "#16a34a", "ok_l": "#dcfce7",
    "bad": "#dc2626", "bad_l": "#fee2e2",
    "frame": "#94a3b8", "frame_l": "#f8fafc",
})
sp.C.update(C)
out = Path(__file__).parent


def label(ax, x, y, text, color, shown=True, **kw):
    """Text drawn in every step (invisible until shown) so the bounding box,
    and so the slide position, never changes between steps."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


def blank(size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_axis_off()
    return fig, ax


# 1. Hook: Horovod against distributed TensorFlow at 128 GPUs --------------
def horovod(k, col=False):
    models = ["Inception V3", "ResNet-101"]
    ideal = [17000, 15000]
    tf = [9000, 6200]
    hv = [15000, 13200]
    fig, ax = plt.subplots(figsize=(3.5, 2.6) if col else (5.8, 2.0))
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, ideal[i], w, facecolor="none", edgecolor=C["text"], lw=1.6,
               ls="--")
        ax.bar(i + w / 2, ideal[i], w, facecolor="none", edgecolor=C["text"], lw=1.6,
               ls="--")
        ax.bar(i - w / 2, tf[i], w, color=C["bad"], alpha=1.0 if k >= 2 else 0.0)
        ax.bar(i + w / 2, hv[i], w, color=C["c2"], alpha=1.0 if k >= 3 else 0.0)
        label(ax, i - w / 2, tf[i] + 500, f"{round(100 * tf[i] / ideal[i])}%", C["bad"],
              k >= 2, ha="center", fontsize=13)
        label(ax, i + w / 2, hv[i] + 500, f"{round(100 * hv[i] / ideal[i])}%", C["c2_t"],
              k >= 3, ha="center", fontsize=13)
        label(ax, i, ideal[i] + 1500, f"ideal {ideal[i]:,}", C["text"], True,
              ha="center", fontsize=11, fontweight="normal")
    if col:
        label(ax, -0.55, 23500, "TensorFlow", C["bad"], k >= 2, fontsize=12)
        label(ax, 0.55, 23500, "Horovod", C["c2_t"], k >= 3, fontsize=12)
    else:
        label(ax, 1.62, 11000, "distributed\nTensorFlow", C["bad"], k >= 2, fontsize=12,
              va="center")
        label(ax, 1.62, 5000, "Horovod\n(ring AllReduce)", C["c2_t"], k >= 3, fontsize=12,
              va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, 1.6 if col else 2.35)
    ax.set_ylim(0, 25500 if col else 19500)
    ax.set_ylabel("Images per second")
    ax.set_yticks([0, 5000, 10000, 15000])
    ax.set_yticklabels(["0", "5,000", "10,000", "15,000"])
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / (f"horovod_col.svg" if col else f"horovod_step{k}.svg"))


# 2. Traffic map: parallelism -> primitive -> binding term -------------------
def traffic(k):
    rows = [
        ("Data parallel", "AllReduce", "bandwidth"),
        ("Sharded (FSDP)", "AllGather +\nReduceScatter", "bandwidth, many calls"),
        ("Tensor parallel", "AllReduce", "critical path, in node"),
        ("Pipeline", "Send / Recv", "latency"),
        ("Expert (MoE)", "AllToAll", "skew, narrowest cut"),
    ]
    fig, ax = blank((5.8, 2.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.5, 5.6)
    heads = [(0.0, "Split"), (4.4, "Primitive"), (8.7, "Binds on")]
    for j, (x, h) in enumerate(heads):
        label(ax, x, 5.15, h, C["neutral"], j == 0 or k > j, fontsize=12)
    for i, (sname, prim, bind) in enumerate(rows):
        y = 4.0 - i * 1.0
        label(ax, 0.0, y, sname, C["c1"], True, fontsize=12, va="center")
        a = 1.0 if k >= 2 else 0.0
        ax.annotate("", xy=(4.3, y), xytext=(3.75, y),
                    arrowprops=dict(arrowstyle="-|>", color=C["c2"], lw=1.8, alpha=a))
        label(ax, 4.4, y, prim, C["c2_t"], k >= 2, fontsize=12, va="center",
              linespacing=0.95)
        label(ax, 8.7, y, bind, C["text"], k >= 3, fontsize=11, va="center",
              fontweight="normal")
    sp.save(fig, out / f"traffic_step{k}.svg")


# 3. Contracts: what each rank holds before and after -----------------------
RANK = ["#c7d7ee", "#fde3b0", "#cdeed7", "#e7d9fb"]   # rank 0..3 fills
RANK_T = ["#1F407A", "#b45309", "#15803d", "#6d28d9"]


def cell(ax, x, y, fill, text="", tcolor=None, a=1.0, w=0.3, h=0.26):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="#475569", lw=0.8,
                           alpha=a))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10,
                fontweight="bold", color=tcolor or C["text"], alpha=a)


def contracts(k):
    names = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]
    fig, ax = blank((5.8, 2.35))
    ax.set_xlim(-0.95, 8.1)
    ax.set_ylim(-2.45, 1.75)
    for r in range(4):
        ax.text(-0.15, 1.13 - r * 0.3, f"R{r}", ha="right", va="center", fontsize=11,
                fontweight="bold", color=C["neutral"])
        ax.text(-0.15, -0.95 - r * 0.3, f"R{r}", ha="right", va="center", fontsize=11,
                fontweight="bold", color=C["neutral"])
    ax.text(-0.75, 0.68, "before", fontsize=11, color=C["neutral"], va="center",
            ha="center", rotation=90)
    ax.text(-0.75, -1.4, "after", fontsize=11, color=C["neutral"], va="center",
            ha="center", rotation=90)
    for p, name in enumerate(names):
        x0 = p * 2.05
        a = 1.0 if p < k else 0.0
        ax.text(x0 + 0.6, 1.62, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color=C["c2_t"], alpha=a)
        for r in range(4):
            yb = 1.0 - r * 0.3
            ya = -1.08 - r * 0.3
            for c in range(4):
                x = x0 + c * 0.3
                if name == "AllReduce":
                    cell(ax, x, yb, RANK[r], a=a)
                    cell(ax, x, ya, "#94a3b8", "Σ", "white", a=a)
                elif name == "AllGather":
                    if c == r:
                        cell(ax, x, yb, RANK[r], a=a)
                    cell(ax, x, ya, RANK[c], a=a)
                elif name == "ReduceScatter":
                    cell(ax, x, yb, RANK[r], a=a)
                    if c == r:
                        cell(ax, x, ya, "#94a3b8", f"Σ{r}", "white", a=a)
                else:
                    cell(ax, x, yb, RANK[r], f"{c}", RANK_T[r], a=a)
                    cell(ax, x, ya, RANK[c], f"{r}", RANK_T[c], a=a)
        ax.annotate("", xy=(x0 + 0.6, -0.27), xytext=(x0 + 0.6, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=C["c2"], lw=2, alpha=a))
    sp.save(fig, out / f"contracts_step{k}.svg")


# 4. Critical message size per interconnect, and the chunks of one bucket ---
def chunk(k):
    rows = [("NVLink 4.0", 700), ("PCIe Gen5", 224), ("Ethernet RoCE", 93.7),
            ("InfiniBand HDR", 87.5), ("InfiniBand NDR", 75)]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    n = len(rows)
    for i, (nm, v) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C["c2"], height=0.6)
        ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v * 0.93, y, f"{v:g} KB", va="center", ha="right", fontsize=11,
                fontweight="bold", color="white")
    a = 1.0 if k >= 2 else 0.0
    for x, txt, ck in [(390.6, "64 ranks:\n390.6 KB chunk", "ok"),
                       (24.4, "1,024 ranks:\n24.4 KB chunk", "bad")]:
        ax.axvline(x, color=C[ck], lw=2, ls="--", alpha=a)
        ax.text(x * 1.06, n - 0.35, txt, color=C[ck], fontsize=11, fontweight="bold",
                va="bottom", ha="left", alpha=a)
    ax.set_xscale("log")
    ax.set_xlim(8, 6000)
    ax.set_ylim(-0.6, n + 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10 KB", "100 KB", "1 MB"])
    ax.set_xlabel(r"Critical size $n^* = \alpha\beta$ (bytes where startup = transfer)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"chunk_step{k}.svg")


# 5. Running scoreboard: 1 GB reference buffer, 64 GPUs, 50 GB/s ------------
def scoreboard():
    rows = [("Parameter server", 1280, "bad", "1,280 ms (derived)"),
            ("Simple tree", 240, "neutral", "240 ms"),
            ("Flat ring", 39.4, "c2", "39.4 ms"),
            ("Hierarchical", 8.3, "ok", "8.3 ms")]
    sp.hbars_steps(str(out / "scoreboard"), rows, "ms", log=True,
                   xlabel="Time for one 1 GB AllReduce, 64 GPUs (ms, log scale)",
                   size=(5.8, 2.3))


# 6. Rounds against rank count -----------------------------------------------
def rounds():
    N = np.arange(2, 1025)
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.plot(N, 2 * (N - 1), color=C["c2"], lw=3)
    ax.plot(N, 2 * np.log2(N), color=C["c1"], lw=3)
    ax.text(120, 1500, "ring: 2(N−1)", color=C["c2_t"], fontsize=13,
            fontweight="bold")
    ax.text(560, 170, "tree: 2 log₂ N", color=C["c1"], fontsize=13, fontweight="bold")
    ax.plot([1024], [2046], "o", color=C["c2"], ms=9)
    ax.plot([1024], [20], "o", color=C["c1"], ms=9)
    ax.text(1000, 2046, "2,046 ", ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["c2_t"])
    ax.text(1000, 120, "20 ", ha="right", va="bottom", fontsize=12, fontweight="bold",
            color=C["c1"])
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 2300)
    ax.set_xlabel("Ranks N")
    ax.set_ylabel("Startup rounds")
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "rounds.svg")


# 7. Ring against tree, N = 64, alpha = 10 us, beta = 10 GB/s ----------------
def crossover():
    N, a, b = 64, 10e-6, 10e9
    M = np.logspace(3, 8, 300)
    ring = 2 * (N - 1) * a + 2 * (N - 1) / N * M / b
    tree = 2 * math.log2(N) * a + 2 * math.log2(N) * M / b
    mx = 114 * a * b / (12 - 2 * 63 / 64)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(M, ring * 1e6, color=C["c2"], lw=3)
    ax.plot(M, tree * 1e6, color=C["c1"], lw=3)
    ax.axvline(mx, color=C["neutral"], lw=1.4, ls="--")
    ax.text(mx * 1.25, 130, "crossover\n≈ 1.1 MB", color=C["neutral"], fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(2e3, 2.0e3, "ring", color=C["c2_t"], fontsize=13, fontweight="bold")
    ax.text(1.5e4, 300, "tree", color=C["c1"], fontsize=13, fontweight="bold")
    ax.text(4e7, 2.0e4, "tree", color=C["c1"], fontsize=13, fontweight="bold")
    ax.text(4e7, 2.2e3, "ring", color=C["c2_t"], fontsize=13, fontweight="bold")
    ax.plot([1e6], [1456.9], "o", color=C["c2"], ms=8)
    ax.plot([1e6], [1320], "o", color=C["c1"], ms=8)
    ax.text(1.3e3, 8e3, "1 MB: ring 1,457 μs\n          tree 1,320 μs", ha="left",
            fontsize=11, fontweight="bold", color=C["text"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(80, 2e5)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"])
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_yticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms"])
    ax.minorticks_off()
    ax.set_xlabel("Buffer size M")
    ax.set_ylabel("Modeled time")
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "crossover.svg")


# 8. Bandwidth tiers on one DGX H100 GPU --------------------------------------
def bw_tiers():
    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    ax.bar([0, 1], [450, 50], color=[C["c2"], C["c1"]], width=0.62)
    ax.text(0, 465, "450", ha="center", fontsize=14, fontweight="bold", color=C["c2_t"])
    ax.text(1, 65, "50", ha="center", fontsize=14, fontweight="bold", color=C["c1"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NVLink\nin node", "InfiniBand\nport"], fontsize=12,
                       fontweight="bold")
    ax.set_ylim(0, 540)
    ax.set_yticks([0, 200, 400])
    ax.set_ylabel("GB/s per GPU,\none direction", fontsize=11)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "bw_tiers.svg")


# 9. Hierarchical AllReduce: three phases on nodes of GPUs -------------------
def hier(k):
    fig, ax = blank((5.8, 2.3))
    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(-0.35, 4.3)
    nodes, gpus = 4, 4
    for n in range(nodes):
        x0 = n * 1.6
        ax.add_patch(FancyBboxPatch((x0, 0), 1.25, 3.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=C["frame_l"], edgecolor=C["frame"], lw=1.2))
        ax.text(x0 + 0.62, 3.85, f"node {n}", ha="center", fontsize=11,
                fontweight="bold", color=C["neutral"])
        for g in range(gpus):
            y = 2.75 - g * 0.85
            ax.add_patch(Rectangle((x0 + 0.08, y), 0.98, 0.55, facecolor=C["c1_l"],
                                   edgecolor=C["c1"], lw=1.2))
            ax.text(x0 + 0.57, y + 0.27, f"GPU {g}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=C["c1"])
        # phase 1 and 3: NVLink bracket inside the node
        ax.plot([x0 + 1.17, x0 + 1.17], [0.3, 3.2], color=C["c2"], lw=3.5,
                alpha=1.0 if k in (1, 3) else (0.25 if k == 2 else 0))
    # phase 2: rails across nodes, one per GPU position
    for g in range(gpus):
        y = 2.75 - g * 0.85 + 0.27
        ax.plot([1.1, 3 * 1.6 + 0.15], [y, y], color=C["c2"], lw=2.4,
                alpha=1.0 if k == 2 else (0.2 if k == 3 else 0.0), zorder=0)
    texts = ["1  ReduceScatter, NVLink\n   each GPU keeps 1/G: 1.94 ms",
             "2  AllReduce of the shard,\n   InfiniBand, one ring per\n   GPU position: 4.40 ms",
             "3  AllGather, NVLink: 1.94 ms"]
    for i, t in enumerate(texts):
        label(ax, 6.55, 3.4 - i * 1.3, t, C["c2_t"] if k == i + 1 else C["neutral"],
              k >= i + 1, fontsize=11, va="top")
    sp.save(fig, out / f"hier_step{k}.svg")


# 10. MoE dispatch: the AllToAll traffic matrix -------------------------------
def a2a_matrix():
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    n = 8
    for s in range(n):
        for d in range(n):
            local = s == d
            ax.add_patch(Rectangle((d, n - 1 - s), 1, 1,
                                   facecolor=C["neutral_l"] if local else C["c2_l"],
                                   edgecolor="white", lw=1.5))
            ax.text(d + 0.5, n - 0.5 - s, "64", ha="center", va="center", fontsize=9,
                    color=C["neutral"] if local else C["c2_t"], fontweight="bold")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=11)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([str(i) for i in range(n)][::-1], fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Receiving GPU", fontsize=12)
    ax.set_ylabel("Sending GPU", fontsize=12)
    ax.tick_params(length=0)
    for s in ["left", "bottom", "top", "right"]:
        ax.spines[s].set_visible(False)
    sp.save(fig, out / "a2a_matrix.svg")


def a2a_hot(k):
    rows = [("Uniform receiver", 36.7, "c2", "36.7 μs  (1.84 MB)"),
            ("Hot receiver, 1.5×", 55.1, "bad", "55.1 μs  (2.75 MB)")]
    fig, ax = plt.subplots(figsize=(5.8, 1.15))
    for i, (nm, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.55, alpha=1.0 if i < k else 0.0)
        ax.text(-1, y, nm, ha="right", va="center", fontsize=12, fontweight="bold",
                color=C["text"])
        label(ax, v + 1, y, txt, C["c2_t"] if i == 0 else C["bad"], i < k, fontsize=12,
              va="center")
    ax.set_xlim(0, 95)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Receive bound on dispatch at 50 GB/s (μs)")
    sp.save(fig, out / f"a2a_hot_step{k}.svg")


# 11. Compression: payback per synchronization, then per run ----------------
def payback():
    fig, ax = plt.subplots(figsize=(5.8, 1.35))
    ax.barh(1, 40, color=C["c2"], height=0.55)
    ax.text(40.8, 1, "40 ms uncompressed", va="center", fontsize=12, fontweight="bold",
            color=C["c2_t"])
    ax.barh(0, 5, color=C["c2"], height=0.55)
    ax.barh(0, 2, left=5, color=C["c1"], height=0.55)
    ax.text(7.8, 0, "5 ms transfer + 2 ms encode = 7 ms", va="center", fontsize=12,
            fontweight="bold", color=C["ok"])
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms per synchronization (8× effective, assumed)")
    sp.save(fig, out / "payback.svg")


def runlevel(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    groups = ["Slow network\n200 ms comm per step", "Fast network\n30 ms comm per step"]
    base = [50000, 33000]
    comp = [38500, 33825]
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, base[i], w, color=C["neutral"])
        ax.text(i - w / 2, base[i] + 900, f"{base[i]:,} s", ha="center", fontsize=11,
                fontweight="bold", color=C["neutral"])
        a = 1.0 if (k >= 2 and i == 0) or (k >= 3 and i == 1) else 0.0
        ax.bar(i + w / 2, comp[i], w, color=C["ok"] if i == 0 else C["bad"], alpha=a)
        label(ax, i + w / 2, comp[i] + 900, f"{comp[i]:,} s", C["ok"] if i == 0 else C["bad"],
              a > 0, ha="center", fontsize=11)
    label(ax, 0.0, 60500, "INT8 + error feedback: 23% faster", C["ok"], k >= 2,
          ha="center", fontsize=12)
    label(ax, 1.0, 60500, "2.5% slower", C["bad"], k >= 3, ha="center", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0, 66000)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Time to target (s)")
    ax.set_yticks([0, 20000, 40000])
    ax.set_yticklabels(["0", "20,000", "40,000"])
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"runlevel_step{k}.svg")


def onebit():
    rows = [("Communication volume", 5.0, "c2", "up to 5× less"),
            ("BERT-Large pretraining", 3.3, "ok", "up to 3.3× throughput"),
            ("SQuAD fine-tuning", 2.9, "ok", "up to 2.9× throughput")]
    sp.hbars(str(out / "onebit.svg"), rows, "", log=False,
             xlabel="1-bit Adam against uncompressed Adam, up to 256 GPUs",
             size=(5.8, 1.8))


# 12. Overlap: sequential against bucketed overlap, 32 layers ----------------
def overlap(k):
    L, tb = 32, 15.0
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    rows = []
    # (y, label, compute blocks, comm blocks, end, shown, color)
    seq_comm = [(480 + i * 27, 27) for i in range(L)]
    ov_comm = [(tb + i * 27, 27) for i in range(L)]
    fast_comm = [(tb * (i + 1), 7.6) for i in range(L)]
    sets = [("Sequential", seq_comm, 1344, 1), ("Overlapped", ov_comm, 879, 2),
            ("Overlapped,\nfaster buckets", fast_comm, 487.6, 3)]
    for j, (nm, comm, end, step) in enumerate(sets):
        y = 2 - j
        a = 1.0 if k >= step else 0.0
        for i in range(L):
            ax.add_patch(Rectangle((i * tb, y + 0.05), tb, 0.33, facecolor=C["c1"],
                                   edgecolor="white", lw=0.4, alpha=a))
        for (s, w) in comm:
            ax.add_patch(Rectangle((s, y - 0.38), w, 0.33, facecolor=C["c2"],
                                   edgecolor="white", lw=0.4, alpha=a))
        ax.text(-20, y, nm, ha="right", va="center", fontsize=11, fontweight="bold",
                color=C["text"], alpha=a)
        ax.text(end + 20, y - 0.2, f"{end:,.1f} ms".replace(".0 ms", " ms"),
                va="center", fontsize=12, fontweight="bold",
                color=[C["bad"], C["c2_t"], C["ok"]][j], alpha=a)
    ax.text(240, 2.6, "backward, 15 ms per layer", ha="center", fontsize=11,
            color=C["c1"], fontweight="bold")
    ax.text(1150, 0.15, "reduction, 27 ms\nper layer", ha="center", va="center", fontsize=11,
            color=C["c2_t"], fontweight="bold")
    ax.set_xlim(0, 1550)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms, one backward pass plus its reductions (7B model, 64 GPUs)")
    sp.save(fig, out / f"overlap_step{k}.svg")


# 13. A four-GPU ring --------------------------------------------------------
def ring():
    fig, ax = blank((2.6, 2.6))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    pos = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for i, (x, y) in enumerate(pos):
        ax.add_patch(FancyBboxPatch((x - 0.45, y - 0.22), 0.9, 0.44,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=C["c1_l"], edgecolor=C["c1"], lw=1.5))
        ax.text(x, y, f"GPU {i}", ha="center", va="center", fontsize=11,
                fontweight="bold", color=C["c1"])
    for i in range(4):
        (x0, y0), (x1, y1) = pos[i], pos[(i + 1) % 4]
        ax.add_patch(FancyArrowPatch((x0 * 0.7 + x1 * 0.3 + 0.12 * (x1 - x0), y0 * 0.7 + y1 * 0.3),
                                     (x0 * 0.3 + x1 * 0.7, y0 * 0.3 + y1 * 0.7 - 0.12 * (y1 - y0)),
                                     arrowstyle="-|>", mutation_scale=16, color=C["c2"],
                                     lw=2.2, connectionstyle="arc3,rad=-0.25"))
    sp.save(fig, out / "ring.svg")


if __name__ == "__main__":
    for k in (1, 2, 3):
        horovod(k)
        traffic(k)
        hier(k)
        runlevel(k)
        overlap(k)
    for k in (1, 2, 3, 4):
        contracts(k)
    for k in (1, 2):
        chunk(k)
        a2a_hot(k)
    scoreboard()
    horovod(3, col=True)
    rounds()
    crossover()
    bw_tiers()
    a2a_matrix()
    payback()
    onebit()
    ring()
    print("figures written to", out)
