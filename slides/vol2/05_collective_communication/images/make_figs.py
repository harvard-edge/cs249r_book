#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 5 (Collective Communication), Volume II.

Every number a figure prints or plots comes from calcs.py, which reads it from
MLSysIM (registry values, Literature, and Scenarios.*); the list below names
the source of each figure. Derived and illustrative values say so on the slide.
  - Horovod hook and trace: slides/vol2/05_collective_communication/images/data/horovod_scaling_128gpu.csv and
    HorovodTimelineTrace, Literature.Communication.Horovod* (Sergeev and Del
    Balso 2018; bar heights and phase edges read from the paper's figures).
  - Traffic map: table 1 (Sec. 5.3). Contracts: Sec. 5.4 and figure 2.
  - Critical size alpha*beta per interconnect: Scenarios.CriticalSizeByInterconnect
    (table 4); bucket chunks: Scenarios.DdpBucketRingNdr (Napkin Math 5.1).
  - Scoreboard: Scenarios.GradientBufferAllReduceNdr (parameter server, simple
    tree, ring) and Scenarios.HierarchicalAllReduce8x8DgxH100.
  - Rounds: ring 2(N-1), tree 2 log2 N, marked at the reference cluster.
  - Crossover: Scenarios.RingVsTreeSmallBuffer; curves are MLSysIM's ring and
    tree time models (Napkin Math 5.3).
  - Bandwidth per GPU: H100 NVLink per direction and the NDR port (Sec. 5.7.1).
  - Hierarchical phases: Scenarios.HierarchicalAllReduce8x8DgxH100 (Napkin Math 5.4).
  - MoE dispatch: Scenarios.MoeDispatch8Gpu64Experts (Napkin Math 5.5, 5.6).
  - Compression: Scenarios.CompressionPayback8x (Napkin Math 5.7) and
    Scenarios.Int8CompressionRun (Sec. 5.9.5).
  - 1-bit Adam: Literature.Communication.OneBitAdam* (Tang et al. 2021).
  - Overlap: Scenarios.OverlapBudget7B64Gpu and its ideal-bucket variant
    (Napkin Math 5.9, Sec. 5.10.4).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402  (records every value; figures read them)
from mlsysim.core.series_loader import load_series  # noqa: E402
from deckvals import val, num  # noqa: E402
from mlsysim import Literature  # noqa: E402
from mlsysim.physics import calc_ring_allreduce_time, calc_tree_allreduce_time  # noqa: E402
from mlsysim.core.units import Q_  # noqa: E402

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
    keys = ["incv3", "resnet"]
    ideal = [num(f"hvd_ideal_{k}") for k in keys]
    tf = [num(f"hvd_tf_{k}") for k in keys]
    hv = [num(f"hvd_hv_{k}") for k in keys]
    fig, ax = plt.subplots(figsize=(3.5, 2.6) if col else (5.8, 2.0))
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, ideal[i], w, facecolor="none", edgecolor=C["text"], lw=1.6,
               ls="--")
        ax.bar(i + w / 2, ideal[i], w, facecolor="none", edgecolor=C["text"], lw=1.6,
               ls="--")
        ax.bar(i - w / 2, tf[i], w, color=C["bad"], alpha=1.0 if k >= 2 else 0.0)
        ax.bar(i + w / 2, hv[i], w, color=C["c2"], alpha=1.0 if k >= 3 else 0.0)
        label(ax, i - w / 2, tf[i] + 500, f"{val(f'hvd_tf_{keys[i]}_pct')}%", C["bad"],
              k >= 2, ha="center", fontsize=13)
        label(ax, i + w / 2, hv[i] + 500, f"{val(f'hvd_hv_{keys[i]}_pct')}%", C["c2_t"],
              k >= 3, ha="center", fontsize=13)
        label(ax, i, ideal[i] + 1500, f"ideal {val(f'hvd_ideal_{keys[i]}')}", C["text"], True,
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
    rows = [("NVLink 4.0", "nstar_nvlink_kb"), ("PCIe Gen5", "nstar_pcie_kb"),
            ("Ethernet RoCE", "nstar_roce_kb"), ("InfiniBand HDR", "nstar_hdr_kb"),
            ("InfiniBand NDR", "nstar_ndr_kb")]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    n = len(rows)
    for i, (nm, key) in enumerate(rows):
        y = n - 1 - i
        v = num(key)
        ax.barh(y, v, color=C["c2"], height=0.6)
        ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        xl = v * 0.93  # right edge of the label; keep it clear of the chunk lines
        for line_x in (num("bkt_chunk_small_kb"), num("bkt_chunk_large_kb")):
            if xl / 2.2 < line_x < xl:
                xl = line_x * 0.93
        ax.text(xl, y, f"{val(key)} KB", va="center", ha="right", fontsize=11,
                fontweight="bold", color="white")
    a = 1.0 if k >= 2 else 0.0
    for x, txt, ck in [(num("bkt_chunk_small_kb"),
                        f"{val('bkt_n_small')} ranks:\n{val('bkt_chunk_small_kb')} KB chunk", "ok"),
                       (num("bkt_chunk_large_kb"),
                        f"{val('bkt_n_large')} ranks:\n{val('bkt_chunk_large_kb')} KB chunk", "bad")]:
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


# 5. Running scoreboard: the reference buffer on the scoreboard cluster -------
def scoreboard():
    rows = [("Parameter server", num("ps_ms"), "bad", f"{val('ps_ms')} ms (derived)"),
            ("Simple tree", num("tree_bytes_ms"), "neutral", f"{val('tree_bytes_ms')} ms"),
            ("Flat ring", num("ring_bytes_ms"), "c2", f"{val('ring_bytes_ms')} ms"),
            ("Hierarchical", num("hier_total_ms"), "ok", f"{val('hier_total_ms')} ms")]
    sp.hbars_steps(str(out / "scoreboard"), rows, "ms", log=True,
                   xlabel=f"Time for one {val('buf_gb')} GB AllReduce, {val('grp_gpus')} GPUs (ms, log scale)",
                   size=(5.8, 2.3))


# 6. Rounds against rank count -----------------------------------------------
def rounds():
    n_ref = int(num("ref_n"))
    N = np.arange(2, n_ref + 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.plot(N, 2 * (N - 1), color=C["c2"], lw=3)
    ax.plot(N, 2 * np.log2(N), color=C["c1"], lw=3)
    ax.text(120, 1500, "ring: 2(N−1)", color=C["c2_t"], fontsize=13,
            fontweight="bold")
    ax.text(560, 170, "tree: 2 log₂ N", color=C["c1"], fontsize=13, fontweight="bold")
    ax.plot([n_ref], [num("ring_rounds_large")], "o", color=C["c2"], ms=9)
    ax.plot([n_ref], [num("tree_rounds_large")], "o", color=C["c1"], ms=9)
    ax.text(1000, num("ring_rounds_large"), f"{val('ring_rounds_large')} ", ha="right", va="center",
            fontsize=12, fontweight="bold", color=C["c2_t"])
    ax.text(1000, 120, f"{val('tree_rounds_large')} ", ha="right", va="bottom", fontsize=12,
            fontweight="bold", color=C["c1"])
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 2300)
    ax.set_xlabel("Ranks N")
    ax.set_ylabel("Startup rounds")
    ax.set_xticks([0, 256, 512, 768, n_ref])
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "rounds.svg")


# 7. Ring against tree for a small buffer (Scenarios.RingVsTreeSmallBuffer) -----
def crossover():
    S = calcs.RVT
    N, a, b = S.group_sizes[0], S.alpha, S.bandwidth
    M = np.logspace(3, 8, 300)
    ring = np.array([calc_ring_allreduce_time(Q_(x, "byte"), N, b, a).to("second").magnitude for x in M])
    tree = np.array([calc_tree_allreduce_time(Q_(x, "byte"), N, b, a).to("second").magnitude for x in M])
    mx = S.result.crossover_exact[N].to("byte").magnitude
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    ax.plot(M, ring * 1e6, color=C["c2"], lw=3)
    ax.plot(M, tree * 1e6, color=C["c1"], lw=3)
    ax.axvline(mx, color=C["neutral"], lw=1.4, ls="--")
    ax.text(mx * 1.25, 130, f"crossover\n≈ {val('rt_cross_mb')} MB", color=C["neutral"], fontsize=12.5,
            fontweight="bold", va="bottom")
    ax.text(2e3, 2.0e3, "ring", color=C["c2_t"], fontsize=13, fontweight="bold")
    ax.text(1.5e4, 300, "tree", color=C["c1"], fontsize=13, fontweight="bold")
    ax.text(4e7, 2.0e4, "tree", color=C["c1"], fontsize=13, fontweight="bold")
    ax.text(4e7, 2.2e3, "ring", color=C["c2_t"], fontsize=13, fontweight="bold")
    m_b = S.message.to("byte").magnitude
    ax.plot([m_b], [num("rt_ring_total_us")], "o", color=C["c2"], ms=8)
    ax.plot([m_b], [num("rt_tree_total_us")], "o", color=C["c1"], ms=8)
    ax.text(1.3e3, 8e3, f"{val('rt_m_mb')} MB: ring {val('rt_ring_total_us0')} μs\n          tree {val('rt_tree_total_us')} μs", ha="left",
            fontsize=12.5, fontweight="bold", color=C["text"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(80, 2e5)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"])
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_yticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms"])
    ax.minorticks_off()
    ax.set_xlabel("Buffer size M", fontsize=13)
    ax.tick_params(labelsize=12.5)
    ax.set_ylabel("Modeled time", fontsize=13)
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "crossover.svg")


# 8. Bandwidth tiers on one DGX H100 GPU --------------------------------------
def bw_tiers():
    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    nv, ib = num("nvlink_gbs"), num("ib_gbs")
    ax.bar([0, 1], [nv, ib], color=[C["c2"], C["c1"]], width=0.62)
    ax.text(0, nv + 15, val("nvlink_gbs"), ha="center", fontsize=14, fontweight="bold", color=C["c2_t"])
    ax.text(1, ib + 15, val("ib_gbs"), ha="center", fontsize=14, fontweight="bold", color=C["c1"])
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
    texts = [f"1  ReduceScatter, NVLink\n   each GPU keeps 1/G: {val('hier_rs_ms')} ms",
             f"2  AllReduce of the shard,\n   InfiniBand, one ring per\n   GPU position: {val('hier_inter_ms')} ms",
             f"3  AllGather, NVLink: {val('hier_rs_ms')} ms"]
    for i, t in enumerate(texts):
        label(ax, 6.55, 3.4 - i * 1.3, t, C["c2_t"] if k == i + 1 else C["neutral"],
              k >= i + 1, fontsize=11, va="top")
    sp.save(fig, out / f"hier_step{k}.svg")


# 10. MoE dispatch: the AllToAll traffic matrix -------------------------------
def a2a_matrix():
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    n = int(num("moe_gpus"))
    for s in range(n):
        for d in range(n):
            local = s == d
            ax.add_patch(Rectangle((d, n - 1 - s), 1, 1,
                                   facecolor=C["neutral_l"] if local else C["c2_l"],
                                   edgecolor="white", lw=1.5))
            ax.text(d + 0.5, n - 0.5 - s, val("moe_pair_tokens"), ha="center", va="center", fontsize=9,
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
    rows = [("Uniform receiver", num("moe_rx_us"), "c2", f"{val('moe_rx_us')} μs  ({val('moe_sent_mb')} MB)"),
            (f"Hot receiver, {val('moe_skew')}×", num("moe_hot_us"), "bad",
             f"{val('moe_hot_us')} μs  ({val('moe_hot_mb')} MB)")]
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
    ax.set_xlabel(f"Receive bound on dispatch at {val('ib_gbs')} GB/s (μs)")
    sp.save(fig, out / f"a2a_hot_step{k}.svg")


# 11. Compression: payback per synchronization, then per run ----------------
def payback():
    fig, ax = plt.subplots(figsize=(5.8, 1.35))
    comm, xfer, codec = num("cp_comm_ms"), num("cp_xfer_ms"), num("cp_codec_ms")
    ax.barh(1, comm, color=C["c2"], height=0.55)
    ax.text(comm + 0.8, 1, f"{val('cp_comm_ms')} ms uncompressed", va="center", fontsize=12,
            fontweight="bold", color=C["c2_t"])
    ax.barh(0, xfer, color=C["c2"], height=0.55)
    ax.barh(0, codec, left=xfer, color=C["c1"], height=0.55)
    ax.text(num("cp_total_ms") + 0.8, 0,
            f"{val('cp_xfer_ms')} ms transfer + {val('cp_codec_ms')} ms encode = {val('cp_total_ms')} ms",
            va="center", fontsize=12, fontweight="bold", color=C["ok"])
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"ms per synchronization ({val('cp_ratio')}× effective, assumed)")
    sp.save(fig, out / "payback.svg")


def runlevel(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    groups = [f"Slow network\n{val('cr_slow_comm_ms')} ms comm per step",
              f"Fast network\n{val('cr_fast_comm_ms')} ms comm per step"]
    base = [num("cr_slow_base_s"), num("cr_fast_base_s")]
    comp = [num("cr_slow_comp_s"), num("cr_fast_comp_s")]
    base_t = [val("cr_slow_base_s"), val("cr_fast_base_s")]
    comp_t = [val("cr_slow_comp_s"), val("cr_fast_comp_s")]
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, base[i], w, color=C["neutral"])
        ax.text(i - w / 2, base[i] + 900, f"{base_t[i]} s", ha="center", fontsize=11,
                fontweight="bold", color=C["neutral"])
        a = 1.0 if (k >= 2 and i == 0) or (k >= 3 and i == 1) else 0.0
        ax.bar(i + w / 2, comp[i], w, color=C["ok"] if i == 0 else C["bad"], alpha=a)
        label(ax, i + w / 2, comp[i] + 900, f"{comp_t[i]} s", C["ok"] if i == 0 else C["bad"],
              a > 0, ha="center", fontsize=11)
    label(ax, 0.0, 60500, f"INT8 + error feedback: {val('cr_slow_gain_pct')}% faster", C["ok"], k >= 2,
          ha="center", fontsize=12)
    label(ax, 1.0, 60500, f"{val('cr_fast_loss_pct')}% slower", C["bad"], k >= 3, ha="center", fontsize=12)
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
    rows = [("Communication volume", num("oba_volume"), "c2", f"up to {val('oba_volume')}× less"),
            ("BERT-Large pretraining", num("oba_bert"), "ok", f"up to {val('oba_bert')}× throughput"),
            ("SQuAD fine-tuning", num("oba_squad"), "ok", f"up to {val('oba_squad')}× throughput")]
    sp.hbars(str(out / "onebit.svg"), rows, "", log=False,
             xlabel=f"1-bit Adam against uncompressed Adam, up to {val('oba_gpus')} GPUs",
             size=(5.8, 1.8))


# 12. Overlap: sequential against bucketed overlap, layer by layer -----------
def overlap(k):
    L, tb = int(num("ov_layers")), num("ov_bwd_ms")
    tc, t_bwd, t_fast = num("ov_comm_ms"), num("ov_total_bwd_ms"), num("ov_ideal_layer_ms")
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    rows = []
    # (y, label, compute blocks, comm blocks, end, shown, color)
    seq_comm = [(t_bwd + i * tc, tc) for i in range(L)]
    ov_comm = [(tb + i * tc, tc) for i in range(L)]
    fast_comm = [(tb * (i + 1), t_fast) for i in range(L)]
    sets = [("Sequential", seq_comm, "ov_seq_ms", 1), ("Overlapped", ov_comm, "ov_overlap_ms", 2),
            ("Overlapped,\nfaster buckets", fast_comm, "ov_fast_ms", 3)]
    for j, (nm, comm, end_key, step) in enumerate(sets):
        end = num(end_key)
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
        ax.text(end + 20, y - 0.2, f"{val(end_key)} ms",
                va="center", fontsize=12, fontweight="bold",
                color=[C["bad"], C["c2_t"], C["ok"]][j], alpha=a)
    ax.text(240, 2.6, f"backward, {val('ov_bwd_ms')} ms per layer", ha="center", fontsize=11,
            color=C["c1"], fontweight="bold")
    ax.text(1150, 0.15, f"reduction, {val('ov_comm_ms')} ms\nper layer", ha="center", va="center", fontsize=11,
            color=C["c2_t"], fontweight="bold")
    ax.set_xlim(0, 1550)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"ms, one backward pass plus its reductions ({val('ov_params_b')}B model, {val('ov_gpus')} GPUs)")
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


# 15. Horovod Timeline (Sergeev and Del Balso 2018, Figure 5), redrawn -------
# Phase edges (read from the screenshot) are slides/vol2/05_collective_communication/images/data/horovod_timeline_trace.csv;
# the ALLREDUCE span of tensors 2 to 4 is printed in its detail pane
# (Literature.Communication.HorovodTimelineAllreduce*). Derived spans: calcs.py.
def horovod_trace():
    trace = load_series("slides/vol2/05_collective_communication/images/data/horovod_timeline_trace.csv").entries
    fast = [(r["phase"], r["start_ms"], r["end_ms"]) for r in trace if r["tensors"] == "0-1"]
    slow = [(r["phase"], r["start_ms"], r["end_ms"]) for r in trace if r["tensors"] == "2-4"]
    color = {"negotiate": C["c3"], "wait": C["neutral_l"], "schedule": C["c3_l"],
             "NCCL": C["c2"], "copy": C["c2"]}
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for row in range(5):
        y = 4 - row
        for name, a, b in (fast if row < 2 else slow):
            ax.barh(y, b - a, left=a, height=0.62, color=color[name],
                    edgecolor="white", lw=0.6)
        ax.text(-6, y, f"tensor {row}", ha="right", va="center", fontsize=12)
    # ALLREDUCE span of tensors 2 to 4, from the detail pane
    span = (num("hvd_ar_start"), num("hvd_ar_end"))
    ax.plot(span, [-1.25, -1.25], color=C["text"], lw=1.4)
    for x in span:
        ax.plot([x, x], [-1.35, -1.15], color=C["text"], lw=1.4)
    ax.text(255.5, -1.45, f"AllReduce {val('hvd_ar_ms')} ms", ha="center", va="top", fontsize=11.5)
    ax.text(88, -0.45, f"negotiate {val('hvd_negotiate_ms')} ms", ha="center", va="top", fontsize=12,
            fontweight="bold", color=C["c3"])
    ax.text(219, -0.45, "wait", ha="center", va="top", fontsize=12, color=C["neutral"])
    ax.text(352, -0.45, f"NCCL {val('hvd_nccl_ms')} ms", ha="right", va="top", fontsize=12,
            fontweight="bold", color=C["c2_t"])
    ax.text(144, 4, "NCCL", ha="left", va="center", fontsize=12, fontweight="bold",
            color=C["c2_t"])
    ax.set_xlim(0, 360)
    ax.set_ylim(-2.2, 4.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel(f"Time in the trace (ms after {val('hvd_t0_ms')} ms)", fontsize=12)
    sp.save(fig, out / "horovod_trace.svg")


if __name__ == "__main__":
    horovod_trace()
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
