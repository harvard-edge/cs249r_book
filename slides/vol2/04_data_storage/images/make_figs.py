#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 4 (Data Storage).

Every number drawn here comes from calcs.py, which reads MLSysIM (registry,
Literature, and the Scenarios.* the chapter also reads): plotted geometry uses
the scenario values, labels use the formatted strings the slides print
(deckvals val/num). Diagram coordinates, axis ranges, and tick positions are
layout. The tier table's capacity column, the archive rate, the latency
column except NVMe, and the "7 to 25 GB/s" upper bound are the chapter's
table 2 entries, typed there and here. The 3FS trace is the DeepSeek 3FS
README read stress test (MIT license), cropped and relabeled on the slide.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Volume II schematic roles (volumes/vol2/palette.md)
V = {
    "c1_fill": "#EAF1F8", "c1": "#1F407A", "c1_text": "#14294E",
    "c2_fill": "#fef3c7", "c2": "#d97706", "c2_text": "#b45309",
    "c3_fill": "#f3e8ff", "c3": "#7c3aed", "c3_text": "#6d28d9",
    "st_fill": "#dcfce7", "st": "#16a34a", "st_text": "#15803d",
    "bad_fill": "#fee2e2", "bad": "#dc2626", "bad_text": "#b91c1c",
    "fr_fill": "#f8fafc", "fr": "#94a3b8", "text": "#0f172a", "mute": "#475569",
}
FONT = "Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# A tiny SVG writer for diagrams (900 x 500 canvas, text kept as text)
# ---------------------------------------------------------------------------
class Svg:
    def __init__(self, w=900, h=500):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, stroke, sw=2, dash=None, rx=6, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=22, color=None, weight="normal", anchor="middle",
             italic=False, op=1.0):
        st = ' font-style="italic"' if italic else ""
        self.items.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or V["text"]}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{st} opacity="{op}">{s}</text>')

    def line(self, x1, y1, x2, y2, color, sw=3, dash=None, arrow=False, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d}{mk} opacity="{op}"/>')
        if arrow:
            self.markers.add(color)

    def poly(self, pts, color, sw=3, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.items.append(f'<polyline points="{p}" fill="none" stroke="{color}" '
                          f'stroke-width="{sw}"{d}{mk}/>')
        if arrow:
            self.markers.add(color)

    def polygon(self, pts, fill, stroke, sw=2):
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.items.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def dots(self, x, y, color):
        """A vertical ellipsis drawn as three dots (no font glyph needed)."""
        for dy in (-10, 0, 10):
            self.items.append(f'<circle cx="{x}" cy="{y + dy}" r="3" fill="{color}"/>')

    def cross(self, x, y, r, color):
        """A bold X drawn as two lines."""
        self.line(x - r, y - r, x + r, y + r, color, 6)
        self.line(x - r, y + r, x + r, y - r, color, 6)

    markers = set()

    def save(self, path):
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            f'markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self.markers))
        body = "\n".join(self.items)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" font-family="{FONT}">\n'
            f'<defs>{defs}</defs>\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n'
            f'{body}\n</svg>\n')
        Svg.markers = set()


def box(s, x, y, w, h, role, title, sub=None, tsize=22, ssize=19, op=1.0, dash=None):
    s.rect(x, y, w, h, V[role + "_fill"], V[role], dash=dash, op=op)
    tc = V.get(role + "_text", V["text"])
    if sub:
        s.text(x + w / 2, y + h / 2 - 4, title, tsize, tc, "bold", op=op)
        s.text(x + w / 2, y + h / 2 + 22, sub, ssize, V["mute"], op=op)
    else:
        s.text(x + w / 2, y + h / 2 + 8, title, tsize, tc, "bold", op=op)


# ---------------------------------------------------------------------------
# 1. Hook: bytes read against bytes written over the 30-day run
# ---------------------------------------------------------------------------
sp.hbars(out / "hook_bytes.svg",
         [("Read: one epoch of tokens", num("corpus_tb"), "data", f"{val('corpus_tb')} TB"),
          (f"Write: checkpoints, {val('run_days')} days", K.m(K.FOOT.result.checkpoint_bytes_written, "TB"), "memory",
           f"{val('written_pb')} PB")],
         "TB", log=True, size=(4.1, 1.05), xlabel="Terabytes moved (log scale)")

# ---------------------------------------------------------------------------
# 2. I/O wall: throughput of an s-times faster accelerator, delivery fixed
# ---------------------------------------------------------------------------
speed = np.array([1, 2, 4, 8, 16])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.35, 1.85))
    ax.plot(speed, speed, color=C["neutral"], lw=2, ls="--", marker="o", ms=6)
    ax.text(1.0, 9.5, "if storage kept up", color=C["neutral"], fontsize=12,
            fontweight="bold", ha="left")
    shown = speed if k == 2 else speed[:3]
    ax.plot(shown, [K.WALL.relative_throughput(x) for x in shown], color=C["danger"], lw=3, marker="o", ms=9)
    a = 1.0 if k == 2 else 0.0
    ax.axhline(K.WALL.plateau, color=C["data"], lw=1.5, ls=":", alpha=a)
    ax.text(17, 3.3, f"plateau: {val('wall_plateau')}× at 16× speed", color=C["danger"],
            fontsize=12.5, fontweight="bold", ha="right", va="top", alpha=a)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(speed, [f"{v}×" for v in speed])
    ax.set_yticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 20)
    ax.set_ylim(0.8, 20)
    ax.set_xlabel("Accelerator speed (relative)")
    ax.set_ylabel("Batches per second")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"io_wall_step{k}.svg")

# ---------------------------------------------------------------------------
# 3. MLPerf Storage v1.0: measured demand (MLSysIM)
# ---------------------------------------------------------------------------
rows = K.BENCH.rows
WL = [("unet3d", "3D U-Net", "memory"), ("cosmoflow", "CosmoFlow", "compute"),
      ("resnet50", "ResNet-50", "data")]

fig, ax = plt.subplots(figsize=(3.9, 1.95))
for w, name, ck in WL:
    for acc, mk in zip(K.BENCH.accelerators, ("o", "^")):
        pts = [(r["num_accelerators"], r["io_GBps"]) for r in rows
               if r["workload"] == w and r["accelerator"] == acc]
        x, y = zip(*pts)
        ax.scatter(x, y, s=34, marker=mk, color=C[ck], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
for (w, name, ck), (tx, ty) in zip(WL, [(200, 2.6), (200, 1.0), (200, 0.38)]):
    ax.text(tx, ty, name, color=C[ck], fontsize=12.5, fontweight="bold")
ax.text(1.1, 400, "circle: A100   triangle: H100", fontsize=11, color=C["text"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Emulated accelerators")
ax.set_ylabel("Read throughput (GB/s)")
ax.grid(color=C["grid"], lw=0.6)
ax.minorticks_off()
sp.save(fig, out / "mlperf_scatter.svg")

# ---------------------------------------------------------------------------
# 4. The six tiers with their scope, bandwidth, latency, and price
# ---------------------------------------------------------------------------
tiers = [("GPU HBM", "c1", f"{val('hbm_cap_gb')} GB", f"{val('hbm_tbs')} TB/s per GPU", f"~{val('hbm_lat_ns')} ns"),
         ("Host DRAM", "c1", "0.5 to 2 TB", f"{val('dram_gbs')} GB/s per node", f"~{val('dram_lat_ns')} ns"),
         ("Local NVMe", "st", "4 to 30 TB", f"{val('gen4_gbs')} to 25 GB/s", f"~{val('nvme_lat_us')} µs"),
         ("Parallel file system", "st", "100+ PB", f"{val('pfs_tbs')}+ TB/s shared", "~1 ms"),
         ("Object storage", "st", "very large", f"{val('obj_agg_gbs')} GB/s shared", "~50 ms"),
         ("Archive", "fr", "very large", "1 GB/s", "min to h")]
for k in (1, 2):
    s = Svg(900, 345)
    top, rh, gap = 36, 47, 4
    s.text(270, 24, "Tier (wider = more capacity)", 20, V["mute"], "bold")
    if k == 2:
        for x, lab in ((555, "Capacity"), (710, "Bandwidth"), (857, "Latency")):
            s.text(x, 24, lab, 20, V["mute"], "bold")
    for i, (name, role, cap, bw, lat) in enumerate(tiers):
        y = top + i * (rh + gap)
        half = 78 + i * 28
        s.polygon([(270 - half, y + rh), (270 + half, y + rh), (270 + half - 22, y),
                   (270 - half + 22, y)], V[role + "_fill"], V[role])
        s.text(270, y + rh / 2 + 8, name, 22, V.get(role + "_text", V["text"]), "bold")
        s.text(4, y + rh / 2 + 8, f"{i}", 22, V["mute"], "bold", anchor="start")
        if k == 2:
            s.text(555, y + rh / 2 + 8, cap, 20)
            s.text(710, y + rh / 2 + 8, bw, 20, V["c1_text"], "bold")
            s.text(857, y + rh / 2 + 8, lat, 20)
    s.save(out / f"tiers_step{k}.svg")

# ---------------------------------------------------------------------------
# 5. Per-client bandwidth ladder with the step ratios
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.7, 1.65))
lad = [("GPU HBM (one GPU)", K.m(K.H100.memory.bandwidth, "GB/s"), "memory", f"{val('hbm_tbs')} TB/s"),
       ("Host DRAM (node)", K.m(K.TECH.HostDram.bandwidth, "GB/s"), "memory", f"{val('dram_gbs')} GB/s"),
       (f"{val('nvme_drives')} NVMe, RAID 0 (node)", K.m(K.raid, "GB/s"), "data", f"{val('raid_gbs')} GB/s"),
       ("File system share (node)", K.m(K.share, "GB/s"), "data", f"{val('pfs_share_gbs')} GB/s")]
n = len(lad)
for i, (lab, v, ck, txt) in enumerate(lad):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=12.5, fontweight="bold", color=C["text"])
    ax.text(v * 1.2, y, txt, va="center", fontsize=12.5, fontweight="bold", color=C[ck])
for i, r in enumerate([f"÷ {val(k)}" for k in ("r_hbm_dram", "r_dram_nvme", "r_nvme_pfs")]):
    ax.text(2e6, n - 1.5 - i, r, va="center", ha="center", fontsize=12.5,
            fontweight="bold", color=C["danger"])
ax.set_xscale("log")
ax.set_xlim(1, 1e7)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("GB/s per client (log scale)")
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
ax.minorticks_off()
sp.save(fig, out / "ladder.svg")

# ---------------------------------------------------------------------------
# 6. Cache: delivered fraction of the hot rate against the hit rate
# ---------------------------------------------------------------------------
h = np.linspace(0.5, 1.0, 400)
fig, ax = plt.subplots(figsize=(4.85, 1.5))
for kmiss, ck in ((K.MULTI.result.miss_penalty, "data"), (K.HOT.result.miss_penalty, "danger")):
    ax.plot(h * 100, 1 / (h + (1 - h) * kmiss) * 100, color=C[ck], lw=3)
hot, mm = K.HOT.result, K.MULTI.result
ax.plot([hot.hit_rate * 100], [hot.delivered_fraction * 100], "o", ms=9, color=C["danger"], mec="white", mew=1.5)
ax.annotate(f"{val('hot_hit_pct')}% hits, misses {val('hot_penalty')}× slower: {val('hot_frac_pct')}%",
            (hot.hit_rate * 100, hot.delivered_fraction * 100), xytext=(53, 10), ha="left", fontsize=11.5,
            fontweight="bold", color=C["danger"],
            arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1.2))
ax.plot([mm.hit_rate * 100], [mm.delivered_fraction * 100], "o", ms=9, color=C["data"], mec="white", mew=1.5)
ax.annotate(f"{val('mm_hit_pct')}% hits, misses {val('mm_penalty')}× slower: {val('mm_frac_pct')}%",
            (mm.hit_rate * 100, mm.delivered_fraction * 100), xytext=(51, 86), ha="left", fontsize=11.5,
            fontweight="bold", color=C["data"],
            arrowprops=dict(arrowstyle="-", color=C["data"], lw=1.2))
ax.set_xlim(50, 100)
ax.set_ylim(0, 105)
ax.set_xlabel("Hit rate in the hot tier (%)")
ax.set_ylabel("% of all-hot rate")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "cache_hit.svg")

# ---------------------------------------------------------------------------
# 7. Input demand: text, images, and the 175B job's average
# ---------------------------------------------------------------------------
sp.hbars(out / "demand.svg",
         [(f"Images, {val('img_per_gpu')} per GPU", K.m(K.IMAGE.result.bandwidth, "GB/s"), "danger", f"{val('img_gbs')} GB/s"),
          (f"Text, {val('seq_tokens')} tokens per GPU", K.m(K.TEXT.result.bandwidth, "GB/s"), "data", f"{val('text_mbs')} MB/s"),
          (f"{val('params_b')}B job, {val('run_days')}-day average", K.m(K.FOOT.result.average_read_rate, "GB/s"), "neutral",
           f"{val('avg_read_mbs')} MB/s")],
         "GB/s", log=True, size=(4.1, 1.05), xlabel=f"GB/s for {val('run_gpus')} GPUs (log scale)")

# ---------------------------------------------------------------------------
# 8. Stall: sequential against overlapped loading, 200 ms and 250 ms
# ---------------------------------------------------------------------------
T_IO, T_COMP = K.m(K.STALL.io_time, "ms"), K.m(K.STALL.compute_time, "ms")
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.65))
    lanes = {"Seq. I/O": 3.0, "Seq. GPU": 2.4, "Overlap I/O": 1.0, "Overlap GPU": 0.4}
    for t0 in (0, K.m(K.STALL.result.sequential_step, "ms")):
        ax.broken_barh([(t0, T_IO)], (lanes["Seq. I/O"] - 0.22, 0.44), color=C["data"])
        ax.broken_barh([(t0 + T_IO, T_COMP)], (lanes["Seq. GPU"] - 0.22, 0.44), color=C["memory"])
    ax.text(0, 3.32, f"Sequential: read {val('io_ms')} ms, then compute {val('step_ms')} ms", ha="left", va="bottom",
            fontsize=11.5, fontweight="bold", color=C["text"])
    ax.text(905, 2.7, f"{val('seq_ms')} ms per step\n{val('seq_stall_pct')}% stalled", ha="left", va="center",
            color=C["danger"], fontsize=12, fontweight="bold")
    a = 1.0 if k == 2 else 0.0
    for i in range(4):
        ax.broken_barh([(i * T_IO, T_IO)], (lanes["Overlap I/O"] - 0.22, 0.44),
                       color=C["data"], alpha=a, edgecolor="white")
        if i < 3:
            ax.broken_barh([(T_IO + i * T_IO, T_COMP)], (lanes["Overlap GPU"] - 0.22, 0.44),
                           color=C["memory"], alpha=a)
    ax.text(905, 0.7, f"{val('ovl_ms')} ms per step\n{val('ovl_stall_pct')}% stalled", ha="left", va="center",
            color=C["danger"], fontsize=12, fontweight="bold", alpha=a)
    ax.set_yticks(list(lanes.values()), ["read", "compute", "read", "compute"], fontsize=11.5)
    ax.text(0, 1.32, "Overlapped", ha="left", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["text"], alpha=a)
    ax.set_xlim(0, 900)
    ax.set_ylim(-0.1, 3.75)
    ax.set_xlabel("Time (ms)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    sp.save(fig, out / f"stall_step{k}.svg")

# ---------------------------------------------------------------------------
# 9. Prefetch depth: one 500 ms p99 read spans 2.5 compute windows
# ---------------------------------------------------------------------------
WIN, P99 = K.m(K.PREFETCH.compute_time, "ms"), K.m(K.PREFETCH.tail_latencies["object"], "ms")
fig, ax = plt.subplots(figsize=(5.0, 0.85))
for i in range(K.PREFETCH.result.depths["object"]):
    ax.broken_barh([(i * WIN + 3, WIN - 6)], (0.55, 0.4), color=C["memory"])
    ax.text(i * WIN + WIN / 2, 0.75, f"batch {i + 1}", ha="center", va="center",
            color="white", fontsize=11.5, fontweight="bold")
ax.broken_barh([(0, P99)], (0.0, 0.4), color=C["danger"])
ax.text(P99 / 2, 0.2, f"one slow read: {val('p99_obj_ms')} ms (p99)", ha="center", va="center", color="white",
        fontsize=11.5, fontweight="bold")
ax.text(610, 0.5, f"{val('p99_obj_ms')} / {val('step_ms')} = {val('span_obj')}\nround up: {val('depth_obj')}", ha="left", va="center",
        fontsize=14, fontweight="bold", color=C["text"])
for x in (WIN, 2 * WIN, 3 * WIN):
    ax.axvline(x, color=C["grid"], lw=1, zorder=0)
ax.set_xlim(0, 860)
ax.set_ylim(-0.1, 1.05)
ax.set_yticks([])
ax.set_xticks([0, WIN, 2 * WIN, 3 * WIN])
ax.set_xlabel(f"Time (ms); one compute window = {val('step_ms')} ms")
ax.spines["left"].set_visible(False)
sp.save(fig, out / "prefetch.svg")

# ---------------------------------------------------------------------------
# 10. GPUDirect Storage: two hops through host DRAM against one direct DMA
# ---------------------------------------------------------------------------
s = Svg(900, 290)
s.text(10, 24, "(a) Through host DRAM", 22, V["text"], "bold", anchor="start")
s.text(470, 24, "(b) GPUDirect Storage", 22, V["text"], "bold", anchor="start")
# (a)
box(s, 10, 42, 120, 60, "fr", "CPU")
s.rect(165, 36, 270, 100, V["c1_fill"], V["c1"])
s.text(300, 60, "Host DRAM", 21, V["c1_text"], "bold")
box(s, 178, 74, 115, 48, "fr", "kernel", tsize=20)
box(s, 308, 74, 115, 48, "fr", "pinned", tsize=20)
s.line(130, 72, 172, 98, V["mute"], 3)
s.text(110, 132, "copy", 20, V["mute"], "bold")
box(s, 10, 222, 150, 60, "st", "NVMe SSD")
box(s, 285, 222, 150, 60, "c1", "GPU HBM")
s.poly([(85, 222), (85, 180), (235, 180), (235, 128)], V["st"], 4, arrow=True)
s.text(150, 172, "DMA 1", 20, V["st_text"], "bold")
s.poly([(365, 122), (365, 216)], V["st"], 4, arrow=True)
s.text(382, 180, "DMA 2", 20, V["st_text"], "bold", anchor="start")
s.line(452, 36, 452, 285, V["fr"], 2, dash="6,6")
# (b)
box(s, 470, 42, 120, 60, "fr", "CPU")
box(s, 625, 36, 270, 100, "c1", "Host DRAM", "not on the path", ssize=20)
box(s, 470, 222, 150, 60, "st", "NVMe SSD")
box(s, 745, 222, 150, 60, "c1", "GPU HBM")
s.line(530, 102, 530, 216, V["mute"], 3, dash="7,6", arrow=True)
s.text(540, 165, "control", 20, V["mute"], "bold", anchor="start")
s.poly([(585, 222), (585, 190), (820, 190), (820, 216)], V["st"], 4, arrow=True)
s.text(702, 180, "one DMA", 20, V["st_text"], "bold")
s.save(out / "gds.svg")

# ---------------------------------------------------------------------------
# 11. Parallel file system: one metadata server, many data servers
# ---------------------------------------------------------------------------
s = Svg(900, 320)
s.text(105, 24, "GPU nodes", 21, V["mute"], "bold")
s.text(480, 24, "Servers", 21, V["mute"], "bold")
for i, y in enumerate((120, 190, 262)):
    box(s, 15, y, 180, 52, "c1", f"Client {['1', '2', 'M'][i]}", tsize=21)
box(s, 370, 34, 220, 66, "c3", "Metadata server", "open, stat, lock", ssize=20)
s.poly([(105, 120), (105, 67), (364, 67)], V["c3"], 3, arrow=True)
s.text(232, 58, "1. look up", 20, V["c3_text"], "bold")
s.text(755, 60, "a few hundred", 21, V["c3_text"], "bold")
s.text(755, 86, "thousand ops/s", 21, V["c3_text"], "bold")
for i, y in enumerate((120, 190, 262)):
    box(s, 390, y, 180, 52, "st", f"OSS {['1', '2', val('oss')][i]}: {val('oss_gbs')} GB/s", tsize=20)
    s.line(300, y + 26, 384, y + 26, V["st"], 3, arrow=True)
s.line(300, 146, 300, 288, V["st"], 3)
for y in (146, 216, 288):
    s.line(195, y, 300, y, V["st"], 3)
s.text(248, 112, "2. read", 20, V["st_text"], "bold")
s.text(755, 210, f"{val('oss')} × {val('oss_gbs')} GB/s", 21, V["st_text"], "bold")
s.text(755, 236, f"= {val('pfs_tbs')} TB/s of data", 21, V["st_text"], "bold")
s.save(out / "pfs.svg")

# ---------------------------------------------------------------------------
# 13. Checkpoint bytes: 14 B per parameter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 1.7))
x = 0
for i, (key, txt, ck) in enumerate([("weights_gb", "FP16\nweights", "memory"),
                                    ("master_gb", "FP32 master\nweights", "neutral"),
                                    ("adam_gb", "Adam moments\n(two FP32)", "compute")]):
    w = num(key)
    ax.barh(0, w, left=x, color=C[ck], height=0.7, edgecolor="white", lw=2)
    ax.text(x + w / 2, 0, val(key), ha="center", va="center", color="white", fontsize=13,
            fontweight="bold")
    if i == 0:
        ax.text(0, 0.42, "FP16 weights", ha="left", va="bottom", color=C[ck], fontsize=12,
                fontweight="bold")
    else:
        ax.text(x + w / 2, -0.45, txt, ha="center", va="top", color=C[ck], fontsize=12,
                fontweight="bold", linespacing=1.05)
    x += w
ax.text(2470, 0, f" = {val('ckpt_gb')} GB", va="center", fontsize=14, fontweight="bold",
        color=C["text"])
ax.text(2490, -0.45, f" {val('bpp_fp16')} + {val('bpp_fp32')} + {val('bpp_adam')} bytes\n per parameter", va="top", fontsize=12,
        color=C["text"], linespacing=1.05)
ax.set_xlim(0, 3200)
ax.set_ylim(-1.35, 0.85)
ax.axis("off")
sp.save(fig, out / "ckpt_bytes.svg")

# ---------------------------------------------------------------------------
# 14. Checkpoint storm: every replica writes against write once
# ---------------------------------------------------------------------------
sp.hbars(out / "storm.svg",
         [("Every replica writes its copy", K.m(K.STORM.result.storm_time, "s"), "danger",
           f"{val('storm_s')} s = {val('storm_min')} min"),
          ("Each unique byte written once", K.m(K.STORM.result.single_copy_time, "s"), "data", f"{val('once_s')} s")],
         "s", log=True, size=(3.6, 0.95), xlabel="Seconds to save (log scale)")

# ---------------------------------------------------------------------------
# 15. Tiered checkpoint timeline (after the chapter's figure 8)
# ---------------------------------------------------------------------------
for k in (1, 2, 3):
    s = Svg(900, 330)
    lanes = [("Training", 8), ("Local NVMe", 72), ("File system", 136), ("Coordinator", 200)]
    for name, y in lanes:
        s.text(140, y + 36, name, 21, V["text"], "bold", anchor="end")
    t0, t1, t2, t3, tend = 222, 362, 640, 760, 890
    ticks = [(t0, "t0"), (t1, "t1")] + ([(t2, "t2")] if k >= 2 else []) + \
        ([(t3, "t3")] if k >= 3 else [])
    for x, lab in ticks:
        s.line(x, 8, x, 268, V["bad"] if x in (t0, t1) else V["c3"], 1.5, dash="5,5")
    bh = 54
    box(s, 150, 8, t0 - 150, bh, "c1", "step i", tsize=20)
    box(s, t0, 8, t1 - t0, bh, "bad", f"pause {val('pause_s')} s", tsize=19)
    box(s, t1, 8, tend - t1, bh, "c1", "training continues on live state", tsize=20)
    box(s, t0, 72, t1 - t0, bh, "st", f"{val('shard_node_gb')} GB", tsize=20)
    if k >= 2:
        s.rect(t1, 72, t3 - t1, bh, "#f0fdf4", V["st"])
        s.text((t1 + t3) / 2, 106, "staged: lost if this node fails", 20, V["st_text"])
        box(s, t1, 136, t2 - t1, bh, "st", f"copy, about {val('drain_s')} s", tsize=20, dash="7,5")
        box(s, t2, 136, tend - t2, bh, "st", "durable", tsize=20)
    if k >= 3:
        box(s, t1, 200, t2 - t1, bh, "fr", "wait for every shard", tsize=20)
        box(s, t2, 200, t3 - t2, bh, "c3", "verify", tsize=20)
        box(s, t3, 200, tend - t3, bh, "c3", "manifest", tsize=20)
    s.line(150, 268, tend, 268, V["mute"], 2, arrow=True)
    for x, lab in ticks:
        s.text(x, 293, lab, 22, V["text"], "bold")
    if k >= 3:
        s.text((t1 + t3) / 2, 322, "staged, not yet restorable", 20, V["mute"], "bold")
        s.text((t3 + tend) / 2 + 4, 322, "restorable", 20, V["st_text"], "bold")
    s.save(out / f"staging_step{k}.svg")

# ---------------------------------------------------------------------------
# 16. Versioned artifacts: every stage writes an immutable, named output
# ---------------------------------------------------------------------------
s = Svg(900, 130)
stages = [("raw corpus", f"{val('source_tb')} TB"), ("cleaned", "dedup v4"), ("tokenized", "tok v2"),
          ("shards", f"{val('corpus_tb')} TB")]
for i, (t, sub) in enumerate(stages):
    x = 5 + i * 175
    box(s, x, 15, 145, 96, "st", t, sub, tsize=22, ssize=20)
    s.line(x + 145, 63, x + 172, 63, V["st"], 3, arrow=True)
s.rect(705, 5, 190, 116, V["c3_fill"], V["c3"])
s.text(800, 38, "manifest", 22, V["c3_text"], "bold")
s.text(800, 68, "shard hashes,", 20, V["mute"])
s.text(800, 96, "code, seed", 20, V["mute"])
s.save(out / "versions.svg")

# ---------------------------------------------------------------------------
# 17. The acknowledgment gap: append against keyed write (Example 4.2)
# ---------------------------------------------------------------------------
for k in (1, 2):
    s = Svg(900, 330)
    s.text(150, 40, "Worker", 22, V["text"], "bold", anchor="end")
    box(s, 165, 8, 150, 54, "fr", "read e42", tsize=21)
    s.line(315, 35, 342, 35, V["mute"], 3, arrow=True)
    box(s, 347, 8, 165, 54, "st", "write row", tsize=21)
    s.cross(545, 35, 13, V["bad"])
    box(s, 578, 8, 312, 54, "c3", "record commit: never", tsize=21, dash="7,5")
    s.text(150, 112, "Replacement", 22, V["text"], "bold", anchor="end")
    box(s, 165, 80, 260, 54, "fr", "replay e42: committed?", tsize=21)
    s.text(660, 112, "crash: written, not committed", 21, V["bad_text"], "bold")
    s.rect(10, 150, 430, 175, V["fr_fill"], V["fr"])
    s.text(25, 180, "(a) Append the row", 22, V["text"], "bold", anchor="start")
    s.rect(25, 195, 400, 44, "white", V["fr"])
    s.text(40, 224, "e42, click, first write", 20, anchor="start")
    s.rect(25, 247, 400, 44, V["bad_fill"], V["bad"])
    s.text(40, 276, "e42, click, replay", 20, anchor="start")
    s.text(412, 276, "duplicate", 20, V["bad_text"], "bold", anchor="end")
    s.text(25, 316, "the click counts twice", 20, V["bad_text"], "bold", anchor="start")
    if k == 2:
        s.rect(460, 150, 430, 175, V["fr_fill"], V["fr"])
        s.text(475, 180, "(b) Write under a stable key", 22, V["text"], "bold", anchor="start")
        s.rect(475, 195, 400, 44, V["st_fill"], V["st"])
        s.text(490, 224, "(feature_version, e42), click", 20, anchor="start")
        s.text(475, 270, "the replay overwrites the same row:", 20, V["st_text"], "bold",
               anchor="start")
        s.text(475, 298, "the click counts once", 20, V["st_text"], "bold", anchor="start")
    s.save(out / f"ackgap_step{k}.svg")

# ---------------------------------------------------------------------------
# 18. Economics: a year of storage against ten reads across a boundary
# ---------------------------------------------------------------------------
sp.hbars(out / "egress.svg",
         [(f"Store {val('ds_tb')} TB for a year", K.BILL.result.storage_cost / 1000, "neutral", f"${val('store_year')}"),
          (f"Read it {val('egress_reads')} times across regions", K.BILL.result.egress_cost / 1000, "danger",
           f"${val('egress_all')}")],
         "$", log=False, size=(3.55, 0.8), xlabel="Thousand US dollars")

ep = np.linspace(0, 20, 200)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.5))
    a = 1.0 if k == 2 else 0.0
    ss = K.PLAN.result
    base = ss.source_storage_cost
    per_epoch = ss.stream_egress_per_run / K.PLAN.epochs * K.PLAN.runs_per_year
    ax.plot(ep, (base + per_epoch * ep) / 1000, color=C["danger"], lw=3, alpha=a)  # stream
    ax.plot(ep, np.full_like(ep, ss.stage_cost / 1000), color=C["data"], lw=3, alpha=a)
    ax.text(5.5, 330, "stream every epoch", color=C["danger"], ha="left", fontsize=12.5,
            fontweight="bold", alpha=a)
    ax.text(19.5, 110, "stage once per run", color=C["data"], ha="right", fontsize=12.5,
            fontweight="bold", alpha=a)
    ax.plot([ss.break_even_epochs], [ss.stage_cost / 1000], "o", ms=10, color=C["text"], mec="white", mew=1.5, alpha=a)
    ax.annotate(f"break-even: {val('ss_break_even')} epochs", (ss.break_even_epochs, ss.stage_cost / 1000), xytext=(6, 8), fontsize=12,
                fontweight="bold", color=C["text"], alpha=a,
                arrowprops=dict(arrowstyle="-", color=C["text"], lw=1.2, alpha=a))
    ax.plot([K.PLAN.epochs] * 2, [ss.stage_cost / 1000, ss.stream_cost / 1000], color=C["neutral"], lw=1, ls=":", alpha=a)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 400)
    ax.set_xlabel(f"Epochs per training run ({val('ss_runs')} runs a year)")
    ax.set_ylabel("Thousand $ per year")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"stage_cost_step{k}.svg")

# ---------------------------------------------------------------------------
# 19. Serving: time to load 350 GB of FP16 weights into one replica
# ---------------------------------------------------------------------------
sp.hbars(out / "coldstart.svg",
         [(lab, K.m(K.LOAD.result.times[link], "s"), ck, f"{val(key)} s")
          for lab, link, key, ck in (
              (f"Object store, {val('load_obj_gbs')} GB/s", "object store", "load_obj_s", "danger"),
              (f"File system share, {val('load_pfs_gbs')} GB/s", "file system share", "load_pfs_s", "compute"),
              (f"One NVMe, {val('gen5_gbs')} GB/s", "one Gen5 NVMe", "load_gen5_s", "data"),
              (f"Eight NVMe, {val('gen4_gbs')} GB/s each", "eight Gen4 NVMe", "load_stripe_s", "data"),
              (f"Host DRAM, {val('dram_gbs')} GB/s", "host DRAM", "load_dram_s", "memory"))],
         "s", log=True, size=(4.05, 1.55), xlabel="Seconds to load (log scale)")

# ---------------------------------------------------------------------------
# 20. The 3FS read trace, cropped to its plot area for slide-size labels
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    im = Image.open(out / "3fs-read-throughput.jpg")
    w, hgt = im.size
    # plot area of the README figure: x 5.4% to 96.3%, y 13.5% to 85.5%
    from PIL import ImageDraw
    cr = im.crop((int(0.054 * w), int(0.135 * hgt), int(0.963 * w), int(0.855 * hgt)))
    cw, ch = cr.size
    # white out the small legend box; the slide labels the line at slide size
    ImageDraw.Draw(cr).rectangle((int(0.86 * cw), int(0.84 * ch), cw, ch), fill="white")
    cr.save(out / "3fs_trace_crop.png")
except FileNotFoundError:
    pass

print("figures written to", out)

# ---------------------------------------------------------------------------
# Hedged read on a timeline (frame "A hedged read pays only when its trigger
# sits in the tail"): the original read lands on a slow server; at the p99
# delay a second read goes to a replica, and the first response wins.
# Schematic, no measured times. Added 2026-10-21.
# ---------------------------------------------------------------------------
s = Svg(900, 220)
s.line(190, 182, 880, 182, V["mute"], 2, arrow=True)
s.text(880, 210, "time", 22, V["mute"], anchor="end")
s.text(175, 62, "First read", 24, V["text"], "bold", anchor="end")
s.rect(195, 36, 640, 44, V["bad_fill"], V["bad"])
s.text(634, 66, "server A: slow, in the tail", 22, V["bad_text"], "bold")
s.line(430, 24, 430, 182, V["c1"], 2.5, dash="7 5")
s.text(436, 210, "p99 delay", 22, V["c1"], "bold", anchor="middle")
s.text(175, 132, "Second read", 24, V["text"], "bold", anchor="end")
s.rect(432, 106, 180, 44, V["st_fill"], V["st"])
s.text(522, 136, "replica B", 22, V["st_text"], "bold")
s.text(626, 136, "first response wins", 24, V["st_text"], "bold", anchor="start")
s.save(out / "hedge_timeline.svg")
