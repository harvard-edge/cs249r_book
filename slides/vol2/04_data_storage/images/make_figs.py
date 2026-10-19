#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 4 (Data Storage).

Every value is from the rendered chapter (Data Storage) unless noted:
  - Running job: 175B parameters, 2,048 H100s in 256 nodes of 8, 30 days,
    6 TB of 4-byte token IDs per epoch, one epoch (Sec. 4.1).
  - Checkpoint contract: 14 B per parameter: FP16 weights 350 GB, FP32
    master 700 GB, Adam moments 1,400 GB; 2,450 GB every 10 minutes;
    4,320 saves, 10.6 PB over 30 days; 1,764x one epoch (Sec. 4.1, 4.6.2,
    table 5).
  - Tier ladder: HBM 3.35 TB/s, host DRAM about 200 GB/s, four NVMe in
    RAID 0 28 GB/s, a node's share of a 1 TB/s file system over 256 nodes
    about 3.9 GB/s (Sec. 4.3.1 figure; Napkin Math 4.7).
  - Tier table (capacity, bandwidth, latency, cost basis): table 2.
  - Cache mean time E[T] = h T_hot + (1 - h) T_miss: 95 percent hits with a
    100x slower miss give 5.95 T_hot (Sec. 4.3.1, equation 1); the 35 TB
    job on 30.7 TB of NVMe: 88 percent hits at 28 GB/s, misses at 4 GB/s,
    about 16 GB/s, 58 percent (Sec. 4.4.3). Curves are derived from eq. 1.
  - I/O wall: delivery takes a quarter of the original compute time and
    overlaps, so throughput is min(s, 4) for an s-times faster accelerator
    (Systems Perspective 4.1). Curve derived.
  - MLPerf Storage v1.0 closed division, passing results (124 rows): read
    throughput and emulated accelerators, from MLSysIM
    Literature.Series.MLPerfStorageV1Training (MLCommons 2024), as the
    chapter's figure 3 uses them.
  - Input demand: 167.8 MB/s text, 303.2 GB/s images, 2,048 GPUs, 200 ms
    (Napkin Math 4.1); the 175B job averages about 2.3 MB/s (Sec. 4.3.1).
  - Stall: 200 ms compute, 250 ms delivery: 450 ms and 55.6 percent
    sequential, 250 ms and 20 percent overlapped (Napkin Math 4.5).
  - Prefetch: 500 ms p99 read against 200 ms compute: ceil(2.5) = 3 ready
    batches (Sec. 4.4.1, equation 5).
  - GDS: 64,000 images/s per node at 120 us against 30 us of CPU per
    request: 7.68 against 1.92 CPU-seconds per second (Napkin Math 4.6,
    illustrative service times).
  - PFS: 100 OSS at 10 GB/s = 1 TB/s; an MDS serves a few hundred thousand
    operations per second (Sec. 4.3.6). Meltdown: 200 million files on a
    500 GB/s file system deliver under 1 percent; 50,000 tar shards, 4,000x
    fewer files (Example 4.1). Running corpus: about 2,000 shards of 3 GB.
  - Tail: 100 servers, 1 percent slow each: 0.99^100 = 0.366 (Sec. 4.3.6);
    curve 1 - 0.99^n derived.
  - Storm: 70B model, 980 GB per replica, 1,024 replicas, 100 GB/s:
    10,035 s (167.3 min) against 9.8 s (Definition 4.5).
  - Staging: 9.6 GB per node, 0.34 s at 28 GB/s, about 2.4 s drain to a
    1 TB/s file system (Napkin Math 4.7); timeline after figure 8.
  - Commit boundary: the e42 trace (Example 4.2, figure 9).
  - Synthetic data: 1 TB payload, 40 percent provenance, 3 verifier reads:
    1.4 TB at rest, 4.4 TB moved (Napkin Math 4.8).
  - Economics: 100 TB: $24,000 a year in object storage; $9,000 egress per
    read at $0.09/GB; ten epochs $90,000 (Sec. 4.8). Stream against stage,
    50 TB, 4 runs a year: stream 12,000 + 18,000 x epochs per run; stage
    $90,000 (Napkin Math 4.9); crossover 4.3 epochs (derived).
  - Serving: 350 GB of FP16 weights: object store 1 GB/s 350 s, file system
    share 4 GB/s 87.5 s, one NVMe at 14 GB/s 25 s, eight NVMe at 7 GB/s
    6.2 s, host DRAM shared at 200 GB/s 1.8 s (Sec. 4.9).
  - 3FS trace: DeepSeek 3FS README (MIT license), read stress test on 180
    storage nodes, about 6.6 TiB/s aggregate; the plot is cropped and
    relabeled at slide size on the slide.

Run from this folder: python3 make_figs.py
"""

import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (every derived value drawn below)

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
         [("Read: one epoch of tokens", K.CORPUS / K.TB, "data", f"{K.CORPUS / K.TB:.0f} TB"),
          ("Write: checkpoints, 30 days", K.CKPT_TOTAL / K.TB, "memory",
           f"{K.CKPT_TOTAL / K.PB:.1f} PB")],
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
    ax.plot(shown, np.minimum(shown, 4), color=C["danger"], lw=3, marker="o", ms=9)
    a = 1.0 if k == 2 else 0.0
    ax.axhline(4, color=C["data"], lw=1.5, ls=":", alpha=a)
    ax.text(17, 3.3, "plateau: 4× at 16× speed", color=C["danger"],
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
rows = [r for r in Literature.Series.MLPerfStorageV1Training.entries
        if r["division"] == "Closed" and r["au_check"] == "pass"]
WL = [("unet3d", "3D U-Net", "memory"), ("cosmoflow", "CosmoFlow", "compute"),
      ("resnet50", "ResNet-50", "data")]
med = {(w, a): statistics.median(r["io_GBps"] / r["num_accelerators"] for r in rows
                                 if r["workload"] == w and r["accelerator"] == a)
       for w, _, _ in WL for a in ("A100", "H100")}

fig, ax = plt.subplots(figsize=(3.9, 1.95))
for w, name, ck in WL:
    for acc, mk in (("A100", "o"), ("H100", "^")):
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

for (w, name, _ck) in WL:
    print(f"MLPerf median {name}: A100 {med[(w, 'A100')]:.2f}, H100 {med[(w, 'H100')]:.2f} GB/s")

# ---------------------------------------------------------------------------
# 4. The six tiers with their scope, bandwidth, latency, and price
# ---------------------------------------------------------------------------
tiers = [("GPU HBM", "c1", "80 GB", "3.35 TB/s per GPU", "~100 ns"),
         ("Host DRAM", "c1", "0.5 to 2 TB", "200 GB/s per node", "~100 ns"),
         ("Local NVMe", "st", "4 to 30 TB", "7 to 25 GB/s", "~100 µs"),
         ("Parallel file system", "st", "100+ PB", "1+ TB/s shared", "~1 ms"),
         ("Object storage", "st", "very large", "100 GB/s shared", "~50 ms"),
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
lad = [("GPU HBM (one GPU)", 3350, "memory", "3.35 TB/s"),
       ("Host DRAM (node)", 200, "memory", "200 GB/s"),
       ("4 NVMe, RAID 0 (node)", 28, "data", "28 GB/s"),
       ("File system share (node)", K.PFS_SHARE / K.GB, "data", f"{K.PFS_SHARE / K.GB:.1f} GB/s")]
n = len(lad)
for i, (lab, v, ck, txt) in enumerate(lad):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=12.5, fontweight="bold", color=C["text"])
    ax.text(v * 1.2, y, txt, va="center", fontsize=12.5, fontweight="bold", color=C[ck])
for i, r in enumerate([f"÷ {v:.1f}" for v in (K.R_HBM_DRAM, K.R_DRAM_NVME, K.R_NVME_PFS)]):
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
for kmiss, ck, lab, ly in ((7, "data", "miss 7× slower (NVMe to file system)", 0.80),
                           (100, "danger", "miss 100× slower", 0.03)):
    ax.plot(h * 100, 1 / (h + (1 - h) * kmiss) * 100, color=C[ck], lw=3)
ax.plot([K.H1 * 100], [K.FRAC1 * 100], "o", ms=9, color=C["danger"], mec="white", mew=1.5)
ax.annotate(f"{K.H1 * 100:.0f}% hits, misses {K.MISS1}× slower: {K.FRAC1 * 100:.0f}%",
            (K.H1 * 100, K.FRAC1 * 100), xytext=(53, 10), ha="left", fontsize=11.5,
            fontweight="bold", color=C["danger"],
            arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1.2))
ax.plot([K.H2 * 100], [K.FRAC2 * 100], "o", ms=9, color=C["data"], mec="white", mew=1.5)
ax.annotate(f"{K.H2 * 100:.0f}% hits, misses {K.MISS2:.0f}× slower: {K.FRAC2 * 100:.0f}%",
            (K.H2 * 100, K.FRAC2 * 100), xytext=(51, 86), ha="left", fontsize=11.5,
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
         [("Images, 256 per GPU", K.IMG_BW / K.GB, "danger", f"{K.IMG_BW / K.GB:.0f} GB/s"),
          ("Text, 4,096 tokens per GPU", K.TEXT_BW / K.GB, "data", f"{K.TEXT_BW / 1e6:.0f} MB/s"),
          ("175B job, 30-day average", K.JOB_AVG_BW / K.GB, "neutral",
           f"{K.JOB_AVG_BW / 1e6:.1f} MB/s")],
         "GB/s", log=True, size=(4.1, 1.05), xlabel="GB/s for 2,048 GPUs (log scale)")

# ---------------------------------------------------------------------------
# 8. Stall: sequential against overlapped loading, 200 ms and 250 ms
# ---------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.65))
    lanes = {"Seq. I/O": 3.0, "Seq. GPU": 2.4, "Overlap I/O": 1.0, "Overlap GPU": 0.4}
    for t0 in (0, 450):
        ax.broken_barh([(t0, 250)], (lanes["Seq. I/O"] - 0.22, 0.44), color=C["data"])
        ax.broken_barh([(t0 + 250, 200)], (lanes["Seq. GPU"] - 0.22, 0.44), color=C["memory"])
    ax.text(0, 3.32, "Sequential: read 250 ms, then compute 200 ms", ha="left", va="bottom",
            fontsize=11.5, fontweight="bold", color=C["text"])
    ax.text(905, 2.7, f"{K.SEQ_STEP} ms per step\n{K.SEQ_STALL * 100:.1f}% stalled", ha="left", va="center",
            color=C["danger"], fontsize=12, fontweight="bold")
    a = 1.0 if k == 2 else 0.0
    for i in range(4):
        ax.broken_barh([(i * 250, 250)], (lanes["Overlap I/O"] - 0.22, 0.44),
                       color=C["data"], alpha=a, edgecolor="white")
        if i < 3:
            ax.broken_barh([(250 + i * 250, 200)], (lanes["Overlap GPU"] - 0.22, 0.44),
                           color=C["memory"], alpha=a)
    ax.text(905, 0.7, f"{K.OVL_STEP} ms per step\n{K.OVL_STALL * 100:.0f}% stalled", ha="left", va="center",
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
fig, ax = plt.subplots(figsize=(5.0, 0.85))
for i in range(3):
    ax.broken_barh([(i * 200 + 3, 194)], (0.55, 0.4), color=C["memory"])
    ax.text(i * 200 + 100, 0.75, f"batch {i + 1}", ha="center", va="center",
            color="white", fontsize=11.5, fontweight="bold")
ax.broken_barh([(0, 500)], (0.0, 0.4), color=C["danger"])
ax.text(250, 0.2, "one slow read: 500 ms (p99)", ha="center", va="center", color="white",
        fontsize=11.5, fontweight="bold")
ax.text(610, 0.5, f"{K.P99_OBJ} / {K.T_COMP} = {K.PREFETCH_SPAN:.1f}\nround up: {K.PREFETCH_DEPTH}", ha="left", va="center",
        fontsize=14, fontweight="bold", color=C["text"])
for x in (200, 400, 600):
    ax.axvline(x, color=C["grid"], lw=1, zorder=0)
ax.set_xlim(0, 860)
ax.set_ylim(-0.1, 1.05)
ax.set_yticks([])
ax.set_xticks([0, 200, 400, 600])
ax.set_xlabel("Time (ms); one compute window = 200 ms")
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
    box(s, 390, y, 180, 52, "st", f"OSS {['1', '2', '100'][i]}: 10 GB/s", tsize=20)
    s.line(300, y + 26, 384, y + 26, V["st"], 3, arrow=True)
s.line(300, 146, 300, 288, V["st"], 3)
for y in (146, 216, 288):
    s.line(195, y, 300, y, V["st"], 3)
s.text(248, 112, "2. read", 20, V["st_text"], "bold")
s.text(755, 210, "100 × 10 GB/s", 21, V["st_text"], "bold")
s.text(755, 236, "= 1 TB/s of data", 21, V["st_text"], "bold")
s.save(out / "pfs.svg")

# ---------------------------------------------------------------------------
# 12. Tail at scale: probability that at least one of n servers is slow
# ---------------------------------------------------------------------------
nn = np.arange(1, 1001)
fig, ax = plt.subplots(figsize=(4.8, 1.5))
ax.plot(nn, (1 - 0.99 ** nn) * 100, color=C["danger"], lw=3)
ax.plot([100], [K.P_SOME_SLOW * 100], "o", ms=10, color=C["danger"], mec="white", mew=1.5)
ax.annotate(f"100 servers: {K.P_SOME_SLOW * 100:.0f}% of steps\nwait on a slow one", (100, K.P_SOME_SLOW * 100), xytext=(140, 25),
            fontsize=12, fontweight="bold", color=C["danger"],
            arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1.2))
ax.set_xscale("log")
ax.set_xlim(1, 1000)
ax.set_ylim(0, 105)
ax.set_xlabel("Servers one step reads from (each slow 1% of the time)")
ax.set_ylabel("% of steps")
ax.grid(color=C["grid"], lw=0.6)
ax.minorticks_off()
sp.save(fig, out / "tail.svg")

# ---------------------------------------------------------------------------
# 13. Checkpoint bytes: 14 B per parameter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 1.7))
x = 0
for w, num, txt, ck in [(350, "350", "FP16\nweights", "memory"),
                        (700, "700", "FP32 master\nweights", "neutral"),
                        (1400, "1,400", "Adam moments\n(two FP32)", "compute")]:
    ax.barh(0, w, left=x, color=C[ck], height=0.7, edgecolor="white", lw=2)
    ax.text(x + w / 2, 0, num, ha="center", va="center", color="white", fontsize=13,
            fontweight="bold")
    if w == 350:
        ax.text(0, 0.42, "FP16 weights", ha="left", va="bottom", color=C[ck], fontsize=12,
                fontweight="bold")
    else:
        ax.text(x + w / 2, -0.45, txt, ha="center", va="top", color=C[ck], fontsize=12,
                fontweight="bold", linespacing=1.05)
    x += w
ax.text(2470, 0, " = 2,450 GB", va="center", fontsize=14, fontweight="bold",
        color=C["text"])
ax.text(2490, -0.45, " 2 + 4 + 8 bytes\n per parameter", va="top", fontsize=12,
        color=C["text"], linespacing=1.05)
ax.set_xlim(0, 3200)
ax.set_ylim(-1.35, 0.85)
ax.axis("off")
sp.save(fig, out / "ckpt_bytes.svg")

# ---------------------------------------------------------------------------
# 14. Checkpoint storm: every replica writes against write once
# ---------------------------------------------------------------------------
sp.hbars(out / "storm.svg",
         [("Every replica writes its copy", K.STORM_S, "danger",
           f"{K.STORM_S:,.0f} s = {K.STORM_S / 60:.0f} min"),
          ("Each unique byte written once", K.ONCE_S, "data", f"{K.ONCE_S:.1f} s")],
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
    box(s, t0, 8, t1 - t0, bh, "bad", f"pause {K.LOCAL_WRITE_S:.2f} s", tsize=19)
    box(s, t1, 8, tend - t1, bh, "c1", "training continues on live state", tsize=20)
    box(s, t0, 72, t1 - t0, bh, "st", f"{K.PER_NODE_CKPT / K.GB:.1f} GB", tsize=20)
    if k >= 2:
        s.rect(t1, 72, t3 - t1, bh, "#f0fdf4", V["st"])
        s.text((t1 + t3) / 2, 106, "staged: lost if this node fails", 20, V["st_text"])
        box(s, t1, 136, t2 - t1, bh, "st", f"copy, about {K.DRAIN_S:.2f} s", tsize=20, dash="7,5")
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
stages = [("raw corpus", "3 TB"), ("cleaned", "dedup v4"), ("tokenized", "tok v2"),
          ("shards", "6 TB")]
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
         [("Store 100 TB for a year", K.STORE_YEAR / 1000, "neutral", f"${K.STORE_YEAR:,.0f}"),
          ("Read it 10 times across regions", K.EGRESS_TEN / 1000, "danger",
           f"${K.EGRESS_TEN:,.0f}")],
         "$", log=False, size=(3.55, 0.8), xlabel="Thousand US dollars")

ep = np.linspace(0, 20, 200)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.5))
    a = 1.0 if k == 2 else 0.0
    base = K.D50 * K.OBJ_RATE * 12
    per_epoch = K.D50 * K.EGRESS * K.RUNS
    ax.plot(ep, (base + per_epoch * ep) / 1000, color=C["danger"], lw=3, alpha=a)  # stream
    ax.plot(ep, np.full_like(ep, K.STAGE / 1000), color=C["data"], lw=3, alpha=a)
    ax.text(5.5, 330, "stream every epoch", color=C["danger"], ha="left", fontsize=12.5,
            fontweight="bold", alpha=a)
    ax.text(19.5, 110, "stage once per run", color=C["data"], ha="right", fontsize=12.5,
            fontweight="bold", alpha=a)
    ax.plot([K.BREAK_EVEN], [K.STAGE / 1000], "o", ms=10, color=C["text"], mec="white", mew=1.5, alpha=a)
    ax.annotate(f"break-even: {K.BREAK_EVEN:.1f} epochs", (K.BREAK_EVEN, K.STAGE / 1000), xytext=(6, 8), fontsize=12,
                fontweight="bold", color=C["text"], alpha=a,
                arrowprops=dict(arrowstyle="-", color=C["text"], lw=1.2, alpha=a))
    ax.plot([20, 20], [K.STAGE / 1000, K.STREAM / 1000], color=C["neutral"], lw=1, ls=":", alpha=a)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Epochs per training run (4 runs a year)")
    ax.set_ylabel("Thousand $ per year")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"stage_cost_step{k}.svg")

# ---------------------------------------------------------------------------
# 19. Serving: time to load 350 GB of FP16 weights into one replica
# ---------------------------------------------------------------------------
sp.hbars(out / "coldstart.svg",
         [(lab, v, ck, f"{v:.1f} s" if v < 10 else f"{v:g} s")
          for lab, v, ck in (("Object store, 1 GB/s", K.LOAD["object store 1 GB/s"], "danger"),
                             ("File system share, 4 GB/s", K.LOAD["file system share 4 GB/s"], "compute"),
                             ("One NVMe, 14 GB/s", K.LOAD["one NVMe 14 GB/s"], "data"),
                             ("Eight NVMe, 7 GB/s each", K.LOAD["eight NVMe 7 GB/s"], "data"),
                             ("Host DRAM, 200 GB/s", K.LOAD["host DRAM 200 GB/s"], "memory"))],
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
