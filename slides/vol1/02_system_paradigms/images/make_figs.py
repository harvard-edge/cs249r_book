#!/usr/bin/env python3
"""Data plots for Lecture 2 (System Paradigms).

Every value is sourced to the rendered chapter (System Paradigms) unless
noted:
  - Power tiers: cloud 3 MW, edge 200 W, mobile 5 W, TinyML 1 mW (Sec. 2.1).
    Memory per example platform: TPU v4 pod 128 TB HBM, DGX Spark 128 GB,
    phone 8-16 GB, ESP32-S3 512 KB SRAM (table 4).
  - Round trip in fiber at 200,000 km/s: 1,500 km gives 15 ms (Napkin Math
    2.2), 3,600 km about 36 ms (Sec. 2.2). Budget 10 ms.
  - Camera bandwidth: 100 x 186.6 MB/s = 18.7 GB/s raw; 10 Gb/s link =
    1.25 GB/s; defect metadata 1 KB x 20 events/s = 20 KB/s (Napkin Math 2.4).
  - Memory wall: compute doubling every 18 months (about 1.6x a year)
    against bandwidth at about 1.2x a year; about 16x apart after ten years
    (Sec. 2.2, illustrative rates).
  - A100 roofline: 312 TFLOP/s FP16, 2.04 TB/s, ridge 153 FLOP/byte;
    ResNet-50 at batch 1: 8.2 GFLOP over 56.3 MB = 146 FLOP/byte (Napkin
    Math 2.1). Batch-1 decode with 16-bit weights: about 1 FLOP/byte (Model
    Serving, Sec. 13.9, as in Lecture 0). Batch B: about B FLOP/byte if
    weight traffic dominates (derived; KV cache and activations ignored).
  - Thermal trace: 12 W model on a 3 W chassis, about 1 C/s, trip at 80 C
    within 60 s, 100 FPS falls to 30 FPS (Napkin Math 2.9, illustrative).
  - Voice assistant: 1B devices x 32 KB/s = 32 TB/s if every microphone
    streams; on-device wake word cuts audio leaving each device about 860x,
    so about 37 GB/s (derived) (Napkin Math 2.10).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
C = sp.C

# ---------------------------------------------------------------------------
# 1. Power ladder with each tier's memory (four steps)
# ---------------------------------------------------------------------------
sp.hbars_steps(out / "power_tiers", [
    ("Cloud", 3e6, "neutral", "3 MW   ·  128 TB"),
    ("Edge", 200, "neutral", "200 W  ·  128 GB"),
    ("Mobile", 5, "neutral", "5 W  ·  8–16 GB"),
    ("TinyML", 1e-3, "neutral", "1 mW  ·  512 KB"),
], "W", xlabel="Power (watts, log scale)   ·   memory of the example platform")


# ---------------------------------------------------------------------------
# 2. Round trip in fiber against a 10 ms safety budget (two steps)
# ---------------------------------------------------------------------------
def rtt(step):
    rows = [("Regional cloud, 1,500 km", 15.0), ("Cross-country, 3,600 km", 36.0)]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.1))
    for i, (label, v) in enumerate(rows):
        y = len(rows) - 1 - i
        shown = i < step
        ax.barh(y, v if shown else 0, color=C["danger"], height=0.55)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
        if shown:
            ax.text(v + 0.6, y, f"{v:.0f} ms", va="center", fontsize=13,
                    fontweight="bold", color=C["danger"])
    ax.axvline(10, color=C["data"], lw=3)
    ax.text(10.5, 1.55, "Safety budget 10 ms", color=C["data"], fontsize=12,
            fontweight="bold", va="center")
    ax.set_xlim(0, 44)
    ax.set_ylim(-0.6, 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Round trip in fiber, propagation only (ms)")
    sp.save(fig, out / f"rtt_distance_step{step}.svg")


for k in (1, 2):
    rtt(k)

# ---------------------------------------------------------------------------
# 3. Factory cameras: raw video against the link (three steps)
# ---------------------------------------------------------------------------
sp.hbars_steps(out / "camera_bandwidth", [
    ("100 raw camera feeds", 18.7e9, "danger", "18.7 GB/s"),
    ("10 Gb/s factory link", 1.25e9, "neutral", "1.25 GB/s: 15× too small"),
    ("Defect metadata only", 20e3, "data", "20 KB/s"),
], "B/s", xlabel="Data rate (bytes per second, log scale)")

# ---------------------------------------------------------------------------
# 4. Memory wall: compute against bandwidth growth over a decade
# ---------------------------------------------------------------------------
yrs = np.linspace(0, 10, 101)
comp = 2 ** (yrs / 1.5)
bw = 1.2 ** yrs
fig, ax = sp.plt.subplots(figsize=(5.8, 2.6))
ax.plot(yrs, comp, color=C["compute"], lw=3.5)
ax.plot(yrs, bw, color=C["memory"], lw=3.5)
ax.fill_between(yrs, bw, comp, color=C["danger_l"], alpha=0.7)
ax.text(10.2, comp[-1], "Compute\n×102", color=C["compute"], fontsize=12,
        fontweight="bold", va="center")
ax.text(10.2, bw[-1], "Memory\nbandwidth\n×6.2", color=C["memory"], fontsize=12,
        fontweight="bold", va="center")
ax.text(6.0, 7.5, "Gap ≈ 16×", color=C["danger"], fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_xlim(0, 10)
ax.set_ylim(0.8, 200)
ax.set_xlabel("Years")
ax.set_ylabel("Growth (log)")
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1×", "10×", "100×"])
ax.text(0.0, -0.38, "Illustrative rates: compute doubles every 18 months, bandwidth +20% a year",
        transform=ax.transAxes, fontsize=9, color=C["neutral"])
sp.save(fig, out / "memory_wall.svg")

# ---------------------------------------------------------------------------
# 5. The A100 roofline, built up, then workloads placed on it
# ---------------------------------------------------------------------------
PEAK, BW = 312.0, 2.04       # TFLOP/s, TB/s
RIDGE = PEAK / BW            # about 153 FLOP/byte


def roofline(name, parts, points=(), peak2=False, arrow=False, qmark=False):
    xs = np.logspace(-1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    slope = xs * BW
    if "slope" in parts:
        m = xs <= (RIDGE if "ceiling" in parts else 1e4)
        ax.plot(xs[m], slope[m], color=C["memory"], lw=3.5)
        ax.text(0.13, 60, "Memory slope:\nintensity × 2.04 TB/s",
                color=C["memory"], fontsize=11, fontweight="bold", ha="left",
                va="top")
    if "ceiling" in parts:
        m = xs >= RIDGE
        ax.plot(xs[m], [PEAK] * m.sum(), color=C["compute"], lw=3.5)
        ax.text(1300, PEAK * 0.7, "Compute ceiling:\n312 TFLOP/s", color=C["compute"],
                fontsize=11, fontweight="bold", ha="center", va="top")
    if "ridge" in parts:
        ax.axvline(RIDGE, color=C["neutral"], lw=1.2, ls="--")
        ax.text(RIDGE * 1.1, 0.25, "Ridge\n153 FLOP/byte", color=C["neutral"],
                fontsize=10, va="bottom")
    if peak2:
        r2 = 2 * PEAK / BW
        m = xs >= r2
        ax.plot(xs[xs <= r2], xs[xs <= r2] * BW, color=C["memory"], lw=1.5, ls=":")
        ax.plot(xs[m], [2 * PEAK] * m.sum(), color=C["compute"], lw=2, ls=":")
        ax.text(3000, 2 * PEAK * 1.25, "A chip with 2× FLOP/s", color=C["compute"], fontsize=10,
                ha="center", va="bottom")
    for label, x, ck, dx, dy, ha in points:
        y = min(PEAK, x * BW)
        ax.plot([x], [y], "o", ms=11, color=C[ck], mec="white", mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=11, fontweight="bold", color=C[ck])
    if arrow:
        ax.annotate("", xy=(256, PEAK), xytext=(1.3, 1.0 * BW * 1.3),
                    arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=2))
        ax.text(0.13, 1200, "Batching: batch B reuses\neach weight B times", color=C["neutral"],
                fontsize=10, ha="left", va="center", style="italic")
    if qmark:
        ax.text(1.0, 6, "???", color=C["danger"], fontsize=18, fontweight="bold",
                ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.15, 3000)
    ax.minorticks_off()
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.set_xlabel("Arithmetic intensity (FLOP per byte moved, log scale)")
    ax.set_ylabel("TFLOP/s (log)")
    sp.save(fig, out / f"{name}.svg")


roofline("roofline_step1", ["slope"])
roofline("roofline_step2", ["slope", "ceiling"])
roofline("roofline_step3", ["slope", "ceiling", "ridge"])
RESNET = ("ResNet-50, batch 1: 146", 146.0, "memory", -12, -4, "right")
roofline("roofline_resnet", ["slope", "ceiling", "ridge"], points=[RESNET])
DECODE = ("Text generation,\nbatch 1: ≈1", 1.0, "danger", 8, -30, "left")
roofline("roofline_decode_q", ["slope", "ceiling", "ridge"], peak2=True, qmark=True)
roofline("roofline_decode_a", ["slope", "ceiling", "ridge"], points=[DECODE],
         peak2=True, arrow=True)


# ---------------------------------------------------------------------------
# 6. Thermal wall: burst then throttle (two steps)
# ---------------------------------------------------------------------------
def thermal(step):
    t = np.linspace(0, 120, 241)
    fps = np.where(t < 60, 100, 30)
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.4))
    m = t < 60 if step == 1 else t >= 0
    ax.plot(t[m], fps[m], color=C["danger"] if step == 2 else C["compute"], lw=3.5)
    ax.text(28, 106, "Burst: 100 FPS at 12 W", color=C["compute"], fontsize=12,
            fontweight="bold", ha="center")
    ax.axvline(60, color=C["neutral"], lw=1.2, ls="--")
    ax.text(61, 70, "Chip reaches 80 °C\n(about 1 °C per second)", color=C["neutral"],
            fontsize=10, va="center")
    if step == 2:
        ax.text(92, 38, "Sustained: 30 FPS at ≈3 W", color=C["danger"], fontsize=12,
                fontweight="bold", ha="center")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 125)
    ax.set_xlabel("Seconds of continuous use")
    ax.set_ylabel("Frames per second")
    ax.text(0.0, -0.4, "Illustrative trace (Napkin Math 2.9)", transform=ax.transAxes,
            fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"thermal_wall_step{step}.svg")


for k in (1, 2):
    thermal(k)

# ---------------------------------------------------------------------------
# 7. Voice assistant: audio leaving one billion devices (two steps)
# ---------------------------------------------------------------------------
sp.hbars_steps(out / "voice_upload", [
    ("Every microphone streams", 32e12, "danger", "32 TB/s"),
    ("Wake word on the device", 32e12 / 860, "data", "≈37 GB/s: 860× less"),
], "B/s", xlabel="Audio sent to the cloud (bytes per second, log scale)",
    size=sp.SHORT)

# ---------------------------------------------------------------------------
# 8. Fifty years of microprocessor trends, redrawn at column size from the
#    same series the book's figure plots (MLSysIM Literature registry:
#    Karl Rupp, Microprocessor Trend Data). Direct labels replace the legend.
# ---------------------------------------------------------------------------
def trends():
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
    from mlsysim import Literature
    rows = Literature.Series.MicroprocessorTrends.entries
    series = [  # key, label, color, marker, label y at the right edge
        ("transistors", "Transistors (K)", C["danger"], "o", 3e7),
        ("specint", "Single-thread\nperf. (SPECint ×10³)", C["memory"], "s", 1.2e5),
        ("frequency", "Clock (MHz)", C["data"], "^", 3.5e3),
        ("watts", "Power (W)", C["compute"], "D", 2.5e2),
        ("cores", "Cores", C["neutral"], "v", 1.6e1),
    ]
    fig, ax = sp.plt.subplots(figsize=(3.3, 2.45))
    ax.axvspan(2004.5, 2006.5, color=C["grid"], alpha=0.6, zorder=0)
    ax.text(2005.5, 2.5e8, "End of Dennard\nscaling", ha="center", va="bottom",
            fontsize=10, color=C["text"])
    for key, label, col, mk, ly in series:
        pts = [(r["year"], r["value"]) for r in rows if r["series"] == key]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=9, color=col,
                   marker=mk, alpha=0.8, edgecolors="none")
        ax.text(2024, ly, label, color=col, fontsize=10, fontweight="bold",
                va="center", ha="left")
    ax.set_yscale("log")
    ax.set_xlim(1970, 2023)
    ax.set_ylim(0.5, 2e8)
    ax.set_xticks([1970, 1990, 2010])
    ax.set_yticks([1, 1e2, 1e4, 1e6, 1e8])
    ax.minorticks_off()
    ax.set_ylabel("Value (log scale)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "microprocessor_trends.svg")


trends()

# ---------------------------------------------------------------------------
# 9. Energy per inference across tiers, redrawn at slide size with the
#    values printed on the book's figure (System Paradigms, fig. 7,
#    EnergyAnchors): 10 μJ, 2.6 mJ, 15.9 mJ, 75.5 mJ, 1 kJ; span 100,000,000×.
# ---------------------------------------------------------------------------
def energy():
    rows = [
        ("TinyML: keyword spotting", 10e-6, "compute", "≈10 μJ"),  # Greek mu: Helvetica slants the micro sign
        ("Mobile: MobileNetV2, phone", 2.6e-3, "memory", "≈2.6 mJ"),
        ("Edge: ResNet-50, Jetson", 15.9e-3, "data", "≈15.9 mJ"),
        ("Cloud: ResNet-50, A100", 75.5e-3, "danger", "≈75.5 mJ"),
        ("Cloud: LLM query (anchor)", 1e3, "danger", "≈1 kJ"),
    ]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.55))
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, left=1e-6, color=C[ck], height=0.62)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v * 2.2, y, txt, va="center", fontsize=12, fontweight="bold", color=C[ck])
    ax.annotate("", xy=(4e5, n - 1), xytext=(4e5, 0),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=2))
    ax.text(6e5, (n - 1) / 2, "100,000,000×", color=C["danger"], fontsize=12,
            fontweight="bold", rotation=90, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e6)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.minorticks_off()
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Energy per inference (joules, log scale)")
    sp.save(fig, out / "energy_per_inference.svg")


energy()

print("figures written")
