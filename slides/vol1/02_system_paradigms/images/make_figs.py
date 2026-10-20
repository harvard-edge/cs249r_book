#!/usr/bin/env python3
"""Data plots for Lecture 2 (System Paradigms).

Every value is sourced to the rendered chapter (System Paradigms) unless
noted:
  - Power tiers: cloud 3 MW, edge 200 W, mobile 5 W, TinyML 1 mW (Sec. 2.1).
    Memory per example platform: TPU v4 pod 128 TB HBM, DGX Spark 128 GB,
    phone 8-16 GB, ESP32-S3 512 KB SRAM (table 4).
  - Round trip in fiber at 200,000 km/s (eq. 1): the arm's 1,500 km gives
    15 ms (Napkin Math 2.2). Measured AWS medians from us-east-1 and the
    great-circle floors come from calcs.py (cloudping.co, 2026-10-18).
  - Camera bandwidth: 100 x 186.6 MB/s = 18.7 GB/s raw; 10 Gb/s link =
    1.25 GB/s; defect metadata 1 KB x 20 events/s = 20 KB/s (Napkin Math 2.4).
  - Memory wall: the slide shows Gholami et al., IEEE Micro 2024, Fig. 1
    (gholami2024-hw-scaling.pdf, from the arXiv 2403.14123 source), not a
    plot drawn here.
  - TPU roofline: Jouppi et al., ISCA 2017, Tables 1 and 3 (inputs in calcs.py).
  - A100 roofline: 312 TFLOP/s FP16, 2.04 TB/s, ridge 153 FLOP/byte;
    ResNet-50 at batch 1: 8.2 GFLOP over 56.3 MB = 146 FLOP/byte (Napkin
    Math 2.1). Batch-1 decode with 16-bit weights: about 1 FLOP/byte (Model
    Serving, Sec. 13.9, as in Lecture 0). Batch B: about B FLOP/byte if
    weight traffic dominates (derived; KV cache and activations ignored).
  - Throttling: the slide shows the 3DMark result screen (GSMArena), not a
    plot drawn here.
  - Voice assistant: 1B devices x 32 KB/s = 32 TB/s if every microphone
    streams; on-device wake word cuts audio leaving each device about 860x,
    so about 37 GB/s (derived) (Napkin Math 2.10).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402

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
# 2. Round trip: the fiber floor, then measured AWS medians (two steps)
# ---------------------------------------------------------------------------
def rtt(step):
    rows = [("Arm to cloud, 1,500 km", 1500, calcs.rtt_floor_ms(1500), None)]
    rows += [(f"Virginia to {lab}, {km:,.0f} km", km, fl, me)
             for lab, km, fl, me in calcs.aws_routes()]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    n = len(rows)
    for i, (label, km, floor, meas) in enumerate(rows):
        y = n - 1 - i
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.barh(y + 0.17, floor, color=C["danger_l"], edgecolor=C["danger"],
                lw=1.2, height=0.32)
        ax.text(max(floor, 10) + 1.5, y + 0.17, f"{floor:.1f}" if floor < 10 else f"{floor:.0f}", va="center", fontsize=11,
                color=C["danger"], fontweight="bold")
        if meas is not None:
            a2 = 1.0 if step == 2 else 0.0
            ax.barh(y - 0.17, meas, color=C["danger"], height=0.32, alpha=a2)
            ax.text(meas + 1.5, y - 0.17, f"{meas:.0f} measured", va="center",
                    fontsize=11, color=C["danger"], fontweight="bold", alpha=a2)
    ax.axvline(10, color=C["data"], lw=2.5)
    ax.text(11, n - 0.25, "Arm budget 10 ms", color=C["data"], fontsize=11,
            fontweight="bold", va="center")
    ax.text(0.99, 0.99, "light: fiber floor\ndark: AWS median", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=C["neutral"])
    ax.set_xlim(0, 185)
    ax.set_ylim(-0.6, n + 0.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Round trip (ms)")
    sp.save(fig, out / f"rtt_measured_step{step}.svg")


for k in (1, 2):
    rtt(k)

# ---------------------------------------------------------------------------
# 3. Factory cameras: raw video against the link (three steps)
# ---------------------------------------------------------------------------
CAM_RAW = 100 * 1920 * 1080 * 3 * 30      # Napkin Math 2.4 inputs
CAM_LINK = 10e9 / 8
CAM_META = 1e3 * 20
# Tick labels at 14 pt so the powers of ten stay legible on the slide.
with sp.plt.rc_context({"xtick.labelsize": 14}):
    sp.hbars_steps(out / "camera_bandwidth", [
      ("100 raw camera feeds", CAM_RAW, "danger", f"{CAM_RAW / 1e9:.1f} GB/s"),
      ("10 Gb/s factory link", CAM_LINK, "neutral",
       f"{CAM_LINK / 1e9:.2f} GB/s: {CAM_RAW / CAM_LINK:.0f}× too small"),
      ("Defect metadata only", CAM_META, "data", f"{CAM_META / 1e3:.0f} KB/s"),
    ], "B/s", xlabel="Data rate (bytes per second, log scale)")

# ---------------------------------------------------------------------------
# 4. Memory wall from published chips: Gholami et al., IEEE Micro 2024,
#    Fig. 1, redrawn at slide size from the markers digitized in calcs.py.
#    Trend lines are least-squares fits on log values, as in the paper.
# ---------------------------------------------------------------------------
def memory_wall():
    pts = calcs.gholami_points()
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    spec = [("flops", C["compute"], "Peak FLOP/s"),
            ("dram", C["memory"], "DRAM bandwidth"),
            ("interconnect", C["network"], "Interconnect bandwidth")]
    for key, col, name in spec:
        xs = [p[0] for p in pts[key]]
        ys = [p[1] for p in pts[key]]
        g, b, a0 = calcs.growth_per_2yr(pts[key])
        ax.scatter(xs, ys, s=22, color=col, edgecolors="white", linewidths=0.6, zorder=3)
        fx = np.array([min(xs), max(xs)])
        ax.plot(fx, 10 ** (a0 + b * fx), color=col, lw=2.5, alpha=0.55, zorder=2)
        ly = {"flops": 4e6, "dram": 500, "interconnect": 12}[key]
        ax.text(2023.3, ly, f"{name}\n{g:.1f}× every 2 years", color=col,
                fontsize=11, fontweight="bold", va="center", ha="left")
    for label, (x, y), off in (("R10000", (1997.07, 1.0), (4, 8)),
                               ("H100", (2022.7, 2.92e6), (-6, 4))):
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points",
                    ha="left" if off[0] > 0 else "right", va="bottom",
                    fontsize=10, color=C["text"])
    ax.set_yscale("log")
    ax.set_xlim(1996, 2023.2)
    ax.set_ylim(0.3, 3e7)
    ax.set_xticks([1996, 2002, 2008, 2014, 2020])
    ax.set_yticks([1, 1e2, 1e4, 1e6])
    ax.set_yticklabels(["1×", "100×", "10,000×", "1,000,000×"])
    ax.minorticks_off()
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Growth from each\nseries' first point (log)")
    sp.save(fig, out / "memory_wall_measured.svg")


memory_wall()

# ---------------------------------------------------------------------------
# 5. The A100 roofline, built up, then workloads placed on it
# ---------------------------------------------------------------------------
PEAK, BW = 312.0, 2.04       # TFLOP/s, TB/s
RIDGE = PEAK / BW            # about 153 FLOP/byte


def roofline(name, parts):
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


def roofline_col(name, point=None, peak2=False, arrow=False, qmark=False):
    """The same A100 roofline, drawn at the size of a half-slide column
    (about 3.3 x 2.05 in) so its 12 pt labels are not shrunk. The slope label
    is left off: the full-width build on the previous slide teaches it."""
    xs = np.logspace(-1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(3.3, 2.05))
    m = xs <= RIDGE
    ax.plot(xs[m], xs[m] * BW, color=C["memory"], lw=3.5)
    ax.plot(xs[~m], [PEAK] * (~m).sum(), color=C["compute"], lw=3.5)
    ax.text(1500, PEAK * 0.72, "312 TFLOP/s", color=C["compute"], fontsize=12,
            fontweight="bold", ha="center", va="top")
    ax.axvline(RIDGE, color=C["neutral"], lw=1.2, ls="--")
    ax.text(RIDGE * 1.15, 0.22, "Ridge\n153", color=C["neutral"], fontsize=12,
            va="bottom")
    if peak2:
        r2 = 2 * PEAK / BW
        ax.plot(xs[xs <= r2], xs[xs <= r2] * BW, color=C["memory"], lw=1.5, ls=":")
        ax.plot(xs[xs >= r2], [2 * PEAK] * (xs >= r2).sum(), color=C["compute"],
                lw=2, ls=":")
        ax.text(2200, 2 * PEAK * 1.2, "2× FLOP/s", color=C["compute"], fontsize=12,
                ha="center", va="bottom")
    if point:
        label, x, ck, dx, dy, ha, va = point
        y = min(PEAK, x * BW)
        ax.plot([x], [y], "o", ms=11, color=C[ck], mec="white", mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=12, fontweight="bold", color=C[ck])
    if arrow:
        ax.annotate("", xy=(256, PEAK), xytext=(1.3, 1.3 * BW),
                    arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=2))
        ax.text(4, 6, "batch B", color=C["neutral"], fontsize=12,
                fontweight="bold", ha="left", va="top")
    if qmark:
        ax.text(1.0, 6, "???", color=C["danger"], fontsize=18, fontweight="bold",
                ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.15, 3000)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=12)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"], fontsize=12)
    ax.set_xlabel("FLOP per byte (log scale)", fontsize=12)
    ax.set_ylabel("TFLOP/s (log scale)", fontsize=12)
    sp.save(fig, out / f"{name}.svg")


roofline_col("roofline_resnet",
             point=("ResNet-50,\nbatch 1: 146", 146.0, "memory", -10, 4, "right", "bottom"))
roofline_col("roofline_decode_q", peak2=True, qmark=True)
roofline_col("roofline_decode_a",
             point=("Decode,\nbatch 1: ≈1", 1.0, "danger", 6, -8, "left", "top"),
             peak2=True, arrow=True)


# ---------------------------------------------------------------------------
# 6. The TPU (2015) roofline with six production applications, measured
#    (Jouppi et al., ISCA 2017, Fig. 5 redrawn from Tables 1 and 3)
# ---------------------------------------------------------------------------
def tpu_roofline():
    peak, ridge = calcs.TPU_PEAK, calcs.TPU_RIDGE
    xs = np.logspace(1, 4, 400)
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.75))
    m = xs <= ridge
    ax.plot(xs[m], xs[m] * peak / ridge, color=C["memory"], lw=3.5)
    ax.plot(xs[~m], [peak] * (~m).sum(), color=C["compute"], lw=3.5)
    ax.axvline(ridge, color=C["neutral"], lw=1.2, ls="--")
    ax.text(ridge * 0.92, 1.25, "Ridge\n1,350", color=C["neutral"], fontsize=11,
            ha="right", va="bottom")
    ax.text(5200, peak * 1.15, "92 TOPS peak", color=C["compute"], fontsize=11,
            fontweight="bold", ha="center", va="bottom")
    offs = {"MLP0": (9, -3, "left", "top"), "MLP1": (-10, 4, "right", "bottom"),
            "LSTM0": (-11, 2, "right", "bottom"), "LSTM1": (9, -3, "left", "top"),
            "CNN0": (4, -9, "left", "top"), "CNN1": (10, 0, "left", "center")}
    for name, x, y in calcs.TPU_APPS:
        ck = "memory" if x < ridge else "compute"
        ax.plot([x], [y], "o", ms=10, color=C[ck], mec="white", mew=1.5, zorder=5)
        dx, dy, ha, va = offs[name]
        ax.annotate(f"{name} {y:g}", (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=11,
                    fontweight="bold", color=C[ck])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e4)
    ax.set_ylim(1, 300)
    ax.minorticks_off()
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_xlabel("Operations per byte of weights read (log scale)")
    ax.set_ylabel("Measured TOPS (log)")
    sp.save(fig, out / "tpu_roofline.svg")


tpu_roofline()

# ---------------------------------------------------------------------------
# 7. Voice assistant: audio leaving one billion devices (two steps)
# ---------------------------------------------------------------------------
sp.hbars_steps(out / "voice_upload", [
    ("Every microphone streams", 32e12, "danger", "32 TB/s"),
    ("Wake word on the device", 32e12 / 860, "data",
     f"≈{32e12 / 860 / 1e9:.0f} GB/s: 860× less"),  # chapter's 860x
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
#    Step 2 adds Napkin Math 2.5: sending 1 MB by radio costs about 100 mJ,
#    39 times the whole phone's 2.6 mJ; the NPU alone uses about 0.1 mJ.
#    A measured row from MLPerf Tiny v1.2 (keyword spotting, energy track,
#    best closed-division result, read by calcs.tiny_kws) sits under the
#    chapter's TinyML anchor.
# ---------------------------------------------------------------------------
def energy(step):
    tiny = min(calcs.tiny_kws(), key=lambda r: r[3])
    rows = [
        ("TinyML: keyword spotting", 10e-6, "compute", "≈10 μJ"),  # Greek mu: Helvetica slants the micro sign
        ("Measured: KWS, Syntiant NDP9120", tiny[3] * 1e-6, "compute",
         f"{tiny[3]:.1f} μJ (MLPerf Tiny)"),
        ("Mobile: MobileNetV2, phone", 2.6e-3, "memory", "≈2.6 mJ"),
        ("Mobile: send 1 MB by radio", 0.1, "danger", f"≈100 mJ: {100 / 2.6:.0f}× the phone"),
        ("Edge: ResNet-50, Jetson", 15.9e-3, "data", "≈15.9 mJ"),
        ("Cloud: ResNet-50, A100", 75.5e-3, "danger", "≈75.5 mJ"),
        ("Cloud: LLM query (anchor)", 1e3, "danger", "≈1 kJ"),
    ]
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.95))
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        send = label.startswith("Mobile: send")
        meas = label.startswith("Measured")
        shown = step == 2 or not send
        alpha = 1.0 if shown else 0.0  # hidden rows keep their slot
        if send:
            ax.barh(y, v, left=1e-6, height=0.62, facecolor="white",
                    edgecolor=C[ck], lw=2.2, ls="--", alpha=alpha)
        elif meas:
            ax.barh(y, v, left=1e-6, facecolor=C["compute_l"], edgecolor=C[ck],
                    lw=1.5, hatch="///", height=0.62)
        else:
            ax.barh(y, v, left=1e-6, color=C[ck], height=0.62)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold",
                color=C["danger"] if send else C["text"], alpha=alpha)
        ax.text(v * 2.2, y, txt, va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=alpha)
    # the phone's NPU alone, inside the whole-phone bar
    ynpu = n - 3
    ax.plot([1e-4, 1e-4], [ynpu - 0.31, ynpu + 0.31], color="white", lw=2.5,
            alpha=1.0 if step == 2 else 0.0)
    ax.text(0.5, ynpu, "(tick: NPU alone ≈0.1 mJ)", color=C["memory"], fontsize=12,
            fontweight="bold", ha="left", va="center", alpha=1.0 if step == 2 else 0.0)
    ax.annotate("", xy=(4e5, n - 1), xytext=(4e5, 0),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=2))
    ax.text(6e5, (n - 1) / 2, f"{1e3 / 10e-6:,.0f}×", color=C["danger"], fontsize=12,
            fontweight="bold", rotation=90, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e6)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.tick_params(axis="x", labelsize=13)
    ax.minorticks_off()
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Energy per inference (joules, log scale)")
    sp.save(fig, out / f"energy_per_inference_step{step}.svg")


for k in (1, 2):
    energy(k)


# ---------------------------------------------------------------------------
# 10. Smart doorbell scorecard on the ESP32-S3 (System Paradigms, S2.3.2 and
#     fig. 3): 0.5 MB of weights fill 11.9% of 4.2 MB flash; 25 MFLOP at
#     about 0.25 GFLOP/s plus 1 ms overhead = 101 ms per frame. Step 1: the
#     200 ms chime, 101 / 200 = 50.5%; step 2: a 50 ms chime, 202% (calcs.py).
# ---------------------------------------------------------------------------
def doorbell(step):
    ms = 25e6 / 0.25e9 * 1e3 + 1          # 25 MFLOP at 0.25 GFLOP/s, +1 ms
    budget = 200 if step == 1 else 50
    lat = ms / budget * 100
    lat_txt = f"{ms:.0f} ms of {budget} ms: {lat:.1f}%".replace(".0%", "%")
    flash = 0.5 / 4.2 * 100
    rows = [("Latency\nper frame", lat, "danger" if lat > 100 else "data", lat_txt),
            ("Weights in\n4.2 MB flash", flash, "data", f"{flash:.1f}%")]
    fig, ax = sp.plt.subplots(figsize=(4.0, 2.3))
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.55)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        if i == 0:  # the latency value sits above its bar, so it never moves
            ax.text(2, y + 0.34, txt, va="bottom", ha="left", fontsize=12,
                    fontweight="bold", color=C[ck], zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        else:
            ax.text(v + 4, y, txt, va="center", ha="left", fontsize=12,
                    fontweight="bold", color=C[ck])
    ax.axvline(100, color=C["neutral"], lw=2, ls="--")
    ax.set_xlim(0, 230)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200])
    ax.set_xticklabels(["0", "100%\nbudget", "200%"], fontsize=12)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / f"doorbell_step{step}.svg")


for k in (1, 2):
    doorbell(k)

print("figures written")
