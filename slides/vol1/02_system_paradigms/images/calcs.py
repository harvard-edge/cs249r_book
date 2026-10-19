#!/usr/bin/env python3
"""Every derived number in Lecture 2 (System Paradigms), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports the inputs and loaders below so figures and
slides share one computation.

Run from this folder: python3 calcs.py
"""

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
FIBER_KM_S = 200_000          # light in fiber (System Paradigms, eq. 1, S2.2)
A100_TFLOPS = 312.0           # A100 FP16 dense (Napkin Math 2.1)
A100_TBS = 2.04               # A100 HBM bandwidth, TB/s (Napkin Math 2.1)

# AWS round trips, median (P50) over one week, from us-east-1 (N. Virginia).
# Source: cloudping.co /api/latencies?percentile=p_50&timeframe=1W, retrieved
# 2026-10-18, saved to data/cloudping_p50_1W_2026-10-18.json.
# Coordinates are of the metro area AWS names for the region (Wikipedia city
# coordinates); AWS does not publish building locations, so distances are
# approximate.
CITY = {
    "us-east-1": ("N. Virginia (Ashburn)", 39.04, -77.49),
    "us-east-2": ("Ohio (Columbus)", 39.96, -83.00),
    "us-west-2": ("Oregon (Boardman)", 45.84, -119.70),
    "eu-west-1": ("Ireland (Dublin)", 53.35, -6.26),
    "ap-northeast-1": ("Tokyo", 35.68, 139.69),
}
ROUTES = [("us-east-2", "Ohio"), ("us-west-2", "Oregon"),
          ("eu-west-1", "Ireland"), ("ap-northeast-1", "Tokyo")]

# Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing
# Unit," ISCA 2017 (arXiv 1704.04760). Intensity: Table 1, "TPU Ops / Weight
# Byte". Measured rate: Table 3, row 9, "TeraOps/sec (92 Peak)". Share of
# deployed TPUs, July 2016: Table 1 (MLPs 61%, LSTMs 29%, CNNs 5%).
TPU_PEAK = 92.0               # TOPS, Table 2 and Table 3
TPU_RIDGE = 1350              # ops per weight byte, Fig. 5 caption
TPU_APPS = [  # name, ops per weight byte, measured TOPS
    ("MLP0", 200, 12.3), ("MLP1", 168, 9.7),
    ("LSTM0", 64, 3.7), ("LSTM1", 96, 2.8),
    ("CNN0", 2888, 86.0), ("CNN1", 1750, 14.1),
]
TPU_SHARE = {"MLP": 61, "LSTM": 29, "CNN": 5}

# MLPerf Tiny v1.2, closed division, keyword spotting, energy track.
# Source: github.com/mlcommons/tiny_results_v1.2, summary.csv (copied to
# data/mlperf_tiny_v1.2_summary.csv).
TINY_CSV = DATA / "mlperf_tiny_v1.2_summary.csv"

# 3DMark Wild Life Stress Test on an iPhone 15 Pro Max, 20 loops: the app's
# own result screen, published in GSMArena's review (2023-09-22 run).
WL_BEST, WL_LOWEST = 10012, 7045


# Gholami et al., "AI and Memory Wall," IEEE Micro 2024 (arXiv 2403.14123),
# Fig. 1. The figure ships as a vector PDF in the arXiv source
# (figs/hw_scaling.pdf); data/gholami2024_hw_scaling_vectors.svg is that file
# converted with pdftocairo -svg. Each marker is a filled circle whose centre
# is read back to (year, normalized scaling) through the figure's own axis
# ticks: major x ticks one year apart starting at 1996 (x = 186.50, 63.92 per
# year), y ticks at 0.01 (88.92) and 1 (286.04), 197.12 per two decades.
GHOLAMI_SVG = DATA / "gholami2024_hw_scaling_vectors.svg"
GHOLAMI_COLORS = {"rgb(0%, 0%, 0%)": "flops", "rgb(0%, 50%, 0%)": "dram",
                  "rgb(0%, 0%, 100%)": "interconnect"}


def gholami_points():
    """{series: [(year, normalized value)]} digitized from the vector figure."""
    import re
    t = GHOLAMI_SVG.read_text()
    out = {k: [] for k in GHOLAMI_COLORS.values()}
    for m in re.finditer(r"<path ([^>]*)/>", t):
        a = m.group(1)
        f = re.search(r'fill="(rgb\([^)]*\))"', a)
        tr = re.search(r'transform="matrix\(([^)]*)\)"', a)
        d = re.search(r'd="([^"]*)"', a).group(1)
        # markers: filled closed curves of radius 11 drawn at a translation
        if not (f and tr and f.group(1) in GHOLAMI_COLORS and d.startswith("M") and "C" in d
                and "stroke-width=\"1\"" in a):
            continue
        v = [float(x) for x in tr.group(1).split(",")]
        x = (v[4] - 0.150904) / 0.999492          # undo the page transform
        y = (1009.97401 - v[5]) / 0.999492
        year = 1996 + (x - 186.498439) / 63.9231
        val = 10 ** (-2 + (y - 88.917577) / 197.1197 * 2)
        out[GHOLAMI_COLORS[f.group(1)]].append((year, val))
    return out


def growth_per_2yr(pts):
    """Least-squares fit of log10(value) on year; growth factor per two years."""
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [math.log10(p[1]) for p in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return 10 ** (2 * b), b, my - b * mx


def haversine_km(a, b):
    """Great-circle distance between two (lat, lon) points, mean Earth radius."""
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def rtt_floor_ms(km):
    return 2 * km / FIBER_KM_S * 1000


def aws_routes():
    """(label, distance km, fiber floor ms, measured median ms) per route."""
    d = json.load(open(DATA / "cloudping_p50_1W_2026-10-18.json"))["data"]
    src = CITY["us-east-1"][1:]
    out = []
    for key, label in ROUTES:
        km = haversine_km(src, CITY[key][1:])
        out.append((label, km, rtt_floor_ms(km), d["us-east-1"][key]))
    return out


def tiny_kws():
    """KWS energy rows from the MLPerf Tiny v1.2 summary (closed, with power)."""
    rows = []
    for r in csv.DictReader(open(TINY_CSV)):
        if r["Model"] == "kws" and r["HasPower"] == "True" and r["Division"] == "closed":
            rows.append((r["Organization"], r["BoardName"], r["SystemDesc"], float(r["Power"])))
    return rows


def main():
    head("Title / power ladder (slide 'Predict: how much power')")
    # Inputs: cloud pod 3 MW, TinyML 1 mW (S2.1); 128 TB pod HBM, 512 KB SRAM (table 4)
    show("power span, orders of magnitude (log10 3e6/1e-3)", f"{math.log10(3e6 / 1e-3):.2f}")
    show("memory drop 128 TB / 512 KB (decimal)", f"{128e12 / 512e3:,.0f}")
    show("memory drop 128 TiB / 512 KiB (binary)", f"{128 * 2**40 / (512 * 2**10):,.0f}")

    head("Silicon contract")
    # TPU v4 chip: 275 TFLOP/s BF16, 32 GiB HBM2, 1,200 GB/s (Jouppi et al., ISCA 2023,
    # cited in the chapter); pod of 4,096 chips (chapter table 4)
    show("TPU v4 pod peak, EFLOP/s (4,096 x 275 TFLOP/s)", f"{4096 * 275 / 1e6:.3f}")
    show("TPU v4 pod HBM, TiB (4,096 x 32 GiB)", f"{4096 * 32 / 1024:.0f}")

    head("Light in fiber / round trips")
    show("robot arm floor, 1,500 km (ms)", f"{rtt_floor_ms(1500):.1f}")
    show("California-Virginia floor, 3,600 km (ms)", f"{rtt_floor_ms(3600):.1f}")
    show("arm overshoot past 10 ms budget (ms)", f"{rtt_floor_ms(1500) - 10:.1f}")
    for label, km, floor, meas in aws_routes():
        show(f"N. Virginia -> {label}: distance km / floor ms / measured median ms / ratio",
             f"{km:,.0f} / {floor:.1f} / {meas:.1f} / {meas / floor:.1f}x")

    head("Factory cameras (Napkin Math 2.4)")
    cam = 1920 * 1080 * 3 * 30           # bytes/s per camera: 1080p, 3 B/pixel, 30 FPS
    total = 100 * cam
    link = 10e9 / 8                      # 10 Gb/s in bytes/s
    meta = 1e3 * 20                      # 1 KB per detection, 20 per second
    show("per camera (MB/s)", f"{cam / 1e6:.1f}")
    show("100 cameras (GB/s)", f"{total / 1e9:.2f}")
    show("10 Gb/s link (GB/s)", f"{link / 1e9:.2f}")
    show("raw / link", f"{total / link:.1f}x")
    show("metadata (KB/s)", f"{meta / 1e3:.0f}")
    show("raw / metadata", f"{total / meta:,.0f}x")

    head("Drone (Napkin Math 2.6)")
    frame = 3840 * 2160 * 3              # 4K, 3 B/pixel
    send_ms = frame * 8 / 100e6 * 1000   # 100 Mb/s uplink
    deadline = 1000 / 60
    show("4K frame (MB)", f"{frame / 1e6:.1f}")
    show("send (ms)", f"{send_ms:.1f}")
    show("remote path lower bound (ms), +110 ms", f"{send_ms + 110:.1f}")
    show("deadline at 60 FPS (ms)", f"{deadline:.1f}")
    show("lower bound / deadline", f"{(send_ms + 110) / deadline:.0f}x")
    show("send after 100-fold compression (ms)", f"{send_ms / 100:.1f}")

    head("Memory wall (Gholami et al., IEEE Micro 2024, Fig. 1)")
    # Printed on the figure: HW FLOPS 60,000x, DRAM BW 100x, interconnect 30x
    # over 20 years (3.0x, 1.6x, 1.4x per two years). Refit here from the
    # digitized markers to confirm the digitization.
    show("FLOPS growth / DRAM bandwidth growth, paper's 20-year factors", f"{60000 / 100:.0f}x")
    pts = gholami_points()
    for k in ("flops", "dram", "interconnect"):
        g = growth_per_2yr(pts[k])[0]
        show(f"{k}: markers / refit growth per 2 years / over 20 years",
             f"{len(pts[k])} / {g:.2f}x / {g ** 10:,.0f}x")

    head("A100 roofline")
    ridge = A100_TFLOPS / A100_TBS
    show("ridge (FLOP/byte)", f"{ridge:.1f}")
    show("10 FLOP/byte: attainable TFLOP/s", f"{10 * A100_TBS:.1f}")
    show("10 FLOP/byte: share of peak", f"{10 * A100_TBS / A100_TFLOPS * 100:.1f}%")

    head("ResNet-50, batch 1 (Napkin Math 2.1)")
    # 8.2 GFLOP, 56.3 MB moved FP16 on A100; phone NPU 35 TOPS, 51.2 GB/s, 28.2 MB INT8
    show("A100 compute (ms)", f"{8.2e9 / 312e12 * 1e3:.4f}")
    show("A100 memory (ms)", f"{56.3e6 / 2.04e12 * 1e3:.4f}")
    show("intensity (FLOP/byte)", f"{8.2e9 / 56.3e6:.1f}")
    show("phone NPU compute (ms)", f"{8.2e9 / 35e12 * 1e3:.3f}")
    show("phone NPU memory (ms)", f"{28.2e6 / 51.2e9 * 1e3:.3f}")

    head("Decode, batch 1")
    show("attainable at 1 FLOP/byte (TFLOP/s)", f"{1 * A100_TBS:.2f}")
    show("share of peak", f"{A100_TBS / A100_TFLOPS * 100:.2f}%")

    head("TPU roofline with production apps (Jouppi et al. 2017)")
    # Fig. 5 draws the slope so it meets the 92 TOPS ceiling at 1,350; the slope
    # is therefore 92 / 1350 TB/s. (Table 1 counts one op per multiply-add in the
    # intensity while the 92 TOPS counts two, so the slope is twice the 34 GB/s
    # DRAM figure in Table 2.)
    slope = TPU_PEAK / TPU_RIDGE
    show("slope implied by Fig. 5 (GB/s)", f"{slope * 1000:.1f}")
    show("2 x 34 GB/s (Table 2)", f"{2 * 34} GB/s; ridge {TPU_PEAK / 0.068:.0f}")
    for name, x, y in TPU_APPS:
        roof = min(TPU_PEAK, x * slope)
        side = "memory slope" if x < TPU_RIDGE else "compute ceiling"
        show(f"{name}: roof / measured / % of roof / side", f"{roof:.1f} / {y} / {y / roof * 100:.0f}% / {side}")
    show("share of deployed TPUs on memory-bound apps (MLP + LSTM)", f"{TPU_SHARE['MLP'] + TPU_SHARE['LSTM']}%")

    head("Power wall")
    show("2x clock, P ~ f^3", f"{2 ** 3}x power")

    head("Throttling (3DMark Wild Life Stress Test, iPhone 15 Pro Max)")
    show("stability = lowest / best", f"{WL_LOWEST / WL_BEST * 100:.1f}%")
    show("loss from best to lowest loop", f"{(1 - WL_LOWEST / WL_BEST) * 100:.1f}%")
    show("napkin math: 12 W cut 4 times (W)", f"{12 / 4:.0f}")

    head("Battery tax (Napkin Math 2.8)")
    show("runtime 15 Wh / 2 W (h)", f"{15 / 2:.1f}")
    show("a day at 2 W (Wh)", f"{2 * 24}")
    show("share of battery", f"{2 * 24 / 15 * 100:.0f}%")

    head("Energy per inference (fig. 7, Napkin Math 2.5, MLPerf Tiny)")
    show("span 1 kJ / 10 uJ", f"{1e3 / 10e-6:,.0f}x")
    show("send 1 MB / whole phone (100 mJ / 2.6 mJ)", f"{100 / 2.6:.1f}x")
    show("send 1 MB / NPU alone (100 mJ / 0.1 mJ)", f"{100 / 0.1:,.0f}x")
    kws = tiny_kws()
    for org, board, desc, uj in kws:
        show(f"MLPerf Tiny KWS {board} ({desc}) uJ", f"{uj}")
    ndp = min(uj for *_, uj in kws)
    m4 = [uj for o, b, d, uj in kws if b == "NUCLEO-L4R5ZI"][0]
    show("Cortex-M4 (NUCLEO-L4R5ZI) / NDP9120 best", f"{m4 / ndp:.0f}x")

    head("Doorbell (S2.3.2, fig. 3)")
    show("weights in flash (0.5 / 4.2 MB)", f"{0.5 / 4.2 * 100:.1f}%")
    lat = 25e6 / 0.25e9 * 1e3 + 1
    show("latency per frame (ms)", f"{lat:.0f}")
    show("share of 200 ms chime", f"{lat / 200 * 100:.1f}%")
    show("share of 50 ms chime", f"{lat / 50 * 100:.0f}%")

    head("Screening gates (Example 2.1)")
    show("100 km/h for 100 ms (m)", f"{100 / 3.6 * 0.1:.2f}")

    head("Voice assistant (Napkin Math 2.10)")
    per_dev = 16_000 * 2                 # 16 kHz, 16-bit
    show("per device (KB/s)", f"{per_dev / 1e3:.0f}")
    show("1B devices (TB/s)", f"{per_dev * 1e9 / 1e12:.0f}")
    show("reduction, 20 queries x 5 s a day", f"{86400 / (20 * 5):.0f}x (chapter: 860x)")
    show("after the chapter's 860x (GB/s)", f"{per_dev * 1e9 / 860 / 1e9:.1f}")
    show("cloud detection at $0.50/device/yr ($M)", f"{0.50 * 1e9 / 1e6:.0f}")
    gpu_h = 20e9 * 0.2 / 3600
    show("GPU-hours a day (20B x 200 ms), millions", f"{gpu_h / 1e6:.2f}")
    show("datacenters, average (10,000 GPUs each)", f"{gpu_h / 24 / 10_000:.2f}")
    show("datacenters at 4.5x peak", f"{gpu_h / 24 / 10_000 * 4.5:.1f}")

    head("Hey Siri two-pass detector (Apple ML Research 2017)")
    show("second-pass width / first-pass width (192 / 32 units)", f"{192 / 32:.0f}x")

    head("Hybrid patterns")
    show("training / inference bytes per parameter (16 / 2)", f"{16 / 2:.0f}x")

    head("True or false: camera pipeline (S2.12)")
    new = 100 + 60 / 10 + 40
    show("pipeline after 10x faster model (ms)", f"{new:.0f}")
    show("speedup", f"{200 / new:.2f}x")
    show("limit with an infinitely fast model", f"{200 / 140:.2f}x")


if __name__ == "__main__":
    main()
