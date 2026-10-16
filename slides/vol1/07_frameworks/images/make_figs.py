#!/usr/bin/env python3
"""Data plots for Lecture 7 (ML Frameworks).

Every value is from the rendered chapter (ML Frameworks):
  - Operator counts (Sec. 7.3.6, figure 6): PyTorch ATen native functions
    2,658 (v2.9, 2025); TensorFlow operator registry 1,563 (v2.20, 2025);
    ONNX 203; Core ATen IR 192; JAX lax primitives about 135; XLA HLO 123.
  - GPT-2 XL batch-1 decoding step on an H100 (Sec. 7.1): 3 GB of FP16
    weights at 3.35 TB/s is about 0.9 ms; 48 blocks x 12 kernels = 576
    launches; 5 us per launch gives 2.9 ms; with 10 us of Python dispatch,
    15 us per operation gives 8.6 ms.
  - Dispatch share (Sec. 7.3.4, illustrative model): 15 us per operation;
    a 1 us kernel spends 94 percent of its step in dispatch, a 100 us
    kernel 13 percent.
  - Five small operations (Sec. 7.3.6, figure 7): eager, each 1 us kernel
    waits on 10 us of Python dispatch; compiled, one 5 us launch and a
    4 us fused kernel.
  - ResNet-50 training memory (Sec. 7.4.3, illustrative): 102.4 MB of FP32
    weights; 8.4 to 12.4 GB in all at batch 64; batch 512 holds 64 to 96 GB
    of saved activations (Sec. 7.10) against an 80 GB H100.
  - Bandwidth (Sec. 7.5.2, table 7): HBM3 3,350 GB/s; NVLink 4.0 450 GB/s
    per direction; PCIe Gen5 x16 64 GB/s.
  - One MNIST training step (Sec. 7.7.3): 13.1 MFLOP at 67 TFLOP/s FP32 is
    about 0.2 us; 1.2 MB at 3.35 TB/s is about 0.4 us; host 95 us.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
C = sp.C
plt = sp.plt

# -- Operator surfaces against compiler cores (two steps) --------------------
ops = [
    ("PyTorch ATen (eager)", 2658, "neutral", "2,658"),
    ("TensorFlow (eager)", 1563, "neutral", "1,563"),
    ("ONNX standard", 203, "compute", "203"),
    ("Core ATen IR", 192, "compute", "192"),
    ("JAX lax primitives", 135, "compute", "about 135"),
    ("XLA HLO", 123, "compute", "123"),
]
for k, step in ((1, 2), (2, None)):
    sp.hbars(out / f"ops_surface_step{k}.svg", ops, "", log=False, step=step,
             xlabel="Operators each backend must implement (2025)")

# -- GPT-2 XL decode: who sets the step time (three steps) -------------------
sp.hbars_steps(out / "decode", [
    ("Read 3 GB of weights (HBM)", 0.9, "memory", "0.9 ms"),
    ("Launch 576 kernels", 2.88, "neutral", "2.9 ms"),
    ("Eager: dispatch + launch", 8.64, "danger", "8.6 ms"),
], "ms", log=False, xlabel="Time for one decoding step on an H100 (ms)")


# -- Dispatch share of a small and a large operation (two steps) -------------
def dispatch_share(path, k):
    rows = [("Small op\n(1 µs kernel)", 15, 1), ("Large op\n(100 µs kernel)", 15, 100)]
    fig, ax = plt.subplots(figsize=sp.SHORT)
    for i, (label, disp, kern) in enumerate(rows):
        y = 1 - i
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
        if i >= k:
            continue
        d = 100 * disp / (disp + kern)
        ax.barh(y, d, color=C["neutral"], height=0.62)
        ax.barh(y, 100 - d, left=d, color=C["compute"], height=0.62)
        ax.text(d / 2, y, f"dispatch {d:.0f}%" if d > 30 else f"{d:.0f}%",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white")
        if 100 - d > 30:
            ax.text(d + (100 - d) / 2, y, f"kernel {100 - d:.0f}%", ha="center",
                    va="center", fontsize=12, fontweight="bold", color="white")
        else:
            ax.text(101.5, y, f"kernel {100 - d:.0f}%", ha="left", va="center",
                    fontsize=12, fontweight="bold", color=C["compute"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the step (15 µs dispatch per operation, model)")
    sp.save(fig, path)


for k in (1, 2):
    dispatch_share(out / f"dispatch_share_step{k}.svg", k)


# -- Five small operations: eager against compiled (two steps) ---------------
def timeline(path, k):
    fig, ax = plt.subplots(figsize=sp.SHORT)
    ys = {"Eager": 1, "Compiled": 0}
    for name, y in ys.items():
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
    t = 0
    for _ in range(5):
        ax.barh(1, 10, left=t, color=C["neutral_l"], edgecolor=C["neutral"], height=0.55)
        ax.barh(1, 1, left=t + 10, color=C["compute"], height=0.55)
        t += 11
    ax.text(2, 1.45, "Python dispatch", color=C["neutral"], fontsize=11, fontweight="bold")
    ax.text(t + 0.8, 1, "55 µs", va="center", fontsize=12, fontweight="bold", color=C["danger"])
    if k == 2:
        ax.barh(0, 5, left=0, color=C["neutral_l"], edgecolor=C["neutral"], height=0.55)
        ax.barh(0, 4, left=5, color=C["compute"], height=0.55)
        ax.text(9.8, 0, "9 µs: one launch, one fused kernel", va="center", fontsize=12,
                fontweight="bold", color=C["compute"])
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.5, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (µs); orange = kernel running on the accelerator")
    sp.save(fig, path)


for k in (1, 2):
    timeline(out / f"python_tax_step{k}.svg", k)


# -- ResNet-50 training memory, log scale (three steps) ----------------------
def train_memory(path, k):
    rows = [
        ("FP32 weights", 102.4, 102.4, "memory", "102.4 MB"),
        ("Training, batch 64", 8400, 12400, "compute", "8.4–12.4 GB in all"),
        ("Saved activations,\nbatch 512", 64000, 96000, "danger", "64–96 GB"),
    ]
    fig, ax = plt.subplots(figsize=sp.SIZE)
    n = len(rows)
    for i, (label, lo, hi, ck, text) in enumerate(rows):
        y = n - 1 - i
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
        if i >= k:
            continue
        ax.barh(y, lo, color=C[ck], height=0.62)
        if hi > lo:
            ax.barh(y, hi - lo, left=lo, color=C[ck + "_l"] if ck + "_l" in C else C[ck],
                    edgecolor=C[ck], height=0.62)
        ax.text(hi * 1.15, y, text, va="center", fontsize=13, fontweight="bold", color=C[ck])
    if k >= 3:
        ax.axvline(80000, ymin=0.0, ymax=0.42, color=C["neutral"], lw=1.5, ls="--")
        ax.text(70000, -0.72, "H100: 80 GB", ha="right", va="center", fontsize=10,
                fontweight="bold", color=C["neutral"])
    ax.set_xscale("log")
    ax.set_xlim(30, 3e6)
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100 MB", "1 GB", "10 GB", "100 GB"])
    ax.minorticks_off()
    ax.set_ylim(-1.0, n - 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Memory (log scale); lighter segment = range of the estimate")
    sp.save(fig, path)


for k in (1, 2, 3):
    train_memory(out / f"train_memory_step{k}.svg", k)

# -- Bandwidth: device memory against the host link (three steps) ------------
sp.hbars_steps(out / "bandwidth", [
    ("HBM3, on device", 3350, "memory", "3,350 GB/s"),
    ("NVLink 4.0, GPU to GPU", 450, "network", "450 GB/s each way"),
    ("PCIe Gen5 x16, host link", 64, "network", "64 GB/s: 52× below HBM"),
], "GB/s", log=False, xlabel="Bandwidth on one H100 (GB/s)")

# -- One MNIST training step: where the time goes (three steps) --------------
sp.hbars_steps(out / "mnist_step", [
    ("Arithmetic, 13.1 MFLOP", 0.2, "compute", "0.2 µs"),
    ("Memory traffic, 1.2 MB", 0.4, "memory", "0.4 µs"),
    ("Host dispatch and launch", 95, "danger", "95 µs"),
], "µs", xlabel="Time on an H100 (µs, log scale; lower bounds and a host model)",
    value_fmt="{:g}")

print("figures written")
