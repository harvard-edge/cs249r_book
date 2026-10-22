#!/usr/bin/env python3
"""Data plots for Lecture 7 (ML Frameworks).

Every value and label comes from calcs.py, which reads MLSysIM: operator
registry counts (local CSV series), the GPT-2 XL decode step and its eager
host work, the dispatch share of a small and a large operation, the PyTorch 2
speedups (Literature.CompilerResults), the ResNet-50 training-memory estimate,
the H100's bandwidths, and the MNIST training step. The measured ResNet-50
memory trace is read from data/resnet50_memory.csv (profile_frameworks.py).

Run from this folder: python3 make_figs.py
"""

# Microseconds are written with the Greek small mu (U+03BC): Helvetica draws
# the micro sign (U+00B5) slanted, which reads as italic next to upright text.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402
from deckvals import val  # noqa: E402

V = calcs.figure_values()

out = Path(__file__).parent
C = sp.C
plt = sp.plt

# -- Operator surfaces against compiler cores (two steps) --------------------
ops = [(label, n, "neutral" if "eager" in label else "compute",
        ("about " if key == "ops_jax" else "") + val(key)) for label, key, n in V["ops"]]
for k, step in ((1, 2), (2, None)):
    sp.hbars(out / f"ops_surface_step{k}.svg", ops, "", log=False, step=step,
             xlabel="Operators defined (2025)")

# -- GPT-2 XL decode: who sets the step time (three steps) -------------------
sp.hbars_steps(out / "decode", [
    (f"Read {val('gpt2_weights_gb')} GB of weights (HBM)", V["decode_read_ms"], "memory", f"{val('gpt2_read_ms')} ms"),
    (f"Launch {val('gpt2_kernels')} kernels", V["decode_launch_ms"], "neutral",
     f"{val('gpt2_launch_ms')} ms"),
    ("Eager: dispatch + launch", V["decode_host_ms"], "danger", f"{val('gpt2_host_ms')} ms"),
], "ms", log=False, xlabel="Time for one decoding step on an H100 (ms)")


# -- Dispatch share of a small and a large operation (two steps) -------------
def dispatch_share(path, k):
    rows = [(f"{name} op\n({val(key)} μs kernel)", op.host_time.to("microsecond").magnitude,
             op.device_time.to("microsecond").magnitude)
            for name, (op, key) in zip(("Small", "Large"), V["dispatch_rows"])]
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
    ax.set_xlabel(f"Share of the step ({val('per_op_us')} μs dispatch per operation, model)")
    sp.save(fig, path)


for k in (1, 2):
    dispatch_share(out / f"dispatch_share_step{k}.svg", k)


# -- PyTorch 2 measured speedups (two steps: inference, then training) -------
def pt2_speedups(path, k):
    suites = list(V["pt2"])
    fig, ax = plt.subplots(figsize=sp.SIZE)
    hgt = 0.36
    for i, suite in enumerate(suites):
        y = len(suites) - 1 - i
        inf, trn = V["pt2"][suite]
        ax.text(-0.01, y, suite, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
        ax.barh(y + hgt / 2, inf, height=hgt, color=C["compute"])
        ax.text(inf + 0.04, y + hgt / 2, f"{inf:.2f}×", va="center", fontsize=12,
                fontweight="bold", color=C["compute"])
        shown = k >= 2
        ax.barh(y - hgt / 2, trn if shown else 0, height=hgt, color=C["memory"],
                alpha=1.0 if shown else 0.0)
        ax.text(trn + 0.04, y - hgt / 2, f"{trn:.2f}×", va="center", fontsize=12,
                fontweight="bold", color=C["memory"], alpha=1.0 if shown else 0.0)
    ax.axvline(1.0, color=C["neutral"], lw=1.5, ls="--")
    ax.text(1.0, len(suites) - 0.45, " eager = 1×", color=C["neutral"], fontsize=10,
            fontweight="bold", va="bottom", ha="left")
    ax.text(2.0, 2.78, "inference", color=C["compute"], fontsize=12, fontweight="bold",
            ha="left", va="bottom")
    ax.text(2.75, 2.78, "training", color=C["memory"], fontsize=12, fontweight="bold",
            ha="left", va="bottom", alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(0, 3.3)
    ax.set_ylim(-0.6, len(suites) + 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Geometric-mean speedup over eager (A100, float32)")
    sp.save(fig, path)


for k in (1, 2):
    pt2_speedups(out / f"pt2_speedup_step{k}.svg", k)


# -- ResNet-50 training memory, log scale (three steps) ----------------------
def train_memory(path, k):
    w = V["resnet_weights_mb"]
    lo, hi = V["resnet_total_gb"]
    a_lo, a_hi = V["resnet_act_big_gb"]
    rows = [
        ("FP32 weights", w, w, "memory", f"{val('res_weights_mb')} MB"),
        (f"Training, batch {val('res_batch')}", lo * 1e3, hi * 1e3, "compute",
         f"{val('res_total_lo_gb')}–{val('res_total_hi_gb')} GB in all"),
        (f"Saved activations,\nbatch {val('res_big_batch')}", a_lo * 1e3, a_hi * 1e3, "danger",
         f"{val('res_big_act_lo_gb')}–{val('res_big_act_hi_gb')} GB"),
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
        ax.axvline(V["h100_gb"] * 1e3, ymin=0.0, ymax=0.42, color=C["neutral"], lw=1.5, ls="--")
        ax.text(V["h100_gb"] * 1e3 * 0.875, -0.72, f"H100: {val('h100_gb')} GB", ha="right", va="center", fontsize=10,
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
    ("HBM3, on device", V["bandwidth_gbps"]["HBM"], "memory", f"{val('hbm_gbps')} GB/s"),
    ("NVLink 4.0, GPU to GPU", V["bandwidth_gbps"]["NVLink"], "network", f"{val('nvlink_gbps')} GB/s each way"),
    ("PCIe Gen5 x16, host link", V["bandwidth_gbps"]["PCIe"], "network",
     f"{val('pcie_gbps')} GB/s: {val('hbm_over_pcie')}× below HBM"),
], "GB/s", log=False, xlabel="Bandwidth on one H100 (GB/s)")

# -- One MNIST training step: where the time goes (three steps) --------------
sp.hbars_steps(out / "mnist_step", [
    (f"Arithmetic, {val('step_mflop')} MFLOP", V["mnist_compute_us"], "compute", f"{val('step_compute_us')} μs"),
    (f"Memory traffic, {val('step_mb')} MB", V["mnist_memory_us"], "memory", f"{val('step_memory_us')} μs"),
    ("Host dispatch and launch", V["mnist_host_us"], "danger", f"{val('step_host_us')} μs"),
], "μs", xlabel="Time on an H100 (μs, log scale; lower bounds and a host model)",
    value_fmt="{:g}")

# -- ResNet-50 memory over two training steps, measured (two steps) ----------
def resnet_measured(path, k):
    rows, n_params = calcs.load_resnet_memory()
    w_gb = n_params * V["fp32_bytes"] / 1e9
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    show = [r for r in rows if r[1] <= k]
    xs = list(range(len(show)))
    gb = [r[4] / 1e9 for r in show]
    n1 = sum(1 for r in rows if r[1] == 1)
    for s in (1, 2):
        if s > k:
            continue
        idx = [i for i, r in enumerate(show) if r[1] == s]
        fwd = [i for i in idx if show[i][2] == "forward"]
        bwd = [i for i in idx if show[i][2] in ("backward", "after_backward")]
        ax.axvspan(fwd[0], fwd[-1], color=C["memory_l"], lw=0)
        ax.axvspan(bwd[0], bwd[-1], color=C["compute_l"], lw=0)
    ax.plot(xs, gb, color=C["memory"], lw=2.0)
    p1 = max(range(n1), key=lambda i: rows[i][4])
    ax.text(p1 + 6, rows[p1][4] / 1e9 + 0.05, f"peak {rows[p1][4] / 1e9:.1f} GB", fontsize=12,
            fontweight="bold", color=C["memory"], va="bottom")
    f1 = [i for i in range(n1) if rows[i][2] == "forward"]
    b1 = [i for i in range(n1) if rows[i][2] == "backward"]
    ax.text((f1[0] + f1[-1]) / 2, 6.55, "forward", ha="center", fontsize=12,
            fontweight="bold", color=C["memory"])
    ax.text((b1[0] + b1[-1]) / 2, 6.55, "backward", ha="center", fontsize=12,
            fontweight="bold", color=C["compute"])
    ax.axhline(w_gb, color=C["neutral"], lw=1.5, ls="--")
    ax.text(n1 * 0.5, w_gb + 0.12, f"weights {w_gb * 1e3:.0f} MB", ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=C["neutral"])
    if k >= 2:
        p2 = max(range(n1, len(rows)), key=lambda i: rows[i][4])
        ax.text(p2 + 6, rows[p2][4] / 1e9 + 0.05, f"{rows[p2][4] / 1e9:.1f} GB", fontsize=12,
                fontweight="bold", color=C["memory"], va="bottom")
    ax.set_xlim(0, len(rows))
    ax.set_ylim(0, 7.2)
    ax.set_xticks([n1 / 2, n1 + n1 / 2])
    ax.set_xticklabels(["training step 1", "training step 2"])
    ax.tick_params(axis="x", length=0, labelsize=12)
    ax.set_ylabel("Allocated (GB)")
    ax.set_xlabel("One sample after each layer's forward and backward", fontsize=11)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2):
    resnet_measured(out / f"resnet_measured_step{k}.svg", k)

print("figures written")
