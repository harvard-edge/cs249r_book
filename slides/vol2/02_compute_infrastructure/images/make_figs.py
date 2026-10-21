#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 2 (Compute Infrastructure).

Every value is from the rendered chapter (Compute Infrastructure, dev build of
18 October 2026) unless marked derived or illustrative:
  - Hook: Llama-3 70B batch-1 decode, 1.4e11 FLOPs at 989 TFLOP/s = 0.14 ms;
    141.2 GB of FP16 weights at 3.35 TB/s = 42.1 ms (Napkin Math 2.1).
  - Walls ladder: HBM 3.35 TB/s against 989 TFLOP/s FP16, ridge 295 FLOP/byte
    (Sec. 2.3); NVLink 450 GB/s against InfiniBand 50 GB/s per direction, 9x
    (Sec. 2.5.1); 33.5 kW four-node DGX H100 rack against 12 kW (Sec. 2.6);
    10,000 GPUs at a per-GPU MTTF of 50,000 h, about 5 h between GPU failures
    (Sec. 2.7.1, Sec. 2.11).
  - Two tests: 141.2 GB of 70B FP16 weights against one H100's 80 GB; decode
    floors 69.2, 42.1, 17.7 ms (calcs.py) at 2.04, 3.35, 8 TB/s (Sec. 2.3.1).
  - Roofline: H100 FP16 989 TFLOP/s, 3.35 TB/s, ridge 295.2 FLOP/byte;
    decode 1 FLOP/byte (0.34 percent of peak), batch 64 (about 22 percent),
    training with 2,048 local tokens (Sec. 2.3.2, Example 2.2).
  - H200 against H100: same peak, 1.4x bandwidth, 1.8x labeled capacity, modeled
    decode speedup 1.4x (Napkin Math 2.2).
  - Die arithmetic share: CPU 5-10, GPU 50-60, TPU 70-80, ASIC over 90
    percent, approximate shares that mark a direction (Sec. 2.2, figure 2).
  - Arithmetic per watt: Epoch AI Machine Learning Hardware, 64 devices with a
    dense FP16 rate and a TDP (via MLSysIM Literature.Series), fit about 1.2x
    per year; V100 0.42, A100 0.78, H100 1.41, B200 2.25 TFLOP/s/W (table 2).
  - Node state: 175B with mixed-precision Adam, 350 + 350 + 700 + 700 + 700 GB
    plus 100-400 GB activations (table 11); DGX H100 640 GB. Per GPU under
    TP-8 without ZeRO: optimizer 175, master 87.5, weights and gradients 87.5,
    total 375 GB; offloading optimizer and master leaves 112.5 GB (Example 2.4).
    The 25 GB of activations and workspace is derived as the remainder.
  - Staircase: 10 GB at 3,350, 450, 50 GB/s: 3.0, 22.2, 200 ms (Napkin 2.3).
  - Tensor-parallel AllReduce: 192 per step, 150.3 ms over NVLink, 1,057 ms
    over PCIe Gen5 at 64 GB/s, against an assumed 200 ms of compute (Sec.
    2.5.2, Example 2.3).
  - Rack power: 12 kW traditional, 33.5 kW four-node DGX H100 scenario,
    100 kW liquid class (Sec. 2.6.2 figure), GB200 NVL72 120 kW (Sec. 2.6.2).
  - Package and system power: MLSysIM Literature.Series.AcceleratorPowerRatings
    (14 parts, vendor conventions) and AISupercomputers (Epoch AI Data on AI
    Supercomputers; Pilz et al. 2025), as in the chapter's figure 13.
  - Cooling: 100 kW of air at 15 K needs 6.6 kg/s, 5.5 m^3/s (Sec. 2.6.2);
    water at 15 K: 100,000 / (4,180 x 15) = 1.6 kg/s, about 1.6 L/s (derived).
  - Pod chain: 3.15e23 FLOPs on 8,192 H100s at 1,979 TFLOP/s FP8 and 45
    percent local efficiency, 12.0 h; / 0.56 = 21.4; x 1.03 = 22.1; x 1.10 =
    24.3; x 1.05 = 25.5 h (Napkin Math 2.4). At 100 percent of the FP8 peak,
    5.4 h (derived).
  - Delivered share of peak: batch-1 decode bound 0.34 percent (FP16, Example
    2.2); assumed local efficiency 45 percent (FP8, Napkin Math 2.4); 8,192-GPU
    scenario 5.4 / 25.5 = 21 percent (derived).

Run from this folder: python3 make_figs.py
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402,F401  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
# Plots here are placed at about 0.75 of their drawn size, so type is set
# larger than slideplot's defaults to land at 10 pt or more on the slide.
plt.rcParams.update({"font.size": 15, "axes.labelsize": 14.5, "axes.titlesize": 15,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5})
out = Path(__file__).parent
MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"

# Volume II C3 palette for schematics (volumes/vol2/palette.md)
V2 = {
    "c1_fill": "#EAF1F8", "c1": "#1F407A", "c1_text": "#14294E",
    "c2_fill": "#fef3c7", "c2": "#d97706", "c2_text": "#b45309",
    "red_fill": "#fee2e2", "red": "#dc2626", "red_text": "#b91c1c",
    "frame_fill": "#f8fafc", "frame": "#94a3b8", "ink": "#0f172a",
    "green_fill": "#dcfce7", "green": "#16a34a",
}
FONT = "Helvetica, Arial, sans-serif"


def svg_write(path, body, w=900, h=500):
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{FONT}">\n'
        f'<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')


def text(x, y, s, size=24, color=V2["ink"], weight="normal", anchor="middle", op=1.0):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}" opacity="{op}">{s}</text>\n')


def rect(x, y, w, h, fill, stroke, sw=2, rx=6, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>\n')


def label(ax, x, y, s, color, shown=True, **kw):
    """Drawn in every step (invisible until shown) so the bounding box and
    the slide position stay fixed between steps."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, s, color=color, alpha=1.0 if shown else 0.0, **kw)


# 1. Hook: arithmetic against weight transfer, batch-1 decode ---------------
sp.hbars(out / "hook_decode.svg",
         [("Arithmetic", num("hook_t_arith_ms"), "compute", f"{val('hook_t_arith_ms')} ms"),
          ("Weights", num("hook_t_mem_ms"), "memory", f"{val('hook_t_mem_ms')} ms")],
         "ms", log=True, xlabel="ms per token (log)", size=(3.6, 1.45))

# 2. The four walls, one scale per step (900 x 500) -------------------------
walls = [
    ("Accelerator", "1 GPU", ("Memory", "wall"), f"{val('h100_bw')} TB/s", ("HBM feeding", f"{val('h100_fp16')} TFLOP/s"), "c1"),
    ("Node", "8 GPUs", ("Communication", "wall"), f"{val('nvlink_gbs')} \u2192 {val('ib_gbs')}", ("GB/s per GPU:", f"a {val('nvlink_ib_ratio')}\u00d7 drop"), "c2"),
    ("Rack", "32 GPUs", ("Power", "wall"), f"{val('dgx_rack_kw')} kW", ("per rack, against", f"{val('trad_rack_kw')} kW for servers"), "frame"),
    ("Pod", f"{val('pod_gpus')} GPUs", ("Reliability", "wall"), f"about {val('pod_fail_h')} h", ("between GPU", "failures"), "red"),
]
for k in range(1, 5):
    b = ""
    b += '<line x1="40" y1="34" x2="850" y2="34" stroke="#64748b" stroke-width="3"/>\n'
    b += '<polygon points="860,34 842,25 842,43" fill="#64748b"/>\n'
    b += text(40, 22, "Smaller physical scale", 21, "#475569", anchor="start")
    b += text(860, 22, "Larger physical scale", 21, "#475569", anchor="end")
    for i, (scale, n, wall, big, small, role) in enumerate(walls):
        op = 1.0 if i < k else 0.0
        x = 8 + i * 224
        cx = x + 106
        fill = {"c1": V2["c1_fill"], "c2": V2["c2_fill"], "frame": "#eef2f7",
                "red": V2["red_fill"]}[role]
        stroke = {"c1": V2["c1"], "c2": V2["c2"], "frame": "#475569", "red": V2["red"]}[role]
        tcol = {"c1": V2["c1_text"], "c2": V2["c2_text"], "frame": V2["ink"],
                "red": V2["red_text"]}[role]
        b += text(cx, 92, scale, 32, V2["ink"], "bold", op=op)
        b += text(cx, 126, n, 27, "#475569", op=op)
        b += f'<line x1="{cx}" y1="138" x2="{cx}" y2="166" stroke="#64748b" stroke-width="2.5" opacity="{op}"/>\n'
        b += f'<polygon points="{cx},176 {cx-8},162 {cx+8},162" fill="#64748b" opacity="{op}"/>\n'
        b += rect(x, 184, 212, 290, fill, stroke, 2.5, 8, op)
        b += text(cx, 228, wall[0], 25 if len(wall[0]) > 11 else 30, tcol, "bold", op=op)
        b += text(cx, 264, wall[1], 30, tcol, "bold", op=op)
        b += f'<line x1="{x+18}" y1="284" x2="{x+194}" y2="284" stroke="{stroke}" stroke-width="1.2" opacity="{op}"/>\n'
        b += text(cx, 340, big, 33, V2["ink"], "bold", op=op)
        b += text(cx, 404, small[0], 24, "#334155", op=op)
        b += text(cx, 438, small[1], 24, "#334155", op=op)
    svg_write(out / f"walls_step{k}.svg", b, h=478)

# 3. Two tests: capacity, then bandwidth (70B FP16) -------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6),
                                 gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.5})
    a1.bar([0], [num("llama70_weights_gb")], 0.55, color=C["memory"])
    a1.axhline(num("h100_hbm_gb"), color=C["danger"], lw=2, ls="--")
    a1.text(0.34, 84, f"one H100\n{val('h100_hbm_gb')} GB", color=C["danger"], fontsize=13.5,
            fontweight="bold", va="bottom", ha="left", linespacing=1.0)
    a1.text(0.12, 146, f"{val('llama70_weights_gb')} GB", ha="center", fontsize=14.5, fontweight="bold", color=C["memory"])
    a1.set_xticks([0], ["70B FP16\nweights"], fontsize=13.5)
    a1.set_xlim(-0.9, 1.6)
    a1.set_ylim(0, 175)
    a1.set_ylabel("GB")
    a1.set_title("Capacity: does it fit?", fontsize=14.5, fontweight="bold", color=C["text"])
    names = [f"A100\n{val('a100_bw')} TB/s", f"H100\n{val('h100_bw')} TB/s", f"B200\n{val('b200_bw')} TB/s"]
    vals = [num("floor_a100_ms"), num("hook_t_mem_ms"), num("floor_b200_ms")]
    shown = k >= 2
    a2.bar(range(3), vals, 0.6, color=C["memory"], alpha=1.0 if shown else 0.0)
    for i, v in enumerate(vals):
        label(a2, i, v + 3, f"{v:g} ms", C["memory"], shown, ha="center", fontsize=14.5)
    a2.set_xticks(range(3), names, fontsize=11.5)
    a2.set_ylim(0, 82)
    a2.set_ylabel("ms per token")
    a2.set_title("Bandwidth: how fast?", fontsize=14.5, fontweight="bold",
                 color=C["text"], alpha=1.0 if shown else 0.0)
    for s in ("left", "bottom"):
        a2.spines[s].set_alpha(1.0 if shown else 0.0)
    a2.tick_params(colors=C["text"] if shown else "white")
    a2.yaxis.label.set_alpha(1.0 if shown else 0.0)
    sp.save(fig, out / f"two_tests_step{k}.svg")

# 4. H100 roofline with real ceilings, three workload points ----------------
PEAK, BW = calcs.m(calcs.H100.compute.peak_flops, "TFLOP/s"), calcs.m(calcs.H100.memory.bandwidth, "TB/s")
RIDGE = PEAK / BW
pts = [(f"Decode, batch 1\n1 FLOP/byte: {val('b1_pct')}%", 1, "memory", (14, -6), "left"),
       (f"Decode, batch 64\n64 FLOP/byte: {val('b64_pct')}%", 64, "memory", (-14, 26), "right"),
       (f"Training, {val('tp_seq')}\nlocal tokens", num("tp_seq"), "compute", (0, -48), "center")]
for k in range(1, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    xs = np.logspace(-0.3, 4, 400)
    roof = np.minimum(PEAK, BW * xs)
    ax.plot(xs[xs <= RIDGE], roof[xs <= RIDGE], color=C["memory"], lw=3)
    ax.plot(xs[xs >= RIDGE], roof[xs >= RIDGE], color=C["compute"], lw=3)
    ax.axvline(RIDGE, color=C["neutral"], lw=1, ls="--")
    ax.text(RIDGE * 1.1, 0.32, f"ridge\n{val('ridge')} FLOP/byte", color=C["neutral"], fontsize=13.5,
            fontweight="bold", va="bottom")
    ax.text(5000, PEAK * 1.25, f"{val('h100_fp16')} TFLOP/s", color=C["compute"], fontsize=13.5,
            fontweight="bold", ha="right", va="bottom")
    ax.text(12, BW * 12 / 4.5, f"{val('h100_bw')} TB/s", color=C["memory"], fontsize=13.5,
            fontweight="bold", rotation=25, ha="center")
    for i, (lab, x, ck, off, ha) in enumerate(pts):
        shown = i < k
        y = min(PEAK, BW * x)
        ax.plot([x], [y], "o", ms=10, color=C[ck], mec="white", mew=1.5,
                zorder=5, alpha=1.0 if shown else 0.0)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=13, fontweight="bold", color=C[ck], va="top",
                    alpha=1.0 if shown else 0.0, linespacing=1.05)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 10000)
    ax.set_ylim(0.25, 4000)
    ax.set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Arithmetic intensity, FLOP per byte from HBM")
    ax.set_ylabel("TFLOP/s (FP16)")
    sp.save(fig, out / f"roofline_step{k}.svg")

# 5. H200 relative to H100 ----------------------------------------------------
fig, ax = plt.subplots(figsize=(3.9, 2.1))
rows = [("FP16 peak", 1.0, "compute"), ("HBM bandwidth", num("h200_bw_ratio"), "memory"),
        ("HBM capacity", num("h200_cap_ratio"), "memory"), ("Decode speed", num("h200_speedup"), "data")]
for i, (lab, v, ck) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, v, 0.6, color=C[ck])
    ax.text(v + 0.03, y, f"{v:.1f}×", va="center", fontsize=15.5, fontweight="bold", color=C[ck])
    ax.text(-0.03, y, lab, ha="right", va="center", fontsize=14.5, fontweight="bold", color=C["text"])
ax.axvline(1.0, color=C["neutral"], lw=1.2, ls="--")
ax.set_yticks([])
ax.set_xlim(0, 1.95)
ax.spines["left"].set_visible(False)
ax.set_xlabel("H200 / H100")
sp.save(fig, out / "h200_ratio.svg")

# 6. Die arithmetic share (approximate direction) ----------------------------
fig, ax = plt.subplots(figsize=(5.4, 2.6))
rows = [("CPU", 5, 10), ("GPU", 50, 60), ("TPU", 70, 80), ("Custom ASIC", 90, 97)]
for i, (lab, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, lo, 0.6, color=C["compute"])
    ax.barh(y, hi - lo, 0.6, left=lo, color=C["compute_l"], edgecolor=C["compute"])
    s = f"{lo} to {hi}%" if lab != "Custom ASIC" else "over 90%"
    ax.text(hi + 2, y, s, va="center", fontsize=16.5, fontweight="bold", color=C["compute"])
    ax.text(-2, y, lab, ha="right", va="center", fontsize=16.5, fontweight="bold", color=C["text"])
ax.set_yticks([])
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Share of die for arithmetic, %", fontsize=15.5)
ax.tick_params(labelsize=12)
sp.save(fig, out / "die_share.svg")


# MLSysIM data series ----------------------------------------------------------
sys.path.insert(0, str(MLSYSIM))
from mlsysim import Literature  # noqa: E402


def dyear(s):
    d = date.fromisoformat(s[:10])
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


# 7. Arithmetic per watt, Epoch AI ---------------------------------------------
rows = Literature.Series.AcceleratorFlopsPerWatt.entries
xs = np.array([dyear(r["release_date"]) for r in rows])
ys = np.array([r["flops_per_w"] / 1e12 for r in rows])
kfit, bfit = np.polyfit(xs, np.log10(ys), 1)
annual = 10 ** kfit
fig, ax = plt.subplots(figsize=(5.8, 2.6))
nv = np.array(["NVIDIA" in r["manufacturer"] for r in rows])
ax.scatter(xs[~nv], ys[~nv], s=26, color=C["neutral_l"], edgecolor=C["neutral"], lw=0.6, zorder=3)
ax.scatter(xs[nv], ys[nv], s=26, color=C["data"], zorder=3)
xx = np.linspace(2017, 2026.3, 50)
ax.plot(xx, 10 ** (kfit * xx + bfit), "--", color=C["text"], lw=1.4, zorder=2)
ax.text(2026.4, 0.118, f"dashed fit: {annual:.1f}× per year",
        fontsize=13.5, fontweight="bold", ha="right", color=C["text"])
for lab, name, off in ((f"V100 {val('perwatt_v100')}", "NVIDIA Tesla V100 SXM2", (60, -26)),
                       (f"A100 {val('perwatt_a100')}", "NVIDIA A100 SXM4 40 GB", (-4, 14)),
                       (f"H100 {val('perwatt_h100')}", "NVIDIA H100 SXM5 80GB", (-8, 14)),
                       (f"B200 {val('perwatt_b200')}", "NVIDIA B200", (-8, 12))):
    r = next(r for r in rows if r["device"] == name)
    p = (dyear(r["release_date"]), r["flops_per_w"] / 1e12)
    ax.scatter([p[0]], [p[1]], s=110, facecolors="none", edgecolors=C["data"], lw=1.6, zorder=4)
    ax.annotate(lab, p, xytext=off, textcoords="offset points", ha="right",
                fontsize=13.5, fontweight="bold", color=C["data"])
ax.set_title("green: NVIDIA    gray: other vendors (Epoch AI, 64 devices)", fontsize=13,
             color=C["neutral"], fontweight="bold")
ax.set_yscale("log")
ax.set_xlim(2016.6, 2026.6)
ax.set_ylim(0.1, 4.5)
ax.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
ax.minorticks_off()
ax.set_xlabel("Release date")
ax.set_ylabel("Dense FP16 TFLOP/s per W")
sp.save(fig, out / "perwatt.svg")
print(f"perwatt: {len(rows)} devices, fit {annual:.2f}x per year")

# 8. Node memory: the 175B state against a node, then one GPU under TP-8 -------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4),
                                 gridspec_kw={"width_ratios": [1, 1], "wspace": 0.6})
    parts = [("weights", num("weights_gb"), C["memory"]), ("gradients", num("weights_gb"), C["network"]),
             ("master", num("master_gb"), C["neutral"]), ("Adam m", num("adam_moment_gb"), C["compute"]),
             ("Adam v", num("adam_moment_gb"), "#B85A0B")]
    bot = 0
    for lab, v, col in parts:
        a1.bar(0, v, 0.6, bottom=bot, color=col, edgecolor="white")
        a1.text(0.36, bot + v / 2, lab, va="center", fontsize=12.5, color=col, fontweight="bold")
        bot += v
    a1.bar(0, 250, 0.6, bottom=bot, color=C["data_l"], edgecolor=C["data"], hatch="//")
    a1.text(0.36, bot + 125, "activations", va="center", fontsize=12.5,
            color=C["data"], fontweight="bold", linespacing=1.0)
    a1.axhline(num("node_hbm_gb"), color=C["danger"], lw=2, ls="--")
    a1.text(-1.85, 700, f"node\n{val('node_hbm_gb')} GB", color=C["danger"], fontsize=13, fontweight="bold",
            ha="left", va="bottom", linespacing=1.0)
    a1.text(0, 3330, f"{val('state_tb')} TB + act.", ha="center", fontsize=13.5, fontweight="bold", color=C["text"])
    a1.set_xlim(-1.9, 1.5)
    a1.set_ylim(0, 3700)
    a1.set_xticks([])
    a1.set_ylabel("GB, whole model")
    a1.set_title("175B training state", fontsize=14.5, fontweight="bold", color=C["text"])
    shown = k >= 2
    al = 1.0 if shown else 0.0
    gp = [("weights, grads", num("tp8_wg_gb"), C["memory"]), ("master", num("tp8_master_gb"), C["neutral"]),
          ("Adam m, v", num("tp8_adam_gb"), C["compute"]), ("other", num("tp8_other_gb"), C["data"])]
    bot = 0
    for lab, v, col in gp:
        a2.bar(0, v, 0.6, bottom=bot, color=col, edgecolor="white", alpha=al)
        label(a2, 0.36, bot + v / 2, lab, col, shown, va="center", fontsize=12.5)
        bot += v
    a2.axhline(num("h100_hbm_gb"), color=C["danger"], lw=2, ls="--", alpha=al)
    label(a2, -1.85, 92, f"H100\n{val('h100_hbm_gb')} GB", C["danger"], shown, fontsize=13, ha="left",
          va="bottom", linespacing=1.0)
    label(a2, 0, 385, f"{val('tp8_total_gb')} GB", C["text"], shown, ha="center", fontsize=13.5)
    a2.set_xlim(-1.9, 1.5)
    a2.set_ylim(0, 430)
    a2.set_xticks([])
    a2.set_ylabel("GB per GPU", alpha=al)
    a2.set_title("One GPU, TP-8, no ZeRO", fontsize=14.5, fontweight="bold",
                 color=C["text"], alpha=al)
    for s in ("left", "bottom"):
        a2.spines[s].set_alpha(al)
    a2.tick_params(colors=C["text"] if shown else "white")
    sp.save(fig, out / f"node_state_step{k}.svg")

# 9. The bandwidth staircase: one 10 GB buffer --------------------------------
sp.hbars_steps(str(out / "staircase"),
               [(f"HBM, {val('h100_bw_gbs')} GB/s", num("stair_hbm_ms"), "memory", f"{val('stair_hbm_ms')} ms"),
                (f"NVLink, {val('nvlink_gbs')} GB/s", num("stair_nvlink_ms"), "compute", f"{val('stair_nvlink_ms')} ms"),
                (f"InfiniBand, {val('ib_gbs')} GB/s", num("stair_ib_ms"), "network", f"{val('stair_ib_ms')} ms")],
               "ms", log=True, xlabel=f"Time to move {val('buffer_gb')} GB, ms (log)", size=(4.3, 2.5))

# 10. Placement: three strategies on two tiers (900 x 500) ---------------------
lanes = [
    ("Tensor parallel", f"Inside one node: NVLink, {val('nvlink_gbs')} GB/s",
     f"{val('tp_payload_mb')} MB, 192 times per step", "on the critical path", "c1"),
    ("Pipeline parallel", f"Between nodes: InfiniBand, {val('ib_gbs')} GB/s",
     f"{val('tp_payload_mb')} MB per micro-batch per stage boundary", "partly hidden", "c2"),
    ("Data parallel", f"Across the cluster: InfiniBand, {val('ib_gbs')} GB/s",
     f"{val('weights_gb')} GB of gradients, once per step", "overlaps backward", "green"),
]
for k in range(1, 4):
    b = ""
    for i, (name, where, size, overlap, role) in enumerate(lanes):
        op = 1.0 if i < k else 0.0
        y = 6 + i * 148
        fill = {"c1": V2["c1_fill"], "c2": V2["c2_fill"], "green": V2["green_fill"]}[role]
        stroke = {"c1": V2["c1"], "c2": V2["c2"], "green": V2["green"]}[role]
        b += rect(6, y, 888, 138, fill, stroke, 2.5, 8, op)
        b += text(24, y + 42, name, 34, stroke, "bold", "start", op)
        b += text(24, y + 84, where, 28, V2["ink"], "bold", "start", op)
        b += text(24, y + 122, size + "; " + overlap, 26, "#334155", "normal", "start", op)
    svg_write(out / f"placement_step{k}.svg", b, h=450)

# 11. Tensor-parallel AllReduce over NVLink and PCIe ---------------------------
fig, ax = plt.subplots(figsize=(5.8, 2.0))
for y, comm, lab, pct, step in ((1, num("tp_nvlink_total_ms"), "NVLink", f"compute {val('tp_nvlink_compute_pct')}%", val("tp_nvlink_step_ms")),
                                (0, num("tp_pcie_total_ms"), "PCIe Gen5", f"compute {val('tp_pcie_compute_pct')}%", val("tp_pcie_step_ms"))):
    ax.barh(y, num("tp_compute_ms"), 0.6, color=C["compute"])
    ax.barh(y, comm, 0.6, left=num("tp_compute_ms"), color=C["network"])
    ax.text(-15, y, lab, ha="right", va="center", fontsize=15.5, fontweight="bold", color=C["text"])
    ax.text(num("tp_compute_ms") + comm + 20, y, f"{step} ms\n{pct}", va="center", fontsize=14,
            fontweight="bold", color=C["text"], linespacing=1.0)
ax.text(num("tp_compute_ms") / 2, 1.48, f"compute {val('tp_compute_ms')} ms (assumed)", ha="center", fontsize=13, fontweight="bold",
        color=C["compute"])
ax.text(730, 0, f"{val('tp_collectives')} AllReduces", ha="center", va="center", fontsize=13.5, fontweight="bold", color="white")
ax.set_yticks([])
ax.set_xlim(0, 1550)
ax.set_ylim(-0.5, 1.75)
ax.spines["left"].set_visible(False)
ax.set_xlabel("ms per step, no overlap")
sp.save(fig, out / "tp_links.svg")

# 12. Rack power ---------------------------------------------------------------
sp.hbars(out / "rack_power.svg",
         [("Server rack", num("trad_rack_kw"), "neutral", f"{val('trad_rack_kw')} kW"),
          ("4 DGX H100", num("dgx_rack_kw"), "compute", f"{val('dgx_rack_kw')} kW"),
          ("Liquid class", num("rack_heat_kw"), "danger", f"{val('rack_heat_kw')} kW"),
          ("GB200 NVL72", num("nvl72_kw"), "danger", f"{val('nvl72_kw')} kW")],
         "kW", log=False, xlabel="kW per rack", size=(4.2, 2.2))

# 13. Package power against system power ---------------------------------------
chips = Literature.Series.AcceleratorPowerRatings.entries
systems = [r for r in Literature.Series.AISupercomputers.entries if r["power_mw"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.6), gridspec_kw={"wspace": 0.45})
cx = [r["release_year"] for r in chips]
cy = [r["power_w"] for r in chips]
a1.scatter(cx, cy, s=30, color=C["compute"], zorder=3)
for r in chips:
    if r["label"] in ("H100", "B300"):
        a1.annotate(f"{r['label']} {r['power_w']:,.0f} W", (r["release_year"], r["power_w"]),
                    xytext=(-6, 8), textcoords="offset points", ha="right",
                    fontsize=12.5, fontweight="bold", color=C["compute"])
a1.set_ylim(0, 1650)
a1.set_xlim(2011, 2026.5)
a1.set_xticks([2013, 2017, 2021, 2025])
a1.set_ylabel("W per package")
a1.set_title(f"One accelerator: {val('chip_growth')}×", fontsize=14.5, fontweight="bold", color=C["text"])
sx = np.array([dyear(r["first_operational_date"]) for r in systems])
syv = np.array([r["power_mw"] for r in systems])
o = np.argsort(sx)
a2.scatter(sx, syv, s=8, color=C["neutral_l"], zorder=2)
run = np.maximum.accumulate(syv[o])
a2.step(sx[o], run, where="post", color=C["danger"], lw=2.2, zorder=3)
a2.annotate(f"Frontier {val('frontier_mw')} MW", (2022.4, num("frontier_mw")), xytext=(2013.0, 150), textcoords="data",
            ha="left", fontsize=12.5, fontweight="bold", color=C["danger"],
            arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1))
a2.annotate(f"{val('largest_2025_mw')} MW", (2025.5, num("largest_2025_mw")), xytext=(-4, 8), textcoords="offset points",
            ha="right", fontsize=12.5, fontweight="bold", color=C["danger"])
a2.set_yscale("log")
a2.set_xlim(2011, 2026.5)
a2.set_ylim(0.3, 1500)
a2.set_xticks([2013, 2017, 2021, 2025])
a2.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
a2.minorticks_off()
a2.set_ylabel("MW per system")
a2.set_title(f"Largest AI system: {val('largest_over_frontier')}×", fontsize=14.5, fontweight="bold", color=C["text"])
sp.save(fig, out / "power_trend.svg")
print(f"power_trend: {len(chips)} packages, {len(systems)} systems")

# 14. Cooling: volumetric flow to remove 100 kW at a 15 K rise -----------------
fig, ax = plt.subplots(figsize=(3.8, 1.5))
air_ls = num("air_m3s") * 1000
for y, lab, v, ck, txt in ((1, "Air", air_ls, "neutral", f"{air_ls:,.0f} L/s"),
                           (0, "Water", num("water_kgs"), "memory", f"{val('water_kgs')} L/s")):
    ax.barh(y, v, 0.6, color=C[ck])
    ax.text(v * 1.3, y, txt, va="center", fontsize=15, fontweight="bold", color=C[ck])
    ax.text(0.55, y, lab + " ", ha="right", va="center", fontsize=15, fontweight="bold", color=C["text"])
ax.set_xscale("log")
ax.set_xlim(0.6, 3e5)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([1, 100, 10000], ["1", "100", "10,000"])
ax.minorticks_off()
ax.set_xlabel(f"L/s for {val('rack_heat_kw')} kW, {val('cool_rise_k')} K rise")
sp.save(fig, out / "cooling_flow.svg")

# 15. Nested failure domains (column slot, 520 x 470) --------------------------
doms = [("Building", "fire, utility feed"),
        ("Hall, row", "row power, coolant"),
        ("Rack", "power shelf, switch"),
        ("Node", "one GPU of 8"),
        ("GPU", "an HBM error")]
b = ""
for i, (nm, what) in enumerate(doms):
    x, y = 6 + i * 22, 6 + i * 84
    w, h = 508 - i * 44, 458 - i * 84
    shade = ["#f8fafc", "#f1f5f9", "#e2e8f0", "#EAF1F8", "#d6e2f0"][i]
    stroke = V2["red"] if i == 0 else ("#64748b" if i < 3 else V2["c1"])
    b += rect(x, y, w, h, shade, stroke, 2.5, 8)
    b += text(x + 12, y + 36, nm, 31, V2["red_text"] if i == 0 else stroke, "bold", "start")
    b += text(x + w - 12, y + 36, what, 27, "#334155", "normal", "end")
svg_write(out / "failure_domains.svg", b, w=520, h=470)

# 16. Pod chain: 175B on 8,192 H100s ----------------------------------------
chain = [("Full\npeak", num("chain_peak_h"), "neutral"), (f"Local\n{val('local_eff_pct')}%", num("chain_compute_h"), "compute"),
         (f"Scaling\n{val('scale_eff')}", num("chain_scaled_h"), "network"), (f"Checkpoint\n+{val('ovh_ckpt_pct')}%", num("chain_ckpt_h"), "memory"),
         (f"Failures\n+{val('ovh_fail_pct')}%", num("chain_fail_h"), "danger"), (f"Upkeep\n+{val('ovh_maint_pct')}%", num("chain_total_h"), "neutral")]
for k, upto in ((1, 2), (2, 3), (3, 6)):
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    prev = 0
    for i, (lab, v, ck) in enumerate(chain):
        shown = i < upto
        a = 1.0 if shown else 0.0
        if i == 0 or i == 1:
            ax.bar(i, v, 0.62, color=C[ck], alpha=a)
        else:
            ax.bar(i, prev, 0.62, color=C["neutral_l"], alpha=a)
            ax.bar(i, v - prev, 0.62, bottom=prev, color=C[ck], alpha=a)
        label(ax, i, v + 0.8, f"{v:.1f} h", C["text"], shown, ha="center", fontsize=14.5)
        prev = v
    ax.set_xticks(range(len(chain)), [c[0] for c in chain], fontsize=13)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Hours of wall clock")
    ax.text(-0.4, 28.2, "derived", fontsize=12.5, color=C["neutral"], fontweight="bold")
    sp.save(fig, out / f"pod_chain_step{k}.svg")

# 17. Delivered share of peak --------------------------------------------------
sp.hbars(out / "delivered.svg",
         [("Decode, batch 1", num("b1_pct"), "memory", f"{val('b1_pct')}%"),
          ("Local training", num("local_eff_pct"), "compute", f"{val('local_eff_pct')}%"),
          ("8,192-GPU run", num("chain_share_pct"), "network", f"{val('chain_share_pct')}%")],
         "%", log=False, xlabel="Share of peak that becomes progress, %",
         size=(4.2, 1.9))

# 18. ZeRO memory per GPU against one H100 (frame "Adding 128-way data
#     parallelism"): weights, then Adam moments, then the subtotal. Added
#     2026-10-21 so the comparison the frame computes is visible. Drawn at
#     its slot (about 8.5 x 3 cm) so type lands near 10 pt.
_W, _A, _CAP = num("tp8_weights_gb"), num("zero_adam_gb"), num("h100_hbm_gb")
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(3.4, 1.15))
    ax.barh(0, _W, 0.6, color=C["memory"])
    label(ax, _W / 2, 0, f"weights {val('tp8_weights_gb')}", "white", ha="center", va="center",
          fontsize=11)
    ax.barh(0, _A, 0.6, left=_W, color=C["compute"], alpha=1.0 if k >= 2 else 0.0)
    label(ax, _W + _A + 1.5, 0.45, f"+ Adam {val('zero_adam_gb')}", C["compute"], k >= 2,
          va="bottom", fontsize=11)
    label(ax, _W + _A + 1.5, -0.05, f"= {val('zero_subtotal_gb')} GB", C["text"], k >= 3,
          va="center", fontsize=12)
    ax.axvline(_CAP, color=C["danger"], lw=2, ls="--")
    label(ax, _CAP - 1, -0.65, f"H100 {val('h100_hbm_gb')} GB", C["danger"], ha="right",
          va="center", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.9, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GB per GPU", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    sp.save(fig, out / f"zero_mem_step{k}.svg")

# 19. The power step and its voltage droop on one H100 core rail (frame "A
#     barrier swings every GPU"). Shape is schematic; the levels, rise times,
#     and droops are the chapter's scenario values. Added 2026-10-21; drawn
#     at its slot (about 8.3 x 4.2 cm).
_lo, _hi = num("step_low_w"), num("tdp_h100")
_ts, _tf = num("rise_slow_us"), num("rise_fast_ns") / 10 ** 3   # microseconds
_ds, _df = num("droop_1us_mv"), num("droop_100ns_mv")
_t0 = 0.5
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 1.85), sharex=True,
                             gridspec_kw={"hspace": 0.3})
for ramp, col in ((_ts, C["memory"]), (_tf, C["danger"])):
    a1.plot([0, _t0, _t0 + ramp, 2.2], [_lo, _lo, _hi, _hi], color=col, lw=2.2)
a1.set_ylim(_lo - 80, _hi + 190)
a1.set_yticks([_lo, _hi])
a1.set_ylabel("W", fontsize=10)
a1.text(_t0 + _ts + 0.05, _hi - 170, f"{val('rise_slow_us')} µs edge", color=C["memory"],
        fontsize=10.5, fontweight="bold")
a1.text(_t0 + _tf + 0.04, _hi + 25, f"{val('rise_fast_ns')} ns edge", color=C["danger"],
        fontsize=10.5, fontweight="bold", va="bottom")
for ramp, d, col in ((_ts, _ds, C["memory"]), (_tf, _df, C["danger"])):
    a2.plot([0, _t0, _t0, _t0 + ramp, _t0 + ramp, 2.2], [0, 0, -d, -d, 0, 0], color=col, lw=2.2)
a2.text(_t0 + _ts + 0.05, -_ds - 8, f"−{val('droop_1us_mv')} mV", color=C["memory"],
        fontsize=10.5, fontweight="bold", va="top")
a2.text(_t0 + _tf + 0.06, -_df + 10, f"−{val('droop_100ns_mv')} mV", color=C["danger"],
        fontsize=10.5, fontweight="bold", va="bottom")
a2.set_ylim(-_df - 30, 30)
a2.set_yticks([0, -round(_df, -2)])
a2.set_ylabel("mV", fontsize=10)
a2.set_xlim(0, 2.2)
a2.set_xlabel("time (µs)", fontsize=10)
for a in (a1, a2):
    a.tick_params(labelsize=9.5)
sp.save(fig, out / "droop.svg")
