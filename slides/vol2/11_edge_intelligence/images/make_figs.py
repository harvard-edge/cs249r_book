#!/usr/bin/env python3
"""Figures for Volume II, Lecture 11 (Edge Intelligence).

Every number on a figure comes from MLSysIM through calcs.py (val/num) or
from the MLSysIM scenario or Literature entry calcs.py names; positions and
sizes are layout only. "EI" is the rendered chapter, Edge Intelligence.
Scenario values are labeled "illustrative", as the chapter labels them.

  - Hook: Hard et al., "Federated Learning for Mobile Keyboard Prediction,"
    arXiv:1811.03604 (2018), Table 4, top-1 recall on client-owned caches:
    n-gram 12.5 +/- 0.2, server CIFG 15.0 +/- 0.5, federated CIFG
    15.8 +/- 0.3 percent. Read from the paper.
  - Device boundary and alternatives: EI figure 2 and table 1 (redrawn).
  - Deadlines: EI Sec. 11.2.1 (control 10 ms, AR/VR 20 ms, camera 33 ms,
    voice 500 ms; network round trip 50 to 200 ms).
  - Training stack: EI figure 3 (10M parameters FP32: weights 40 MB,
    gradients 40, Adam 80, activations 200 assumed, 360 MB, 9x) and the
    keyboard allowance 200 to 300 MB (Sec. 11.3.1).
  - Memory ladder: EI Sec. 11.3.4 figure (phone 16 GB, IoT device up to
    1 GB, microcontroller 256 KB).
  - Apple Neural Engine ratings: MLSysIM vol2/11_edge_intelligence/data/apple_neural_engine.csv
    (Apple statements; A13 derived; precision never stated), as in EI
    figure 5. A18 Pro (secondary source) is left out, as in the book.
  - Edge inference energy: MLSysIM Literature.EdgeInferenceBenchmarks
    (MLPerf Tiny and vendor data), as in EI figure 6. Training-second band
    2 to 3 J from the 2 to 3 W sustained envelope (Sec. 11.3.2, 11.3.4).
  - Budget terms: EI equation 1 (schematic, not to scale).
  - Personalization: EI Napkin Math 11.2 (400 MB against 42 MB).
  - Battery: EI Napkin Math 11.1 (2.25 Wh of 15 Wh).
  - Round timeline: EI Sec. 11.1 and figure 10 (redrawn, schematic).
  - FedAvg rounds: McMahan et al., AISTATS 2017, Table 2 (MNIST CNN, rounds
    to 99 percent; FedSGD 626 IID, 483 non-IID; FedAvg rows by u).
  - Uplink: EI Napkin Math 11.3 (200 MB raw per week; 2.5 MB update;
    break-even 80 rounds; 7 and 168 rounds a week).
  - Secure aggregation: EI figure 16 and Sec. 11.7.4 (redrawn; cell values
    illustrative).
  - Session outcomes: Bonawitz et al., "Towards Federated Learning at Scale,"
    MLSys 2019, Appendix A Table 1 (1,116,401 / 327,478 / 29,771).
  - AI Benchmark: MLSysIM vol2/11_edge_intelligence/data/ai_benchmark_socs.csv (ETH Zurich
    AI Benchmark ranking, snapshot 2026-10-12), as in EI figure 17.
  - Coverage funnel: illustrative cohort shares, after EI figure 13.
  - Forgetting: illustrative curves, after EI Sec. 11.5.2 figure.
  - Shadow gate: EI figure 20 (redrawn; letters illustrative).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import calcs as K  # noqa: E402  (records every value; figures read them)
from mlsysim.core.series_loader import load_series  # noqa: E402
from deckvals import val, num  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from slideplot import C  # noqa: E402,F401  (sets the shared rcParams)

# Volume II C3 schematic roles (volumes/vol2/palette.md)
V = {
    "c1": "#1F407A", "c1_l": "#EAF1F8",          # compute and substrate
    "c2": "#d97706", "c2_l": "#fef3c7", "c2_t": "#b45309",  # communication
    "c3": "#7c3aed", "c3_l": "#f3e8ff", "c3_t": "#6d28d9",  # coordination
    "ok": "#16a34a", "ok_l": "#dcfce7", "ok_t": "#15803d",  # data, healthy
    "bad": "#dc2626", "bad_l": "#fee2e2", "bad_t": "#b91c1c",  # risk
    "fr": "#94a3b8", "fr_l": "#f8fafc", "ink": "#0f172a", "mute": "#64748b",
}
TXT = C["text"]


def out(fig, name, tight=True):
    fig.savefig(HERE / f"{name}.svg", format="svg",
                bbox_inches="tight" if tight else None,
                pad_inches=0.08, facecolor="white")
    plt.close(fig)


def canvas(w, h):
    """A diagram canvas in inches: data units equal inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fill, stroke, text="", fs=12, color=None, bold=False,
        dashed=False, alpha=1.0, r=0.05, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=stroke, lw=lw, ls="--" if dashed else "-",
                                alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=color or TXT, fontweight="bold" if bold else "normal", alpha=alpha)


def arrow(ax, x0, y0, x1, y1, color, lw=1.8, alpha=1.0, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=14, alpha=alpha,
                                shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------
# Hook: Gboard recall (Hard et al. 2018, Table 4)
# --------------------------------------------------------------------------
def hook():
    rows = [("N-gram (shipped)", num("gb_ngram"), num("gb_ngram_ci"), C["neutral"]),
            ("Data-center RNN", num("gb_server"), num("gb_server_ci"), C["memory"]),
            ("Federated RNN", num("gb_fed"), num("gb_fed_ci"), C["data"])]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        for i, (lab, v, e, col) in enumerate(rows):
            y = len(rows) - 1 - i
            shown = step == 2 or i < 2
            ax.barh(y, v if shown else 0, color=col, height=0.6)
            ax.errorbar(v, y, xerr=e, color=TXT, capsize=4, lw=1.4,
                        alpha=1 if shown else 0)
            ax.text(-0.3, y, lab, ha="right", va="center", fontsize=16,
                    fontweight="bold", color=TXT)
            ax.text(v + 0.8, y, f"{v:.1f}%" if shown else "", va="center",
                    fontsize=16, fontweight="bold", color=col)
            if not shown:
                ax.text(1.0, y, "?", va="center", fontsize=22, fontweight="bold",
                        color=C["danger"])
        ax.set_xlim(0, 19.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([])
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Top-1 next-word recall on phones' own typing (%)", fontsize=14)
        out(fig, f"hook_step{step}")
    # thumbnail for goals, synthesis, takeaways
    fig, ax = plt.subplots(figsize=(3.0, 1.3))
    for i, (lab, v, e, col) in enumerate(rows):
        ax.barh(2 - i, v, color=col, height=0.6)
    ax.set_xlim(0, 17)
    ax.axis("off")
    out(fig, "hook_thumb")


# --------------------------------------------------------------------------
# Fleet Law read for one federated round (EI Sec. 11.1)
# --------------------------------------------------------------------------
def fleetlaw_edge():
    fig, ax = canvas(4.8, 3.5)
    ax.text(2.4, 3.25, r"$T_{\mathrm{round}} = T_{\mathrm{compute}} + T_{\mathrm{comm}} + T_{\mathrm{sync}} - T_{\mathrm{overlap}}$",
            ha="center", va="center", fontsize=15, color=TXT)
    terms = [
        ("Compute", "training on the slowest\nphone the round awaits\n(not divided by N)", V["c1_l"], V["c1"]),
        ("Communication", f"update upload over\na {val('uplink_lo')} to {val('uplink_hi')} Mb/s uplink", V["c2_l"], V["c2"]),
        ("Synchronization", "wait until K valid\nreports arrive", V["c3_l"], V["c3"]),
        ("Overlap", "download and training\nthat run concurrently", V["fr_l"], V["fr"]),
    ]
    w, h = 2.3, 1.4
    for i, (t, d, f, st) in enumerate(terms):
        x = 0.05 + (i % 2) * (w + 0.1)
        y = 1.55 - (i // 2) * (h + 0.1)
        box(ax, x, y, w, h, f, st)
        ax.text(x + w / 2, y + h - 0.25, t, ha="center", va="center", fontsize=14,
                fontweight="bold", color=st if st != V["fr"] else V["mute"])
        ax.text(x + w / 2, y + 0.55, d, ha="center", va="center", fontsize=13, color=TXT,
                linespacing=1.2)
    out(fig, "fleetlaw_edge")


# --------------------------------------------------------------------------
# What crosses the device boundary (EI figure 2, table 1)
# --------------------------------------------------------------------------
def boundary():
    W, H = 6.95, 2.75
    panels = [
        ("Frozen model\n+ local features", "nothing trains", "0"),
        ("Local-only", "nothing leaves", "A"),
        ("Centralized", "raw data leaves", "B"),
        ("Federated", "only updates leave", "C"),
    ]
    pw = 1.5
    for step in (1, 2, 3):
        fig, ax = canvas(W, H)
        yb = 1.45  # device boundary
        ax.plot([0.05, W - 0.05], [yb, yb], ls="--", color=V["mute"], lw=1.4)
        ax.text(0.05, yb + 0.07, "server", ha="left", va="bottom", fontsize=12.6, color=V["mute"])
        ax.text(0.05, yb - 0.07, "device", ha="left", va="top", fontsize=12.6, color=V["mute"])
        for i, (title, foot, tag) in enumerate(panels):
            show = (i == 0) or (step >= 2 and i in (1, 2)) or (step == 3 and i == 3)
            a = 1.0 if show else 0.0
            x = 0.62 + i * (pw + 0.08)
            ax.text(x + pw / 2, H - 0.12, f"{tag}. {title}" if tag != "0" else title,
                    ha="center", va="top", fontsize=13.8, fontweight="bold", color=TXT,
                    alpha=a, linespacing=1.05)
            # device with data squares
            box(ax, x + 0.15, 0.42, pw - 0.3, 0.75, V["fr_l"], V["fr"], alpha=a)
            for k in range(4):
                ax.add_patch(Rectangle((x + 0.28 + k * 0.17, 0.55), 0.13, 0.13,
                                       fc=V["ok_l"], ec=V["ok"], lw=1, alpha=a))
            ax.text(x + pw / 2, 0.98, "phone", ha="center", va="center", fontsize=12.6,
                    color=V["mute"], alpha=a)
            ax.text(x + pw / 2, 0.2, foot, ha="center", va="center", fontsize=12.2,
                    color=V["c1"] if i != 2 else V["bad_t"], fontweight="bold", alpha=a)
            cx = x + pw / 2
            if i == 0:
                box(ax, x + 0.25, 1.65, pw - 0.5, 0.45, V["c1_l"], V["c1"],
                    "released\nmodel", fs=12.1, alpha=a)
                arrow(ax, cx, 1.62, cx, 1.22, V["mute"], alpha=a)
            if i == 1:
                ax.annotate("", xy=(x + pw - 0.2, 1.0), xytext=(x + pw - 0.2, 0.62),
                            arrowprops=dict(arrowstyle="-|>", color=V["c1"], lw=1.6,
                                            connectionstyle="arc3,rad=-1.2", alpha=a))
                ax.text(x + pw - 0.02, 0.82, "train", ha="left", va="center", fontsize=11.5,
                        color=V["c1"], alpha=a)
            if i == 2:
                box(ax, x + 0.2, 1.65, pw - 0.4, 0.45, V["c1_l"], V["c1"],
                    "cloud trainer", fs=12.6, alpha=a)
                arrow(ax, cx, 1.2, cx, 1.62, V["bad"], alpha=a, lw=2.2)
                ax.add_patch(Rectangle((cx + 0.07, 1.33), 0.12, 0.12, fc=V["ok_l"],
                                       ec=V["ok"], alpha=a))
            if i == 3:
                box(ax, x + 0.2, 1.65, pw - 0.4, 0.45, V["c3_l"], V["c3"],
                    "FedAvg", fs=13, alpha=a)
                arrow(ax, cx - 0.22, 1.2, cx - 0.22, 1.62, V["c2"], alpha=a, lw=2.2)
                arrow(ax, cx + 0.22, 1.62, cx + 0.22, 1.2, V["mute"], alpha=a, lw=1.8)
                ax.text(cx - 0.3, 1.33, "Δθ", ha="right", va="center", fontsize=13.8,
                        color=V["c2_t"], alpha=a, fontweight="bold")
                ax.text(cx + 0.3, 1.33, "θ", ha="left", va="center", fontsize=13.8,
                        color=V["mute"], alpha=a, fontweight="bold")
        out(fig, f"boundary_step{step}", tight=False)


# --------------------------------------------------------------------------
# Deadlines against the network round trip (EI Sec. 11.2.1)
# --------------------------------------------------------------------------
def deadlines():
    rows = [("Safety control", num("dl_control")), ("AR/VR motion to photon", num("dl_arvr")),
            ("Camera frame", num("dl_camera")), ("Voice response", num("dl_voice"))]
    lo, hi = num("rtt_lo"), num("rtt_hi")
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.axvspan(lo, hi, color=C["network_l"], alpha=0.9, zorder=0)
    ax.text((lo * hi) ** 0.5, 3.75, f"network round trip\n{val('rtt_lo')} to {val('rtt_hi')} ms", ha="center", va="center",
            fontsize=13.2, color=C["network"], fontweight="bold")
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        col = C["danger"] if v < lo else C["data"]
        ax.barh(y, v, color=col, height=0.55, zorder=2)
        ax.text(v * 1.12, y, f"{v:.0f} ms", va="center", fontsize=13.2, color=col, fontweight="bold")
        ax.text(5.6, y, lab, ha="right", va="center", fontsize=13.2, fontweight="bold", color=TXT)
    ax.set_xscale("log")
    ax.set_xlim(6, 1500)
    ax.set_ylim(-0.6, 4.4)
    ax.set_yticks([])
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1,000"])
    ax.set_xlabel("Foreground deadline (ms, log scale)", fontsize=13.2)
    out(fig, "deadlines")


# --------------------------------------------------------------------------
# Training working set against the background allowance (EI figure 3)
# --------------------------------------------------------------------------
def amplifier():
    segs = [("Weights", num("stack_weights_mb"), "#9CA3AF"), ("Gradients", num("stack_grads_mb"), "#93C5FD"),
            ("Adam moments", num("stack_adam_mb"), "#60A5FA"),
            ("Saved activations", num("stack_acts_mb"), V["c1"])]
    a_lo, a_hi, w = num("allow_lo_mb"), num("allow_hi_mb"), num("stack_weights_mb")
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.axhspan(a_lo, a_hi, color=C["danger_l"], zorder=0)
        ax.text(-0.12, (a_lo + a_hi) / 2, f"background\nallowance\n{val('allow_lo_mb')} to\n{val('allow_hi_mb')} MB",
                ha="center", va="center", fontsize=13, color=C["danger"], fontweight="bold")
        ax.bar(0, w, width=0.6, color="#9CA3AF", zorder=2)
        ax.text(0, w + 8, f"{val('stack_weights_mb')} MB", ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=TXT)
        upto = {1: 1, 2: 3, 3: 4}[step]
        bottom = 0
        for k, (lab, v, col) in enumerate(segs):
            shown = k < upto
            ax.bar(0.9, v if shown else 0, bottom=bottom, width=0.6, color=col, zorder=2,
                   edgecolor="white", lw=1)
            if shown and v >= w:
                ax.text(1.25, bottom + v / 2, f"{lab} {v:.0f}", va="center", ha="left",
                        fontsize=15, color=TXT)
            if shown:
                bottom += v
        total = sum(s[1] for s in segs[:upto])
        lab = f"{total:.0f} MB" + (f"  ({val('stack_amp')}×)" if step == 3 else "")
        ax.text(0.9, total + 8, lab, ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=C["danger"] if total > a_hi else TXT)
        ax.set_xticks([0, 0.9])
        ax.set_xticklabels(["Inference\nweights", "Full\nfine-tune"], fontsize=15)
        ax.set_xlim(-0.62, 2.75)
        ax.set_ylim(0, 410)
        ax.set_ylabel("MB (FP32)", fontsize=15)
        ax.tick_params(axis="y", labelsize=14)
        out(fig, f"amp_step{step}")


# --------------------------------------------------------------------------
# Device memory ladder (EI Sec. 11.3.4 figure)
# --------------------------------------------------------------------------
def memladder():
    sp_ = K.SPECTRUM
    rows = [("Phone", K.m(sp_.FlagshipSmartphoneRamHigh, "byte"), f"{val('phone_gb')} GB"),
            ("IoT device", K.m(sp_.IotDeviceRamHigh, "byte"), f"up to {val('iot_gb')} GB"),
            ("Microcontroller", K.m(sp_.MicrocontrollerSram, "byte"), f"{val('mcu_kb')} KB")]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (lab, v, s) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=C["memory"], height=0.55)
        ax.text(v * 1.6, y, s, va="center", fontsize=13, fontweight="bold", color=C["memory"])
        ax.text(5e4, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=TXT)
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e12)
    ax.set_yticks([])
    ax.set_xticks([1e6, 1e9])
    ax.set_xticklabels(["1 MB", "1 GB"], fontsize=12)
    ax.set_xlabel("Memory (log scale)", fontsize=12)
    out(fig, "memladder")


# --------------------------------------------------------------------------
# Apple Neural Engine ratings (MLSysIM series; EI figure 5)
# --------------------------------------------------------------------------
def npu_tops():
    from mlsysim import Literature
    rows = [r for r in load_series("vol2/11_edge_intelligence/data/apple_neural_engine.csv").entries
            if r["basis"] in ("stated", "derived")]
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    xs = [r["year"] for r in rows]
    ys = [r["tops"] for r in rows]
    ax.plot(xs, ys, color=C["compute"], lw=2.4, zorder=1)
    for r in rows:
        hollow = r["basis"] == "derived"
        ax.scatter(r["year"], r["tops"], s=70, zorder=2,
                   facecolor="white" if hollow else C["compute"], edgecolor=C["compute"], lw=2)
    ax.plot([2022.5, 2022.5], [0, 29], color=C["neutral"], ls=":", lw=1.5)
    ax.text(2022.25, 24, "rating precision\nnot stated", ha="right", va="center", fontsize=13,
            color=C["neutral"])
    ax.text(2016.9, 9.0, f"A11: {val('ane_a11')}", ha="left", va="bottom", fontsize=14, color=TXT, fontweight="bold")
    ax.text(2023.3, 37.0, f"A17 Pro: {val('ane_a17')}", ha="right", va="bottom", fontsize=14, color=TXT,
            fontweight="bold")
    ax.set_xticks([2017, 2019, 2021, 2023])
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 44)
    ax.set_xlim(2016.6, 2023.5)
    ax.set_ylabel("Peak rating (TOPS)", fontsize=14)
    out(fig, "npu_tops")


# --------------------------------------------------------------------------
# Equation 1 as a bar: which term each method shrinks (schematic)
# --------------------------------------------------------------------------
def terms():
    W, H = 6.4, 2.6
    segs = [("Resident\nweights", 1.55, "#D1D5DB", V["fr"]),
            ("Trainable\nstate", 1.4, "#BFDBFE", V["c1"]),
            ("Peak activations\n+ workspace", 1.9, V["c1"], V["c1"]),
            ("Reserved for\nOS and apps", 1.25, "white", V["mute"])]
    methods = {
        1: [],
        2: [(1, "bias-only, LoRA,\nadapters", V["c1"])],
        3: [(0, "quantize, prune,\ndistill", V["mute"]), (1, "bias-only, LoRA,\nadapters", V["c1"]),
            (2, "TinyTL, head-only,\ncheckpointing", V["c1"])],
    }
    for step in (1, 2, 3):
        fig, ax = canvas(W, H)
        x = 0.1
        y0, h = 1.25, 0.75
        xs = []
        for k, (lab, w, f, s) in enumerate(segs):
            ax.add_patch(Rectangle((x, y0), w, h, fc=f, ec=s, lw=1.4,
                                   ls="--" if k == 3 else "-"))
            ax.text(x + w / 2, y0 + h / 2, lab, ha="center", va="center", fontsize=12.9,
                    color="white" if k == 2 else TXT, fontweight="bold", linespacing=1.1)
            xs.append((x, w))
            x += w
        ax.plot([x, x], [y0 - 0.1, y0 + h + 0.12], color=V["bad"], lw=2.2)
        ax.text(x - 0.04, y0 + h + 0.16, "device memory", ha="right", va="bottom", fontsize=12.3,
                color=V["bad_t"], fontweight="bold")
        # bracket for M_peak
        xa, xb = xs[0][0], xs[2][0] + xs[2][1]
        ax.plot([xa, xa, xb, xb], [y0 + h + 0.08, y0 + h + 0.16, y0 + h + 0.16, y0 + h + 0.08],
                color=TXT, lw=1.2)
        ax.text((xa + xb) / 2, y0 + h + 0.2, "update working set (equation 1)", ha="center",
                va="bottom", fontsize=13.4, color=TXT)
        for (k, lab, col) in methods[step]:
            sx, sw = xs[k]
            arrow(ax, sx + sw / 2, 0.68, sx + sw / 2, y0 - 0.04, col, lw=2)
            ax.text(sx + sw / 2, 0.42, lab, ha="center", va="center", fontsize=12.3, color=col,
                    fontweight="bold", linespacing=1.1)
        ax.text(W - 0.05, 0.05, "schematic, not to scale", ha="right", va="bottom", fontsize=11.2,
                color=V["mute"])
        out(fig, f"terms_step{step}", tight=False)


# --------------------------------------------------------------------------
# Personalization storage (EI Napkin Math 11.2)
# --------------------------------------------------------------------------
def personalize():
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    rows = [(f"{val('wall_contexts')} full copies", num("wall_copies_mb"), C["neutral"], f"{val('wall_copies_mb')} MB"),
            (f"Base + {val('wall_contexts')} adapters", num("wall_shared_mb"), C["memory"],
             f"{val('wall_shared_mb')} MB")]
    for i, (lab, v, col, t) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.55)
        ax.text(v + 8, y, t, va="center", fontsize=12, color=col, fontweight="bold")
        ax.text(-8, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=TXT)
    ax.set_xlim(0, 520)
    ax.set_yticks([])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Storage on the phone (MB)", fontsize=12)
    out(fig, "personalize")


# --------------------------------------------------------------------------
# Battery share (EI Napkin Math 11.1)
# --------------------------------------------------------------------------
def battery():
    fig, ax = plt.subplots(figsize=(3.7, 1.7))
    b, e = num("battery_wh"), num("job_wh")
    ax.barh(0, b, color=C["neutral_l"], height=0.55, edgecolor=C["neutral"])
    ax.barh(0, e, color=C["danger"], height=0.55)
    ax.text(e + 0.15, 0.5, f"one update: {val('job_wh')} Wh", ha="left", va="bottom", fontsize=16,
            color=C["danger"], fontweight="bold")
    ax.text(b - 0.2, 0, f"{val('battery_wh')} Wh battery", ha="right", va="center", fontsize=16, color=TXT)
    ax.set_xlim(0, 15.3)
    ax.set_ylim(-0.5, 1.1)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15])
    ax.tick_params(labelsize=15)
    ax.set_xlabel("Energy (Wh)", fontsize=16)
    out(fig, "battery")


# --------------------------------------------------------------------------
# Edge inference energy against a training second (MLSysIM; EI figure 6)
# --------------------------------------------------------------------------
def tiny_energy():
    from mlsysim import Literature
    keep = {
        "Syntiant Core 2\n(KWS, low-energy)": "Syntiant Core 2, keyword spotting",
        "STM32N6 NPU\n(KWS, low-power)": "STM32N6 NPU, keyword spotting",
        "STM32N6 NPU\n(IC, high-perf)": "STM32N6 NPU, image classification",
        "Google Coral\nEdge TPU": "Google Coral Edge TPU",
        "Jetson AGX Orin\n(ResNet-50)": "Jetson AGX Orin, ResNet-50",
    }
    rows = [(keep[d["name"]], d["energy_uj"]) for d in Literature.EdgeInferenceBenchmarks.list
            if d["name"] in keep]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    t_lo, t_hi = (K.m(q, "microjoule") for q in (K.TRAIN_SEC.training_energy_low, K.TRAIN_SEC.training_energy_high))
    ax.axvspan(t_lo, t_hi, color=C["danger_l"], zorder=0)
    ax.text(t_lo * 1.2, 2.0, f"one second\nof training\nat {val('power_lo')} to {val('power_sus_hi')} W",
            ha="center", va="center",
            fontsize=13.2, color=C["danger"], fontweight="bold")
    for i, (lab, e) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(e, y, s=70, color=C["compute"], zorder=3)
        txt = f"{e:,.0f} µJ" if e < 1000 else f"{e / 1000:,.1f} mJ"
        ax.text(e * 1.5, y, txt, va="center", fontsize=13.2, color=C["compute"], fontweight="bold")
        ax.text(12, y, lab, ha="right", va="center", fontsize=13.2, color=TXT)
    ax.set_xscale("log")
    ax.set_xlim(15, 1.2e7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([1e2, 1e4, 1e6])
    ax.set_xticklabels(["100 µJ", "10 mJ", "1 J"], fontsize=13.2)
    ax.set_xlabel("Energy per inference (log scale)", fontsize=13.2)
    out(fig, "tiny_energy")


# --------------------------------------------------------------------------
# One federated round on a time axis (EI Sec. 11.1, figure 10; schematic)
# --------------------------------------------------------------------------
def round_timeline():
    W, H = 6.15, 2.85
    clients = [  # (download, train, upload)
        (0.25, 1.4, 0.45), (0.25, 1.0, 0.6), (0.3, 1.7, 0.45), (0.25, 1.2, 0.5), (0.35, 3.1, 0.4)]
    x0, deadline = 1.15, 4.45
    for step in (1, 2, 3):
        fig, ax = canvas(W, H)
        # coordinator lane
        ax.text(0.02, 2.51, "Server", ha="left", va="center", fontsize=15.5, fontweight="bold", color=V["c3_t"])
        box(ax, 0.85, 2.3, 1.2, 0.42, V["c3_l"], V["c3"], f"select {val('overselect')}K", fs=15.5)
        if step == 3:
            box(ax, 5.0, 2.3, 1.1, 0.42, V["c3_l"], V["c3"], "average K", fs=15.5)
        for i, (d, t, u) in enumerate(clients):
            y = 1.95 - i * 0.33
            ax.text(0.05, y, f"phone {i + 1}", ha="left", va="center", fontsize=15.5, color=TXT)
            ax.add_patch(Rectangle((x0, y - 0.1), d, 0.2, fc="#E5E7EB", ec=V["fr"]))
            if step >= 2:
                late = x0 + d + t > deadline
                tw = (deadline - x0 - d) if late else t
                ax.add_patch(Rectangle((x0 + d, y - 0.1), tw, 0.2, fc=V["c1_l"], ec=V["c1"]))
                if late:
                    ax.add_patch(Rectangle((deadline, y - 0.1), x0 + d + t - deadline, 0.2,
                                           fc=V["bad_l"], ec=V["bad"], hatch="///"))
                    if step == 3:
                        ax.text(x0 + d + t + 0.08, y, "dropped", va="center", fontsize=15.5,
                                color=V["bad_t"], fontweight="bold")
                elif step == 3:
                    ax.add_patch(Rectangle((x0 + d + t, y - 0.1), u, 0.2, fc=V["c2_l"], ec=V["c2"]))
        if step >= 2:
            ax.text(x0 + 2.0, 2.17, "local epochs E", ha="center", va="center", fontsize=13, color=V["c1"])
        if step == 3:
            ax.plot([deadline, deadline], [0.55, 2.2], ls="--", color=V["bad"], lw=1.8)
            ax.text(deadline, 2.22, "deadline", ha="center", va="bottom", fontsize=15.5, color=V["bad_t"],
                    fontweight="bold")
            # term brackets
            def br(xa, xb, lab, col):
                ax.plot([xa, xa, xb, xb], [0.42, 0.34, 0.34, 0.42], color=col, lw=1.5)
                ax.text((xa + xb) / 2, 0.26, lab, ha="center", va="top", fontsize=14, color=col,
                        fontweight="bold")
            br(x0 + 0.3, x0 + 1.98, "compute", V["c1"])
            br(x0 + 2.02, x0 + 2.45, "comm", V["c2_t"])
            br(x0 + 2.49, deadline, "sync", V["c3_t"])

        out(fig, f"round_step{step}", tight=False)


# --------------------------------------------------------------------------
# FedAvg rounds to target (McMahan et al. 2017, Table 2, MNIST CNN, 99%)
# --------------------------------------------------------------------------
def fedavg():
    # (E, B, u, IID rounds, non-IID rounds); FedSGD first (MLSysIM series)
    from mlsysim import Literature
    rows = [(r["local_epochs"], r["batch_size"], r["updates_per_round"], r["iid_rounds"], r["non_iid_rounds"])
            for r in load_series("slides/vol2/11_edge_intelligence/images/data/fedavg_mnist_cnn_rounds.csv").entries]
    sgd_iid, sgd_non = num("fedsgd_iid"), num("fedsgd_noniid")
    best_u, best_iid, best_non = num("fedavg_best_u"), num("fedavg_best_iid"), num("fedavg_best_noniid")
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.55))
        u = [r[2] for r in rows]
        ax.scatter(u[1:], [r[3] for r in rows[1:]], s=55, color=C["memory"], zorder=3)
        ax.axhline(sgd_iid, color=C["memory"], ls="--", lw=1.4)
        ax.text(2600, sgd_iid * 1.08, f"one step per round: {val('fedsgd_iid')}", ha="right", va="bottom", fontsize=12.1,
                color=C["memory"])
        ax.text(best_u, best_iid * 0.75, f"IID: {val('fedavg_best_iid')}", ha="center", va="top", fontsize=13.2,
                color=C["memory"], fontweight="bold")
        if step == 2:
            ax.scatter(u[1:], [r[4] for r in rows[1:]], s=55, color=C["danger"], zorder=3, marker="s")
            ax.axhline(sgd_non, color=C["danger"], ls="--", lw=1.4)
            ax.text(2600, sgd_non * 0.92, val("fedsgd_noniid"), ha="right", va="top", fontsize=12.1, color=C["danger"])
            ax.text(6.5, 1100, f"non-IID, E = {val('fedavg_e5_epochs')}: {val('fedavg_e5_noniid')}", ha="left", va="bottom",
                    fontsize=13.2, color=C["danger"], fontweight="bold")
            ax.text(best_u, best_non * 0.72, f"non-IID: {val('fedavg_best_noniid')}", ha="center", va="top", fontsize=13.2,
                    color=C["danger"], fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 3000)
        ax.set_ylim(7, 2600)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.set_yticks([10, 100, 1000])
        ax.set_yticklabels(["10", "100", "1,000"])
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Local updates per round, u (log)", fontsize=13.2)
        ax.set_ylabel("Rounds to 99%", fontsize=13.2)
        out(fig, f"fedavg_step{step}")


# --------------------------------------------------------------------------
# Cumulative uploads against raw data (EI Napkin Math 11.3)
# --------------------------------------------------------------------------
def uplink():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    r = list(range(0, 181))
    upd, raw = num("upd_mb"), num("raw_mb")
    be, d_r, d_mb, h_r, h_mb = (num(k) for k in ("break_even", "daily_rounds", "daily_mb", "hourly_rounds", "hourly_mb"))
    ax.plot(r, [upd * k for k in r], color=C["network"], lw=2.6)
    ax.axhline(raw, color=C["data"], lw=2.2, ls="--")
    ax.text(178, raw - 12, f"raw images:\n{val('raw_mb')} MB a week", ha="right", va="top", fontsize=15, color=C["data"], fontweight="bold")
    ax.scatter([be], [raw], s=70, color=TXT, zorder=3)
    ax.text(be - 14, raw + 22, f"break-even: {val('break_even')}", fontsize=15, ha="right", va="bottom", color=TXT)
    ax.scatter([d_r], [d_mb], s=70, color=C["data"], zorder=3)
    ax.text(d_r + 11, d_mb, f"daily: {val('daily_rounds')} rounds", fontsize=15, va="center", color=C["data"])
    ax.scatter([h_r], [h_mb], s=70, color=C["danger"], zorder=3)
    ax.text(h_r - 8, h_mb + 12, f"hourly: {val('hourly_rounds')}", fontsize=15, ha="right", va="bottom",
            color=C["danger"])
    ax.set_xlim(0, 182)
    ax.set_ylim(0, 470)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Update rounds per week", fontsize=15)
    ax.set_ylabel("Uploaded per phone (MB)", fontsize=15)
    out(fig, "uplink")


# --------------------------------------------------------------------------
# Secure aggregation: masks cancel (EI figure 16, redrawn; values illustrative)
# --------------------------------------------------------------------------
def secagg():
    M = K.MASK
    da, db, s = M.update_a, M.update_b, M.mask
    ya, yb, tot = M.upload_a, M.upload_b, M.total
    W, H = 6.4, 2.75
    cw = 0.42

    def cells(ax, x, y, vals, fill, stroke, label, alpha=1.0, lab_left=True):
        for k, v in enumerate(vals):
            ax.add_patch(Rectangle((x + k * cw, y), cw, 0.32, fc=fill, ec=stroke, lw=1.2, alpha=alpha))
            ax.text(x + k * cw + cw / 2, y + 0.16, f"{v:+.1f}", ha="center", va="center", fontsize=11.8,
                    color=TXT, alpha=alpha)
        ax.text(x - 0.08 if lab_left else x + 4 * cw + 0.08, y + 0.16, label,
                ha="right" if lab_left else "left", va="center", fontsize=13.4, fontweight="bold",
                color=stroke, alpha=alpha)

    for step in (1, 2, 3):
        fig, ax = canvas(W, H)
        yb_ = 1.55
        ax.plot([0.05, W - 0.05], [yb_, yb_], ls="--", color=V["mute"], lw=1.3)
        ax.text(W - 0.05, yb_ + 0.05, "visible to server", ha="right", va="bottom", fontsize=11.8, color=V["mute"])
        ax.text(W - 0.05, yb_ - 0.05, "on the phones", ha="right", va="top", fontsize=11.8, color=V["mute"])
        cells(ax, 0.75, 0.95, da, V["c1_l"], V["c1"], "Δθ_A")
        cells(ax, 0.75, 0.2, db, V["c1_l"], V["c1"], "Δθ_B")
        cells(ax, 3.2, 0.58, s, "#E5E7EB", V["mute"], "shared mask s", lab_left=False)
        a2 = 1.0 if step >= 2 else 0.0
        cells(ax, 0.75, 1.75, ya, V["c2_l"], V["c2"], "A + s", alpha=a2)
        cells(ax, 3.2, 1.75, yb, V["c2_l"], V["c2"], "B − s", alpha=a2, lab_left=False)
        if step >= 2:
            ax.text(1.59, 2.2, "looks random", ha="center", va="bottom", fontsize=12.3, color=V["c2_t"])
            ax.text(4.04, 2.2, "looks random", ha="center", va="bottom", fontsize=12.3, color=V["c2_t"])
        a3 = 1.0 if step == 3 else 0.0
        cells(ax, 1.98, 2.35, tot, V["ok_l"], V["ok"], "sum = Δθ_A + Δθ_B", alpha=a3, lab_left=False)
        if step == 3:
            ax.text(1.9, 2.51, "s − s = 0", ha="right", va="center", fontsize=13.4, color=V["ok_t"],
                    fontweight="bold")
        out(fig, f"secagg_step{step}", tight=False)


def secagg_drop():
    W, H = 6.4, 2.6
    fig, ax = canvas(W, H)
    names = ["A", "B", "C", "D", "E"]
    xs = [0.6 + i * 0.75 for i in range(5)]
    for x, n in zip(xs, names):
        dropped = n == "C"
        ax.add_patch(plt.Circle((x, 0.95), 0.24, fc=V["bad_l"] if dropped else V["c1_l"],
                                ec=V["bad"] if dropped else V["c1"], lw=1.6))
        ax.text(x, 0.95, n, ha="center", va="center", fontsize=14.6, fontweight="bold",
                color=V["bad_t"] if dropped else V["c1"])
        if dropped:
            ax.plot([x - 0.2, x + 0.2], [0.75, 1.15], color=V["bad"], lw=2)
            ax.text(x, 0.6, "drops after\nagreeing masks", ha="center", va="top", fontsize=11.8,
                    color=V["bad_t"])
        else:
            arrow(ax, x, 1.21, x, 1.62, V["c2"], lw=1.8)
    box(ax, 0.3, 1.65, 3.6, 0.55, V["c3_l"], V["c3"],
        "sum still holds C's masks with A, B, D, E", fs=12.9)
    box(ax, 4.15, 1.65, 2.15, 0.55, V["ok_l"], V["ok"], "survivors send shares\nof C's mask seeds", fs=12.3)
    box(ax, 4.15, 0.55, 2.15, 0.7, V["bad_l"], V["bad"],
        "fewer than t survivors:\nround is abandoned", fs=12.3)
    ax.text(5.22, 2.42, "server rebuilds and\nsubtracts C's masks", ha="center", va="center",
            fontsize=12.3, color=V["ok_t"])
    out(fig, "secagg_drop", tight=False)


# --------------------------------------------------------------------------
# Session outcomes (Bonawitz et al. 2019, Appendix A, Table 1)
# --------------------------------------------------------------------------
def sessions():
    rows = [("Completed and accepted", num("sess_ok"), C["data"], val("sess_ok_pct")),
            ("Completed, rejected as late", num("sess_late"), C["compute"], val("sess_late_pct")),
            ("Interrupted (left idle)", num("sess_intr"), C["danger"], val("sess_intr_pct"))]
    tot = sum(r[1] for r in rows)
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    for i, (lab, v, col, pct) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v / tot * 100, color=col, height=0.55)
        ax.text(v / tot * 100 + 1.5, y, f"{pct}%  ({v:,.0f} sessions)", va="center",
                fontsize=13.2, color=col, fontweight="bold")
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=13.2, fontweight="bold", color=TXT)
    ax.set_xlim(0, 125)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Share of on-device training sessions (%)", fontsize=13.2)
    out(fig, "sessions")


# --------------------------------------------------------------------------
# AI Benchmark spread (MLSysIM series; EI figure 17)
# --------------------------------------------------------------------------
def aibench():
    from mlsysim import Literature
    rows = [r for r in load_series("vol2/11_edge_intelligence/data/ai_benchmark_socs.csv").entries if r["ai_score"]]
    fam = [("Snapdragon", C["danger"]), ("Dimensity", C["compute"]), ("Exynos", C["memory"]),
           ("Google", C["data"]), ("Kirin", C["network"])]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.axvspan(2019.5, 2026.5, color=C["neutral_l"], alpha=0.5, zorder=0)
    for r in rows:
        f = r["processor"].split()[0]
        col = dict(fam).get(f, "#BBBBBB")
        ax.scatter(r["year"], r["ai_score"], s=30, color=col, zorder=3)
    lo, hi = K.SPREAD.lowest("fleet"), K.SPREAD.highest("fleet")
    g4 = next(r for r in rows if r["processor"] == "Google Tensor G4")
    ax.annotate("", xy=(2027.0, hi["ai_score"]), xytext=(2027.0, lo["ai_score"]),
                arrowprops=dict(arrowstyle="<->", color=TXT, lw=1.4))
    ax.text(2027.25, (hi["ai_score"] * lo["ai_score"]) ** 0.5,
            f"{val('ai_spread')}×", va="center", fontsize=14.3, fontweight="bold")
    ax.text(lo["year"] + 0.2, lo["ai_score"] * 0.75, lo["processor"], fontsize=12.1, va="top", color=TXT)
    ax.text(hi["year"] - 0.2, hi["ai_score"] * 1.6, hi["processor"], fontsize=12.1, ha="right",
            va="bottom", color=TXT)
    ax.text(g4["year"], g4["ai_score"] * 0.3, "Tensor G4", fontsize=12.1, ha="center", va="top",
            color=C["data"], fontweight="bold")
    ax.text(2020.0, 0.6, f"phones a fleet may still hold ({val('ai_first_year')} on)", fontsize=12.1, color=C["neutral"])
    ax.set_yscale("log")
    ax.set_ylim(0.3, 2e5)
    ax.set_xlim(2009.5, 2028.6)
    ax.set_yticks([1, 100, 10000])
    ax.set_yticklabels(["1", "100", "10,000"])
    ax.set_xticks([2010, 2014, 2018, 2022, 2026])
    ax.tick_params(labelsize=11)
    ax.set_ylabel("AI Benchmark score", fontsize=13.2)
    for k, (f, col) in enumerate(fam):
        ax.text(2010, 6e4 / (3.4 ** k), {"Dimensity": "Dimensity", "Google": "Tensor"}.get(f, f),
                fontsize=12.1, color=col, fontweight="bold", va="center")
    out(fig, "aibench")


# --------------------------------------------------------------------------
# Coverage funnel and coverage ratio (illustrative; after EI figure 13)
# --------------------------------------------------------------------------
def coverage():
    stages = ["Deployed", "Admitted", "Selected", "Completed"]
    # counts per 100 deployed phones: flagship, mid-tier, budget (illustrative, MLSysIM scenario)
    cov = K.COVER
    data = [tuple(cov.counts[c][i] for c in ("flagship", "mid-tier", "budget")) for i in range(len(cov.stages))]
    cols = [C["memory"], "#AAB2BD", C["danger"]]
    names = ["flagship", "mid-tier", "budget"]
    for step in (1, 2):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
        for i, d in enumerate(data):
            b = 0
            for k, v in enumerate(reversed(d)):
                kk = 2 - k
                a1.bar(i, v, bottom=b, color=cols[kk], width=0.62, edgecolor="white")
                b += v
            a1.text(i, b + 2, f"{sum(d)}", ha="center", va="bottom", fontsize=13.4, fontweight="bold")
        a1.text(3.35, 1, f"budget\n{val('cov_budget_done_pct')}%", fontsize=12.3, color=C["danger"], va="bottom", fontweight="bold")
        a1.text(-0.3, 4, f"{val('cov_budget_fleet_pct')}%", fontsize=12.3, color="white", fontweight="bold")
        a1.set_xticks(range(4))
        a1.set_xticklabels(["Deployed", "\nAdmitted", "Selected", "\nCompleted"], fontsize=11.8)
        a1.set_xlim(-0.5, 4.1)
        a1.set_ylim(0, 112)
        a1.set_yticks([])
        a1.spines["left"].set_visible(False)
        a1.set_title("Phones per 100 deployed", fontsize=13.4)
        ratios = [cov.coverage_ratio(c) for c in ("flagship", "mid-tier", "budget")]
        for k, (n, r) in enumerate(zip(names, ratios)):
            y = 2 - k
            show = step == 2
            a2.barh(y, r if show else 0, color=cols[k], height=0.55)
            a2.text(max(r, 1) + 0.06, y, f"{r:.2f}" if show else "", va="center", fontsize=13.4,
                    fontweight="bold", color=cols[k] if k != 1 else TXT)
            a2.text(-0.06, y, n, ha="right", va="center", fontsize=13.4, fontweight="bold")
        a2.axvline(1, color=C["data"], ls="--", lw=1.6)
        a2.set_xlim(0, 2.6)
        a2.set_ylim(-0.6, 2.6)
        a2.set_yticks([])
        a2.set_xticks([0, 1, 2])
        a2.tick_params(labelsize=11)
        a2.set_title("Coverage ratio", fontsize=13.4)
        fig.tight_layout(w_pad=2.5)
        out(fig, f"coverage_step{step}")


# --------------------------------------------------------------------------
# Forgetting (illustrative; after EI Sec. 11.5.2 figure)
# --------------------------------------------------------------------------
def forget():
    import numpy as np
    t = np.linspace(0, 10, 50)
    new = 55 + 35 * (1 - np.exp(-t / 2.5))
    old = 90 - 35 * (1 - np.exp(-t / 3.0))
    rep = 90 - 7 * (1 - np.exp(-t / 3.0))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(t, new, color=C["data"], lw=2.8)
    ax.plot(t, old, color=C["danger"], lw=2.8)
    ax.plot(t, rep, color=C["danger"], lw=2.2, ls="--")
    ax.text(10.2, new[-1] + 1.5, "new task", va="bottom", fontsize=14, color=C["data"], fontweight="bold")
    ax.text(10.2, old[-1], "old task", va="center", fontsize=14, color=C["danger"], fontweight="bold")
    ax.text(10.2, rep[-1] - 1.0, "old task,\nwith replay", va="top", fontsize=13, color=C["danger"])
    ax.set_xlim(0, 13.8)
    ax.set_ylim(40, 100)
    ax.set_xticks([])
    ax.set_yticks([50, 70, 90])
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Local updates over months", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    out(fig, "forget")


# --------------------------------------------------------------------------
# Shadow gate (EI figure 20, redrawn; letters illustrative)
# --------------------------------------------------------------------------
def gate():
    base, cand = list(K.GATE.frozen), list(K.GATE.adapted)
    W, H = 6.4, 2.7
    for step in (1, 2):
        fig, ax = canvas(W, H)
        x0, cw = 1.55, 0.38
        ax.text(x0 - 0.1, 2.35, "input", ha="right", va="center", fontsize=12.3, color=V["mute"])
        ax.text(x0 - 0.1, 1.95, "frozen v1", ha="right", va="center", fontsize=12.3, color=V["mute"],
                fontweight="bold")
        ax.text(x0 - 0.1, 1.55, "adapted v1.1", ha="right", va="center", fontsize=12.3, color=V["c1"],
                fontweight="bold")
        ax.text(x0 - 0.1, 1.15, "agree?", ha="right", va="center", fontsize=12.3, color=V["c3_t"],
                fontweight="bold")
        for k in range(8):
            x = x0 + k * cw
            ax.text(x + cw / 2, 2.35, f"x{k + 1}", ha="center", va="center", fontsize=11.8, color=V["mute"])
            box(ax, x + 0.03, 1.8, cw - 0.06, 0.3, "#F3F4F6", V["fr"], base[k], fs=13.4, r=0.03)
            box(ax, x + 0.03, 1.4, cw - 0.06, 0.3, V["c1_l"], V["c1"], cand[k], fs=13.4, r=0.03)
            same = base[k] == cand[k]
            box(ax, x + 0.03, 1.0, cw - 0.06, 0.3, "white" if same else V["c3_l"],
                V["fr"] if same else V["c3"], "=" if same else "≠", fs=13.4, r=0.03,
                color=TXT if same else V["c3_t"])
        ax.text(x0, 0.62, f"{val('gate_dis')} of {val('gate_n')} disagree: {val('gate_pct')}% > τ = {val('gate_tau_pct')}%", ha="left", va="center",
                fontsize=13.4, color=V["bad_t"], fontweight="bold")
        box(ax, 4.75, 1.45, 1.6, 0.6, V["bad_l"], V["bad"], "hold: candidate\nout of service", fs=12.3)
        arrow(ax, x0 + 8 * cw + 0.05, 1.15, 4.75, 1.6, V["bad"])
        if step == 2:
            box(ax, 4.75, 0.3, 1.6, 0.9, V["ok_l"], V["ok"],
                "validated outcome\nclears it or\nconfirms rollback", fs=11.8)
            arrow(ax, 5.55, 1.2, 5.55, 1.44, V["ok"])
        ax.text(x0, 0.22, "label-free: flags change, not which is right", ha="left",
                va="center", fontsize=12.3, color=V["mute"])
        out(fig, f"gate_step{step}", tight=False)


if __name__ == "__main__":
    hook()
    fleetlaw_edge()
    boundary()
    deadlines()
    amplifier()
    memladder()
    npu_tops()
    terms()
    personalize()
    battery()
    tiny_energy()
    round_timeline()
    fedavg()
    uplink()
    secagg()
    secagg_drop()
    sessions()
    aibench()
    coverage()
    forget()
    gate()
    print("figures written")
