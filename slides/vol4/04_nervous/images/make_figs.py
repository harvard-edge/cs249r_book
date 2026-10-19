#!/usr/bin/env python3
"""Figures for Lecture 4 (Volume IV, The Nervous System).

Every figure is drawn at the size of its slot on the slide (centimeters on
the 16 x 9 cm page) and saved at exactly that size, so the slide places it at
scale 1 and every label keeps the point size written here. Derived values come
from calcs.py (imported as K); inputs carry their source there.

Colors follow the Volume IV causal-boundary palette (volumes/vol4/palette.md):
navy for cognition, blue (dashed) for unprivileged proposals, green for the
deterministic permission path, red for vetoes, limits, and hazards, slate for
the physical plant, amber for latency.

Sources, by figure (NS: the rendered chapter, The Nervous System):
  rate_ladder   NS 4.1 figure and 4.3.1 (500 ms, 50 ms, 1 ms, 50 us; L/R 2.7 ms).
  phase         NS 4.3.1, equation 1 (20 Hz crossover and 5 degree allowance, illustrative).
  jitter        data/jitter_*.npy, measured by the lecture authors (tick.c, load.c).
  relay         Pilz PNOZ s4 datasheet 1001742-EN-05, Technical details, Times
                (delay-on de-energisation with E-STOP typ. 10 ms, max. 20 ms); jitter data.
  twoproc       NS 4.2 figure 2 and 4.6 figure 7, redrawn.
  queue         NS 4.3.2 (event against state messages), new diagram.
  mailbox_meas  data/mailbox_*.npy, measured by the lecture authors (mailbox.c).
  seqlock       NS 4.3.2 figure 4 panel b, redrawn as three cases.
  lease         NS 4.4 figure 4 panel a, redrawn.
  budget        NM 4.1 (stopping budget terms), derived in calcs.py.
  openvla       Franklin 2024, NVIDIA Jetson AI Lab OpenVLA tutorial, Bridge table:
                Jetson AGX Orin 64GB, FP16 840 ms, FP8 471 ms, INT4 336 ms.
  leaserule     NS 4.9.1, equation 4 and table 1.
  timers        NS 4.4.1 (three supervisory timers, four ending events).
  events        NS table 1.
  rta           NS 4.2 figure 1 and NM 4.2.
  runmax        data/jitter_heavy.npy and jitter_light.npy.
  bus           NM 4.3.

Run from this folder: python3 make_figs.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py

plt = sp.plt
OUT = Path(__file__).parent

# Volume IV causal-boundary palette
NAVY, NAVY_L = "#1F407A", "#E6EDF8"
BLUE, BLUE_L = "#2563EB", "#EFF6FF"
GREEN, GREEN_D, GREEN_L = "#059669", "#047857", "#DCFCE7"
RED, RED_D, RED_L = "#DC2626", "#991B1B", "#FEE2E2"
SLATE, SLATE_L = "#334155", "#F1F5F9"
AMBER, AMBER_L = "#D97706", "#FEF3C7"
GRAY, GRAY_L = "#666666", "#DDDDDD"
TEXT = "#1A1A2E"
FS = 10          # label size in points, at the placed size
FS_S = 9


def cm(w, h):
    return (w / 2.54, h / 2.54)


def save_slot(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white")
    plt.close(fig)


def canvas(w, h):
    """A diagram canvas whose data units are centimeters."""
    fig = plt.figure(figsize=cm(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc="white", ec=SLATE, tc=TEXT, fs=FS, bold=False,
        ls="-", lw=1.2, r=0.08, ha="center", hatch=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, hatch=hatch)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.15
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color=SLATE, ls="-", lw=1.5, ms=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def plot_axes(w, h, left, bottom, right=0.2, top=0.15):
    """A plot whose axes sit at fixed margins (cm) inside a w x h cm figure."""
    fig = plt.figure(figsize=cm(w, h))
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
    ax.tick_params(labelsize=FS_S)
    return fig, ax


# ---------------------------------------------------------------- rate ladder
def rate_ladder(step):
    w, h = 9.6, 5.9
    fig, ax = plot_axes(w, h, left=0.45, bottom=1.0, right=0.15, top=0.1)
    rows = [("Intent model", 500.0, NAVY, "500 ms  (2 Hz)",
             "set by prefill and weight sweeps"),
            ("Chunk policy", 50.0, BLUE, "50 ms  (20 Hz)",
             "set by one weight sweep per 16-setpoint chunk"),
            ("Permission tick", 1.0, GREEN, "1 ms  (1 kHz)",
             f"set by travel: {K.V * K.T_TICK:.1f} mm per tick at 1.5 m/s"),
            ("Current loop", 0.05, SLATE, "50 \u00b5s  (20 kHz)",
             f"set by the winding: L/R = 2.7 ms, pole near {1 / (2 * np.pi * K.TAU_E):.0f} Hz")]
    lo = 0.01
    for i, (lab, t, c, val, why) in enumerate(rows):
        y = 3 - i
        ax.barh(y, t - lo, left=lo, height=0.28, color=c)
        ax.text(lo * 1.15, y + 0.2, lab, ha="left", va="bottom", fontsize=FS,
                fontweight="bold", color=c)
        ax.text(t * 1.25, y, val, ha="left", va="center", fontsize=FS_S, color=c,
                fontweight="bold")
        ax.text(lo * 1.15, y - 0.2, why, ha="left", va="top", fontsize=FS_S,
                color=GRAY, alpha=1.0 if step >= 2 else 0.0)
    ax.set_xscale("log")
    ax.set_xlim(lo, 30000)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.set_xlabel("Period (ms, log scale)", fontsize=FS_S)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.minorticks_off()
    save_slot(fig, f"rate_ladder_step{step}")


# ---------------------------------------------------------------- phase lag
def phase(step):
    w, h = 14.6, 4.3
    fig, ax = plot_axes(w, h, left=3.3, bottom=1.0, right=0.3, top=0.15)
    hold, comp, bus = (K.lag_deg(K.HOLD), K.lag_deg(K.COMPUTE), K.lag_deg(K.BUS))
    y1, y2 = 1.0, 0.0
    ax.barh(y1, hold, color=AMBER, height=0.5)
    ax.barh(y1, comp, left=hold, color=GREEN, height=0.5)
    ax.text(hold / 2, y1, f"hold\n{hold:.1f}°", ha="center", va="center", fontsize=FS_S,
            color="white", fontweight="bold")
    ax.text(hold + comp / 2, y1 - 0.42, f"compute {comp:.1f}°", ha="center", va="top",
            fontsize=FS_S, color=GREEN_D)
    x0 = hold + comp
    if step == 1:
        ax.add_patch(Rectangle((x0, y1 - 0.25), bus, 0.5, fc="white", ec=SLATE, ls="--", lw=1.2))
        ax.text(x0 + bus / 2, y1, "bus cycle 1 ms:  ???", ha="center", va="center",
                fontsize=FS, color=SLATE, fontweight="bold")
    else:
        ax.barh(y1, bus, left=x0, color=SLATE, height=0.5)
        ax.text(x0 + bus / 2, y1, f"bus {bus:.1f}°", ha="center", va="center",
                fontsize=FS, color="white", fontweight="bold")
        tot = x0 + bus
        ax.text(tot + 0.15, y1, f"{tot:.1f}° = {tot / K.ALLOW:.1f}× the allowance",
                ha="left", va="center", fontsize=FS, color=RED_D, fontweight="bold")
    if step >= 3:
        ax.barh(y2, bus, color=SLATE, height=0.5)
        ax.text(bus / 2, y2, f"bus alone {bus:.1f}°", ha="center", va="center",
                fontsize=FS, color="white", fontweight="bold")
        ax.text(bus + 0.15, y2, "still over: no tick rate fixes it", ha="left", va="center",
                fontsize=FS, color=RED_D, fontweight="bold")
    ax.axvline(K.ALLOW, color=RED, ls="--", lw=1.5)
    ax.text(K.ALLOW, 1.62, f"allowed {K.ALLOW:.0f}°", ha="center", va="bottom",
            fontsize=FS_S, color=RED, fontweight="bold")
    ax.set_yticks([y1, y2])
    ax.set_yticklabels(["1 kHz tick", "Infinitely fast tick"], fontsize=FS, fontweight="bold")
    ax.set_ylim(-0.45, 1.95)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Phase lag at a 20 Hz crossover (degrees)", fontsize=FS_S)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save_slot(fig, f"phase_step{step}")


# ---------------------------------------------------------------- measured OS jitter
def jitter():
    w, h = 9.4, 4.6
    fig, ax = plot_axes(w, h, left=1.6, bottom=1.0, right=0.2, top=0.55)
    rows = [("p50", "p50"), ("p99", "p99"), ("p99.9", "p999"), ("max", "mx")]
    for i, (lab, key) in enumerate(rows):
        y = 3 - i
        lv, hv = K.JSTAT["light"][key], K.JSTAT["heavy"][key]
        ax.barh(y + 0.18, lv, color=GRAY, height=0.32)
        ax.barh(y - 0.18, hv, color=RED, height=0.32)
        ax.text(lv * 1.12, y + 0.18, f"{lv:,.0f}", va="center", fontsize=FS_S, color=GRAY)
        ax.text(hv * 1.12, y - 0.18, f"{hv:,.0f}", va="center", fontsize=FS_S, color=RED,
                fontweight="bold")
    ax.axvline(1000, color=SLATE, ls="--", lw=1.2)
    ax.text(1000, 3.62, "one period, 1 ms", ha="center", va="bottom", fontsize=FS_S,
            color=SLATE)
    ax.text(380000, 3.2, "ordinary load", color=GRAY, fontsize=FS_S, fontweight="bold",
            va="center", ha="right")
    ax.text(380000, 2.75, "all 18 cores loaded", color=RED, fontsize=FS_S,
            fontweight="bold", va="center", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(100, 400000)
    ax.set_ylim(-0.5, 3.6)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("How late the release woke (µs, log scale)", fontsize=FS_S)
    ax.minorticks_off()
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
    save_slot(fig, "jitter")


# ---------------------------------------------------------------- relay vs laptop
def relay():
    w, h = 8.6, 4.3
    fig, ax = plot_axes(w, h, left=0.2, bottom=1.0, right=0.2, top=0.1)
    rows = [("Safety relay, E-STOP, typical", K.PNOZ["estop_typ"], GREEN, "datasheet"),
            ("Safety relay, E-STOP, maximum", K.PNOZ["estop_max"], GREEN_D, "datasheet"),
            ("Laptop 1 ms task, worst seen", K.JSTAT["heavy"]["mx"] / 1000, RED, "measured")]
    for i, (lab, v, c, kind) in enumerate(rows):
        y = 2 - i
        ax.barh(y - 0.12, v, color=c, height=0.36)
        ax.text(0.3, y + 0.25, lab, fontsize=FS_S, color=c, fontweight="bold", va="bottom")
        ax.text(v + 0.6, y - 0.12, f"{v:.1f} ms = {K.V * v:.1f} mm" if i == 2 else
                f"{v:.0f} ms = {K.V * v:.0f} mm", fontsize=FS_S, va="center", color=c)
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.5, 2.65)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Delay before the drives lose torque (ms)", fontsize=FS_S)
    save_slot(fig, "relay")


# ---------------------------------------------------------------- two processors
def twoproc(step):
    w, h = 14.6, 5.0
    fig, ax = canvas(w, h)
    # application processor: data flows right to left, so the proposal drops at the left
    box(ax, 0.1, 3.4, 9.6, 1.55, fc=NAVY_L, ec=NAVY, lw=1.4)
    ax.text(0.25, 4.72, "Application processor (may only propose)", fontsize=FS, color=NAVY,
            fontweight="bold", va="center")
    box(ax, 0.3, 3.5, 2.7, 0.85, "Chunk policy\n20 Hz", fc="white", ec=BLUE, ls="--")
    box(ax, 3.5, 3.5, 2.7, 0.85, "Intent model\n2 Hz", fc="white", ec=NAVY)
    box(ax, 6.7, 3.5, 2.8, 0.85, "Perception", fc="white", ec=NAVY)
    arrow(ax, 6.7, 3.92, 6.2, 3.92, NAVY)
    arrow(ax, 3.5, 3.92, 3.0, 3.92, NAVY)
    box(ax, 0.1, 2.72, 9.6, 0.5, fc="white", ec=GRAY, ls="--")
    ax.text(2.1, 2.97, "Untrusted board link: CRC and sequence number",
            fontsize=FS_S, color=GRAY, va="center")
    if step >= 2:
        box(ax, 0.1, 0.1, 9.6, 2.45, fc=GREEN_L, ec=GREEN, lw=1.4)
        ax.text(0.25, 2.33, "Safety microcontroller (may command)", fontsize=FS,
                color=GREEN_D, fontweight="bold", va="center")
        arrow(ax, 1.65, 3.5, 1.65, 2.05, BLUE, ls="--")
        box(ax, 0.3, 0.95, 2.7, 1.1, "Mailbox\n(own SRAM)", fc="white", ec=GREEN)
        box(ax, 3.5, 0.8, 2.9, 1.3, "Permission tick\n1 kHz: freshness,\nlease, limits",
            fc="white", ec=GREEN_D, lw=1.6, fs=FS_S)
        arrow(ax, 3.0, 1.5, 3.5, 1.5, GREEN_D)
        box(ax, 6.9, 0.95, 2.6, 1.1, "Drive\nregisters", fc=SLATE_L, ec=SLATE, r=0.0)
        arrow(ax, 6.4, 1.5, 6.9, 1.5, GREEN, lw=2.2)
        ax.text(8.2, 2.33, "sole writer", fontsize=FS_S, color=GREEN_D, fontweight="bold",
                ha="center", va="center")
        box(ax, 10.3, 0.1, 4.2, 2.0, "Drives and motors\n(current loops)", fc="white", ec=SLATE,
            r=0.0)
        arrow(ax, 9.5, 1.5, 10.3, 1.5, SLATE, lw=2)
        ax.plot([10.3, 4.95], [0.38, 0.38], color=SLATE, lw=1.3)
        arrow(ax, 4.95, 0.38, 4.95, 0.8, SLATE, lw=1.3)
        ax.text(7.7, 0.45, "encoders: measured state", fontsize=FS_S, color=SLATE,
                ha="center", va="bottom")
    if step >= 3:
        ax.text(10.0, 4.6, "No wire from the host\nto any drive register", fontsize=FS,
                color=RED_D, fontweight="bold", va="center")
        arrow(ax, 9.75, 4.0, 11.2, 2.15, RED, ls="--", lw=1.6)
        ax.plot([10.15, 10.65], [2.75, 3.25], color=RED, lw=3)
        ax.plot([10.15, 10.65], [3.25, 2.75], color=RED, lw=3)
        ax.text(14.5, 3.6, "Two chips are a hypothesis\nloaded tests must confirm", fontsize=FS_S,
                color=TEXT, ha="right", va="top")
    save_slot(fig, f"twoproc_step{step}")


# ---------------------------------------------------------------- queue vs mailbox
def queue(step):
    w, h = 14.6, 4.4
    fig, ax = canvas(w, h)

    def row(y, title, color):
        ax.text(0.1, y + 1.45, title, fontsize=FS, fontweight="bold", color=color, va="center")
        box(ax, 0.1, y, 2.3, 1.0, "Chunk policy\nwrites at 20 Hz", fc=BLUE_L, ec=BLUE, ls="--",
            fs=FS_S)
        box(ax, 11.9, y, 2.6, 1.0, "Permission tick\nreads at 1 kHz", fc=GREEN_L, ec=GREEN_D,
            fs=FS_S)

    row(2.55, "FIFO queue (event messages): every proposal is kept and read in order", RED_D)
    arrow(ax, 2.4, 3.05, 3.0, 3.05, BLUE)
    for i, lab in enumerate(["#9", "#8", "#7", "#6", "#5"]):
        box(ax, 3.1 + i * 1.45, 2.65, 1.25, 0.8, lab, fc="white", ec=BLUE, fs=FS)
    ax.text(3.1, 2.42, "newest", ha="left", va="top", fontsize=FS_S, color=GRAY)
    ax.text(3.1 + 4 * 1.45 + 1.25, 2.42, "oldest", ha="right", va="top", fontsize=FS_S, color=GRAY)
    arrow(ax, 10.35, 3.05, 11.9, 3.05, RED)
    ax.text(11.1, 3.32, "#5", fontsize=FS, color=RED_D, fontweight="bold", ha="center")
    ax.text(14.5, 2.4, "acts on the oldest plan", fontsize=FS_S, color=RED_D,
            fontweight="bold", ha="right", va="top")
    if step >= 2:
        row(0.1, "Mailbox (state message): one slot, overwritten in place", GREEN_D)
        arrow(ax, 2.4, 0.6, 5.3, 0.6, BLUE)
        box(ax, 5.4, 0.2, 2.4, 0.8, "#9", fc="white", ec=GREEN_D, fs=FS, lw=1.6)
        arrow(ax, 7.8, 0.6, 11.9, 0.6, GREEN_D)
        ax.text(9.85, 0.83, "#9, the freshest", fontsize=FS, color=GREEN_D,
                fontweight="bold", ha="center")
    save_slot(fig, f"queue_step{step}")


# ---------------------------------------------------------------- measured mailbox
def mailbox_meas(step):
    w, h = 11.0, 3.9
    fig, ax = plot_axes(w, h, left=0.2, bottom=1.0, right=0.2, top=0.1)
    rows = [("Blocking mutex", K.MBX["mutex"], RED,
             f"{{v:,.0f}} µs; {K.MBX['mutex']['over1ms']} reads over 1 ms"),
            ("One try-lock (as ros2_control)", K.MBX["trylock"], GREEN, "{v:.1f} µs"),
            ("One-attempt seqlock", K.MBX["seqlock"], GREEN_D,
             f"{{v:.1f}} µs; {K.MBX['seqlock']['torn']} torn reads rejected")]
    for i, (lab, s, c, fmt) in enumerate(rows):
        y = 2 - i
        ax.text(0.12, y + 0.26, lab, fontsize=FS_S, fontweight="bold", color=c, va="bottom")
        if step >= 2:
            ax.barh(y - 0.1, s["mx"], left=0.1, color=c, height=0.36)
            ax.text(s["mx"] * 1.25, y - 0.1, fmt.format(v=s["mx"]), fontsize=FS_S, color=c,
                    va="center", fontweight="bold", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.5))
        else:
            ax.text(0.15, y - 0.1, "?", fontsize=FS, color=c, va="center", fontweight="bold")
    ax.axvline(1000, color=SLATE, ls="--", lw=1.2)
    ax.text(1000 * 1.15, 2.45, "one tick", fontsize=FS_S, color=SLATE, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3e7)
    ax.set_ylim(-0.55, 2.7)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Longest time the reader spent getting a proposal (µs, log scale)",
                  fontsize=FS_S)
    save_slot(fig, f"mailbox_meas_step{step}")


# ---------------------------------------------------------------- seqlock cases
def seqlock(step):
    w, h = 14.6, 4.9
    fig, ax = canvas(w, h)
    cw = 4.75
    cases = [("1  Clean exchange", ["7", "8"], "reads 8, then 8:\naccept the chunk", GREEN_D),
             ("2  Tick preempts a write", ["9", "10"], "reads 9 (odd): reject,\nkeep the active chunk",
              AMBER),
             ("3  Host stalls", ["10", "10"], "reads 10 again: nothing new,\nlease keeps counting",
              SLATE)]
    for k, (title, ctr, out, c) in enumerate(cases):
        x0 = 0.1 + k * (cw + 0.1)
        if step < k + 1:
            continue
        box(ax, x0, 0.1, cw, 4.7, fc="white", ec=GRAY_L, lw=1.0)
        ax.text(x0 + 0.15, 4.5, title, fontsize=FS, fontweight="bold", color=c, va="center")
        ax.text(x0 + 0.15, 3.85, "Writer", fontsize=FS_S, color=BLUE, va="center")
        ax.text(x0 + 0.15, 2.85, "Counter", fontsize=FS_S, color=SLATE, va="center")
        ax.text(x0 + 0.15, 1.85, "Tick", fontsize=FS_S, color=GREEN_D, va="center")
        if k == 0:
            box(ax, x0 + 1.5, 3.6, 1.9, 0.5, "copy frame", fc=BLUE_L, ec=BLUE, ls="--", fs=FS_S)
            box(ax, x0 + 1.5, 2.6, 0.9, 0.5, "7 odd", fc=AMBER_L, ec=AMBER, fs=FS_S)
            box(ax, x0 + 3.45, 2.6, 0.9, 0.5, "8", fc=GREEN_L, ec=GREEN, fs=FS_S)
            for xx in (3.65, 4.15):
                arrow(ax, x0 + xx, 2.6, x0 + xx, 2.1, GREEN_D)
        elif k == 1:
            box(ax, x0 + 1.5, 3.6, 0.7, 0.5, "", fc=BLUE_L, ec=BLUE, ls="--")
            ax.text(x0 + 2.3, 3.85, "paused", fontsize=FS_S, color=GRAY, va="center")
            box(ax, x0 + 3.55, 3.6, 0.6, 0.5, "", fc=BLUE_L, ec=BLUE, ls="--")
            box(ax, x0 + 1.5, 2.6, 2.4, 0.5, "9 odd", fc=AMBER_L, ec=AMBER, fs=FS_S)
            box(ax, x0 + 3.95, 2.6, 0.7, 0.5, "10", fc=GREEN_L, ec=GREEN, fs=FS_S)
            arrow(ax, x0 + 2.6, 2.6, x0 + 2.6, 2.1, AMBER)
            arrow(ax, x0 + 4.3, 2.6, x0 + 4.3, 2.1, GREEN_D)
        else:
            ax.text(x0 + 2.8, 3.85, "no frame arrives", fontsize=FS_S, color=GRAY,
                    va="center", ha="center")
            box(ax, x0 + 1.5, 2.6, 3.1, 0.5, "10", fc=GREEN_L, ec=GREEN, fs=FS_S)
            for xx in (2.4, 3.8):
                arrow(ax, x0 + xx, 2.6, x0 + xx, 2.1, SLATE)
        ax.plot([x0 + 1.3, x0 + cw - 0.1], [1.85, 1.85], color=GREEN_D, lw=1.2)
        ax.text(x0 + cw / 2, 1.0, out, fontsize=FS_S, ha="center", va="center", color=c,
                fontweight="bold")
    save_slot(fig, f"seqlock_step{step}")


# ---------------------------------------------------------------- lease timeline
def lease(step):
    w, h = 14.6, 4.9
    fig, ax = plot_axes(w, h, left=2.4, bottom=1.0, right=0.2, top=0.15)
    yP, yA, yL, yD = 3.0, 2.0, 1.0, 0.0
    ax.barh(yP, 40, left=0, color=BLUE, height=0.45)
    ax.barh(yP, 40, left=50, color=BLUE, height=0.45)
    for x in (40, 90):
        ax.plot([x], [yA], marker="v", ms=9, color=GREEN_D)
    ax.text(90, yA + 0.32, "last admission", fontsize=FS_S, color=GREEN_D, ha="center")
    ax.add_patch(Rectangle((40, yL - 0.22), 60, 0.44, fc="none", ec=GREEN, ls="--", lw=1.2))
    if step == 1:
        ax.barh(yD, 100, left=0, color=SLATE, height=0.45)
        ax.text(50, yD, "tracking admitted setpoints", color="white", fontsize=FS_S,
                ha="center", va="center")
    if step >= 2:
        ax.add_patch(Rectangle((100, yP - 0.22), 110, 0.44, fc=RED_L, ec=RED, hatch="///", lw=1))
        ax.text(125, yP + 0.33, "stalled", fontsize=FS_S, color=RED_D,
                ha="center", fontweight="bold")
        ax.barh(yL, 60, left=90, color=GREEN, height=0.44)
        ax.text(120, yL, f"lease {K.T_LEASE:.0f} ms", color="white", fontsize=FS,
                fontweight="bold", ha="center", va="center")
        ax.barh(yD, 150, left=0, color=SLATE, height=0.45)
        ax.text(75, yD, "tracking the last admitted chunk", color="white", fontsize=FS_S,
                ha="center", va="center")
    if step >= 3:
        ax.barh(yD, 22, left=150, color=AMBER, height=0.45)
        ax.barh(yD, 38, left=172, color=RED, height=0.45)
        ax.text(191, yD, "stop", color="white", fontsize=FS_S, ha="center", va="center",
                fontweight="bold")
        ax.axvline(150, color=GREEN_D, ls="--", lw=1)
        ax.axvline(172, color=RED, ls="--", lw=1.2)
        ax.text(191, yA, "tick + bus\n+ onset 22 ms", fontsize=FS_S - 1, color=AMBER,
                ha="center", va="center", fontweight="bold")
        ax.annotate("", xy=(172, -0.72), xytext=(90, -0.72),
                    arrowprops=dict(arrowstyle="<->", color=RED_D, lw=1.2))
        ax.text(131, -0.64, f"{K.LEASE_PATH_MS + K.T_ONSET:.0f} ms = "
                f"{K.V * (K.LEASE_PATH_MS + K.T_ONSET):.0f} mm at 1.5 m/s", fontsize=FS_S,
                color=RED_D, ha="center", va="bottom", fontweight="bold", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_yticks([yP, yA, yL, yD])
    ax.set_yticklabels(["Chunk policy", "Admission", "Lease", "Drives"], fontsize=FS,
                       fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 210)
    ax.set_ylim(-0.95, 3.55)
    ax.set_xlabel("Time (ms)", fontsize=FS_S)
    ax.spines["left"].set_visible(False)
    save_slot(fig, f"lease_step{step}")


# ---------------------------------------------------------------- stopping budget bar
def budget(step):
    w, h = 9.0, 4.0
    fig, ax = plot_axes(w, h, left=0.2, bottom=1.0, right=0.35, top=0.1)
    y = 1.0
    x = 0
    ax.barh(y, K.BRAKE, left=0, color=SLATE, height=0.6, ec="white")
    ax.text(K.BRAKE / 2, y, f"braking\n{K.BRAKE:.1f} mm", color="white", fontsize=FS_S,
            ha="center", va="center", fontweight="bold")
    x = K.BRAKE
    ov = K.LOC + K.PROT
    ax.barh(y, ov, left=x, color=GRAY, height=0.6, ec="white")
    ax.text(x + ov / 2, y, f"{ov:.0f}", color="white", fontsize=FS_S, ha="center", va="center",
            fontweight="bold")
    ax.text(x + ov - 10, y - 0.4, "localization + clearance", color=GRAY, fontsize=FS_S,
            ha="right", va="top")
    x += ov
    ax.barh(y, K.ONSET_MM, left=x, color=AMBER, height=0.6, ec="white")
    ax.text(x + K.ONSET_MM - 8, y + 0.4, f"onset {K.ONSET_MM:.0f}", color=AMBER, fontsize=FS_S,
            ha="right", va="bottom", fontweight="bold")
    x += K.ONSET_MM
    if step >= 2:
        ax.barh(y, K.LEASE_PATH, left=x, color=GREEN, height=0.6, ec="white")
        ax.text(x + K.LEASE_PATH / 2, y, f"{K.LEASE_PATH:.0f}", color="white", fontsize=FS_S,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 8, y + 0.4, "lease path", color=GREEN_D, fontsize=FS_S, ha="left",
                va="bottom", fontweight="bold")
        x2 = x + K.LEASE_PATH
        ax.text(x2 + (K.D_CLEAR - x2) / 2, y, f"left\n{K.D_CLEAR - x2:.1f}", color=GRAY,
                fontsize=FS_S, ha="center", va="center")
        ax.text(x2 + 8, y - 0.4, f"{x2:.1f} mm", color=GREEN_D, fontsize=FS_S, ha="left",
                va="top", fontweight="bold")
    else:
        ax.text(x + 8, y - 0.4, f"{x:.1f} mm", color=SLATE, fontsize=FS_S, ha="left", va="top",
                fontweight="bold")
    ax.axvline(K.D_CLEAR, color=RED, ls="--", lw=1.5)
    ax.text(K.D_CLEAR - 10, 1.95, "clear distance 1,100 mm", color=RED, fontsize=FS_S,
            ha="right", va="bottom", fontweight="bold")
    ax.set_xlim(0, 1150)
    ax.set_ylim(0.15, 2.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Distance from detection (mm)", fontsize=FS_S)
    save_slot(fig, f"budget_step{step}")


# ---------------------------------------------------------------- measured OpenVLA
def openvla():
    w, h = 9.2, 3.8
    fig, ax = plot_axes(w, h, left=0.2, bottom=1.0, right=0.2, top=0.45)
    for i, (lab, t) in enumerate(K.OPENVLA):
        y = 2 - i
        ax.barh(y, t, color=NAVY, height=0.42)
        ax.text(t + 12, y, f"{lab}  {t:.0f} ms", fontsize=FS_S, va="center", color=NAVY,
                fontweight="bold")
    ax.axvline(K.T_LEASE, color=GREEN_D, ls="-", lw=2)
    ax.text(K.T_LEASE, 2.62, "lease 60", color=GREEN_D, fontsize=FS_S, ha="center",
            va="bottom", fontweight="bold")
    ax.axvline(K.CEIL, color=RED, ls="--", lw=1.6)
    ax.text(K.CEIL, 2.62, f"pre-brake ceiling {K.CEIL:.1f}", color=RED, fontsize=FS_S,
            ha="left", va="bottom", fontweight="bold")
    ax.set_xlim(0, 1060)
    ax.set_ylim(-0.5, 2.55)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("OpenVLA-7B, measured time per frame (ms)", fontsize=FS_S)
    save_slot(fig, "openvla")


# ---------------------------------------------------------------- lease rule waterfall
def leaserule(step):
    w, h = 14.6, 4.7
    fig, ax = plot_axes(w, h, left=1.2, bottom=1.25, right=0.2, top=0.2)
    cols = [("Pre-brake\nceiling", 0, K.CEIL, RED),
            ("tick, bus,\nonset", K.UP0, K.CEIL, AMBER),
            ("observation\nage\n(Lecture 8)", K.UP0 - K.OBS_AGE, K.UP0, AMBER),
            ("dispatch to\nadmission\n(Lecture 13)", K.UPPER, K.UP0 - K.OBS_AGE, AMBER),
            ("Upper limit\non the lease", 0, K.UPPER, GREEN_D)]
    labels = []
    for i, (lab, lo, hi, c) in enumerate(cols):
        hatch = None
        ax.bar(i, hi - lo, bottom=lo, color=c, width=0.6, hatch=hatch)
        if i in (0, 4):
            ax.text(i, hi + 6, f"{hi:.1f}", ha="center", va="bottom", fontsize=FS, color=c,
                    fontweight="bold")
        else:
            ax.text(i, hi + 6, f"−{hi - lo:.1f}", ha="center", va="bottom", fontsize=FS,
                    color=c, fontweight="bold")
        labels.append(lab)
    if step >= 2:
        ax.bar(5.2, K.T_RENEW + K.T_JIT, color=BLUE, width=0.45)
        ax.text(5.2, K.T_RENEW + K.T_JIT + 6, f"{K.T_RENEW + K.T_JIT:.0f}", ha="center",
                va="bottom", fontsize=FS, color=BLUE, fontweight="bold")
        ax.bar(5.8, K.T_LEASE, color=GREEN, width=0.45)
        ax.text(5.8, K.T_LEASE + 6, f"{K.T_LEASE:.0f}", ha="center", va="bottom", fontsize=FS,
                color=GREEN_D, fontweight="bold")
        labels_extra = True
    else:
        labels_extra = False
    ax.set_xticks(list(range(5)) + ([5.2, 5.8] if labels_extra else []))
    ax.set_xticklabels(labels + (["floor", "lease"] if labels_extra else []), fontsize=FS_S)
    ax.set_xlim(-0.5, 6.2)
    ax.set_ylim(0, 300)
    ax.set_ylabel("ms", fontsize=FS_S)
    ax.tick_params(axis="x", length=0)
    save_slot(fig, f"leaserule_step{step}")


# ---------------------------------------------------------------- authority timers
def timers():
    w, h = 8.2, 5.0
    fig, ax = canvas(w, h)
    items = [("Evidence deadline passes", NAVY),
             (f"Chunk lease ({K.T_LEASE:.0f} ms) expires", GREEN_D),
             ("Host heartbeat silent 30 ms", SLATE),
             ("Protective scanner trips", RED)]
    for i, (lab, c) in enumerate(items):
        y = 4.0 - i * 0.95
        box(ax, 0.1, y, 4.3, 0.72, lab, fc="white", ec=c, tc=c, fs=FS_S, bold=False)
        arrow(ax, 4.4, y + 0.36, 5.0, 2.92, c, lw=1.2)
    box(ax, 5.0, 2.35, 3.1, 1.15, "Earliest one\nends authority", fc=RED_L, ec=RED, tc=RED_D,
        bold=True, fs=FS_S)
    arrow(ax, 6.55, 2.35, 6.55, 1.72, RED)
    box(ax, 5.0, 1.05, 3.1, 0.65, "State-matched stop", fc="white", ec=RED, tc=RED_D, fs=FS_S,
        bold=True)
    box(ax, 0.1, 0.1, 8.0, 0.8, "Only an admitted chunk renews the lease.\nA heartbeat never does.",
        fc=GREEN_L, ec=GREEN, tc=GREEN_D, fs=FS_S, bold=True)
    save_slot(fig, "timers")


# ---------------------------------------------------------------- event timing
def events():
    w, h = 14.6, 2.9
    fig, ax = canvas(w, h)
    x0, scale = 0.9, 12.6 / 200.0
    t = [0, K.OBS_AGE, K.OBS_AGE + K.DISP_ADM, K.OBS_AGE + K.DISP_ADM + K.STALL]
    names = ["capture", "dispatch", "admission", "brake onset"]
    ax.plot([x0, x0 + t[-1] * scale], [1.6, 1.6], color=SLATE, lw=2)
    for ti, n in zip(t, names):
        x = x0 + ti * scale
        ax.plot([x, x], [1.4, 1.8], color=SLATE, lw=2)
        ax.text(x, 1.95, n, ha="center", va="bottom", fontsize=FS, fontweight="bold",
                color=RED_D if n == "brake onset" else TEXT)
    spans = [("observation age\n51.6 ms (Lecture 8)", AMBER),
             ("dispatch to admission\n63.3 ms (Lecture 13)", AMBER),
             (f"stalled proposer {K.STALL:.0f} ms\n(lease + tick + bus + onset)", GREEN_D)]
    for k, (lab, c) in enumerate(spans):
        a, b = x0 + t[k] * scale, x0 + t[k + 1] * scale
        ax.annotate("", xy=(b, 1.15), xytext=(a, 1.15),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
        ax.text((a + b) / 2, 1.0, lab, ha="center", va="top", fontsize=FS_S, color=c)
    save_slot(fig, "events")


# ---------------------------------------------------------------- response time
def rta(step):
    w, h = 14.6, 4.3
    fig, ax = plot_axes(w, h, left=2.6, bottom=1.0, right=0.3, top=0.35)
    y1, y2 = 1.0, 0.0
    ax.barh(y1, K.C_UNLOADED, color=GRAY, height=0.45)
    ax.text(K.C_UNLOADED + 4, y1, f"{K.C_UNLOADED:.0f} µs, one unloaded run", va="center",
            fontsize=FS_S, color=GRAY)
    ax.barh(y2, K.C_UNLOADED, color=GREEN, height=0.45)
    ax.add_patch(Rectangle((K.C_UNLOADED, y2 - 0.225), K.C_DECL - K.C_UNLOADED, 0.45,
                           fc=GREEN_L, ec=GREEN, hatch="///"))
    ax.text(K.C_DECL / 2, y2 - 0.38, f"C = {K.C_DECL:.0f} (declared WCET)", ha="center",
            va="top", fontsize=FS_S, color=GREEN_D, fontweight="bold")
    if step >= 2:
        ax.barh(y2, K.B, left=K.C_DECL, color=AMBER, height=0.45)
        ax.barh(y2, K.I, left=K.C_DECL + K.B, color="#B45309", height=0.45)
        ax.text(K.C_DECL + K.B / 2, y2 + 0.33, f"B {K.B:.0f}", ha="center", va="bottom",
                fontsize=FS_S, color=AMBER, fontweight="bold")
        ax.text(K.C_DECL + K.B + K.I / 2, y2 + 0.33, f"I {K.I:.0f}", ha="center", va="bottom",
                fontsize=FS_S, color="#B45309", fontweight="bold")
        ax.text(K.R / 2 + 60, y2 - 0.38, "", fontsize=FS_S)
        ax.text(K.R + 4, y2 - 0.38, f"R = {K.R:.0f}", ha="left", va="top", fontsize=FS,
                color=TEXT, fontweight="bold")
    if step >= 3:
        ax.axvline(K.DEADLINE, color=RED, ls="--", lw=1.6)
        ax.text(K.DEADLINE - 4, 1.5, f"deadline D = {K.DEADLINE:.0f} µs", color=RED,
                fontsize=FS_S, fontweight="bold", va="center", ha="right")
        ax.annotate("", xy=(K.DEADLINE, y2), xytext=(K.R, y2),
                    arrowprops=dict(arrowstyle="<->", color=RED_D, lw=1.2))
        ax.text(K.DEADLINE + 4, y2, f"margin {K.DEADLINE - K.R:.0f}", va="center",
                fontsize=FS_S, color=RED_D, fontweight="bold")
    ax.set_yticks([y1, y2])
    ax.set_yticklabels(["Observed", "Bound"], fontsize=FS, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 480)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("Time after the tick's release (µs)", fontsize=FS_S)
    save_slot(fig, f"rta_step{step}")


# ---------------------------------------------------------------- running maximum
def runmax():
    w, h = 9.6, 4.7
    fig, ax = plot_axes(w, h, left=1.9, bottom=1.0, right=0.25, top=0.15)
    ns = np.unique(np.logspace(1, np.log10(60000), 60).astype(int))
    for key, c, lab in (("heavy", RED, "all cores loaded"), ("light", GRAY, "ordinary load")):
        a = K.JIT[key]
        mx = np.maximum.accumulate(a)
        ax.plot(ns, mx[ns - 1], color=c, lw=2)
        p99 = [np.percentile(a[:n], 99) for n in ns]
        ax.plot(ns, p99, color=c, lw=1.5, ls="--")
    ax.text(60000, K.JSTAT["heavy"]["mx"] * 1.15, f"max {K.JSTAT['heavy']['mx'] / 1000:.1f} ms",
            ha="right", va="bottom", fontsize=FS_S, color=RED, fontweight="bold")
    ax.text(60000, K.JSTAT["heavy"]["p99"] * 1.2, f"p99 {K.JSTAT['heavy']['p99']:.0f} µs",
            ha="right", va="bottom", fontsize=FS_S, color=RED)
    ax.text(12, 25000, "solid: largest so far\ndashed: p99 so far", fontsize=FS_S, color=TEXT,
            va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 70000)
    ax.set_ylim(150, 150000)
    ax.minorticks_off()
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([1000, 10000, 100000])
    ax.set_yticklabels(["1,000", "10,000", "100,000"])
    ax.set_xlabel("Releases tested", fontsize=FS_S)
    ax.set_ylabel("Lateness (µs)", fontsize=FS_S)
    save_slot(fig, "runmax")


# ---------------------------------------------------------------- bus occupancy
def bus():
    w, h = 9.0, 4.4
    fig, ax = plot_axes(w, h, left=0.2, bottom=1.0, right=0.2, top=0.45)
    rows = [(K.BUSES[0][0], K.BUSES[0][1], RED), (K.BUSES[1][0], K.BUSES[1][1], AMBER),
            (K.BUSES[2][0], K.BUSES[2][1], AMBER), ("EtherCAT, one frame, 9 axes", K.ECAT, GREEN)]
    for i, (lab, v, c) in enumerate(rows):
        y = 3 - i
        ax.barh(y - 0.12, v, color=c, height=0.38)
        ax.text(5, y + 0.22, lab, fontsize=FS_S, color=c, fontweight="bold", va="bottom")
        txt = f"{v:,.1f} µs ({100 * v / K.CYCLE:.1f}%)"
        if v > 600:
            ax.text(v - 15, y - 0.12, txt, fontsize=FS_S, color="white", va="center", ha="right",
                    fontweight="bold")
        else:
            ax.text(v + 15, y - 0.12, txt, fontsize=FS_S, color=c, va="center")
    ax.axvline(K.CYCLE, color=SLATE, ls="--", lw=1.4)
    ax.text(K.CYCLE, 3.62, "1 ms bus cycle", fontsize=FS_S, color=SLATE, ha="center",
            va="bottom", fontweight="bold")
    ax.set_xlim(0, 1500)
    ax.set_ylim(-0.5, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Serialization time per cycle (µs)", fontsize=FS_S)
    save_slot(fig, "bus")


if __name__ == "__main__":
    for s in (1, 2):
        rate_ladder(s)
        queue(s)
        mailbox_meas(s)
        budget(s)
        leaserule(s)
    for s in (1, 2, 3):
        phase(s)
        twoproc(s)
        seqlock(s)
        lease(s)
        rta(s)
    jitter()
    relay()
    openvla()
    timers()
    events()
    runmax()
    bus()
    print("figures written")
