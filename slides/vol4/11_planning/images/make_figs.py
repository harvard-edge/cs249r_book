#!/usr/bin/env python3
"""Figures for Lecture 11 (Trajectory Planning), Volume IV.

Every value is sourced. "TP" is the rendered chapter, Trajectory Planning.
Every derived value comes from calcs.py (imported as K); values labeled
"illustrative" on a figure are the chapter's illustrative inputs.

  - hook: TP Napkin Math 11.1 (3,000 handoffs, p99, mission target 1e-3).
  - obligations: the arm stall of TP Napkin Math 11.4 (1 m/s, lease 60 ms,
    onset at 82 ms, at rest 357 ms); the seam mismatch is schematic.
  - jac_ellipse, dls: TP Sec. 11.2.1 (Panda planar links 0.316 and 0.384 m,
    damping 50 mm illustrative); computed in calcs.py.
  - record: TP Sec. 11.2.2 and 11.8 (fields, deadlines of the alternate
    schedule, stop of NM 11.4).
  - repl: TP equation 3 (51.6 + 63.3 = 114.9 ms; 40 ms inference, 18 ms
    validation); 50 ms renewal and 60 ms lease (TP Sec. 11.2.2).
  - chunks: published chunk configurations: Diffusion Policy Push-T 16/6
    (Chi et al. 2025, via TP Sec. 11.3.2); pi0 50/25 at 50 Hz (Black et al.
    2024, via TP table 4); the running machine 16 setpoints, renewed every
    50 ms (2 setpoints executed).
  - stale: TP Sec. 11.3.3 (0.50 rad/s, 80 ms, illustrative).
  - overshoot: TP equation 9 and 10, schematic.
  - seam: TP Napkin Math 11.2 (reflected inertia from the MLSysIM registry).
  - phase: TP Napkin Math 11.3.
  - ruckig_arm, base: Ruckig 0.19.4 output (ruckig_stops.csv, written by
    ruckig_run.py) with the chapter's bounds; constant-deceleration and
    polynomial curves from TP equations 16 and 17.
  - runway: TP Napkin Math 11.4 and Sec. 11.5.2 (open mug).
  - timeline: TP figure 2 values (both schedules).
  - gen_timing: Ruckig 19.8 us mean, 123 us worst (Berscheid and Kroger
    2021, via TP table 4); 1 ms tick, 18 ms validation, 40 ms chunk pass
    (illustrative), 114.9 ms replacement (TP Sec. 11.1, 11.3.1); pi0 0.5 s
    between inferences (TP table 4).
  - base_view: TP figure 6 (differential drive), schematic.

Run from this folder: python3 make_figs.py
"""

import contextlib
import csv
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Volume IV causal-boundary roles (volumes/vol4/palette.md)
V4 = {
    "proposal": "#2563EB", "proposal_l": "#EFF6FF",
    "nervous": "#059669", "nervous_l": "#DCFCE7",
    "veto": "#DC2626", "veto_l": "#FEE2E2",
    "lag": "#D97706", "lag_l": "#FEF3C7",
    "plant": "#334155", "plant_l": "#F8FAFC",
    "navy": "#1F407A", "gray": "#666666", "gray_l": "#DDDDDD",
    "text": "#1A1A2E",
}
FS = 11      # label size at slot scale
FS_S = 10    # small labels


def cm(w, h):
    return (w / 2.54, h / 2.54)


def save_slot(fig, name):
    fig.savefig(out / name, format="svg", facecolor="white")
    plt.close(fig)


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def read_ruckig():
    data = {}
    with (out / "ruckig_stops.csv").open() as f:
        for r in csv.DictReader(f):
            d = data.setdefault(r["case"], {"t": [], "p": [], "v": [], "a": []})
            d["t"].append(float(r["t_s"]))
            d["p"].append(float(r["position_m"]))
            d["v"].append(float(r["velocity_mps"]))
            d["a"].append(float(r["acceleration_mps2"]))
    return {k: {kk: np.array(vv) for kk, vv in d.items()} for k, d in data.items()}


RK = read_ruckig()


# ============================================================== hook
def hook(step):
    rows = [
        ("p99 budget,\nper handoff", K.P_LATE, V4["gray"], "1 in 100"),
        ("Mission target,\nwhole run", K.MISSION_BUDGET, V4["lag"], "1 in 1,000"),
        ("Design target,\nper handoff", K.P_TARGET, V4["veto"],
         f"1 in {1 / K.P_TARGET:,.0f}"),
    ]
    fig, ax = plt.subplots(figsize=cm(14.2, 4.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.27)
    for i, (lab, p, col, txt) in enumerate(rows):
        y = 2 - i
        shown = i < step
        n = -math.log10(p)
        ax.barh(y, n, color=col, height=0.6, alpha=1 if shown else 0)
        ax.text(-0.15, y, lab, ha="right", va="center", fontsize=FS,
                fontweight="bold", color=V4["text"])
        ax.text(n + 0.12, y, txt, va="center", fontsize=FS, fontweight="bold",
                color=col, alpha=1 if shown else 0)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.65, 2.45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("How rare a late handoff must be (nines: $-\\log_{10}$ of the probability)",
                  fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    if step >= 3:
        ax.annotate("", xy=(-math.log10(K.P_TARGET), -0.42), xytext=(2, -0.42),
                    arrowprops=dict(arrowstyle="<->", color=V4["veto"], lw=1.4))
        ax.text(-math.log10(K.P_TARGET) + 0.15, -0.42,
                f"{math.log10(K.P_LATE / K.P_TARGET):.1f} orders deeper than p99",
                fontsize=FS_S, color=V4["veto"], ha="left", va="center")
    save_slot(fig, f"hook_step{step}.svg")


for s in (1, 2, 3):
    hook(s)


# ============================================================== obligations
def obligations(step):
    """Tool speed against time: the three things a plan carries."""
    fig, ax = plt.subplots(figsize=cm(9.4, 5.6))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.2)
    t_pred = np.linspace(0, 400, 200)
    # proposed continuation, dashed (not authorized)
    ax.plot(t_pred, np.ones_like(t_pred), color=V4["proposal"], lw=1.6, ls="--")
    ax.text(395, 0.95, "predicted to 400 ms", color=V4["proposal"],
            fontsize=FS_S, ha="right", va="top")
    # admitted prefix with setpoints
    ax.plot([0, 50], [1, 1], color=V4["nervous"], lw=3.5)
    ax.plot([0, 25, 50], [1, 1, 1], "o", color=V4["nervous"], ms=6)
    ax.text(5, 0.86, "1. timed prefix", color=V4["nervous"], fontsize=FS,
            fontweight="bold", va="top")
    if step >= 2:
        # incoming chunk B 0.2 faster, joined by a quintic bridge (schematic)
        tb = np.linspace(0, 1, 60)
        vb = 1 + 0.2 * (10 * tb ** 3 - 15 * tb ** 4 + 6 * tb ** 5)
        ab = 0.3 if step >= 3 else 1.0
        ax.plot(50 + 50 * tb, vb, color=V4["nervous"], lw=3.5, alpha=ab)
        ax.plot([100, 160], [1.2, 1.2], color=V4["proposal"], lw=1.6, ls="--", alpha=ab)
        ax.plot([50, 50], [1.0, 1.2], color=V4["veto"], lw=1.4, ls=":", alpha=ab)
        ax.text(105, 1.27, "2. smooth join", color=V4["nervous"],
                fontsize=FS, fontweight="bold", va="bottom", alpha=ab)
    if step >= 3:
        d = RK["arm"]
        t0 = K.T_PATH
        ax.plot(t0 + 1000 * d["t"], d["v"], color=V4["veto"], lw=3)
        ax.plot([K.T_LEASE, t0], [1, 1], color=V4["lag"], lw=3)
        ax.text(88, 0.22, "3. stop without\nthe proposer",
                color=V4["veto"], fontsize=FS, fontweight="bold", ha="left", va="center")
    ax.set_xlim(0, 410)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("Time after admission (ms)", fontsize=FS)
    ax.set_ylabel("Tool speed (m/s)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f"obligations_step{step}.svg")


for s in (1, 2, 3):
    obligations(s)


# ============================================================== Jacobian ellipse
def jac_ellipse():
    fig, axes = plt.subplots(1, 2, figsize=cm(9.6, 5.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05)
    for ax, (q2d, q1d) in zip(axes, ((90, 15), (10, 30))):
        q1, q2 = math.radians(q1d), math.radians(q2d)
        p0 = np.array([0.0, 0.0])
        p1 = p0 + K.L1 * np.array([math.cos(q1), math.sin(q1)])
        p2 = p1 + K.L2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
        th = np.linspace(0, math.pi / 2 + 0.6, 60)
        R = K.L1 + K.L2
        ax.plot(R * np.cos(th - 0.3), R * np.sin(th - 0.3), color=V4["gray"], lw=1, ls="--")
        ax.plot(*zip(p0, p1, p2), color=V4["plant"], lw=5, solid_capstyle="round")
        ax.plot(*p0, "s", color=V4["plant"], ms=10)
        ax.plot(*p1, "o", color="white", mec=V4["plant"], mew=2.5, ms=9)
        # ellipse: image of unit joint-rate circle, scaled for display
        J = np.array([[-K.L1 * math.sin(q1) - K.L2 * math.sin(q1 + q2), -K.L2 * math.sin(q1 + q2)],
                      [K.L1 * math.cos(q1) + K.L2 * math.cos(q1 + q2), K.L2 * math.cos(q1 + q2)]])
        U, S, _ = np.linalg.svd(J)
        sc = 0.22
        ang = math.degrees(math.atan2(U[1, 0], U[0, 0]))
        ax.add_patch(Ellipse(p2, 2 * sc * S[0], 2 * sc * S[1] + 0.004, angle=ang,
                             fc=V4["proposal_l"], ec=V4["proposal"], lw=1.6))
        minor = U[:, 1] * 0.13
        ax.annotate("", xy=p2 + minor, xytext=p2 - minor,
                    arrowprops=dict(arrowstyle="<->", color=V4["veto"], lw=1.8))
        ax.plot(*p2, "o", color=V4["proposal"], ms=6)
        key = q2d
        ax.text(0.98, 0.17, f"Elbow {q2d}°", transform=ax.transAxes, fontsize=FS,
                fontweight="bold", va="bottom", ha="right", color=V4["text"],
                bbox=dict(fc="white", ec="none", pad=1))
        ax.text(0.98, 0.03, f"{K.JAC[key]['exact']:.1f} rad/s", transform=ax.transAxes,
                fontsize=FS + 2, fontweight="bold", va="bottom", ha="right", color=V4["veto"],
                bbox=dict(fc="white", ec="none", pad=1))
        ax.set_aspect("equal")
        ax.set_xlim(-0.12, 0.85)
        ax.set_ylim(-0.30, 0.72)
        hide_axis(ax)
    save_slot(fig, "jac_ellipse.svg")


jac_ellipse()


# ============================================================== DLS curve
def dls():
    fig, ax = plt.subplots(figsize=cm(14.2, 5.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.24)
    el = np.linspace(90, 3, 300)
    ex, dm = [], []
    for e in el:
        smin = K.sing_vals(e)[1]
        ex.append(1 / smin)
        dm.append(K.dls_rate(smin))
    ax.plot(el, ex, color=V4["veto"], lw=2.5)
    ax.plot(el, dm, color=V4["nervous"], lw=2.5)
    ax.axhline(1 / (2 * K.LAMBDA), color=V4["nervous"], lw=1.2, ls="--")
    ax.axhline(K.QDOT_MAX, color=V4["plant"], lw=1.2, ls=":")
    ax.text(72, 1 / (2 * K.LAMBDA) * 1.08, "damped ceiling 10 rad/s", color=V4["nervous"],
            fontsize=FS_S, ha="left")
    ax.text(88, K.QDOT_MAX * 0.80, "joint limit 2.175 rad/s", color=V4["plant"],
            fontsize=FS_S, ha="left", va="top")
    ax.text(22, 40, "exact inverse", color=V4["veto"], fontsize=FS, fontweight="bold", ha="right")
    ax.text(62, 3.1, "damped least squares", color=V4["nervous"], fontsize=FS,
            fontweight="bold", ha="left", va="center")
    for e in (90, 10):
        ax.plot([e], [K.JAC[e]["exact"]], "o", color=V4["veto"], ms=7)
        ax.plot([e], [K.JAC[e]["damped"]], "o", color=V4["nervous"], ms=7)
    ax.annotate(f"{K.JAC[10]['exact']:.1f}", (10, K.JAC[10]["exact"]), xytext=(8, -2),
                textcoords="offset points", fontsize=FS, color=V4["veto"], fontweight="bold")
    ax.annotate(f"{K.JAC[10]['damped']:.1f}", (10, K.JAC[10]["damped"]), xytext=(-6, -16), ha="right",
                textcoords="offset points", fontsize=FS, color=V4["nervous"], fontweight="bold")
    ax.annotate(f"{K.JAC[90]['exact']:.1f} / {K.JAC[90]['damped']:.1f}", (90, K.JAC[90]["exact"]),
                xytext=(6, 9), textcoords="offset points", fontsize=FS_S, color=V4["text"],
                ha="left")
    ax.set_yscale("log")
    ax.set_ylim(1.5, 150)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(92, 0)
    ax.set_xlabel("Elbow angle (degrees; 0 is straight)", fontsize=FS)
    ax.set_ylabel("Joint rate (rad/s)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "dls.svg")


dls()


# ============================================================== trajectory record
def box(ax, x, y, w, h, title, lines, ec, fc, alpha=1.0, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                ec=ec, fc=fc, lw=1.8, alpha=alpha,
                                ls="--" if dashed else "-"))
    ax.text(x + 0.015, y + h - 0.03, title, fontsize=FS, fontweight="bold", color=ec,
            va="top", alpha=alpha)
    for i, ln in enumerate(lines):
        ax.text(x + 0.015, y + h - 0.12 - i * 0.082, ln, fontsize=FS_S, color=V4["text"],
                va="top", alpha=alpha)


def record(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 5.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hide_axis(ax)
    box(ax, 0.01, 0.52, 0.30, 0.46, "Header",
        ["frame: robot base", "evidence time", "parent lease digest"],
        V4["navy"], "#EEF2F8")
    box(ax, 0.01, 0.02, 0.30, 0.46, "Motion",
        ["16 setpoints, 25 ms apart", "seam state: position,", "velocity, acceleration"],
        V4["proposal"], V4["proposal_l"], dashed=True)
    a = 1 if step >= 2 else 0
    box(ax, 0.34, 0.52, 0.32, 0.46, "Stop (latched)",
        [f"S-curve: {K.A_ARM:.0f} m/s², {K.J_ARM:.1f} m/s³",
         f"{K.D_STOP_ARM:.1f} mm in {1000 * K.T_STOP_ARM:.0f} ms",
         "runs without the proposer"],
        V4["veto"], V4["veto_l"], alpha=a)
    box(ax, 0.34, 0.02, 0.32, 0.46, "Deadlines (MCU clock)",
        [f"chunk lease: {K.T_LEASE:.0f} ms", f"commitment: {K.T_COMMIT:.0f} ms",
         f"nominal end: {K.N_SETPOINTS * K.T_SETPOINT:.0f} ms", f"at rest: {K.T_END:.0f} ms"],
        V4["lag"], V4["lag_l"], alpha=a)
    b = 1 if step >= 3 else 0
    box(ax, 0.69, 0.52, 0.30, 0.46, "Sidecar",
        ["payload at most 3 kg", "obstacle-map epoch", "inside the ODD"],
        V4["plant"], V4["plant_l"], alpha=b)
    box(ax, 0.69, 0.02, 0.30, 0.46, "Read by",
        ["tracking controller:", "  setpoints, feedforward", "permission path:",
         "  every field, every tick"],
        V4["nervous"], V4["nervous_l"], alpha=b)
    save_slot(fig, f"record_step{step}.svg")


for s in (1, 2, 3):
    record(s)


# ============================================================== replacement latency
def repl(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 5.4))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.2)
    y0 = 3.0
    ax.barh(y0, K.T_AGE, color=V4["lag"], height=0.6)
    ax.text(K.T_AGE / 2, y0, f"{K.T_AGE}", ha="center", va="center",
            fontsize=FS_S, color="white", fontweight="bold")
    ax.text(K.T_AGE / 2, y0 - 0.38, "capture to\ndispatch", ha="center", va="top",
            fontsize=FS_S - 1, color=V4["lag"])
    a2 = 1 if step >= 2 else 0
    segs = [("inference", K.T_INFER, V4["proposal"]),
            ("check", K.T_VALIDATE, V4["nervous"]),
            ("", K.T_DA - K.T_INFER - K.T_VALIDATE, V4["gray"])]
    x = K.T_AGE
    for lab, w, col in segs:
        ax.barh(y0, w, left=x, color=col, height=0.6, alpha=a2)
        if lab:
            ax.text(x + w / 2, y0, f"{w:.0f}", ha="center", va="center", fontsize=FS_S,
                    color="white", fontweight="bold", alpha=a2)
            ax.text(x + w / 2, y0 - (0.75 if lab == "check" else 0.38), lab, ha="center",
                    va="top", fontsize=FS_S - 1,
                    color=col, alpha=a2)
        x += w
    ax.text(x + 2, y0, f"{K.T_REPL:.1f} ms", va="center", fontsize=FS + 1,
            fontweight="bold", color=V4["text"], alpha=a2)
    ax.text(-3, y0, "One replacement", ha="right", va="center", fontsize=FS, fontweight="bold")
    for xv, lab, ls, ha in ((K.T_RENEW, "renewal 50 ", "--", "right"), (K.T_LEASE, " lease 60", ":", "left")):
        ax.plot([xv, xv], [2.55, 3.55], color=V4["veto"], lw=1.5, ls=ls)
        ax.text(xv, 3.58, lab, color=V4["veto"], fontsize=FS_S, ha=ha, va="bottom")
    a3 = 1 if step >= 3 else 0
    for i in range(3):
        st = i * K.T_RENEW
        y = 1.6 - i * 0.62
        ax.barh(y - 0.35, K.T_REPL, left=st, color=V4["proposal_l"], ec=V4["proposal"],
                height=0.45, alpha=a3)
        ax.plot([st + K.T_REPL], [y - 0.35], "v", color=V4["nervous"], ms=8, alpha=a3)
        ax.text(-3, y - 0.35, f"request {i + 1}", ha="right", va="center", fontsize=FS_S, alpha=a3)
    ax.text(K.T_REPL + 2 * K.T_RENEW + 6, 1.6 - 2 * 0.62 - 0.35, "admitted", color=V4["nervous"],
            fontsize=FS_S, va="center", alpha=a3)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.45, 3.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds from capture (allocations, illustrative)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f"repl_step{step}.svg")


for s in (1, 2, 3):
    repl(s)


# ============================================================== chunk configs
def chunks():
    fig, ax = plt.subplots(figsize=cm(14.2, 4.4))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.95, bottom=0.27)
    for i, (n, h, e, _) in enumerate(K.CHUNKS):
        y = 2 - i
        ax.barh(y, h, color=V4["proposal_l"], ec=V4["proposal"], height=0.6, ls="--")
        ax.barh(y, e, color=V4["nervous"], height=0.6)
        ax.text(h + 0.8, y, f"executes {e} of {h}", va="center", fontsize=FS, color=V4["text"])
        ax.text(-1, y, n, ha="right", va="center", fontsize=FS, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Steps per chunk; green: executed before replacement",
                  fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "chunks.svg")


chunks()


# ============================================================== stale anchor
def stale():
    fig, ax = plt.subplots(figsize=cm(8.4, 5.6))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.93, bottom=0.21)
    t = np.linspace(-20, 130, 100)
    q = K.W_JOINT * t / 1000
    ax.plot(t, q, color=V4["nervous"], lw=2.5)
    ax.text(125, K.W_JOINT * 0.125, "active chunk", color=V4["nervous"], fontsize=FS_S,
            ha="right", va="bottom", fontweight="bold")
    t_s = 1000 * K.T_DELAY
    ax.plot([0], [0], "o", color=V4["veto"], ms=7)
    ax.plot([t_s], [K.GAP], "o", color=V4["nervous"], ms=8)
    ax.plot([t_s, t_s], [K.GAP, 0], color=V4["veto"], lw=2, ls="--")
    ax.plot([0, t_s], [0, 0], color=V4["veto"], lw=1, ls=":")
    ax.text(8, -0.0105, "request: stale anchor", color=V4["veto"], fontsize=FS_S, ha="left",
            va="bottom")
    ax.text(t_s + 4, K.GAP / 2, f"{K.GAP:.3f} rad\nstep back", color=V4["veto"],
            fontsize=FS_S, va="center")
    ax.text(t_s - 4, K.GAP + 0.004, "seam: plan\nfrom here", color=V4["nervous"],
            fontsize=FS_S, ha="right", va="bottom")
    ax.set_xlabel("Time after the request (ms)", fontsize=FS)
    ax.set_ylabel("Joint angle (rad)", fontsize=FS)
    ax.set_ylim(-0.012, 0.07)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "stale.svg")


stale()


# ============================================================== overshoot between samples
def overshoot():
    fig, ax = plt.subplots(figsize=cm(8.4, 5.0))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.18)
    t = np.linspace(0, 1, 100)
    vb = 0.95
    v = np.where(t < 0.5, vb + 0.6 * t, vb + 0.6 * (1 - t))
    ax.plot(t, v, color=V4["veto"], lw=2.5)
    ax.plot([0, 1], [vb, vb], "o", color=V4["proposal"], ms=9, zorder=5)
    ax.axhline(1.0, color=V4["plant"], lw=1.4, ls="--")
    ax.axhline(0.70, color=V4["nervous"], lw=1.4, ls=":")
    ax.text(0.5, 0.97, "limit", fontsize=FS_S, color=V4["plant"], va="top", ha="center")
    ax.text(0.5, 0.66, "contracted limit", fontsize=FS_S, color=V4["nervous"], va="top",
            ha="center")
    ax.text(0.5, 1.3, "overspeed between\ntwo legal samples", fontsize=FS_S,
            color=V4["veto"], ha="center", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sample k", "sample k + 1"])
    ax.set_xlim(-0.12, 1.12)
    ax.set_yticks([])
    ax.set_ylim(0.45, 1.6)
    ax.set_ylabel("Joint speed", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "overshoot.svg")


overshoot()


# ============================================================== seam spike
def seam(step):
    fig, axes = plt.subplots(2, 2, figsize=cm(14.6, 5.9), sharex="col",
                             gridspec_kw=dict(height_ratios=[1, 1.2]))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.9, bottom=0.17, hspace=0.25, wspace=0.18)
    # left: one-tick step
    t = np.linspace(-10, 60, 701)
    v_step = np.where(t < 0, 0, np.where(t < 1, 0.2 * t, 0.2))
    tau_step = np.where((t >= 0) & (t < 1), K.TAU_STEP, 0)
    tb = np.clip(t / 50, 0, 1)
    v_br = 0.2 * (10 * tb ** 3 - 15 * tb ** 4 + 6 * tb ** 5)
    acc_br = np.where((t >= 0) & (t <= 50),
                      0.2 / 0.05 * (30 * tb ** 2 - 60 * tb ** 3 + 30 * tb ** 4), 0)
    tau_br = K.J_REFL * acc_br
    cols = [(v_step, tau_step, "One servo tick", V4["veto"]),
            (v_br, tau_br, "50 ms quintic bridge", V4["nervous"])]
    for c, (v, tau, title, col) in enumerate(cols):
        a = 1 if (c == 0 or step >= 2) else 0
        axv, axt = axes[0, c], axes[1, c]
        axv.plot(t, v, color=col, lw=2.5, alpha=a)
        axv.set_title(title, fontsize=FS, fontweight="bold", color=col, alpha=a)
        axt.fill_between(t, 0, tau, color=col, alpha=0.35 * a, lw=0)
        axt.plot(t, tau, color=col, lw=2, alpha=a)
        for yv, lab in ((K.TAU_RATED, f"rated {K.TAU_RATED:.0f}"), (K.TAU_PEAK, f"peak {K.TAU_PEAK:.0f}")):
            axt.axhline(yv, color=V4["plant"], lw=1, ls="--")
            axt.text(58, yv + 3, lab, fontsize=FS_S - 1, color=V4["plant"], ha="right", va="bottom")
        pk = K.TAU_STEP if c == 0 else K.J_REFL * K.A_BRIDGE
        axt.text(5 if c == 0 else 25, 70 if c == 0 else pk + 12, f"{pk:.1f} N·m", fontsize=FS, color=col,
                 fontweight="bold", ha="left" if c == 0 else "center", alpha=a)
        axv.set_ylim(-0.03, 0.26)
        axt.set_ylim(0, 150)
        axt.set_xlim(-10, 60)
        axv.tick_params(labelsize=FS_S - 1)
        axt.tick_params(labelsize=FS_S - 1)
        axt.set_xlabel("ms from the seam", fontsize=FS_S)
        if c == 0:
            axv.set_ylabel("rad/s", fontsize=FS_S)
            axt.set_ylabel("N·m", fontsize=FS_S)
        if c == 1 and step < 2:
            for ax_ in (axv, axt):
                for s_ in ax_.spines.values():
                    s_.set_alpha(0)
                ax_.tick_params(colors="white")
                ax_.xaxis.label.set_color("white")
    save_slot(fig, f"seam_step{step}.svg")


for s in (1, 2):
    seam(s)


# ============================================================== phase plane
def scurve_profile(dq, a, j):
    """Rest-to-rest jerk-limited swing, no cruise: (position, speed) samples."""
    ramp = a / j
    t1 = (-ramp + math.sqrt(ramp ** 2 + 4 * dq / a)) / 2
    T = 2 * (t1 + ramp)
    tt = np.linspace(0, T, 4000)
    acc = np.zeros_like(tt)
    # acceleration: ramp up, hold, ramp down to zero at T/2, mirror for braking
    h = T / 2
    for i, x in enumerate(tt):
        s = x if x <= h else T - x
        sign = 1 if x <= h else -1
        if s < ramp:
            acc[i] = sign * j * s
        elif s < h - ramp:
            acc[i] = sign * a
        else:
            acc[i] = sign * j * (h - s)
    v = np.concatenate([[0], np.cumsum((acc[1:] + acc[:-1]) / 2 * np.diff(tt))])
    p = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(tt))])
    return p, v


def phase():
    fig, ax = plt.subplots(figsize=cm(14.2, 5.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.22)
    s = np.linspace(0, K.SWING, 300)
    up = (K.TAU_MAX - K.TAU_GRAV) / K.J_EFF
    dn = (K.TAU_MAX + K.TAU_GRAV) / K.J_EFF
    ax.plot(s, np.sqrt(2 * up * s), color=V4["gray"], lw=1.3, ls="--")
    ax.plot(s, np.sqrt(2 * dn * (K.SWING - s)), color=V4["gray"], lw=1.3, ls="--")
    ax.text(0.03, 3.3, "torque limit alone", fontsize=FS_S, color=V4["gray"])
    a = K.QDDOT_MAX
    bang = np.minimum(np.sqrt(2 * a * s), np.sqrt(2 * a * (K.SWING - s)))
    ax.plot(s, bang, color=V4["veto"], lw=4.5)
    p, v = scurve_profile(K.SWING, a, K.JERK_SW)
    ax.plot(p, v, color=V4["nervous"], lw=1.8)
    ax.axhline(K.QDOT_MAX, color=V4["plant"], lw=1.2, ls=":")
    ax.text(0.10, K.QDOT_MAX + 0.06, "velocity limit 2.175", fontsize=FS_S,
            color=V4["plant"], ha="left", va="bottom")
    ax.plot([K.SWING / 2], [K.V_PEAK_SW], "o", color=V4["veto"], ms=7)
    ax.annotate(f"switch point: {2 * K.J_EFF * a:.1f} N·m torque step",
                (K.SWING / 2, K.V_PEAK_SW), xytext=(0.25, 2.9), fontsize=FS_S, ha="center",
                color=V4["veto"], arrowprops=dict(arrowstyle="-", color=V4["veto"], lw=1))
    ax.text(0.13, 0.75, f"bang-bang {1000 * K.T_BANG:.1f} ms", fontsize=FS, color=V4["veto"],
            fontweight="bold", ha="left")
    ax.text(0.13, 0.42, f"jerk-limited {1000 * K.T_S:.1f} ms", fontsize=FS,
            color=V4["nervous"], fontweight="bold", ha="left")
    ax.set_xlim(0, K.SWING)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel("Shoulder position along the swing (rad)", fontsize=FS)
    ax.set_ylabel("Speed (rad/s)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "phase.svg")


phase()


# ============================================================== Ruckig arm stop
def ruckig_arm():
    d = RK["arm"]
    fig, (av, aa) = plt.subplots(2, 1, figsize=cm(9.6, 6.0), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.17, hspace=0.12)
    t = 1000 * d["t"]
    tc = np.array([0, 1000 * K.V_TCP / K.A_ARM])
    av.plot(tc, [K.V_TCP, 0], color=V4["gray"], lw=1.6, ls="--")
    av.plot(t, d["v"], color=V4["veto"], lw=2.6)
    av.text(8, 0.06, "constant 5 m/s²:\n200 ms, 100 mm", fontsize=FS_S, color=V4["gray"])
    av.text(1000 * K.T_STOP_ARM - 4, 0.55,
            f"Ruckig: {1000 * d['t'][-1]:.0f} ms,\n{1000 * d['p'][-1]:.1f} mm",
            fontsize=FS, color=V4["veto"], fontweight="bold", ha="right")
    av.set_ylabel("m/s", fontsize=FS_S)
    aa.plot([0, 0, tc[1], tc[1]], [0, -K.A_ARM, -K.A_ARM, 0], color=V4["gray"], lw=1.6, ls="--")
    aa.plot(t, d["a"], color=V4["veto"], lw=2.6)
    aa.text(85, -2.6, "constant: steps,\ninfinite jerk", fontsize=FS_S, color=V4["gray"])
    aa.set_ylabel("m/s²", fontsize=FS_S)
    aa.set_xlabel("ms from stop onset", fontsize=FS_S)
    for ax in (av, aa):
        ax.tick_params(labelsize=FS_S - 1)
    aa.set_xlim(-5, 290)
    save_slot(fig, "ruckig_arm.svg")


ruckig_arm()


# ============================================================== stall runway
def runway(step):
    fig, ax = plt.subplots(figsize=cm(14.2, 3.9))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.93, bottom=0.3)
    rows = [("Empty gripper", K.D_STOP_ARM, K.TOTAL_ARM, f"{K.A_ARM:.0f} m/s²"),
            ("Open mug", K.D_STOP_MUG, K.TOTAL_MUG, f"{K.A_MUG:.2f} m/s²")]
    for i, (lab, stop, tot, acc) in enumerate(rows):
        y = 1 - i
        a = 1 if i < step else 0
        ax.barh(y, K.D_PATH, color=V4["lag"], height=0.55, alpha=a)
        ax.barh(y, stop, left=K.D_PATH, color=V4["veto"], height=0.55, alpha=a)
        ax.text(K.D_PATH / 2, y, f"{K.D_PATH:.0f}", ha="center", va="center", color="white",
                fontsize=FS, fontweight="bold", alpha=a)
        ax.text(K.D_PATH + stop / 2, y, f"stop {stop:.1f} ({acc})", ha="center",
                va="center", color="white", fontsize=FS_S, fontweight="bold", alpha=a)
        ax.text(tot + 4, y, f"{tot:.1f}", va="center", fontsize=FS + 1, fontweight="bold",
                color=V4["veto"] if tot > K.D_CLEAR_ARM else V4["nervous"], alpha=a)
        ax.text(-5, y, lab, ha="right", va="center", fontsize=FS, fontweight="bold")
    ax.axvline(K.D_CLEAR_ARM, color=V4["plant"], lw=2, ls="--")
    ax.text(K.D_CLEAR_ARM + 4, 1.45, "clearance 300", ha="left", fontsize=FS_S, color=V4["plant"])
    ax.text(K.D_PATH / 2, 1.45, "lease path", ha="center", fontsize=FS_S, color=V4["lag"])
    ax.set_xlim(0, 410)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Travel after the last admission at 1 m/s (mm)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f"runway_step{step}.svg")


for s in (1, 2):
    runway(s)


# ============================================================== two schedules
def timeline(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 4.8))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.93, bottom=0.25)
    h = 0.42

    def seg(y, x0, x1, col, lab=None, a=1.0, hatch=None, fc=None):
        ax.barh(y, x1 - x0, left=x0, height=h, color=fc or col, ec=col, hatch=hatch,
                alpha=a, lw=1.2)
        if lab:
            ax.text((x0 + x1) / 2, y, lab, ha="center", va="center", fontsize=FS_S,
                    color="white" if fc is None else col, fontweight="bold", alpha=a)

    y1, y2 = 1.0, 0.0
    seg(y1, 0, K.T_LEASE, V4["nervous"], "track")
    seg(y1, K.T_LEASE, K.T_PATH, V4["lag"])
    seg(y1, K.T_PATH, K.T_PATH + 1000 * K.T_STOP_ARM, V4["veto"], "S-curve stop")
    ax.text(K.T_PATH + 1000 * K.T_STOP_ARM + 6, y1,
            f"{K.TOTAL_ARM:.1f} of 300 mm", va="center", fontsize=FS, color=V4["text"])
    ax.text(K.T_LEASE, y1 + 0.26, "lease lapses 60", ha="center", fontsize=FS_S, color=V4["lag"])
    ax.text(-8, y1, "Running lease", ha="right", va="center", fontsize=FS, fontweight="bold")
    a = 1 if step >= 2 else 0
    seg(y2, 0, K.T_COMMIT, V4["nervous"], "track", a=a)
    ax.plot([K.T_COMMIT, K.N_SETPOINTS * K.T_SETPOINT], [y2 - 0.48, y2 - 0.48], color=V4["proposal"],
            lw=2, ls="--", alpha=a)
    ax.text(K.N_SETPOINTS * K.T_SETPOINT + 6, y2 - 0.48, "setpoints to 400, unused",
            color=V4["proposal"], fontsize=FS_S - 1, va="center", alpha=a)
    seg(y2, K.T_COMMIT, K.T_END, V4["veto"], a=a)
    ax.text((K.T_COMMIT + K.T_END) / 2, y2, "stop", ha="center", va="center",
            fontsize=FS_S, color="white", fontweight="bold", alpha=a)
    for xv, lab, ha in ((K.T_PREF, "preferred 240 ", "right"), (K.T_CUTOFF, "cutoff 278 ", "below"),
                        (K.T_COMMIT, " commit 300", "left")):
        ax.plot([xv, xv], [y2 - 0.22, y2 + 0.32], color=V4["text"], lw=1.5, alpha=a)
        below = ha == "below"
        ax.text(xv, y2 - 0.24 if below else y2 + 0.34, lab, ha="right" if below else ha,
                fontsize=FS_S, color=V4["text"], alpha=a, va="top" if below else "bottom")
    ax.text(K.T_END + 6, y2, f"{K.TOTAL_ALT:.1f} of 300 mm", va="center", fontsize=FS,
            color=V4["text"], alpha=a)
    ax.text(-8, y2, "Longer lease", ha="right", va="center", fontsize=FS, fontweight="bold",
            alpha=a)
    ax.set_xlim(0, 760)
    ax.set_ylim(-0.75, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds after tracking starts", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, f"timeline_step{step}.svg")


for s in (1, 2):
    timeline(s)


# ============================================================== base stop
def base(step):
    fig, (av, ab) = plt.subplots(1, 2, figsize=cm(14.6, 5.3),
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.21, wspace=0.55)
    d = RK["base"]
    tc = 1000 * K.V_BASE / K.A_BASE
    av.plot([0, tc], [K.V_BASE, 0], color=V4["gray"], lw=2, ls="--")
    av.plot(1000 * d["t"], d["v"], color=V4["veto"], lw=2.6)
    tp = np.linspace(0, 1, 200)
    av.plot(K.T_POLY * tp, K.V_BASE * (1 - 3 * tp ** 2 + 2 * tp ** 3), color=V4["lag"], lw=2)
    for i, (lab, col) in enumerate(((f"constant {tc:.0f} ms", V4["gray"]),
                                    (f"S-curve {K.T_SC:,.2f} ms", V4["veto"]),
                                    (f"polynomial {K.T_POLY:,.0f} ms", V4["lag"]))):
        av.text(1290, 1.55 - i * 0.2, lab, color=col, fontsize=FS_S - 1, ha="right",
                va="top", fontweight="bold")
    av.set_xlabel("ms from stop onset", fontsize=FS_S)
    av.set_ylabel("Base speed (m/s)", fontsize=FS_S)
    av.set_xlim(0, 1300)
    av.set_ylim(0, 1.65)
    av.tick_params(labelsize=FS_S - 1)
    a = 1 if step >= 2 else 0
    rows = [("constant", K.D_CONST, V4["gray"]), ("S-curve", K.D_SC, V4["veto"]),
            ("polynomial", K.D_POLY, V4["lag"])]
    other = K.BUDGET_BEFORE - K.D_CONST
    for i, (lab, dd, col) in enumerate(rows):
        y = 2 - i
        ab.barh(y, other, color=V4["gray_l"], height=0.6, alpha=a)
        ab.barh(y, dd, left=other, color=col, height=0.6, alpha=a)
        tot = other + dd
        ab.text(tot + 15, y, f"{tot:,.0f}", va="center", fontsize=FS_S, fontweight="bold",
                color=col, alpha=a)
        ab.text(-20, y, lab, ha="right", va="center", fontsize=FS_S - 1,
                fontweight="bold", color=col)
    ab.axvline(K.D_CLEAR_BASE, color=V4["plant"], lw=2, ls="--")
    ab.text(K.D_CLEAR_BASE + 20, 2.55, "1,100", ha="left", fontsize=FS_S, color=V4["plant"])
    ab.text(0, 2.42, f"gray: delay {K.DELAY_BASE:.1f}\n+ allowances {K.ALLOW_BASE:.0f}",
            ha="left", va="bottom", fontsize=FS_S - 1, color=V4["plant"], alpha=a)
    ab.set_xlim(0, 1450)
    ab.set_ylim(-0.5, 3.45)
    ab.set_yticks([])
    ab.spines["left"].set_visible(False)
    ab.set_xlabel("Stopping budget at 1.5 m/s (mm)", fontsize=FS_S)
    ab.tick_params(labelsize=FS_S - 1)
    if step < 2:
        for s_ in ab.spines.values():
            s_.set_alpha(0)
        ab.tick_params(colors="white")
        ab.xaxis.label.set_color("white")
    save_slot(fig, f"base_step{step}.svg")


for s in (1, 2):
    base(s)


# ============================================================== generator timing
def gen_timing():
    rows = [
        ("Ruckig, mean", K.RUCKIG_MEAN, "nervous", "19.8 µs"),
        ("Ruckig, worst case", K.RUCKIG_WORST, "nervous", "123 µs"),
        ("Servo tick", 1.0, "neutral", "1 ms"),
        ("Plan validation", K.T_VALIDATE, "neutral", "18 ms"),
        ("Chunk policy pass", K.T_INFER, "memory", "40 ms"),
        ("Capture to admission", K.T_REPL, "danger", f"{K.T_REPL:.1f} ms"),
    ]
    fig, ax = plt.subplots(figsize=cm(14.2, 5.2))
    fig.subplots_adjust(left=0.31, right=0.97, top=0.97, bottom=0.24)
    n = len(rows)
    cmap = {"nervous": V4["nervous"], "neutral": V4["gray"], "memory": V4["proposal"],
            "danger": V4["veto"]}
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=cmap[ck], height=0.62)
        ax.text(v * 1.2, y, txt, va="center", fontsize=FS, fontweight="bold", color=cmap[ck])
        ax.text(0.0075, y, lab, ha="right", va="center", fontsize=FS, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3000)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds (log scale)", fontsize=FS)
    ax.tick_params(labelsize=FS_S)
    save_slot(fig, "gen_timing.svg")


gen_timing()


# ============================================================== base top view
def base_view():
    fig, axes = plt.subplots(1, 2, figsize=cm(9.6, 4.6))
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98, wspace=0.08)

    def body(ax, x, y, th, a=1.0):
        c, s = math.cos(th), math.sin(th)
        pts = np.array([[-0.3, -0.22], [0.3, -0.22], [0.3, 0.22], [-0.3, 0.22]])
        R = np.array([[c, -s], [s, c]])
        P = pts @ R.T + [x, y]
        ax.add_patch(Polygon(P, closed=True, fc=V4["plant_l"], ec=V4["plant"], lw=1.8, alpha=a))
        for wy in (-0.26, 0.26):
            w = np.array([[-0.12, wy - 0.04], [0.12, wy - 0.04], [0.12, wy + 0.04], [-0.12, wy + 0.04]])
            ax.add_patch(Polygon(w @ R.T + [x, y], closed=True, fc=V4["plant"], ec=V4["plant"], alpha=a))

    ax = axes[0]
    body(ax, 0, 0, 0)
    ax.annotate("", xy=(0.75, 0), xytext=(0.32, 0),
                arrowprops=dict(arrowstyle="-|>", color=V4["nervous"], lw=2.5))
    ax.annotate("", xy=(0, 0.75), xytext=(0, 0.32),
                arrowprops=dict(arrowstyle="-|>", color=V4["veto"], lw=2, ls="--"))
    ax.plot([-0.12, 0.12], [0.45, 0.62], color=V4["veto"], lw=3)
    ax.plot([-0.12, 0.12], [0.62, 0.45], color=V4["veto"], lw=3)
    ax.text(0.52, -0.12, "drive", fontsize=FS_S, color=V4["nervous"], ha="center", va="top")
    ax.text(0.16, 0.78, "no sideways step", fontsize=FS_S, color=V4["veto"], ha="left", va="center")
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.6, 0.95)
    ax.set_aspect("equal")
    hide_axis(ax)
    ax = axes[1]
    body(ax, 0, 0, 0, a=0.4)
    body(ax, 0, 0.9, 0)
    ax.plot([0, 0], [0.25, 0.65], color=V4["nervous"], lw=2.5)
    for xy, n in (((0.38, 0.05), "1 turn"), ((0.12, 0.45), "2 drive"), ((0.38, 0.95), "3 turn")):
        ax.text(*xy, n, fontsize=FS_S, color=V4["nervous"], fontweight="bold", ha="left")
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.45, 1.3)
    ax.set_aspect("equal")
    hide_axis(ax)
    save_slot(fig, "base_view.svg")


base_view()
print("figures written")
