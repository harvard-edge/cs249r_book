#!/usr/bin/env python3
"""Figures for Lecture 12 (Safety Enforcement).

Every value comes from calcs.py (imported as K), which reads MLSysIM: the
running machine and its aisle record, the finished stopping budgets, the
loaded arm joint and its kinetic barrier, the two-joint projection, the
permission check's stage budget, Literature.ContactForceMeasurements
(Kirschner et al. 2022, Table II, hand, B1), and Literature.IncidentReports
(Cruise, Therac-25). Labels print the strings calcs.py recorded, through
deckvals.val; geometry uses the full-precision values. "SE" is the rendered
chapter, Safety Enforcement (Volume IV, Chapter 12).
Diagram colors follow the Volume IV causal-boundary palette
(rules/volumes/vol4/palette.md); red only for veto, hazard, and limits.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402

import calcs as K  # noqa: E402  (prints its table on import)
from deckvals import val  # noqa: E402

C = sp.C
# Volume IV roles
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL = (
    "#1F407A", "#2563EB", "#059669", "#DC2626", "#D97706", "#334155", "#0F766E")
GREEN_F, BLUE_F, RED_F, SLATE_F = "#DCFCE7", "#EFF6FF", "#FEE2E2", "#F8FAFC"
TXT = "#1A1A2E"
FONT = "Helvetica, Arial, sans-serif"


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", format="svg", bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)


# =============================================================================
# SVG helpers
# =============================================================================
DEFS = f"""<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>
<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>
<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>
<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>
</defs>"""


def svg(name, w, h, body):
    (HERE / f"{name}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + "\n".join(body) + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, dash=None, rx=8, sw=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=22, fill=TXT, anchor="middle", weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}>{s}</text>')


def lines(x, y, rows, size=22, fill=TXT, anchor="middle", weight="normal", gap=None):
    gap = gap or size * 1.25
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]


def arrow(x1, y1, x2, y2, color=SLATE, sw=3, dash=None, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


# =============================================================================
# Hook: Therac-20 and Therac-25 (SE War Story 12.1); drawn for a 4.3 cm slot
# =============================================================================
def therac(name):
    b = []
    rows = [("Therac-20", 40, True), ("Therac-25", 200, False)]
    for title, y0, hw in rows:
        b.append(text(10, y0 + 2, title, 26, TXT, anchor="start", weight="bold"))
        y = y0 + 18
        xs = [10, 240, 470, 700]
        b.append(box(xs[0], y, 190, 92, BLUE_F, NAVY))
        b += lines(xs[0] + 95, y + 40, ["Operator", "console"], 24)
        b.append(box(xs[1], y, 190, 92, GREEN_F, GREEN))
        b += lines(xs[1] + 95, y + 40, ["Software", "check"], 24)
        if hw:
            b.append(box(xs[2], y, 190, 92, RED_F, RED))
            b += lines(xs[2] + 95, y + 40, ["Turntable", "microswitch"], 24, "#991B1B", weight="bold")
        else:
            b.append(box(xs[2], y, 190, 92, "white", "#94A3B8", dash="8,6"))
            b += lines(xs[2] + 95, y + 40, ["removed"], 24, "#64748B")
        b.append(box(xs[3], y, 190, 92, SLATE_F, SLATE, rx=0))
        b += lines(xs[3] + 95, y + 40, ["Beam", "fires"], 24)
        for i in range(3):
            b.append(arrow(xs[i] + 192, y + 46, xs[i + 1] - 2, y + 46))
    svg(name, 900, 320, b)


therac("therac_hook")


# =============================================================================
# The per-tick gatekeeper: proposer, permission path, drives (SE 12.1, 12.8)
# =============================================================================
def gatekeeper(step):
    b = []
    b.append(box(10, 20, 230, 120, BLUE_F, BLUE, dash="8,5"))
    b += lines(125, 60, ["Learned", "proposer"], 25, NAVY, weight="bold")
    b.append(text(125, 122, "proposes only", 22, BLUE))
    b.append(f'<line x1="262" y1="5" x2="262" y2="390" stroke="#475569" stroke-width="2" stroke-dasharray="6,4"/>')
    b.append(text(20, 175, "proposal", 22, "#475569", anchor="start"))
    b.append(text(20, 200, "boundary", 22, "#475569", anchor="start"))
    b.append(arrow(240, 80, 288, 80, BLUE, marker="ahb", dash="8,5"))
    b.append(box(290, 20, 300, 120, GREEN_F, GREEN))
    b += lines(440, 58, ["Enforcer on the MCU", f"every {val('t_tick')} ms tick"], 25, "#047857", weight="bold")
    b.append(text(440, 124, "admit, correct, fall back", 22, "#047857"))
    b.append(arrow(590, 80, 638, 80, GREEN, marker="ahg"))
    b.append(box(640, 20, 250, 120, SLATE_F, SLATE, rx=0))
    b += lines(765, 72, ["Drives and", "brakes"], 25, weight="bold")
    if step >= 2:
        b.append(box(290, 245, 300, 120, RED_F, RED))
        b += lines(440, 290, ["Certified floor:", "scanner, STO, brakes"], 24, "#991B1B", weight="bold")
        b.append(f'<path d="M590,305 L765,305 L765,143" fill="none" stroke="{RED}" stroke-width="3" marker-end="url(#ahr)"/>')
        b.append(text(612, 345, "hardware path", 22, RED, anchor="start"))
        b.append(arrow(440, 243, 440, 143, RED, dash="6,4", marker="ahr"))
        b.append(text(450, 200, "if the MCU fails", 22, RED, anchor="start"))
    svg(f"gate_step{step}", 900, 375, b)


for s in (1, 2):
    gatekeeper(s)

# =============================================================================
# Recall thumbnail (Lecture 11): constant deceleration against the S-curve
# =============================================================================
fig, ax = plt.subplots(figsize=(4.4, 2.2))
t = np.linspace(0, K.T_SCURVE_DRIVE, 400)
ax.plot([0, K.T_CONST_DRIVE], [K.V_DRIVE, 0], color=C["neutral"], lw=3, ls="--")
v = []
tj = K.A_BRAKE / K.J
for ti in t:
    if ti < tj:
        v.append(K.V_DRIVE - 0.5 * K.J * ti ** 2)
    elif ti < K.T_SCURVE_DRIVE - tj:
        v.append(K.V_DRIVE - 0.5 * K.A_BRAKE * tj - K.A_BRAKE * (ti - tj))
    else:
        tr = K.T_SCURVE_DRIVE - ti
        v.append(0.5 * K.J * tr ** 2)
ax.plot(t, v, color=GREEN, lw=3.5)
ax.set_xticks([]); ax.set_yticks([])
save(fig, "recall_stop")

# The budget build is the course kit's stopping-budget bar (assets/kit/vol4,
# budget_bar_step5 to step7 and budget_bar_aisle_step7); its goals and
# takeaways thumbnail, thumb_budget_bar_aisle_step7.svg, is the kit bar run
# through tools/greek_thumbs.py.

# =============================================================================
# Three ceilings (SE figure 3)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.9, 2.7))
vs = np.linspace(0.02, 1.7, 300)
fixed = K.FIXED * 1e3


def need(v, a=K.A_BRAKE, walker=0.0):
    return (v * K.T_DELAY + K.stop_dist(v, a) + walker * (K.T_DELAY + K.stop_time(v, a))) * 1e3 + fixed


ax.plot(vs, [need(x) for x in vs], color=SLATE, lw=3)
ax.plot(vs, [need(x, K.A_INSP) for x in vs], color=AMBER, lw=3, ls="--")
vw = vs[vs < 0.6]
ax.plot(vw, [need(x, walker=K.V_WALK) for x in vw], color=RED, lw=3, ls=":")
ax.axhline(K.D_CLEAR * 1e3, color=C["neutral"], lw=1.5, ls="--")
ax.text(0.62, 1130, f"clear distance {val('d_clear_mm')} mm", ha="left", fontsize=10.5, color=C["neutral"])
for vv, key, col, lab, y in [(K.V_CEIL, "ceiling", SLATE, "dry floor", 440),
                             (K.V_CEIL_INSP, "ceiling_inspected", AMBER,
                              f"floor at friction {val('mu_inspected')}", 300),
                             (K.V_WALKER, "v_walk", RED, "person walking toward the base", 160)]:
    ax.plot([vv], [K.D_CLEAR * 1e3], "o", ms=9, color=col, mec="white", mew=1.5, zorder=5)
    ax.text(0.9, y, f"{lab}: {val(key)} m/s", color=col, fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.7)
ax.set_ylim(0, 1600)
ax.set_xlabel("Speed when the hazard is seen (m/s)", fontsize=11)
ax.set_ylabel("Stop needs (mm)", fontsize=11)
ax.tick_params(labelsize=10)
save(fig, "ceilings")

# =============================================================================
# Phase portrait and the per-tick governor (SE figure 5)
# =============================================================================
def portrait(step):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    D = np.linspace(0, 2.0, 200)
    ax.plot(D, np.sqrt(2 * K.A_BRAKE * D), color=C["neutral"], lw=2.5, ls="--")
    ax.text(0.33, 1.9, "braking only", color=C["neutral"], fontsize=12)
    if step >= 2:
        inset = [K.ceiling_at(d) for d in D]
        ax.fill_between(D, 0, inset, color=GREEN_F)
        ax.plot(D, inset, color=GREEN, lw=3)
        ax.text(1.6, 0.35, "stopping\nenvelope", color="#047857", fontsize=13,
                fontweight="bold", ha="center")
        ax.plot([K.D_CLEAR], [K.V_CEIL], "o", color=GREEN, ms=9, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"ceiling {val('ceiling')} m/s", (K.D_CLEAR, K.V_CEIL), xytext=(1.22, 0.98),
                    ha="left", fontsize=12, color="#047857",
                    fontweight="bold", arrowprops=dict(arrowstyle="-", color="#047857"))
    if step >= 3:
        ax.plot([K.D_CLEAR], [K.V_AISLE], "s", color=NAVY, ms=8, zorder=6)
        ax.text(1.22, 1.2, f"aisle {val('v_aisle')} m/s", color=NAVY, fontsize=12, fontweight="bold")
        ax.plot([0.6], [K.V_AISLE], "X", color=RED, ms=11, zorder=6)
        ax.annotate("outside:\nSTOP", (0.6, K.V_AISLE), xytext=(0.03, 1.66), color=RED, fontsize=12,
                    fontweight="bold", ha="left", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Clear distance ahead (m)", fontsize=12)
    ax.set_ylabel("Speed (m/s)", fontsize=12)
    ax.tick_params(labelsize=11)
    save(fig, f"portrait_step{step}")


for s in (1, 2, 3):
    portrait(s)

# =============================================================================
# Gravity braking tax (SE Napkin Math 12.2, figure 6b)
# =============================================================================
fig, ax = plt.subplots(figsize=(4.6, 2.4))
data = [("Ascending", K.STOP_UP, "stop_up", GREEN),
        ("No bias", K.STOP_FREE, "stop_free", SLATE),
        ("Descending", K.STOP_DOWN, "stop_down", RED)]
for i, (lab, w, key, col) in enumerate(data):
    ax.barh(i, w, color=col, height=0.6)
    ax.text(w + 1.5, i, f"{val(key)} mm", va="center", fontsize=13, color=col, fontweight="bold")
    ax.text(-1.5, i, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=TXT)
ax.set_xlim(0, 105)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Tool travel before the joint stops (mm)", fontsize=12)
ax.tick_params(labelsize=11)
save(fig, "gravity")

# =============================================================================
# Relative degree (SE 12.4.1)
# =============================================================================
def reldeg(step):
    b = []
    xs = [20, 250, 480, 710]
    labs = [["Motor", "torque"], ["Accel-", "eration"], ["Velocity"], ["Position"]]
    for i, (x, lab) in enumerate(zip(xs, labs)):
        b.append(box(x, 40, 170, 90, GREEN_F if i == 0 else SLATE_F, GREEN if i == 0 else SLATE))
        b += lines(x + 85, 78 if len(lab) == 2 else 92, lab, 23, weight="bold")
    b.append(arrow(190, 85, 248, 85)); b.append(text(219, 72, "÷ m", 19))
    b.append(arrow(420, 85, 478, 85)); b.append(text(449, 72, "∫ dt", 19))
    b.append(arrow(650, 85, 708, 85)); b.append(text(679, 72, "∫ dt", 19))
    # barrier on position
    b.append(box(560, 180, 320, 110, RED_F, RED))
    b += lines(720, 218, ["Barrier on distance alone:", "its rate is −velocity"], 21, "#991B1B", weight="bold")
    b.append(text(720, 276, "torque has no term in it", 20, "#991B1B"))
    b.append(arrow(795, 130, 760, 178, RED, marker="ahr"))
    if step >= 2:
        b.append(box(20, 180, 500, 110, GREEN_F, GREEN))
        b += lines(270, 218, ["Kinetic barrier: clearance minus", "the distance needed to stop"], 21, "#047857", weight="bold")
        b.append(text(270, 276, "its rate contains torque", 20, "#047857"))
        b.append(arrow(565, 130, 400, 178, GREEN, marker="ahg"))
    svg(f"reldeg_step{step}", 900, 300, b)


for s in (1, 2):
    reldeg(s)

# =============================================================================
# Barrier torque demand (SE 12.4.1 figure)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.9, 2.7))
cl = np.linspace(0.060, 0.140, 200)
dem = [K.barrier_demand(c) for c in cl]
ax.axhspan(K.TAU_MAX, 100, color=RED_F)
ax.axhline(K.TAU_MAX, color=RED, ls="--", lw=2)
ax.text(138, K.TAU_MAX + 1.6, f"joint peak {val('tau_max')} N·m", color=RED, ha="right", fontsize=11.5,
        fontweight="bold")
ax.plot(cl * 1e3, dem, color=GREEN, lw=3)
for c, ck, tk, col, dx, dy in [(K.CLEAR_FAR, "clear_far", "brake_far", GREEN, -10, 12),
                               (K.STOP_DOWN / 1e3, "stop_down", "brake_zero", NAVY, 20, -60),
                               (K.CLEAR_NEAR, "clear_near", "brake_near", RED, -6, 12)]:
    tq = K.barrier_demand(c)
    ax.plot([c * 1e3], [tq], "o", color=col, ms=9, mec="white", mew=1.5, zorder=5)
    ax.annotate(f"{val(ck)} mm: {val(tk)} N·m", (c * 1e3, tq), xytext=(dx, dy), textcoords="offset points",
                fontsize=11.5, color=col, fontweight="bold")
ax.text(64, 70, f"gravity alone\nasks {val('bias')} N·m", fontsize=10.5, color=C["neutral"])
ax.axhline(K.TAU_BIAS, color=C["neutral"], lw=1.2, ls=":")
ax.set_xlim(60, 140)
ax.set_ylim(60, 95)
ax.set_xlabel("Tool clearance to the packing station (mm)", fontsize=11)
ax.set_ylabel("Braking torque\nthe barrier asks (N·m)", fontsize=11)
ax.tick_params(labelsize=10)
save(fig, "barrier_torque")

# =============================================================================
# Projection against per-axis clamp (SE 12.6.1, figure 7b)
# =============================================================================
def projection(step):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    x = np.linspace(-1, 5, 50)
    yb = K.b - K.a[0] * x
    ax.fill_between(x, yb, -1, color=GREEN_F)
    ax.plot(x, yb, color=GREEN, lw=2.5)
    ax.text(0.62, 0.05, "admissible", color="#047857", fontsize=12, fontweight="bold")
    ax.text(0.95, 4.05, f"{val('proj_a1')}τ₁ + τ₂ = {val('proj_b')}", color="#047857", fontsize=11.5)
    ax.plot([0, K.u[0]], [0, K.u[1]], color=BLUE, lw=2, ls="--")
    ax.plot([K.u[0]], [K.u[1]], "o", color=BLUE, ms=10, zorder=5)
    ax.text(K.u[0] + 0.12, K.u[1] + 0.12, f"proposal ({val('proj_u1')}, {val('proj_u2')})", color=BLUE, fontsize=12, fontweight="bold")
    if step >= 2:
        ax.plot([K.clamp[0]], [K.clamp[1]], "s", color=RED, ms=9, zorder=5)
        ax.annotate("", (K.clamp[0], K.clamp[1]), (K.u[0], K.u[1]),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.text(K.clamp[0] - 0.15, K.clamp[1] + 0.35, (f"clamp ({val('proj_clamp1')}, {val('proj_clamp2')})\n"
                 f"{val('proj_a1')}·{val('proj_clamp1')} + {val('proj_clamp2')} = {val('proj_clamp_value')}"), color=RED,
                fontsize=11.5, fontweight="bold", ha="center")
    if step >= 3:
        ax.plot([K.ustar[0]], [K.ustar[1]], "D", color=GREEN, ms=9, zorder=5)
        ax.annotate("", (K.ustar[0], K.ustar[1]), (K.u[0], K.u[1]),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
        ax.text(0.1, 1.15, f"projection\n({val('proj_us1')}, {val('proj_us2')})", color="#047857",
                fontsize=12, fontweight="bold", ha="left")
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.2, 4.4)
    ax.set_aspect("equal")
    ax.set_xlabel("Joint 1 torque τ₁ (N·m)", fontsize=12)
    ax.set_ylabel("Joint 2 torque τ₂ (N·m)", fontsize=12)
    ax.tick_params(labelsize=11)
    save(fig, f"proj_step{step}")


for s in (1, 2, 3):
    projection(s)

# =============================================================================
# Latch force against time (SE equations 11, 12; figure 8)
# =============================================================================
def latch(step):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    v = K.V_LATCH
    lo = K.LATCH_LO
    rate = lo["df_dt"].to("N/s").magnitude
    f_on = lo["f_at_brake_onset"].to("N").magnitude
    acc = lo["a_arm_needed"].to("m/s**2").magnitude
    tstop = lo["t_arm_stop"].to("s").magnitude
    t_trip = K.TRIP / rate
    t_b = t_trip + K.T_LOOP
    tt = np.linspace(0, t_b, 50)
    ax.plot(tt * 1e3, rate * tt, color=GREEN, lw=3)
    # braked: force continues to rise as the arm slows (plate compresses v t - a t^2 / 2)
    t2 = np.linspace(0, tstop, 60)
    f2 = f_on + K.K_PLATE * (v * t2 - 0.5 * acc * t2 ** 2)
    ax.plot((t_b + t2) * 1e3, f2, color=GREEN, lw=3)
    ax.plot([(t_b + tstop) * 1e3, 30], [f2[-1], f2[-1]], color=GREEN, lw=3)
    tn = np.linspace(t_b, t_b + 6e-3, 20)
    ax.plot(tn * 1e3, rate * tn, color=RED, lw=2, ls="--")
    ax.text((t_b + 6.2e-3) * 1e3, 104, "no brake", color=RED, fontsize=11, fontweight="bold")
    ax.axvspan(t_trip * 1e3, t_b * 1e3, color=AMBER, alpha=0.18)
    ax.axhline(K.TRIP, color=C["neutral"], ls=":", lw=1.5)
    ax.text(29.5, K.TRIP + 3, f"tripwire {val('f_trip')} N", ha="right", fontsize=10.5, color=C["neutral"])
    ax.axhline(K.F_LATCH, color=RED, ls="--", lw=2)
    ax.axhspan(K.F_LATCH, 150, color=RED_F, alpha=0.6)
    ax.text(29.5, K.F_LATCH + 4, f"latch limit {val('f_latch')} N", ha="right", fontsize=11, color=RED, fontweight="bold")
    ax.plot([t_b * 1e3], [f_on], "o", color=GREEN, ms=9, mec="white", mew=1.5, zorder=5)
    ax.text(t_b * 1e3 + 0.6, f_on - 14, f"{val('v_latch')} m/s: brake at {val('f_cmd')} N", color="#047857",
            fontsize=11.5, fontweight="bold")
    if step >= 2:
        r2 = K.LATCH_HI["df_dt"].to("N/s").magnitude
        fo2 = K.LATCH_HI["f_at_brake_onset"].to("N").magnitude
        tt2 = np.linspace(0, (K.TRIP / r2 + K.T_LOOP), 30)
        ax.plot(tt2 * 1e3, r2 * tt2, color=BLUE, lw=3)
        ax.plot([tt2[-1] * 1e3], [fo2], "o", color=BLUE, ms=9, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{val('v_latch_high')} m/s: {val('f_cmd_hi')} N at brake,\nstop needs {val('a_arm_hi')} m/s²", (tt2[-1] * 1e3, fo2),
                    xytext=(0.3, 132), color=BLUE, fontsize=11.5, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=BLUE))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Time since first contact (ms)", fontsize=11)
    ax.set_ylabel("Force on plate (N)", fontsize=11)
    ax.tick_params(labelsize=10)
    save(fig, f"latch_step{step}")


for s in (1, 2):
    latch(s)

# =============================================================================
# Energy tank (SE 12.6.2, equations 9, 10); drawn for a 0.58-column slot
# =============================================================================
b = []
b.append(box(200, 30, 190, 280, "white", GREEN, rx=4))
b.append(f'<rect x="202" y="130" width="186" height="178" fill="{GREEN_F}"/>')
b.append(f'<line x1="200" y1="255" x2="390" y2="255" stroke="{RED}" stroke-width="3" stroke-dasharray="8,5"/>')
b.append(text(295, 290, "reserve", 25, RED, weight="bold"))
b.append(text(295, 190, "tank", 26, "#047857", weight="bold"))
b.append(arrow(110, 115, 196, 160, GREEN, marker="ahg"))
b += lines(10, 50, ["credit:", "damping"], 25, "#047857", anchor="start", weight="bold")
b.append(arrow(394, 160, 480, 115, AMBER))
b += lines(440, 50, ["debit:", "active work"], 25, AMBER, anchor="start", weight="bold")
b += lines(420, 205, ["at the reserve,", "no active work"], 24, RED, anchor="start", weight="bold")
b.append(box(140, 340, 400, 62, SLATE_F, SLATE))
b.append(text(340, 380, "force: its own sensor", 25))
svg("tank", 640, 410, b)

# =============================================================================
# Hand force (SE Napkin Math 12.3); drawn for a 2.6 cm by 9.5 cm slot
# =============================================================================
fig, ax = plt.subplots(figsize=(3.6, 1.3))
for i, (lab, w, key, col) in enumerate([("Clamped", K.F_CLAMP, "f_clamp", RED),
                                        ("Free", K.F_FREE, "f_free", GREEN)]):
    ax.barh(i, w, color=col, height=0.62)
    ax.text(w + 1.5, i, f"{val(key)} N", va="center", fontsize=12, color=col, fontweight="bold")
    ax.text(-1.5, i, lab, ha="right", va="center", fontsize=12, fontweight="bold")
ax.axvline(K.F_CRIT, color=RED, ls="--", lw=2)
ax.text(K.F_CRIT + 1.5, 1.5, f"{val('f_crit')} N criterion", color=RED, fontsize=12, fontweight="bold")
ax.set_xlim(0, 125); ax.set_ylim(-0.5, 1.95)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(f"Peak force at {val('v_handover')} m/s (N)", fontsize=12)
ax.tick_params(labelsize=11)
save(fig, "hand_force")
# =============================================================================
# Measured clamped-hand forces (Kirschner et al. 2022, Table II, hand, B1)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.4, 2.7))
names = list(K.KIRSCHNER)
short = {"UR10e": "UR10e", "Franka (FE)": "Franka", "KUKA LBR iiwa 14": "KUKA iiwa", "Techman TM5": "TM5"}
xs = np.arange(len(names))
cvals = [K.KIRSCHNER[n][1] for n in names]
nvals = [K.KIRSCHNER[n][2] for n in names]
ax.bar(xs - 0.2, cvals, width=0.38, color=NAVY)
ax.bar(xs + 0.2, nvals, width=0.38, color="#93C5FD")
for x, c, n in zip(xs, cvals, nvals):
    ax.text(x - 0.2, c + 6, f"{c:.0f}", ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(x + 0.2, n + 6, f"{n:.0f}", ha="center", fontsize=11, color="#1D4ED8", fontweight="bold")
ax.axhline(K.F_TS, color=RED, ls="--", lw=2)
ax.text(3.42, K.F_TS + 12, f"set for\n{val('f_ts')} N", color=RED, ha="left", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"{short[n]}\n{K.KIRSCHNER[n][0]:.2f} m/s" for n in names], fontsize=10.5)
ax.set_xlim(-0.5, 4.4)
ax.set_ylim(0, 480)
ax.set_ylabel("Measured force (N)", fontsize=11)
ax.tick_params(axis="y", labelsize=10)
save(fig, "kirschner")

# =============================================================================
# Protective separation distance (SE 12.8.3)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.9, 1.8))
parts = [("person walks", K.S_H, RED), ("robot reacts", K.S_R, AMBER), ("robot stops", K.S_STOP, SLATE)]
x = 0
for lab, w, col in parts:
    ax.barh(0, w, left=x, color=col, height=0.6, edgecolor="white")
    if w > 200:
        ax.text(x + w / 2, 0, f"{lab}\n{w:.0f} mm", ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold")
    else:
        ax.text(x + w / 2, 0.42 if lab.endswith("reacts") else -0.42, f"{lab} {w:.0f} mm", ha="center",
                va="bottom" if lab.endswith("reacts") else "top", fontsize=10.5, color=col, fontweight="bold")
    x += w
ax.text(x + 10, 0, f"{x:.0f} mm", va="center", fontsize=12, fontweight="bold", color=TXT)
ax.set_xlim(0, 800); ax.set_ylim(-0.9, 0.9); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Separation the arm must keep, before sensor and position margins (mm)", fontsize=10.5)
ax.tick_params(labelsize=10)
save(fig, "ssm")

# =============================================================================
# Five premises and their detectors (SE 12.6.3); drawn for a 5 cm slot
# =============================================================================
b = []
rows = [("Model fidelity", "observer: predicted vs measured"),
        ("Bounded push", "observer against the force bound"),
        ("Torque available", "current shunts, saturation flag"),
        ("Solver on time", "watchdog and cycle counter"),
        ("Short tick", f"bus timestamps; {val('hold_drift')} mm drift")]
b.append(text(155, 30, "Premise", 25, "#047857", weight="bold"))
b.append(text(615, 30, "Detector, every tick", 25, RED, weight="bold"))
for i, (p, d) in enumerate(rows):
    y = 48 + i * 72
    b.append(box(10, y, 290, 58, GREEN_F, GREEN))
    b.append(text(155, y + 38, p, 24, weight="bold"))
    b.append(arrow(302, y + 29, 338, y + 29))
    b.append(box(340, y, 550, 58, "white", "#94A3B8"))
    b.append(text(615, y + 38, d, 24))
svg("premises", 900, 415, b)

# =============================================================================
# Cycle budget (SE 12.6.4)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.9, 1.75))
x = 0
cols = [SLATE, "#64748B", "#94A3B8", AMBER, "#64748B", "#CBD5E1"]
for (lab, w), col in zip(K.STAGES, cols):
    ax.barh(0, w, left=x, color=col, height=0.55, edgecolor="white")
    if w >= 100:
        ax.text(x + w / 2, 0, f"{lab} {w:.0f}", ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")
    x += w
ax.text(0, 0.36, f"state in {val('cyc_state')}, lease {val('cyc_lease')}, tracking {val('cyc_tracking')}",
        fontsize=10, color=SLATE, va="bottom")
ax.text(205, 0.36, f"output {val('cyc_output')}, reserve {val('cyc_reserve')}", fontsize=10, color=SLATE,
        va="bottom")
ax.axvline(K.DEADLINE, color=RED, ls="--", lw=2)
ax.text(K.DEADLINE - 4, -0.42, f"deadline {val('deadline')} µs", color=RED, ha="right", fontsize=11,
        fontweight="bold", va="center")
ax.plot([K.UNLOADED], [-0.42], "^", color=NAVY, ms=8)
ax.text(K.UNLOADED + 7, -0.42, f"unloaded run {val('unloaded')}", color=NAVY, fontsize=10, va="center")
ax.text(K.WCET + 6, 0, f"declared WCET {val('wcet')}", fontsize=11, fontweight="bold", va="center")
ax.set_xlim(0, 420); ax.set_ylim(-0.65, 0.65)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(f"Microseconds within one {val('t_tick')} ms tick", fontsize=11)
ax.tick_params(labelsize=10)
save(fig, "cycle")

# =============================================================================
# The fallback ladder (SE 12.8.1, table 6, figure 10); drawn for a 0.70-column slot
# =============================================================================
RUNGS = [("PROJECT", ["fresh proposal,", "correction exists"], "least change", GREEN, GREEN_F),
         ("HOLD", ["lease lost,", "standing still"], "powered hold", GREEN, GREEN_F),
         ("STOP", ["lease lost or", "breach, moving"], "resident stop", AMBER, "#FEF3C7"),
         ("INHIBIT", ["MCU fails", "(watchdog)"], "STO + brakes", RED, RED_F)]


def ladder(step):
    b = []
    b.append(text(155, 28, "Failed check, state", 23, TXT, weight="bold"))
    b.append(text(640, 28, "Response", 23, TXT, weight="bold"))
    for i, (name, cond, resp, col, fill) in enumerate(RUNGS):
        if i >= step:
            continue
        y = 44 + i * 90
        b.append(box(10, y, 290, 74, "white", "#94A3B8"))
        b += lines(155, y + 31, cond, 22, gap=26)
        mk = "ahr" if col == RED else ("ahg" if col == GREEN else "ah")
        b.append(arrow(302, y + 37, 346, y + 37, col, marker=mk))
        b.append(box(348, y, 170, 74, fill, col))
        b.append(text(433, y + 46, name, 25, col if col != GREEN else "#047857", weight="bold"))
        b.append(text(535, y + 46, resp, 23, anchor="start"))
    if step >= 4:
        b.append(f'<line x1="5" y1="{44 + 3 * 90 - 8}" x2="795" y2="{44 + 3 * 90 - 8}" stroke="{RED}" stroke-width="2" stroke-dasharray="6,4"/>')
        b.append(text(795, 44 + 3 * 90 + 96, "below the line: hardware, not firmware", 21, RED, anchor="end"))
    svg(f"ladder_step{step}", 800, 420, b)


for s in (1, 2, 3, 4):
    ladder(s)

# =============================================================================
# STOP against STO from the aisle speed (SE 12.8.1, 12.8.2)
# =============================================================================
fig, ax = plt.subplots(figsize=(5.9, 2.3))
T = K.stop_time(K.V_AISLE)
tj = K.A_BRAKE / K.J
tt = np.linspace(0, T, 300)
vel = []
for ti in tt:
    if ti < tj:
        vel.append(K.V_AISLE - 0.5 * K.J * ti ** 2)
    elif ti < T - tj:
        vel.append(K.V_AISLE - 0.5 * K.A_BRAKE * tj - K.A_BRAKE * (ti - tj))
    else:
        vel.append(0.5 * K.J * (T - ti) ** 2)
ax.plot(tt * 1e3, vel, color=GREEN, lw=3.5)
ax.text(560, 0.95, f"STOP: {val('t_stop_aisle')} ms, {val('d_stop_aisle')} mm",
        color="#047857", fontsize=12, fontweight="bold")
ax.plot([0, 1200], [K.V_AISLE, K.V_AISLE], color=RED, lw=2.5, ls="--")
ax.text(1190, 1.38, "STO alone: no braking torque, the base coasts", color=RED,
        fontsize=11.5, fontweight="bold", ha="right")
ax.axvspan(*K.BRAKE_SET, color=AMBER, alpha=0.25)
ax.text(K.BRAKE_SET[1] + 8, 0.1, f"spring brakes set\n{val('brake_set_min')} to {val('brake_set_max')} ms after STO",
        fontsize=11.5, color="#92400E")
ax.set_xlim(0, 1200); ax.set_ylim(0, 1.6)
ax.set_xlabel("Time after the fault (ms)", fontsize=11)
ax.set_ylabel("Base speed (m/s)", fontsize=11)
ax.tick_params(labelsize=10)
save(fig, "stop_sto")

# =============================================================================
# Cruise, October 2, 2023 (SE War Story 12.2); vertical, for a column slot
# =============================================================================
b = []
ev = [("Another car strikes a pedestrian", SLATE),
      ("Robotaxi hits her; emergency stop", SLATE),
      ("Classifier: side collision", AMBER),
      ("Pullover; lane read as clear", RED),
      (f"Dragged {val('cruise_drag_m')} m, up to {val('cruise_ms')} m/s", RED)]
b.append(f'<line x1="30" y1="30" x2="30" y2="{30 + 4 * 66}" stroke="{SLATE}" stroke-width="3"/>')
for i, (lab, col) in enumerate(ev):
    y = 30 + i * 66
    b.append(f'<circle cx="30" cy="{y}" r="13" fill="{col}"/>')
    b.append(text(60, y + 9, lab, 26, col if col != SLATE else TXT, anchor="start",
                  weight="bold" if col == RED else "normal"))
svg("cruise", 520, 320, b)


# =============================================================================
# Synthesis: Therac-25 against the running machine; drawn for a 0.55-column slot
# =============================================================================
b = []
b.append(text(120, 32, "Therac-25", 27, weight="bold"))
b.append(box(20, 60, 200, 70, GREEN_F, GREEN)); b.append(text(120, 104, "software", 25))
b.append(arrow(120, 130, 120, 250))
b.append(text(132, 198, "nothing", 24, RED, anchor="start", weight="bold"))
b.append(text(132, 224, "else", 24, RED, anchor="start", weight="bold"))
b.append(box(20, 252, 200, 66, SLATE_F, SLATE, rx=0)); b.append(text(120, 294, "beam", 25))
b.append(f'<line x1="260" y1="15" x2="260" y2="325" stroke="#CBD5E1" stroke-width="2"/>')
b.append(text(450, 32, "Running machine", 27, weight="bold"))
b.append(box(285, 60, 170, 70, GREEN_F, GREEN)); b.append(text(370, 104, "enforcer", 25))
b.append(box(470, 60, 175, 70, RED_F, RED)); b.append(text(557, 104, "floor", 25, "#991B1B"))
b.append(arrow(370, 130, 430, 250, GREEN, marker="ahg"))
b.append(arrow(557, 130, 500, 250, RED, marker="ahr"))
b.append(box(365, 252, 200, 66, SLATE_F, SLATE, rx=0)); b.append(text(465, 294, "drives", 25))
svg("synthesis", 660, 330, b)

print("\nfigures written")
