#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 17 (Responsible Physical AI), Volume IV.

Every derived value comes from calcs.py (inputs from the rendered chapter,
Responsible Physical AI, the MLSysIM registry the chapter prints from, and
two published per-group detection tables, each named beside its data). Each
figure is drawn at the size of its slot (full width about 5.8 in wide; a
column about 5.4 by 3.6 in with 15 pt labels) so its type stays legible on
the slide. A figure revealed in steps is written as name_step1.svg ...
name_stepN.svg with every element drawn in every step (hidden ones
transparent), so the canvas and the axes never move between steps.

Colors follow the Volume IV palette (volumes/vol4/palette.md): green for the
deterministic permission side and its records, slate for people, parties,
and the plant, red only for hazards, missing records, and harm, amber for
the approval record (the cover's approval seal), navy for learned proposers.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402,F401  (sets the shared rcParams)

import calcs  # noqa: E402

plt = sp.plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

V = calcs.VALUES
OUT = Path(__file__).parent

P = {
    "navy": "#1F407A", "navy_l": "#EFF6FF",
    "green": "#059669", "green_l": "#DCFCE7", "green_d": "#047857",
    "red": "#DC2626", "red_l": "#FEE2E2", "red_d": "#991B1B",
    "slate": "#334155", "slate_l": "#F1F5F9",
    "amber": "#D97706", "amber_l": "#FEF3C7", "amber_d": "#92400E",
    "gray": "#64748B", "gray_l": "#E2E8F0",
    "petrol": "#0D9488",
    "text": "#1A1A2E",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white")
    plt.close(fig)


def a(shown):
    return 1.0 if shown else 0.0


def box(ax, x, y, w, h, text, fc, ec, shown=True, fs=13, tc=None, bold=True,
        rad=0.08, lw=1.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, ls=ls, alpha=a(shown)))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or P["text"], fontweight="bold" if bold else "normal",
            alpha=a(shown))


def arrow(ax, x0, y0, x1, y1, color, shown=True, lw=2.0, ls="-", ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 alpha=a(shown)))


def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Hook: the Cruise collision, the motion and the record (ch Sec. 17.1, 17.4.2)
# ---------------------------------------------------------------------------
def cruise_timeline():
    c = V["cruise"]
    for step in (1, 2):
        fig, ax = canvas(6.0, 2.5, (0, 30), (0, 10))
        s2 = step >= 2
        # Row 1: the motion (2 October 2023)
        ax.text(0.2, 8.6, "The motion", fontsize=12.5, fontweight="bold", color=P["red_d"], va="center")
        box(ax, 7.0, 7.3, 22.6, 2.6,
            f"2 Oct 2023: ran over a pedestrian thrown into its path,\nthen pulled over {c['drag_m']:.1f} m with her beneath it",
            P["red_l"], P["red"], fs=11.5, bold=False)
        # Row 2: the record (3 to 24 October)
        ax.text(0.2, 5.0, "The record", fontsize=12.5, fontweight="bold", color=P["slate"], va="center")
        ax.plot([7.0, 29.6], [5.0, 5.0], color=P["gray"], lw=2)
        events = [(8.0, "3 Oct", "regulator told\nit stopped"),
                  (14.3, "3 to 18 Oct", "15 days without\na full account"),
                  (20.4, "19 Oct", "full video\nto the CPUC"),
                  (26.6, "24 Oct", "DMV suspends\ndriverless permits")]
        for x, d, t in events:
            ax.plot([x], [5.0], "o", color=P["slate"], ms=8)
            ax.text(x, 5.75, d, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=P["slate"])
            ax.text(x, 4.3, t, ha="center", va="top", fontsize=11, color=P["text"], linespacing=1.05)
        # Row 3 (step 2): penalties for the record, 2024
        ax.text(0.2, 0.9, "2024", fontsize=12.5, fontweight="bold", color=P["amber_d"], va="center", alpha=a(s2))
        pens = [(f"CPUC  ${c['cpuc']:,}", 7.0), (f"NHTSA  ${c['nhtsa']:,}", 14.6),
                (f"DOJ fine  ${c['doj']:,}", 22.2)]
        for t, x in pens:
            box(ax, x, 0.05, 7.3, 1.7, t, P["amber_l"], P["amber"], fs=11.5, shown=s2)
        save(fig, f"cruise_timeline_step{step}")


# ---------------------------------------------------------------------------
# Decisions behind one admitted command (ch Sec. 17.2, figure 1)
# ---------------------------------------------------------------------------
DECISIONS = [
    ("Policy weights", "learning team", "policy manifest"),
    ("Enforcer thresholds", "safety engineering", "enforcement record"),
    ("Authority state", "assistant on shift", "authority record"),
    ("Operating envelope", "systems adjudicator", "release manifest"),
    ("Crossing rule", "facility management", "no record"),
    ("Staffing ratio", "operations", "no record"),
]


def decision_chain():
    for step in (1, 2, 3):
        fig, ax = canvas(6.4, 3.0, (0, 70), (-4.5, 29.5))
        box(ax, 0.2, 9.0, 9.6, 10.0, "Admitted\ncommand\n1.3 m/s,\none tick",
            P["green_l"], P["green"], fs=11.5)
        ax.text(21.8, 27.4, "Decision", ha="center", fontsize=11.5, color=P["gray"], fontweight="bold")
        ax.text(41.0, 27.4, "Made by", ha="center", fontsize=11.5, color=P["gray"], fontweight="bold")
        ax.text(58.5, 27.4, "Held in", ha="center", fontsize=11.5, color=P["gray"], fontweight="bold")
        for i, (dec, who, rec) in enumerate(DECISIONS):
            y = 22.6 - i * 4.4
            missing = rec == "no record"
            arrow(ax, 9.8, 14.0, 12.4, y + 1.7, P["slate"], lw=1.3, ms=9)
            box(ax, 12.4, y, 18.8, 3.4, dec, P["slate_l"], P["slate"], fs=10.5)
            ax.text(41.0, y + 1.7, who, ha="center", va="center", fontsize=11, color=P["text"])
            red = missing and step >= 2
            col = P["red"] if red else (P["green_d"] if not missing else P["text"])
            ax.text(58.5, y + 1.7, rec, ha="center", va="center", fontsize=11, color=col,
                    fontweight="bold" if red else "normal")
        s3 = step >= 3
        ax.add_patch(Rectangle((31.8, -0.5), 37.9, 27.0, fc="none", ec=P["amber"], lw=2.2, ls="--",
                               alpha=a(s3)))
        ax.text(50.7, -3.0, "Approval record: each decision has an owner",
                ha="center", va="center", fontsize=11.5, color=P["amber_d"], fontweight="bold", alpha=a(s3))
        save(fig, f"decision_chain_step{step}")


# ---------------------------------------------------------------------------
# A chain is weaker than any link (ch Sec. 17.2.1)
# ---------------------------------------------------------------------------
def chain_prob():
    p = np.linspace(0.85, 1.0, 200)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.2, top=0.95)
        ax.plot(p, p, color=P["gray"], lw=2.2)
        ax.text(0.862, 0.905, "one link", fontsize=15, color=P["gray"])
        s2 = step >= 2
        ax.plot(p, p**6, color=P["red"], lw=3, alpha=a(s2))
        ax.text(0.952, 0.48, "six links", fontsize=15, color=P["red"], fontweight="bold", alpha=a(s2))
        for pk, ck in V["chain"].items():
            ax.plot([pk], [ck], "o", color=P["red"], ms=9, alpha=a(s2))
            if pk > 0.98:
                ax.annotate(f"{ck:.2f}", xy=(pk, ck), xytext=(0.995, 0.79), ha="right", fontsize=15,
                            color=P["red_d"], fontweight="bold", alpha=a(s2),
                            arrowprops=dict(arrowstyle="-", color=P["red"], lw=1.0, alpha=a(s2)))
            else:
                ax.text(pk - 0.004, ck + 0.035, f"{ck:.2f}", ha="right", fontsize=15, color=P["red_d"],
                        fontweight="bold", alpha=a(s2))
        ax.set_xlim(0.85, 1.0)
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks([0.85, 0.90, 0.95, 1.0])
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Each link retrievable", fontsize=15)
        ax.set_ylabel("Whole chain found", fontsize=15)
        save(fig, f"chain_prob_step{step}")


# ---------------------------------------------------------------------------
# Time to act at the rack end (ch Sec. 17.3.1)
# ---------------------------------------------------------------------------
def time_test():
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.3, top=0.95)
    rows = [("Time the hazard leaves", V["t_window"], P["green"], f"{V['t_window']:.2f} s (1.10 m at 1.3 m/s)"),
            ("Out-of-loop takeover", V["t_ool"], P["red"], f"{V['t_ool']:.1f} s: travels {V['travel']:.1f} m")]
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.55)
        ax.text(-0.08, y, lab, ha="right", va="center", fontsize=12.5, fontweight="bold", color=P["text"])
        ax.text(v + 0.08, y, txt, ha="left", va="center", fontsize=12.5, color=col, fontweight="bold")
    ax.set_xlim(0, 8.2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Seconds", fontsize=12)
    ax.tick_params(labelsize=11.5)
    save(fig, "time_test")


# ---------------------------------------------------------------------------
# One assistant, many machines (Napkin Math 17.1)
# ---------------------------------------------------------------------------
def queue():
    n = np.arange(1, 35)
    lam = n * calcs.RATE_PER_MACHINE_H / 3600.0
    s = calcs.T_SERVICE_S
    rho = lam * s
    wq = rho * s / (1 - rho) / 60.0
    Q = V["queue"]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.2, top=0.95)
        ax.plot(n, wq, color=P["slate"], lw=3)
        pts = [10, 20] + ([32] if step >= 2 else [])
        for k in (10, 20, 32):
            r, w, _ = Q[k]
            sh = k in pts
            ax.plot([k], [w / 60], "o", color=P["red"], ms=9, alpha=a(sh))
            tx, ty = {10: (3.0, 4.0), 20: (12.0, 7.4), 32: (17.5, 12.2)}[k]
            ax.annotate(f"{k} machines\n{r:.2f} busy, {w / 60:.0f} min", xy=(k, w / 60), xytext=(tx, ty),
                        ha="left", va="bottom", fontsize=14, color=P["red_d"], alpha=a(sh), linespacing=1.0,
                        arrowprops=dict(arrowstyle="-", color=P["red"], lw=1.0, alpha=a(sh)))
        ax.set_xlim(0, 34)
        ax.set_ylim(0, 15)
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Machines per assistant", fontsize=15)
        ax.set_ylabel("Mean wait (min)", fontsize=15)
        save(fig, f"queue_step{step}")


# ---------------------------------------------------------------------------
# Permission in stages (ch Sec. 17.4)
# ---------------------------------------------------------------------------
def ladder():
    road = ["Testing with a\nsafety driver", "Driverless\ntesting", "Deployment to\nthe public"]
    floor = ["Commissioning\nwith a spotter", "Unattended in a\nzone people avoid", "Open floor\namong workers"]
    for step in (1, 2):
        fig, ax = canvas(6.0, 2.7, (0, 60), (0, 28))
        s2 = step >= 2
        ax.text(0.3, 21.5, "California\nroads", fontsize=12, fontweight="bold", color=P["slate"], va="center")
        ax.text(0.3, 7.5, "A warehouse", fontsize=12, fontweight="bold", color=P["slate"], va="center",
                alpha=a(s2))
        for i in range(3):
            x = 11.5 + i * 16.0
            box(ax, x, 17.5 + i * 1.6, 13.0, 6.2, road[i], P["green_l"], P["green"], fs=11.5)
            box(ax, x, 3.5 + i * 1.6, 13.0, 6.2, floor[i], P["green_l"], P["green"], fs=11.5, shown=s2)
            if i < 2:
                arrow(ax, x + 13.0, 20.6 + i * 1.6, x + 16.0, 22.2 + i * 1.6, P["slate"], lw=1.6)
                arrow(ax, x + 13.0, 6.6 + i * 1.6, x + 16.0, 8.2 + i * 1.6, P["slate"], lw=1.6, shown=s2)
        ax.text(11.5, 14.8, "Each rung is its own DMV permit; the CPUC adds passenger rungs",
                fontsize=11, color=P["gray"], va="center")
        ax.text(11.5, 0.9, "Granted by the owner, the integrator, and workers' representatives",
                fontsize=11, color=P["gray"], va="center", alpha=a(s2))
        save(fig, f"ladder_step{step}")


# ---------------------------------------------------------------------------
# Leaving a stage on evidence (ch Sec. 17.4.1)
# ---------------------------------------------------------------------------
def stage_hours():
    days = np.linspace(0, 24, 300)
    hrs = days * V["fleet_h_day"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
    ax.plot(days, hrs, color=P["green"], lw=3)
    ax.axhline(V["t_req"], color=P["red"], lw=1.8, ls="--")
    ax.text(0.5, V["t_req"] + 90, f"{V['t_req']:,.0f} failure-free hours", fontsize=15, color=P["red_d"])
    ax.plot([V["days_req"]], [V["t_req"]], "o", color=P["red"], ms=9)
    ax.text(V["days_req"] + 0.6, V["t_req"] - 380, f"day {V['days_req']:.0f}", fontsize=15,
            color=P["red_d"], fontweight="bold")
    ax.text(11, 600, "10 machines,\n16 h a day", fontsize=15, color=P["green_d"], ha="left")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3800)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Days at the supervised stage", fontsize=15)
    ax.set_ylabel("Fleet hours", fontsize=15)
    save(fig, "stage_hours")


# ---------------------------------------------------------------------------
# Penalties for the record (ch Sec. 17.4.2)
# ---------------------------------------------------------------------------
def fines():
    c = V["cruise"]
    rows = [("CPUC", c["cpuc"], "15 days x $7,500"), ("NHTSA", c["nhtsa"], "reports omitted the drag"),
            ("DOJ", c["doj"], "false report, criminal fine")]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.22, top=0.95)
    for i, (lab, v, why) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v / 1e6, color=P["amber"], height=0.6)
        ax.text(-0.03, y, lab, ha="right", va="center", fontsize=15, fontweight="bold",
                transform=ax.get_yaxis_transform())
        ax.text(v / 1e6 + 0.04, y + 0.05, f"${v:,}", ha="left", va="bottom", fontsize=14,
                fontweight="bold", color=P["amber_d"])
        ax.text(v / 1e6 + 0.04, y - 0.05, why, ha="left", va="top", fontsize=13, color=P["text"])
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Penalty (million USD)", fontsize=15)
    save(fig, "fines")


# ---------------------------------------------------------------------------
# Published per-group detection: Li et al. 2025 Table 7; Wilson et al. 2019 Table 4
# ---------------------------------------------------------------------------
def detect_data():
    fig = plt.figure(figsize=(6.0, 2.8))
    ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.68])
    ax2 = fig.add_axes([0.69, 0.2, 0.3, 0.68])
    li = V["li"]
    names = list(li)
    for i, k in enumerate(names):
        y = len(names) - 1 - i
        ad, ch = li[k]
        ax1.plot([ad, ch], [y, y], color=P["gray_l"], lw=3, zorder=1)
        ax1.plot([ad], [y], "o", color=P["slate"], ms=6.5, zorder=2)
        ax1.plot([ch], [y], "o", color=P["red"], ms=6.5, zorder=2)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names[::-1], fontsize=10.5)
    ax1.set_xlim(0, 70)
    ax1.set_xlabel("Miss rate (%)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=10.5)
    ax1.text(3, 8.15, "adults", color=P["slate"], fontsize=11, fontweight="bold")
    ax1.text(38, 8.15, "children", color=P["red"], fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.6, 8.9)
    ax1.set_title("Li et al. 2025, Table 7", fontsize=11.5, loc="left", x=-0.55, pad=14)
    w = V["wilson"]
    wn = list(w)
    short = [s.replace("Faster R-CNN ", "Faster ").replace("Mask R-CNN ", "Mask ") for s in wn]
    for i, k in enumerate(wn):
        y = len(wn) - 1 - i
        ls_, ds_ = w[k]
        ax2.plot([ds_, ls_], [y, y], color=P["gray_l"], lw=3, zorder=1)
        ax2.plot([ls_], [y], "o", color=P["slate"], ms=6.5, zorder=2)
        ax2.plot([ds_], [y], "o", color=P["red"], ms=6.5, zorder=2)
    ax2.set_yticks(range(len(wn)))
    ax2.set_yticklabels(short[::-1], fontsize=10.5)
    ax2.set_xlim(45, 75)
    ax2.set_xlabel("AP75 (%)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=10.5)
    ax2.text(46.2, 8.15, "darker", color=P["red"], fontsize=11, fontweight="bold")
    ax2.text(62, 8.15, "lighter skin", color=P["slate"], fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.6, 8.9)
    ax2.set_title("Wilson et al. 2019, Table 4", fontsize=11.5, loc="left", x=-0.62, pad=14)
    save(fig, "detect_data")


# ---------------------------------------------------------------------------
# Frames needed (ch equation 3)
# ---------------------------------------------------------------------------
def miss_frames():
    n = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.2, right=0.96, bottom=0.2, top=0.95)
    for p, col, lab, dy in ((0.95, P["slate"], "seen in 95% of frames", 0.0),
                            (0.75, P["red"], "seen in 75%", 0.0)):
        ax.semilogy(n, (1 - p) ** n, "o-", color=col, lw=2.6, ms=7)
    ax.axhline(V["p_miss"], color=P["green"], lw=1.8, ls="--")
    ax.text(3.75, V["p_miss"] * 0.45, "1 in 1,000", fontsize=14, ha="left",
            color=P["green_d"], va="top", fontweight="bold")
    ax.annotate("3 frames", xy=(3, 0.05**3), xytext=(1.5, 1.2e-5), fontsize=15, color=P["slate"],
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=P["slate"]))
    ax.annotate("5 frames", xy=(5, 0.25**5), xytext=(5.15, 0.02), fontsize=15, color=P["red"],
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=P["red"]))
    ax.text(1.1, 3e-7, "95% per frame", fontsize=14, color=P["slate"])
    ax.text(1.4, 0.3, "75% per frame", fontsize=14, color=P["red"])
    ax.set_xlim(0.8, 6.4)
    ax.set_ylim(1e-8, 1.5)
    ax.set_xticks(n)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Consecutive frames in view", fontsize=15)
    ax.set_ylabel("Missed in every frame", fontsize=15)
    save(fig, "miss_frames")


# ---------------------------------------------------------------------------
# One stop, two groups (ch figure 2, Table 1)
# ---------------------------------------------------------------------------
def shortfall():
    v0, acc, tau, tf = calcs.V_AISLE, calcs.A_BRAKE, calcs.TAU_DELAY, calcs.T_FRAME
    x_ref = v0 * (V["n_ref"] * tf + tau)
    x_person = x_ref + v0**2 / (2 * acc)
    rows = V["rows"]

    def prof(xb):
        x = np.linspace(0, x_person, 600)
        v = np.where(x < xb, v0, np.sqrt(np.clip(v0**2 - 2 * acc * (x - xb), 0, None)))
        return x, v

    groups = [(None, P["slate"]), (rows[1], P["amber"]), (rows[2], P["red"])]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.95)
        for k, (row, col) in enumerate(groups):
            sh = k < step
            dr = 0 if row is None else row[3]
            x, v = prof(x_ref + dr)
            ax.plot(x, v, color=col, lw=3, alpha=a(sh))
            if row is not None:
                ax.plot([x_person], [row[4]], "o", color=col, ms=10, alpha=a(sh), zorder=5)
                ax.annotate(f"seen in {row[0]:.0%}: {row[4]:.2f} m/s, {row[5]:.0f} J",
                            xy=(x_person, row[4]), xytext=(0.52, 0.44 if k == 1 else 0.68),
                            ha="right", va="center", fontsize=14, color=col, fontweight="bold", alpha=a(sh),
                            arrowprops=dict(arrowstyle="-", color=col, lw=1.0, alpha=a(sh)))
        ax.text(0.03, 0.06, "seen in 95%:\nstops short", fontsize=14, color=P["slate"], fontweight="bold")
        ax.axvline(x_person, color=P["text"], lw=1.4, ls="--")
        ax.text(x_person - 0.02, 1.45, "person", ha="right", fontsize=14)
        ax.set_xlim(0, x_person * 1.04)
        ax.set_ylim(0, 1.6)
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Distance from first view (m)", fontsize=15)
        ax.set_ylabel("Speed (m/s)", fontsize=15)
        save(fig, f"shortfall_step{step}")


# ---------------------------------------------------------------------------
# Published miss rates in meters and joules (derived from Li et al. Table 7)
# ---------------------------------------------------------------------------
def li_conv():
    conv = V["li_conv"]
    names = list(conv)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.22, top=0.97)
    for i, k in enumerate(names):
        y = len(names) - 1 - i
        na, nc, dr, vi, ei = conv[k]
        ax.barh(y, dr * 1e3, color=P["red"], height=0.62)
        ax.text(dr * 1e3 + 2, y, f"{nc - na} frames later: {vi:.2f} m/s, {ei:.0f} J", va="center",
                fontsize=11, color=P["red_d"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=11)
    ax.set_xlim(0, 260)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("Clearance lost for children (mm), aisle speed 1.3 m/s", fontsize=11.5)
    save(fig, "li_conv")


# ---------------------------------------------------------------------------
# Sample sizes for each group (ch Sec. 17.5.2)
# ---------------------------------------------------------------------------
def intervals():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.subplots_adjust(left=0.3, right=0.96, bottom=0.22, top=0.95)
    rows = [(f"{V['n_large']:,}\npeople", V["ci_l"], P["slate"]), (f"{V['n_small']}\npeople", V["ci_s"], P["red"])]
    for i, (lab, (lo, hi), col) in enumerate(rows):
        y = 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle="butt")
        ax.plot([0.95], [y], "o", color="white", mec=col, mew=2.5, ms=11)
        ax.text(0.9035, y, lab, ha="right", va="center", fontsize=15, fontweight="bold", color=col)
        ax.text(0.95, y - 0.22, f"{lo:.3f} to {hi:.3f}", fontsize=14, color=col, ha="center", va="top")
    ax.axvline(0.92, color=P["amber"], lw=1.8, ls="--")
    ax.text(0.9215, -0.62, "0.92", fontsize=14, color=P["amber_d"])
    ax.set_xlim(0.905, 0.99)
    ax.set_ylim(-0.75, 1.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Detection rate, exact 95% interval", fontsize=15)
    save(fig, "intervals")


# ---------------------------------------------------------------------------
# Who bears the risk (ch Sec. 17.6)
# ---------------------------------------------------------------------------
def exposure():
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.24, top=0.95)
    rows = [("10% who use\nthree crossings", V["exp_ratio"], P["red"]),
            ("Site average", 1.0, P["gray"]),
            ("Everyone else", V["rest_ratio"], P["slate"])]
    for i, (lab, v, col) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(-0.15, y, lab, ha="right", va="center", fontsize=15, fontweight="bold")
        ax.text(v + 0.15, y, f"{v:.0f}x" if v >= 1 else f"{v:.2f}x", va="center", fontsize=15,
                fontweight="bold", color=col)
    ax.set_xlim(0, 9.5)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Exposure per person", fontsize=15)
    save(fig, "exposure")


# ---------------------------------------------------------------------------
# Keeping less (ch Sec. 17.7.2)
# ---------------------------------------------------------------------------
def retention():
    day_s = V["day_h"] * 3600
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.3, top=0.97)
    ax.barh(1, V["day_h"], color=P["red_l"], ec=P["red"], height=0.6)
    ax.text(V["day_h"] / 2, 1, "every frame, all day", ha="center", va="center",
            fontsize=15, color=P["red_d"], fontweight="bold")
    rng = np.random.default_rng(17)
    starts = np.sort(rng.uniform(0, day_s - V["window"], V["events"]))
    for s0 in starts:
        ax.barh(0, V["window"] / 3600 * 6, left=s0 / 3600, color=P["green"], height=0.6)
    ax.text(-0.4, 1, "Continuous", ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(-0.4, 0, "Triggered", ha="right", va="center", fontsize=15, fontweight="bold")
    ax.set_xlim(0, V["day_h"])
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Hour of a 16 h day (marks 6x wide)", fontsize=14)
    save(fig, "retention")


# ---------------------------------------------------------------------------
# Where the labor went (Napkin Math 17.3)
# ---------------------------------------------------------------------------
def labor():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.95)
    parts = [("requests", V["assist_h"], P["red"]), ("watching", V["idle_h"], P["amber"]),
             ("sweeps", V["sweep_h"], P["slate"])]
    left = 0
    for lab, v, col in parts:
        ax.barh(0, v, left=left, color=col, height=0.7)
        if v >= 2:
            ax.text(left + v / 2, 0, f"{lab}\n{v:.0f} h", ha="center",
                    va="center", fontsize=14, color="white", fontweight="bold")
        else:
            ax.text(left + 0.4, -0.6, f"{lab} {v:.1f} h", ha="left", va="center",
                    fontsize=14, color=col, fontweight="bold")
        left += v
    for k in (1, 2):
        ax.plot([8 * k, 8 * k], [-0.8, 0.36], color=P["text"], lw=1.5, ls="--")
        ax.text(8 * k - 0.2, -0.6, f"{k} shift" + ("s" if k > 1 else ""), ha="right", va="center", fontsize=14)
    ax.text(left, 0.5, f"{V['total_h']:.1f} person-hours a day", ha="right", fontsize=15,
            fontweight="bold", color=P["text"])
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Hours per operating day, 20 machines", fontsize=15)
    save(fig, "labor")


# The record chain is the course kit's (assets/kit/vol4/record_chain_l17 and
# thumb_record_chain).


def main():
    cruise_timeline()
    decision_chain()
    chain_prob()
    time_test()
    queue()
    ladder()
    stage_hours()
    fines()
    detect_data()
    miss_frames()
    shortfall()
    li_conv()
    intervals()
    exposure()
    retention()
    labor()


if __name__ == "__main__":
    main()
