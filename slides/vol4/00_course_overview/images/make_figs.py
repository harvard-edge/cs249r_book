"""Figures for Volume IV, Lecture 0 (Course overview).

Run:  python3 make_figs.py          (the figures this deck shows)
      python3 make_figs.py --kit    (also the two kit candidates below)
Writes SVGs here; `make deck V=4 D=00_course_overview` converts them to PDF.
Data plots read the MLSysIM registry; derived values come from calcs.py.

Kit candidates: the stopping-budget bar and the record chain are kept here
until they move to slides/assets/kit (course plan, "The stack figure"). Lecture
0 no longer shows them; Lecture 1 owns both ideas.
"""

import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))   # slides/assets/kit
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

# Volume IV roles (rules/volumes/vol4/palette.md)
NAVY = "#1F407A"
NERVE = "#059669"
RED = "#DC2626"
PLANT = "#334155"
AMBER = "#D97706"
GRAY = "#94A3B8"
TEXT = "#1A1A2E"


def out(name):
    return str(HERE / name)


def _year(ym):
    y, m = (int(s) for s in ym.split("-"))
    d = date(y, m, 28)
    return y + (d.timetuple().tm_yday - 1) / 365


# ---------------------------------------------------------------------------
# Hook: Waymo cumulative rider-only miles (Waymo Safety Impact Data Hub,
# MLSysIM Literature.Series.WaymoRiderOnlyMiles). Step 2 marks the last year.
# ---------------------------------------------------------------------------
def waymo_miles(step):
    rows = calcs.waymo_rows()
    xs = [_year(d) for d, _ in rows]
    ys = [m for _, m in rows]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, ys, "o-", color=sp.C["data"], lw=2.5, ms=6, zorder=3)
    ax.text(xs[0], ys[0] + 22, f"{ys[0]:.1f} million\nJune 2024", color=sp.C["data"],
            fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax.text(xs[-1] - 0.06, ys[-1], f"{ys[-1]:.1f} million\nJune 2026", color=sp.C["data"],
            fontsize=12, fontweight="bold", ha="right", va="center")
    if step >= 2:
        d = dict(rows)
        x0, x1 = _year("2025-06"), _year("2026-06")
        ax.plot([x0, x0], [0, d["2025-06"]], color=AMBER, lw=1.5, ls="--")
        ax.plot([x1, x1], [0, d["2026-06"]], color=AMBER, lw=1.5, ls="--")
        ax.annotate("", xy=(x1, 30), xytext=(x0, 30),
                    arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.8))
        ax.text((x0 + x1) / 2, 40, f"{calcs.waymo_last_year():.0f} million miles\nin 12 months",
                color=AMBER, fontsize=12, fontweight="bold", ha="center", va="bottom")
    ax.set_xlim(2024.25, 2026.6)
    ax.set_ylim(0, 300)
    ax.set_xticks([2024.5, 2025, 2025.5, 2026, 2026.5])
    ax.set_xticklabels(["mid 2024", "2025", "mid 2025", "2026", "mid 2026"])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("Driverless miles,\ncumulative (millions)")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out(f"waymo_miles_step{step}.svg"))


# ---------------------------------------------------------------------------
# The stopping-budget bar, two segments (Napkin Math 1.1)
# ---------------------------------------------------------------------------
def stop_bar(step):
    d1, d2 = calcs.stop_mm(calcs.V_DRIVE)
    clear = calcs.D_CLEAR * 1000
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    ax.barh(0, d1, color=AMBER, height=0.5)
    ax.text(d1 / 2, 0, f"{d1:.1f}", color="white", fontsize=13, ha="center", va="center",
            fontweight="bold")
    ax.text(d1 / 2, 0.33, "delay  v·τ", color=AMBER, fontsize=12, ha="center", va="bottom",
            fontweight="bold")
    a2 = 1 if step >= 2 else 0
    ax.barh(0, d2, left=d1, color=PLANT, height=0.5, alpha=a2)
    ax.text(d1 + d2 / 2, 0, f"{d2:.1f}", color="white", fontsize=13, ha="center", va="center",
            fontweight="bold", alpha=a2)
    ax.text(d1 + d2 / 2, 0.33, "braking  v²/2a", color=PLANT, fontsize=12, ha="center",
            va="bottom", fontweight="bold", alpha=a2)
    a3 = 1 if step >= 3 else 0
    ax.axvline(clear, color=RED, lw=2, ls="--", alpha=a3)
    ax.text(clear - 12, 0.33, "clear 1,100", color=RED, fontsize=12, ha="right", va="bottom",
            fontweight="bold", alpha=a3)
    ax.text((d1 + d2 + clear) / 2, 0, f"spare\n{clear - d1 - d2:.1f}", color=RED, fontsize=11,
            ha="center", va="center", fontweight="bold", alpha=a3)
    ax.set_xlim(0, 1250)
    ax.set_ylim(-0.4, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mm traveled from detection, 1.5 m/s")
    sp.save(fig, out(f"stop_bar_step{step}.svg"))


# ---------------------------------------------------------------------------
# The record chain (sec. 1.11, figure 9): one record per lecture
# ---------------------------------------------------------------------------
def record_chain():
    rows = [
        ("Part I", "irreversibility", PLANT, [(2, "limit"), (4, "handoff")]),
        ("Part II", "endogenous data", sp.C["data"], [(5, "provenance"), (6, "policy manifest"),
                                                      (7, "evaluation")]),
        ("Part III", "proposal is not", NERVE, [(8, "observation contract"), (9, "belief"),
                                                (10, "intent lease")]),
        ("", "permission", NERVE, [(11, "trajectory"), (12, "enforcement"),
                                    (13, "placement")]),
        ("Part IV", "permission must", NAVY, [(14, "authority"), (15, "fault"),
                                              (16, "release manifest")]),
        ("", "be earned", NAVY, [(17, "approval")]),
        ("Close", "", RED, [(18, "residual-claims register")]),
    ]
    fig = plt.figure(figsize=(5.8, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(-2.55, 0.2)
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(t):
        bb = t.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    y = 0
    for part, law, c, recs in rows:
        ax.text(0.05, y, part, color=c, fontsize=11, fontweight="bold", va="center")
        ax.text(0.78, y, law, color=c, fontsize=10, va="center")
        x = 2.0
        for lec, name in recs:
            t1 = ax.text(x + 0.06, y, f"L{lec}", color=c, fontsize=10, va="center", fontweight="bold")
            w1 = width(t1)
            t2 = ax.text(x + 0.06 + w1 + 0.06, y, name, color=TEXT, fontsize=10, va="center")
            w = w1 + width(t2) + 0.18
            ax.add_patch(FancyBboxPatch((x, y - 0.14), w, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc="white", ec=c, lw=1.5, clip_on=False))
            x += w + 0.07
        y -= 0.38
    sp.save(fig, out("record_chain.svg"))

def main():
    for k in (1, 2):
        waymo_miles(k)
    if "--kit" in sys.argv:
        for k in (1, 2, 3):
            stop_bar(k)
        record_chain()


if __name__ == "__main__":
    main()
