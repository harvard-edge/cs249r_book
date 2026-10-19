#!/usr/bin/env python3
"""Figures for Lecture 18 (Multi-Agent Coordination), Volume III.

Every value comes from calcs.py, which this script imports, so figures and
slides agree. "MA" is the rendered chapter, Multi-Agent Coordination. The book
SVGs (coordination topologies, supervisor isolation, OCC worktrees, write
collision, correlated ensemble, barrier odds, USL retrograde) are page-sized
and fall below the legibility floor on a slide, so they are redrawn here at
slide size. Full-width diagrams are 900 px wide with type of 18 px or more;
column plots are drawn at 5.4 x 3.6 in with 14 to 16 pt type. Plots use
slideplot.

Run from this folder: python3 make_figs.py, then from slides/: python3 tools/greek_thumbs.py on screen, structures_step3, merge_step3, floor_step2, cancel_step3, recall_bars, and next_bars (the thumb_*.svg files), then make deck.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume III agentic palette (volumes/vol3/palette.md): fill, stroke, text.
P = {
    "store": ("#f3e8ff", "#7c3aed", "#6d28d9"),   # supervisor, records, runtime
    "tool": ("#fef3c7", "#d97706", "#b45309"),    # tools, workers' actions
    "fault": ("#fee2e2", "#dc2626", "#b91c1c"),   # faults, rejections
    "ok": ("#dcfce7", "#16a34a", "#15803d"),      # accepted, published, settled
    "think": ("#e0f2fe", "#0284c7", "#0369a1"),   # model calls, workers
    "plain": ("#ffffff", "#94a3b8", "#0f172a"),
    "band": ("#f8fafc", "#cbd5e1", "#0f172a"),
    "gray": ("#f1f5f9", "#94a3b8", "#475569"),
}
FONT = "Helvetica, Arial, sans-serif"
VIO, RED, GRN, AMB, SKY, GRY = "#7c3aed", "#dc2626", "#16a34a", "#d97706", "#0284c7", "#64748b"
INK = "#0f172a"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, s, show=True):
        self.items.append(s if show else f'<g opacity="0">{s}</g>')

    def _t(self, x, y, s, size, color, bold=False, anchor="middle", italic=False):
        w = ' font-weight="bold"' if bold else ""
        it = ' font-style="italic"' if italic else ""
        return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}"'
                f' fill="{color}" text-anchor="{anchor}"{w}{it}>{s}</text>')

    def text(self, x, y, s, size=19, color=INK, show=True, **kw):
        self.add(self._t(x, y, s, size, color, **kw), show)

    def box(self, x, y, w, h, title, subs=(), role="plain", show=True, dash=False,
            tsize=20, ssize=18, rx=6, sw=2.4, opacity=1.0):
        fill, stroke, text = P[role]
        d = ' stroke-dasharray="8,5"' if dash else ""
        s = (f'<g opacity="{opacity}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
        n = 1 + len(subs)
        lh = [tsize] + [ssize] * len(subs)
        gap = 6
        total = sum(lh) + gap * (n - 1)
        yy = y + (h - total) / 2 + tsize * 0.8
        cx = x + w / 2
        s += self._t(cx, yy, title, tsize, text, bold=True)
        for sub in subs:
            yy += gap + ssize
            s += self._t(cx, yy, sub, ssize, "#1e293b")
        self.add(s + "</g>", show)

    def arrow(self, pts, color=GRY, show=True, dash=False, width=2.6, both=False):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
        da = ' stroke-dasharray="8,5"' if dash else ""

        def head(p1, p2):
            (x1, y1), (x2, y2) = p1, p2
            a = math.atan2(y2 - y1, x2 - x1)
            L, Wd = 14, 7
            bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
            q1 = (bx + Wd * math.sin(a), by - Wd * math.cos(a))
            q2 = (bx - Wd * math.sin(a), by + Wd * math.cos(a))
            return (f'<polygon points="{x2},{y2} {q1[0]:.1f},{q1[1]:.1f} '
                    f'{q2[0]:.1f},{q2[1]:.1f}" fill="{color}"/>')
        s = f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>'
        s += head(pts[-2], pts[-1])
        if both:
            s += head(pts[1], pts[0])
        self.add(s, show)

    def rect(self, x, y, w, h, fill, stroke="none", show=True, rx=0, sw=2, dash=False,
             opacity=1.0):
        d = ' stroke-dasharray="8,5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>', show)

    def line(self, x1, y1, x2, y2, color=GRY, width=2, dash=False, show=True):
        d = ' stroke-dasharray="6,5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>', show)

    def circle(self, x, y, r, fill, stroke="none", sw=2, show=True):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"/>', show)

    def save(self, name):
        body = "\n".join(self.items)
        (out / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')


def col_axes(size=(5.4, 3.6), fs=15):
    fig, ax = plt.subplots(figsize=size)
    ax.tick_params(labelsize=fs - 2)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


# ============================================================ recall (L17)
def recall_bars():
    r = calcs.recall()
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.barh([1], [r["true_accepts"]], color=C["data"], height=0.6)
    ax.barh([0], [r["false_accepts"]], color=C["danger"], height=0.6)
    ax.set_yticks([0, 1], ["false accepts", "true accepts"], fontsize=14)
    ax.set_xlabel("Rewarded rollouts of 8,192", fontsize=13)
    ax.text(r["false_accepts"] + 15, 0, f"{100 * r['false_share']:.1f}% of rewards",
            va="center", fontsize=14, color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 1000)
    sp.save(fig, out / "recall_bars.svg")


recall_bars()


# ======================================================= split (delegation)
def split(k):
    s = Svg(900, 400)
    # left: one controller, parallel calls inside one trajectory
    s.rect(10, 10, 420, 380, P["band"][0], P["band"][1], rx=8)
    s.text(220, 44, "One controller", 21, INK, bold=True)
    s.box(130, 66, 180, 64, "Controller", ("one context",), "store")
    for i, lab in enumerate(("search", "tests", "sample")):
        x = 30 + i * 135
        s.box(x, 200, 120, 56, lab, (), "tool", tsize=19)
        s.arrow([(220, 130), (x + 60, 198)], GRY)
    s.text(220, 300, "parallel calls inside", 18, "#334155")
    s.text(220, 324, "one trajectory, one grant,", 18, "#334155")
    s.text(220, 348, "one budget", 18, "#334155")
    # right: delegated design
    sh = k >= 2
    s.rect(470, 10, 420, 380, P["band"][0], P["band"][1], rx=8, show=sh)
    s.text(680, 44, "Delegated", 21, INK, bold=True, show=sh)
    s.box(590, 66, 180, 64, "Supervisor", ("own context",), "store", show=sh)
    for i, (lab, sub) in enumerate((("interface", "worker"), ("caller", "worker"))):
        x = 500 + i * 200
        s.box(x, 196, 160, 76, lab, (sub,), "think", tsize=19, show=sh)
        s.arrow([(680, 130), (x + 80, 194)], VIO, show=sh)
        # the boundary: envelope down, receipt up
        s.rect(x + 8, 280, 144, 34, P["fault"][0], P["fault"][1], rx=4, dash=True, show=sh)
        s.text(x + 80, 303, "new boundary", 17, P["fault"][2], bold=True, show=sh)
    s.text(680, 342, "each child: its own context,", 18, "#334155", show=sh)
    s.text(680, 366, "grant, budget, and lifecycle", 18, "#334155", show=sh)
    s.save(f"split_step{k}.svg")


for k in (1, 2):
    split(k)


# ================================================== NM 18.1 screen (speedup)
def screen_plot():
    a, b = calcs.screen(0.80, 0.02), calcs.screen(0.80, 0.01)
    ns = list(range(1, 17))
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.axhline(a["ceiling"], color=C["neutral"], ls="--", lw=1.8)
    ax.text(16, a["ceiling"] - 0.12, "ceiling 1/(1−p) = 5× (no overhead)", ha="right",
            va="top", fontsize=12, color=C["neutral"], fontweight="bold")
    ax.plot(ns, [1 / a["t"][n] for n in ns], "-o", color=C["network"], lw=2.4, ms=5)
    ax.plot(ns, [1 / b["t"][n] for n in ns], "-o", color=C["memory"], lw=2.0, ms=4, alpha=0.8)
    ax.plot([a["best"]], [a["speedup"]], "o", ms=12, color=C["danger"], zorder=5)
    ax.annotate(f"c = 0.02: best {a['best']} workers, {a['speedup']:.1f}×",
                (a["best"], a["speedup"]), xytext=(10, -40), textcoords="offset points",
                fontsize=12, color=C["network"], fontweight="bold")
    ax.annotate(f"c = 0.01: best {b['best']}, {b['speedup']:.1f}×", (b["best"], b["speedup"]),
                xytext=(14, 8), textcoords="offset points", fontsize=12,
                color=C["memory"], fontweight="bold")
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 5.4)
    ax.set_xlabel("Workers N (p = 0.80, overhead c per added worker)", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.grid(color=C["grid"], linewidth=0.6)
    sp.save(fig, out / "screen.svg")


screen_plot()


# ======================================================= USL (retrograde)
def usl_plot():
    u, r = calcs.usl_summary(), calcs.usl_summary(0.10, 0.002, nmax=600)
    ns = list(range(1, 61))
    fig, ax = col_axes()
    ax.axvspan(u["first_slower"], 60, color=C["danger_l"], zorder=0)
    ax.axhline(1, color=C["neutral"], ls="--", lw=1.6)
    ax.plot(ns, [calcs.usl(n, 0.2, 0.02) for n in ns], color=C["network"], lw=2.8)
    ax.plot(ns, [calcs.usl(n, 0.1, 0.002) for n in ns], color=C["data"], lw=2.8)
    ax.plot([u["best"]], [u["peak"]], "o", ms=10, color=C["network"])
    ax.plot([r["best"]], [r["peak"]], "o", ms=10, color=C["data"])
    ax.text(u["best"] + 2, u["peak"] + 0.15, f"repair: peak {u['best']}, {u['peak']:.1f}×",
            fontsize=14, color=C["network"], fontweight="bold")
    ax.text(r["best"] - 3, r["peak"] + 0.25, f"research: {r['best']}, {r['peak']:.1f}×",
            fontsize=14, color=C["data"], fontweight="bold")
    ax.text(42, 1.25, f"slower than one\nworker from {u['first_slower']}", fontsize=14,
            color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 6.6)
    ax.set_xlabel("Workers N", fontsize=14)
    ax.set_ylabel("Capacity (× one worker)", fontsize=14)
    sp.save(fig, out / "usl.svg")


usl_plot()


# =================================================== NM 18.6 break-even
def breakeven_plot():
    h = calcs.hook()
    m, be = h["token_multiple"], h["breakeven_single_success"]
    fig, ax = col_axes()
    xs = [i / 200 for i in range(0, 201)]
    ax.fill_between([x for x in xs if x >= be], 0, 1.0, color=C["danger_l"], zorder=0)
    ax.plot([x for x in xs if x <= be], [m * x for x in xs if x <= be], color=C["network"], lw=3)
    ax.plot([be], [1.0], "o", ms=11, color=C["danger"], zorder=5)
    ax.text(be + 0.04, 0.62, f"single agent above\n{100 * be:.1f}%: no acceptance\nrate can pay back",
            fontsize=14, color=C["danger"], fontweight="bold", va="top")
    ax.text(0.10, 0.20, f"slope\n{m:.2f}", fontsize=14,
            color=C["network"], fontweight="bold", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Single-agent acceptance", fontsize=14)
    ax.set_ylabel("Delegated acceptance\nneeded to break even", fontsize=14)
    sp.save(fig, out / "breakeven.svg")


breakeven_plot()


# ================================================== matched trial diagram
def matched():
    s = Svg(900, 360)
    s.box(20, 20, 270, 70, "Single controller", ("parallel tools",), "store")
    s.box(610, 20, 270, 70, "Delegated design", ("supervisor + workers",), "think")
    rows = ["same task sample and acceptance contract", "same model and tool concurrency limits",
            "same token and dollar ceilings", "same fixtures and evidence access",
            "score the integrated task; count every start"]
    for i, r in enumerate(rows):
        y = 112 + i * 48
        s.rect(20, y, 860, 40, P["band"][0] if i % 2 == 0 else "#ffffff", P["band"][1], rx=4)
        s.text(450, y + 27, r, 19, INK)
    s.save("matched.svg")


matched()


# ================================================= three structures (build)
def structures(k):
    s = Svg(900, 400)
    titles = [("Dependency graph", "what must finish first", "think"),
              ("Communication", "who sends messages", "tool"),
              ("Supervision tree", "who owns each lifecycle", "store")]
    for i, (t, sub, role) in enumerate(titles):
        sh = k >= i + 1
        x0 = 10 + i * 300
        s.rect(x0, 10, 280, 380, P["band"][0], P["band"][1], rx=8, show=sh)
        s.text(x0 + 140, 44, t, 21, P[role][2], bold=True, show=sh)
        s.text(x0 + 140, 70, sub, 18, "#334155", show=sh)
        # nodes: S supervisor (top), I interface, C caller, X test container
        S, I, Cn, X = (x0 + 140, 125), (x0 + 65, 235), (x0 + 215, 235), (x0 + 215, 340)
        names = (("S", S, "store"), ("I", I, "think"), ("C", Cn, "think"), ("T", X, "tool"))
        if i == 0:
            s.arrow([(I[0] + 34, I[1]), (Cn[0] - 36, Cn[1])], P["think"][1], show=sh, width=3)
            s.text((I[0] + Cn[0]) / 2, I[1] - 14, "interface", 17, P["think"][2], show=sh)
            s.arrow([(Cn[0], Cn[1] + 34), (X[0], X[1] - 36)], P["think"][1], show=sh, width=3)
        if i == 1:
            for nd in (I, Cn):
                s.arrow([(S[0], S[1] + 32), (nd[0], nd[1] - 34)], P["tool"][1], show=sh, both=True)
            s.text(x0 + 70, 345, "I and C never", 17, "#334155", show=sh)
            s.text(x0 + 70, 367, "talk directly", 17, "#334155", show=sh)
        if i == 2:
            for nd in (I, Cn):
                s.arrow([(S[0], S[1] + 32), (nd[0], nd[1] - 34)], P["store"][1], show=sh, width=3)
            s.arrow([(Cn[0], Cn[1] + 34), (X[0], X[1] - 36)], P["store"][1], show=sh, width=3)
            s.text(x0 + 70, 345, "C owns its", 17, "#334155", show=sh)
            s.text(x0 + 70, 367, "test container", 17, "#334155", show=sh)
        for lab, (cx, cy), role in names:
            if i == 0 and lab == "S":
                continue
            if i == 1 and lab == "T":
                continue
            s.add(f'<circle cx="{cx}" cy="{cy}" r="32" fill="{P[role][0]}" stroke="{P[role][1]}" '
                  f'stroke-width="2.6"/>', sh)
            s.text(cx, cy + 8, lab, 22, P[role][2], bold=True, show=sh)
    s.save(f"structures_step{k}.svg")


for k in (1, 2, 3):
    structures(k)


# ===================================================== mesh vs supervisor
def mesh_plot():
    ns = list(range(1, 11))
    fig, ax = col_axes()
    ax.plot(ns, [calcs.deliveries(n)["mesh"] for n in ns], "-o", color=C["danger"], lw=2.8)
    ax.plot(ns, [calcs.deliveries(n)["supervisor"] for n in ns], "-o", color=C["network"], lw=2.8)
    d8 = calcs.deliveries(8)
    ax.text(0.9, 72, f"full mesh N(N−1):\n{d8['mesh']} at N = 8", fontsize=14,
            color=C["danger"], fontweight="bold")
    ax.text(6.2, 25, f"supervisor 2N: {d8['supervisor']}", fontsize=14, color=C["network"],
            fontweight="bold")
    ax.set_xlabel("Agents N", fontsize=14)
    ax.set_ylabel("Deliveries per round", fontsize=14)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 95)
    sp.save(fig, out / "mesh.svg")


mesh_plot()


# =============================================================== DAG Gantt
def gantt(k):
    g = calcs.dag()
    rows = [("v1 spec", "v1"), ("v2 module", "v2"), ("v3 module", "v3"), ("v4 module", "v4"),
            ("v5 assess", "v5"), ("v6 bindings", "v6"), ("v7 integrate", "v7")]
    crit = {"v1", "v2", "v5", "v7"}
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    n = len(rows)
    for i, (lab, key) in enumerate(rows):
        y = n - 1 - i
        dur = calcs.DAG[key][0]
        oncrit = k >= 2 and key in crit
        color = C["network"] if oncrit else (C["memory"] if key != "v7" else C["data"])
        ax.barh(y, dur, left=g["start"][key], height=0.62, color=color,
                edgecolor=C["danger"] if oncrit else "none", linewidth=2.4 if oncrit else 0)
        ax.text(-3, y, lab, ha="right", va="center", fontsize=12, color=C["text"],
                fontweight="bold")
    if k >= 2:
        ys = n - 1 - 5
        ax.annotate("", xy=(g["start"]["v7"], ys), xytext=(g["finish"]["v6"], ys),
                    arrowprops=dict(arrowstyle="<->", color=C["data"], lw=2))
        ax.text(g["start"]["v7"] + 2, ys, f"{g['slack_v6']:.1f} s slack", fontsize=12,
                color=C["data"], fontweight="bold", va="center")
        ax.text(g["makespan"], n - 0.8, f"done at {g['makespan']:.1f} s", ha="right",
                fontsize=12, color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Seconds (4 slots; illustrative durations)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"gantt_step{k}.svg")


for k in (1, 2):
    gantt(k)


# ======================================================== admission bounds
def bounds():
    s = Svg(540, 400)
    s.box(185, 10, 170, 56, "Lead", (), "store", tsize=23)
    xs = [12, 144, 276, 408]
    for i, x in enumerate(xs):
        s.arrow([(270, 66), (x + 60, 120)], GRY)
        s.box(x, 122, 120, 52, f"child {i + 1}", (), "think", tsize=22)
    s.arrow([(72, 174), (72, 232)], GRY)
    s.box(12, 234, 160, 52, "grandchild", (), "think", tsize=22)
    s.text(470, 100, "breadth", 22, P["fault"][2], bold=True)
    s.line(12, 190, 528, 190, RED, 2, dash=True)
    s.text(370, 222, "concurrent slots", 22, P["fault"][2], bold=True)
    s.text(232, 268, "depth", 22, P["fault"][2], bold=True)
    s.rect(6, 304, 528, 90, P["store"][0], P["store"][1], rx=6)
    s.text(270, 336, "reserve each child's maximum", 22, P["store"][2], bold=True)
    s.text(270, 364, "charge atomically before it starts,", 21, P["store"][2])
    s.text(270, 388, "and set a deadline", 21, P["store"][2])
    s.save("bounds.svg")


bounds()


# =================================================== supervision blast radius
def blast():
    s = Svg(540, 400)
    for j, (name, sub, kill) in enumerate((("one_for_one", "failed child only", (1,)),
                                           ("one_for_all", "whole group", (0, 1, 2)))):
        y0 = 6 + j * 198
        s.rect(6, y0, 528, 186, P["band"][0], P["band"][1], rx=8)
        s.text(20, y0 + 34, name, 23, P["store"][2], bold=True, anchor="start")
        s.text(522, y0 + 34, sub, 21, "#334155", anchor="end")
        for i in range(3):
            x = 18 + i * 172
            failed = i == 1
            role = "fault" if failed else ("tool" if i in kill else "think")
            lab = "fails" if failed else ("restarts" if i in kill else "continues")
            s.box(x, y0 + 60, 158, 104, f"child {i + 1}", (lab,), role, tsize=22, ssize=21)
    s.save("blast.svg")


blast()


# =============================================== clean merge, broken change
def merge(k):
    s = Svg(900, 330)
    s.line(30, 80, 870, 80, "#334155", 3)
    s.circle(80, 80, 14, P["plain"][0], P["plain"][1], 3)
    s.text(80, 50, "repo_base_01", 21, INK, bold=True)
    s.text(98, 120, "foo(a)", 21, "#334155", anchor="start")
    sh2 = k >= 2
    s.circle(400, 80, 14, P["ok"][0], P["ok"][1], 3, show=sh2)
    s.text(400, 50, "repo_head_02", 21, P["ok"][2], bold=True, show=sh2)
    s.text(400, 120, "interface lands: foo(a, b)", 21, P["ok"][2], show=sh2)
    s.arrow([(80, 94), (80, 200), (250, 200)], P["think"][1])
    s.box(252, 160, 310, 80, "caller patch", ("calls foo(a), built on base",), "think",
          tsize=22, ssize=20)
    sh3 = k >= 3
    s.arrow([(562, 200), (720, 200), (720, 96)], P["think"][1], show=sh3)
    s.circle(720, 80, 14, P["fault"][0], P["fault"][1], 3, show=sh3)
    s.text(720, 50, "merge: no conflict", 21, P["ok"][2], bold=True, show=sh3)
    s.rect(480, 252, 412, 74, P["fault"][0], P["fault"][1], rx=6, show=sh3)
    s.text(686, 282, "tests: foo(a) no longer exists", 21, P["fault"][2], bold=True, show=sh3)
    s.text(686, 312, "different files, broken change", 21, P["fault"][2], show=sh3)
    s.save(f"merge_step{k}.svg")


for k in (1, 2, 3):
    merge(k)


# ================================================ equation 4: shared writes
def birthday_plot():
    c = calcs.collisions()
    ns = list(range(1, 17))
    fig, ax = col_axes()
    ax.axhline(0.5, color=C["neutral"], ls="--", lw=1.6)
    ax.plot(ns, [c["p"][n] for n in ns], "-", color=C["network"], lw=3)
    for n, dx, dy in ((2, 1.0, -0.07), (c["even"], 0.6, -0.1), (8, 0.6, -0.05)):
        ax.plot([n], [c["p"][n]], "o", ms=10, color=C["danger"] if n == c["even"] else C["network"])
        ax.text(n + dx, c["p"][n] + dy, f"{n} workers: {100 * c['p'][n]:.1f}%", fontsize=14,
                color=C["danger"] if n == c["even"] else C["network"], fontweight="bold")
    ax.text(1, 0.53, "even odds", ha="left", fontsize=13, color=C["neutral"])
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Workers (1,000 files; 10 written each)", fontsize=14)
    ax.set_ylabel("P(at least one shared write)", fontsize=14)
    sp.save(fig, out / "birthday.svg")


birthday_plot()


# ===================================================== guarded publication
def gate(k):
    s = Svg(540, 420)
    steps = [("1 Revalidate", "dependencies vs. current head"),
             ("2 Build", "the exact merge candidate"),
             ("3 Check", "that candidate, sealed tests"),
             ("4 Publish", "only if head is unchanged")]
    for i, (t, sub) in enumerate(steps):
        sh = k >= i + 1
        y = 14 + i * 100
        role = "ok" if i == 3 else "store"
        s.box(90, y, 360, 74, t, (sub,), role, tsize=20, ssize=18, show=sh)
        if i < 3:
            s.arrow([(270, y + 74), (270, y + 98)], GRY, show=k >= i + 2)
    sh = k >= 4
    s.arrow([(450, 365), (510, 365), (510, 51), (452, 51)], RED, show=sh, dash=True)
    s.add(f'<text x="528" y="210" font-family="{FONT}" font-size="17" fill="{P["fault"][2]}" '
          f'font-weight="bold" text-anchor="middle" transform="rotate(90 528 210)">'
          f'head moved: start over</text>', sh)
    s.save(f"gate_step{k}.svg")


for k in (1, 2, 3, 4):
    gate(k)


# ===================================================== Kim et al. (real data)
def kim_plot():
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.barh([1], [100 * calcs.KIM_RANDOM], color=C["neutral_l"], height=0.6)
    ax.barh([0], [100 * calcs.KIM_MEASURED], color=C["danger"], height=0.6)
    ax.set_yticks([0, 1], ["measured", "if errors\nwere random"], fontsize=14)
    ax.text(100 * calcs.KIM_RANDOM + 1.5, 1, "33%", va="center", fontsize=15,
            color=C["neutral"], fontweight="bold")
    ax.text(100 * calcs.KIM_MEASURED + 1.5, 0, "about 60%", va="center", fontsize=15,
            color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Same wrong answer, when both models err (%)", fontsize=13)
    ax.tick_params(labelsize=13)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / "kim.svg")


kim_plot()


# ============================================ NM 18.4 the correlated floor
def floor_plot(k):
    r = calcs.reviewers()
    ns = sorted(r["ind"])
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.plot(ns, [100 * r["ind"][n] for n in ns], "-o", color=C["memory"], lw=2.6)
    ax.text(9.4, 100 * r["ind"][9] + 2.0, "independent errors", fontsize=12,
            color=C["memory"], fontweight="bold")
    sh = k >= 2
    ax.axhline(100 * r["floor"], color=C["danger"], ls="--", lw=1.8, alpha=1 if sh else 0)
    ax.plot(ns, [100 * r["mix"][n] for n in ns], "-o", color=C["network"], lw=2.6,
            alpha=1 if sh else 0)
    ax.text(15, 100 * r["mix"][15] + 1.2, f"shared difficulty: {100 * r['mix'][15]:.1f}% at 15",
            ha="right", fontsize=12, color=C["network"], fontweight="bold", alpha=1 if sh else 0)
    ax.text(15, 100 * r["floor"] - 1.0, f"floor {100 * r['floor']:.1f}%", ha="right", va="top",
            fontsize=12, color=C["danger"], fontweight="bold", alpha=1 if sh else 0)
    ax.text(5.2, 100 * r["mix"][5] + 1.6, f"{100 * r['mix'][5]:.2f}%", fontsize=12,
            color=C["network"], fontweight="bold", alpha=1 if sh else 0)
    ax.text(5.3, 100 * r["ind"][5] + 0.8, f"{100 * r['ind'][5]:.2f}%", fontsize=12,
            color=C["memory"], fontweight="bold")
    ax.set_xticks(ns)
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0, 22)
    ax.set_xlabel("Reviewers (each falsely approves 20%; correlation 0.40)", fontsize=12)
    ax.set_ylabel("Majority false\napproval (%)", fontsize=12)
    ax.grid(color=C["grid"], linewidth=0.6)
    sp.save(fig, out / f"floor_step{k}.svg")


for k in (1, 2):
    floor_plot(k)


# =================================================== NM 18.5 the barrier
def barrier_plot():
    b = calcs.barrier()
    ns = list(range(1, 17))
    fig, ax = col_axes()
    ax.axhline(0.5, color=C["neutral"], ls="--", lw=1.6)
    ax.plot(ns, [b["all"][n] for n in ns], "-", color=C["network"], lw=3)
    n7 = b["below_half"]
    ax.plot([n7], [b["all"][n7]], "o", ms=11, color=C["danger"])
    ax.text(1, 0.30, f"{n7} workers:\n{100 * b['all'][n7]:.0f}%, below even",
            fontsize=14, color=C["danger"], fontweight="bold")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.plot([8], [b["quorum"]], "s", ms=11, color=C["data"])
    ax.text(8.6, b["quorum"] - 0.02, f"any 6 of 8: {100 * b['quorum']:.2f}%", fontsize=14,
            color=C["data"], fontweight="bold", va="top")
    ax.text(15.5, 0.53, "even odds", ha="right", fontsize=13, color=C["neutral"])
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Workers, each on time 90%", fontsize=14)
    ax.set_ylabel("P(all on time)", fontsize=14)
    sp.save(fig, out / "barrier.svg")


barrier_plot()


# ===================================================== credit window bars
def credit_plot():
    c = calcs.credit()
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    vals = [c["wait_h"] * 3600, c["residence"]]
    ax.barh([1, 0], vals, color=[C["danger"], C["data"]], height=0.62)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["window of 8", "no window"], fontsize=16)
    ax.text(vals[0] / 1.15, 1, f"{c['wait_h']:.1f} h", ha="right", va="center", fontsize=17,
            color="white", fontweight="bold")
    ax.text(vals[1] * 1.2, 0, f"{c['residence']:.0f} s", va="center", fontsize=17,
            color=C["data"], fontweight="bold")
    ax.set_xlim(10, 1e5)
    ax.set_xticks([10, 100, 1000, 10000, 100000], ["10 s", "100 s", "17 min", "2.8 h", "28 h"])
    ax.minorticks_off()
    ax.set_xlabel("Wait of the newest diff (log scale)", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / "credit.svg")


credit_plot()


# ================================================= cancellation (build 3)
def cancel(k):
    s = Svg(900, 420)
    s.box(310, 8, 280, 72, "migration_task_01", ("generation 7",), "store", tsize=21, ssize=20)
    if k >= 2:
        s.box(310, 8, 280, 72, "migration_task_01", ("canceled: gen 8",), "fault", tsize=21, ssize=20)
    for x, name in ((40, "worker_interface_01"), (560, "worker_caller_01")):
        s.arrow([(450, 80), (x + 150, 118)], GRY)
        s.box(x, 120, 300, 68, name, ("grant bound to gen 7",), "think", tsize=21, ssize=20)
    s.rect(20, 222, 860, 48, P["band"][0], P["band"][1], rx=6)
    s.text(450, 254, "gateway checks every ancestor's generation", 21, INK, bold=True)
    s.arrow([(190, 188), (190, 290)], AMB)
    s.box(40, 292, 300, 84, "test job, admitted", ("before the cancel",), "tool", tsize=21, ssize=20)
    sh2 = k >= 2
    s.arrow([(710, 188), (710, 220)], RED, show=sh2)
    s.box(560, 292, 300, 84, "new tool call", ("rejected: gen 8 ≠ 7",), "fault", tsize=21,
          ssize=20, show=sh2)
    sh3 = k >= 3
    s.rect(40, 292, 300, 84, "none", P["ok"][1], rx=6, sw=5, show=sh3)
    s.text(40, 408, "keeps its owner, reservation, and record until it settles", 21,
           P["ok"][2], bold=True, anchor="start", show=sh3)
    s.save(f"cancel_step{k}.svg")


for k in (1, 2, 3):
    cancel(k)


# ================================================ attenuation (build 2)
def atten(k):
    s = Svg(900, 400)
    s.box(290, 8, 320, 100, "Root", ("read repo, edit callers,", "publish; $39 available"), "store",
          tsize=22, ssize=20)
    s.arrow([(450, 108), (210, 150)], VIO)
    s.box(30, 152, 360, 100, "worker_caller_01", ("edit callers only;", "$13 reserved"), "think",
          tsize=22, ssize=20)
    s.arrow([(210, 252), (210, 290)], VIO)
    s.box(30, 292, 360, 100, "inspection child", ("read one file;", "$5 out of the $13"), "think",
          tsize=22, ssize=20)
    sh = k >= 2
    s.arrow([(390, 342), (600, 342), (600, 112)], RED, dash=True, show=sh)
    s.text(620, 160, "names the root", 21, P["fault"][2], bold=True, anchor="start", show=sh)
    s.text(620, 186, "as its parent", 21, P["fault"][2], bold=True, anchor="start", show=sh)
    s.box(620, 220, 270, 100, "rejected", ("authority comes from", "the grant it holds"), "fault",
          tsize=22, ssize=20, show=sh)
    s.save(f"atten_step{k}.svg")


for k in (1, 2):
    atten(k)


# =============================================================== MAST bars
def mast_plot():
    rows = list(reversed(calcs.MAST_MODES))
    col = {"design": C["network"], "align": C["compute"], "verify": C["danger"]}
    short = {"Reasoning-action mismatch": "Reasoning-action\nmismatch",
             "Unaware of termination": "Unaware of\ntermination",
             "Disobey task specification": "Disobey task spec",
             "No or incomplete verification": "No or incomplete\nverification",
             "Incorrect verification": "Incorrect\nverification"}
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (lab, v, cat) in enumerate(rows):
        ax.barh(i, v, color=col[cat], height=0.7)
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=14, color=col[cat], fontweight="bold")
        ax.text(-0.4, i, short.get(lab, lab), ha="right", va="center", fontsize=13,
                color=C["text"], linespacing=0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Share of failures, 1,642 traces (%)", fontsize=13)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / "mast.svg")


mast_plot()


# ======================================================== next lecture thumb
def next_bars():
    co = calcs.cohort()
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.bar([0, 1], [co["single"]["per_acc"], co["delegated"]["per_acc"]],
           color=[C["memory"], C["network"]], width=0.6)
    ax.set_xticks([0, 1], ["single", "delegated"])
    ax.set_ylabel("Cost per accepted task")
    sp.save(fig, out / "next_bars.svg")


next_bars()
print("figures written")
