#!/usr/bin/env python3
"""Figures for Lecture 14 (Operating Agents), Volume III.

Every value comes from calcs.py, which this script imports, so figures and
slides agree. "OA" is the rendered chapter, Operating Agents. The book SVGs
(tail-sampling-pipeline, canary-deployment-pipeline, SPRT corridor) are
page-sized and fall below the legibility floor on a slide, so they are
redrawn here at slide size:
  - tail_pipe_step*: OA figure 1 (tail-sampling-pipeline.svg).
  - stages_step*: OA figure 2 (vol3-canary-deployment-pipeline.svg).
  - sprt_step*: OA figure 3 (vol3_eval_margin_sprt_corridor.svg), with
    illustrative walks added.
Full-width diagrams are 900 px wide with type of 18 px or more; column
figures are drawn at 5.4 x 3.6 in (or 540 px wide). Plots use slideplot.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""

import math
import random
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
    "store": ("#f3e8ff", "#7c3aed", "#6d28d9"),   # logs, records, supervisor
    "tool": ("#fef3c7", "#d97706", "#b45309"),    # tools, actuation, rollout
    "fault": ("#fee2e2", "#dc2626", "#b91c1c"),   # faults, holds, violations
    "ok": ("#dcfce7", "#16a34a", "#15803d"),      # accepted, kept, settled
    "think": ("#e0f2fe", "#0284c7", "#0369a1"),   # model calls
    "plain": ("#ffffff", "#94a3b8", "#0f172a"),
    "band": ("#f8fafc", "#cbd5e1", "#0f172a"),
    "gray": ("#f1f5f9", "#94a3b8", "#475569"),
}
FONT = "Helvetica, Arial, sans-serif"
VIO, RED, GRN, AMB, SKY, GRY = "#7c3aed", "#dc2626", "#16a34a", "#d97706", "#0284c7", "#64748b"


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

    def text(self, x, y, s, size=19, color="#0f172a", show=True, **kw):
        self.add(self._t(x, y, s, size, color, **kw), show)

    def box(self, x, y, w, h, title, subs=(), role="plain", show=True, dash=False,
            tsize=21, ssize=18, rx=6, sw=2.4, title_color=None):
        fill, stroke, text = P[role]
        d = ' stroke-dasharray="8,5"' if dash else ""
        s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="{sw}"{d}/>')
        n = 1 + len(subs)
        lh = [tsize] + [ssize] * len(subs)
        gap = 7
        total = sum(lh) + gap * (n - 1)
        y0 = y + (h - total) / 2
        cx = x + w / 2
        yy = y0 + tsize * 0.8
        s += self._t(cx, yy, title, tsize, title_color or text, bold=True)
        for sub in subs:
            yy += gap + ssize
            s += self._t(cx, yy, sub, ssize, "#1e293b")
        self.add(s, show)

    def arrow(self, pts, color=GRY, show=True, dash=False, width=2.6):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
        da = ' stroke-dasharray="8,5"' if dash else ""
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        L, Wd = 14, 7
        bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
        p1 = (bx + Wd * math.sin(a), by - Wd * math.cos(a))
        p2 = (bx - Wd * math.sin(a), by + Wd * math.cos(a))
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>'
                 f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
                 f'fill="{color}"/>', show)

    def rect(self, x, y, w, h, fill, stroke="none", show=True, rx=0, sw=2, dash=False,
             opacity=1.0):
        d = ' stroke-dasharray="8,5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>', show)

    def line(self, x1, y1, x2, y2, color=GRY, width=2, dash=False, show=True):
        d = ' stroke-dasharray="6,5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{width}"{d}/>', show)

    def save(self, name):
        body = "\n".join(self.items)
        (out / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')


# ===================================================== hook timeline (OpenAI)
def hook_timeline(k):
    h = calcs.HOOK
    s = Svg(900, 232 if k < 4 else 304)
    x0, day = 262, 105          # x of 24 Apr, pixels per day

    def X(d):
        return x0 + (d - 24) * day
    s.box(10, 14, 215, 140, "Before launch", ("offline evals:", "looked good", "A/B test:", "looked good"),
          "ok", tsize=19, ssize=18)
    s.line(X(24), 180, X(29.4), 180, "#334155", 2)
    names = {24: "Thu 24", 25: "Fri 25", 26: "Sat 26", 27: "Sun 27", 28: "Mon 28", 29: "Tue 29"}
    for d, nm in names.items():
        s.line(X(d), 172, X(d), 188, "#334155", 2)
        s.text(X(d), 214, nm, 18, "#334155")
    s.rect(X(h["rollout_start"]), 14, day * (h["rollout_done"] - h["rollout_start"]), 40,
           P["tool"][0], P["tool"][1], rx=5, sw=2.4)
    s.text(X(24.5), 41, "Rollout", 19, P["tool"][2], bold=True)
    sh2 = k >= 2
    s.rect(X(25), 64, day * 2, 40, P["think"][0], P["think"][1], rx=5, sw=2.4, show=sh2)
    s.text(X(26), 91, "Live; monitored", 19, P["think"][2], bold=True, show=sh2)
    sh3 = k >= 3
    s.line(X(27.85), 10, X(27.85), 172, RED, 2.4, dash=True, show=sh3)
    s.text(X(27.85) - 8, 140, "Prompt patch", 18, P["fault"][2], anchor="end", bold=True, show=sh3)
    s.rect(X(h["rollback_start"]), 114, day * h["rollback_hours"] / 24, 40,
           P["fault"][0], P["fault"][1], rx=5, sw=2.4, show=sh3)
    s.text(X(28.5), 141, "Rollback", 19, P["fault"][2], bold=True, show=sh3)
    s.text(X(29) + 8, 141, "about 24 h", 18, P["fault"][2], anchor="start", show=sh3)
    if k >= 4:
        s.rect(10, 230, 880, 68, P["band"][0], P["band"][1], rx=6)
        for x, a, b in ((117, "No criterion", "for sycophancy"), (X(26), "Detected outside", "the declared gate"),
                        (X(28.5) + 20, "Admission stop,", "then settlement")):
            s.text(x, 258, a, 18, "#0f172a", bold=True)
            s.text(x, 284, b, 18)
    s.save(f"hook_step{k}.svg")


for k in range(1, 5):
    hook_timeline(k)


# ======================================================= Gantt helpers (mpl)
def gantt(name, rows, xmax, notes=(), size=(5.4, 3.4), crit=(), ghost=(), qmark=None,
          gap=None, fs=15):
    """rows: list of (label, start, end, color_key). crit: labels outlined in
    violet (the critical path). ghost: (label, start, end) drawn faint.
    gap: (start, end) hatched as unattributed time."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (label, s0, e0, ck) in enumerate(rows):
        y = n - 1 - i
        for gl, gs, ge in ghost:
            if gl == label:
                ax.barh(y, ge - gs, left=gs, height=0.6, color="white",
                        edgecolor=C["neutral"], hatch="///", linewidth=1.2, zorder=1)
        if e0 is not None:
            ax.barh(y, e0 - s0, left=s0, height=0.6, color=C[ck], zorder=2,
                    edgecolor=VIO if label in crit else "none",
                    linewidth=3.0 if label in crit else 0)
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=fs,
                color=C["text"], fontweight="bold")
    if gap:
        ax.axvspan(gap[0], gap[1], color=C["danger_l"], zorder=0)
    if qmark:
        y = n - 1 - [r[0] for r in rows].index(qmark[0])
        ax.text(qmark[1], y, "?", ha="center", va="center", fontsize=22,
                color=C["danger"], fontweight="bold")
    for (x, y, txt, color, ha) in notes:
        ax.text(x, y, txt, ha=ha, va="center", fontsize=fs - 1, color=color,
                fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Seconds after segment start", fontsize=fs - 2)
    ax.tick_params(axis="x", labelsize=fs - 2)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / name)


# ============================================ real OpenTelemetry run (Gantt)
run = calcs.run_spans()
S = run["spans"]
LBL = [("root span", "invoke_agent repair_agent", "neutral"),
       ("retrieve", "retrieve supporting_context", "data"),
       ("model: propose", "chat propose_checks", "memory"),
       ("run_tests", "execute_tool run_tests", "compute"),
       ("run_lint", "execute_tool run_lint", "compute"),
       ("model: report", "chat synthesize_report", "memory")]
rows = [(lbl, S[k][0], S[k][1], ck) for lbl, k, ck in LBL]
gantt("run_step1.svg", rows, 12.6)
gantt("run_step2.svg", rows, 12.6,
      crit=("retrieve", "model: propose", "run_tests", "model: report"),
      gap=(S["execute_tool run_tests"][1], S["chat synthesize_report"][0]),
      notes=[(S["execute_tool run_lint"][1] + 0.3, 1, "off the path", C["neutral"], "left"),
             (9.9, 2, f"{run['gap']:.1f} s gap", C["danger"], "left")])

# ==================================== cut the tests: which span limits now?
d = calcs.NM141
t_ret, t_pro = d["retrieve"], d["retrieve"] + d["propose"]
t_chk = t_pro
nm = calcs.nm141()
base = [("retrieve", 0, t_ret, "data"), ("model: propose", t_ret, t_pro, "memory"),
        ("run_tests", t_chk, t_chk + 1.0, "compute"), ("run_lint", t_chk, t_chk + d["lint"], "compute"),
        ("model: report", None, None, "memory")]
gantt("cut_step1.svg", base, 12.6, ghost=[("run_tests", t_chk + 1.0, t_chk + d["tests"])],
      qmark=("model: report", 8.0),
      notes=[(t_chk + 6.2, 2, "was 6 s", C["neutral"], "left")])
end_l = t_chk + d["lint"] + d["gap"]
done = [r if r[0] != "model: report" else ("model: report", end_l, end_l + d["synth"], "memory")
        for r in base]
gantt("cut_step2.svg", done, 12.6, ghost=[("run_tests", t_chk + 1.0, t_chk + d["tests"])],
      crit=("retrieve", "model: propose", "run_lint", "model: report"),
      notes=[(end_l + d["synth"] + 0.25, 0, f"{nm['tests_cut_to_1']:.1f} s", C["text"], "left"),
             (t_chk + 6.2, 2, "was 6 s", C["neutral"], "left")])


# ====================================================== three records (SVG)
def records(k):
    s = Svg(900, 400)
    s.box(300, 14, 300, 64, "One runtime event", ("model call, tool run, or check",),
          "plain", tsize=20, ssize=18)
    cards = [
        (30, "Diagnostic trace", ("timing and debugging", "may sample, cut, or drop"), "think"),
        (330, "Trajectory log", ("recovery rebuilds from it", "never sampled"), "store"),
        (630, "Accounting record", ("every trial, verdict,", "spend, and drop"), "ok"),
    ]
    for i, (x, t, subs, role) in enumerate(cards):
        sh = k >= i + 1
        s.arrow([(450, 78), (x + 120, 160)], P[role][1], show=sh)
        s.box(x, 162, 240, 120, t, subs, role, show=sh, tsize=21, ssize=18)
    s.text(150, 318, "Explains one trial", 18, P["think"][2], bold=True)
    s.text(450, 318, "Lecture 11", 18, P["store"][2], bold=True, show=k >= 2)
    s.text(750, 318, "Every rate comes from here", 18, P["ok"][2], bold=True, show=k >= 3)
    s.text(450, 375, "A span's status is not a task verdict.", 20, P["fault"][2],
           bold=True, show=k >= 3)
    s.save(f"records_step{k}.svg")


for k in range(1, 4):
    records(k)

# ========================================================== Dapper Table 2
fig, ax = plt.subplots(figsize=(5.8, 2.5))
rows_d = calcs.DAPPER
for i, (lab, v) in enumerate(rows_d):
    y = len(rows_d) - 1 - i
    ck = "danger" if v > 5 else ("neutral" if v > 2.5 else "data")
    ax.barh(y, v, height=0.62, color=C[ck])
    ax.text(-1.0, y, f"1 trace per {lab.split('/')[1]}" if lab != "1/1" else "every request",
            ha="right", va="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.text(max(v, 0) + 0.3, y, f"{v:+.2f}%".replace("+", "+") if v < 0 else f"+{v:g}%",
            va="center", fontsize=12, fontweight="bold", color=C[ck])
ax.axvspan(-2.5, 2.5, color=C["neutral_l"], alpha=0.6, zorder=0)
ax.text(2.7, len(rows_d) - 0.35, "within error (±2.5%)", fontsize=11, color=C["neutral"],
        va="bottom")
ax.set_xlim(-3, 21)
ax.set_ylim(-0.6, len(rows_d) - 0.1)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Change in average latency, web search cluster (%)")
ax.axvline(0, color="#888888", lw=1)
sp.save(fig, out / "dapper_bars.svg")


# ============================================ head sampling miss probability
def head_miss(k):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ns = list(range(1, 61))
    curves = [(0.01, "p = 1%", "danger"), (0.10, "p = 10%", "memory"),
              (0.05, "p = 5%", "compute")]
    for idx, (p, lab, ck) in enumerate(curves):
        if idx == 2 and k < 2:
            continue
        ax.plot(ns, [100 * calcs.miss(p, n) for n in ns], color=C[ck], lw=3)
    ax.text(42, 100 * calcs.miss(0.01, 42) + 4, "p = 1%", color=C["danger"],
            fontsize=15, fontweight="bold", ha="center")
    ax.text(1.5, 16, "p = 10%", color=C["memory"],
            fontsize=15, fontweight="bold", ha="left")
    pts = [(0.01, 10, "danger", (8, 10)), (0.10, 5, "memory", (-40, -22))]
    if k >= 2:
        pts.append((0.05, 20, "compute", (10, 6)))
        ax.text(42, 100 * calcs.miss(0.05, 42) + 5, "p = 5%", color=C["compute"],
                fontsize=15, fontweight="bold", ha="left")
    for p, n, ck, off in pts:
        v = 100 * calcs.miss(p, n)
        ax.plot([n], [v], "o", ms=10, color=C[ck], mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{v:.0f}%", (n, v), xytext=off, textcoords="offset points",
                    fontsize=15, fontweight="bold", color=C[ck])
    if k < 2:
        ax.plot([20], [36], "o", ms=12, mfc="white", mec=C["compute"], mew=2.5)
        ax.annotate("?", (20, 36), xytext=(10, 4), textcoords="offset points",
                    fontsize=18, fontweight="bold", color=C["compute"])
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Times the failure mode occurs", fontsize=14)
    ax.set_ylabel("P(no detailed trace kept), %", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"miss_step{k}.svg")


for k in (1, 2):
    head_miss(k)


# ============================================ tail selection pipeline (SVG)
def tail_pipe(k):
    s = Svg(900, 330)
    s.box(10, 150, 160, 80, "Admitted", ("trials",), "plain", tsize=21, ssize=20)
    s.box(240, 10, 360, 80, "Accounting record", ("every trial, verdict, drop",),
          "ok", tsize=21, ssize=18)
    s.arrow([(90, 150), (90, 50), (236, 50)], P["ok"][1])
    s.box(240, 140, 240, 100, "Staging buffer", ("held until the", "verdict window ends"),
          "store", tsize=20, ssize=18)
    s.arrow([(170, 190), (236, 190)], GRY)
    sh = k >= 2
    cls = [("Failed or violation: keep", "fault"), ("Pending: keep, flag", "tool"),
           ("Slow or costly: keep", "think"), ("Ordinary: 2% control", "gray")]
    for i, (t_, role) in enumerate(cls):
        y = 104 + i * 56
        s.box(604, y, 286, 46, t_, (), role, show=sh, tsize=19)
        s.arrow([(480, 190), (540, 190), (540, y + 23), (600, y + 23)], P[role][1], show=sh)
    s.text(747, 98, "evict from the bottom first", 18, "#475569", show=sh, italic=True)
    s.arrow([(360, 240), (360, 290), (200, 290), (200, 94)], RED, dash=True, show=sh)
    s.text(372, 284, "buffer expired: the record", 18, P["fault"][2], anchor="start", show=sh)
    s.text(372, 308, "flags the missing detail", 18, P["fault"][2], anchor="start", show=sh)
    s.save(f"tail_step{k}.svg")


for k in (1, 2):
    tail_pipe(k)

# ======================================================= NM 14.2 retention
r142 = calcs.nm142()
sp.hbars(str(out / "retention.svg"), [
    ("Keep all", r142["full_gb"], "neutral", f"{r142['full_gb']:.0f} GB"),
    ("Head 5%", r142["head_gb"], "memory",
     f"{r142['head_gb']:.1f} GB: {r142['head_failed']:.0f} of 400 failures/h"),
    ("Tail", r142["tail_gb"], "data",
     f"{r142['tail_gb']:.1f} GB: all 400 failures/h"),
], "GB", log=False, xlabel="Retained trace bodies per day (GB, decimal)", size=(5.8, 2.0))


# =================================================== postmortem findings
def findings(k):
    s = Svg(560, 384)
    items = [("Observation adapter", "cut the result, no notice", "fault", "interface", "defect"),
             ("Model", "proposed an out-of-scope edit", "think", "model", "behavior"),
             ("Runtime gate", "refused before dispatch", "ok", "gate", "worked")]
    for i, (t_, sub, role, v1, v2) in enumerate(items):
        y = 6 + i * 128
        sh = k >= i + 1
        s.box(6, y, 404, 100, t_, (sub,), role, show=sh, tsize=24, ssize=21)
        s.text(422, y + 46, v1, 21, P[role][2], anchor="start", bold=True, show=sh)
        s.text(422, y + 72, v2, 21, P[role][2], anchor="start", bold=True, show=sh)
        if i < 2:
            s.arrow([(208, y + 100), (208, y + 126)], GRY, show=k >= i + 2)
    s.save(f"findings_step{k}.svg")


for k in (1, 2, 3):
    findings(k)

# ============================================================ replay modes
s = Svg(900, 360)
s.box(250, 10, 400, 64, "Recorded trajectory", ("saved prefix and captured results",),
      "store", tsize=20, ssize=18)
modes = [(20, "Time-travel replay", ("shows what the", "model received"), "store",
          "no new outcome"),
         (320, "Regression replay", ("old vs. new adapter", "on captured output"), "tool",
          "no model call"),
         (620, "Counterfactual branch", ("fork the prefix,", "change one thing"), "think",
          "fresh model calls")]
for x, t, subs, role, foot in modes:
    s.arrow([(450, 74), (x + 130, 128)], GRY)
    s.box(x, 130, 260, 132, t, subs, role, tsize=21, ssize=18)
    s.text(x + 130, 300, foot, 19, P[role][2], bold=True)
s.text(450, 345, "Only the branch tests whether a fix changes the outcome.", 19,
       "#0f172a", bold=True)
s.save("replay.svg")


# ========================================================= Example 14.1 bars
def branches(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows_b = calcs.EX141
    for i, (name, acc, forb, bud) in enumerate(rows_b):
        y = len(rows_b) - 1 - i
        ax.text(-0.4, y, name, ha="right", va="center", fontsize=13, fontweight="bold",
                color=C["text"])
        if k < 2:
            ax.barh(y, 20, height=0.6, color="white", edgecolor=C["neutral"], lw=1.2)
            ax.text(10, y, "? of 20 accepted", ha="center", va="center", fontsize=13,
                    color=C["neutral"])
            continue
        left = 0
        for v, ck, lab in ((acc, "data", "accepted"), (forb, "danger", "forbidden edit"),
                           (bud, "neutral", "budget stop")):
            if v:
                ax.barh(y, v, left=left, height=0.6, color=C[ck])
                txt = f"{v} {lab}" if v >= 6 else f"{v}"
                ax.text(left + v / 2, y, txt, ha="center", va="center", fontsize=12,
                        color="white", fontweight="bold")
            left += v
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(-0.6, len(rows_b) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("20 continuations per branch from one saved prefix (illustrative)")
    sp.save(fig, out / f"branches_step{k}.svg")


for k in (1, 2):
    branches(k)

# ================================================= Wilson bound, 0 failures
ex = calcs.ex141()
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ns = list(range(5, 121))
ax.plot(ns, [100 * v for v in calcs.wilson_curve(ns)], color=VIO, lw=3)
for n, ck in ((20, "danger"), (ex["runs_for_5pct"], "data")):
    v = 100 * calcs.wilson(0, n)[1]
    ax.plot([n], [v], "o", ms=10, color=C[ck], mec="white", mew=1.5, zorder=5)
    lab = f"20 of 20: up to {v:.0f}%" if n == 20 else f"{n} clean runs: under 5%"
    ax.annotate(lab, (n, v), xytext=(10, 8), textcoords="offset points", fontsize=15,
                fontweight="bold", color=C[ck])
ax.axhline(5, color=C["neutral"], lw=1.5, ls="--")
ax.set_xlim(0, 122)
ax.set_ylim(0, 45)
ax.set_xlabel("Consecutive accepted runs, no failures", fontsize=14)
ax.set_ylabel("Upper limit on failure rate, %", fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "wilson.svg")


# =================================================== staged release (SVG)
def stages(k):
    s = Svg(560, 436)
    st = [("Offline gate", "paired comparison", "tool"),
          ("Shadow", "copied inputs, no live effects", "think"),
          ("Canary", "fraction f of live tasks", "tool"),
          ("Wider release", "next stage, own question", "ok")]
    for i, (t_, sub, role) in enumerate(st):
        y = 6 + i * 108
        sh = k >= i + 1
        s.box(6, y, 360, 84, t_, (sub,), role, show=sh, tsize=24, ssize=20)
        if i < 3:
            s.arrow([(186, y + 84), (186, y + 106)], GRY, show=k >= i + 2)
            s.arrow([(366, y + 42), (420, y + 42), (420, 190 if i < 2 else 262)],
                    RED, show=sh, dash=True)
    s.box(384, 192, 170, 68, "Admission", ("hold",), "fault", tsize=22, ssize=21)
    s.save(f"stages_step{k}.svg")


for k in range(1, 5):
    stages(k)

# ======================================================= Jensen-Shannon bars
jv = calcs.js(calcs.JS_BASE, calcs.JS_CAND)
fig, ax = plt.subplots(figsize=(5.4, 2.4))
for i, (name, dist) in enumerate((("Baseline", calcs.JS_BASE), ("Candidate", calcs.JS_CAND))):
    y = 1 - i
    ax.barh(y, dist[0] * 100, height=0.6, color=C["compute"])
    ax.barh(y, dist[1] * 100, left=dist[0] * 100, height=0.6, color=C["memory"])
    ax.text(dist[0] * 50, y, f"tests {dist[0]*100:.0f}%", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")
    ax.text(dist[0] * 100 + dist[1] * 50, y, f"lint {dist[1]*100:.0f}%", ha="center",
            va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(-2, y, name, ha="right", va="center", fontsize=15, fontweight="bold",
            color=C["text"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xticks([])
for sname in ("left", "bottom"):
    ax.spines[sname].set_visible(False)
ax.set_title(f"JS divergence = {jv:.3f} nats, {jv / math.log(2) * 100:.0f}% of ln 2",
             fontsize=15, fontweight="bold", color=C["text"])
sp.save(fig, out / "js_bars.svg")


# ============================================================ SPRT corridor
def _first_cross_seed(p, lo_n, hi_n):
    s3 = calcs.nm143()
    for seed in range(1, 500):
        rng = random.Random(seed)
        x, y = 0, 0.0
        while s3["lower"] < y < s3["upper"] and x < 70:
            y += s3["dn_success"] if rng.random() < p else s3["up_fail"]
            x += 1
        if y >= s3["upper"] and lo_n <= x <= hi_n:
            return seed
    return 1


RED_SEED = _first_cross_seed(0.75, 30, 50)

def sprt_fig(k):
    s3 = calcs.nm143()
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    xs = [0, 70]
    ax.fill_between(xs, s3["lower"], s3["upper"], color=C["neutral_l"], alpha=0.5)
    ax.axhline(s3["upper"], color=C["danger"], lw=3)
    ax.axhline(s3["lower"], color=C["data"], lw=3)
    ax.text(69, s3["upper"] + 0.25, f"hold: degraded ({s3['upper']:.2f})", ha="right",
            va="bottom", fontsize=14, fontweight="bold", color=C["danger"])
    ax.text(69, s3["lower"] - 0.25, f"select p0 = 0.90 ({s3['lower']:.2f})", ha="right",
            va="top", fontsize=14, fontweight="bold", color=C["data"])
    ax.text(3, 0.5, "keep sampling", fontsize=14, color=C["neutral"], fontweight="bold")
    if k >= 2:
        for p, ck, seed in ((0.75, "danger", RED_SEED), (0.90, "data", 3)):
            rng = random.Random(seed)
            x, y, xsw, ysw = 0, 0.0, [0], [0.0]
            while s3["lower"] < y < s3["upper"] and x < 70:
                ok = rng.random() < p
                y += s3["dn_success"] if ok else s3["up_fail"]
                x += 1
                xsw.append(x)
                ysw.append(y)
            ax.step(xsw, ysw, where="post", color=C[ck], lw=2.4)
            ax.plot([xsw[-1]], [ysw[-1]], "o", ms=9, color=C[ck], mec="white", mew=1.5)
        ax.text(3, 3.4, "illustrative walks", fontsize=13, color=C["neutral"],
                style="italic")
    ax.set_xlim(0, 70)
    ax.set_ylim(s3["lower"] - 1.4, s3["upper"] + 1.4)
    ax.set_xlabel("Labeled canary tasks", fontsize=14)
    ax.set_ylabel("Evidence for degradation", fontsize=14)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / f"sprt_step{k}.svg")


for k in (1, 2):
    sprt_fig(k)

# ===================================================== calendar time vs f
fig, ax = plt.subplots(figsize=(5.4, 3.4))
fs_ = [f / 100 for f in range(1, 101)]
ax.plot([100 * f for f in fs_], [calcs.nm143(f)["decide_h"] for f in fs_], color=VIO, lw=3)
for f, off in ((0.05, (10, 4)), (0.10, (10, 4)), (1.0, (-14, -16))):
    v = calcs.nm143(f)["decide_h"]
    ax.plot([100 * f], [v], "o", ms=10, color=C["compute"], mec="white", mew=1.5, zorder=5)
    ax.annotate(f"{100*f:.0f}%: {v:.1f} h", (100 * f, v), xytext=off,
                textcoords="offset points", fontsize=15, fontweight="bold",
                color=C["compute"], ha="left" if f < 1 else "right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 5, 10, 50, 100])
ax.set_xticklabels(["1", "5", "10", "50", "100"])
ax.set_ylim(0.75, 130)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.minorticks_off()
ax.set_xlabel("Canary share of eligible tasks, %", fontsize=14)
ax.set_ylabel("Hours to a decision", fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "calendar.svg")

# ================================================= which harm the fraction sets
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
e5, e100 = calcs.nm143(0.05)["excess"], calcs.nm143(1.0)["excess"]
ax = axes[0]
ax.bar([0, 1], [e5, e100], color=[C["compute"], C["neutral"]], width=0.6)
for i, v in enumerate((e5, e100)):
    ax.text(i, v + 0.4, f"{v:.0f}", ha="center", fontsize=15, fontweight="bold")
ax.set_title("Measured regression:\nexcess failures", fontsize=13, fontweight="bold")
ax.set_ylim(0, 14)
ax2 = axes[1]
ax2.bar([0, 1], [5, 100], color=[C["compute"], C["neutral"]], width=0.6)
for i, v in enumerate((5, 100)):
    ax2.text(i, v + 4, f"{v}%", ha="center", fontsize=15, fontweight="bold")
ax2.set_title("Unmeasured harm:\nexposure rate", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 125)
for a in axes:
    a.set_xticks([0, 1])
    a.set_xticklabels(["5% canary", "full"], fontsize=13)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
sp.save(fig, out / "harm_bars.svg")

# ======================================================= rollback intervals
s = Svg(900, 252)
marks = [(250, "detected", AMB, 210), (430, "admission stopped", RED, 210),
         (790, "last effect settled", GRN, 210)]
rng = random.Random(4)
for i in range(9):
    st0 = 40 + i * 42
    end = min(st0 + 120 + rng.random() * 260, 770)
    s.rect(st0, 8 + i * 18, end - st0, 13, P["tool"][0], P["tool"][1], rx=3, sw=1.6)
s.line(20, 182, 880, 182, "#334155", 2)
for x, lab, col, y in marks:
    s.line(x, 4, x, 190, col, 3, dash=(lab != "admission stopped"))
    s.text(x, y, lab, 20, col, bold=True)
s.text(450, 244, "in-flight candidate tasks keep their identities until settled", 20,
       "#475569", italic=True)
s.save("rollback.svg")

# ============================================= paired interval (AE NM 13.1)
pi = calcs.PAIRED
half = (pi["gain"] - pi["lo"]) * math.sqrt(pi["pairs"] / pi["need"])


def paired(name, show_217=True, size=(5.4, 2.4)):
    fig, ax = plt.subplots(figsize=size)
    ax.axvline(-pi["margin"], color=C["danger"], lw=2.5, ls="--")
    ax.text(-pi["margin"] - 0.3, 1.55, "margin\n−2 points", ha="right", va="center",
            fontsize=14, fontweight="bold", color=C["danger"])
    ax.plot([pi["lo"], pi["hi"]], [1, 1], color=VIO, lw=5)
    ax.plot([pi["gain"]], [1], "o", ms=11, color=VIO)
    ax.text(pi["hi"] + 0.3, 1, "150 pairs", va="center", fontsize=14, fontweight="bold",
            color=VIO)
    ax.text(pi["lo"] - 0.3, 0.72, f"{pi['lo']:.2f}", ha="right", fontsize=13,
            fontweight="bold", color=VIO)
    if show_217:
        ax.plot([pi["gain"] - half, pi["gain"] + half], [0.25, 0.25], color=C["neutral"],
                lw=5, alpha=0.6)
        ax.plot([pi["gain"]], [0.25], "o", ms=11, color=C["neutral"], alpha=0.6)
        ax.text(pi["gain"] + half + 0.3, 0.25, f"{pi['need']} pairs (planned)", va="center",
                fontsize=14, color=C["neutral"], fontweight="bold")
    ax.set_xlim(-8, 21)
    ax.set_ylim(-0.2, 2.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Candidate minus baseline acceptance (points)", fontsize=13)
    ax.tick_params(labelsize=12)
    sp.save(fig, out / name)


paired("paired.svg")

# ============================================ text-free thumbnails 260x100
def thumb(name, body):
    (out / name).write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 100" '
        'width="260" height="100"><rect width="260" height="100" fill="white"/>'
        + body + "</svg>\n")


# recall: a paired interval crossing a margin
thumb("thumb_recall.svg",
      f'<line x1="60" y1="10" x2="60" y2="90" stroke="{RED}" stroke-width="4" stroke-dasharray="8,6"/>'
      f'<line x1="40" y1="50" x2="230" y2="50" stroke="{VIO}" stroke-width="8"/>'
      f'<circle cx="110" cy="50" r="11" fill="{VIO}"/>')
# goals and takeaways
thumb("thumb_trace.svg",
      f'<rect x="8" y="8" width="244" height="12" fill="#cbd5e1"/>'
      f'<rect x="8" y="26" width="8" height="12" fill="{GRN}"/>'
      f'<rect x="16" y="42" width="60" height="12" fill="{SKY}" stroke="{VIO}" stroke-width="3"/>'
      f'<rect x="76" y="58" width="120" height="12" fill="{AMB}" stroke="{VIO}" stroke-width="3"/>'
      f'<rect x="76" y="74" width="40" height="12" fill="{AMB}"/>'
      f'<rect x="206" y="88" width="46" height="10" fill="{SKY}" stroke="{VIO}" stroke-width="3"/>')
thumb("thumb_retain.svg",
      f'<rect x="8" y="35" width="50" height="30" rx="4" fill="#ffffff" stroke="#94a3b8" stroke-width="3"/>'
      f'<rect x="88" y="35" width="60" height="30" rx="4" fill="#f3e8ff" stroke="{VIO}" stroke-width="3"/>'
      f'<line x1="58" y1="50" x2="88" y2="50" stroke="{GRY}" stroke-width="3"/>'
      + "".join(f'<rect x="185" y="{8 + i * 23}" width="66" height="17" rx="3" fill="{f}" stroke="{c}" stroke-width="2.5"/>'
                f'<line x1="148" y1="50" x2="185" y2="{16 + i * 23}" stroke="{c}" stroke-width="2"/>'
                for i, (f, c) in enumerate((("#fee2e2", RED), ("#fef3c7", AMB),
                                            ("#e0f2fe", SKY), ("#f1f5f9", "#94a3b8")))))
thumb("thumb_branch.svg",
      f'<rect x="10" y="10" width="12" height="22" fill="{GRN}"/><rect x="22" y="10" width="228" height="22" fill="{RED}"/>'
      f'<rect x="10" y="39" width="96" height="22" fill="{GRN}"/><rect x="106" y="39" width="144" height="22" fill="#94a3b8"/>'
      f'<rect x="10" y="68" width="240" height="22" fill="{GRN}"/>')
thumb("thumb_canary.svg",
      "".join(f'<rect x="{10 + i * 62}" y="34" width="52" height="32" rx="4" fill="{f}" stroke="{c}" stroke-width="3"/>'
              for i, (f, c) in enumerate((("#fef3c7", AMB), ("#e0f2fe", SKY), ("#fef3c7", AMB), ("#dcfce7", GRN))))
      + "".join(f'<line x1="{62 + i * 62}" y1="50" x2="{72 + i * 62}" y2="50" stroke="{GRY}" stroke-width="3"/>' for i in range(3))
      + f'<rect x="100" y="80" width="60" height="16" rx="3" fill="#fee2e2" stroke="{RED}" stroke-width="2.5"/>')
thumb("thumb_gate.svg",
      "".join(f'<rect x="{12 + i * 62}" y="30" width="48" height="40" rx="5" fill="{f}" stroke="{c}" stroke-width="3"/>'
              for i, (f, c) in enumerate((("#fee2e2", RED), ("#f1f5f9", "#94a3b8"),
                                          ("#f1f5f9", "#94a3b8"), ("#f1f5f9", "#94a3b8")))))
# next lecture: one record fanned into several learning artifacts
nx = f'<rect x="10" y="30" width="70" height="40" rx="5" fill="#f3e8ff" stroke="{VIO}" stroke-width="3"/>'
for i, (f, c) in enumerate((("#dcfce7", GRN), ("#fef3c7", AMB), ("#e0f2fe", SKY))):
    y = 8 + i * 31
    nx += (f'<line x1="80" y1="50" x2="170" y2="{y + 12}" stroke="{c}" stroke-width="2.5"/>'
           f'<rect x="170" y="{y}" width="80" height="24" rx="4" fill="{f}" stroke="{c}" stroke-width="3"/>')
thumb("thumb_next.svg", nx)

print("figures written")
