#!/usr/bin/env python3
"""Figures for Lecture 11 (Durable Execution), Volume III.

Diagrams are SVG in the Volume III agentic-stack palette (runtime violet,
tools amber, model sky blue, memory green, faults red, structure slate).
Plots use slides/assets/kit/slideplot.py. Every number drawn comes from
calcs.py, which names its source (DE = the chapter, Durable Execution):

  restarts          measured: data/resample.json (30 restarts from the prompt)
  lost_step1-2      DE Sec. 11.1 (what a crash erases), drawn for the slide
  fold_step1-3      DE Sec. 11.2.1, figure 1 (event sourcing), redrawn
  intent_step1-2    DE Sec. 11.3.2, figure 2 (crash position), redrawn
  crashrun_step1-2  measured: data/summary_Repair*.json (Temporal crash runs)
  acks              measured: data/ack_latency.json
  replica_step1-2   DE Sec. 11.3.3 (completeness rule), drawn for the slide
  groupcommit       measured: data/ack_latency.json (group commit)
  ckpt_step1-3      DE Sec. 11.4.1 (consistent recovery boundary), drawn
  snapshots         DE equation 3 with Sec. 11.4.2 inputs (illustrative)
  cadence           DE NM 11.1 and equation 5 (illustrative)
  resume_step1-3    DE Sec. 11.5 (five stages), drawn for the slide
  fence_step1-3     DE Sec. 11.5.1 (fencing tokens), drawn for the slide
  prefill           measured: data/resume_prefill.json
  feynman           Thinking Machines measurement (Literature.InferenceDeterminism), drawn
  replay_step1-2    DE Sec. 11.6, figure 3 (live vs replay), redrawn
  diverge_step1-2   DE Sec. 11.7 (first divergence, r18 example), drawn
  retention         DE NM 11.2 (illustrative)
  thumb_*           text-free thumbnails for the goals and takeaways (log,
                    intent, cadence, replay)

Run from this folder with the MLSysBook venv: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from deckvals import val  # noqa: E402

C = sp.C
plt = sp.plt
OUT = HERE

# Volume III palette (volumes/vol3/palette.md): fill, stroke, text
ROLE = {
    "model": ("#e0f2fe", "#0284c7", "#0369a1"),
    "memory": ("#dcfce7", "#16a34a", "#15803d"),
    "tool": ("#fef3c7", "#d97706", "#b45309"),
    "sys": ("#f3e8ff", "#7c3aed", "#6d28d9"),
    "fault": ("#fee2e2", "#dc2626", "#b91c1c"),
    "struct": ("#f8fafc", "#94a3b8", "#0f172a"),
    "white": ("#ffffff", "#94a3b8", "#0f172a"),
}
INK = "#0f172a"
GRAY = "#64748b"
FONT = "Helvetica, Arial, sans-serif"
MONO = "Courier New, Courier, monospace"
FLOOR = 22   # smallest SVG font in viewBox units for a 900-wide figure
VIOLET, RED, BLUE, AMBER, GREEN = (ROLE["sys"][1], ROLE["fault"][1], ROLE["model"][1],
                                   ROLE["tool"][1], ROLE["memory"][1])


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def save(self, name):
        markers = "".join(
            f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{col}"/></marker>'
            for k, col in (("blue", BLUE), ("violet", VIOLET), ("amber", AMBER),
                           ("green", GREEN), ("red", RED), ("gray", GRAY), ("ink", INK)))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" font-family="{FONT}">'
               f'<defs>{markers}<pattern id="hatch" width="10" height="10" '
               f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               f'<line x1="0" y1="0" x2="0" y2="10" stroke="#fca5a5" stroke-width="4"/>'
               f'</pattern></defs><rect x="0" y="0" width="{self.w}" height="{self.h}" '
               f'fill="white"/>' + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)

    def rect(self, x, y, w, h, role="struct", dash=False, rx=8, sw=2, fill=None, stroke=None,
             opacity=1.0):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill or f}" stroke="{stroke or s}" stroke-width="{sw}"{d} '
                 f'opacity="{opacity}"/>')

    def text(self, x, y, s, fs=22, color=INK, anchor="middle", weight="normal", mono=False,
             italic=False, opacity=1.0, floor=None):
        fs = max(fs, floor if floor is not None else FLOOR)
        fam = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{fam}{it} '
                 f'opacity="{opacity}">{s}</text>')

    def box(self, x, y, w, h, role, title, sub=None, fs=24, sfs=22, dash=False, mono_sub=False,
            title_color=None, sw=2, mono_title=False, opacity=1.0):
        self.rect(x, y, w, h, role, dash=dash, sw=sw, opacity=opacity)
        _, _, tc = ROLE[role]
        if sub:
            self.text(x + w / 2, y + h / 2 - 5, title, fs, title_color or tc, weight="bold",
                      mono=mono_title, opacity=opacity)
            self.text(x + w / 2, y + h / 2 + sfs + 1, sub, sfs, GRAY, mono=mono_sub,
                      opacity=opacity)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, title, fs, title_color or tc,
                      weight="bold", mono=mono_title, opacity=opacity)

    def arrow(self, x1, y1, x2, y2, color="gray", dash=False, sw=2.5, path=None):
        col = {"blue": BLUE, "violet": VIOLET, "amber": AMBER, "green": GREEN, "red": RED,
               "gray": GRAY, "ink": INK}[color]
        d = ' stroke-dasharray="8 6"' if dash else ""
        geom = path or f"M{x1},{y1} L{x2},{y2}"
        self.add(f'<path d="{geom}" fill="none" stroke="{col}" stroke-width="{sw}"{d} '
                 f'marker-end="url(#a{color})"/>')

    def line(self, x1, y1, x2, y2, color=GRAY, sw=2, dash=False):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}"{d}/>')

    def cross(self, x, y, r=11, color=None, sw=5):
        col = color or RED
        self.add(f'<path d="M{x-r},{y-r} L{x+r},{y+r} M{x-r},{y+r} L{x+r},{y-r}" '
                 f'stroke="{col}" stroke-width="{sw}"/>')

    def check(self, x, y, r=12, color=None):
        col = color or GREEN
        self.add(f'<path d="M{x-r},{y} L{x-r/3},{y+r*0.7} L{x+r},{y-r*0.8}" fill="none" '
                 f'stroke="{col}" stroke-width="5"/>')

    def dot(self, x, y, r=9, color=INK):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def num(self, x, y, k, color=None):
        col = color or VIOLET
        self.add(f'<circle cx="{x}" cy="{y}" r="16" fill="{col}"/>')
        self.text(x, y + 7, str(k), 21, "white", weight="bold")


# =====================================================================
# Hook
# =====================================================================
def restarts():
    """Column figure: first steps chosen by 30 restarts from the prompt."""
    r = calcs.resample()
    order = [("read_file", "read a file"), ("grep", "search the code"),
             ("run_tests", "run the tests")]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ys = np.arange(len(order))[::-1]
    for y, (key, lab) in zip(ys, order):
        v = r["by_tool"].get(key, 0)
        ax.barh(y, v, color=C["memory"] if key != "run_tests" else C["danger"], height=0.6)
        ax.text(v + 0.4, y, f"{v}", va="center", fontsize=16, fontweight="bold",
                color=C["memory"] if key != "run_tests" else C["danger"])
        ax.text(-0.4, y, lab, va="center", ha="right", fontsize=15, fontweight="bold",
                color=C["text"])
    ax.set_yticks([])
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f"Restarts, of {r['n']}", fontsize=14)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{r['distinct']} distinct first calls", fontsize=15, color=C["text"],
                 fontweight="bold", loc="left")
    sp.save(fig, OUT / "restarts.svg")


# =====================================================================
# Segment 1: the log
# =====================================================================
def lost():
    """What one crash erases: worker memory vs the context window vs a log."""
    for k in (1, 2):
        s = Svg(900, 470)
        # worker
        s.rect(20, 20, 520, 300, "struct", rx=12, sw=2, dash=False)
        s.text(40, 56, "Worker process (memory)", 24, INK, anchor="start", weight="bold")
        items = [("test_r18 dispatched, no result", "tool"),
                 (f"${val('res_usd')} reserved for the test", "sys"),
                 ("grants and control_epoch", "sys"),
                 ("approval pending for r19", "sys")]
        for i, (lab, role) in enumerate(items):
            s.rect(40, 76 + i * 56, 480, 46, role, rx=6, sw=1.5, opacity=1.0 if k == 1 else 0.35)
            s.text(56, 107 + i * 56, lab, 23, INK, anchor="start",
                   opacity=1.0 if k == 1 else 0.35)
        # context window
        s.rect(580, 20, 300, 300, "model", rx=12, sw=2)
        s.text(600, 56, "Context window", 24, ROLE["model"][2], anchor="start", weight="bold")
        for i, lab in enumerate(("prompt", "turns 1 to 17,", "compacted", "last observation")):
            s.text(600, 108 + i * 40, lab, 23, INK, anchor="start")
        s.text(600, 290, "no operations, no budget", 21, GRAY, anchor="start")
        if k >= 2:
            s.cross(280, 190, 70, RED, sw=10)
            s.text(280, 370, "a crash erases all of it", 25, ROLE["fault"][2],
                   weight="bold")
            s.text(730, 370, "a transcript, not a record", 23, ROLE["model"][2],
                   weight="bold")
            s.rect(20, 400, 860, 56, "sys", rx=10, sw=2.5)
            s.text(450, 437, "Only what reached durable storage survives: the trajectory log",
                   23, ROLE["sys"][2], weight="bold")
        s.save(f"lost_step{k}")


def fold():
    """Event sourcing: the log is folded into the record (DE figure 1)."""
    ev = [("TaskStarted", None), ("ContextBuilt", None), ("ModelReplied", None),
          ("Authorized", "+ reserve"), ("ToolIntent", "test_r18")]
    for k in (1, 2, 3):
        s = Svg(900, 470)
        # log at the bottom
        s.rect(10, 286, 880, 178, "white", rx=12, stroke="#94a3b8", sw=2)
        s.text(28, 318, "Trajectory log: durable, append-only", 24, INK, anchor="start",
               weight="bold")
        x = 20
        w = 148
        for i, (a, b) in enumerate(ev):
            s.rect(x + i * w, 366, w, 84, "sys", rx=0, sw=2)
            s.text(x + i * w + w / 2, 356, f"e{i + 1}", 22, VIOLET, weight="bold")
            if b:
                s.text(x + i * w + w / 2, 400, a, 22, INK)
                s.text(x + i * w + w / 2, 430, b, 22, INK, mono=(b == "test_r18"))
            else:
                s.text(x + i * w + w / 2, 416, a, 22, INK)
        s.line(768, 340, 768, 456, VIOLET, 4)
        s.rect(776, 366, 108, 84, "white", dash=True, rx=6)
        s.text(830, 400, "result", 22, GRAY)
        s.text(830, 430, "missing", 22, GRAY)
        if k >= 2:
            s.box(330, 40, 250, 120, "struct", "Reducer", fs=26)
            s.text(455, 128, "apply(R, e)", 23, INK, mono=True)
            s.text(455, 196, "no model call, no tool call", 22, GRAY)
            s.arrow(455, 284, 455, 210, "violet", sw=3.5)
            s.text(470, 258, "fold, in log order", 22, VIOLET, anchor="start")
            s.box(20, 50, 240, 100, "memory", "R0", "task, grants, limits", fs=26, sfs=22)
            s.arrow(262, 100, 328, 100, "gray")
        if k >= 3:
            s.rect(650, 30, 240, 200, "memory", rx=10, sw=2)
            s.text(770, 62, "Rebuilt record", 24, ROLE["memory"][2], weight="bold")
            s.text(770, 104, "admission held", 22, INK)
            s.text(770, 140, f"${val('res_usd')} still reserved", 22, INK)
            s.rect(664, 162, 212, 50, "fault", dash=True, rx=6)
            s.text(770, 195, "test_r18 unresolved", 22, ROLE["fault"][2], weight="bold")
            s.arrow(582, 100, 648, 100, "gray")
        s.save(f"fold_step{k}")


# =====================================================================
# Segment 2: intent before dispatch
# =====================================================================
def intent():
    """Durable intent before dispatch, and what each crash position proves."""
    chain = [("propose", "model reply", "sys"), ("admit", "grant, reserve", "sys"),
             ("intent ack", "durable", "sys"), ("dispatch", "request leaves", "tool"),
             ("evidence", "result logged", "sys")]
    for k in (1, 2):
        s = Svg(900, 460)
        w, gap, y = 158, 22, 66
        for i, (a, b, role) in enumerate(chain):
            x = 10 + i * (w + gap)
            s.box(x, y, w, 96, role, a, b, fs=25, sfs=22)
            if i < 4:
                s.arrow(x + w + 1, y + 48, x + w + gap - 1, y + 48,
                        "amber" if i == 2 else "gray")
        s.line(539, 42, 539, 300 if k == 1 else 440, VIOLET, 3.5, dash=True)
        s.text(539, 30, "durability barrier", 22, VIOLET, weight="bold")
        s.line(10, 216, 890, 216, GRAY, 2)
        s.arrow(860, 216, 890, 216, "gray")
        s.text(890, 244, "time", 22, GRAY, anchor="end")
        s.text(268, 198, "runtime can still withhold", 22, GRAY)
        s.text(712, 198, "request may have left", 22, GRAY)
        if k >= 2:
            zones = [(10, 439, "struct", "Not dispatched", "absence proves it only",
                      "in a complete log"),
                     (449, 281, "fault", "In doubt", "reconcile before", "dependent work"),
                     (730, 160, "memory", "Established", "per endpoint", "contract")]
            for x, w2, role, a, b, c in zones:
                s.rect(x, 262, w2, 180, role, rx=0, sw=2.5)
                s.text(x + w2 / 2, 312, a, 25, ROLE[role][2], weight="bold")
                s.text(x + w2 / 2, 362, b, 22, INK)
                s.text(x + w2 / 2, 394, c, 22, INK)
        s.save(f"intent_step{k}")


def crashrun():
    """The two measured Temporal runs as lanes on one time axis."""
    c = calcs.crash_runs()
    for k in (1, 2):
        s = Svg(900, 420)
        x0, sc = 250, 140     # px per second; 0 to 4.5 s shown
        lanes = [("RepairNaive", "Unkeyed test", 130), ("RepairKeyed", "Keyed test", 275)]
        s.line(x0, 380, x0 + 4.6 * sc, 380, GRAY, 2)
        for t in range(0, 5):
            s.line(x0 + t * sc, 376, x0 + t * sc, 386, GRAY, 2)
            s.text(x0 + t * sc, 412, f"{t} s", 22, GRAY)
        for name, lab, y in lanes:
            run = c[name]
            s.text(10, y + 8, lab, 22, INK, anchor="start", weight="bold")
            s.line(x0, y, x0 + 4.6 * sc, y, "#cbd5e1", 3)
            for ev, job, att, t in run["ledger"]:
                if ev == "model_call":
                    continue
                if att == 1:
                    s.rect(x0 + t * sc - 6, y - 26, 26, 52, "tool", rx=4)
                    s.text(x0 + t * sc + 7, y + 52, "job_tests_1", 22, ROLE["tool"][2])
                    s.text(x0 + t * sc + 7, y + 78, "launched", 22, ROLE["tool"][2])
                elif k >= 2:
                    if ev == "job_launched":
                        s.rect(x0 + t * sc - 6, y - 26, 26, 52, "fault", rx=4)
                        s.text(x0 + t * sc + 24, y - 36, f"{job} launched again", 22,
                               ROLE["fault"][2], anchor="end", weight="bold")
                    else:
                        s.rect(x0 + t * sc - 6, y - 26, 26, 52, "memory", rx=4)
                        s.text(x0 + t * sc + 24, y - 36, "job_tests_1 found, reused", 22,
                               ROLE["memory"][2], anchor="end", weight="bold")
        # crash_run.py kills the worker 1.0 s after the test job is launched
        # (it waits for the launch marker, then sleeps 1.0 s)
        kx = x0 + 1.0 * sc
        s.line(kx, 46, kx, 365, RED, 3, dash=True)
        s.text(kx, 36, "worker killed (SIGKILL)", 22, ROLE["fault"][2], weight="bold")
        if k == 1:
            s.text(x0 + 2.75 * sc, 210, "new worker: what runs?", 24, INK,
                   weight="bold")
        s.save(f"crashrun_step{k}")


def acks():
    a = calcs.acks()
    rows = [("write()", a["write"], "neutral", f"{val('ack_write')} ms"),
            ("fsync()", a["fsync"], "compute", f"{val('ack_fsync')} ms"),
            ("F_FULLFSYNC", a["full"], "danger", f"{val('ack_full')} ms")]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    n = len(rows)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v * 1.25, y, txt, va="center", fontsize=15, fontweight="bold", color=C[ck])
        ax.text(0.0045, y, lab, va="center", ha="right", fontsize=15, fontweight="bold",
                color=C["text"], family="monospace")
    ax.set_xscale("log")
    ax.set_xlim(0.005, 60)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.tick_params(labelsize=13)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Median time to return, ms (log scale)", fontsize=13)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "acks.svg")


def replica():
    """Completeness rule: a lagging replica cannot prove non-dispatch."""
    for k in (1, 2):
        s = Svg(900, 420)
        s.text(10, 40, "Authoritative log", 24, INK, anchor="start", weight="bold")
        s.text(10, 200, "Asynchronous replica", 24, INK, anchor="start", weight="bold")
        for i in range(7):
            x = 10 + i * 100
            s.rect(x, 60, 92, 70, "sys", rx=4)
            s.text(x + 46, 104, f"e{i + 1}", 23, VIOLET, weight="bold")
            if i < 6:
                s.rect(x, 220, 92, 70, "sys", rx=4)
                s.text(x + 46, 264, f"e{i + 1}", 23, VIOLET, weight="bold")
        s.rect(610, 60, 92, 70, "tool", rx=4, sw=3)
        s.text(656, 104, "intent", 22, ROLE["tool"][2], weight="bold")
        s.rect(610, 220, 92, 70, "white", dash=True, rx=4)
        s.text(656, 264, "?", 26, GRAY, weight="bold")
        s.arrow(656, 134, 656, 214, "gray", dash=True)
        s.text(670, 182, "still in flight", 22, GRAY, anchor="start")
        s.text(720, 104, "test_r18 sent", 22, ROLE["tool"][2], anchor="start")
        if k >= 2:
            s.rect(10, 316, 880, 92, "fault", rx=8)
            s.text(450, 352, "Reading the replica: “no intent, so never sent”",
                   24, ROLE["fault"][2], weight="bold")
            s.text(450, 388, "Dispatching again launches a second test", 23, INK)
        s.save(f"replica_step{k}")


def groupcommit():
    a = calcs.acks()
    ks = sorted(a["group"])
    per = [a["group"][k]["per_record_ms"] for k in ks]
    flush = [a["group"][k]["flush_ms_median"] for k in ks]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(ks, flush, "o-", color=C["danger"], lw=2.4, ms=7, mec="white")
    ax.plot(ks, per, "o-", color=C["memory"], lw=2.4, ms=7, mec="white")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks([0.05, 0.1, 0.5, 1, 5])
    ax.set_yticklabels(["0.05", "0.1", "0.5", "1", "5"])
    ax.minorticks_off()
    ax.set_ylim(0.03, 9)
    ax.tick_params(labelsize=12)
    ax.text(ks[-1], flush[-1] * 1.35, "every record waits the flush", ha="right",
            fontsize=12.5, fontweight="bold", color=C["danger"])
    ax.text(ks[0], 0.075, f"storage work per record:\n{val('gc_per_hi')} ms to {val('gc_per_lo')} ms",
            ha="left", va="center", fontsize=12.5, fontweight="bold", color=C["memory"])
    ax.set_xlabel("Records per flush (batch size)")
    ax.set_ylabel("ms")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "groupcommit.svg")


# =====================================================================
# Segment 3: checkpoints and resume
# =====================================================================
def ckpt():
    """A checkpoint binds record and workspace to one log position c."""
    for k in (1, 2, 3):
        s = Svg(900, 450)
        # the log line
        s.text(10, 40, "Log", 24, INK, anchor="start", weight="bold")
        for i in range(13):
            x = 70 + i * 62
            s.rect(x, 18, 54, 40, "sys" if i <= 7 else "white", rx=3, sw=1.5,
                   stroke=None if i <= 7 else VIOLET)
        cx = 70 + 7 * 62 + 58
        s.line(cx, 6, cx, 300, VIOLET, 4)
        s.text(cx, 330, "position c", 23, VIOLET, weight="bold")
        s.text(cx + 214, 92, "suffix: folded at restore", 22, VIOLET)
        s.arrow(cx + 8, 70, cx + 330, 70, "violet")
        # manifest and its parts
        s.box(40, 110, 300, 90, "memory", "record_R_ckpt", "reservations, operations",
              fs=24, sfs=22, mono_title=True)
        s.box(40, 220, 300, 90, "tool", "ws_r18", "workspace snapshot", fs=24, sfs=22,
              mono_title=True)
        s.arrow(342, 155, cx - 6, 155, "gray")
        s.arrow(342, 265, cx - 6, 265, "gray")
        if k >= 2:
            s.rect(380, 110, 150, 200, "white", rx=8, sw=2)
            s.text(455, 146, "manifest", 23, INK, weight="bold")
            for i, lab in enumerate(("position c", "hashes", "reducer v3")):
                s.text(455, 190 + i * 36, lab, 22, GRAY)
            s.num(612, 210, 1)
            s.text(640, 202, "bytes durable first,", 22, INK, anchor="start")
            s.text(640, 232, "then publish manifest", 22, INK, anchor="start")
        if k >= 3:
            s.rect(40, 360, 820, 80, "fault", rx=8)
            s.cross(80, 400, 16)
            s.text(110, 392, "Torn checkpoint: record at c, files copied seconds later", 23,
                   ROLE["fault"][2], anchor="start", weight="bold")
            s.text(110, 424, "restores a state that never existed", 22, INK, anchor="start")
        s.save(f"ckpt_step{k}")


def snapshots():
    s = calcs.snapshots()
    ks, full, delta = calcs.snapshot_curves()
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    ax.plot(ks, full, color=C["danger"], lw=2.8)
    ax.plot(ks, delta, color=C["neutral"], lw=2.8)
    ax.text(3, 3.2, f"full snapshots: {val('sn_full')} GB", ha="left",
            va="center", fontsize=13, fontweight="bold", color=C["danger"])
    ax.text(ks[-1], delta[-1] + 0.15, f"deltas: {val('sn_delta')} MB", ha="right",
            va="bottom", fontsize=13, fontweight="bold", color=C["neutral"])
    ax.set_xlabel("Checkpoints taken (one every 5 turns)")
    ax.set_ylabel("Bytes written (GB)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.6)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "snapshots.svg")


def cadence():
    k = calcs.cadence()
    st = np.logspace(0, np.log10(80), 300)
    log_c = [calcs.expected_overhead_curve(x, calcs.PER_TURN) for x in st]
    unl_c = [calcs.expected_overhead_curve(x, calcs.REEXEC) for x in st]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(st, unl_c, color=C["danger"], lw=2.6)
    ax.plot(st, log_c, color=C["memory"], lw=2.6)
    for x, curve_per, ck in ((k["s_unlog"], calcs.REEXEC, "danger"),
                             (k["s_log"], calcs.PER_TURN, "memory")):
        y = calcs.expected_overhead_curve(x, curve_per)
        ax.plot([x], [y], "o", ms=10, color=C[ck], mec="white", mew=1.5, zorder=5)
    ax.text(0.03, 0.96, f"turns re-executed: optimum {val('cd_k_unlog')}", transform=ax.transAxes,
            va="top", fontsize=13.5, fontweight="bold", color=C["danger"])
    ax.annotate(f"turns logged: optimum {val('cd_k_log')}", (k["s_log"],
                calcs.expected_overhead_curve(k["s_log"], calcs.PER_TURN)),
                xytext=(0, -32), textcoords="offset points", ha="center", fontsize=13.5,
                fontweight="bold", color=C["memory"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 25, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "50"])
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_ylim(0.008, 30)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Checkpoint stride (turns)", fontsize=14)
    ax.set_ylabel("Overhead (s per turn)", fontsize=14)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "cadence.svg")


def resume():
    """Resume in stages: prove, own, restore, reconcile, admit."""
    stages = [("Prove", "full log", "struct"), ("Own", "epoch 7 to 8", "sys"),
              ("Restore", "fold + ws_r18", "memory"), ("Reconcile", "test_r18", "tool"),
              ("Admit", "new work", "model")]
    show = {1: 2, 2: 4, 3: 5}
    for k in (1, 2, 3):
        s = Svg(900, 330)
        w, gap = 164, 17
        for i, (a, b, role) in enumerate(stages):
            x = 8 + i * (w + gap)
            on = i < show[k]
            s.box(x, 40, w, 110, role, a, b, fs=27, sfs=21, opacity=1.0 if on else 0.25,
                  title_color=INK if role == "struct" else None)
            s.num(x + 20, 40, i + 1, VIOLET if on else "#cbd5e1")
            if i < 4:
                s.arrow(x + w + 1, 95, x + w + gap - 1, 95, "gray")
        cx = [8 + i * (w + gap) + w / 2 for i in range(5)]
        notes = [("replica lags:", "say unavailable"), ("stale writes", "are rejected"),
                 ("no model call,", "no tool call"), ("probe by id,", "never resend"),
                 ("recheck limits,", "then prefill")]
        for i, (a, b) in enumerate(notes):
            if i < show[k]:
                s.text(cx[i], 200, a, 22, GRAY)
                s.text(cx[i], 230, b, 22, GRAY)
        if k >= 3:
            s.rect(8, 268, 884, 54, "sys", rx=8)
            s.text(450, 304, "Missing evidence or ownership: hold the trajectory, do not guess",
                   23, ROLE["sys"][2], weight="bold")
        s.save(f"resume_step{k}")


def fence():
    """Fencing: the resource checks the owner epoch on every write."""
    for k in (1, 2, 3):
        s = Svg(900, 400)
        s.text(10, 48, "Worker A", 24, INK, anchor="start", weight="bold")
        s.text(10, 168, "Worker B", 24, INK, anchor="start", weight="bold")
        s.text(10, 310, "Log or gateway", 24, INK, anchor="start", weight="bold")
        s.text(10, 340, "(checks epoch)", 22, GRAY, anchor="start")
        s.line(200, 40, 890, 40, "#cbd5e1", 3)
        s.line(200, 160, 890, 160, "#cbd5e1", 3)
        s.rect(200, 280, 690, 70, "sys", rx=8)
        s.box(210, 18, 150, 44, "sys", "epoch 7", fs=22)
        s.rect(380, 22, 260, 36, "fault", dash=True, rx=6)
        s.text(510, 48, "stalled, lease expires", 22, ROLE["fault"][2])
        if k >= 2:
            s.box(500, 138, 150, 44, "sys", "epoch 8", fs=22)
            s.arrow(575, 184, 575, 276, "violet", sw=3)
            s.text(562, 236, "write, epoch 8: accepted", 22, ROLE["sys"][2], anchor="end")
            s.check(575, 316, 14)
        if k >= 3:
            s.arrow(740, 62, 740, 276, "red", sw=3)
            s.text(752, 110, "late write,", 22, ROLE["fault"][2], anchor="start")
            s.text(752, 136, "epoch 7", 22, ROLE["fault"][2], anchor="start")
            s.cross(740, 316, 14)
            s.text(766, 324, "rejected", 22, ROLE["fault"][2], anchor="start", weight="bold")
        s.save(f"fence_step{k}")


def prefill():
    p = calcs.prefill()
    rows = [("Same server,\nprefix held", p["warm"], "memory", f"{val('pf_warm')} s"),
            ("Model reloaded,\nfull prefill", p["cold"], "danger", f"{val('pf_cold')} s")]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v + 0.15, y, txt, va="center", fontsize=15, fontweight="bold", color=C[ck])
        ax.text(-0.15, y, lab, va="center", ha="right", fontsize=13.5, fontweight="bold",
                color=C["text"])
    ax.set_xlim(0, 11)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(f"Time to the first token of the next call, s ({val('pf_tokens')} tokens)",
                  fontsize=12)
    sp.save(fig, OUT / "prefill.svg")


# =====================================================================
# Segment 4: replay, divergence, retention
# =====================================================================
def feynman():
    """1,000 temperature-0 completions split at token 103 (Thinking Machines).
    Drawn for a column slot (about 7.7 by 3.4 cm)."""
    s = Svg(600, 270)
    s.rect(4, 82, 236, 106, "model", rx=8)
    s.text(122, 116, f"tokens 1 to {val('tm_prefix')}", 24, ROLE["model"][2], weight="bold")
    s.text(122, 146, "identical", 24, ROLE["model"][2], weight="bold")
    s.text(122, 174, "\u201c...1918, in\u201d", 22, INK)
    s.arrow(242, 118, 318, 66, "gray", sw=3)
    s.arrow(242, 152, 318, 204, "gray", sw=3)
    s.rect(322, 10, 274, 104, "memory", rx=8)
    s.text(459, 50, "\u201cQueens, New York\u201d", 24, ROLE["memory"][2], weight="bold")
    s.text(459, 88, f"{val('tm_major')} of {val('tm_n')}", 24, INK)
    s.rect(322, 156, 274, 104, "fault", rx=8)
    s.text(459, 196, "\u201cNew York City\u201d", 24, ROLE["fault"][2], weight="bold")
    s.text(459, 234, f"{val('tm_minor')} of {val('tm_n')}", 24, INK)
    s.save("feynman")


def replay():
    """Live execution vs replay (DE figure 3), redrawn for the slide."""
    nodes = [("Harness", "records", "struct"), ("Model", "sampled", "model"),
             ("Gate", "authorizes", "struct"), ("Tools", "real effects", "tool")]
    log = ["ContextBuilt", "ModelReplied", "ToolIntent", "ToolResult"]
    global FLOOR
    FLOOR = 26   # shown height-limited beside nothing; larger type
    for k in (1, 2):
        s = Svg(900, 490)
        w = 190
        xs = [10 + i * 230 for i in range(4)]
        s.text(10, 30, "Live", 24, INK, anchor="start", weight="bold")
        for i, (a, b, role) in enumerate(nodes):
            s.box(xs[i], 44, w, 84, role, a, b, fs=28, sfs=26,
                  title_color=INK if role == "struct" else None)
            if i < 3:
                s.arrow(xs[i] + w + 1, 86, xs[i + 1] - 1, 86, "gray")
            s.arrow(xs[i] + w / 2, 130, xs[i] + w / 2, 160, "violet")
        for i, lab in enumerate(log):
            s.rect(xs[i], 162, w + (40 if i < 3 else 0), 44, "sys", rx=0)
            s.text(xs[i] + w / 2, 194, lab, 26, INK)
        if k >= 2:
            s.line(10, 232, 890, 232, "#cbd5e1", 2)
            s.text(10, 270, "Replay", 24, INK, anchor="start", weight="bold")
            for i, lab in enumerate(log):
                s.rect(xs[i], 284, w + (40 if i < 3 else 0), 44, "sys", rx=0)
                s.text(xs[i] + w / 2, 316, lab, 26, INK)
            states = [("Harness", "replaying", "struct", False), ("Model", "stubbed", "white", True),
                      ("Gate", "authorizes", "struct", False), ("Tools", "disconnected", "white", True)]
            for i, (a, b, role, dash) in enumerate(states):
                s.box(xs[i], 396, w, 84, role, a, b, fs=28, sfs=26, dash=dash,
                      title_color=GRAY if dash else INK)
            s.arrow(0, 0, 0, 0, "violet", sw=3, path="M780,330 L780,346 L105,346 L105,394")
            s.text(240, 382, "recorded replies stream in, in order", 26, VIOLET,
                   anchor="start")
            s.line(xs[2] + w + 4, 438, xs[3] - 6, 438, RED, 4)
            s.line(xs[3] - 6, 420, xs[3] - 6, 456, RED, 5)
        s.save(f"replay_step{k}")
    FLOOR = 22


def diverge():
    """Two runs compared boundary by boundary; the first mismatch localizes."""
    bounds = ["context", "model", "decision", "tool"]
    for k in (1, 2):
        s = Svg(900, 400)
        s.text(10, 120, "Original", 24, INK, anchor="start", weight="bold")
        s.text(10, 240, "Replay", 24, INK, anchor="start", weight="bold")
        s.text(10, 270, "new harness", 22, GRAY, anchor="start")
        x0 = 180
        cols = []
        for t, turn in enumerate(("turn 18", "turn 19")):
            s.text(x0 + t * 350 + 160, 40, turn, 23, GRAY, weight="bold")
            for j, b in enumerate(bounds):
                x = x0 + t * 350 + j * 82
                cols.append((t, j, x))
                s.text(x + 36, 78, b, 21, GRAY, floor=21)
                s.rect(x, 96, 72, 40, "sys", rx=4)
        for t, j, x in cols:
            idx = t * 4 + j
            first = (t == 1 and j == 0)
            if idx < 4:
                role = "sys"
            elif first:
                role = "fault"
            else:
                role = "white"
            s.rect(x, 216, 72, 40, role, rx=4, dash=(role == "white"))
            if idx < 4 or k >= 1 and idx < 4:
                s.check(x + 36, 182, 11)
        if k >= 1:
            s.rect(x0 + 350, 216, 72, 40, "fault", rx=4, sw=3)
            s.cross(x0 + 350 + 36, 182, 11)
        if k >= 2:
            s.text(x0 + 330, 306, "first mismatch: compaction dropped", 22,
                   ROLE["fault"][2], weight="bold", anchor="start")
            s.text(x0 + 330, 336, "“regression suite omitted”", 22,
                   ROLE["fault"][2], anchor="start")
            s.text(x0 + 350 + 164 + 36, 190, "may be consequences", 22, GRAY)
        s.save(f"diverge_step{k}")


def retention():
    t = calcs.retention()
    rows = [(f"Log ({val('rt_turns')} turns)", t["log_mb"], "network", f"{val('rt_log')} MB"),
            ("Workspace snapshot", t["snap_mb"], "compute", f"{val('rt_ws')} MB")]
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v + 1, y, txt, va="center", fontsize=14, fontweight="bold", color=C[ck])
        ax.text(-1, y, lab, va="center", ha="right", fontsize=14, fontweight="bold",
                color=C["text"])
    ax.set_xlim(0, 65)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Bytes kept for recovery and replay (MB)", fontsize=12)
    sp.save(fig, OUT / "retention.svg")


# =====================================================================
# Text-free thumbnails (goals, takeaways): 260 x 100 slots
# =====================================================================
def thumbs():
    # log folded into a record
    s = Svg(260, 100)
    for i in range(5):
        s.rect(6 + i * 34, 60, 30, 30, "sys", rx=2, sw=1.5)
    s.arrow(90, 56, 90, 34, "violet", sw=3)
    s.rect(60, 4, 60, 28, "memory", rx=4, sw=1.5)
    s.rect(190, 60, 30, 30, "white", dash=True, rx=2, sw=1.5)
    s.save("thumb_log")
    # intent before dispatch
    s = Svg(260, 100)
    for i in range(5):
        s.rect(6 + i * 52, 30, 40, 40, "tool" if i == 3 else "sys", rx=4, sw=1.5)
        if i < 4:
            s.arrow(47 + i * 52, 50, 56 + i * 52, 50, "gray", sw=2)
    s.line(152, 8, 152, 92, VIOLET, 3, dash=True)
    s.save("thumb_intent")
    # checkpoint cadence U-curve
    s = Svg(260, 100)
    pts = " ".join(f"{20 + i * 11},{82 - 0.68 * (i - 10) ** 2:.1f}" for i in range(21))
    s.add(f'<polyline points="{pts}" fill="none" stroke="{GREEN}" stroke-width="4"/>')
    s.dot(130, 82, 7, GREEN)
    s.line(10, 94, 250, 94, GRAY, 2)
    s.save("thumb_cadence")
    # replay and first mismatch
    s = Svg(260, 100)
    for j in range(6):
        s.rect(8 + j * 41, 14, 34, 28, "sys", rx=3, sw=1.5)
        s.rect(8 + j * 41, 58, 34, 28, "sys" if j < 3 else ("fault" if j == 3 else "white"),
               rx=3, sw=1.5, dash=(j > 3))
    s.save("thumb_replay")


if __name__ == "__main__":
    restarts()
    lost()
    fold()
    intent()
    crashrun()
    acks()
    replica()
    groupcommit()
    ckpt()
    snapshots()
    cadence()
    resume()
    fence()
    prefill()
    feynman()
    replay()
    diverge()
    retention()
    thumbs()
    print("figures written")
