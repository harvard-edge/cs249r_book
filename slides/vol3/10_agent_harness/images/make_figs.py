#!/usr/bin/env python3
"""Figures for Lecture 10 (The Agent Harness), Volume III.

Diagrams are SVG in the Volume III agentic-stack palette (runtime violet,
tools amber, model sky blue, memory green, faults red, structure slate).
Plots use slides/assets/kit/slideplot.py. Every number drawn comes from
calcs.py (which names its source); diagrams that carry no data are
schematic redraws of the chapter's figures (AH = The Agent Harness):

  compound_step1-2   AH Sec. 10.10 figure (0.99^H) plus measured median
                     calls of five public repair agents (MLSysIM)
  harness_step1-2    AH Sec. 10.2, figure 1 (prototype loop vs harness), redrawn
  turn_step1-4       AH Sec. 10.2.2, table 1 (one supervised turn of r18)
  record_fields      AH Sec. 10.3, table 2 (field groups), drawn for a column
  states_step1-3     AH Sec. 10.4, figure 2 (trajectory state machine), redrawn
  late_step1-3       AH Sec. 10.4.2 (settlement vs observation staging)
  pause_step1-3      measured: data/pause_latency.json; AH Sec. 10.5.1 reference
  escrow_step1-3     AH Sec. 10.7, figure 3 (approval gate), redrawn
  drr_step1-2        AH NM 10.1 (illustrative; simulated in calcs)
  runaway_step1-3    AH Sec. 10.9.1, figure 4 and table 9, redrawn
  admit_step1-2      AH NM 10.2 (illustrative)
  overshoot          measured: one public mini-SWE-agent trajectory (data/)
  meter_step1-2      AH NM 10.3 (illustrative)
  ceiling_cdf        measured: AH figure 5 data, one configuration, column size
  ceiling_bars       measured: AH figure 5 data, five configurations
  osworld            measured: AH figure 6 data (Xie et al. 2024 via Epoch AI)

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
from deckvals import val, num  # noqa: E402

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
FLOOR = 22   # smallest SVG font, in viewBox units (about 8 pt at slide size;
             # full-width diagrams are height-limited to about 12 cm)
VIOLET, RED, BLUE, AMBER, GREEN = (ROLE["sys"][1], ROLE["fault"][1], ROLE["model"][1],
                                   ROLE["tool"][1], ROLE["memory"][1])


# ------------------------------------------------------------ SVG helpers
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
               f'</pattern></defs><rect x="0" y="0" width="{self.w}" height="{self.h}" fill="white"/>'
               + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)

    def rect(self, x, y, w, h, role="struct", dash=False, rx=8, sw=2, fill=None, stroke=None,
             opacity=1.0):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill or f}" stroke="{stroke or s}" stroke-width="{sw}"{d} '
                 f'opacity="{opacity}"/>')

    def text(self, x, y, s, fs=22, color=INK, anchor="middle", weight="normal", mono=False,
             italic=False, opacity=1.0):
        fs = max(fs, FLOOR)
        fam = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{fam}{it} '
                 f'opacity="{opacity}">{s}</text>')

    def box(self, x, y, w, h, role, title, sub=None, fs=22, sfs=20, dash=False, mono_sub=False,
            title_color=None, sw=2):
        self.rect(x, y, w, h, role, dash=dash, sw=sw)
        _, _, tc = ROLE[role]
        if sub:
            self.text(x + w / 2, y + h / 2 - 5, title, fs, title_color or tc, weight="bold")
            self.text(x + w / 2, y + h / 2 + sfs + 1, sub, sfs, GRAY, mono=mono_sub)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, title, fs, title_color or tc,
                      weight="bold")

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

    def cross(self, x, y, r=11, color=None):
        col = color or RED
        self.add(f'<path d="M{x-r},{y-r} L{x+r},{y+r} M{x-r},{y+r} L{x+r},{y-r}" '
                 f'stroke="{col}" stroke-width="5"/>')

    def check(self, x, y, r=12, color=None):
        col = color or GREEN
        self.add(f'<path d="M{x-r},{y} L{x-r/3},{y+r*0.7} L{x+r},{y-r*0.8}" fill="none" '
                 f'stroke="{col}" stroke-width="5"/>')

    def num(self, x, y, k, color=None):
        col = color or VIOLET
        self.add(f'<circle cx="{x}" cy="{y}" r="15" fill="{col}"/>')
        self.text(x, y + 7, str(k), 20, "white", weight="bold")


# =====================================================================
# Segment 1: continuation
# =====================================================================
def compound():
    hs = np.arange(0, 121)
    p_turn, h_run = calcs.RULE.step_success, calcs.RULE.short_horizon
    curve = 100 * p_turn ** hs
    lo, hi = calcs.CEIL.median_calls_low, calcs.CEIL.median_calls_high
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        a2 = 1.0 if k >= 2 else 0.0
        ax.axvspan(lo, hi, color=C["network_l"], alpha=0.8 * a2, lw=0)
        ax.text((lo + hi) / 2, 8, f"median model calls,\n{val('swe_configs_words')} public repair agents:\n"
                f"{val('med_lo')} to {val('med_hi')}", ha="center", va="bottom", fontsize=10.5,
                color=C["network"], fontweight="bold", alpha=a2)
        ax.plot(hs, curve, color=C["memory"], lw=2.6)
        p30 = 100 * calcs.RULE.result.success[h_run]
        ax.plot([h_run], [p30], "o", ms=10, color=C["danger"], mec="white", mew=1.5,
                zorder=5)
        ax.annotate(f"{val('fr_turns')} turns: {p30:.0f}%", (h_run, p30), xytext=(10, 8),
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C["danger"])
        p64 = 100 * p_turn ** hi
        ax.plot([hi], [p64], "o", ms=8, color=C["network"], mec="white", mew=1.2, zorder=5,
                alpha=a2)
        ax.annotate(f"{val('med_hi')} calls: {val('fr_hold_hi_pct')}%", (hi, p64), xytext=(10, 6),
                    textcoords="offset points", fontsize=11.5, fontweight="bold",
                    color=C["network"], alpha=a2)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Turns in the trajectory")
        ax.set_ylabel("Rule held on\nevery turn (%)")
        ax.text(118, 96, f"rule kept on {val('fr_step_pct')}% of turns", ha="right", fontsize=11,
                color=C["memory"], fontweight="bold")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, OUT / f"compound_step{k}.svg")


def harness():
    for k in (1, 2):
        s = Svg(900, 500)
        # (a) prototype loop
        s.text(10, 30, "(a) Prototype loop", 22, INK, anchor="start", weight="bold")
        s.box(20, 95, 170, 64, "model", "Model", fs=24)
        s.box(710, 95, 170, 64, "tool", "Tools", fs=24)
        s.arrow(190, 127, 708, 127, "amber")
        s.text(450, 152, "proposed call runs at once", 21, ROLE["tool"][2])
        for x, lab in ((290, "pause"), (450, "spend ceiling"), (610, "approval")):
            s.text(x, 64, lab, 21, GRAY)
            s.line(x, 72, x, 96, GRAY, 2, dash=True)
            s.cross(x, 104, 9)
        s.box(360, 178, 180, 42, "memory", "message list", fs=21)
        s.text(346, 234, "all state lives here", 20, GRAY, anchor="end")
        s.arrow(0, 0, 0, 0, "gray", path="M795,159 L795,199 L542,199")
        s.arrow(0, 0, 0, 0, "gray", path="M358,199 L105,199 L105,161")
        s.line(10, 250, 890, 250, "#cbd5e1", 2, dash=True)
        if k >= 2:
            s.text(10, 272, "(b) The same loop inside a harness", 22, INK, anchor="start",
                   weight="bold")
            s.box(20, 365, 170, 74, "model", "Model", "proposes only", fs=24)
            s.box(710, 365, 170, 74, "tool", "Tools", "settle outcomes", fs=24)
            s.rect(290, 300, 320, 192, "sys", rx=12, fill="#faf5ff", sw=2.5)
            s.text(306, 330, "Harness", 23, ROLE["sys"][2], anchor="start", weight="bold")
            for i, lab in enumerate(("trajectory record", "guarded transitions",
                                     "control and approval", "budgets, reserved first")):
                s.rect(306, 342 + i * 37, 288, 32, "white", rx=6, sw=1.5)
                s.text(320, 365 + i * 37, lab, 21, INK, anchor="start")
            s.arrow(190, 388, 288, 388, "blue")
            s.text(239, 378, "proposal", 20, ROLE["model"][2])
            s.arrow(288, 420, 192, 420, "blue")
            s.text(239, 445, "context", 20, ROLE["model"][2])
            s.arrow(612, 388, 708, 388, "amber")
            s.text(660, 378, "admitted", 20, ROLE["tool"][2])
            s.arrow(708, 420, 614, 420, "amber")
            s.text(660, 445, "result", 20, ROLE["tool"][2])
            s.text(750, 285, "pause, ceiling, approval", 20, VIOLET, anchor="middle")
            s.arrow(0, 0, 0, 0, "violet", path="M640,292 L640,318 L612,318")
        s.save(f"harness_step{k}")


def turn():
    """One supervised turn of the r18 repair (AH table 1), in four steps."""
    x0 = 160
    segs = [("READY", 150, 240, "white"), ("CALLING_MODEL", 240, 465, "model"),
            ("WAITING_TOOL", 465, 670, "tool"), ("", 670, 684, "white"),
            ("PAUSED", 684, 800, "struct"), ("READY", 800, 890, "white")]
    show_until = {1: 2, 2: 3, 3: 5, 4: 6}
    for k in range(1, 5):
        s = Svg(900, 350)
        for name, y in (("State", 44), ("Model", 120), ("Test job", 196), ("Operator", 262)):
            s.text(10, y + 8, name, 22, INK, anchor="start", weight="bold")
            if name != "State":
                s.line(x0, y, 890, y, "#cbd5e1", 3)
        for i, (name, a, b, role) in enumerate(segs):
            if i >= show_until[k]:
                break
            s.rect(a, 20, b - a, 48, role, rx=4, sw=2)
            if name:
                s.text((a + b) / 2, 52, name, 20, ROLE[role][2] if role != "white" else INK,
                       weight="bold")
        # step 1: reserve, then send the model call
        s.num(240, 90, 1)
        s.rect(240, 108, 225, 24, "model", rx=4)
        s.text(252, 160, "reserve, then send", 20, ROLE["model"][2], anchor="start")
        if k >= 2:
            s.num(465, 90, 2)
            s.rect(465, 184, 295 if k >= 4 else (160 if k == 2 else 295), 24, "tool", rx=4)
            s.text(475, 234, "reserve, register,", 20, ROLE["tool"][2], anchor="start")
            s.text(475, 258, "then send the test", 20, ROLE["tool"][2], anchor="start")
        if k >= 3:
            s.num(610, 290, 3, RED)
            s.text(600, 320, "pause: stop admitting,", 20, ROLE["fault"][2], anchor="end")
            s.text(600, 344, "keep owning the test", 20, ROLE["fault"][2], anchor="end")
        if k >= 4:
            s.num(760, 196, 4, GREEN)
            s.text(886, 150, "result settles its charge;", 20, ROLE["memory"][2], anchor="end")
            s.text(886, 174, "the turn stays paused", 20, ROLE["memory"][2], anchor="end")
            s.num(800, 290, 5)
            s.text(886, 320, "resume:", 20, ROLE["sys"][2], anchor="end")
            s.text(886, 344, "recheck all", 20, ROLE["sys"][2], anchor="end")
        s.save(f"turn_step{k}")


# =====================================================================
# Segment 2: record and states
# =====================================================================
def record_fields():
    rows = [("Identity, task", "task contract, version"),
            ("Control", "state, control_epoch"),
            ("Continuation", "context ref, plan rev"),
            ("Accounting", "actual, reserved, final"),
            ("Authority", "grants, workspace lease"),
            ("Invocations", "eligible call IDs"),
            ("Held approval", "digest, expiry"),
            ("Obligations", "operation table")]
    s = Svg(640, 470)
    s.text(10, 28, "Trajectory record, written only by the harness", 22, ROLE["sys"][2],
           anchor="start", weight="bold")
    for i, (a, b) in enumerate(rows):
        y = 46 + i * 52
        s.rect(10, y, 620, 44, "sys" if i % 2 == 0 else "white", rx=6, sw=1.5)
        s.text(24, y + 30, a, 22, INK, anchor="start", weight="bold")
        s.text(250, y + 30, b, 21, ROLE["sys"][2], anchor="start", mono=True)
    s.save("record_fields")


def states():
    """The trajectory state machine (AH figure 2), revealed in three steps."""
    P = {"PAUSED": (40, 22, 230, 70), "READY": (40, 190, 230, 76),
         "CALLING_MODEL": (380, 190, 250, 76), "WAITING_TOOL": (40, 372, 230, 76),
         "AWAITING_APPROVAL": (366, 372, 280, 76), "FINALIZING": (690, 190, 200, 76)}
    sub = {"PAUSED": "no new work", "READY": "admission checks", "CALLING_MODEL": "call reserved",
           "WAITING_TOOL": "tool work runs", "AWAITING_APPROVAL": "proposal held",
           "FINALIZING": "admission stopped"}
    role = {"PAUSED": "struct", "READY": "white", "CALLING_MODEL": "model",
            "WAITING_TOOL": "tool", "AWAITING_APPROVAL": "tool", "FINALIZING": "sys"}
    when = {"READY": 1, "CALLING_MODEL": 1, "WAITING_TOOL": 1, "PAUSED": 2,
            "AWAITING_APPROVAL": 2, "FINALIZING": 3}
    global FLOOR
    FLOOR = 23   # this figure is height-limited on the slide; larger type
    for k in (1, 2, 3):
        s = Svg(900, 500)
        s.rect(14, 8, 650, 476, "white", dash=True, rx=10, stroke="#cbd5e1", sw=1.5)
        s.text(650, 36, "live states", 20, GRAY, anchor="end")
        for name, (x, y, w, h) in P.items():
            if when[name] > k:
                continue
            s.box(x, y, w, h, role[name], name, sub[name], fs=24, sfs=23,
                  dash=(name == "AWAITING_APPROVAL"),
                  title_color=INK if role[name] in ("white", "struct") else None)
        # ordinary cycle (heavy)
        s.arrow(270, 214, 378, 214, "ink", sw=3.5)
        s.text(324, 204, "admit", 20, INK)
        s.arrow(0, 0, 0, 0, "ink", sw=3.5, path="M450,266 L450,318 L210,318 L210,370")
        s.text(316, 310, "tool call admitted", 20, INK)
        s.arrow(0, 0, 0, 0, "ink", sw=3.5, path="M100,372 L100,268")
        s.text(110, 330, "wait met", 20, INK, anchor="start")
        if k >= 2:
            s.arrow(120, 190, 120, 94, "gray")
            s.text(110, 150, "pause", 20, GRAY, anchor="end")
            s.arrow(200, 94, 200, 188, "gray")
            s.text(210, 150, "resume", 20, GRAY, anchor="start")
            s.arrow(378, 248, 272, 248, "gray")
            s.text(324, 284, "none", 20, GRAY)
            s.arrow(560, 268, 560, 370, "amber", dash=True)
            s.text(570, 330, "review", 20, ROLE["tool"][2], anchor="start")
            s.arrow(364, 430, 272, 430, "amber")
            s.text(330, 474, "approved, guards pass", 20, ROLE["tool"][2])
        if k >= 3:
            s.arrow(630, 228, 688, 228, "violet", sw=3)
            s.text(894, 152, "from any live state:", 20, VIOLET, anchor="end")
            s.text(894, 178, "done, cancel, budget", 20, VIOLET, anchor="end")
            for i, (t, r) in enumerate((("COMPLETED", "white"), ("FAILED", "fault"),
                                        ("CANCELED", "struct"))):
                y = 300 + i * 58
                s.rect(712, y, 178, 46, r, rx=6, sw=3)
                s.text(801, y + 31, t, 23, INK if r != "fault" else ROLE["fault"][2],
                       weight="bold")
                s.arrow(0, 0, 0, 0, "gray", path=f"M700,266 L700,{y + 23} L710,{y + 23}")
            s.text(801, 494, "absorbing: final", 20, GRAY)
        s.save(f"states_step{k}")
    FLOOR = 22


def late():
    """A late result: settlement always, observation only on a current epoch."""
    for k in (1, 2, 3):
        s = Svg(900, 440)
        s.text(10, 30, "Test of r18", 22, INK, anchor="start", weight="bold")
        s.line(150, 80, 890, 80, "#cbd5e1", 3)
        s.rect(170, 66, 560, 28, "tool", rx=4)
        s.text(178, 124, "sent under epoch 7", 20, ROLE["tool"][2], anchor="start")
        s.line(400, 50, 400, 160, RED, 2.5, dash=True)
        s.text(410, 150, "deadline: outcome unknown", 20, ROLE["fault"][2], anchor="start")
        s.text(410, 174, "epoch becomes 8", 20, ROLE["fault"][2], anchor="start")
        s.add(f'<circle cx="730" cy="80" r="11" fill="{GREEN}"/>')
        s.text(886, 52, "passed, arrives late", 20, ROLE["memory"][2], anchor="end")
        if k >= 2:
            s.arrow(0, 0, 0, 0, "green", path="M730,92 L730,196 L480,196 L480,214")
            s.box(250, 216, 460, 82, "memory", "Operation table", "evidence kept; reserved $ becomes actual",
                  fs=22, sfs=20)
            s.check(680, 236, 10)
        if k >= 3:
            s.arrow(0, 0, 0, 0, "red", path="M730,196 L800,196 L800,330")
            s.box(560, 332, 330, 82, "fault", "Current turn", "epoch 7 is not 8: not staged",
                  fs=22, sfs=20)
            s.cross(860, 350, 9)
            s.text(20, 360, "Every authentic result settles.", 21, INK, anchor="start",
                   weight="bold")
            s.text(20, 390, "Only a current one enters the turn.", 21, INK, anchor="start",
                   weight="bold")
        s.save(f"late_step{k}")


# =====================================================================
# Segment 3: control
# =====================================================================
def pause_plot():
    m = calcs.PAUSE
    ks = np.array(m["k"], dtype=float)
    mean = np.array([m["end"][k][1] for k in m["k"]])
    lo = mean - np.array([m["end"][k][0] for k in m["k"]])
    hi = np.array([m["end"][k][2] for k in m["k"]]) - mean
    canc = np.array([m["cancel"][k][2] for k in m["k"]])
    xs = np.linspace(0, 4300, 50)
    t_dec_s = calcs.mag(calcs.CALL.floors.result.memory_time, "s")
    k_max, wait = calcs.CALL.output_allowance, calcs.mag(calcs.CALL.allowance_decode_time, "s")
    ref = xs * t_dec_s
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.55))
        a2 = 1.0 if k >= 2 else 0.0
        a3 = 1.0 if k >= 3 else 0.0
        ax.plot(xs, ref, color=C["neutral"], lw=1.8, ls="--", alpha=a2)
        ax.text(250, 88, f"chapter reference,\n{val('bw_tdec')} ms per token",
                fontsize=10.5, color=C["neutral"], ha="left", va="top", alpha=a2)
        ax.annotate(f"{val('bw_wait')} s", (k_max, wait),
                    xytext=(-4, 6), textcoords="offset points", ha="right", va="bottom",
                    fontsize=11, fontweight="bold", color=C["neutral"], alpha=a2)
        ax.errorbar(ks, mean, yerr=[lo, hi], fmt="o", ms=8, color=C["danger"], lw=1.6,
                    capsize=3)
        ax.annotate(f"wait for the\nreply: {val('ps_wait_hi_round')} s", (ks[-1], mean[-1]),
                    xytext=(4150, 50), textcoords="data", ha="right", va="center", fontsize=11.5,
                    fontweight="bold", color=C["danger"])
        ax.plot(ks, canc, "s", ms=8, color=C["data"], alpha=a3)
        ax.annotate(f"cancel the stream: at most {val('ps_cancel_ms')} ms", (ks[-1], canc[-1]),
                    xytext=(-12, 10), textcoords="offset points", ha="right", fontsize=11.5,
                    fontweight="bold", color=C["data"], alpha=a3)
        ax.set_xlim(0, 4400)
        ax.set_ylim(-4, 100)
        ax.set_xticks(m["k"])
        ax.set_xlabel("Output allowance of the call in flight (tokens)")
        ax.set_ylabel("Seconds until\nthe pause holds")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, OUT / f"pause_step{k}.svg")


def escrow():
    for k in (1, 2, 3):
        s = Svg(900, 470)
        s.text(10, 28, "(a) What the reviewer approves", 22, INK, anchor="start", weight="bold")
        s.rect(10, 42, 880, 54, "tool", rx=6)
        s.text(24, 77, "submit_review(r18 patch, named checks)", 22, ROLE["tool"][2],
               anchor="start", mono=True)
        for i, (a, b) in enumerate((("destination", "internal"), ("evidence", "named checks"),
                                    ("precondition", "base version"), ("expiry", "deadline"))):
            x = 10 + i * 220
            s.rect(x, 96, 220, 66, "white", rx=0, sw=1.5)
            s.text(x + 14, 124, a, 21, INK, anchor="start", weight="bold")
            s.text(x + 14, 150, b, 20, GRAY, anchor="start")
            s.line(x + 110, 162, x + 110, 180, VIOLET, 2)
        s.rect(10, 180, 880, 40, "sys", rx=6)
        s.text(450, 207, "approval digest binds the proposal and every field", 21,
               ROLE["sys"][2], weight="bold")
        s.line(10, 238, 890, 238, "#cbd5e1", 2, dash=True)
        if k >= 2:
            s.text(10, 268, "(b) The gate", 22, INK, anchor="start", weight="bold")
            s.box(10, 300, 200, 74, "tool", "1 Hold", "not sent", fs=22, sfs=20, dash=True)
            s.arrow(210, 337, 238, 337, "gray")
            s.box(240, 300, 170, 74, "struct", "2 Reviewer", "sees (a)", fs=22, sfs=20,
                  title_color=INK)
            s.arrow(410, 337, 438, 337, "gray")
            s.box(440, 300, 180, 74, "struct", "3 Decision", "model not asked", fs=22,
                  sfs=20, title_color=INK)
        if k >= 3:
            s.rect(652, 286, 12, 34, "sys", rx=2, fill=VIOLET)
            s.rect(652, 356, 12, 106, "sys", rx=2, fill=VIOLET)
            s.arrow(620, 337, 704, 337, "amber", sw=3)
            s.box(706, 300, 188, 74, "tool", "WAITING_TOOL", "sent", fs=22, sfs=22)
            s.text(676, 284 - 8, "4 recheck at dispatch", 20, VIOLET, weight="bold")
            s.arrow(530, 374, 530, 400, "gray", dash=True)
            s.box(440, 402, 180, 60, "struct", "REFUSED", None, fs=21, title_color=INK)
            s.text(430, 440, "denied, expired, stale", 20, GRAY, anchor="end")
            s.arrow(650, 432, 622, 432, "gray", dash=True)
        s.save(f"escrow_step{k}")


# =====================================================================
# Segment 4: sharing and budgets
# =====================================================================
def drr():
    sim = calcs.DRR
    rows = [("Token charge", sim.simulated_charge_share), ("Calls admitted", sim.simulated_call_share)]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 1.75))
        for i, (lab, sh) in enumerate(rows):
            y = 1 - i
            a = 1.0 if i < k else 0.0
            ia, ma = sh["interactive"], sh["maintenance"]
            ax.barh(y, 100 * ia, color=C["memory"], height=0.6, alpha=a)
            ax.barh(y, 100 * ma, left=100 * ia, color=C["neutral_l"], height=0.6, alpha=a,
                    edgecolor=C["neutral"], lw=0.8)
            ax.text(-1, y, lab, ha="right", va="center", fontsize=12.5, fontweight="bold",
                    color=C["text"])
            ax.text(100 * ia / 2, y, f"interactive {100 * ia:.0f}%", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold", alpha=a)
            ax.text(101, y, f"maintenance {100 * ma:.0f}%" if ma > 0.1 else
                    f"maintenance {100 * ma:.0f}%", ha="left", va="center", fontsize=11.5,
                    color=C["neutral"], fontweight="bold", alpha=a)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_xticks([])
        for sname in ("left", "bottom"):
            ax.spines[sname].set_visible(False)
        sp.save(fig, OUT / f"drr_step{k}.svg")


def runaway():
    global FLOOR
    FLOOR = 24   # height-limited on the slide; larger type
    for k in (1, 2, 3):
        s = Svg(900, 450)
        # loop on the left
        cx, cy, r = 210, 190, 120
        s.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{RED}" '
              f'stroke-width="3" stroke-dasharray="10 6"/>')
        nodes = [("propose variant", cx, cy - r), ("test fails", cx + r, cy),
                 ("error appended", cx, cy + r), ("retry", cx - r, cy)]
        for lab, x, y in nodes:
            w = 24 + 13.5 * len(lab)
            s.rect(x - w / 2, y - 22, w, 44, "fault", rx=8)
            s.text(x, y + 8, lab, 24, ROLE["fault"][2], weight="bold")
        s.text(cx, 366, "each call well formed", 20, GRAY)
        s.text(cx, 392, "and within per-call limits", 20, GRAY)
        if k >= 2:
            s.rect(420, 30, 470, 280, "sys", rx=12, fill="#faf5ff")
            s.text(436, 62, "Budget vector (outside context)", 24, ROLE["sys"][2],
                   anchor="start", weight="bold")
            used = calcs.METER.used_with_reuse
            for i, (lab, key) in enumerate((("turns", "turns"), ("tokens", "tokens"),
                                            ("wall time", "time"), ("dollars", "dollars"))):
                y = 80 + i * 46
                s.text(436, y + 24, lab, 21, INK, anchor="start")
                s.rect(560, y + 4, 300, 28, "white", rx=4, sw=1.5)
                s.rect(560, y + 4, 300 * used[key], 28, "sys", rx=4, fill="#c4b5fd", sw=0)
                s.text(560 + 300 * used[key] + 8, y + 26, f"{100 * used[key]:.0f}%", 20,
                       ROLE["sys"][2], anchor="start")
            s.text(436, 296, "illustrative (NM 10.3, with reuse)", 20, GRAY, anchor="start")
            s.arrow(0, 0, 0, 0, "violet", path="M420,276 L303,276")
            s.cross(352, 276, 10)
        if k >= 3:
            for i, (lab, sub) in enumerate((("Steer", "show the remaining allowance"),
                                            ("Restrict", "narrow grants"),
                                            ("Finalize", "checks, report, cleanup"))):
                x = 420 + i * 164
                s.rect(x, 338, 140, 54, "sys", rx=8)
                s.text(x + 70, 374, lab, 25, ROLE["sys"][2], weight="bold")
                if i < 2:
                    s.arrow(x + 140, 365, x + 162, 365, "violet")
            s.text(420, 425, "then the hard stop at admission", 20, GRAY, anchor="start")
        s.save(f"runaway_step{k}")
    FLOOR = 22


def admit():
    """Drawn for a 0.6-width column (about 3.6 in on the slide)."""
    a = calcs.ADMIT
    settled, reserved = calcs.mag(calcs.ADMIT_S.settled, "USD"), calcs.mag(calcs.ADMIT_S.reserved, "USD")
    final, allow = calcs.mag(calcs.ADMIT_S.finalization_reserve, "USD"), calcs.mag(calcs.ADMIT_S.allowance, "USD")
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.0, 2.6))
        rows = [("Spend\ncheck", [(settled, "neutral_l"), (a.spend_headroom, "memory_l")]),
                ("Admission\ntest", [(settled, "neutral_l"), (reserved, "compute_l"),
                                      (final, "network_l"), (a.headroom, "memory_l")])]
        for i, (lab, parts) in enumerate(rows):
            y = 1 - i
            al = 1.0 if i < k else 0.0
            left = 0
            for v, ck in parts:
                ax.barh(y, v, left=left, color=C[ck], height=0.5, alpha=al, edgecolor="white", lw=1)
                left += v
            start = settled + (0 if i == 0 else reserved + final)
            fits = a.spend_check_admits if i == 0 else a.admitted
            ax.barh(y, a.liability, left=start, color="none", height=0.26, alpha=al,
                    edgecolor=C["data"] if fits else C["danger"], lw=2.2,
                    hatch=None if fits else "////")
            ax.text(-0.06, y, lab, ha="right", va="center", fontsize=11, fontweight="bold",
                    color=C["text"])
            free = val("ad_spend_head") if i == 0 else val("ad_head")
            ax.text(allow + 0.12, y, f"free ${free}\n" + ("call fits" if fits else
                    "call does\nnot fit"), ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color=C["data"] if fits else C["danger"], alpha=al)
        ax.text(settled / 2, 1, f"settled\n${val('ad_settled')}", ha="center", va="center", fontsize=10.5)
        ax.text(settled / 2, 0, f"settled\n${val('ad_settled')}", ha="center", va="center", fontsize=10.5,
                alpha=1.0 if k >= 2 else 0.0)
        ax.text(settled + reserved / 2, 0.42, f"test ${val('ad_reserved')}", ha="center", fontsize=10,
                alpha=1.0 if k >= 2 else 0.0)
        ax.text(allow, -0.5, f"final ${val('ad_final')}", ha="right", fontsize=10,
                alpha=1.0 if k >= 2 else 0.0)
        ax.axvline(allow, color=C["text"], lw=1.6)
        ax.text(allow, 1.45, f"allowance ${val('ad_allow')}", ha="right", fontsize=10.5,
                color=C["text"])
        ax.set_xlim(0, 4.6)
        ax.set_ylim(-0.65, 1.6)
        ax.set_yticks([])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["$0", "$1", "$2", "$3"])
        ax.spines["left"].set_visible(False)
        sp.save(fig, OUT / f"admit_step{k}.svg")


def overshoot():
    t = calcs.TRAJ
    cum = np.array(t["cumulative_cost"])
    o = dict(limit=t["limits"]["cost_limit"], calls=len(cum), before_last=cum[-2], final=cum[-1])
    n = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.step(n, cum, where="post", color=C["memory"], lw=2.2)
    ax.axhline(o["limit"], color=C["danger"], lw=1.6, ls="--")
    ax.text(3, o["limit"] + 0.07, f"cost_limit = {val('tj_limit_cfg')}", color=C["danger"],
            fontsize=11, fontweight="bold")
    ax.plot([o["calls"] - 1], [o["before_last"]], "o", ms=7, color=C["neutral"])
    ax.annotate(f"before call {val('tj_calls')}: ${val('tj_before')},\nunder the limit, so admitted",
                (o["calls"] - 1, o["before_last"]), xytext=(18, 2.2), textcoords="data",
                ha="left", va="center", fontsize=11, color=C["text"],
                arrowprops=dict(arrowstyle="->", color=C["neutral"], lw=1.2))
    ax.plot([o["calls"]], [o["final"]], "o", ms=9, color=C["danger"])
    ax.annotate(f"stopped at ${val('tj_final')}", (o["calls"], o["final"]), xytext=(-10, 6),
                textcoords="offset points", ha="right", fontsize=11.5, fontweight="bold",
                color=C["danger"])
    ax.set_xlim(0, o["calls"] + 3)
    ax.set_ylim(0, 3.75)
    ax.set_xlabel("Model call")
    ax.set_ylabel("Settled cost ($)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "overshoot.svg")


def meter():
    mt = calcs.METER
    labels = [("Turns", "turns"), ("Tokens", "tokens"), ("Time", "time"), ("Dollars", "dollars")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.3))
        for i, (lab, key) in enumerate(labels):
            y = 3 - i
            v = 100 * mt.used_with_reuse[key]
            ax.barh(y + 0.17, v, height=0.32, color=C["memory"])
            ax.text(v + 1.2, y + 0.17, f"{v:.1f}%" if key != "turns" else f"{v:.0f}%",
                    va="center", fontsize=11, color=C["memory"], fontweight="bold")
            w = 100 * mt.used_without_reuse[key]
            a = 1.0 if k >= 2 else 0.0
            ax.barh(y - 0.17, w, height=0.32, color=C["compute"], alpha=a)
            ax.text(w + 1.2, y - 0.17, f"{w:.1f}%" if key != "turns" else f"{w:.0f}%",
                    va="center", fontsize=11, color=C["compute"], fontweight="bold", alpha=a)
            ax.text(-1.5, y, lab, ha="right", va="center", fontsize=12.5, fontweight="bold",
                    color=C["text"])
        ax.text(99, 2.25, "with prefix reuse", ha="right", fontsize=11, color=C["memory"],
                fontweight="bold")
        ax.text(99, 1.85, "cache missed", ha="right", fontsize=11, color=C["compute"],
                fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 3.5)
        ax.set_yticks([])
        ax.set_xlabel(f"Share of the allowance used after {val('mt_turns')} turns (%)")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        sp.save(fig, OUT / f"meter_step{k}.svg")


def ceiling():
    rows = calcs.CONFIGS
    pick = calcs.PICK
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ok, bad = pick["accepted_calls"], pick["failed_calls"]
    ax.plot(ok, np.arange(1, len(ok) + 1) / len(ok), color=C["memory"], lw=2.2)
    ax.plot(bad, np.arange(1, len(bad) + 1) / len(bad), color=C["danger"], lw=2.2, ls="--")
    ax.axvline(pick["cap"], color=C["neutral"], lw=1.4)
    ax.text(pick["cap"] + 3, 0.08, f"ceiling\n{pick['cap']:.0f} calls", fontsize=10.5,
            color=C["neutral"])
    ax.text(2, 0.97, f"accepted ({pick['accepted']})", fontsize=11, color=C["memory"],
            fontweight="bold", va="center")
    ax.text(70, 0.52, f"failed ({pick['runs'] - pick['accepted']})", fontsize=11,
            color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Model calls per run")
    ax.set_ylabel("Fraction of runs")
    sp.save(fig, OUT / "ceiling_cdf.svg")

    # drawn at its slot: with its labels about 2.9 x 2.0 in (half the slide, 5.0 cm tall)
    fig, ax = plt.subplots(figsize=(2.0, 1.85))
    short = {"SWE-agent 0.x (2024)": "SWE-agent 0.x", "SWE-agent 1.1": "SWE-agent 1.1",
             "OpenHands": "OpenHands", "mini-SWE-agent 2.x": "mini-SWE-agent"}
    model = {"Claude 3.5 Sonnet": "Sonnet 3.5", "Claude Sonnet 4": "Sonnet 4",
             "Claude Opus 4.6": "Opus 4.6", "GPT-5.2-Codex": "GPT-5.2-Codex"}
    rows = sorted(rows, key=lambda r: r["failed_halted"])
    for i, r in enumerate(rows):
        y = i
        ax.barh(y + 0.18, 100 * r["failed_halted"], height=0.34, color=C["danger"])
        ax.barh(y - 0.18, 100 * r["accepted_halted"], height=0.34, color=C["memory"])
        ax.text(100 * r["failed_halted"] + 1, y + 0.18, f"{100 * r['failed_halted']:.0f}%",
                va="center", fontsize=8.5, color=C["danger"], fontweight="bold")
        ax.text(100 * r["accepted_halted"] + 1, y - 0.18, f"{100 * r['accepted_halted']:.0f}%",
                va="center", fontsize=8.5, color=C["memory"], fontweight="bold")
        ax.text(-1, y, f"{short[r['scaffold']]}\n{model[r['model']]}, cap {r['cap']:.0f}",
                ha="right", va="center", fontsize=8.5, color=C["text"])
    ax.set_yticks([])
    ax.set_xlim(0, 40)
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("Runs the ceiling halts (%)", fontsize=9)
    ax.spines["left"].set_visible(False)
    sp.save(fig, OUT / "ceiling_bars.svg")


def osworld():
    steps = calcs.STEPS.steps
    hi = max(steps)
    names = {"claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
             "claude-sonnet-4-20250514": "Claude Sonnet 4",
             "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
             "computer-use-preview-2025-03-11": "computer-use-preview",
             "o3-2025-04-16_medium": "o3"}
    cols = ["danger", "memory", "network", "compute", "data"]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    order = sorted(calcs.STEPS.scores.items(), key=lambda kv: -kv[1][hi])
    offs = {"Claude Sonnet 4": 1.6, "Claude 3.7 Sonnet": 1.6, "computer-use-preview": -1.8}
    for (m, v), ck in zip(order, cols):
        xs = steps
        ax.plot(xs, [v[x] for x in xs], "-o", color=C[ck], lw=2.2, ms=6)
        nm = names.get(m, m)
        ax.text(104, v[hi] + offs.get(nm, 0), f"{nm}  {v[hi]:.0f}%", va="center",
                fontsize=10.5, color=C[ck], fontweight="bold")
    ax.set_xticks(steps)
    ax.set_xlim(8, 165)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Step ceiling enforced on the agent")
    ax.set_ylabel("Tasks completed (%)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "osworld.svg")


# =====================================================================
# Thumbnails for the goals and takeaways slides: text-free recognition
# cues drawn at their slot (2.5 x 0.95 cm, a 260 x 100 canvas).
# =====================================================================
def thumbs():
    def svg(name, body):
        (OUT / f"{name}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 100" width="260" '
            'height="100"><rect width="260" height="100" fill="white"/>' + body + "</svg>")
    # horizon compounding: a decaying curve, the 30-turn point
    p_turn, h_run = calcs.RULE.step_success, calcs.RULE.short_horizon
    pts = " ".join(f"{10 + 2 * h:.1f},{90 - 80 * p_turn ** h:.1f}" for h in range(0, 121))
    svg("thumb_compound", f'<line x1="10" y1="92" x2="250" y2="92" stroke="#94a3b8" stroke-width="3"/>'
        f'<line x1="10" y1="8" x2="10" y2="92" stroke="#94a3b8" stroke-width="3"/>'
        f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="5"/>'
        f'<circle cx="{10 + 2 * h_run}" cy="{90 - 80 * p_turn ** h_run:.1f}" r="9" fill="{RED}"/>')
    # state machine: READY -> CALLING -> WAITING loop and FINALIZING
    b = ""
    for x, y, f, st in ((12, 8, "#ffffff", "#0f172a"), (100, 8, ROLE["model"][0], BLUE),
                        (12, 62, ROLE["tool"][0], AMBER), (190, 34, ROLE["sys"][0], VIOLET)):
        b += f'<rect x="{x}" y="{y}" width="62" height="30" rx="5" fill="{f}" stroke="{st}" stroke-width="3"/>'
    b += (f'<path d="M74,23 L98,23" stroke="#0f172a" stroke-width="4"/>'
          f'<path d="M131,38 L131,77 L76,77" fill="none" stroke="#0f172a" stroke-width="4"/>'
          f'<path d="M43,62 L43,40" stroke="#0f172a" stroke-width="4"/>'
          f'<path d="M162,23 L188,45" stroke="{VIOLET}" stroke-width="4"/>')
    svg("thumb_states", b)
    # late result: a bar, a green dot, settle (green) and blocked (red)
    svg("thumb_late", f'<rect x="10" y="12" width="170" height="18" rx="4" fill="{ROLE["tool"][0]}" stroke="{AMBER}" stroke-width="3"/>'
        f'<line x1="95" y1="4" x2="95" y2="40" stroke="{RED}" stroke-width="3" stroke-dasharray="6 4"/>'
        f'<circle cx="180" cy="21" r="9" fill="{GREEN}"/>'
        f'<path d="M180,30 L180,50 L80,50 L80,60" fill="none" stroke="{GREEN}" stroke-width="4"/>'
        f'<rect x="30" y="62" width="110" height="30" rx="5" fill="{ROLE["memory"][0]}" stroke="{GREEN}" stroke-width="3"/>'
        f'<path d="M180,50 L215,50 L215,60" fill="none" stroke="{RED}" stroke-width="4"/>'
        f'<rect x="170" y="62" width="85" height="30" rx="5" fill="{ROLE["fault"][0]}" stroke="{RED}" stroke-width="3"/>'
        f'<path d="M203,69 L227,85 M203,85 L227,69" stroke="{RED}" stroke-width="4"/>')
    # pause: waits rise with allowance (red), cancels stay flat (green)
    b = (f'<line x1="10" y1="92" x2="250" y2="92" stroke="#94a3b8" stroke-width="3"/>'
         f'<line x1="10" y1="92" x2="240" y2="12" stroke="#94a3b8" stroke-width="3" stroke-dasharray="7 5"/>')
    for x, y in ((30, 85), (80, 72), (140, 52), (230, 28)):
        b += f'<circle cx="{x}" cy="{y}" r="8" fill="{RED}"/>'
        b += f'<rect x="{x - 7}" y="80" width="14" height="14" fill="{GREEN}"/>'
    svg("thumb_pause", b)
    # admission: spend check fits; admission test overflows the allowance
    b = (f'<rect x="10" y="12" width="150" height="28" fill="#dddddd"/>'
         f'<rect x="160" y="12" width="54" height="28" fill="#d1e6f3"/>'
         f'<rect x="168" y="18" width="14" height="16" fill="none" stroke="{GREEN}" stroke-width="3"/>'
         f'<rect x="10" y="58" width="150" height="28" fill="#dddddd"/>'
         f'<rect x="160" y="58" width="29" height="28" fill="#fce4cc"/>'
         f'<rect x="189" y="58" width="18" height="28" fill="#e6d4e5"/>'
         f'<rect x="207" y="58" width="7" height="28" fill="#d1e6f3"/>'
         f'<rect x="207" y="62" width="14" height="20" fill="#fee2e2" stroke="{RED}" stroke-width="3"/>'
         f'<line x1="214" y1="4" x2="214" y2="96" stroke="#0f172a" stroke-width="3"/>')
    svg("thumb_admit", b)



def main():
    compound()
    harness()
    turn()
    record_fields()
    states()
    late()
    pause_plot()
    escrow()
    drr()
    runaway()
    admit()
    overshoot()
    meter()
    ceiling()
    osworld()
    thumbs()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
