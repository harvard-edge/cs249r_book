#!/usr/bin/env python3
"""Figures for Volume III, Lecture 1 (Introduction).

Every number drawn comes from calcs.py (which names its sources) or is typed
here with its source. Diagrams are SVG drawn at the size of their slot, in the
Volume III agentic-stack palette (rules/volumes/vol3/palette.md); data plots
use the shared slideplot helper. The slides Makefile converts SVG to PDF.

Redrawn from the chapter (slide variants, larger type, split into steps):
  loop        Introduction, figure 1 (a person or a runtime closes the loop)
  stochastic  Introduction, figure 2 (the stochastic computer map)
  hsa         Introduction, figure 3 (the H·S·A exposure space)
  faults      Introduction, section 1.5.1 margin figure (three fault models)
  sixphase    Introduction, figure 5 (the six-phase turn)
New: cleanup (Example 1.1), traj (trajectory strip with a real run),
authority (side-effect classes), ladder (closure evidence levels), workflow
(section 1.6.1), and the data plots calls, cmds, claimed, metr, horizon,
costbars, realcost.

Run from this folder: python3 make_figs.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs  # noqa: E402
import slideplot as sp  # noqa: E402

C = sp.C
plt = sp.plt
OUT = Path(__file__).resolve().parent

# Volume III agentic-stack palette: (fill, stroke, text)
MODEL = ("#e0f2fe", "#0284c7", "#0369a1")
MEMORY = ("#dcfce7", "#16a34a", "#15803d")
TOOLS = ("#fef3c7", "#d97706", "#b45309")
RUNTIME = ("#f3e8ff", "#7c3aed", "#6d28d9")
FAULT = ("#fee2e2", "#dc2626", "#b91c1c")
NEUTRAL = ("#f8fafc", "#94a3b8", "#0f172a")
OK = ("#dcfce7", "#16a34a", "#15803d")
INK = "#0f172a"
GRAY = "#64748b"
FONT = "Helvetica, Arial, sans-serif"
MONO = "Courier New, Courier, monospace"
SUBS = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u029c": "H"}


# ---------------------------------------------------------------- SVG kit
class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def box(self, x, y, w, h, role=NEUTRAL, rx=6, sw=2, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{role[0]}" '
                 f'stroke="{role[1]}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=20, weight="normal", color=INK, anchor="start",
             italic=False, font=FONT, op=1.0):
        st = ' font-style="italic"' if italic else ""
        lines = s.split("\n")
        out = (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
               f'fill="{color}" text-anchor="{anchor}"{st} opacity="{op}">')
        for i, ln in enumerate(lines):
            ln = ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            for u, d in SUBS.items():
                ln = ln.replace(u, f'<tspan baseline-shift="sub" font-size="72%">{d}</tspan>')
            out += f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.2}">{ln}</tspan>'
        self.add(out + "</text>")

    def line(self, x1, y1, x2, y2, color=GRAY, sw=2.5, dash=None, arrow=True, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}"{d}{m} opacity="{op}"/>')

    def path(self, d, color=GRAY, sw=2.5, dash=None, arrow=True, fill="none"):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{ds}{m}/>')

    def save(self, name):
        cols = {GRAY, MODEL[1], MEMORY[1], TOOLS[1], RUNTIME[1], FAULT[1], OK[1], INK}
        marks = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in cols)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}"><defs>{marks}</defs>'
               f'<rect width="{self.w}" height="{self.h}" fill="white"/>' + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)


def role_box(s, x, y, w, h, title, sub, role, size=22, op=1.0, sub_size=19):
    s.box(x, y, w, h, role, op=op)
    s.text(x + w / 2, y + h / 2 - 4, title, size, "bold", INK, "middle", op=op)
    if sub:
        s.text(x + w / 2, y + h / 2 + 22, sub, sub_size, "normal", role[2], "middle", op=op)


# ------------------------------------------------- Example 1.1: two workspaces
def cleanup():
    """Example 1.1 (CH section 1.4): one cleanup command, two environments.
    Step 2 adds the H·S·A cell each run lands in."""
    for k in (1, 2):
        s = Svg(900, 330)
        role_box(s, 300, 8, 300, 74, "Same model, same prompt", "same horizon, same command", MODEL)
        s.line(390, 82, 210, 120, MODEL[1])
        s.line(510, 82, 690, 120, MODEL[1])
        for x, title, sub, items, role in (
                (20, "A. Disposable working copy", "an isolated sandbox clone",
                 ["build/  *.o  tmp_*", "the patch under test"], TOOLS),
                (470, "B. The developer's only workspace", "holds uncommitted edits",
                 ["build/  *.o  tmp_*", "unstaged edits, no snapshot"], FAULT if k == 2 else TOOLS)):
            s.box(x, 124, 410, 150, NEUTRAL)
            s.text(x + 16, 156, title, 22, "bold")
            s.text(x + 16, 184, sub, 21, color=GRAY)
            s.box(x + 16, 196, 378, 36, ("#0f172a", "#0f172a", "#fff"), rx=4)
            s.text(x + 24, 222, "$ rm -rf build tmp_* *.orig", 20, color="#e2e8f0", font=MONO)
            s.text(x + 16, 260, items[1], 21, color=role[2] if k == 2 else GRAY)
        if k == 2:
            s.text(225, 312, "S₂ · A₁: discard the sandbox", 21, "bold", TOOLS[2], "middle")
            s.text(675, 312, "S₃ · A₃: the edits are gone", 21, "bold", FAULT[2], "middle")
        s.save(f"cleanup_step{k}")


# ---------------------------------------------------- figure 1: who closes the loop
def loop():
    for k in (1, 2):
        s = Svg(900, 360)
        s.add('<line x1="450" y1="10" x2="450" y2="350" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="6 6"/>')
        for x0, op in ((0, 1.0), (460, 1.0 if k == 2 else 0.0)):
            s.text(x0 + 220, 30, "A. A person closes the loop" if x0 == 0 else "B. A runtime closes the loop",
                   24, "bold", INK, "middle", op=op)
            s.add(f'<line x1="{x0 + 20}" y1="138" x2="{x0 + 420}" y2="138" stroke="#94a3b8" '
                  f'stroke-width="1.5" stroke-dasharray="5 5" opacity="{op}"/>')
            s.text(x0 + 420, 128, "proposes", 20, color=GRAY, anchor="end", op=op)
            s.text(x0 + 420, 160, "acts", 20, color=GRAY, anchor="end", op=op)
        # panel A
        role_box(s, 110, 50, 220, 70, "Model endpoint", "answers only", MODEL)
        role_box(s, 110, 172, 220, 70, "Developer", "carries everything", NEUTRAL)
        role_box(s, 110, 278, 220, 70, "Terminal", "make test", TOOLS)
        s.line(170, 172, 170, 122, GRAY); s.text(160, 164, "prompt + error", 20, color=GRAY, anchor="end")
        s.line(270, 120, 270, 170, GRAY); s.text(280, 164, "patch", 20, color=GRAY)
        s.line(170, 242, 170, 276, GRAY); s.text(160, 266, "runs", 20, color=GRAY, anchor="end")
        s.line(270, 278, 270, 244, GRAY); s.text(280, 266, "error", 20, color=GRAY)
        # panel B
        op = 1.0 if k == 2 else 0.0
        role_box(s, 570, 50, 220, 70, "Foundation model", "holds no authority", MODEL, op=op)
        role_box(s, 570, 172, 220, 70, "Agent runtime", "grants each action", RUNTIME, op=op)
        role_box(s, 490, 278, 180, 70, "Sandbox", "isolated", TOOLS, op=op, size=21)
        role_box(s, 700, 278, 180, 70, "Verifiers", "tests, linters", TOOLS, op=op, size=21)
        s.line(630, 172, 630, 122, GRAY, op=op); s.text(620, 164, "context", 18, color=GRAY, anchor="end", op=op)
        s.line(730, 120, 730, 170, GRAY, op=op); s.text(740, 164, "proposal", 18, color=GRAY, op=op)
        s.line(600, 242, 600, 276, RUNTIME[1], sw=3, op=op)
        s.text(590, 268, "dispatch", 20, color=RUNTIME[2], anchor="end", op=op)
        s.line(640, 278, 640, 244, GRAY, op=op)
        s.line(770, 278, 770, 244, GRAY, op=op); s.text(780, 268, "evidence", 18, color=GRAY, op=op)
        s.line(670, 313, 698, 313, GRAY, op=op)
        s.save(f"loop_step{k}")


# ----------------------------------------------------- figure 2: stochastic computer
def stochastic():
    """Slide variant of the chapter's map, drawn for a 0.64-width column.
    Step 2 marks where the analogy breaks (CH section 1.3)."""
    for k in (1, 2):
        s = Svg(640, 470)
        s.box(6, 6, 628, 330, NEUTRAL, rx=8, sw=1.5)
        s.text(20, 32, "Live loop: one trajectory", 21, "bold", GRAY)
        role_box(s, 20, 44, 270, 74, "Model", "≈ processor", MODEL, sub_size=20)
        role_box(s, 350, 44, 270, 74, "Agent memory", "≈ memory hierarchy", MEMORY, sub_size=20)
        role_box(s, 20, 156, 600, 66, "Agent runtime", "≈ operating system", RUNTIME, sub_size=20)
        role_box(s, 20, 258, 270, 70, "Tool use", "≈ input and output", TOOLS, sub_size=20)
        s.box(350, 258, 270, 70, NEUTRAL, dash="6 5")
        s.text(485, 288, "Environment", 21, "bold", INK, "middle")
        s.text(485, 312, "repositories, services", 20, color=GRAY, anchor="middle")
        s.line(110, 118, 110, 154, GRAY); s.text(118, 142, "proposal", 20, color=GRAY)
        s.line(220, 156, 220, 120, GRAY); s.text(228, 142, "context", 20, color=GRAY)
        s.line(450, 118, 450, 154, GRAY); s.line(540, 156, 540, 120, GRAY)
        s.line(110, 222, 110, 256, GRAY); s.line(220, 258, 220, 224, GRAY)
        s.text(118, 246, "call", 20, color=GRAY); s.text(228, 246, "result", 20, color=GRAY)
        s.line(290, 284, 348, 284, GRAY); s.line(348, 306, 292, 306, GRAY)
        s.box(6, 356, 628, 108, NEUTRAL, rx=8, sw=1.5, dash="6 5")
        s.text(20, 380, "Across tasks", 21, "bold", GRAY)
        role_box(s, 20, 388, 290, 66, "Learning", "≈ recompiling", MODEL, size=21, sub_size=20)
        role_box(s, 330, 388, 290, 66, "Agents at scale", "≈ scale-out", NEUTRAL, size=21, sub_size=20)
        if k == 2:
            for x, y, w, h in ((14, 38, 282, 86),):
                s.box(x, y, w, h, ("none", FAULT[1], ""), rx=8, sw=4)
            s.box(14, 150, 612, 78, ("none", FAULT[1], ""), rx=8, sw=4)
            pass
        s.save(f"stochastic_step{k}")


# -------------------------------------------- trajectory strip with a real run
TRAJ = [  # (step, proposed command, observation), verbatim excerpts of the real run (RUN)
    ("11", "python reproduce_issue.py",
     "\u2717 mask * constant: unsupported\n  operand type(s) for |", True),
    ("13", "sed -i '521a\\  elif operand ...'", "(no output)  returncode 0", False),
    ("14", "python reproduce_issue.py",
     "IndentationError: expected an\nindented block  returncode 1", True),
    ("16", "git checkout .../ndarithmetic.py", "Updated 1 path from the index", False),
]


def traj():
    """Four turns of the real astropy repair, a 2 x 2 grid read left to right.
    Step 1: turns 11 and 13; step 2: turns 14 and 16; step 3: the record's total."""
    for k in (1, 2, 3):
        s = Svg(900, 400)
        for i, (st, cmd, obs, red) in enumerate(TRAJ):
            op = 1.0 if (i < 2 or k >= 2) else 0.0
            x0, y0 = 6 + (i % 2) * 450, 4 + (i // 2) * 172
            s.box(x0, y0, 438, 56, MODEL, rx=4, op=op)
            s.text(x0 + 10, y0 + 22, f"turn {st}: proposal", 19, "bold", MODEL[2], op=op)
            s.text(x0 + 10, y0 + 47, cmd, 20, color=INK, font=MONO, op=op)
            s.box(x0, y0 + 60, 438, 32, RUNTIME, rx=4, op=op)
            s.text(x0 + 10, y0 + 83, "runtime: run it in the container", 19, color=RUNTIME[2], op=op)
            s.box(x0, y0 + 96, 438, 66, FAULT if red else TOOLS, rx=4, op=op)
            s.text(x0 + 10, y0 + 122, obs, 19, color=INK, font=MONO, op=op)
        if k == 3:
            s.add('<path d="M6,350 L6,358 L894,358 L894,350" stroke="#7c3aed" stroke-width="2.5" fill="none"/>')
            s.text(450, 390, "The record keeps the failed patch: 28 calls, $0.34",
                   23, "bold", RUNTIME[2], "middle")
        s.save(f"traj_step{k}")


# ------------------------------------------------------------- data: model calls
def calls_hist():
    r = calcs.real_run()
    edges = list(range(0, 130, 10))
    ok = [int(x["model_calls"]) for x in r["rows"] if x["resolved"] == "1"]
    bad = [int(x["model_calls"]) for x in r["rows"] if x["resolved"] != "1"]
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    ax.hist([ok, bad], bins=edges, stacked=True, color=[C["data"], C["danger"]],
            edgecolor="white", linewidth=0.8)
    med = r["calls_median"]
    ax.axvline(med, color=C["text"], lw=1.5, ls="--")
    ax.text(med + 2, ax.get_ylim()[1] * 0.92, f"median {med:.0f} calls", fontsize=12,
            fontweight="bold", va="top")
    ax.text(70, ax.get_ylim()[1] * 0.62, f"accepted ({r['acc']})", color=C["data"], fontsize=12,
            fontweight="bold")
    ax.text(70, ax.get_ylim()[1] * 0.42, f"rejected ({r['rejected']})", color=C["danger"],
            fontsize=12, fontweight="bold")
    ax.set_xlabel("Model calls in one task's trajectory")
    ax.set_ylabel("Tasks")
    ax.set_xlim(0, 125)
    sp.save(fig, OUT / "calls_hist.svg")


def cmd_classes():
    r = calcs.real_run()
    rows = [("Read files", r["cmd"]["read"], "memory"), ("Write the copy", r["cmd"]["write"], "compute"),
            ("Run code", r["cmd"]["run"], "network"), ("Reach network", r["cmd"]["net"], "danger")]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, v, ck) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        ax.text(-150, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
        ax.text(v + 120, y, f"{v:,}  ({v / r['cmds']:.0%})" if v > 100 else f"{v}", va="center",
                fontsize=13, fontweight="bold", color=C[ck])
    ax.set_xlim(0, 11500)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Commands, all {r['cmds']:,} in one run of 500 tasks")
    sp.save(fig, OUT / "cmd_classes.svg")


# ------------------------------------------------------------ side-effect classes
def authority():
    s = Svg(900, 360)
    rows = [("A₃ irreversible", "no automated reversal: refund paid out, unversioned record", FAULT),
            ("A₂ compensable", "external effect with a tested undo: a seat hold released", TOOLS),
            ("A₁ workspace", "mutates a disposable copy: discarded on teardown", RUNTIME),
            ("A₀ read only", "changes nothing, yet secrets it reads can leak", MODEL)]
    for i, (lab, ex, role) in enumerate(rows):
        y = 10 + i * 68
        s.box(10, y, 880, 58, role, rx=4, sw=1.5)
        s.text(26, y + 37, lab, 23, "bold", role[2])
        s.text(270, y + 37, ex, 20, color=INK)
    s.text(450, 312, "Same delete: A\u2081 in a sandbox clone, A\u2083 on the only copy of the edits",
           21, "bold", FAULT[2], "middle")
    s.text(450, 346, "The level follows the consequence in this environment.",
           19, color=GRAY, anchor="middle", italic=True)
    s.save("authority")


# ------------------------------------------------------------- figure 3: H·S·A
def hsa():
    import math
    W, H = 900, 430
    x0, y0 = 250, 300            # origin
    xw = 600                     # 1 to 10,000 turns, log
    def hx(h):
        return x0 + xw * math.log10(h) / 3.6
    band = 64
    for k in (1, 2, 3):
        s = Svg(W, H)
        if k >= 2:
            names = [("A₀ read only", MODEL), ("A₁ workspace", RUNTIME),
                     ("A₂ compensable", TOOLS), ("A₃ irreversible", FAULT)]
            for i, (lab, role) in enumerate(names):
                yb = y0 - (i + 1) * band
                s.box(x0, yb, xw + 20, band, (role[0], "#e2e8f0", ""), rx=0, sw=1)
                s.text(x0 - 12, yb + band / 2 + 8, lab, 23, "normal", role[2] if i == 3 else INK, "end")
            for i, lab in enumerate(["S₁ context", "S₂ working copy", "S₃ durable"]):
                xx, yy = x0 - 22 * (i + 1), y0 + 20 * (i + 1)
                s.line(xx - 6, yy, xx + 6, yy, INK, sw=2, arrow=False)
                s.text(xx - 12, yy + 7, lab, 20, color=INK, anchor="end")
        s.line(x0, y0, x0, 20, INK, sw=2.5)
        s.line(x0, y0, x0 + xw + 30, y0, INK, sw=2.5)
        s.line(x0, y0, x0 - 80, y0 + 76, INK, sw=2.5)
        s.text(x0 + 8, 24, "Authority A", 22, "bold")
        s.text(x0 + xw + 30, y0 + 52, "Horizon H (turns)", 22, "bold", INK, "end")
        s.text(x0 - 84, y0 + 104, "State S", 22, "bold", INK, "middle")
        for h in (1, 10, 100, 1000):
            s.text(hx(h), y0 + 28, f"{h:,}", 21, color=GRAY, anchor="middle")
        s.add(f'<circle cx="{x0}" cy="{y0}" r="8" fill="{INK}"/>')
        s.text(x0 + 40, y0 + 54, "origin: one model call", 20, italic=True, color=GRAY)
        if k == 3:
            pts = [("Support agent", 3, 0, 1, INK, "H ≤ 3 · S₁ · A₀"),
                   ("Repair agent", 100, 1, 2, INK, "H ≈ 100 · S₂ · A₁"),
                   ("Research agent", 1000, 0, 3, INK, "H ≥ 1,000 · S₃ · A₀"),
                   ("Remediation agent", 20, 3, 3, FAULT[1], "H ≈ 20 · S₃ · A₃")]
            for name, h, a, st, col, coord in pts:
                px, py = hx(h), y0 - a * band - band / 2
                tx, ty = px - 14 * st, py + 12 * st
                s.add(f'<line x1="{tx}" y1="{ty}" x2="{px}" y2="{py}" stroke="{col}" stroke-width="2"/>')
                s.add(f'<circle cx="{px}" cy="{py}" r="9" fill="{col}"/>')
                right = name in ("Support agent", "Remediation agent")
                dy = -14 if name == "Research agent" else 0
                s.text(px + (14 if right else -24), py - 4 + dy, name, 23, "bold", col, "start" if right else "end")
                s.text(px + (14 if right else -24), py + 20 + dy, coord, 20, color=GRAY, anchor="start" if right else "end")
        s.save(f"hsa_step{k}")


# ------------------------------------------------------------- data: claimed vs accepted
def claimed():
    r = calcs.real_run()
    rows = [("Claimed done", r["submitted"], "neutral", f"{r['submitted']}  (all)"),
            ("Accepted", r["acc"], "data", f"{r['acc']}  ({r['acc'] / r['n']:.0%})"),
            ("Rejected", r["rejected"], "danger", f"{r['rejected']}  ({r['rejected'] / r['n']:.0%})")]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-12, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
        ax.text(v + 10, y, txt, va="center", fontsize=13, fontweight="bold", color=C[ck])
    ax.set_xlim(0, 760)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Tasks, hidden tests as judge")
    sp.save(fig, OUT / "claimed.svg")


# ------------------------------------------------------------- three fault models
def faults():
    cards = [("Fail-stop", "halts, so a heartbeat\nor exit code detects it", NEUTRAL),
             ("Byzantine", "arbitrary, but bounded:\n3f + 1 replicas outvote it", NEUTRAL),
             ("Fail-plausible", "well formed, credible, wrong:\nlooks like success", FAULT)]
    for k in (1, 2, 3):
        s = Svg(900, 230)
        for i, (t, d, role) in enumerate(cards):
            op = 1.0 if i < k else 0.0
            x = 10 + i * 300
            s.box(x, 10, 280, 150, role, sw=3 if role is FAULT else 2, op=op)
            s.text(x + 140, 54, t, 26, "bold", role[2] if role is FAULT else INK, "middle", op=op)
            s.text(x + 140, 96, d, 20, color=INK, anchor="middle", op=op)
        s.text(450, 206, "Crash monitors log it as a success; samples from one model can agree on it.",
               20, "bold", FAULT[2], "middle", op=1.0 if k == 3 else 0.0)
        s.save(f"faults_step{k}")


# ------------------------------------------------------------- closure evidence levels
def ladder():
    lv = [("Self-report", "the agent's own account", "“The fix is working perfectly.”"),
          ("Static analysis", "types and structure, no run", "(not used in this run)"),
          ("Visible tests", "cases the agent can read", "pytest test_ndarithmetic.py"),
          ("Sealed tests", "withheld: 1 new, 179 old tests", "test_nddata_bitmask_arithmetic"),
          ("Formal proof", "meets a written specification", "(not used)")]
    for k in (1, 2):
        s = Svg(900, 360)
        for i, (name, what, ex) in enumerate(lv):
            x = 20 + i * 58
            y = 290 - i * 62
            role = RUNTIME if i in (0, 2, 3) and k == 2 else NEUTRAL
            s.box(x, y, 300, 52, (role[0], RUNTIME[1] if role is RUNTIME else "#94a3b8", ""), rx=3, sw=2)
            s.text(x + 12, y + 34, f"{i + 1}  {name}", 23, "bold", INK)
            s.text(x + 312, y + 22, what, 20, color=GRAY)
            if k == 2:
                bold = "bold" if i in (0, 2, 3) else "normal"
                s.text(x + 312, y + 47, ex, 19, bold, RUNTIME[2] if i in (0, 2, 3) else GRAY, font=MONO)
        s.line(890, 340, 890, 20, GRAY, sw=2)
        s.text(880, 16, "stronger by default", 19, color=GRAY, anchor="end", italic=True)
        s.save(f"ladder_step{k}")


# ------------------------------------------------------------- data: METR horizon
def metr():
    import matplotlib.dates as mdates
    import numpy as np
    m = calcs.metr()
    rows = m["rows"]
    date = lambda r: dt.datetime.fromisoformat(r["release_date"])  # noqa: E731
    fig, ax = plt.subplots(figsize=(5.8, 2.55))
    for front in (False, True):
        pts = [r for r in rows if bool(r["is_sota"]) == front]
        x = [date(r) for r in pts]
        y = np.array([r["p50_min"] for r in pts])
        err = [y - np.array([r["p50_ci_low"] for r in pts]), np.array([r["p50_ci_high"] for r in pts]) - y]
        if front:
            ax.errorbar(x, y, yerr=err, fmt="o", color=C["memory"], ms=6, elinewidth=1, zorder=5)
        else:
            ax.errorbar(x, y, yerr=err, fmt="o", mfc="white", mec=C["neutral"], ecolor=C["grid"],
                        ms=5, elinewidth=1, zorder=4)
    fit = [r for r in rows if r["is_sota"] and r["release_date"] >= "2023-01-01" and r["p50_min"] <= 960]
    fx = mdates.date2num([date(r) for r in fit])
    slope, icpt = np.polyfit(fx, np.log2([r["p50_min"] for r in fit]), 1)
    xs = np.linspace(mdates.datestr2num("2023-01-01"), mdates.datestr2num("2026-07-01"), 50)
    ax.plot(mdates.num2date(xs), 2 ** (icpt + slope * xs), "--", color=C["neutral"], lw=1.5)
    ax.set_yscale("log")
    ticks = {"1 s": 1 / 60, "1 min": 1, "1 hr": 60, "8 hr": 480, "40 hr": 2400}
    ax.set_yticks(list(ticks.values()))
    ax.set_yticklabels(list(ticks))
    ax.minorticks_off()
    ax.set_ylim(1 / 90, 6000)
    ax.set_ylabel("Task length done\nhalf the time")
    first = rows[0]
    last = [r for r in rows if r["is_sota"]][-1]
    ax.annotate(f"{m['first_s']:.0f} s", (date(first), first["p50_min"]), xytext=(8, -2),
                textcoords="offset points", fontsize=12, fontweight="bold", color=C["memory"])
    ax.annotate(f"{m['last_h']:.0f} hr", (date(last), last["p50_min"]), xytext=(-10, 6),
                textcoords="offset points", fontsize=12, fontweight="bold", color=C["memory"], ha="right")
    ax.text(dt.datetime(2018, 12, 1), 200,
            f"Frontier horizon doubles every\n{m['recent_months']:.1f} months since 2023",
            fontsize=12, fontweight="bold", color=C["text"])
    ax.set_xlabel("Model release date (filled: frontier at release)")
    sp.save(fig, OUT / "metr.svg")


# ------------------------------------------------------------- horizon compounding
def horizon_curve():
    import numpy as np
    h = calcs.horizon()
    H = np.logspace(0, np.log10(300), 300)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.45))
        ax.plot(H, calcs.STEP_OK ** H, color=C["danger"], lw=3)
        ax.text(1.3, 0.2, "99% per step,\nno checks", color=C["danger"], fontsize=12, fontweight="bold")
        for hh, v, dx, dy in ((10, h["p10"], -48, -24), (100, h["p100"], 6, 6)):
            ax.plot([hh], [v], "o", color=C["danger"], ms=8, zorder=5)
            ax.annotate(f"{v:.1%}", (hh, v), xytext=(dx, dy), textcoords="offset points",
                        fontsize=13, fontweight="bold", color=C["danger"])
        if k >= 2:
            ax.plot(H, (1 - h["eps_req"]) ** H, color=C["data"], lw=3)
            ax.text(1.3, 0.52, f"{h['eps_req']:.2%} per step: 90% at 100 turns",
                        ha="left",
                        fontsize=12, fontweight="bold", color=C["data"])
        if k >= 3:
            v200 = calcs.STEP_OK ** 200
            ax.plot([200], [v200], "o", color=C["danger"], ms=8, zorder=5)
            ax.annotate(f"twice the turns: {v200:.1%} = {h['p100']:.3f}²", (200, v200), xytext=(-16, -4),
                        textcoords="offset points", ha="right", va="center", fontsize=12, fontweight="bold",
                        color=C["text"])
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 300])
        ax.set_xticklabels(["1", "10", "100", "300"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xlabel("Horizon H (turns)")
        ax.set_ylabel("Task success")
        sp.save(fig, OUT / f"horizon_step{k}.svg")


# ------------------------------------------------------------- NM 1.2 bars
def cost_bars():
    c = calcs.cost()
    rows = [("Baseline", c["base"], "neutral"), ("Half token price", c["price_cut"], "memory"),
            ("Acceptance 60%", c["better"], "data")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(3.3, 1.5))
        for i, (lab, v, ck) in enumerate(rows):
            y = 2 - i
            shown = i == 0 or k == 2
            ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if shown else 0.0)
            ax.text(-0.15, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
            txt = f"${v:.2f}" if shown else "?"
            ax.text((v if shown else 0) + 0.12, y, txt, va="center", fontsize=13, fontweight="bold",
                    color=C[ck] if shown else C["danger"])
        ax.set_xlim(0, 6.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Dollars per accepted task")
        sp.save(fig, OUT / f"costbars_step{k}.svg")


def real_cost():
    r = calcs.real_run()
    rows = [("Accepted runs only", r["winners_only"], "neutral"),
            ("Mean attempt", r["mean_attempt"], "memory"),
            ("Per accepted task", r["per_accepted"], "danger")]
    fig, ax = plt.subplots(figsize=(5.8, 1.75))
    for i, (lab, v, ck) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-0.01, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
        ax.text(v + 0.01, y, f"${v:.2f}", va="center", fontsize=13, fontweight="bold", color=C[ck])
    ax.set_xlim(0, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Dollars per task; {r['n']} tasks, one attempt each, {r['acc']} accepted")
    sp.save(fig, OUT / "realcost.svg")


# ------------------------------------------------------------- workflow vs loop
def workflow():
    s = Svg(900, 250)
    steps = ["Clone", "Install deps", "Run linter"]
    for i, t in enumerate(steps):
        x = 10 + i * 150
        s.box(x, 70, 130, 60, NEUTRAL)
        s.text(x + 65, 107, t, 20, "bold", INK, "middle")
        s.line(x + 130, 100, x + 148, 100, GRAY)
    s.text(205, 52, "Fixed workflow: order never changes", 20, "bold", GRAY, "middle")
    s.box(470, 30, 260, 140, MODEL, rx=10)
    s.text(600, 62, "Model-directed loop", 21, "bold", MODEL[2], "middle")
    s.text(600, 96, "inspect → hypothesize", 20, color=INK, anchor="middle")
    s.text(600, 124, "→ edit → read the log", 20, color=INK, anchor="middle")
    s.path("M 700 170 C 700 200, 500 200, 500 172", MODEL[1])
    s.text(600, 218, "next step depends on what the last one found", 18, color=GRAY, anchor="middle", italic=True)
    s.line(730, 100, 768, 100, GRAY)
    s.box(770, 70, 120, 60, NEUTRAL)
    s.text(830, 98, "Sealed", 20, "bold", INK, "middle")
    s.text(830, 120, "tests", 20, "bold", INK, "middle")
    s.save("workflow")


# ------------------------------------------------------------- figure 5: six phases
def sixphase():
    P = [(1, "Context assembly", "runtime", RUNTIME, 20, 20), (2, "Model invocation", "model", MODEL, 320, 20),
         (3, "Validate, authorize", "runtime", RUNTIME, 620, 20), (4, "Tool execution", "tool", TOOLS, 620, 210),
         (5, "Observation, evidence", "runtime", RUNTIME, 320, 210),
         (6, "Assess vs. contract", "runtime", RUNTIME, 20, 210)]
    for k in (1, 2, 3):
        s = Svg(900, 420)
        for n, t, who, role, x, y in P:
            op = 1.0 if (n <= 3 or k >= 2) else 0.0
            s.box(x, y, 260, 76, role, op=op, sw=3 if role is MODEL else 2)
            s.text(x + 130, y + 33, f"{n}  {t}", 21, "bold", INK, "middle", op=op)
            s.text(x + 130, y + 62, who, 21, color=role[2], anchor="middle", op=op)
        s.line(280, 58, 318, 58, GRAY); s.line(580, 58, 618, 58, GRAY)
        s.text(299, 48, "", 1)
        if k >= 2:
            s.line(780, 96, 780, 208, RUNTIME[1], sw=4)
            s.text(792, 160, "authorized", 19, "bold", RUNTIME[2])
            s.line(620, 248, 582, 248, GRAY); s.line(320, 248, 282, 248, GRAY)
            s.line(70, 210, 70, 98, GRAY); s.text(80, 160, "next turn", 19, color=GRAY)
            s.text(600, 312, "unknown if timed out", 20, italic=True, color=GRAY, anchor="middle")
        if k >= 3:
            s.path("M 680 96 L 680 140 L 460 140 L 460 208", FAULT[1], dash="7 5")
            s.text(570, 132, "rejected: no execution", 18, "bold", FAULT[2], "middle")
            s.box(20, 330, 250, 62, OK)
            s.text(145, 357, "Accept deliverable", 20, "bold", OK[2], "middle")
            s.text(145, 383, "evidence meets contract", 19, color=INK, anchor="middle")
            s.box(290, 330, 280, 62, FAULT)
            s.text(430, 357, "Stop without acceptance", 20, "bold", FAULT[2], "middle")
            s.text(430, 383, "unresolved or escalated", 19, color=INK, anchor="middle")
            s.line(110, 286, 110, 328, OK[1]); s.line(220, 286, 330, 328, FAULT[1])
            s.box(600, 330, 290, 62, NEUTRAL, dash="6 5")
            s.text(745, 357, "Wait: approval or a long test", 18, "bold", INK, "middle")
            s.text(745, 383, "task alive, model idle", 19, color=GRAY, anchor="middle")
        s.save(f"sixphase_step{k}")


if __name__ == "__main__":
    cleanup(); loop(); stochastic(); traj(); calls_hist(); cmd_classes(); authority(); hsa()
    claimed(); faults(); ladder(); metr(); horizon_curve(); cost_bars(); real_cost(); workflow()
    sixphase()
    print("figures written")
