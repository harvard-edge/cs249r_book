#!/usr/bin/env python3
"""Figures for Lecture 7 (Tool Calling), Volume III.

Diagrams are SVG in the Volume III agentic-stack palette (runtime violet,
tools amber, model sky blue, memory green, faults red, structure slate).
Plots use slides/assets/kit/slideplot.py. Every number drawn comes from
calcs.py (which names its source); diagrams that carry no data are
schematic redraws of the chapter's figures (TC = Tool Calling chapter):

  loop_step1-4      TC Sec. 7.2.1 figure 1 (mediated round trip), redrawn
  cas_step1-3       TC Sec. 7.3.3 (conditional update, r17 -> r18)
  gh_bars           measured: 131 GitHub MCP server descriptors (calcs)
  incident_step1-2  TC War Story 7.1 (Invariant Labs, May 2025)
  worlds_step1-3    TC Sec. 7.5 (three outcomes behind one timeout)
  dup_plot          measured retry harness vs NM 7.1 formula (calcs)
  delayed_step1-3   TC Sec. 7.5.1 figure (delayed attempt), redrawn
  opid_step1-2      TC Sec. 7.5.3 figure 2 (operation-ID lookup), redrawn
  headtail_step1-3  TC Sec. 7.6.1 figure 3 (head, notice, tail), redrawn
  obs_hist          measured: 18,076 observations of 500 real repair runs
  job_states        TC Sec. 7.8.2 (job handle states)
  poll_plot         NM 7.3 (illustrative inputs; computed in calcs)
  catalog_step1-3   measured GitHub catalog vs NM 7.4 (calcs)
  deep_shallow      TC Sec. 7.9.2 figure 4, redrawn for a column
  click_step1-3     TC Sec. 7.10.1 (stale click target), schematic

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from deckvals import num, val  # noqa: E402

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
FLOOR = 20   # smallest SVG font, in viewBox units (about 8 pt at slide size)


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
            for k, col in (("blue", ROLE["model"][1]), ("violet", ROLE["sys"][1]),
                           ("amber", ROLE["tool"][1]), ("green", ROLE["memory"][1]),
                           ("red", ROLE["fault"][1]), ("gray", GRAY)))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" font-family="{FONT}">'
               f'<defs>{markers}<pattern id="hatch" width="10" height="10" '
               f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               f'<line x1="0" y1="0" x2="0" y2="10" stroke="#cbd5e1" stroke-width="4"/>'
               f'</pattern></defs><rect x="0" y="0" width="{self.w}" height="{self.h}" fill="white"/>'
               + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)

    # primitives
    def rect(self, x, y, w, h, role="struct", dash=False, rx=8, sw=2, fill=None):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill or f}" stroke="{s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, fs=22, color=INK, anchor="middle", weight="normal", mono=False,
             italic=False):
        fs = max(fs, FLOOR)   # legibility floor at slide size
        fam = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{fam}{it}>{s}</text>')

    def box(self, x, y, w, h, role, title, sub=None, fs=22, sfs=19, dash=False, mono_sub=False,
            mono_title=False):
        self.rect(x, y, w, h, role, dash=dash)
        _, _, tc = ROLE[role]
        if sub:
            self.text(x + w / 2, y + h / 2 - 4, title, fs, INK, weight="bold", mono=mono_title)
            self.text(x + w / 2, y + h / 2 + sfs + 2, sub, sfs, tc, mono=mono_sub)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, title, fs, INK, weight="bold",
                      mono=mono_title)

    def arrow(self, x1, y1, x2, y2, color="gray", dash=False, sw=2.5, path=None):
        col = {"blue": ROLE["model"][1], "violet": ROLE["sys"][1], "amber": ROLE["tool"][1],
               "green": ROLE["memory"][1], "red": ROLE["fault"][1], "gray": GRAY}[color]
        d = ' stroke-dasharray="8 6"' if dash else ""
        geom = path or f"M{x1},{y1} L{x2},{y2}"
        self.add(f'<path d="{geom}" fill="none" stroke="{col}" stroke-width="{sw}"{d} '
                 f'marker-end="url(#a{color})"/>')

    def line(self, x1, y1, x2, y2, color=GRAY, sw=2, dash=False):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}"{d}/>')

    def cross(self, x, y, r=11, color=None):
        col = color or ROLE["fault"][1]
        self.add(f'<path d="M{x-r},{y-r} L{x+r},{y+r} M{x-r},{y+r} L{x+r},{y-r}" '
                 f'stroke="{col}" stroke-width="5"/>')

    def dot(self, x, y, r=12, color=None):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color or ROLE["sys"][1]}"/>')


# ------------------------------------------------------- 1. mediated loop
def loop():
    for k in range(1, 5):
        s = Svg(900, 440)
        s.box(10, 50, 160, 150, "model", "Model", "turn t", fs=26, sfs=20)
        s.text(90, 185, "call_1..3", 19, ROLE["model"][2], mono=True)
        s.rect(200, 14, 480, 300, "sys", rx=12, fill="#faf5ff")
        s.text(216, 44, "Runtime", 22, ROLE["sys"][2], anchor="start", weight="bold")
        s.arrow(170, 125, 215, 125, "blue")
        s.box(218, 85, 135, 80, "sys", "Validate", "schema", fs=23)
        if k >= 2:
            s.arrow(353, 125, 372, 125, "violet")
            s.box(374, 85, 135, 80, "sys", "Authorize", "grant", fs=23)
        if k >= 3:
            s.arrow(509, 125, 528, 125, "violet")
        if k >= 3:
            s.box(530, 85, 140, 80, "sys", "Dispatch", "track it", fs=23)
            s.box(505, 205, 170, 80, "sys", "Shape", "bound, match ID", fs=23, sfs=17)
            s.arrow(665, 125, 712, 125, "amber")
            s.box(714, 70, 176, 230, "tool", "Tool", "effect lands here", fs=26, sfs=18)
            s.arrow(714, 245, 667, 245, "amber")
        # results strip
        s.rect(200, 345, 480, 86, "memory", rx=10)
        s.line(285, 165, 285, 352, ROLE["fault"][1], 2.5, dash=True)
        s.box(215, 355, 140, 66, "white", "call_3", "invalid", fs=20, sfs=18, mono_title=True)
        if k >= 2:
            s.line(441, 165, 441, 352, ROLE["fault"][1], 2.5, dash=True)
            s.box(371, 355, 140, 66, "white", "call_2", "not granted", fs=20, sfs=18,
                  mono_title=True)
        if k >= 3:
            s.arrow(597, 285, 597, 352, "green")
            s.box(527, 355, 140, 66, "white", "call_1", "result", fs=20, sfs=18, mono_title=True)
        if k >= 4:
            s.arrow(0, 0, 0, 0, "gray", path="M200,388 L90,388 L90,203")
            s.text(140, 375, "next turn", 19, GRAY)
            s.text(780, 388, "one observation", 19, ROLE["memory"][2])
            s.text(780, 412, "per call ID", 19, ROLE["memory"][2])
        s.save(f"loop_step{k}")


# ------------------------------------------------------- 2. lost update
def cas():
    for k in range(1, 4):
        s = Svg(900, 420)
        lanes = [("Agent", 70, "model"), ("pool.py", 200, "tool"), ("Teammate", 330, "memory")]
        for name, y, role in lanes:
            s.text(20, y + 8, name, 23, ROLE[role][2], anchor="start", weight="bold")
            s.line(140, y, 880, y, "#cbd5e1", 3)
        # file state band
        s.box(150, 178, 170, 44, "tool", "r17", fs=22, mono_title=True)
        s.arrow(240, 178, 240, 82, "blue")
        s.text(250, 130, "read r17", 19, ROLE["model"][2], anchor="start")
        s.box(335, 178, 165, 44, "tool", "r18", fs=22, mono_title=True)
        s.arrow(400, 320, 400, 224, "green")
        s.text(410, 280, "teammate commits", 19, ROLE["memory"][2], anchor="start")
        if k == 2:
            s.arrow(560, 82, 560, 176, "red")
            s.text(570, 130, "write patch", 19, ROLE["fault"][2], anchor="start")
            s.box(515, 178, 360, 44, "fault", "r19: teammate change lost", fs=21)
        if k == 3:
            s.arrow(560, 82, 560, 176, "violet")
            s.text(550, 112, "write if version = r17", 19, ROLE["sys"][2], anchor="end")
            s.box(515, 178, 360, 44, "sys", "rejected: version conflict", fs=21)
            s.arrow(820, 224, 820, 300, "violet", dash=True)
            s.text(810, 290, "re-read r18,", 19, ROLE["sys"][2], anchor="end")
            s.text(810, 314, "propose again", 19, ROLE["sys"][2], anchor="end")
        s.text(880, 405, "check and write commit as one step at the file service",
               18, GRAY, anchor="end", italic=True)
        s.save(f"cas_step{k}")


# ------------------------------------------- 3. real GitHub descriptors
def gh_bars():
    rows = [("Tools listed", num("gh_n"), "neutral"),
            ("Mutating", num("gh_mut"), "compute"),
            ("Repo is any string", num("gh_repo"), "danger"),
            ("Declared destructive", num("gh_destr"), "network"),
            ("Declared idempotent", num("gh_idem"), "data")]
    sp.hbars(str(OUT / "gh_bars.svg"), rows, "", log=False, size=(3.6, 2.0),
             value_fmt="{:,.0f}", xlabel="Tools in the GitHub MCP server (measured)")


# ---------------------------------------------------- 4. the incident
def incident():
    calls = [("1", "list_issues", "public repo", "memory"),
             ("2", "get_file_contents", "private repo", "fault"),
             ("3", "create_pull_request", "public repo", "tool")]
    for k in (1, 2):
        s = Svg(900, 400)
        s.text(20, 36, "Task: address the issues in one public repo", 22, INK, anchor="start",
               weight="bold")
        for i, (n, name, target, role) in enumerate(calls):
            x = 20 + i * 295
            r = "white" if (k == 1 or i == 0) else role
            s.rect(x, 70, 270, 120, r)
            s.text(x + 135, 112, name, 21, INK, weight="bold", mono=True)
            s.text(x + 135, 145, target, 20, GRAY)
            s.text(x + 135, 176, "schema ok, token ok", 18, ROLE["memory"][2])
            if i < 2:
                s.arrow(x + 270, 130, x + 293, 130, "gray")
        s.text(155, 230, "reads the attacker's issue", 19, GRAY)
        if k == 2:
            s.cross(450, 70, 16)
            s.box(315, 255, 270, 95, "fault", "Grant check refuses",
                  "target not in the task", fs=21, sfs=18)
            s.arrow(450, 255, 450, 194, "red")
            s.text(745, 230, "never dispatched", 19, ROLE["fault"][2])
            s.text(20, 390, "The first private read is read-only, and a per-call grant check refuses it.",
                   19, GRAY, anchor="start", italic=True)
        s.save(f"incident_step{k}")
    # The refused chain again, stacked for a half-width column (the retrieval slide)
    s = Svg(640, 400)
    s.text(20, 32, "Task: address the issues in one public repo", 22, INK, anchor="start",
           weight="bold")
    for i, (n, name, target, role) in enumerate(calls):
        y = 52 + i * 116
        s.rect(20, y, 280, 96, "white" if i == 0 else role)
        s.text(160, y + 30, name, 21, INK, weight="bold", mono=True)
        s.text(160, y + 57, target, 20, GRAY)
        s.text(160, y + 83, "schema ok, token ok", 20, ROLE["memory"][2])
        if i < 2:
            s.arrow(160, y + 96, 160, y + 114, "gray")
    s.text(320, 108, "reads the attacker's issue", 20, GRAY, anchor="start")
    s.cross(300, 168, 14)
    s.box(360, 172, 270, 96, "fault", "Grant check refuses", "target not in the task",
          fs=21, sfs=20)
    s.arrow(360, 220, 304, 220, "red")
    s.text(320, 342, "never dispatched", 20, ROLE["fault"][2], anchor="start")
    s.save("incident_col")


# ------------------------------------------------ 5. three worlds
def worlds():
    titles = [("Request lost", "no effect"), ("Reply lost", "effect happened"),
              ("Still in flight", "effect still to come")]
    for k in range(1, 4):
        s = Svg(900, 430)
        for i in range(k):
            x0 = 20 + i * 295
            s.rect(x0, 10, 270, 410, "struct", rx=10)
            s.text(x0 + 135, 44, titles[i][0], 23, INK, weight="bold")
            s.text(x0 + 135, 72, titles[i][1], 20, ROLE["fault"][2] if i else GRAY)
            rx, sx = x0 + 45, x0 + 225
            s.text(rx, 108, "runtime", 18, GRAY)
            s.text(sx, 108, "service", 18, GRAY)
            s.line(rx, 118, rx, 405, "#94a3b8", 3)
            s.line(sx, 118, sx, 405, "#94a3b8", 3)
            s.line(rx - 14, 270, rx + 14, 270, INK, 4)
            s.text(rx + 18, 276, "timeout", 17, INK, anchor="start")
            if i == 0:
                s.line(rx, 140, rx + 85, 175, ROLE["sys"][1], 3)
                s.cross(rx + 92, 178, 10)
            elif i == 1:
                s.arrow(rx, 140, sx - 4, 190, "violet")
                s.dot(sx, 195, 11, ROLE["fault"][1])
                s.text(sx - 5, 265, "restart", 18, ROLE["fault"][2], anchor="end")
                s.line(sx, 200, sx - 80, 240, "#94a3b8", 3)
                s.cross(sx - 86, 243, 10)
            else:
                s.arrow(rx, 140, sx - 4, 360, "amber", dash=True)
                s.dot(sx, 365, 11, ROLE["fault"][1])
                s.text(sx - 10, 398, "restart", 18, ROLE["fault"][2], anchor="end")
        s.save(f"worlds_step{k}")


# ------------------------------------------------ 6. duplicates measured
def dup_plot():
    rows = calcs.HARNESS
    # blind retries by rising reply loss, then the deduplicated run
    order = sorted((r for r in rows if not r["dedup"]), key=lambda r: r["p_rep"]) + \
        [r for r in rows if r["dedup"]]
    labels = [f"{r['p_rep']:.0%}" + (" +\ndedup" if r["dedup"] else "") for r in order]
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    x = np.arange(len(order))
    pred = [r["predicted"] for r in order]
    meas = [r["duplicates"] for r in order]
    ax.bar(x - 0.19, pred, 0.36, color=C["neutral_l"], edgecolor=C["neutral"], label="formula")
    ax.bar(x + 0.19, meas, 0.36, color=[C["danger"]] * 3 + [C["data"]])
    for xi, p, m in zip(x, pred, meas):
        if p:
            ax.text(xi - 0.21, p + 15, f"{p:.0f}", ha="right", fontsize=10, color=C["neutral"])
        ax.text(xi + 0.21, m + 15, f"{m}", ha="left", fontsize=11, fontweight="bold",
                color=C["data"] if m == 0 else C["danger"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel(f"Reply loss (request loss {val('hx_req')}%)")
    ax.set_ylabel(f"Duplicate restarts\nper {val('hx_n')} ops")
    ax.set_ylim(0, 1100)
    ax.text(-0.45, 990, "gray = formula, red = measured", fontsize=11, color=C["text"])
    sp.save(fig, OUT / "dup_plot.svg")


# --------------------------------------------- 7. delayed attempt
def delayed():
    for k in range(1, 4):
        s = Svg(900, 430)
        rx, sx = 120, 520
        s.text(rx, 34, "runtime", 23, GRAY, weight="bold")
        s.text(sx, 34, "service", 23, GRAY, weight="bold")
        s.line(rx, 48, rx, 420, "#94a3b8", 4)
        s.line(sx, 48, sx, 420, "#94a3b8", 4)
        s.text(rx - 16, 80, "A", 26, ROLE["tool"][2], anchor="end", weight="bold")
        s.arrow(rx, 72, sx - 6, 372, "amber", dash=True, sw=3)
        s.line(rx - 12, 150, rx + 12, 150, INK, 4)
        s.text(rx - 18, 157, "timeout", 19, INK, anchor="end")
        if k >= 2:
            s.arrow(rx, 185, sx - 6, 200, "gray")
            s.arrow(sx, 222, rx + 6, 237, "gray")
            s.text(430, 186, "probe", 19, GRAY)
            s.text(200, 262, "unchanged", 19, GRAY)
        if k >= 3:
            s.text(rx - 16, 300, "B", 26, ROLE["sys"][2], anchor="end", weight="bold")
            s.arrow(rx, 292, sx - 8, 326, "violet", sw=3)
            s.dot(sx, 330, 12, ROLE["sys"][1])
            s.text(sx + 20, 337, "restart 1", 20, ROLE["sys"][2], anchor="start", weight="bold")
            s.dot(sx, 376, 12, ROLE["fault"][1])
            s.text(sx + 20, 383, "restart 2", 20, ROLE["fault"][2], anchor="start", weight="bold")
            s.rect(640, 60, 250, 170, "struct")
            s.text(765, 95, "Probe shows now,", 19, INK)
            s.text(765, 121, "not what is still", 19, INK)
            s.text(765, 147, "in flight.", 19, INK)
            s.text(765, 185, "Retry only on proof", 19, ROLE["fault"][2], weight="bold")
            s.text(765, 210, "the attempt expired", 19, ROLE["fault"][2], weight="bold")
        s.save(f"delayed_step{k}")


# -------------------------------------------------- 8. operation ID
def opid():
    # Drawn on the slot's own shape (about 2.9 to 1), so no label falls below the floor.
    for k in (1, 2):
        s = Svg(1100, 380)
        s.rect(8, 6, 340, 368, "sys", rx=12, fill="#faf5ff")
        s.text(24, 34, "Runtime", 22, ROLE["sys"][2], anchor="start", weight="bold")
        s.box(28, 48, 300, 64, "sys", "Admit intent", "op ID made once", fs=22, sfs=20)
        s.arrow(178, 112, 178, 132, "violet")
        s.box(28, 134, 300, 64, "sys", "Attempt k", "same op ID, new attempt ID", fs=22, sfs=20)
        s.arrow(178, 198, 178, 218, "red", dash=True)
        s.box(28, 220, 300, 64, "fault", "Timeout", "effect unknown", fs=22, sfs=20)
        s.arrow(0, 0, 0, 0, "violet", path="M28,252 L16,252 L16,166 L26,166")
        s.arrow(178, 284, 178, 304, "red", dash=True)
        s.box(28, 306, 300, 62, "white", "Unresolved", "attempts used up", fs=22, sfs=20,
              dash=True)
        if k == 2:
            s.rect(372, 6, 720, 368, "tool", rx=12, fill="#fffbeb")
            s.text(388, 34, "Endpoint", 22, ROLE["tool"][2], anchor="start", weight="bold")
            s.arrow(328, 166, 398, 166, "violet")
            s.box(400, 126, 250, 80, "tool", "Look up op ID", "caller, scope, args", fs=22,
                  sfs=20)
            outs = [("Completed", "return kept result", "memory", 48),
                    ("Still running", "return pending", "struct", 128),
                    ("Args differ", "reject the reuse", "fault", 208),
                    ("Not found", "execute + record", "tool", 288)]
            for t, sub, role, y in outs:
                s.arrow(0, 0, 0, 0, "gray", path=f"M650,166 L680,166 L680,{y + 32} L708,{y + 32}")
                s.box(710, y, 362, 64, role, t, sub, fs=22, sfs=20)
            s.text(525, 330, "effect + record:", 20, GRAY)
            s.text(525, 356, "one transaction", 20, GRAY)
        s.save(f"opid_step{k}")


# ---------------------------------------------------- 9. head and tail
def headtail():
    for k in range(1, 4):
        s = Svg(900, 400)
        s.text(20, 34, "output stream, any length", 20, GRAY, anchor="start")
        s.box(20, 48, 130, 60, "memory", "head", fs=22)
        s.add('<rect x="150" y="48" width="390" height="60" fill="url(#hatch)" '
              'stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 5"/>')
        s.text(345, 85, "middle", 21, GRAY)
        s.box(540, 48, 130, 60, "memory", "tail", fs=22)
        s.box(20, 160, 200, 90, "memory", "head buffer", "fills once", fs=21, sfs=18)
        s.arrow(85, 108, 85, 158, "green")
        if k >= 2:
            s.box(250, 160, 200, 90, "struct", "omission count", "bytes, none kept",
                  fs=20, sfs=18)
            s.arrow(345, 108, 345, 158, "gray", dash=True)
            s.box(480, 160, 200, 90, "memory", "tail ring", "overwrites oldest",
                  fs=21, sfs=18)
            s.arrow(605, 108, 605, 158, "green")
            s.text(350, 290, "capture memory is fixed, however long the log",
                   19, ROLE["memory"][2], weight="bold")
        if k >= 3:
            s.text(800, 34, "staged view", 21, INK, weight="bold")
            items = [("exit status", "struct"), ("head", "memory"),
                     ("notice: N omitted", "struct"), ("tail", "memory"),
                     ("handle obs_42", "sys")]
            for i, (t, role) in enumerate(items):
                s.box(712, 50 + i * 56, 176, 46, role, t, fs=18)
            s.arrow(680, 205, 708, 205, "green")
            s.text(445, 345, "budget = head + notice + tail", 22, INK, weight="bold")
            s.text(445, 378, "a handle reads the dropped middle on request", 19, GRAY)
        s.save(f"headtail_step{k}")


# ---------------------------------------- 10. real observation sizes
def obs_hist():
    raw = np.array([max(1, r) for r in calcs.OBS_RAW])
    bins = np.logspace(0, 6, 25)
    fig, ax = plt.subplots(figsize=(4.0, 2.05))
    cnt, edges = np.histogram(raw, bins=bins)
    for c_, lo, hi in zip(cnt, edges[:-1], edges[1:]):
        col = C["danger"] if lo >= calcs.STAGE_CAP else C["memory"]
        ax.bar(lo, c_, width=hi - lo, align="edge", color=col, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(calcs.STAGE_CAP, color=C["danger"], ls="--", lw=1.5)
    ax.text(calcs.STAGE_CAP * 1.3, 2.5e4, f"cap:\n{val('so_cap')}", fontsize=10, color=C["danger"], va="top")
    ax.annotate(f"{val('so_cut')} cut\n({val('so_cut_pct')}%)", xy=(3e4, 40), xytext=(2e5, 300),
                fontsize=10, color=C["danger"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["danger"]))
    ax.set_ylim(0.7, 6e4)
    ax.set_xlabel("Characters of output per command (log)")
    ax.set_ylabel("Commands (log)")
    ax.set_xlim(1, 1.5e6)
    sp.save(fig, OUT / "obs_hist.svg")


# ------------------------------------------------- 11. job states
def job_states():
    s = Svg(620, 470)
    s.box(10, 10, 270, 90, "tool", "accepted", "queued (202)", fs=26, sfs=22)
    s.arrow(145, 100, 145, 140, "amber")
    s.box(10, 142, 270, 90, "tool", "running", "last seen at t", fs=26, sfs=22)
    s.arrow(280, 165, 338, 80, "green")
    s.box(340, 10, 270, 90, "memory", "completed", "result + scope", fs=26, sfs=22)
    s.arrow(280, 200, 338, 260, "red")
    s.box(340, 230, 270, 90, "fault", "failed", "diagnostic", fs=26, sfs=22)
    s.arrow(145, 232, 145, 290, "gray", dash=True)
    s.box(10, 292, 270, 96, "struct", "unavailable", "last observed running", fs=26, sfs=21,
          dash=True)
    s.text(10, 425, "unavailable is no failure; no rollback,", 22, ROLE["fault"][2], anchor="start",
           weight="bold")
    s.text(10, 455, "the runtime keeps watching", 22, GRAY, anchor="start")
    s.save("job_states")


# ------------------------------------------------- 12. polling cost
def poll_plot():
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    ivs = [s.poll_interval.to("s").magnitude for s in calcs.POLL_SWEEP]
    costs = [calcs.usd(s.result.cost) for s in calcs.POLL_SWEEP]
    ax.plot(ivs, costs, "-o", color=C["compute"], lw=2.5, ms=7)
    for i, c_ in zip(ivs, costs):
        ax.text(i + 1.2, c_ + 0.03, f"${c_:.2f}", fontsize=11,
                color=C["compute"], fontweight="bold")
    runtime = calcs.RUNTIME_POLL.poll_interval.to("s").magnitude
    ax.plot([runtime], [0], "s", color=C["data"], ms=10, clip_on=False)
    ax.annotate(f"runtime polls every {runtime:.0f} s: $0", xy=(10.5, 0.01), xytext=(24, 0.45),
                fontsize=11, color=C["data"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["data"]))
    ax.text(22, 0.80, "model polls: halving the delay\ndoubles the cost", fontsize=11,
            color=C["compute"])
    ax.set_xlabel("Poll interval = worst detection delay (s)")
    ax.set_ylabel(f"Extra $ per {val('pc_job')} s job")
    ax.set_xlim(0, 66)
    ax.set_ylim(0, 1.0)
    sp.save(fig, OUT / "poll_plot.svg")


# ----------------------------------------------- 13. catalog tokens
def catalog():
    rows = [(f"All {val('gh_n')} GitHub tools", num("gh_total"), "danger", f"{val('gh_total')} (measured)"),
            (f"{val('cat_reg')} tools × {val('cat_desc')}", num("cat_full"), "neutral", val("cat_full")),
            (f"{val('cat_sel')} selected + {val('cat_over')}", num("cat_selected"), "data", val("cat_selected"))]
    for k in range(1, 4):
        sp.hbars(str(OUT / f"catalog_step{k}.svg"), rows, "", log=False, step=k,
                 size=(5.8, 1.45), xlabel="Descriptor tokens on every model call")


# ------------------------------------------- 14. deep vs shallow (column)
def deep_shallow():
    s = Svg(600, 430)
    s.text(10, 30, "Shallow: 4 model calls", 22, INK, anchor="start", weight="bold")
    names = ["open", "seek", "write", "close"]
    for i, n in enumerate(names):
        x = 10 + i * 148
        s.box(x, 46, 128, 56, "model", f"call {i + 1}", fs=20)
        s.box(x, 132, 128, 52, "tool", n, fs=20, mono_title=True)
        s.arrow(x + 50, 102, x + 50, 130, "blue")
        s.arrow(x + 80, 132, x + 80, 104, "green")
        if i < 3:
            s.arrow(x + 128, 74, x + 146, 74, "gray")
    s.line(10, 214, 590, 214, "#cbd5e1", 2)
    s.text(10, 250, "Deep: 1 model call", 22, INK, anchor="start", weight="bold")
    s.box(10, 266, 128, 56, "model", "call 1", fs=20)
    s.rect(10, 350, 580, 70, "tool")
    s.text(24, 392, "apply_patch", 20, INK, anchor="start", mono=True, weight="bold")
    s.box(190, 360, 170, 50, "white", "check version", fs=18)
    s.box(400, 360, 170, 50, "white", "apply change", fs=18)
    s.arrow(360, 385, 398, 385, "gray")
    s.arrow(74, 322, 74, 348, "blue")
    s.box(190, 266, 170, 56, "fault", "conflict", fs=19)
    s.box(400, 266, 170, 56, "memory", "diff, new rev", fs=19)
    s.arrow(275, 360, 275, 324, "red")
    s.arrow(485, 360, 485, 324, "green")
    s.save("deep_shallow")


# --------------------------------------------- 15. stale click target
def click():
    for k in range(1, 4):
        s = Svg(900, 420)
        s.rect(10, 10, 560, 400, "struct", rx=6)
        s.rect(10, 10, 560, 34, "struct", rx=6, fill="#e2e8f0")
        s.text(26, 34, "ops dashboard (schematic)", 18, GRAY, anchor="start")
        if k == 1:
            s.rect(110, 120, 360, 200, "white", rx=8)
            s.text(290, 170, "Restart srv-stage-04?", 26, INK, weight="bold")
        else:
            s.rect(110, 120, 360, 200, "fault", rx=8)
            s.text(290, 170, "Restart prod-04?", 26, ROLE["fault"][2], weight="bold")
            s.text(290, 206, "(popped up just now)", 22, ROLE["fault"][2])
        s.rect(200, 248, 180, 46, "sys", rx=6)
        s.text(262, 278, "Restart", 20, INK, weight="bold")
        # crosshair at the proposed point, on the right of the button
        s.add('<circle cx="345" cy="271" r="17" fill="none" stroke="#0284c7" stroke-width="3"/>')
        s.line(320, 271, 370, 271, "#0284c7", 2)
        s.line(345, 246, 345, 296, "#0284c7", 2)
        s.text(300, 352, "proposed click (640, 420)", 23, ROLE["model"][2])
        s.text(300, 384, "from frame f_12", 22, GRAY)
        if k == 3:
            s.rect(600, 40, 290, 340, "sys", rx=10, fill="#faf5ff")
            s.text(745, 78, "Before dispatch", 24, ROLE["sys"][2], weight="bold")
            checks = ["window and focus?", "dialog, resource ID?", "grant for this restart?"]
            for i, t in enumerate(checks):
                s.text(616, 128 + i * 44, t, 21, INK, anchor="start")
            s.box(612, 262, 268, 100, "fault", "ID changed: abort", "capture a new frame",
                  fs=23, sfs=21)
        s.save(f"click_step{k}")


if __name__ == "__main__":
    loop()
    cas()
    gh_bars()
    incident()
    worlds()
    dup_plot()
    delayed()
    opid()
    headtail()
    obs_hist()
    job_states()
    poll_plot()
    catalog()
    deep_shallow()
    click()
    print("figures written")
