#!/usr/bin/env python3
"""Figures for Lecture 9 (Agent Security), Volume III.

Diagrams are SVG in the Volume III agentic-stack palette (runtime violet,
tools amber, model sky blue, memory green, faults red, structure slate).
Plots use slides/assets/kit/slideplot.py. Every number drawn comes from
calcs.py (which names its source). AS = the Agent Security chapter.

  echo_step1-3      EchoLeak chain (AS War Story 9.1; Reddy and Gujral 2025)
  seq_step1-3       AS Sec. 9.2.1 figure 1 (one token sequence), redrawn
  exp_step1-2       NM 9.1, equation 1 (Scenarios.InjectionExposureByReadVolume; calcs)
  checks_step1-2    AS Sec. 9.3.1 and 9.4.1 (four checks on one proposal)
  approval_step1-3  AS Sec. 9.7.3 (approval bound to a digest), schematic
  reader_step1-2    AS Sec. 9.5 figure 3 (quarantined reader), redrawn
  labels_step1-3    AS Sec. 9.6.3 figure 4 (two labels, two checks), redrawn
  launder_step1-3   AS Sec. 9.8.2 (a blocked attack survives in memory)
  ladder            AS Sec. 9.9.1 (outcome ladder of one attacked episode)
  dojo_step1-2      measured: AgentDojo results table (calcs.dojo_row)
  nm92              NM 9.2 table 4 (Scenarios.FixtureReleasePolicyComparison; calcs)
  zero_bound        AS Sec. 9.9.3 footnote 11 (MLSysIM zero-failure bound; calcs)
  thumb_*           text-free pictograms for the goals and takeaways slides

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
from deckvals import val, num  # noqa: E402
from mlsysim.physics import (  # noqa: E402
    calc_clopper_pearson_zero_failure_bound,
    calc_injection_exposure,
    calc_zero_failure_rate_upper_bound,
)

# Inputs of the exposure and zero-success figures, from the scenarios calcs reads
Q = calcs.EXPOSURE.payload_fraction
MISS = calcs.EXPOSURE.miss_rate
READS = calcs.EXPOSURE.reads
CONF = calcs.CLEAN50.confidence


def exposure(q, n):
    return calc_injection_exposure(q, n)


def bound(n):
    """Exact 95 percent zero-success upper bound, for a scalar or an array of n."""
    return 1 - np.vectorize(calc_clopper_pearson_zero_failure_bound)(n, CONF)


def agentdojo():
    """The AgentDojo rows the dojo figures draw, selected by calcs."""
    return dict(plain=calcs.DOJO_PLAIN, early=calcs.dojo_row(calcs.EARLY),
                late=calcs.dojo_row(calcs.LATE), base=calcs.dojo_row(calcs.ABLATED),
                defenses={d: calcs.dojo_row(calcs.ABLATED, d) for d in calcs.DEFENSES})

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
FLOOR = 20   # smallest SVG font, in viewBox units (about 9 pt at slide size)
COLS = {"blue": ROLE["model"][1], "violet": ROLE["sys"][1], "amber": ROLE["tool"][1],
        "green": ROLE["memory"][1], "red": ROLE["fault"][1], "gray": GRAY}


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
            f'fill="{col}"/></marker>' for k, col in COLS.items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" font-family="{FONT}">'
               f'<defs>{markers}</defs><rect x="0" y="0" width="{self.w}" height="{self.h}" '
               f'fill="white"/>' + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)

    def rect(self, x, y, w, h, role="struct", dash=False, rx=8, sw=2, fill=None):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill or f}" stroke="{s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, fs=22, color=INK, anchor="middle", weight="normal", mono=False,
             italic=False):
        fs = max(fs, FLOOR)
        fam = f' font-family="{MONO}"' if mono else ""
        it = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{fam}{it}>{s}</text>')

    def box(self, x, y, w, h, role, title, sub=None, fs=22, sfs=20, dash=False, mono_sub=False,
            mono_title=False, sub2=None):
        self.rect(x, y, w, h, role, dash=dash)
        _, _, tc = ROLE[role]
        if sub2:
            self.text(x + w / 2, y + h / 2 - 14, title, fs, INK, weight="bold", mono=mono_title)
            self.text(x + w / 2, y + h / 2 + 12, sub, sfs, tc, mono=mono_sub)
            self.text(x + w / 2, y + h / 2 + 36, sub2, sfs, tc, mono=mono_sub)
        elif sub:
            self.text(x + w / 2, y + h / 2 - 4, title, fs, INK, weight="bold", mono=mono_title)
            self.text(x + w / 2, y + h / 2 + sfs + 2, sub, sfs, tc, mono=mono_sub)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, title, fs, INK, weight="bold",
                      mono=mono_title)

    def arrow(self, x1, y1, x2, y2, color="gray", dash=False, sw=2.5, path=None):
        d = ' stroke-dasharray="8 6"' if dash else ""
        geom = path or f"M{x1},{y1} L{x2},{y2}"
        self.add(f'<path d="{geom}" fill="none" stroke="{COLS[color]}" stroke-width="{sw}"{d} '
                 f'marker-end="url(#a{color})"/>')

    def line(self, x1, y1, x2, y2, color=GRAY, sw=2, dash=False):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}"{d}/>')

    def cross(self, x, y, r=11, color=None):
        col = color or ROLE["fault"][1]
        self.add(f'<path d="M{x-r},{y-r} L{x+r},{y+r} M{x-r},{y+r} L{x+r},{y-r}" '
                 f'stroke="{col}" stroke-width="5"/>')

    def tick(self, x, y, r=12, color=None):
        col = color or ROLE["memory"][1]
        self.add(f'<path d="M{x-r},{y} L{x-r/3},{y+r*0.7} L{x+r},{y-r*0.8}" fill="none" '
                 f'stroke="{col}" stroke-width="5"/>')


# --------------------------------------------------- 1. EchoLeak chain
def _echo_base(s, fix=False):
    s.box(15, 20, 250, 90, "fault", "Attacker email",
          "label: untrusted" if fix else "phrased for a human", fs=23)
    s.box(325, 20, 250, 90, "memory", "Retrieval",
          "label: mailbox only" if fix else "mail, files, chats", fs=23)
    s.box(635, 20, 250, 90, "model", "Assistant model",
          "output joins both" if fix else "drafts the answer", fs=23)
    s.arrow(265, 65, 322, 65, "red")
    s.arrow(575, 65, 632, 65, "blue")
    s.arrow(760, 110, 140, 197, "blue", path="M760,110 L760,160 L140,160 L140,197")
    s.box(15, 200, 250, 90, "tool", "Reply", "![image][ref]", fs=23, mono_sub=True)
    s.arrow(265, 245, 322, 245, "amber")


def echo():
    """War Story 9.1: one email to a fetched image URL; three defenses passed."""
    for k in (1, 2, 3):
        s = Svg(900, 385)
        _echo_base(s)
        s.box(325, 200, 250, 90, "tool", "Client fetches", "the image URL", fs=23)
        s.box(635, 200, 250, 90, "fault", "Attacker server", "logs ?secret", fs=23)
        s.arrow(575, 245, 632, 245, "red")
        s.text(604, 190, "via teams.microsoft.com", 20, ROLE["fault"][2])
        if k >= 2:
            s.text(295, 140, "classifier: passed", 20, ROLE["fault"][2], weight="bold")
            s.text(140, 320, "link redaction: missed [ref]", 20, ROLE["fault"][2], weight="bold")
            s.text(604, 320, "domain allowlist: passed", 20, ROLE["fault"][2], weight="bold")
        if k == 3:
            s.rect(15, 340, 870, 38, "sys", rx=8)
            s.text(450, 366, "No check asked where mailbox data may go", 21, ROLE["sys"][2],
                   weight="bold")
        s.save(f"echo_step{k}")
    s = Svg(900, 385)
    _echo_base(s, fix=True)
    s.rect(325, 190, 270, 110, "sys", rx=10)
    s.text(460, 224, "Release check", 23, ROLE["sys"][2], weight="bold")
    s.text(460, 254, "attacker URL in", 20, INK)
    s.text(460, 280, "mailbox readers? no", 20, INK)
    s.box(655, 200, 230, 90, "fault", "Not fetched", "rejected first", fs=23)
    s.arrow(595, 245, 652, 245, "red", dash=True)
    s.text(450, 350, "The URL is a recipient; the image tag is an outbound call", 21,
           ROLE["sys"][2], weight="bold")
    s.save("echo_fix")


# -------------------------------------------- 2. one token sequence
def seq():
    groups = [("App instructions", "developer", 4, "struct"),
              ("Tool definitions", "runtime", 4, "struct"),
              ("User request", "task owner", 3, "struct"),
              ("README", "attacker", 4, "fault")]
    for k in (1, 2, 3):
        s = Svg(900, 400)
        x = 20
        cells = []
        for t, sub, n, role in groups:
            w = n * 52 - 6
            col = ROLE["fault"][2] if role == "fault" else INK
            s.text(x + w / 2, 26, t, 21, col, weight="bold")
            s.text(x + w / 2, 52, sub, 20, ROLE["fault"][2] if role == "fault" else GRAY)
            for i in range(n):
                cx = x + i * 52
                fill = "#fecaca" if role == "fault" else "#e2e8f0"
                s.add(f'<rect x="{cx}" y="66" width="46" height="46" rx="4" fill="{fill}" '
                      f'stroke="{ROLE[role][1]}" stroke-width="1.5"/>')
                cells.append((cx + 23, role))
            x += w + 26
        s.text(20, 392, "role markers are ordinary tokens", 20, GRAY, anchor="start", italic=True)
        if k >= 2:
            for cx, role in cells:
                col = "#f87171" if role == "fault" else "#cbd5e1"
                s.line(cx, 112, 300, 195, col, sw=1.6)
            s.box(150, 195, 300, 64, "model", "Foundation model", fs=24)
            s.arrow(300, 259, 300, 287, "amber")
            s.box(80, 290, 440, 76, "tool", "Proposed call", "package_search(fixture...)",
                  fs=23, mono_sub=True)
        if k >= 3:
            s.line(575, 185, 575, 380, GRAY, sw=2, dash=True)
            s.text(560, 396, "model", 20, GRAY, anchor="end")
            s.text(590, 396, "runtime", 20, GRAY, anchor="start")
            s.arrow(520, 328, 615, 328, "amber")
            s.rect(618, 240, 270, 136, "sys", rx=10)
            s.text(753, 274, "Runtime check", 23, ROLE["sys"][2], weight="bold")
            s.text(753, 302, "on the proposal,", 20, INK)
            s.text(753, 326, "not on the text", 20, INK)
            s.rect(640, 338, 226, 30, "fault", rx=6)
            s.text(753, 360, "rejected", 20, ROLE["fault"][2], weight="bold")
        s.save(f"seq_step{k}")

# ------------------------------------------------ 3. exposure plot
def exposure_plot():
    n = np.arange(0, 121)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.1, 2.5))
        ax.plot(n, 100 * exposure(Q, n), color=C["danger"], lw=2.4)
        ax.axhline(100, color=C["neutral"], lw=1, ls="--")
        pos = {"repair": (2, 72, "left"), "research": (118, 92, "right")}
        for key in ("repair", "research"):
            r = READS[key]
            v = num(f"exp_{key}")
            ax.plot([r], [v], "o", color=C["text"], ms=7)
            x, y, ha = pos[key]
            ax.text(x, y, f"{key}, {val('exp_n_' + key)} reads: {val('exp_' + key)}%", fontsize=12,
                    ha=ha, va="center",
                    color=C["text"])
            if key == "repair":
                ax.annotate("", (r, v + 2), (10, y - 5),
                            arrowprops=dict(arrowstyle="-", color=C["text"], lw=1))
        if k == 2:
            ax.plot(n, 100 * exposure(Q * MISS, n),
                    color=C["memory"], lw=2.4)
            v1, v2 = num("exp_repair_scr"), num("exp_research_scr")
            ax.plot([READS["repair"], READS["research"]], [v1, v2], "o", color=C["memory"], ms=7)
            ax.text(118, 12, f"screen missing {val('exp_miss')}%: {val('exp_repair_scr')}%, "
                    f"{val('exp_research_scr')}%",
                    fontsize=12, ha="right", color=C["memory"])
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 105)
        ax.set_xlabel("untrusted documents read per task")
        ax.set_ylabel("tasks meeting a payload (%)")
        sp.save(fig, OUT / f"exp_step{k}.svg")


# ----------------------------------------- 4. four checks on one call
def checks():
    for k in (1, 2):
        s = Svg(900, 400)
        s.rect(20, 20, 860, 70, "tool", rx=10)
        s.text(450, 64, 'package_search(query="dependency diagnosis PRIVATE_CASE_EXAMPLE")',
               20, INK, mono=True)
        rows = [("A", "Tool name registered and granted", True),
                ("B", "Arguments match the schema", True),
                ("C", "Endpoint on the egress allowlist", True),
                ("D", "Query data may reach this endpoint", False)]
        for i, (lab, t, ok) in enumerate(rows):
            y = 116 + i * 68
            s.rect(20, y, 860, 56, "white", rx=8)
            s.text(52, y + 37, lab, 24, ROLE["sys"][2], weight="bold")
            s.text(90, y + 37, t, 23, INK, anchor="start")
            if k == 2:
                if ok:
                    s.tick(840, y + 28)
                    s.text(810, y + 37, "passes", 21, ROLE["memory"][2], anchor="end")
                else:
                    s.rect(20, y, 860, 56, "fault", rx=8, fill="none", sw=3)
                    s.cross(840, y + 28)
                    s.text(810, y + 37, "fails", 21, ROLE["fault"][2], anchor="end",
                           weight="bold")
            else:
                s.text(840, y + 37, "?", 26, GRAY, weight="bold")
        s.save(f"checks_step{k}")


# ---------------------------------------- 5. action-bound approval
def approval():
    fields = [("task", "repair_42"), ("repository", "connection-pool"),
              ("target branch", "main"), ("candidate", "r18"),
              ("artifact digest", "sha256:9f2c..."), ("base commit", "a41e...")]
    for k in (1, 2, 3):
        s = Svg(900, 400)
        s.rect(15, 10, 360, 285, "struct", rx=10)
        s.text(195, 42, "Canonical action", 23, INK, weight="bold")
        for i, (f, v) in enumerate(fields):
            y = 80 + i * 37
            s.text(35, y, f, 20, GRAY, anchor="start")
            s.text(355, y, v, 20, INK, anchor="end", mono=True)
        s.text(195, 325, 'model\'s note: "Updating dependencies"', 20, GRAY, italic=True)
        s.text(195, 351, "shown, labeled untrusted", 20, GRAY)
        s.arrow(375, 140, 455, 140, "violet")
        s.rect(458, 80, 200, 120, "sys", rx=10)
        s.text(558, 118, "Human approves", 21, ROLE["sys"][2], weight="bold")
        s.text(558, 146, "out of band", 20, INK)
        s.text(558, 174, "signs digest d", 20, INK)
        if k >= 2:
            s.arrow(658, 140, 690, 140, "violet")
            s.rect(693, 80, 195, 120, "sys", rx=10)
            s.text(790, 118, "At dispatch", 21, ROLE["sys"][2], weight="bold")
            s.text(790, 146, "recompute d,", 20, INK)
            s.text(790, 174, "recheck base", 20, INK)
            s.box(693, 235, 195, 80, "memory", "match", "dispatch once", fs=22)
            s.arrow(790, 200, 790, 232, "green")
        if k == 3:
            s.rect(430, 235, 240, 150, "fault", rx=10)
            s.text(550, 268, "Patch moved to r19", 21, ROLE["fault"][2], weight="bold")
            s.text(550, 298, "after sign-off:", 20, INK)
            s.text(550, 326, "digest differs,", 20, INK)
            s.text(550, 354, "approval void", 20, ROLE["fault"][2], weight="bold")
        s.save(f"approval_step{k}")


# ------------------------------------------- 6. quarantined reader
def reader():
    for k in (1, 2):
        s = Svg(900, 400)
        s.rect(10, 10, 345, 175, "struct", dash=True, rx=12)
        s.text(24, 38, "quarantine: no tools, no credentials", 20, GRAY, anchor="start")
        s.box(25, 60, 140, 96, "fault", "README", "readme_7", fs=23, mono_sub=True)
        s.box(200, 60, 140, 96, "model", "Reader", "extracts", fs=23)
        s.arrow(165, 108, 197, 108, "red")
        s.arrow(340, 108, 387, 108, "blue")
        s.box(390, 60, 150, 96, "sys", "Adapter", "schema check", fs=23)
        s.arrow(540, 108, 587, 108, "violet")
        s.rect(590, 10, 300, 200, "white", rx=10)
        s.text(605, 40, "admitted record", 21, INK, anchor="start", weight="bold")
        s.text(605, 72, "claimed_tests_passed:", 20, ROLE["fault"][2], anchor="start", mono=True)
        s.text(625, 96, "true", 20, ROLE["fault"][2], anchor="start", mono=True)
        s.text(605, 134, "host adds:", 20, ROLE["sys"][2], anchor="start")
        s.text(605, 160, "readme_7 @ doc_v3", 20, ROLE["sys"][2], anchor="start")
        s.text(605, 186, "untrusted_external", 20, ROLE["sys"][2], anchor="start")
        s.arrow(760, 210, 760, 242, "gray")
        s.box(640, 245, 240, 76, "model", "Planner", "sees the claim only", fs=23)
        if k == 2:
            s.arrow(640, 283, 455, 283, "amber")
            s.rect(30, 228, 420, 162, "sys", rx=10)
            s.text(240, 260, "Evidence check: record a pass?", 22, ROLE["sys"][2],
                   weight="bold")
            s.text(240, 292, "accepted source: protected runner", 20, INK)
            s.text(240, 318, "claim source: readme_7", 20, INK)
            s.rect(110, 336, 260, 38, "fault", rx=6)
            s.text(240, 362, "not recorded as a pass", 20, ROLE["fault"][2], weight="bold")
        s.save(f"reader_step{k}")


# ------------------------------------------ 7. two labels, two checks
def labels():
    for k in (1, 2, 3):
        s = Svg(900, 395)
        s.rect(15, 10, 250, 100, "fault", rx=10)
        s.text(140, 40, "README", 22, ROLE["fault"][2], weight="bold")
        s.text(140, 68, "ι = {readme_7}", 21, INK)
        s.text(140, 94, "κ = every recipient", 21, INK)
        s.rect(15, 125, 250, 100, "memory", rx=10)
        s.text(140, 155, "Private fixture", 22, ROLE["memory"][2], weight="bold")
        s.text(140, 183, "ι = {fixture}", 21, INK)
        s.text(140, 209, "κ = {test runner}", 21, INK)
        s.text(15, 262, "ι: sources that may have influenced it", 20, GRAY, anchor="start")
        s.text(15, 288, "κ: recipients allowed to receive it", 20, GRAY, anchor="start")
        if k >= 2:
            s.arrow(265, 60, 312, 105, "gray")
            s.arrow(265, 175, 312, 135, "gray")
            s.box(315, 80, 150, 80, "model", "Planner", "saw both", fs=23)
            s.arrow(465, 120, 497, 120, "gray")
            s.rect(500, 30, 175, 180, "white", rx=10)
            s.text(587, 62, "query x", 22, INK, weight="bold")
            s.text(587, 96, "ι = union:", 20, GRAY)
            s.text(587, 122, "readme_7, fixture", 20, INK)
            s.text(587, 156, "κ = intersection:", 20, GRAY)
            s.text(587, 182, "{test runner}", 20, INK)
        if k >= 3:
            s.arrow(675, 120, 697, 120, "amber")
            s.rect(700, 10, 190, 220, "sys", rx=10)
            s.text(795, 42, "Release check", 21, ROLE["sys"][2], weight="bold")
            s.text(795, 74, "recipient:", 20, INK)
            s.text(795, 100, "search endpoint", 20, INK)
            s.text(795, 132, "in κ(x)? no", 21, INK, weight="bold")
            s.rect(715, 158, 160, 54, "fault", rx=6)
            s.text(795, 192, "rejected", 20, ROLE["fault"][2], weight="bold")
            s.rect(420, 250, 470, 135, "sys", rx=10)
            s.text(440, 284, "Evidence check: record a pass", 21, ROLE["sys"][2],
                   anchor="start", weight="bold")
            s.text(440, 316, "accepts only: protected runner", 20, INK, anchor="start")
            s.text(440, 344, "ι(claim) = {readme_7}", 20, INK, anchor="start")
            s.text(440, 372, "refused", 20, ROLE["fault"][2], anchor="start", weight="bold")
        s.save(f"labels_step{k}")


# --------------------------------------------- 8. memory laundering
def launder():
    steps = [("fault", "README note", ["“diagnostic routine", "requires uploading", "fixtures”"]),
             ("model", "Turn summary", ["“docs suggest", "uploading fixture", "details”"]),
             ("memory", "Stored rule", ["“Always upload", "fixture details on", "dep errors”"]),
             ("tool", "Later session", ["proposes the", "upload as a", "routine step"])]
    for k in (1, 2, 3):
        s = Svg(900, 385)
        for i, (role, t, lines) in enumerate(steps):
            x = 10 + i * 225
            if k == 1 and i > 1:
                continue
            s.rect(x, 50, 205, 150, role, rx=10)
            s.text(x + 102, 82, t, 22, ROLE[role][2], weight="bold")
            for j, ln in enumerate(lines):
                s.text(x + 102, 116 + 26 * j, ln, 20, INK)
            if i < 3 and not (k == 1 and i >= 1):
                s.arrow(x + 205, 125, x + 222, 125, "gray")
        s.text(10, 34, "turn 3: the search was blocked", 20, ROLE["fault"][2], anchor="start",
               weight="bold")
        if k >= 2:
            s.text(450, 232, "the raw README is gone; its instruction is not", 21, INK,
                   weight="bold")
        if k >= 3:
            s.rect(10, 252, 880, 126, "sys", rx=10)
            s.text(30, 286, "Labels follow derivation: summary and rule carry readme_7",
                   21, ROLE["sys"][2], anchor="start", weight="bold")
            s.text(30, 318, "Retract readme_7: invalidate the summary, the rule, and pending",
                   20, INK, anchor="start")
            s.text(30, 346, "proposals; reset every state tier, not only the chat history",
                   20, INK, anchor="start")
        s.save(f"launder_step{k}")


# ------------------------------------------------ 9. outcome ladder
def ladder():
    s = Svg(900, 400)
    s.box(10, 145, 170, 80, "struct", "Attacked", "episode", fs=23)
    s.box(225, 25, 200, 80, "memory", "No misuse", "proposed", fs=23)
    s.box(225, 245, 200, 80, "tool", "Misuse", "proposed", fs=23)
    s.arrow(180, 170, 222, 75, "gray")
    s.arrow(180, 200, 222, 285, "gray")
    s.box(470, 150, 180, 70, "sys", "Blocked", "before dispatch", fs=22)
    s.box(470, 280, 180, 70, "tool", "Dispatched", fs=22)
    s.arrow(425, 275, 467, 195, "violet")
    s.arrow(425, 295, 467, 315, "amber")
    for i, (role, t) in enumerate((("fault", "Confirmed harm"), ("memory", "No observed harm"),
                                   ("struct", "Unresolved"))):
        y = 210 + i * 54
        s.rect(690, y, 200, 46, role, rx=6)
        s.text(790, y + 30, t, 20, INK, weight="bold")
        s.arrow(650, 315, 687, y + 23, "gray", sw=2)
    s.text(450, 392, "Unresolved is reported, never counted as a defense", 21,
           ROLE["sys"][2], weight="bold")
    s.save("ladder")

# ------------------------------------------------ 10. AgentDojo plot
SHORT = {"spotlighting_with_delimiting": "delimiters", "transformers_pi_detector": "detector",
         "tool_filter": "tool filter", "repeat_user_prompt": "repeat prompt"}


def dojo():
    a = agentdojo()
    base = a["base"]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        xs = [r["utility_pct"] for r in a["plain"]]
        ys = [r["targeted_asr_pct"] for r in a["plain"]]
        ax.scatter(xs, ys, s=38, color=C["neutral"], alpha=0.75, zorder=2)
        e, l = a["early"], a["late"]
        if k == 1:
            ax.annotate(f"same model family, same utility:\n{val('dj_early')}% vs {val('dj_late')}% attack success",
                        (e["utility_pct"], e["targeted_asr_pct"]), xytext=(40, 52),
                        fontsize=11, ha="center", color=C["text"],
                        arrowprops=dict(arrowstyle="-", color=C["text"], lw=1))
            ax.plot([e["utility_pct"], l["utility_pct"]],
                    [e["targeted_asr_pct"], l["targeted_asr_pct"]], color=C["text"], lw=1,
                    ls=":")
        if k == 2:
            ax.scatter([base["utility_pct"]], [base["targeted_asr_pct"]], s=60,
                       color=C["text"], zorder=4)
            ax.text(base["utility_pct"] - 1.5, base["targeted_asr_pct"] + 2.5, "GPT-4o",
                    fontsize=11, ha="right", color=C["text"])
            place = {"spotlighting_with_delimiting": (76, 37, "left"),
                     "transformers_pi_detector": (53, 1.5, "center"),
                     "tool_filter": (79, 13, "left"), "repeat_user_prompt": (87, 22, "center")}
            cols = {"spotlighting_with_delimiting": C["danger"],
                    "transformers_pi_detector": C["compute"], "tool_filter": C["data"],
                    "repeat_user_prompt": C["memory"]}
            for d, r in a["defenses"].items():
                ax.annotate("", (r["utility_pct"], r["targeted_asr_pct"]),
                            (base["utility_pct"], base["targeted_asr_pct"]),
                            arrowprops=dict(arrowstyle="-|>", color=cols[d], lw=2))
                x, y, ha = place[d]
                ax.text(x, y, f"{SHORT[d]}: {r['utility_pct']:.0f}%, {r['targeted_asr_pct']:.0f}%",
                        fontsize=11, color=cols[d], ha=ha, fontweight="bold")
        ax.set_xlim(20, 95)
        ax.set_ylim(-2, 62)
        ax.set_xlabel("benign utility (%)")
        ax.set_ylabel("attack success (%)")
        ax.text(94, 0, "safe corner", fontsize=11, ha="right", va="bottom", color=C["data"])
        sp.save(fig, OUT / f"dojo_step{k}.svg")


# ------------------------------------------------ 11. NM 9.2 bars
def nm92():
    n = calcs.POLICY.result
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 1.45))
    pols = ["A", "B"]
    conf = [100 * n.baseline_lower, 100 * n.candidate_lower]
    unres = [100 * (n.baseline_upper - n.baseline_lower), 100 * (n.candidate_upper - n.candidate_lower)]
    a1.barh(pols, conf, color=C["danger"], height=0.55)
    a1.barh(pols, unres, left=conf, color=C["neutral_l"], height=0.55, edgecolor=C["neutral"],
            hatch="//")
    for i in range(2):
        a1.text(conf[i] + unres[i] + 0.8, i, f"{conf[i]:.0f}% + {unres[i]:.0f}% unresolved",
                va="center", fontsize=11)
    a1.set_xlim(0, 45)
    a1.invert_yaxis()
    a1.set_title("attacked: disclosures", fontsize=12)
    acc = [100 * n.baseline_acceptance, 100 * n.candidate_acceptance]
    a2.barh(pols, acc, color=C["data"], height=0.55)
    for i in range(2):
        a2.text(acc[i] + 1, i, f"{acc[i]:.0f} of {val('pc_benign')}", va="center", fontsize=11)
    a2.set_xlim(0, 125)
    a2.invert_yaxis()
    a2.set_title("benign: accepted repairs", fontsize=12)
    fig.tight_layout()
    sp.save(fig, OUT / "nm92.svg")


# -------------------------------------------- 12. zero-success bound
def zero_bound():
    n = np.logspace(1, 3, 100)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.loglog(n, 100 * bound(n), color=C["network"], lw=2.4)
    ax.loglog(n, 100 * np.vectorize(calc_zero_failure_rate_upper_bound)(n, CONF), color=C["neutral"],
              lw=1.2, ls="--")
    for m in (10, 50, 100, 1000):
        v = 100 * float(bound(m))
        ax.plot([m], [v], "o", color=C["network"], ms=6)
        ax.text(m * 1.12, v * 1.12, f"{v:.1f}%" if v < 10 else f"{v:.0f}%", fontsize=11,
                color=C["network"])
    ax.text(400, 1.4, "3/n", fontsize=11, color=C["neutral"])
    ax.set_xlabel("attacked episodes, zero successes")
    ax.set_ylabel(f"{val('zb_conf')}% bound (%)")
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax.set_xticks([10, 50, 100, 1000])
    ax.set_xticklabels(["10", "50", "100", "1000"])
    ax.minorticks_off()
    sp.save(fig, OUT / "zero_bound.svg")


# ------------------------------------- 13. text-free thumbnails
# Goals and takeaways show figures at 2.5 by 0.95 cm, too small for labels,
# so each thumbnail is a text-free pictogram of the figure that teaches it.
def thumbs():
    W, H = 260, 100
    # exposure curve
    s = Svg(W, H)
    s.line(20, 90, 250, 90, GRAY, sw=3)
    s.line(20, 90, 20, 8, GRAY, sw=3)
    pts = " ".join(f"{20 + 2 * n:.0f},{90 - 80 * exposure(Q, n):.1f}"
                   for n in range(0, 116, 5))
    s.add(f'<polyline points="{pts}" fill="none" stroke="{ROLE["fault"][1]}" stroke-width="6"/>')
    pts = " ".join(f"{20 + 2 * n:.0f},{90 - 80 * exposure(Q * MISS, n):.1f}"
                   for n in range(0, 116, 5))
    s.add(f'<polyline points="{pts}" fill="none" stroke="{ROLE["model"][1]}" stroke-width="6"/>')
    s.save("thumb_exp")
    # four checks, last fails
    s = Svg(W, H)
    for i in range(4):
        y = 6 + i * 23
        role = "fault" if i == 3 else "white"
        s.rect(10, y, 240, 19, role, rx=4, sw=2)
        if i < 3:
            s.tick(228, y + 9, r=7)
        else:
            s.cross(228, y + 9, r=7)
    s.save("thumb_checks")
    # one sequence converging on the model
    s = Svg(W, H)
    for i in range(12):
        role = "fault" if i >= 9 else "struct"
        s.rect(10 + i * 20, 6, 16, 16, role, rx=2, sw=1.5)
        s.line(18 + i * 20, 22, 95, 58, "#f87171" if i >= 9 else "#94a3b8", sw=1.5)
    s.rect(55, 58, 80, 34, "model", rx=6)
    s.arrow(135, 75, 170, 75, "amber", sw=3)
    s.rect(172, 52, 80, 44, "sys", rx=6)
    s.cross(212, 74, r=10)
    s.save("thumb_seq")
    # two labels, two checks
    s = Svg(W, H)
    s.rect(8, 8, 60, 36, "fault", rx=5)
    s.rect(8, 56, 60, 36, "memory", rx=5)
    s.arrow(68, 26, 98, 46, "gray", sw=3)
    s.arrow(68, 74, 98, 56, "gray", sw=3)
    s.rect(100, 32, 60, 36, "model", rx=5)
    s.arrow(160, 50, 185, 50, "amber", sw=3)
    s.rect(188, 6, 66, 88, "sys", rx=6)
    s.cross(221, 30, r=10)
    s.cross(221, 70, r=10)
    s.save("thumb_labels")
    # laundering chain
    s = Svg(W, H)
    for i, role in enumerate(("fault", "model", "memory", "tool")):
        s.rect(6 + i * 64, 20, 52, 50, role, rx=6)
        if i < 3:
            s.arrow(58 + i * 64, 45, 68 + i * 64, 45, "gray", sw=3)
    s.rect(6, 78, 244, 16, "sys", rx=4)
    s.save("thumb_launder")
    # AgentDojo: dots and four arrows from one point
    a = agentdojo()
    s = Svg(W, H)

    def xy(u, v):
        return 10 + (u - 20) / 75 * 240, 92 - v / 60 * 84
    for r in a["plain"]:
        x, y = xy(r["utility_pct"], r["targeted_asr_pct"])
        s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#94a3b8"/>')
    bx, by = xy(a["base"]["utility_pct"], a["base"]["targeted_asr_pct"])
    for d, col in (("spotlighting_with_delimiting", "red"), ("transformers_pi_detector", "amber"),
                   ("tool_filter", "green"), ("repeat_user_prompt", "blue")):
        r = a["defenses"][d]
        x, y = xy(r["utility_pct"], r["targeted_asr_pct"])
        s.arrow(bx, by, x, y, col, sw=3)
    s.add(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="{INK}"/>')
    s.save("thumb_dojo")


if __name__ == "__main__":
    echo()
    seq()
    exposure_plot()
    checks()
    approval()
    reader()
    labels()
    launder()
    ladder()
    dojo()
    nm92()
    zero_bound()
    thumbs()
    print("figures written")
