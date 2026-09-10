#!/usr/bin/env python3
"""Generate candidate README/series figures for the four-volume set.

Three different answers to "how do the four volumes tie together":
  R1  scope ladder      - four volume bands ordered by what a mistake can reach
  R2  shared spine      - the concerns every volume shares, across four columns
  R3  interleaved stack - every layer in the series, coloured by owning volume

Renders plain SVG with no external CSS so it survives a GitHub README.
"""
from pathlib import Path

VOL = {
    1: ("#A51C30", "Volume I",   "Machine Learning Systems"),
    2: ("#1F407A", "Volume II",  "Systems at Fleet Scale"),
    3: ("#581C87", "Volume III", "Agentic Systems"),
    4: ("#1A4D3E", "Volume IV",  "Physical AI"),
}
INK, MUTED, HAIR = "#1A1A1A", "#6B7280", "#D9DCE1"
FONT = ("ui-serif, Georgia, 'Times New Roman', "
        "'Palatino Linotype', Palatino, serif")

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def txt(x, y, s, size=13, fill=INK, weight="400", anchor="start", ls="0"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{ls}">{esc(s)}</text>')

def rect(x, y, w, h, fill, rx=2, op=1.0, stroke="none", sw=0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')

def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">\n'
            f'{rect(0,0,w,h,"#FFFFFF",rx=0)}\n{body}\n</svg>\n')

# --------------------------------------------------------------------------
# R1  scope ladder: four bands, ordered by reach. Bottom = smallest blast
#     radius. Each band lists the layers that volume actually owns.
# --------------------------------------------------------------------------
def r1():
    W, H = 860, 560
    rows = [
        (4, "Matter", "a mistake strips a gearbox",
            "Body · Nervous System · Brain · Governance"),
        (3, "Other people's systems", "a mistake drops a production table",
            "Sandbox · Tools · Serving & Memory · Policy · Control · Fleet"),
        (2, "A fleet", "a mistake stalls a cluster",
            "Facility · Fabric · Execution · Control · Serving · Ops · Assurance · Governance"),
        (1, "One machine", "a mistake wastes an accelerator-hour",
            "Hardware · Frameworks · Models · Training · Serving · Ops · Applications"),
    ]
    o, bh, gap = [], 96, 12
    x0, top, bw = 168, 96, 600
    o.append(txt(40, 46, "One series, four blast radii", 25, INK, "600"))
    o.append(txt(40, 70, "Each volume inherits the constraints of the one below it "
                         "and adds a consequence that cannot be rolled back.", 13, MUTED))
    for i, (v, name, cost, layers) in enumerate(rows):
        y = top + i * (bh + gap)
        c = VOL[v][0]
        o.append(rect(x0, y, bw, bh, c, op=0.07))
        o.append(rect(x0, y, 5, bh, c, rx=0))
        o.append(txt(x0 + 22, y + 33, name, 17, c, "600"))
        o.append(txt(x0 + 22, y + 55, cost, 12.5, MUTED))
        o.append(txt(x0 + 22, y + 79, layers, 11, c, "400", ls="0.15"))
        o.append(txt(x0 - 22, y + 33, VOL[v][1], 13, c, "600", anchor="end"))
        o.append(txt(x0 - 22, y + 51, VOL[v][2], 11, MUTED, anchor="end"))
    ay = top - 8
    ah = 4 * bh + 3 * gap + 16
    o.append(f'<line x1="{x0+bw+34}" y1="{ay+ah}" x2="{x0+bw+34}" y2="{ay}" '
             f'stroke="{MUTED}" stroke-width="1.1" marker-end="url(#a)"/>')
    o.append(f'<text x="{x0+bw+52}" y="{ay+ah/2}" font-family="{FONT}" font-size="11.5" '
             f'fill="{MUTED}" transform="rotate(-90 {x0+bw+52} {ay+ah/2})" '
             f'text-anchor="middle">what a mistake can reach</text>')
    defs = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>')
    return svg(W, H, defs + "\n" + "\n".join(o))

# --------------------------------------------------------------------------
# R2  shared spine: the questions every volume must answer, and how each
#     volume answers them. Shows interaction rather than ordering.
# --------------------------------------------------------------------------
def r2():
    W, H = 980, 560
    concerns = [
        ("Where compute happens", "one accelerator", "a fleet of them",
         "a model plus its tools", "silicon beside a motor"),
        ("What limits it", "memory bandwidth", "network and power",
         "context and tool latency", "physics and the control rate"),
        ("What moves", "tensors", "gradients and checkpoints",
         "trajectories", "torque"),
        ("How it fails", "a slow kernel", "a stalled rank",
         "a wrong action taken", "matter is damaged"),
        ("What it must prove", "accuracy", "utilisation and cost",
         "the action was permitted", "the machine may operate"),
    ]
    o = []
    o.append(txt(40, 46, "Four volumes, one set of questions", 25, INK, "600"))
    o.append(txt(40, 70, "The questions do not change as you move up the series. "
                         "Only the answers do.", 13, MUTED))
    cx, top, cw, rh = 300, 108, 160, 84
    for v in (1, 2, 3, 4):
        x = cx + (v - 1) * cw
        c = VOL[v][0]
        o.append(rect(x, top - 40, cw - 10, 30, c, op=0.10))
        o.append(txt(x + (cw - 10) / 2, top - 20, VOL[v][1], 13, c, "600", "middle"))
    for i, row in enumerate(concerns):
        y = top + i * rh
        o.append(f'<line x1="40" y1="{y-10}" x2="{W-40}" y2="{y-10}" '
                 f'stroke="{HAIR}" stroke-width="1"/>')
        o.append(txt(40, y + 22, row[0], 14, INK, "600"))
        for v in (1, 2, 3, 4):
            x = cx + (v - 1) * cw
            c = VOL[v][0]
            o.append(f'<circle cx="{x+8}" cy="{y+17}" r="3" fill="{c}"/>')
            words, line, lines = row[v].split(), "", []
            for wd in words:
                if len(line + " " + wd) > 20:
                    lines.append(line); line = wd
                else:
                    line = (line + " " + wd).strip()
            lines.append(line)
            for k, ln in enumerate(lines[:2]):
                o.append(txt(x + 18, y + 21 + k * 15, ln, 11.5, INK))
    return svg(W, H, "\n".join(o))

# --------------------------------------------------------------------------
# R3  interleaved stack: one ladder of every layer in the series, each rung
#     coloured by the volume that owns it. The honest structural claim.
# --------------------------------------------------------------------------
def r3():
    W, H = 980, 700
    rungs = [
        (0, "Governance and assurance", "every volume ends here"),
        (3, "Agent policy and control plane", ""),
        (3, "Tools, sandboxes, trajectories", ""),
        (2, "Fleet serving and operations", ""),
        (2, "Distributed execution and fabric", ""),
        (1, "Training, serving, operations", ""),
        (1, "Frameworks and models", ""),
        (1, "Silicon, memory, accelerators", ""),
        (4, "Real-time reflex and actuation", ""),
        (4, "Matter, inertia, thermodynamics", "the floor nothing negotiates"),
    ]
    o = []
    o.append(txt(40, 46, "One ladder, four owners", 25, INK, "600"))
    o.append(txt(40, 70, "Ordered by physical distance from matter. Each volume owns a "
                         "contiguous span; governance is the shared roof.", 13, MUTED))
    x0, top, bw, bh, gap = 210, 104, 400, 50, 6
    for i, (v, name, note) in enumerate(rungs):
        y = top + i * (bh + gap)
        c = VOL[v][0] if v else MUTED
        o.append(rect(x0, y, bw, bh, c, op=0.09 if v else 0.06))
        o.append(rect(x0, y, 5, bh, c, rx=0))
        o.append(txt(x0 + 20, y + 30, name, 14, INK if v else MUTED, "500"))
        lbl = VOL[v][1] if v else "shared"
        o.append(txt(x0 - 18, y + 30, lbl, 12, c, "600", anchor="end"))
        if note:
            o.append(txt(x0 + bw + 16, y + 30, note, 11, MUTED))
    return svg(W, H, "\n".join(o))

for name, fn in (("R1-scope-ladder", r1), ("R2-shared-spine", r2),
                 ("R3-interleaved", r3)):
    Path(f"{name}.svg").write_text(fn())
    print("wrote", name + ".svg")
