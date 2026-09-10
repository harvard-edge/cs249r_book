#!/usr/bin/env python3
"""README figures for the four-volume MLSysBook series.

R1  hero      - four scopes, the figure that answers "which book do I need"
R2  explainer - the questions every volume answers, below the fold

Both emit a light and a dark variant so the README can pair them in a
<picture> with prefers-color-scheme. R3 was cut in review: it placed vol1's
silicon above vol4's actuation, when vol1 hardware sits inside a vol4 machine,
and dissolved vol4's governance tier into a shared roof, which contradicts the
authority boundary that tier exists to enforce.
"""
from pathlib import Path

VOL = {
    1: ("#A51C30", "#E8798C", "Volume I",   "Machine Learning Systems"),
    2: ("#1F407A", "#7FA3DC", "Volume II",  "Systems at Fleet Scale"),
    3: ("#581C87", "#C08FE0", "Volume III", "Agentic Systems"),
    4: ("#1A4D3E", "#6FB79C", "Volume IV",  "Physical AI"),
}
FONT = ("ui-serif, Georgia, 'Times New Roman', 'Palatino Linotype', Palatino, serif")

class Theme:
    def __init__(self, dark):
        self.dark = dark
        self.bg    = "#0D1117" if dark else "#FFFFFF"
        self.ink   = "#E6EDF3" if dark else "#1A1A1A"
        self.muted = "#8B949E" if dark else "#6B7280"
        self.hair  = "#30363D" if dark else "#D9DCE1"
        self.band  = 0.16 if dark else 0.07
    def accent(self, v):
        return VOL[v][1] if self.dark else VOL[v][0]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def txt(x, y, s, size=13, fill="#000", weight="400", anchor="start", ls="0"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{ls}">{esc(s)}</text>')

def rect(x, y, w, h, fill, rx=3, op=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" fill-opacity="{op}"/>')

def svg(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">\n{rect(0,0,w,h,t.bg,rx=0)}\n'
            f'{body}\n</svg>\n')

# --------------------------------------------------------------------------
# R1 hero. The scope name is the largest type on the page, because "which of
# these four is mine" is the only question a README visitor is asking.
# --------------------------------------------------------------------------
ROWS = [
    (4, "Matter", "your system moves something in the physical world",
     "Body · Nervous System · Brain · Governance", "a mistake damages matter"),
    (3, "Other people's systems", "your model takes actions in systems you do not own",
     "Sandbox · Tools · Memory · Policy · Control", "a mistake drops a production table"),
    (2, "A fleet", "you run many machines as one system",
     "Facility · Fabric · Execution · Serving · Governance", "a mistake stalls a cluster"),
    (1, "One machine", "you are making a single machine train or serve a model well",
     "Hardware · Frameworks · Models · Training · Serving", "a mistake costs a GPU-hour"),
]

def r1(dark=False):
    t = Theme(dark)
    W, H = 1600, 900
    o = []
    o.append(txt(72, 96, "Four books, four scopes", 46, t.ink, "600"))
    o.append(txt(72, 136, "Each volume covers a different thing a system can act on. "
                          "Start where your system lives.", 21, t.muted))
    x0, top, bw, bh, gap = 300, 186, 1180, 152, 20
    for i, (v, scope, whenread, layers, cost) in enumerate(ROWS):
        y = top + i * (bh + gap)
        c = t.accent(v)
        o.append(rect(x0, y, bw, bh, c, op=t.band))
        o.append(rect(x0, y, 7, bh, c, rx=0))
        o.append(txt(x0 + 34, y + 54, scope, 33, c, "600"))
        o.append(txt(x0 + 34, y + 90, "Read this if " + whenread, 19, t.ink))
        o.append(txt(x0 + 34, y + 124, layers, 16, c, "400", ls="0.2"))
        o.append(txt(x0 + bw - 30, y + 124, cost, 15, t.muted, anchor="end"))
        o.append(txt(x0 - 32, y + 50, VOL[v][2], 22, c, "600", anchor="end"))
        o.append(txt(x0 - 32, y + 76, VOL[v][3], 16, t.muted, anchor="end"))
        if v == 1:
            o.append(rect(x0 - 214, y + 92, 182, 26, c, op=0.22))
            o.append(txt(x0 - 123, y + 110, "MIT Press · in print", 14, c,
                         "600", "middle"))
    return svg(W, H, "\n".join(o), t)

# --------------------------------------------------------------------------
# R2 explainer.
# --------------------------------------------------------------------------
CONCERNS = [
    ("Where compute happens", "one accelerator", "a fleet of them",
     "a model plus its tools", "silicon beside a motor"),
    ("What limits it", "memory bandwidth", "network and power",
     "context and tool latency", "physics and the control rate"),
    ("What moves", "tensors", "gradients and checkpoints", "trajectories", "torque"),
    ("How it fails", "the accelerator starves", "a rank stalls the job",
     "a wrong action is taken", "matter is damaged"),
    ("What it must prove", "accuracy", "utilization and cost",
     "the action was permitted", "the machine may operate"),
]

def r2(dark=False):
    t = Theme(dark)
    W, H = 1600, 820
    o = []
    o.append(txt(72, 92, "The questions do not change. The answers do.", 40, t.ink, "600"))
    o.append(txt(72, 130, "Every volume answers the same five questions about the system "
                          "it covers.", 20, t.muted))
    cx, top, cw, rh = 500, 214, 268, 116
    for v in (1, 2, 3, 4):
        x = cx + (v - 1) * cw
        c = t.accent(v)
        o.append(rect(x, top - 58, cw - 18, 42, c, op=0.16))
        o.append(txt(x + (cw - 18) / 2, top - 29, VOL[v][2], 20, c, "600", "middle"))
    for i, row in enumerate(CONCERNS):
        y = top + i * rh
        o.append(f'<line x1="72" y1="{y-14}" x2="{W-72}" y2="{y-14}" '
                 f'stroke="{t.hair}" stroke-width="1"/>')
        o.append(txt(72, y + 34, row[0], 21, t.ink, "600"))
        for v in (1, 2, 3, 4):
            x = cx + (v - 1) * cw
            c = t.accent(v)
            o.append(f'<circle cx="{x+9}" cy="{y+28}" r="4" fill="{c}"/>')
            words, line, lines = row[v].split(), "", []
            for wd in words:
                if len(line + " " + wd) > 21:
                    lines.append(line); line = wd
                else:
                    line = (line + " " + wd).strip()
            lines.append(line)
            for k, ln in enumerate(lines[:2]):
                o.append(txt(x + 26, y + 33 + k * 24, ln, 17, t.ink))
    return svg(W, H, "\n".join(o), t)

for stem, fn in (("series-hero", r1), ("series-questions", r2)):
    Path(f"{stem}.svg").write_text(fn(False))
    Path(f"{stem}-dark.svg").write_text(fn(True))
    print("wrote", stem, "+ dark")
