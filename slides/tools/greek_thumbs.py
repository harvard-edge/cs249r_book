#!/usr/bin/env python3
"""Make text-free thumbnails of figures for goals and takeaways slides.

A goals or takeaways line shows its figure at about 2.5 by 0.95 cm, where no
label can be read. This writes `thumb_<name>.svg` beside each figure, with
every text run replaced by a faint bar of the same length and position, so
the thumbnail keeps the figure's shape without carrying unreadable words.

Usage (from slides/):
    python3 tools/greek_thumbs.py vol3/03_test_time_compute/images/coverage_step3.svg [...]
Then `make deck` converts the new SVGs to PDF.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def style_of(el):
    st = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            st[k.strip()] = v.strip()
    for k in ("font-size", "text-anchor", "fill", "font-family"):
        if el.get(k) is not None:
            st.setdefault(k, el.get(k))
    return st


def num(v, default=0.0):
    if v is None:
        return default
    m = re.match(r"\s*([-\d.eE+]+)", str(v))
    return float(m.group(1)) if m else default


def bars(el, inherited):
    """Yield (x, y, width, height, fill) per line of a text element."""
    st = dict(inherited)
    st.update(style_of(el))
    fs = num(st.get("font-size"), 12.0)
    mono = "mono" in st.get("font-family", "").lower() or "menlo" in st.get("font-family", "").lower()
    x, y = num(el.get("x")), num(el.get("y"))
    lines = []
    own = (el.text or "").strip()
    if own:
        lines.append((x, y, own, st, fs))
    cy = y
    for ts in el.findall(f"{{{SVG}}}tspan"):
        tst = dict(st)
        tst.update(style_of(ts))
        tfs = num(tst.get("font-size"), fs)
        tx = num(ts.get("x"), x)
        cy = num(ts.get("y"), cy) + num(ts.get("dy"), 0.0)
        t = "".join(ts.itertext()).strip()
        if t:
            lines.append((tx, cy, t, tst, tfs))
    for lx, ly, t, s, f in lines:
        w = len(t) * f * (0.6 if mono else 0.47)
        anchor = s.get("text-anchor", "start")
        if anchor == "middle":
            lx -= w / 2
        elif anchor == "end":
            lx -= w
        yield lx, ly - 0.66 * f, w, 0.5 * f, s.get("fill", "#0f172a")


def greek(src):
    tree = ET.parse(src)
    root = tree.getroot()
    parents = {c: p for p in root.iter() for c in p}
    for el in list(root.iter(f"{{{SVG}}}text")):
        p = parents[el]
        idx = list(p).index(el)
        # Inherit font settings from enclosing groups.
        inh, a = {}, p
        chain = []
        while a is not None:
            chain.append(a)
            a = parents.get(a)
        for g in reversed(chain):
            inh.update(style_of(g))
        new = []
        for bx, by, bw, bh, fill in bars(el, inh):
            if fill in ("none", ""):
                fill = "#0f172a"
            r = ET.Element(f"{{{SVG}}}rect", {
                "x": f"{bx:.2f}", "y": f"{by:.2f}", "width": f"{bw:.2f}", "height": f"{bh:.2f}",
                "rx": f"{bh / 2:.2f}", "fill": fill, "fill-opacity": "0.35"})
            if el.get("transform"):
                r.set("transform", el.get("transform"))
            new.append(r)
        p.remove(el)
        for k, r in enumerate(new):
            p.insert(idx + k, r)
    out = src.with_name(f"thumb_{src.stem}.svg")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(greek(Path(a)))
