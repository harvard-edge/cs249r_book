#!/usr/bin/env python3
"""
Applicability matrix: YAML source of truth → app_matrix.tex

  python3 generate_app_matrix_from_yaml.py          # write app_matrix.tex
  python3 generate_app_matrix_from_yaml.py --export # write applicability_matrix.yaml from app_matrix.tex

Edit interviews/paper/applicability_matrix.yaml (categories, subtopics, booleans per track).
Tracks order: [cloud, edge, mobile, tinyml].
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent
YAML_PATH = DIR / "applicability_matrix.yaml"
TEX_PATH = DIR / "app_matrix.tex"

MASK = "green!60!black"
MASK_TOKEN = "greenBLACKNOSPLIT"

CHECK = r"\textcolor{green!60!black}{\checkmark}"
CROSS = r"\textcolor{red}{\ding{55}}"


def split_row(line: str) -> list[str]:
    line2 = line.replace(r"\&", "__AMP__")
    line2 = line2.replace(MASK, MASK_TOKEN)
    parts = re.split(r"\s*&\s*", line2)
    if len(parts) != 15:
        raise ValueError(f"Expected 15 cells, got {len(parts)}")
    return [p.replace(MASK_TOKEN, MASK).replace("__AMP__", r"\&") for p in parts]


def marks_to_bool(marks: list[str]) -> list[bool]:
    out: list[bool] = []
    for m in marks:
        m = m.strip()
        if CHECK in m or r"\checkmark" in m:
            out.append(True)
        elif CROSS in m or r"\ding{55}" in m:
            out.append(False)
        else:
            raise ValueError(f"Unknown mark cell: {m[:80]!r}")
    return out


def bool_to_marks(applicable: list[bool]) -> list[str]:
    return [CHECK if x else CROSS for x in applicable]


def escape_topic(s: str) -> str:
    """Topic name for LaTeX body (ampersands already \\& in YAML as needed)."""
    return s.replace("&", r"\&")


def parse_tex_to_panels(tex: str) -> list[list[dict]]:
    """Return 3 panels: each category is {name, count|None, implicit, topics: [{name, tracks}]}"""
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip() and ln.endswith(r"\\")]

    panels: list[list[dict]] = [[], [], []]
    cur: list[int | None] = [None, None, None]
    IMPLICIT = "__implicit__"

    def ensure_cat(col: int, header_tex: str | None, implicit: bool) -> None:
        if header_tex is None:
            name = None
            count = None
        else:
            m = re.match(r"\\textbf\{(.+?)\s*\((\d+)\)\}", header_tex)
            if not m:
                raise ValueError(f"Bad header: {header_tex!r}")
            name = m.group(1).strip()
            count = int(m.group(2))
        panels[col].append(
            {"name": name, "count": count, "implicit": implicit, "topics": []}
        )
        cur[col] = len(panels[col]) - 1

    for line in lines:
        line_stripped = line[:-2].strip()
        parts = split_row(line_stripped)
        for col in range(3):
            cell = parts[col * 5]
            marks = parts[col * 5 + 1 : col * 5 + 5]
            if cell.startswith("\\textbf{"):
                ensure_cat(col, cell, implicit=False)
            elif cell.strip() == "" and all(m.strip() == "" for m in marks):
                continue
            else:
                if cur[col] is None:
                    ensure_cat(col, None, implicit=True)
                panels[col][cur[col]]["topics"].append(
                    {"name": cell.replace(r"\&", "&"), "tracks": marks_to_bool(marks)}
                )
    return panels


def export_yaml_from_tex() -> None:
    text = TEX_PATH.read_text()
    panels = parse_tex_to_panels(text)
    doc = {
        "tracks": ["cloud", "edge", "mobile", "tinyml"],
        "sort_panels": True,
        "panels": [],
    }
    for pi, panel in enumerate(panels):
        pdoc: dict = {"categories": []}
        for cat in panel:
            cdoc: dict = {}
            if cat["implicit"]:
                cdoc["implicit"] = True
            else:
                cdoc["name"] = cat["name"]
                cdoc["count"] = cat["count"]
            cdoc["topics"] = []
            for t in cat["topics"]:
                cdoc["topics"].append(
                    {"name": t["name"], "tracks": t["tracks"]}
                )
            pdoc["categories"].append(cdoc)
        doc["panels"].append(pdoc)

    yaml_body = yaml.safe_dump(
        doc,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    header = (
        "# Applicability matrix — edit categories, subtopics, and track booleans.\n"
        "# Regenerate LaTeX: python3 generate_app_matrix_from_yaml.py\n"
        "# Category \"count\" is the (n) in the PDF bold header; it may differ from\n"
        "# the number of topic rows when the label is editorial (e.g. Cross-Cutting).\n\n"
    )
    YAML_PATH.write_text(header + yaml_body)
    print(f"Wrote {YAML_PATH}")


def flatten_panels(panels: list[list[dict]]) -> list[list[tuple]]:
    """Each panel → list of ('header', str) | ('topic', name, marks_tex x4)."""
    flat: list[list[tuple]] = [[], [], []]
    for col in range(3):
        for cat in panels[col]:
            if not cat.get("implicit"):
                title = cat["name"]
                n = cat["count"]
                flat[col].append(("header", f"\\textbf{{{title} ({n})}}"))
            for t in cat["topics"]:
                name = escape_topic(t["name"])
                marks = bool_to_marks(t["tracks"])
                flat[col].append(("topic", name, marks))
    return flat


def emit_table_body(flat: list[list[tuple]]) -> str:
    max_len = max(len(x) for x in flat)
    for c in range(3):
        while len(flat[c]) < max_len:
            flat[c].append(("empty",))

    lines_out: list[str] = []
    for i in range(max_len):
        parts: list[str] = []
        for col in range(3):
            item = flat[col][i]
            if item[0] == "header":
                parts.extend([item[1], "", "", "", ""])
            elif item[0] == "topic":
                parts.extend([item[1]] + item[2])
            else:
                parts.extend(["", "", "", "", ""])
        lines_out.append(" & ".join(parts) + r" \\")
    return "\n".join(lines_out)


TEX_PREFIX = r"""% Applicability matrix — generated by generate_app_matrix_from_yaml.py from applicability_matrix.yaml; do not edit by hand.
\begin{table*}[p]
\centering
\begin{adjustbox}{max width=\textwidth,max height=\textheight,keepaspectratio,center}
\begin{minipage}{\textwidth}
\centering
\begingroup
\captionsetup{skip=4pt}
\caption{\textbf{Applicability matrix.} Rows are interview topics (bold rows mark category headers); columns are deployment tracks (Cloud, Edge, Mobile, and TinyML), corresponding to the hardware regime a question targets. Each panel is ordered for lookup: bold categories A--Z, then topics A--Z within each category. The three panels are filled independently, so a horizontal table row does not pair topics across columns. \textcolor{green!60!black}{\checkmark}: the topic has a physically meaningful substrate on that tier; \textcolor{red}{\ding{55}}: excluded because the concept has no defensible realization there (memory footprint, interconnect, power budget, or operational envelope). The grid records semantic validity for topic $\times$ track authoring, not item counts. Of the $\numtopics{} \times 4$ possible pairs, \numapplicablepairs{} are applicable and \numexcludedpairs{} are excluded.}
\label{tab:applicability-matrix}
\endgroup
\vspace{1pt}
\footnotesize
\renewcommand{\arraystretch}{1.25}
\setlength{\tabcolsep}{0.5pt}
\providecommand{\mtxHdr}[1]{\rotatebox{90}{\tiny #1}}
% tabularx: three X columns = equal Topic width; m{1.52em} = identical Cloud–TinyML in each panel.
\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}X >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} @{\hspace{5pt}} >{\raggedright\arraybackslash}X >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} @{\hspace{5pt}} >{\raggedright\arraybackslash}X >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} >{\centering\arraybackslash}m{1.52em} @{}}
\toprule
\textbf{Topic} & \mtxHdr{Cloud} & \mtxHdr{Edge} & \mtxHdr{Mobile} & \mtxHdr{TinyML} & \textbf{Topic} & \mtxHdr{Cloud} & \mtxHdr{Edge} & \mtxHdr{Mobile} & \mtxHdr{TinyML} & \textbf{Topic} & \mtxHdr{Cloud} & \mtxHdr{Edge} & \mtxHdr{Mobile} & \mtxHdr{TinyML} \\
\midrule
"""

TEX_SUFFIX = r"""\bottomrule
\end{tabularx}
\end{minipage}
\end{adjustbox}
\end{table*}
\FloatBarrier
"""


def sort_panels_alphabetically(panels: list[list[dict]]) -> None:
    """Sort categories A--Z and topics within each category (same as sort_app_matrix)."""

    def cat_key(cat: dict) -> str:
        if cat.get("implicit"):
            return "data"
        return (cat["name"] or "").lower()

    def topic_key(t: dict) -> str:
        return t["name"].lower()

    for col in range(3):
        cats = panels[col]
        cats.sort(key=cat_key)
        for cat in cats:
            cat["topics"].sort(key=topic_key)


def load_panels_from_yaml() -> list[list[dict]]:
    raw = yaml.safe_load(YAML_PATH.read_text())
    panels_data = raw["panels"]
    panels: list[list[dict]] = [[], [], []]
    for col in range(3):
        for cat in panels_data[col]["categories"]:
            implicit = cat.get("implicit", False)
            if implicit:
                entry = {"name": None, "count": None, "implicit": True, "topics": []}
            else:
                entry = {
                    "name": cat["name"],
                    "count": cat["count"],
                    "implicit": False,
                    "topics": [],
                }
            for t in cat["topics"]:
                tr = t["tracks"]
                if len(tr) != 4:
                    raise ValueError(f"Need 4 tracks for topic {t['name']!r}")
                entry["topics"].append({"name": t["name"], "tracks": tr})
            panels[col].append(entry)
    if raw.get("sort_panels", True):
        sort_panels_alphabetically(panels)
    return panels


def generate_tex() -> None:
    panels = load_panels_from_yaml()
    flat = flatten_panels(panels)
    body = emit_table_body(flat)
    TEX_PATH.write_text(TEX_PREFIX + body + "\n" + TEX_SUFFIX)
    print(f"Wrote {TEX_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export",
        action="store_true",
        help="Write applicability_matrix.yaml from current app_matrix.tex",
    )
    args = ap.parse_args()
    if args.export:
        export_yaml_from_tex()
    else:
        if not YAML_PATH.is_file():
            raise SystemExit(
                f"Missing {YAML_PATH}; run with --export once if app_matrix.tex exists."
            )
        generate_tex()


if __name__ == "__main__":
    main()
