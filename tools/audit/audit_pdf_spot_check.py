#!/usr/bin/env python3
"""
For each saved PDF in audit-pdf-output/, locate the page that contains
the specific math content we fixed and emit a navigation manifest.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "audit-pdf-output"

# (volume, chapter, friendly description, regex to find on page)
TARGETS = [
    ("vol1", "introduction",
     "mem_span/compute_span (10^N) in engineering crux callout",
     r"engineering crux|memory.*span|compute.*span|10\s*[\^to]+\s*\d"),
    ("vol1", "benchmarking",
     "BERT arithmetic intensity equations (md-wrapped $..$)",
     r"BERT.*arithmetic intensity|FLOPs|arithmetic intensity"),
    ("vol1", "appendix_dam",
     "ex4 GPU old/new costs (md-wrapped $\\times$)",
     r"ex4|GPU.*old|GPU.*new|\$[\\\.0-9 ]*times"),
    ("vol1", "appendix_algorithm",
     "P_str scientific notation (md_math + sci_latex)",
     r"power.*=|\\?times.*10|sustained.*power"),
    ("vol1", "model_compression",
     'Callout title: "The 4× MobileNet Win"',
     r"4×\s*MobileNet Win|The 4.*MobileNet"),
    ("vol2", "conclusion",
     "machine_ops_sci / brain_ops_sci (md_math)",
     r"machine.*op|brain.*ops?|10\s*\^?\s*16|FLOPs?/s"),
    ("vol2", "inference",
     "lifetime_queries_latex_str + KV cache fig-cap (Unicode)",
     r"lifetime queries|KV Cache Wall|35 GB \+ 82 GB"),
    ("vol2", "introduction",
     'AI Triad fig-cap: "$10^{25}$ FLOPS"',
     r"AI Triad at Scale|10.*FLOPS|frontier training"),
    ("vol2", "appendix_dam",
     "ex4 GPU old/new costs (md-wrapped)",
     r"ex4|GPU.*old|GPU.*new"),
    ("vol2", "appendix_c3",
     'Callout title: "The C³ Tax on a 100,000-GPU Cluster"',
     r"C³ Tax|100,?000-?GPU"),
    ("vol2", "robust_ai",
     "p_rate_meta_str (md_math wrapped 10^{-4})",
     r"p_rate_meta|meta.*p.*rate|10\^?\{?-4\}?"),
]


def page_text(pdf: Path, page_no: int) -> str:
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
             str(pdf), "-"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout
    except Exception:
        return ""


def total_pages(pdf: Path) -> int:
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                             text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


def find_pages(pdf: Path, pattern: str) -> list[int]:
    pat = re.compile(pattern, re.I)
    matches: list[int] = []
    n = total_pages(pdf)
    for p in range(1, n + 1):
        if pat.search(page_text(pdf, p)):
            matches.append(p)
    return matches


def main():
    lines = ["# Math Rendering — Visual Spot-Check Map", "",
             "For each chapter we patched, the relevant PDF pages and PNG ",
             "filenames are listed below. Open the PNG in Preview to verify ",
             "math renders correctly.", "",
             "PDFs and PNGs live under `audit-pdf-output/<vol>/<chap>/`.", ""]

    for vol, chap, desc, pattern in TARGETS:
        chap_dir = OUT / vol / chap
        pdf = chap_dir / f"{chap}.pdf"
        if not pdf.exists():
            lines += [f"## `{vol}/{chap}` — _PDF missing_", ""]
            continue

        pages = find_pages(pdf, pattern)
        lines += [f"## `{vol}/{chap}` — {desc}",
                  f"- pdf: `{pdf.relative_to(REPO)}`",
                  f"- total pages: **{total_pages(pdf)}**"]
        if pages:
            shown = pages[:8]
            lines.append(f"- candidate pages (regex hits): "
                         + ", ".join(f"**{p}**" for p in shown)
                         + (" ..." if len(pages) > len(shown) else ""))
            for p in shown:
                lines.append(f"  - `audit-pdf-output/{vol}/{chap}/pages/page-"
                             f"{p:0{max(2, len(str(total_pages(pdf))))}d}.png`")
        else:
            lines.append("- _no regex match — open the chapter and search "
                         "manually_")
        lines.append("")

    out_md = REPO / "audit-pdf-spot-check.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
