"""
Printed size of matplotlib body figures.

Every numbered body figure is scaled to the text block, so the size a reader
sees is the matplotlib font size times the print scale (text-block width over
the width of the tightly cropped image). The style module fixes the font sizes;
this check measures the other half, which no default can fix because each cell
picks its own ``figsize`` and labels outside the axes widen the crop.

Each chapter runs in its own subprocess: its ``{python}`` cells execute in
order (later cells read values from earlier LEGO cells), and every
``plt.show()`` inside a ``::: {#fig-...}`` div is measured instead of drawn.

Findings per figure:

* ``print-scale``: scale outside ``PRINT_SCALE_RANGE``. Below the floor the
  plot text prints visibly smaller than the surrounding figures; above it,
  larger than the body text.
* ``figure-font``: any text set below ``MIN_FONT_PT`` before scaling, i.e. a
  cell overriding the style's sizes with a smaller literal.
* ``raw-figure``: the cell creates a figure with ``plt.subplots``/``plt.figure``
  without calling ``setup_plot``/``set_book_style`` itself, so its look depends
  on whichever earlier cell last set the rcParams.
* ``figure-exec``: the cell raised, so nothing could be measured.

The worker half of this module (``__main__``) uses only the standard library
and matplotlib so it can run under any interpreter that builds the book.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

# Thresholds are not defined here: the worker reads PRINT_TEXT_WIDTH_IN,
# PRINT_SCALE_RANGE, and MIN_FONT_PT from mlsysim.viz.style, the same module
# that sets the figure fonts, and reports them with its measurements.
MEASURE_DPI = 100

_CELL = re.compile(
    r"^::: *\{#(fig-[\w-]+)(?P<attrs>[^}]*)\}|^:::+\s*$|^```\{python\}\n(.*?)^```\s*$",
    re.S | re.M,
)
_RAW_FIGURE = re.compile(r"\bplt\.(subplots|figure)\(")
_STYLED = re.compile(r"\b(setup_plot|set_book_style|plot_\w+|bar_compare)\(")


def _cells(src: str):
    """Yield (fig_id or None, code) for each python cell, tagged with its enclosing figure div."""
    div_id = None
    for m in _CELL.finditer(src):
        if m.group(1):
            # Margin and reduced-width figures are not scaled to the text block.
            attrs = m.group("attrs") or ""
            div_id = None if ("column-margin" in attrs or "width=" in attrs) else m.group(1)
        elif m.group(3) is not None:
            yield div_id, m.group(3)
        else:
            div_id = None


def _line_of(src: str, fig_id: str) -> int:
    """1-based line number of a figure div's opening fence, or 0."""
    idx = src.find("{#" + fig_id)
    return src.count("\n", 0, idx) + 1 if idx >= 0 else 0


def measure_chapter(qmd: Path, repo_root: Path, timeout: int = 900) -> Dict:
    """Run the worker on one chapter and return its JSON report."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(qmd), str(repo_root)],
        cwd=str(repo_root / "books"), capture_output=True, text=True, timeout=timeout,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (proc.stderr or proc.stdout).strip()[-300:], "figures": []}


def check_chapters(qmds: List[Path], repo_root: Path, workers: int = 4) -> List[Dict]:
    """Measure every chapter; returns flat finding dicts (file, line, code, message, severity)."""
    findings: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reports = list(pool.map(lambda q: (q, measure_chapter(q, repo_root)), qmds))
    for qmd, rep in reports:
        rel = str(qmd.relative_to(repo_root))
        lim = rep.get("limits") or {}
        TEXT_WIDTH_IN = lim.get("text_width_in", 0)
        SCALE_MIN, SCALE_MAX = lim.get("scale_range", (0, 0))
        FONT_MIN = lim.get("min_font_pt", 0)
        src = qmd.read_text(encoding="utf-8")
        if rep.get("error"):
            findings.append(dict(file=rel, line=0, code="figure-exec", severity="warning",
                                 message=f"chapter worker failed: {rep['error']}"))
        for f in rep.get("figures", []):
            line = _line_of(src, f["id"])
            if f.get("error"):
                findings.append(dict(file=rel, line=line, code="figure-exec", severity="warning",
                                     message=f"{f['id']}: cell raised {f['error']}"))
                continue
            if f.get("raw"):
                findings.append(dict(file=rel, line=line, code="raw-figure", severity="warning",
                                     message=f"{f['id']}: plt.subplots/plt.figure without setup_plot() or set_book_style()"))
            if "width_in" not in f or not TEXT_WIDTH_IN:
                continue
            scale = TEXT_WIDTH_IN / f["width_in"]
            if not SCALE_MIN <= scale <= SCALE_MAX:
                findings.append(dict(
                    file=rel, line=line, code="print-scale", severity="warning",
                    message=(f"{f['id']}: prints at {scale:.2f}x (cropped {f['width_in']:.2f} in, "
                             f"figsize width {f['figsize_w']:.1f} in); 10 pt text prints at "
                             f"{10 * scale:.1f} pt. Target {SCALE_MIN:.2f}-{SCALE_MAX:.2f}x, "
                             f"a cropped width near {TEXT_WIDTH_IN / 0.9:.1f} in."),
                ))
            if f["min_font"] < FONT_MIN:
                findings.append(dict(
                    file=rel, line=line, code="figure-font", severity="warning",
                    message=(f"{f['id']}: text set at {f['min_font']:g} pt (prints at "
                             f"{f['min_font'] * scale:.1f} pt); minimum is {FONT_MIN:g} pt"),
                ))
    return findings


# ---------------------------------------------------------------------------
# Worker: executes one chapter's cells and measures each figure.
# ---------------------------------------------------------------------------

def _worker(qmd_path: str, repo_root: str) -> None:
    import io
    import os

    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "mlsysim"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path(qmd_path).read_text(encoding="utf-8")
    figures: List[Dict] = []
    current: Dict = {"id": None, "raw": False}

    def _shown(t) -> bool:
        # alpha=0 spacer labels reserve layout room and never print.
        return t.get_visible() and t.get_alpha() != 0 and bool(t.get_text().strip())

    def _min_font(fig) -> float:
        sizes = []
        for ax in fig.axes:
            texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
            texts += ax.get_xticklabels() + ax.get_yticklabels()
            leg = ax.get_legend()
            if leg:
                texts += list(leg.get_texts())
            sizes += [t.get_fontsize() for t in texts if _shown(t)]
        sizes += [t.get_fontsize() for t in fig.texts if _shown(t)]
        return min(sizes) if sizes else 99.0

    def _show(*_a, **_k):
        fid = current["id"]
        if not fid or any(f["id"] == fid for f in figures):
            return
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=MEASURE_DPI, bbox_inches="tight")
        buf.seek(16)
        width_px = int.from_bytes(buf.read(4), "big")  # PNG IHDR width
        figures.append({
            "id": fid, "raw": current["raw"],
            "width_in": width_px / MEASURE_DPI,
            "figsize_w": float(fig.get_size_inches()[0]),
            "min_font": float(_min_font(fig)),
        })

    plt.show = _show
    ns: Dict = {"__name__": "__main__"}
    for fid, code in _cells(src):
        current["id"] = fid
        current["raw"] = bool(fid and _RAW_FIGURE.search(code) and not _STYLED.search(code))
        try:
            with open(os.devnull, "w") as devnull:
                old = sys.stdout
                sys.stdout = devnull
                try:
                    exec(compile(code, f"{qmd_path}:{fid}", "exec"), ns)  # noqa: S102
                finally:
                    sys.stdout = old
        except Exception as e:  # noqa: BLE001
            if fid:
                figures.append({"id": fid, "error": repr(e)[:160]})
        plt.close("all")

    from mlsysim.viz import style
    limits = {"text_width_in": style.PRINT_TEXT_WIDTH_IN,
              "scale_range": list(style.PRINT_SCALE_RANGE),
              "min_font_pt": style.MIN_FONT_PT}
    print(json.dumps({"figures": figures, "limits": limits}))


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
