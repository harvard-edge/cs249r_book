"""Build and check chapter study sheets.

A sheet is plain LaTeX on the shared class `studysheet.cls` (this folder), stored next to its
chapter as `books/volN/NN_chapter/<stem>_studysheet.tex`. Its numbers come from
`<stem>_studysheet_numbers.py`, which computes them from mlsysim and writes
`<stem>_studysheet_numbers.tex` (\\setnum lines, generated, not committed).

The volume's accent color is read from the book itself, `books/volN/tex/theme-colors.tex`
(`accentcolor`), so a sheet always matches its volume.

    python3 books/shared/studysheet/build.py                  # every sheet
    python3 books/shared/studysheet/build.py vol1             # one volume
    python3 books/shared/studysheet/build.py path/to/x_studysheet.tex
PDFs go to books/_build/studysheets/volN/.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOOKS = ROOT / "books"
INCLUDES = BOOKS / "shared" / "_includes"
OUT = BOOKS / "_build" / "studysheets"
MIN_FREE_PT = 24.0            # every column ends with at least this much free space (studysheet.md)
MAX_IDEA_LINES = 3           # each key idea at most 3 printed lines (studysheet.md)


def sheets(scope: str | None = None) -> list[Path]:
    if scope and scope.endswith(".tex"):
        return [Path(scope).resolve()]
    vols = [scope] if scope else ["vol1", "vol2", "vol3", "vol4"]
    return sorted(p for v in vols for p in (BOOKS / v).glob("*/*_studysheet.tex"))


def volume_of(tex: Path) -> int:
    m = re.search(r"\\sheetvolume\{(\d)\}", tex.read_text())
    if not m:
        raise SystemExit(f"{tex}: no \\sheetvolume{{N}}")
    return int(m.group(1))


def accent(volume: int) -> str:
    """The volume's accentcolor, as the book defines it (last definition wins, as in LaTeX)."""
    theme = (BOOKS / f"vol{volume}" / "tex" / "theme-colors.tex").read_text()
    found = re.findall(r"^\s*\\definecolor\{accentcolor\}\{HTML\}\{([0-9A-Fa-f]{6})\}", theme, flags=re.M)
    if not found:
        raise SystemExit(f"vol{volume}/tex/theme-colors.tex defines no accentcolor")
    return found[-1].upper()


def numbers(tex: Path) -> None:
    script = tex.with_name(tex.stem + "_numbers.py")
    if script.exists():
        env = {**os.environ, "PYTHONPATH": f"{ROOT / 'mlsysim'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, env=env)


def ensure_macros() -> None:
    if not (INCLUDES / "_macros.tex").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_notation.py")], cwd=ROOT, check=True)


def compile_tex(tex: Path, out: Path, volume: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TEXINPUTS": f"{tex.parent}//:{HERE}//:{INCLUDES}//:"}
    first_line = rf"\def\SheetAccent{{{accent(volume)}}}\input{{{tex.name}}}"
    subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={tex.stem}",
                    f"-output-directory={out}", first_line], cwd=tex.parent, capture_output=True, text=True, env=env)
    logf = out / f"{tex.stem}.log"
    log = logf.read_text(errors="replace") if logf.exists() else ""
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return {"pdf": out / f"{tex.stem}.pdf", "pages": int(pages.group(1)) if pages else 0,
            "free": [(int(p), float(a), float(b)) for p, a, b in
                     re.findall(r"STUDYSHEET-FREE page=(\d+) left=([-\d.]+)pt right=([-\d.]+)pt", log)],
            "idea_lines": [int(x) for x in re.findall(r"STUDYSHEET-IDEA lines=(\d+)", log)],
            "overfull": [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)],
            "errors": re.findall(r"^! (.*)$", log, flags=re.M)}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def problems(rep: dict) -> list[str]:
    out = [f"LaTeX error: {e}" for e in rep["errors"]]
    if rep["pages"] != 2:
        out.append(f"sheet has {rep['pages']} pages, not 2")
    for page, a, b in rep["free"]:
        out += [f"page {page} {side} column: {f:.1f} pt free (< {MIN_FREE_PT:g})"
                for side, f in (("left", a), ("right", b)) if f < MIN_FREE_PT]
    out += [f"key idea {i + 1} runs {n} lines (> {MAX_IDEA_LINES})"
            for i, n in enumerate(rep.get("idea_lines", [])) if n > MAX_IDEA_LINES]
    return out + [f"overfull box {x:.1f} pt" for x in rep["overfull"] if x > 1.0]


def build(tex: Path) -> dict:
    volume = volume_of(tex)
    numbers(tex)
    ensure_macros()
    return compile_tex(tex, OUT / f"vol{volume}", volume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scope", nargs="?", help="volN, or a *_studysheet.tex path; default every sheet")
    bad = 0
    for tex in sheets(ap.parse_args().scope):
        rep = build(tex)
        probs = problems(rep)
        bad += bool(probs)
        print(f"{'✗' if probs else '✓'} {rel(tex)} -> {rel(rep['pdf'])}  free {rep['free']}")
        for p in probs:
            print("    ", p)
    raise SystemExit(1 if bad else 0)
