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

Publishing (the studysheets-*.yml workflows; no LaTeX needed for --manifest or --live-volumes):

    build.py --stage DIR [volN ...]   # after a build: flat volN_<stem>_studysheet.pdf files plus one
                                      # MLSysBook-StudySheets-VolN.pdf per volume, in book order
    build.py --manifest PATH          # the book's chapter -> sheet URL map (HTML pre-render)
    build.py --live-volumes           # print LIVE_VOLUMES for the workflows

Released volumes are served from the `studysheets-latest` release; every volume, released or not,
is in the `studysheets-dev` prerelease, which the dev preview links to for unreleased volumes.
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

# Volumes whose sheets are published live. Matches LIVE_VOLUMES in slides-publish-live.yml and the
# release gate in shared/config/site-head.html (Volumes III and IV staging-only, 2026-10-23).
# To release a volume, add it here; the workflows read this list through --live-volumes.
LIVE_VOLUMES = ("vol1", "vol2")
VOLUMES = ("vol1", "vol2", "vol3", "vol4")
RELEASES = "https://github.com/harvard-edge/cs249r_book/releases/download"
LIVE_TAG, DEV_TAG = "studysheets-latest", "studysheets-dev"


def sheets(scope: str | None = None) -> list[Path]:
    if scope and scope.endswith(".tex"):
        return [Path(scope).resolve()]
    vols = [scope] if scope else list(VOLUMES)
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


def asset_name(tex: Path) -> str:
    """Release asset name: GitHub releases are flat, so the volume prefixes the file."""
    return f"{tex.parents[1].name}_{tex.stem}.pdf"


def sheet_url(tex: Path) -> str:
    vol = tex.parents[1].name
    return f"{RELEASES}/{LIVE_TAG if vol in LIVE_VOLUMES else DEV_TAG}/{asset_name(tex)}"


def book_order(vol: str) -> list[Path]:
    """The volume's sheets in the chapter order of its PDF config (the canonical order)."""
    config = (BOOKS / "config" / f"_quarto-pdf-{vol}.yml").read_text()
    order = []
    for chapter in re.findall(rf"^\s*-\s*{vol}/(\d+_[^/\s]+)/\1\.qmd\s*$", config, flags=re.M):
        for tex in sorted((BOOKS / vol / chapter).glob("*_studysheet.tex")):
            if tex not in order:
                order.append(tex)
    missing = [t for t in sheets(vol) if t not in order]
    if missing:
        raise SystemExit(f"{vol}: sheets for chapters not in _quarto-pdf-{vol}.yml: {[rel(t) for t in missing]}")
    return order


def chapter_title(tex: Path) -> str:
    """The chapter's title as the book shows it: front-matter `title:` or its first `# ` heading."""
    qmd = (tex.parent / f"{tex.parent.name}.qmd").read_text()
    found = (re.search(r'^title:\s*"?([^"\n]+)"?\s*$', qmd, flags=re.M)
             or re.search(r"^# (.+)$", qmd, flags=re.M))
    return re.sub(r"\s*\{[^}]*\}", "", found.group(1)).strip() if found else tex.parent.name


def book_parts(vol: str) -> dict[str, str]:
    """Chapter folder -> the sidebar section it is listed under in _quarto-html-volN.yml ("" if none)."""
    import yaml
    config = yaml.safe_load((BOOKS / "config" / f"_quarto-html-{vol}.yml").read_text())
    sidebars = (config.get("website") or {}).get("sidebar") or []
    out: dict[str, str] = {}

    def walk(items, part):
        for item in items or []:
            if isinstance(item, dict) and "contents" in item:
                walk(item["contents"], item.get("section", part))
            else:
                href = item.get("href", "") if isinstance(item, dict) else str(item)
                m = re.match(rf"^{vol}/(\d+_[^/]+)/\1\.qmd$", str(href))
                if m:
                    out[m.group(1)] = part
    for sb in sidebars if isinstance(sidebars, list) else [sidebars]:
        walk(sb.get("contents"), "")
    return out


def volume_url(vol: str) -> str:
    return f"{RELEASES}/{LIVE_TAG if vol in LIVE_VOLUMES else DEV_TAG}/MLSysBook-StudySheets-Vol{vol[3:]}.pdf"


def write_manifest(path: Path) -> None:
    """Per volume: chapter folder -> {"url"} (the line under each chapter title), and "_volume": the
    combined PDF plus every sheet in book order with its title and part (the Study Sheets page)."""
    import json
    manifest = {}
    for vol in VOLUMES:
        order, parts = book_order(vol), book_parts(vol)
        entry = {t.parent.name: {"url": sheet_url(t)} for t in order}
        entry["_volume"] = {"url": volume_url(vol), "count": len(order), "pages": 2 * len(order),
                            "sheets": [{"chapter": t.parent.name, "title": chapter_title(t),
                                        "part": parts.get(t.parent.name, ""), "url": sheet_url(t)}
                                       for t in order]}
        manifest[vol] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"build.py: {sum(e['_volume']['count'] for e in manifest.values())} study sheets -> {path}")


def stage(dest: Path, vols: list[str]) -> None:
    """Copy built PDFs to flat release names and combine each volume in book order."""
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    for vol in vols:
        pdfs = []
        for tex in book_order(vol):
            pdf = OUT / vol / f"{tex.stem}.pdf"
            if not pdf.exists():
                raise SystemExit(f"{rel(pdf)} missing; build {vol} first")
            shutil.copy2(pdf, dest / asset_name(tex))
            pdfs.append(pdf)
        combined = dest / f"MLSysBook-StudySheets-Vol{vol[3:]}.pdf"
        subprocess.run(["pdfunite", *map(str, pdfs), str(combined)], check=True)
        print(f"{vol}: {len(pdfs)} sheets -> {rel(combined)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scope", nargs="*", help="volN, or a *_studysheet.tex path; default every sheet")
    ap.add_argument("--manifest", type=Path, help="write the book's chapter -> sheet URL manifest and exit")
    ap.add_argument("--stage", type=Path, help="stage built PDFs for a release into this folder and exit")
    ap.add_argument("--live-volumes", action="store_true", help="print the live volumes and exit")
    args = ap.parse_args()
    if args.live_volumes:
        raise SystemExit(print(" ".join(LIVE_VOLUMES)))
    if args.manifest:
        raise SystemExit(write_manifest(args.manifest))
    if args.stage:
        raise SystemExit(stage(args.stage, args.scope or list(VOLUMES)))
    bad = 0
    targets = [t for s in (args.scope or [None]) for t in sheets(s)]
    for tex in targets:
        rep = build(tex)
        probs = problems(rep)
        bad += bool(probs)
        print(f"{'✗' if probs else '✓'} {rel(tex)} -> {rel(rep['pdf'])}  free {rep['free']}")
        for p in probs:
            print("    ", p)
    raise SystemExit(1 if bad else 0)
