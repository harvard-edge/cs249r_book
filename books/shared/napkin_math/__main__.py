"""Build Napkin Math sheets.

Two steps, so the PDF printer does not need mlsysim and the content step
does not need a browser:
  python3 -m napkin_math html vol1 [vol2 ...] [--out DIR]   # needs mlsysim
  python3 -m napkin_math pdf  vol1 [vol2 ...] [--out DIR]   # needs playwright
Run from books/shared. The pdf step prints a layout report and exits 1 if any
item runs past its page or any equation is wider than its column.
"""
import argparse
import importlib
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["html", "pdf"])
    ap.add_argument("vols", nargs="+")
    ap.add_argument("--out", default="_build/napkin-math")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = False
    for vol in a.vols:
        html_path, pdf_path = out / f"napkin-math-{vol}.html", out / f"napkin-math-{vol}.pdf"
        if a.cmd == "html":
            from .core import render
            html_path.write_text(render(importlib.import_module(f".{vol}", __package__).build()))
            print(f"{vol}: {html_path}")
            continue
        from .core import to_pdf
        report = to_pdf(html_path, pdf_path)
        print(f"{vol}: {pdf_path}")
        for r in report:
            print("   ", r)
        bad |= any("free px" not in r for r in report)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
