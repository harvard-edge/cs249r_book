#!/usr/bin/env python3
"""Post-render script: save LaTeX build log to the output directory.

Quarto deletes intermediate files (including index.log) after post-render
scripts finish. This script copies the log while it still exists, so the
binder post-build validator can scan it for overfull hbox, underfull vbox,
and other LaTeX diagnostics.
"""
import re
import shutil
import sys
from pathlib import Path

OVERFULL_RE = re.compile(
    r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too (?:wide|high)\).*?lines (\d+)"
)

def main():
    script_dir = Path(__file__).resolve().parent.parent  # quarto/

    # The log file name matches the output-file in _quarto.yml
    # Try common names: index.log, Machine-Learning-Systems-Vol1.log, Vol2.log
    log_src = None
    for candidate in (
        "index.log",
        "Machine-Learning-Systems-Vol1.log",
        "Machine-Learning-Systems-Vol2.log",
    ):
        p = script_dir / candidate
        if p.is_file():
            log_src = p
            break

    if log_src is None:
        # Also check the Quarto temp render directory
        for d in script_dir.glob(".quarto/**/index.log"):
            log_src = d
            break

    if log_src is None:
        print("[latex-log] No LaTeX log found (non-PDF build or already cleaned)")
        return

    # Find the output directory
    for vol in ("vol1", "vol2"):
        out_dir = script_dir / "_build" / f"pdf-{vol}"
        if out_dir.is_dir():
            dst = out_dir / "latex-build.log"
            shutil.copy2(log_src, dst)
            print(f"[latex-log] Saved {log_src.name} -> {dst.relative_to(script_dir)}")

            # Print a quick summary of severe issues for the build output
            text = log_src.read_text(errors="replace")
            severe = [(float(m.group(1)), int(m.group(2)))
                      for m in OVERFULL_RE.finditer(text)
                      if float(m.group(1)) >= 20.0]
            if severe:
                print(f"[latex-log] WARNING: {len(severe)} severe layout overflows (>= 20pt)")
                for pts, line in sorted(severe, key=lambda x: -x[0])[:5]:
                    print(f"[latex-log]   {pts:.1f}pt overflow at .tex line {line}")
            else:
                print(f"[latex-log] No severe layout overflows detected")
            return

    print("[latex-log] No _build/pdf-vol*/ output directory found")

if __name__ == "__main__":
    main()
