#!/usr/bin/env python3
"""Render all Matplotlib figures from QMD files to PNG for visual inspection.

Usage:
    cd book/quarto
    python ../tools/scripts/testing/render_figures.py [chapter_path ...]

Examples:
    # Render figures from specific chapters:
    python ../tools/scripts/testing/render_figures.py contents/vol1/training/training.qmd

    # Render figures from all modified chapters:
    python ../tools/scripts/testing/render_figures.py \
        contents/vol1/training/training.qmd \
        contents/vol1/data_selection/data_selection.qmd \
        contents/vol1/ops/ops.qmd

    # Render ALL vol1 figures:
    python ../tools/scripts/testing/render_figures.py contents/vol1/*/*.qmd

Output goes to /tmp/rendered_figures/<chapter>/<fig-label>.png
"""

import re
import sys
import os
import textwrap
from pathlib import Path


def extract_python_figures(qmd_path: str) -> list[dict]:
    """Extract Python code blocks that have a fig-* label from a QMD file."""
    with open(qmd_path, "r") as f:
        content = f.read()

    # Match ```{python} ... ``` blocks
    pattern = re.compile(
        r"```\{python\}\s*\n(.*?)```",
        re.DOTALL,
    )

    figures = []
    for match in pattern.finditer(content):
        block = match.group(1)

        # Check if it has a figure label
        label_match = re.search(r"#\|\s*label:\s*(fig-[\w-]+)", block)
        if not label_match:
            continue

        label = label_match.group(1)

        # Strip the #| directives to get just the Python code
        lines = block.split("\n")
        code_lines = [
            line for line in lines if not line.strip().startswith("#|")
        ]
        code = "\n".join(code_lines).strip()

        figures.append({"label": label, "code": code})

    return figures


def render_figure(code: str, output_path: str) -> bool:
    """Execute a figure's Python code and save to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Replace plt.show() with savefig
    modified_code = code.replace("plt.show()", f"plt.savefig('{output_path}', dpi=150, bbox_inches='tight')\nplt.close('all')")

    # If code doesn't call plt.show(), add savefig at the end
    if "plt.show()" not in code and "savefig" not in code:
        modified_code += f"\nplt.savefig('{output_path}', dpi=150, bbox_inches='tight')\nplt.close('all')"

    try:
        exec(modified_code, {"__name__": "__main__"})
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Ensure we're in the quarto directory for imports
    cwd = os.getcwd()
    if not os.path.exists("physx"):
        quarto_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "quarto",
        )
        if os.path.exists(os.path.join(quarto_dir, "physx")):
            os.chdir(quarto_dir)
            print(f"Changed to {quarto_dir}")

    sys.path.insert(0, ".")

    output_base = Path("/tmp/rendered_figures")
    total, success, failed = 0, 0, 0

    for qmd_path in sys.argv[1:]:
        if not os.path.exists(qmd_path):
            print(f"File not found: {qmd_path}")
            continue

        chapter_name = Path(qmd_path).stem
        figures = extract_python_figures(qmd_path)

        if not figures:
            print(f"\n{chapter_name}: no Matplotlib figures found")
            continue

        print(f"\n{'='*60}")
        print(f"{chapter_name}: {len(figures)} figures")
        print(f"{'='*60}")

        out_dir = output_base / chapter_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for fig in figures:
            total += 1
            out_file = out_dir / f"{fig['label']}.png"
            print(f"  Rendering {fig['label']}...", end=" ", flush=True)

            if render_figure(fig["code"], str(out_file)):
                success += 1
                print(f"OK -> {out_file}")
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{total} succeeded, {failed} failed")
    print(f"Output: {output_base}/")
    print(f"{'='*60}")

    if sys.platform == "darwin":
        print(f"\nTo view all figures:\n  open {output_base}/")


if __name__ == "__main__":
    main()
