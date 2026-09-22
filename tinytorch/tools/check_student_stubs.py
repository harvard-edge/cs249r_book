#!/usr/bin/env python3
"""
Inspect and verify student exercise stubs and scaffold docstring hygiene across all 20 modules.

Usage:
    python3 tools/check_student_stubs.py           # Summary table and verification
    python3 tools/check_student_stubs.py --verbose # Full detail of every exercise stub
    python3 tools/check_student_stubs.py --module 01  # Inspect one module
"""

import argparse
import glob
import re
import sys
from pathlib import Path

# Ensure tito is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tito.core.solutions import (
    SOLUTION_BEGIN_MARKER,
    apply_release_tier,
    clear_solution_regions,
)


def audit_module(py_path: Path, verbose: bool = False):
    """Analyze one module's source for student exercises and scaffold docstrings."""
    content = py_path.read_text(encoding="utf-8")
    cells = re.split(r"\n# %%\s*", content)

    exercises = []
    scaffolds = []
    leaked_scaffold = []

    for cell_idx, cell in enumerate(cells):
        if SOLUTION_BEGIN_MARKER not in cell:
            continue

        res, errors = apply_release_tier(cell, "student")
        cleared, clear_errs = clear_solution_regions(res)

        def_matches = list(
            re.finditer(
                r"(def\s+\w+\([^)]*\)[^:]*:.*?(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\'))",
                cleared,
                re.DOTALL,
            )
        )

        for m in def_matches:
            fn_head = m.group(0).splitlines()[0].strip()
            doc = m.group(2)
            pos = m.end()
            following = cleared[pos : pos + 400].strip()

            is_exercise = (
                "# BEGIN" in following
                and "raise NotImplementedError() #delete this line" in following
            )

            # Extract docstring summary (first non-empty line)
            doc_lines = [l.strip() for l in doc.splitlines() if l.strip()]
            summary = doc_lines[0] if doc_lines else "(no docstring summary)"
            todo = next((l for l in doc_lines if l.startswith("TODO:")), None)

            if is_exercise:
                # Extract the exact stub block
                stub_lines = []
                for line in cleared[pos : pos + 400].splitlines():
                    if any(
                        kw in line
                        for kw in (
                            "YOUR CODE HERE",
                            "BEGIN",
                            "NotImplementedError",
                            "END",
                        )
                    ):
                        stub_lines.append(line)
                    elif stub_lines:
                        break

                exercises.append(
                    {
                        "function": fn_head,
                        "summary": summary,
                        "todo": todo,
                        "stub": "\n".join(stub_lines),
                        "cell": cell_idx,
                    }
                )
            else:
                scaffolds.append(
                    {
                        "function": fn_head,
                        "summary": summary,
                        "cell": cell_idx,
                    }
                )
                if any(kw in doc for kw in ("TODO:", "HINT:", "HINTS:")):
                    leaked_scaffold.append(
                        {
                            "function": fn_head,
                            "doc": doc,
                            "cell": cell_idx,
                        }
                    )

    return {
        "module": py_path.parent.name,
        "exercises": exercises,
        "scaffolds": scaffolds,
        "leaked": leaked_scaffold,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit student exercise stubs")
    parser.add_argument(
        "--module", help="Specific module to inspect (e.g., 01, 01_tensor)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print full details for every exercise"
    )
    args = parser.parse_args()

    files = sorted(glob.glob(str(REPO_ROOT / "src/*/*.py")))
    if args.module:
        mod_key = args.module.zfill(2)
        files = [f for f in files if Path(f).parent.name.startswith(mod_key)]
        if not files:
            print(f"❌ Module {args.module} not found.")
            return 1

    total_exercises = 0
    total_scaffolds = 0
    total_leaks = 0

    print("=" * 90)
    print("TinyTorch Student Exercise & Scaffold Hygiene Audit")
    print("=" * 90)

    for f in files:
        data = audit_module(Path(f), verbose=args.verbose)
        mod_name = data["module"]
        ex_count = len(data["exercises"])
        scaf_count = len(data["scaffolds"])
        leak_count = len(data["leaked"])

        total_exercises += ex_count
        total_scaffolds += scaf_count
        total_leaks += leak_count

        status = "✅ PASS" if leak_count == 0 else f"❌ {leak_count} LEAKS"
        print(f"[{status}] {mod_name:20s} : {ex_count:2d} exercises, {scaf_count:2d} scaffold methods")

        if args.verbose or args.module:
            for idx, ex in enumerate(data["exercises"], 1):
                print(f"\n   Exercise {idx}: {ex['function']}")
                print(f"      Summary: {ex['summary']}")
                if ex["todo"]:
                    print(f"      {ex['todo']}")
                print("      Generated Stub:")
                for sline in ex["stub"].splitlines():
                    print(f"         {sline}")

        if data["leaked"]:
            for leak in data["leaked"]:
                print(f"   ⚠️  LEAK in {leak['function']}: contains scaffold TODO/HINTS in docstring!")

    print("=" * 90)
    print(f"Summary: {len(files)} modules audited")
    print(f"  • Total student exercises : {total_exercises}")
    print(f"  • Total scaffold methods   : {total_scaffolds}")
    print(f"  • Leaked scaffold comments : {total_leaks}")
    print("=" * 90)

    if total_leaks > 0:
        print("❌ Audit failed with leaked scaffold comments.")
        return 1

    print("✅ All student exercise stubs and scaffold docstrings are clean and consistent!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
