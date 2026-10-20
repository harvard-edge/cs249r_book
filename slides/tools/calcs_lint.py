#!/usr/bin/env python3
"""Fail if a deck's calcs.py types a number instead of reading it from MLSysIM.

Applies to every images/calcs.py that writes numbers.tex through deckvals.
Such a file only selects values from MLSysIM, converts units, and formats
them, so the only numeric literals it may contain are:
  - 0, 1, 10, and 100 (complements, ratios, percent);
  - an exponent of 10 (``10 ** 11``, scientific-notation mantissas);
  - the digits argument of round(), and an index such as parents[3].
Anything else is an input that belongs in MLSysIM (registry or Scenarios.*).

usage: calcs_lint.py [calcs.py ...]   (default: every deckvals calcs.py)
"""

import ast
import sys
from pathlib import Path

ALLOWED = {0, 1, 10, 100}


def offending(path):
    tree = ast.parse(path.read_text(), str(path))
    exempt = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exempt.add(id(node.right))
            if isinstance(node.right, ast.UnaryOp):
                exempt.add(id(node.right.operand))
        if isinstance(node, ast.Subscript):  # an index (parents[3]) is not data
            exempt.add(id(node.slice))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "round" and len(node.args) > 1:
            exempt.add(id(node.args[1]))
            if isinstance(node.args[1], ast.UnaryOp):
                exempt.add(id(node.args[1].operand))
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and type(node.value) in (int, float)
                and id(node) not in exempt and node.value not in ALLOWED):
            bad.append((node.lineno, node.value))
    return sorted(bad)


def main(argv):
    root = Path(__file__).resolve().parents[1]
    files = [Path(a) for a in argv] or [
        p for p in sorted(root.glob("vol*/*/images/calcs.py")) if "from deckvals import" in p.read_text()]
    failed = 0
    for f in files:
        bad = offending(f)
        if bad:
            failed += 1
            shown = f.relative_to(root) if f.is_relative_to(root) else f
            print(f"  ✗ {shown}: typed numbers belong in MLSysIM")
            for line, value in bad:
                print(f"      line {line}: {value!r}")
    if not failed:
        print(f"  ✓ {len(files)} calcs.py file(s) read every number from MLSysIM")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
