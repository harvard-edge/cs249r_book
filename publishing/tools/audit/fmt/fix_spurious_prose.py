#!/usr/bin/env python3
"""Fix fmt() calls whose rendered output triggers spurious-.0 prose flags.

See ``fmt/README.md`` for the full chapter workflow.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
FMT_ASSIGN = re.compile(
    r"^(\s*)(\w+_str)\s*=\s*(fmt(?:_percent)?|fmt_int)\((.+)\)\s*(#.*)?$"
)
SPURIOUS = re.compile(r"^\d[\d,]*\.0$")


def _fresh_ns(prior: list[str]) -> dict:
    ns: dict = {"__builtins__": __builtins__}
    for code in prior:
        exec(compile(code, "<string>", "exec"), ns)  # noqa: S102
    return ns


def _rendered_spurious(line: str, ns: dict, prior_lines: list[str]) -> bool:
    m = FMT_ASSIGN.match(line)
    if not m or m.group(3) == "fmt_int":
        return False
    indent, var, fn, inner, comment = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or ""
    trial = dict(ns)
    try:
        exec(compile("\n".join(prior_lines), "<string>", "exec"), trial)  # noqa: S102
        exec(f"_x = {fn}({inner})", trial)  # noqa: S102
        out = str(trial["_x"])
    except Exception:
        return False
    # strip prefix/suffix for numeric check
    num = re.sub(r"^[^\d-]*", "", out)
    num = re.sub(r"[^\d.,-].*$", "", num)
    return bool(SPURIOUS.match(num))


def _to_fmt_int(line: str) -> str | None:
    m = FMT_ASSIGN.match(line)
    if not m or m.group(3) != "fmt":
        return None
    indent, var, _, inner, comment = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or ""
    parts = inner.split(",", 1)
    expr = parts[0].strip()
    rest = parts[1] if len(parts) > 1 else ""
    rest = re.sub(r",?\s*precision\s*=\s*\d+", "", rest)
    rest = re.sub(r",\s*,", ",", rest).strip(", ")
    if rest:
        return f"{indent}{var} = fmt_int({expr}, {rest}){comment}"
    return f"{indent}{var} = fmt_int({expr}){comment}"


def fix_file(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    prior: list[str] = []
    total = 0
    out: list[str] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            out.append(lines[i])
            start = i + 1
            i += 1
            while i < len(lines) and not CELL_END.match(lines[i]):
                i += 1
            cell_lines = lines[start:i]
            ns = _fresh_ns(prior)
            changed = True
            while changed:
                changed = False
                for j, ln in enumerate(cell_lines):
                    if not _rendered_spurious(ln, ns, cell_lines[:j]):
                        continue
                    new_ln = _to_fmt_int(ln)
                    if new_ln and new_ln != ln:
                        cell_lines[j] = new_ln
                        total += 1
                        changed = True
                        break
                if changed:
                    ns = _fresh_ns(prior)
                    exec(compile("\n".join(cell_lines), "<string>", "exec"), ns)  # noqa: S102
            cell_code = "\n".join(cell_lines)
            if "fmt_int(" in cell_code and "fmt_int" not in "\n".join(cell_lines[:5]):
                for j, ln in enumerate(cell_lines):
                    if "from mlsysim.fmt import" in ln and "fmt_int" not in ln:
                        cell_lines[j] = ln.replace(
                            "from mlsysim.fmt import",
                            "from mlsysim.fmt import fmt_int, ",
                            1,
                        )
                        break
                cell_code = "\n".join(cell_lines)
            prior.append(cell_code)
            out.extend(cell_code.splitlines())
            if i < len(lines):
                out.append(lines[i])
            i += 1
            continue
        out.append(lines[i])
        i += 1
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return total


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]]
    total = 0
    for f in files:
        n = fix_file(f)
        if n:
            print(f"OK {f}: {n} prose-spurious fixes")
        total += n
    print(f"Total: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
