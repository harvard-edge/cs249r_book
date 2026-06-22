#!/usr/bin/env python3
"""Fix fmt() precision errors using exec traceback line numbers.

See ``fmt/README.md`` for the full chapter workflow.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
REMOVE_KWARGS = re.compile(r",?\s*(?:drop_zero|allow_zero)\s*=\s*(?:True|False)")


def _clean_fmt_args(rest: str) -> str:
    rest = REMOVE_KWARGS.sub("", rest)
    rest = re.sub(r",?\s*precision\s*=\s*0", "", rest)
    rest = re.sub(r",\s*,", ",", rest).strip(", ")
    return rest


def _fix_fmt_call(call: str, msg: str) -> str | None:
    if "integer-like" in msg and re.search(r"precision\s*=\s*[1-9]", call):
        return re.sub(r"precision\s*=\s*\d+", "precision=0", call)
    if "not integer-like" in msg and re.search(r"precision\s*=\s*0", call):
        return re.sub(r"precision\s*=\s*0", "precision=1", call)
    if "formatted as '0'" in msg:
        pm = re.search(r"precision\s*=\s*(\d+)", call)
        n = int(pm.group(1)) + 1 if pm else 2
        return re.sub(r"precision\s*=\s*\d+", f"precision={n}", call)
    return None


def _fmt_call_expr(call: str) -> str | None:
    m = re.match(r"fmt(?:_percent)?\(\s*(.+)\)\s*$", call)
    if not m:
        return None
    inner = m.group(1)
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return inner[:i].strip()
    return inner.strip()


def _apply_fix(line: str, msg: str, ns: dict | None = None) -> str | None:
    vm = re.search(r"Value ([^\s]+)", msg)
    err_val = float(vm.group(1)) if vm else None
    if err_val is not None and ns is not None:
        calls = list(re.finditer(r"fmt(?:_percent)?\([^)]*(?:\([^)]*\)[^)]*)*\)", line))
        if len(calls) > 1:
            for m in calls:
                call = m.group(0)
                expr = _fmt_call_expr(call)
                if not expr:
                    continue
                try:
                    val = float(eval(expr, ns))  # noqa: S307
                except Exception:
                    continue
                if abs(val - err_val) <= max(1e-6, abs(err_val) * 1e-9):
                    fixed = _fix_fmt_call(call, msg)
                    if fixed and fixed != call:
                        return line[: m.start()] + fixed + line[m.end() :]
    if "integer-like" in msg and re.search(r"precision\s*=\s*[1-9]", line):
        return re.sub(r"precision\s*=\s*\d+", "precision=0", line)
    if "not integer-like" in msg and re.search(r"precision\s*=\s*0", line):
        if vm:
            val = float(vm.group(1))
            if abs(val - round(val)) <= 0.5:
                m = re.match(
                    r"^(\s*)(\w+_str)\s*=\s*fmt(?:_percent)?\(\s*(.+)\)\s*(#.*)?$",
                    line,
                )
                if m:
                    indent, var, inner, comment = m.group(1), m.group(2), m.group(3), m.group(4) or ""
                    parts = inner.split(",", 1)
                    expr = parts[0].strip()
                    rest = _clean_fmt_args(parts[1] if len(parts) > 1 else "")
                    if rest:
                        return f"{indent}{var} = fmt_int({expr}, {rest}){comment}"
                    return f"{indent}{var} = fmt_int({expr}){comment}"
        return re.sub(r"precision\s*=\s*0", "precision=1", line)
    if "formatted as '0'" in msg:
        pm = re.search(r"precision\s*=\s*(\d+)", line)
        n = int(pm.group(1)) + 1 if pm else 2
        return re.sub(r"precision\s*=\s*\d+", f"precision={n}", line)
    return None


def _preprocess(content: str) -> str:
    content = REMOVE_KWARGS.sub("", content)
    content = re.sub(
        r"fmt\(\s*int\(([^)]+)\)\s*,\s*precision\s*=\s*0",
        r"fmt_int(\1",
        content,
    )
    return content


def _error_lineno(exc: BaseException) -> int | None:
    tb = traceback.extract_tb(exc.__traceback__)
    for frame in reversed(tb):
        if frame.filename == "<string>":
            return frame.lineno
    return None


def _fresh_ns(prior_cells: list[str]) -> dict:
    ns: dict = {"__builtins__": __builtins__}
    for prior in prior_cells:
        exec(compile(prior, "<string>", "exec"), ns)  # noqa: S102
    return ns


def _statement_block(lines: list[str], lineno: int) -> tuple[int, int, str]:
    """Return (start, end_exclusive, joined text) for a possibly multiline assignment."""
    idx = lineno - 1
    start = idx
    while start > 0 and lines[start - 1].rstrip().endswith("\\"):
        start -= 1
    while start > 0 and not lines[start - 1].strip():
        start -= 1
    if start > 0 and "=" not in lines[start] and re.match(r"^\s*\w+_str\s*=", lines[start - 1]):
        start -= 1
    end = idx + 1
    depth = lines[start].count("(") - lines[start].count(")")
    while end < len(lines) and depth > 0:
        depth += lines[end].count("(") - lines[end].count(")")
        end += 1
    return start, end, "\n".join(lines[start:end])


def _apply_fix_block(block: str, msg: str, ns: dict | None = None) -> str | None:
    """Apply precision fixes to a single- or multi-line fmt assignment."""
    flat = " ".join(ln.strip() for ln in block.splitlines())
    fixed_flat = _apply_fix(flat, msg, ns)
    if not fixed_flat or fixed_flat == flat:
        return None
    indent = re.match(r"^(\s*)", block.splitlines()[0]).group(1)
    lhs = block.splitlines()[0].split("=")[0].strip()
    rhs = fixed_flat.strip().split("=", 1)[1].strip()
    return f"{indent}{lhs} = {rhs}"


def _fix_cell(code: str, prior_cells: list[str]) -> tuple[str, int, dict]:
    fixes = 0
    lines = code.splitlines()
    for _ in range(500):
        trial_ns = _fresh_ns(prior_cells)
        try:
            exec(compile(code, "<string>", "exec"), trial_ns)  # noqa: S102
            return code, fixes, trial_ns
        except ValueError as exc:
            if "Formatting Precision Error" not in str(exc):
                raise
            lineno = _error_lineno(exc)
            msg = str(exc)
            if lineno is None or lineno < 1 or lineno > len(lines):
                raise ValueError(f"No lineno for: {msg}") from exc
            start, end, block = _statement_block(lines, lineno)
            new_block = _apply_fix_block(block, msg, trial_ns)
            if not new_block:
                line = lines[lineno - 1]
                new_line = _apply_fix(line, msg, trial_ns)
                if not new_line or new_line == line:
                    raise ValueError(f"Cannot fix line {lineno}: {msg}\n  {line}") from exc
                lines[lineno - 1] = new_line
            else:
                new_lines = new_block.splitlines()
                lines[start:end] = new_lines
            code = "\n".join(lines)
            if "fmt_int(" in code:
                for j, ln in enumerate(lines):
                    if "from mlsysim.fmt import" in ln and "fmt_int" not in ln:
                        lines[j] = ln.replace(
                            "from mlsysim.fmt import",
                            "from mlsysim.fmt import fmt_int,",
                            1,
                        )
                        code = "\n".join(lines)
                        break
                    if re.match(r"from mlsysim import \*", ln):
                        lines.insert(j + 1, "from mlsysim.fmt import fmt_int")
                        code = "\n".join(lines)
                        break
            fixes += 1
        except Exception:
            raise
    raise RuntimeError("max iterations in cell")


def fix_file(path: Path) -> int:
    content = _preprocess(path.read_text(encoding="utf-8"))
    lines = content.splitlines()
    prior_cells: list[str] = []
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
            cell_code = "\n".join(lines[start:i])
            cell_code, n, _ = _fix_cell(cell_code, prior_cells)
            prior_cells.append(cell_code)
            total += n
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
    failed = 0
    total = 0
    for f in files:
        try:
            n = fix_file(f)
            print(f"OK {f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f}: {n} fixes")
            total += n
        except Exception as exc:
            failed += 1
            print(f"FAIL {f}: {exc}", file=sys.stderr)
    print(f"Total fixes: {total}, failed files: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
