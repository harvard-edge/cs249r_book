#!/usr/bin/env python3
"""
codemod_fmt.py — paired (cell + prose) rewrites for the typed-fmt migration.

Two strictly separated lanes (PLAN.md §6 WS1, ASSESSMENT.md §0a):

PROVABLE (auto, value-identical) — the multiplier relocation
    cell:  ``X = fmt(f, ...,  suffix='×')``  →  ``X = fmt_multiple(f, ...)``
    prose: ``…`{python} C.X`…``              →  ``…`{python} C.X`$\\times$…``
  fmt_multiple formats the *same* magnitude; only the glyph moves from the
  string into prose. This is Regime 2: assess_equiv proves the visible prose is
  unchanged, fmt_prose_contract proves the glyph isn't duplicated.

AMBIGUOUS (never auto-touched) — emitted to an adjudication queue
    * suffix='%'  — fmt_percent needs a 0-1 *ratio*; whether the source value is
      a ratio or already-scaled 0-100 is NOT statically knowable. Human decides
      the source scale + MIT style (body 'percent' vs table '%').
    * suffix in {K,M,B,T,…} — fmt_count needs raw value + scale; the division
      may already be in the expression. Human decides.
    * multi-line multiplier calls — surgical single-line replace can't prove them.

DEFAULT IS DRY-RUN. Nothing is written without ``--write``. After ``--write``,
run the gauntlet: assess_equiv (value+visible-prose diff) then
fmt_prose_contract, then the render checks (MIGRATION.md Phase 3).

Usage::

    # preview the provable multiplier rewrites for one chapter
    python3 codemod_fmt.py multiple book/quarto/contents/vol1/training/training.qmd

    # apply them (cell + prose), then verify with assess_equiv
    python3 codemod_fmt.py multiple <qmd> --write

    # emit the human-adjudication queue (percent / scale / multi-line)
    python3 codemod_fmt.py queue --root book/quarto/contents --out /tmp/fmt_queue.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fmt_usage import extract_python_cells  # noqa: E402

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")

MULT_GLYPHS = {"×", "x", " ×", "× "}
PERCENT_GLYPHS = {"%", " %", "%%"}
SCALE_GLYPHS = {"K", "M", "B", "T", "k", "G"}
REWRITABLE_FNS = {"fmt", "fmt_int"}

_TIMES_AFTER = re.compile(r"^\s*(\$\\times\$|\\times|×)")


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _assign_target(node: ast.Assign):
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            return tgt.id
        if isinstance(tgt, ast.Attribute):
            return tgt.attr
    return None


@dataclass
class MultEdit:
    line: int          # 1-based file line of the call
    var: str
    old: str           # exact call source segment
    new: str           # rewritten call


@dataclass
class QueueItem:
    file: str
    line: int
    var: str | None
    fn: str
    suffix: str
    kind: str          # 'percent' | 'scale' | 'multiline-multiplier' | 'other'
    call: str
    action: str        # the human decision needed


def _rewrite_fmt_int_multiplier(call, src: str) -> str | None:
    """fmt_int(arg, …, suffix='×') → fmt_multiple(round(arg), precision=0, commas=…).

    fmt_int is documented as fmt(round(val), precision=0, …) with commas=True,
    so this is provably value-equivalent (only the glyph moves to prose).
    """
    if not call.args:
        return None
    arg0 = ast.get_source_segment(src, call.args[0])
    if arg0 is None or "\n" in arg0:
        return None
    if "prefix=" in (ast.get_source_segment(src, call) or ""):
        return None
    commas = "True"  # fmt_int default
    for kw in call.keywords:
        if kw.arg == "commas":
            v = _const_str(kw.value)
            commas = "False" if (isinstance(kw.value, ast.Constant) and kw.value.value is False) else \
                     ("True" if (isinstance(kw.value, ast.Constant) and kw.value.value is True) else commas)
    inner = arg0 if arg0.strip().startswith("round(") else f"round({arg0})"
    return f"fmt_multiple({inner}, precision=0, commas={commas})"


def _rewrite_call_to_multiple(seg: str) -> str | None:
    """Textual rewrite of a single-line ``fmt(...×)`` call → ``fmt_multiple(...)``.

    Only plain ``fmt`` is auto-rewritten: ``fmt`` and ``fmt_multiple`` share the
    exact same precision guard, so dropping ``suffix='×'`` and renaming preserves
    the rendered magnitude byte-for-byte. ``fmt_int`` is deliberately NOT handled
    here — it *rounds* its argument (5.032→"5") whereas ``fmt_multiple`` guards
    against silent rounding, so the conversion is not provably value-identical
    and is routed to the human queue instead.
    """
    if "\n" in seg:
        return None
    if not seg.lstrip().startswith("fmt("):
        return None
    if "prefix=" in seg:
        return None  # fmt_multiple has no prefix= — not a clean rewrite
    new = re.sub(r"^fmt\(", "fmt_multiple(", seg, count=1)
    # drop the suffix kwarg (always comma-preceded; positional factor comes first)
    new = re.sub(r",\s*suffix\s*=\s*(?:'[^']*'|\"[^\"]*\")", "", new, count=1)
    # fmt defaults commas=True but fmt_multiple defaults commas=False; if the
    # original relied on the default, pin commas=True so grouping is preserved.
    if "commas=" not in new and new.endswith(")"):
        new = new[:-1] + ", commas=True)"
    return new


def scan_file(path: Path, variants: bool = False):
    """Return (mult_edits, mult_vars, queue_items) for one chapter.

    variants=False (default): only the provable, value-identical lane —
      fmt(…suffix='×') -> fmt_multiple (exact glyph). Everything else queued.
    variants=True: ALSO emit edits for the intended-normalization multiplier
      forms — fmt(…suffix='x'|'X'|' ×') and fmt_int(…×) — which standardize the
      glyph to × and relocate it to prose. These change rendered output (x->×,
      rounding made explicit) and MUST be verified with the transformation-aware
      gate (run_multiplier_lane --variants), not the byte-identical gate.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    mult_edits: list[MultEdit] = []
    mult_vars: set[str] = set()
    queue: list[QueueItem] = []

    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # map each line to its innermost enclosing class so exports are tracked
        # as Class.attr (prose refs them that way) — avoids the name-collision
        # hazard where two cells export the same bare attr but only one is rewritten.
        classes = [(n.name, n.lineno, n.end_lineno or n.lineno)
                   for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        def _qualifier(lineno: int) -> str:
            best = None
            for name, s, e in classes:
                if s <= lineno <= e and (best is None or s > best[1]):
                    best = (name, s, e)
            return best[0] + "." if best else ""

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in REWRITABLE_FNS:
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            sfx = suffix.strip()

            if suffix == "×":
                # exact glyph — provable lane for fmt; fmt_int handled in variants
                new = _rewrite_call_to_multiple(seg)
                if not new and variants and var and fname == "fmt_int":
                    new = _rewrite_fmt_int_multiplier(call, src)
                if new and var:
                    mult_edits.append(MultEdit(file_line, var, seg, new))
                    mult_vars.add(_qualifier(node.lineno) + var)
                elif fname == "fmt_int":
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "fmt_int-multiplier", seg.replace("\n", " ⏎ "),
                        "fmt_int rounds; pick fmt_multiple(round(x), precision=0) if "
                        "integer display is intended, else fmt_multiple(x, precision=N). "
                        "Then add $\\times$ in prose."))
                else:
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "multiline-multiplier", seg.replace("\n", " ⏎ "),
                        "Convert by hand to fmt_multiple(...) + add $\\times$ in prose."))
            elif sfx in {"×", "x", "X"}:
                # space-padded glyph (' ×') or a LITERAL letter x, or an fmt_int
                # multiplier. In variants mode these are converted (intended
                # normalization, transformation-gated); otherwise queued.
                new = None
                if variants and var:
                    if fname == "fmt":
                        new = _rewrite_call_to_multiple(seg)
                    elif fname == "fmt_int":
                        new = _rewrite_fmt_int_multiplier(call, src)
                if new and var:
                    mult_edits.append(MultEdit(file_line, var, seg, new))
                    mult_vars.add(_qualifier(node.lineno) + var)
                else:
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "multiplier-variant", seg.replace("\n", " ⏎ "),
                        f"suffix={suffix!r}: confirm it is a multiplier, then "
                        "fmt_multiple(...) + $\\times$ in prose (drop the literal x/space)."))
            elif sfx in {g.strip() for g in PERCENT_GLYPHS}:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "percent", seg.replace("\n", " ⏎ "),
                    "Decide source scale (ratio vs 0-100) → fmt_percent(ratio, "
                    "style=symbol|prose by context). Body=prose, table/eq=symbol."))
            elif sfx in SCALE_GLYPHS:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "scale", seg.replace("\n", " ⏎ "),
                    "Move scale division into fmt_count(raw, scale=…) or keep as unit."))

    return mult_edits, mult_vars, queue


def _ref_is_mult(ref: str, mult_vars: set[str]) -> bool:
    """True if a prose ref names a rewritten multiplier export.

    mult_vars holds qualified names ('Class.attr' for class exports, bare
    'attr' for module-level). A prose ref 'Class.attr' must match the qualified
    entry exactly; a bare ref matches a bare entry. This prevents patching a
    same-named export from a *different* class that was not rewritten.
    """
    if ref in mult_vars:
        return True
    # module-level export referenced bare
    return "." not in ref and ref in mult_vars


def _patch_prose(text: str, mult_vars: set[str]) -> tuple[str, list[tuple[int, str]]]:
    """Append $\\times$ after refs to multiplier vars that lack it. Returns
    (new_text, [(line, ref)])."""
    if not mult_vars:
        return text, []
    out_lines: list[str] = []
    patches: list[tuple[int, str]] = []
    in_cell = False
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"^```\{python\}", line):
            in_cell = True
            out_lines.append(line)
            continue
        if in_cell and re.match(r"^```\s*$", line):
            in_cell = False
            out_lines.append(line)
            continue
        if in_cell:
            out_lines.append(line)
            continue

        # collect insertion points (end offsets) for qualifying refs, then splice
        # right-to-left so repeated identical refs on one line each get exactly
        # one $\times$ (str.replace would collide on identical tokens).
        inserts: list[tuple[int, str]] = []
        for m in INLINE_PY.finditer(line):
            ref = m.group(1)
            if not _ref_is_mult(ref, mult_vars):
                continue
            if _TIMES_AFTER.match(line[m.end():]):
                continue  # already followed by the glyph
            inserts.append((m.end(), ref))
        new_line = line
        for pos, ref in sorted(inserts, key=lambda t: -t[0]):
            new_line = new_line[:pos] + "$\\times$" + new_line[pos:]
            patches.append((i, ref))
        out_lines.append(new_line)
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_nl, patches


def _apply_cell_edits(text: str, edits: list[MultEdit]) -> str:
    lines = text.splitlines()
    for e in edits:
        idx = e.line - 1
        if 0 <= idx < len(lines) and e.old in lines[idx]:
            lines[idx] = lines[idx].replace(e.old, e.new, 1)
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing_nl


def cmd_multiple(args) -> int:
    files = [Path(p) for p in args.qmd]
    total_cell = total_prose = 0
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        edits, mult_vars, _ = scan_file(path)
        new_text = _apply_cell_edits(text, edits)
        new_text, prose_patches = _patch_prose(new_text, mult_vars)
        rel = str(path).split("contents/")[-1]
        if not edits and not prose_patches:
            continue
        print(f"\n=== {rel} ===")
        for e in edits:
            print(f"  cell  L{e.line}  {e.var}")
            print(f"    -  {e.old}")
            print(f"    +  {e.new}")
        for ln, ref in prose_patches:
            print(f"  prose L{ln}  +$\\times$ after `{{python}} {ref}`")
        total_cell += len(edits)
        total_prose += len(prose_patches)
        if args.write:
            path.write_text(new_text, encoding="utf-8")
    verb = "APPLIED" if args.write else "DRY-RUN (use --write to apply)"
    print(f"\n{verb}: {total_cell} cell rewrites, {total_prose} prose $\\times$ insertions.")
    if not args.write and (total_cell or total_prose):
        print("After --write, verify: assess_equiv (values change, visible prose IDENTICAL) "
              "+ fmt_prose_contract (clean).")
    return 0


def cmd_queue(args) -> int:
    files = [Path(p) for p in args.qmd]
    if args.root:
        files += sorted(Path(args.root).rglob("*.qmd"))
    items: list[QueueItem] = []
    for path in files:
        if path.is_file():
            _, _, q = scan_file(path)
            items.extend(q)
    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    payload = {"summary": by_kind, "count": len(items), "items": [asdict(i) for i in items]}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[queue] {len(items)} item(s) -> {args.out}")
    print(f"by kind: {by_kind}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("multiple", help="provable multiplier relocation (cell+prose)")
    m.add_argument("qmd", nargs="+")
    m.add_argument("--write", action="store_true", help="apply edits (default: dry-run)")
    m.set_defaults(func=cmd_multiple)

    q = sub.add_parser("queue", help="emit adjudication queue (percent/scale/multiline)")
    q.add_argument("qmd", nargs="*")
    q.add_argument("--root")
    q.add_argument("--out")
    q.set_defaults(func=cmd_queue)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
