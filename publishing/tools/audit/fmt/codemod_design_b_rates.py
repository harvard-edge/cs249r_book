#!/usr/bin/env python3
"""Normalize fixed-unit rate/FLOP output names and refs.

This is the Design-B companion for output-name semantics. It is intentionally
unit-driven, not suffix-driven: the displayed unit is inferred from the
formatter call, then the export name is updated so the token before ``_str``
matches what the reader sees.

Examples:

* ``*_gb_s_str`` / ``*_gbs_str`` -> ``*_gb_per_s_str`` for ``GB/s``.
* ``*_gbps_str`` -> ``*_gbit_per_s_str`` for ``Gb/s``; but if the call
  actually formats ``GB/second``, it becomes ``*_gb_per_s_str``.
* ``*_tflops_str`` -> ``*_tflop_per_s_str`` for ``TFLOP/s``.
* ``*_tflops_str`` -> ``*_tflop_str`` for total FLOP work (``TFLOP``).

The script rewrites the full QMD file text so cell header comments and prose
refs stay in sync. It refuses per-file name conflicts.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fmt_usage import extract_python_cells  # noqa: E402

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")

RATE_SUFFIX_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9])gbps_s(?![A-Za-z0-9])"), "gbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mbps_s(?![A-Za-z0-9])"), "mbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gb_s(?![A-Za-z0-9])"), "gb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gbs(?![A-Za-z0-9])"), "gb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tb_s(?![A-Za-z0-9])"), "tb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tbs(?![A-Za-z0-9])"), "tb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mb_s(?![A-Za-z0-9])"), "mb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mbs(?![A-Za-z0-9])"), "mb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])kb_s(?![A-Za-z0-9])"), "kb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])kbs(?![A-Za-z0-9])"), "kb_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gbps(?![A-Za-z0-9])"), "gbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mbps(?![A-Za-z0-9])"), "mbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gbit_s(?![A-Za-z0-9])"), "gbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mbit_s(?![A-Za-z0-9])"), "mbit_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])km_s(?![A-Za-z0-9])"), "km_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])kms_s(?![A-Za-z0-9])"), "km_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gw_s(?![A-Za-z0-9])"), "gw_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gws_s(?![A-Za-z0-9])"), "gw_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tokens_s(?![A-Za-z0-9])"), "tokens_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tokens_per_sec(?![A-Za-z0-9])"), "tokens_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])req_s(?![A-Za-z0-9])"), "req_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])requests_s(?![A-Za-z0-9])"), "requests_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])img_per_sec(?![A-Za-z0-9])"), "img_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])img_sec(?![A-Za-z0-9])"), "img_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])images_per_sec(?![A-Za-z0-9])"), "images_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tflop_s(?![A-Za-z0-9])"), "tflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])tflops(?![A-Za-z0-9])"), "tflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])gflops(?![A-Za-z0-9])"), "gflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])mflops(?![A-Za-z0-9])"), "mflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])pflops(?![A-Za-z0-9])"), "pflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])eflops(?![A-Za-z0-9])"), "eflop_per_s"),
    (re.compile(r"(?<![A-Za-z0-9])zflops(?![A-Za-z0-9])"), "zflop_per_s"),
)

FLOP_WORK_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9])mflops(?![A-Za-z0-9])"), "mflop"),
    (re.compile(r"(?<![A-Za-z0-9])gflops(?![A-Za-z0-9])"), "gflop"),
    (re.compile(r"(?<![A-Za-z0-9])tflops(?![A-Za-z0-9])"), "tflop"),
    (re.compile(r"(?<![A-Za-z0-9])pflops(?![A-Za-z0-9])"), "pflop"),
    (re.compile(r"(?<![A-Za-z0-9])eflops(?![A-Za-z0-9])"), "eflop"),
    (re.compile(r"(?<![A-Za-z0-9])zflops(?![A-Za-z0-9])"), "zflop"),
)

TARGET_PATTERNS = {
    "gb_per_s": re.compile(r"(?:^|_)gb_per_s(?:_|$)"),
    "tb_per_s": re.compile(r"(?:^|_)tb_per_s(?:_|$)"),
    "mb_per_s": re.compile(r"(?:^|_)mb_per_s(?:_|$)"),
    "kb_per_s": re.compile(r"(?:^|_)kb_per_s(?:_|$)"),
    "gbit_per_s": re.compile(r"(?:^|_)gbit_per_s(?:_|$)"),
    "mbit_per_s": re.compile(r"(?:^|_)mbit_per_s(?:_|$)"),
    "km_per_s": re.compile(r"(?:^|_)km_per_s(?:_|$)"),
    "gw_per_s": re.compile(r"(?:^|_)gw_per_s(?:_|$)"),
    "tokens_per_s": re.compile(r"(?:^|_)tokens_per_s(?:_|$)"),
    "tokens_per_hour": re.compile(r"(?:^|_)tokens_per_hour(?:_|$)"),
    "req_per_s": re.compile(r"(?:^|_)(?:req_per_s|rps)(?:_|$)"),
    "requests_per_s": re.compile(r"(?:^|_)(?:requests_per_s|rps|qps)(?:_|$)"),
    "img_per_s": re.compile(r"(?:^|_)img_per_s(?:_|$)"),
    "images_per_s": re.compile(r"(?:^|_)images_per_s(?:_|$)"),
    "tflop_per_s": re.compile(r"(?:^|_)tflop_per_s(?:_|$)"),
    "gflop_per_s": re.compile(r"(?:^|_)gflop_per_s(?:_|$)"),
    "mflop_per_s": re.compile(r"(?:^|_)mflop_per_s(?:_|$)"),
    "pflop_per_s": re.compile(r"(?:^|_)pflop_per_s(?:_|$)"),
    "eflop_per_s": re.compile(r"(?:^|_)eflop_per_s(?:_|$)"),
    "zflop_per_s": re.compile(r"(?:^|_)zflop_per_s(?:_|$)"),
    "mflop": re.compile(r"(?:^|_)mflop(?:_|$)"),
    "gflop": re.compile(r"(?:^|_)gflop(?:_|$)"),
    "tflop": re.compile(r"(?:^|_)tflop(?:_|$)"),
    "pflop": re.compile(r"(?:^|_)pflop(?:_|$)"),
    "eflop": re.compile(r"(?:^|_)eflop(?:_|$)"),
    "zflop": re.compile(r"(?:^|_)zflop(?:_|$)"),
}


@dataclass(frozen=True)
class Rename:
    file: str
    line: int
    qualified: str
    old: str
    new: str
    formatter: str
    target: str
    unit_source: str
    context: str


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    out: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Attribute):
            out.append(target.attr)
    return out


def _classes(tree: ast.AST) -> list[tuple[str, int, int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return out


def _qualifier(classes: list[tuple[str, int, int]], lineno: int) -> str:
    best = None
    for name, start, end in classes:
        if start <= lineno <= end and (best is None or start > best[1]):
            best = (name, start, end)
    return f"{best[0]}." if best else ""


def _node_src(src: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.get_source_segment(src, node) or ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _unit_source(fname: str, call: ast.Call, src: str) -> str:
    if fname in {"fmt_bandwidth", "fmt_flop_rate", "fmt_ops_rate", "fmt_flops"}:
        kw = _keyword(call, "unit")
        return _node_src(src, kw) if kw is not None else ""
    if fname in {"fmt_qty", "fmt_qty_int"}:
        kw = _keyword(call, "unit")
        if kw is not None:
            return _node_src(src, kw)
        if len(call.args) >= 2:
            return _node_src(src, call.args[1])
        return ""
    if fname == "fmt_rate":
        if len(call.args) >= 2:
            lit = _literal_str(call.args[1])
            return lit if lit is not None else _node_src(src, call.args[1])
        kw = _keyword(call, "unit")
        lit = _literal_str(kw)
        return lit if lit is not None else _node_src(src, kw)
    if fname == "MarkdownStr" and call.args:
        lit = _literal_str(call.args[0])
        return lit if lit is not None else _node_src(src, call.args[0])
    return ""


def _compact_unit(unit: str) -> str:
    return (
        unit.strip()
        .replace(" ", "")
        .replace("ureg.", "")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace('"', "")
    )


def _target_from_unit(fname: str, unit_source: str, old_name: str) -> str | None:
    compact = _compact_unit(unit_source)
    lower = compact.lower()

    if fname == "fmt_rate":
        unit = unit_source.strip().strip("'\"")
        if unit == "tokens/s":
            return "tokens_per_s"
        if unit == "tokens/hour":
            return "tokens_per_hour"
        if unit == "req/s":
            return "req_per_s"
        if unit == "requests/s":
            return "requests_per_s"
        if unit == "img/s":
            return "img_per_s"
        if unit == "images/s":
            return "images_per_s"
        return None

    # MarkdownStr literals are handled by visible text snippets.
    if "tflop/s" in lower:
        return "tflop_per_s"
    if "gflop/s" in lower:
        return "gflop_per_s"
    if "mflop/s" in lower:
        return "mflop_per_s"
    if "pflop/s" in lower:
        return "pflop_per_s"
    if "eflop/s" in lower:
        return "eflop_per_s"
    if "zflop/s" in lower:
        return "zflop_per_s"
    if "gb/s" in lower:
        return "gb_per_s"
    if "tb/s" in lower:
        return "tb_per_s"
    if "mb/s" in lower:
        return "mb_per_s"
    if "kb/s" in lower:
        return "kb_per_s"

    if re.search(r"(?:^|[^A-Za-z])TB/?second", compact):
        return "tb_per_s"
    if re.search(r"(?:^|[^A-Za-z])GB/?second", compact):
        return "gb_per_s"
    if re.search(r"(?:^|[^A-Za-z])MB/?second", compact):
        return "mb_per_s"
    if re.search(r"(?:^|[^A-Za-z])KB/?second", compact):
        return "kb_per_s"
    if "Mbps" in compact or "megabit/second" in lower:
        return "mbit_per_s"
    if "Gbps" in compact or "gigabit/second" in lower:
        return "gbit_per_s"
    if "kilometer/second" in lower or "km/second" in lower:
        return "km_per_s"
    if re.search(r"(?:^|[^A-Za-z])GW/?second", compact):
        return "gw_per_s"

    # FLOP throughput versus FLOP work. fmt_flop_rate is a rate even if the
    # unit source is omitted; fmt_flops is work even when old names say flops.
    if "TFLOP/second" in compact or "TFLOPs/second" in compact:
        return "tflop_per_s"
    if "GFLOP/second" in compact or "GFLOPs/second" in compact:
        return "gflop_per_s"
    if "MFLOP/second" in compact or "MFLOPs/second" in compact:
        return "mflop_per_s"
    if "PFLOP/second" in compact or "PFLOPs/second" in compact:
        return "pflop_per_s"
    if "EFLOP/second" in compact or "EFLOPs/second" in compact:
        return "eflop_per_s"
    if "ZFLOP/second" in compact or "ZFLOPs/second" in compact:
        return "zflop_per_s"

    if fname == "fmt_flop_rate":
        if "tflop" in old_name:
            return "tflop_per_s"
        if "gflop" in old_name:
            return "gflop_per_s"
        if "mflop" in old_name:
            return "mflop_per_s"
        if "pflop" in old_name:
            return "pflop_per_s"
        if "eflop" in old_name:
            return "eflop_per_s"
        if "zflop" in old_name:
            return "zflop_per_s"

    if "TFLOPs" in compact or "TFLOP" in compact:
        return "tflop"
    if "GFLOPs" in compact or "GFLOP" in compact:
        return "gflop"
    if "MFLOPs" in compact or "MFLOP" in compact:
        return "mflop"
    if "PFLOPs" in compact or "PFLOP" in compact:
        return "pflop"
    if "EFLOPs" in compact or "EFLOP" in compact:
        return "eflop"
    if "ZFLOPs" in compact or "ZFLOP" in compact:
        return "zflop"

    return None


def _contains_target(name: str, target: str) -> bool:
    pattern = TARGET_PATTERNS.get(target)
    return bool(pattern and pattern.search(name[:-4] if name.endswith("_str") else name))


def _replace_tokens(base: str, target: str) -> str:
    replacements = FLOP_WORK_REPLACEMENTS if target in {
        "mflop", "gflop", "tflop", "pflop", "eflop", "zflop",
    } else RATE_SUFFIX_REPLACEMENTS
    out = base
    for pattern, replacement in replacements:
        out = pattern.sub(replacement, out)

    if target in {"gb_per_s", "tb_per_s", "mb_per_s", "kb_per_s", "gbit_per_s", "mbit_per_s"}:
        unit_prefix = target.removesuffix("_per_s")
        out = re.sub(rf"_{unit_prefix}$", f"_{target}", out)
        out = re.sub(r"_bw_s$", f"_bw_{target}", out)
    if target == "tokens_per_s":
        out = re.sub(r"_toks$", "_tokens_per_s", out)

    # Older work quantities sometimes used a trailing scale token instead of a
    # unit token, e.g. ``flops_m_str`` for a value rendered as ``MFLOP``.
    # Replace that terminal scale token with the actual rendered unit.
    flop_scale = {
        "mflop": "m",
        "gflop": "g",
        "tflop": "t",
        "pflop": "p",
        "eflop": "e",
        "zflop": "z",
    }.get(target)
    if flop_scale:
        out = re.sub(rf"_{flop_scale}$", f"_{target}", out)

    # If the only visible unit token was ambiguous (for example ``gbps``) and
    # the formatter proved it renders a different unit, force the terminal token
    # to the inferred target.
    if not TARGET_PATTERNS.get(target, re.compile(r"$^")).search(out):
        ambiguous_tail = re.compile(
            r"(gbit_per_s|mbit_per_s|gb_per_s|tb_per_s|mb_per_s|kb_per_s|"
            r"tflop_per_s|gflop_per_s|mflop_per_s|pflop_per_s|eflop_per_s|zflop_per_s|"
            r"tflop|gflop|mflop|pflop|eflop|zflop)$"
        )
        out = ambiguous_tail.sub(target, out)
    if not TARGET_PATTERNS.get(target, re.compile(r"$^")).search(out):
        out = f"{out}_{target}"

    # Collapse mechanical duplicates from names such as
    # ``prefill_tokens_per_s_tokens_s_str``.
    out = re.sub(r"(tokens_per_s)_\1", r"\1", out)
    out = re.sub(r"tokens_tokens_per_s", "tokens_per_s", out)
    out = re.sub(r"(req_per_s)_\1", r"\1", out)
    out = re.sub(r"(gb_per_s)_\1", r"\1", out)
    out = re.sub(r"(mflop|gflop|tflop|pflop|eflop|zflop)_\1", r"\1", out)
    return out


def _move_target_to_suffix(base: str, target: str) -> str:
    """Move context qualifiers before the rendered unit token.

    ``a100_tflop_per_s_fp16`` renders a TFLOP/s string, so the clearer export is
    ``a100_fp16_tflop_per_s``. Keep ``gflop_per_inference`` in place because the
    formatter's ``per="inference"`` text is part of the rendered unit.
    """
    if base == target or base.endswith(f"_{target}"):
        return base

    if base.startswith(f"{target}_"):
        prefix = ""
        trailing = base[len(target) + 1:]
    else:
        marker = f"_{target}_"
        if marker not in base:
            return base
        prefix, trailing = base.split(marker, 1)

    if target in {"mflop", "gflop", "tflop", "pflop", "eflop", "zflop"} and trailing.startswith("per_"):
        return base

    parts = [part for part in (prefix, trailing, target) if part]
    return "_".join(parts)


def canonical_name(name: str, target: str) -> str:
    if not name.endswith("_str"):
        return name
    base = name[:-4]
    rewritten = _move_target_to_suffix(_replace_tokens(base, target), target)
    if _contains_target(name, target) and rewritten == base:
        return name
    return f"{rewritten}_str"


def collect_renames(root: Path) -> list[Rename]:
    out: list[Rename] = []
    for qmd in sorted(root.rglob("*.qmd")):
        text = qmd.read_text(encoding="utf-8", errors="replace")
        raw_lines = text.splitlines()
        for fence_line, src in extract_python_cells(text):
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            classes = _classes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if not isinstance(node.value, ast.Call):
                    continue
                fname = _call_name(node.value)
                if fname not in {
                    "fmt_bandwidth", "fmt_flop_rate", "fmt_flops", "fmt_qty",
                    "fmt_qty_int", "fmt_rate", "MarkdownStr",
                }:
                    continue
                unit_source = _unit_source(fname, node.value, src)
                for old in _target_names(node):
                    if not old.endswith("_str"):
                        continue
                    target = _target_from_unit(fname, unit_source, old)
                    if target is None:
                        continue
                    new = canonical_name(old, target)
                    if new == old:
                        continue
                    file_line = fence_line + (node.lineno or 0)
                    out.append(Rename(
                        file=str(qmd),
                        line=file_line,
                        qualified=_qualifier(classes, node.lineno) + old,
                        old=old,
                        new=new,
                        formatter=fname,
                        target=target,
                        unit_source=unit_source,
                        context=raw_lines[file_line - 1].strip()
                        if 0 < file_line <= len(raw_lines) else "",
                    ))
    return out


def _replace_word(text: str, old: str, new: str) -> tuple[str, int]:
    return re.subn(rf"\b{re.escape(old)}\b", new, text)


def _validate_file_map(rows: list[Rename]) -> dict[str, str]:
    by_old: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_old[row.old].add(row.new)
    conflicts = {old: sorted(new) for old, new in by_old.items() if len(new) > 1}
    if conflicts:
        details = "; ".join(f"{old}->{news}" for old, news in sorted(conflicts.items()))
        raise SystemExit(f"Refusing conflicting per-file rename map: {details}")
    return {old: next(iter(new)) for old, new in by_old.items()}


def apply_renames(renames: list[Rename], *, write: bool) -> dict:
    by_file: dict[Path, list[Rename]] = defaultdict(list)
    for row in renames:
        by_file[Path(row.file)].append(row)

    totals = Counter()
    changed_files: list[str] = []
    for path, rows in sorted(by_file.items()):
        text = path.read_text(encoding="utf-8")
        original = text
        file_map = _validate_file_map(rows)
        for old, new in sorted(file_map.items(), key=lambda item: -len(item[0])):
            text, n = _replace_word(text, old, new)
            totals["name_replacements"] += n
        if text != original:
            changed_files.append(str(path))
            totals["files_changed"] += 1
            if write:
                path.write_text(text, encoding="utf-8")
    totals["files_seen"] = len(by_file)
    totals["rename_rows"] = len(renames)
    totals["unique_names"] = len({(r.file, r.old, r.new) for r in renames})
    return {"totals": dict(totals), "changed_files": changed_files}


def _scan_refs(root: Path, old_qualified: set[str]) -> int:
    count = 0
    for qmd in sorted(root.rglob("*.qmd")):
        text = qmd.read_text(encoding="utf-8", errors="replace")
        for match in INLINE_PY.finditer(text):
            if match.group(1) in old_qualified:
                count += 1
    return count


def run(root: Path, *, write: bool, json_path: Path | None) -> int:
    renames = collect_renames(root)
    result = apply_renames(renames, write=write)
    result["mode"] = "write" if write else "dry-run"
    result["by_target"] = dict(Counter(row.target for row in renames))
    result["by_formatter"] = dict(Counter(row.formatter for row in renames))
    result["old_refs"] = _scan_refs(root, {row.qualified for row in renames})
    result["renames"] = [row.__dict__ for row in renames]
    print(json.dumps({k: v for k, v in result.items() if k != "renames"}, indent=2, sort_keys=True))
    for path in result["changed_files"][:50]:
        print(path)
    if len(result["changed_files"]) > 50:
        print(f"... {len(result['changed_files']) - 50} more")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[codemod-rates] wrote {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("book/quarto/contents"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    return run(args.root, write=args.write, json_path=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
