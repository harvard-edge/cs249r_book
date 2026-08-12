#!/usr/bin/env python3
"""Design B codemod for fmt_multiple exports and inline refs.

Reads the frozen inventory from ``inventory_design_b_rates.py`` and performs
only the mechanical part of Design B:

* ``foo_str = fmt_multiple(...)`` -> ``foo_mult_str = fmt_multiple(...)``
* ``foo_multiplier_str`` / ``foo_multiple_str`` -> ``foo_mult_str``
* ``{python} Class.foo_mult_str`$\times$`` -> ``{python} Class.foo_mult_str``

It does not infer new multiplier sites. Run inventory and manual review first.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TIMES_AFTER_REF = r"(?:\$\\times\$|\\times|×|x\b)"


def has_mult_token(name: str) -> bool:
    """Return True when the canonical ``mult`` token appears before ``_str``."""
    if not name.endswith("_str"):
        return False
    return "mult" in name[:-4].split("_")


def multiplier_name(name: str) -> str:
    """Return the Design B export name for a fmt_multiple output."""
    if not name.endswith("_str"):
        return name
    if has_mult_token(name):
        return name
    base = name[:-4]
    for suffix in ("_mult", "_multiplier", "_multiple"):
        if base.endswith(suffix):
            return f"{base[: -len(suffix)]}_mult_str"
    if base.endswith("_x"):
        return f"{base[:-2]}_mult_str"
    return f"{base}_mult_str"


def _load_inventory(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_word(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    return pattern.subn(new, text)


def _strip_times_after_refs(text: str, ref_map: dict[str, str]) -> tuple[str, int]:
    count = 0
    for old_ref, new_ref in ref_map.items():
        pattern = re.compile(
            rf"(`\{{python\}}\s+{re.escape(new_ref)}`)\s*{TIMES_AFTER_REF}"
        )
        text, n = pattern.subn(r"\1", text)
        count += n
    return text, count


def transform_file(path: Path, exports: list[dict]) -> tuple[str, dict]:
    original = path.read_text(encoding="utf-8")
    text = original
    replacements: dict[str, str] = {}
    ref_map: dict[str, str] = {}
    unique_names = {row["name"] for row in exports}
    rows_requiring_rename = 0

    for row in exports:
        old_name = row["name"]
        new_name = multiplier_name(old_name)
        if old_name == new_name:
            continue
        rows_requiring_rename += 1
        replacements[old_name] = new_name
        old_ref = row["qualified"]
        if "." in old_ref:
            cls, _ = old_ref.rsplit(".", 1)
            ref_map[old_ref] = f"{cls}.{new_name}"

    renamed = 0
    for old_name, new_name in sorted(replacements.items(), key=lambda x: -len(x[0])):
        text, n = _replace_word(text, old_name, new_name)
        renamed += n

    text, stripped = _strip_times_after_refs(text, ref_map)

    stats = {
        "changed": text != original,
        "export_rows": len(exports),
        "export_rows_requiring_rename": rows_requiring_rename,
        "unique_export_names": len(unique_names),
        "already_semantic_export_names": len(unique_names) - len(replacements),
        "duplicate_export_rows": len(exports) - len(unique_names),
        "export_names_rewritten": len(replacements),
        "name_replacements": renamed,
        "times_stripped": stripped,
    }
    return text, stats


def run(inventory: Path, *, write: bool) -> int:
    payload = _load_inventory(inventory)
    by_file: dict[Path, list[dict]] = defaultdict(list)
    for row in payload["mult_exports"]:
        by_file[Path(row["file"])].append(row)

    totals = {
        "files_seen": len(by_file),
        "files_changed": 0,
        "export_rows": 0,
        "export_rows_requiring_rename": 0,
        "unique_export_names": 0,
        "already_semantic_export_names": 0,
        "duplicate_export_rows": 0,
        "export_names_rewritten": 0,
        "name_replacements": 0,
        "times_stripped": 0,
    }
    changed_files: list[str] = []
    for path, exports in sorted(by_file.items()):
        new_text, stats = transform_file(path, exports)
        if stats["changed"]:
            totals["files_changed"] += 1
            changed_files.append(str(path))
            if write:
                path.write_text(new_text, encoding="utf-8")
        for key in (
            "export_rows",
            "export_rows_requiring_rename",
            "unique_export_names",
            "already_semantic_export_names",
            "duplicate_export_rows",
            "export_names_rewritten",
            "name_replacements",
            "times_stripped",
        ):
            totals[key] += stats[key]

    mode = "write" if write else "dry-run"
    print(json.dumps({"mode": mode, **totals}, indent=2, sort_keys=True))
    for path in changed_files[:40]:
        print(path)
    if len(changed_files) > 40:
        print(f"... {len(changed_files) - 40} more")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("book/tools/audit/artifacts/fmt_design_b_inventory.json"),
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    return run(args.inventory, write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
