#!/usr/bin/env python3
"""Migrate raw mathematical notation in QMD files to canonical mlsysim macros.

Scans QMD files in specified volume(s), isolates math spans ($...$ and $$...$$)
outside of frontmatter and fenced code blocks, and substitutes raw LaTeX variables
with their semantic macros defined in mlsysim.Notation.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Add project root and mlsysim to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
sys.path.insert(0, str(PROJECT_ROOT))

from binder.tools.audit.checks import notation_macros


def get_rules_for_volume(volume: str) -> list[tuple[re.Pattern, str, str]]:
    """Return deduplicated rules for volume sorted by length of raw latex descending."""
    raw_rules = list(notation_macros._VOLUME_BANNED.get(volume, []))
    raw_rules.extend(notation_macros._VOLUME_BANNED.get("core", []))
    
    # Sort by length of raw_latex descending so longer patterns match first
    raw_rules.sort(key=lambda x: len(x[1]), reverse=True)
    
    seen = set()
    unique_rules = []
    for p, raw, macro in raw_rules:
        # Prefer StateTransitionMatrix over FundamentalMatrix for \mathbf{F} in vol4
        if raw == "\\mathbf{F}" and macro != "\\StateTransitionMatrix":
            continue
        if raw not in seen:
            seen.add(raw)
            unique_rules.append((p, raw, macro))
    return unique_rules


def migrate_content(content: str, rules: list[tuple[re.Pattern, str, str]]) -> tuple[str, int]:
    """Replace raw math notations inside math spans with macros."""
    placeholders: list[str] = []

    def repl_block(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"__NOTATION_MASK_{len(placeholders) - 1}__"

    # 1. Mask YAML frontmatter
    masked = re.sub(r"\A---\n.*?\n---", repl_block, content, flags=re.DOTALL)
    # 2. Mask fenced code blocks
    masked = re.sub(r"```.*?```", repl_block, masked, flags=re.DOTALL)

    repl_count = 0

    def transform_math(math_body: str) -> str:
        nonlocal repl_count
        res = math_body
        for pattern, _, macro in rules:
            if pattern.search(res):
                matches = pattern.findall(res)
                repl_count += len(matches)
                res = pattern.sub(lambda m: macro, res)
        return res

    # 3. Transform display math ($$...$$)
    masked = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + transform_math(m.group(1)) + "$$", masked, flags=re.DOTALL)

    # 4. Transform inline math ($...$)
    masked = re.sub(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)", lambda m: "$" + transform_math(m.group(1)) + "$", masked)

    # 5. Restore placeholders in reverse order
    for i in range(len(placeholders) - 1, -1, -1):
        masked = masked.replace(f"__NOTATION_MASK_{i}__", placeholders[i])

    return masked, repl_count


def process_volume(volume: str, dry_run: bool = False) -> None:
    """Migrate all QMD files in the given volume."""
    vol_dir = PROJECT_ROOT / "books" / volume
    if not vol_dir.exists():
        print(f"Directory {vol_dir} does not exist.")
        return

    rules = get_rules_for_volume(volume)
    print(f"Loaded {len(rules)} unique notation rules for {volume}")

    total_files_modified = 0
    total_replacements = 0

    qmd_files = sorted(vol_dir.glob("**/*.qmd"))
    for qmd in qmd_files:
        if qmd.name.startswith("_notation"):
            continue

        orig = qmd.read_text(encoding="utf-8")
        migrated, count = migrate_content(orig, rules)

        if count > 0 and migrated != orig:
            total_files_modified += 1
            total_replacements += count
            rel_path = qmd.relative_to(PROJECT_ROOT)
            print(f"  [{volume}] {rel_path}: {count} replacements")
            if not dry_run:
                qmd.write_text(migrated, encoding="utf-8")

    print(f"[{volume}] Done: {total_files_modified} files modified, {total_replacements} replacements made.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate raw math notations to macros.")
    parser.add_argument("--volume", choices=["vol1", "vol2", "vol3", "vol4", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    volumes = ["vol1", "vol2", "vol3", "vol4"] if args.volume == "all" else [args.volume]
    for v in volumes:
        process_volume(v, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
