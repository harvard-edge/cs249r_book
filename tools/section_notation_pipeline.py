#!/usr/bin/env python3
"""Section-by-Section Notation Inspector and Validation Pipeline.

Enables agents to review, repair, and verify mathematical notation section-by-section
without blind global regular expressions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Suspicious macros frequently over-substituted by blind regex
SUSPICIOUS_MACROS = {
    r"\PermessageLatency": "Often over-substituted for statistical significance alpha=0.05",
    r"\LinkBandwidth": "Often over-substituted for statistical power beta=0.10",
    r"\HiddenDimensionFour": "Often over-substituted for probability/proportion p",
    r"\NumberOfDevices": "Often over-substituted for sample size N or count",
    r"\ScaledResource": "Often over-substituted for count/successes X",
    r"\Loss": "Often over-substituted for lower limit L in [L, U]",
    r"\StateTransitionMatrix": "Often over-substituted for force F or general matrix F",
    r"\CoefficientOfRestitutionTwo": "Often over-substituted for upper bound 1 in [0, 1]",
    r"\CommunicationcomputationRatio": "Often over-substituted for ratio rho",
}

CORRUPTION_PATTERNS = [
    (re.compile(r"\\(?:dealt|Dealt)\\"), "Truncated dealta macro (e.g. \\dealt\\ProfileAccelerationBound)"),
    (re.compile(r"\$\$[^$]*`\{python\}"), "Inline Python inside display math ($$...`{python}`...)"),
]


@dataclass
class SectionInfo:
    index: int
    title: str
    sec_id: str
    start_line: int
    end_line: int
    level: int
    display_math_count: int
    inline_math_count: int
    duplicate_labels: list[str]
    corruptions: list[str]
    suspicious_macros: list[str]


def parse_sections(file_path: Path) -> list[SectionInfo]:
    """Parse a QMD file into sections delimited by headings."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    heading_re = re.compile(r"^(#{1,4})\s+(.*?)(?:\s+\{#([^}]+)\})?\s*$")
    label_re = re.compile(r"\{#(eq-[^}]+)\}")

    sections: list[SectionInfo] = []
    current_title = "Frontmatter / Introduction"
    current_sec_id = ""
    current_start = 1
    current_level = 1

    section_lines_map: list[tuple[str, str, int, int, int]] = []

    for i, line in enumerate(lines, 1):
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            if level <= 3:  # H1, H2, H3
                if i > current_start:
                    section_lines_map.append(
                        (current_title, current_sec_id, current_start, i - 1, current_level)
                    )
                current_title = m.group(2).strip()
                current_sec_id = m.group(3) or ""
                current_start = i
                current_level = level

    if current_start <= len(lines):
        section_lines_map.append(
            (current_title, current_sec_id, current_start, len(lines), current_level)
        )

    # Analyze each section
    for idx, (title, sec_id, start, end, level) in enumerate(section_lines_map):
        sec_lines = lines[start - 1 : end]
        sec_text = "\n".join(sec_lines)

        # Count math
        display_math = len(re.findall(r"\$\$.*?\$\$", sec_text, re.DOTALL))
        # Remove display math before counting inline math
        clean_text = re.sub(r"\$\$.*?\$\$", "", sec_text, flags=re.DOTALL)
        inline_math = len(re.findall(r"\$[^$\n]+\$", clean_text))

        # Check duplicate equation labels within the section
        labels = label_re.findall(sec_text)
        seen_labels = set()
        dup_labels = []
        for lbl in labels:
            if lbl in seen_labels and lbl not in dup_labels:
                dup_labels.append(lbl)
            seen_labels.add(lbl)

        # Check corruptions
        corruptions = []
        for pat, desc in CORRUPTION_PATTERNS:
            if pat.search(sec_text):
                corruptions.append(desc)

        # Check suspicious macros
        suspicious = []
        for macro, reason in SUSPICIOUS_MACROS.items():
            if macro in sec_text:
                suspicious.append(f"{macro} ({reason})")

        sections.append(
            SectionInfo(
                index=idx,
                title=title,
                sec_id=sec_id,
                start_line=start,
                end_line=end,
                level=level,
                display_math_count=display_math,
                inline_math_count=inline_math,
                duplicate_labels=dup_labels,
                corruptions=corruptions,
                suspicious_macros=suspicious,
            )
        )

    return sections


def list_command(file_path: Path) -> None:
    """Print an executive overview of all sections in a chapter."""
    sections = parse_sections(file_path)
    print(f"\n📂 File: {file_path}")
    print(f"Total sections: {len(sections)}\n")
    print(f"{'Idx':<4} {'Lines':<12} {'Level':<6} {'Math($$/$)':<12} {'Flags':<8} {'Title / Sec ID'}")
    print("-" * 80)

    flagged_count = 0
    for s in sections:
        flags = []
        if s.duplicate_labels:
            flags.append(f"DUP_LABEL({len(s.duplicate_labels)})")
        if s.corruptions:
            flags.append(f"CORRUPT({len(s.corruptions)})")
        if s.suspicious_macros:
            flags.append(f"SUSPICIOUS({len(s.suspicious_macros)})")

        flag_str = " ".join(flags) if flags else "OK"
        if flags:
            flagged_count += 1
            status_icon = "⚠️"
        else:
            status_icon = "🟢"

        math_str = f"{s.display_math_count}/{s.inline_math_count}"
        line_range = f"{s.start_line}-{s.end_line}"
        id_str = f" (#{s.sec_id})" if s.sec_id else ""
        print(f"{s.index:<4} {line_range:<12} H{s.level:<5} {math_str:<12} {status_icon} {s.title[:40]}{id_str}")
        if s.duplicate_labels:
            print(f"     ↳ Duplicate Labels: {', '.join(s.duplicate_labels)}")
        if s.corruptions:
            print(f"     ↳ Corruptions: {', '.join(s.corruptions)}")
        if s.suspicious_macros:
            print(f"     ↳ Suspicious Macros: {len(s.suspicious_macros)} found")

    print("-" * 80)
    print(f"Sections requiring attention: {flagged_count} / {len(sections)}\n")


def extract_command(file_path: Path, section_query: str, out_path: Path | None = None) -> None:
    """Extract a specific section by index, ID, or title match."""
    sections = parse_sections(file_path)
    target: SectionInfo | None = None

    if section_query.isdigit():
        idx = int(section_query)
        if 0 <= idx < len(sections):
            target = sections[idx]
    else:
        for s in sections:
            if section_query.lower() in s.sec_id.lower() or section_query.lower() in s.title.lower():
                target = s
                break

    if not target:
        print(f"❌ Error: Section matching '{section_query}' not found.")
        sys.exit(1)

    lines = file_path.read_text(encoding="utf-8").split("\n")
    sec_content = "\n".join(lines[target.start_line - 1 : target.end_line])

    payload = {
        "file": str(file_path),
        "index": target.index,
        "title": target.title,
        "sec_id": target.sec_id,
        "start_line": target.start_line,
        "end_line": target.end_line,
        "duplicate_labels": target.duplicate_labels,
        "corruptions": target.corruptions,
        "suspicious_macros": target.suspicious_macros,
        "content": sec_content,
    }

    if out_path:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"✅ Extracted section {target.index} ('{target.title}') to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


def apply_command(file_path: Path, patch_file: Path) -> None:
    """Apply an updated section payload back into the QMD file."""
    data = json.loads(patch_file.read_text(encoding="utf-8"))
    start_line = data["start_line"]
    end_line = data["end_line"]
    new_content = data["content"]

    lines = file_path.read_text(encoding="utf-8").split("\n")
    before = lines[: start_line - 1]
    after = lines[end_line:]

    updated = before + new_content.split("\n") + after
    file_path.write_text("\n".join(updated), encoding="utf-8")
    print(f"✅ Replaced lines {start_line}-{end_line} in {file_path}")


def verify_command(file_path: Path) -> int:
    """Run integrity checks on the file using Binder's official check suite."""
    print(f"\n🔍 Verifying notation & syntax integrity for: {file_path}")
    failed_groups = []

    # Run check groups relevant to math and notation
    for group, extra_args in [("markup", []), ("labels", ["--scope", "duplicates"]), ("math", [])]:
        cmd = ["./binder/binder", "check", group, "--path", str(file_path)] + extra_args
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ binder check {group} FAILED:")
            for line in res.stdout.split("\n"):
                if "ERROR" in line or "FAIL" in line or "source:" in line:
                    print(f"   {line.strip()}")
            failed_groups.append(group)
        else:
            print(f"🟢 binder check {group} PASSED.")

    # Also check for truncated dealta macros (\dealt\ or \Dealt\)
    content = file_path.read_text(encoding="utf-8")
    dealt_matches = re.findall(r"\\(?:dealt|Dealt)\\", content)
    if dealt_matches:
        print(f"❌ Truncated dealta macros found: {len(dealt_matches)} instance(s) of \\dealt\\ or \\Dealt\\")
        failed_groups.append("truncated_dealta")
    else:
        print("🟢 No truncated dealta macros found.")

    if not failed_groups:
        print(f"🎉 Verification SUCCESS: All checks passed for {file_path.name}!\n")
        return 0
    else:
        print(f"❌ Verification FAILED for {file_path.name}: {', '.join(failed_groups)}\n")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Section-by-section notation pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all sections and notation flags in a QMD")
    p_list.add_argument("file", type=Path, help="Path to .qmd file")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract section payload for agent inspection")
    p_extract.add_argument("file", type=Path, help="Path to .qmd file")
    p_extract.add_argument("--section", required=True, help="Section index, ID, or title")
    p_extract.add_argument("--out", type=Path, help="Output JSON file path")

    # apply
    p_apply = subparsers.add_parser("apply", help="Apply fixed section back to QMD")
    p_apply.add_argument("file", type=Path, help="Path to .qmd file")
    p_apply.add_argument("--patch", required=True, type=Path, help="JSON file with updated content")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify syntax and notation integrity")
    p_verify.add_argument("file", type=Path, help="Path to .qmd file")

    args = parser.parse_args()

    if args.cmd == "list":
        list_command(args.file)
    elif args.cmd == "extract":
        extract_command(args.file, args.section, args.out)
    elif args.cmd == "apply":
        apply_command(args.file, args.patch)
    elif args.cmd == "verify":
        sys.exit(verify_command(args.file))


if __name__ == "__main__":
    main()
