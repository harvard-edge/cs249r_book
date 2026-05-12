#!/usr/bin/env python3
"""Build chapter packets for semantic cross-reference validation.

The mechanical audit proves that references resolve. These packets support the
next question: whether each resolved target is editorially useful and canonical.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "book" / "quarto" / "contents").exists():
            return parent
    raise RuntimeError("Could not locate repository root containing book/quarto/contents")


ROOT = find_repo_root()


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def read_lines(path: str, cache: dict[str, list[str]]) -> list[str]:
    if path not in cache:
        cache[path] = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return cache[path]


def paragraph_at(lines: list[str], line_no: int, max_chars: int = 1200) -> str:
    idx = max(0, min(line_no - 1, len(lines) - 1))
    start = idx
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    text = " ".join(line.strip() for line in lines[start : end + 1]).strip()
    return text[:max_chars]


def target_excerpt(lines: list[str], line_no: int, max_lines: int = 18, max_chars: int = 1600) -> str:
    idx = max(0, min(line_no - 1, len(lines) - 1))
    end = min(len(lines), idx + max_lines)
    excerpt = "\n".join(lines[idx:end]).strip()
    return excerpt[:max_chars]


def packet_name(file_path: str) -> str:
    return file_path.removeprefix("book/quarto/contents/").replace("/", "__").removesuffix(".qmd") + ".semantic.yml"


def build_packets(inventory_path: Path, out_dir: Path) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    anchors_by_key = {(item["volume"], item["id"], item["file"]): item for item in inventory["anchors"]}
    anchors_by_volume = defaultdict(list)
    for item in inventory["anchors"]:
        anchors_by_volume[(item["volume"], item["id"])].append(item)
    files: dict[str, list[str]] = {}
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ref in inventory["references"]:
        if ref["status"] != "ok":
            continue
        anchor = anchors_by_key.get((ref["volume"], ref["id"], ref.get("target_file") or ""))
        if not anchor:
            candidates = anchors_by_volume.get((ref["volume"], ref["id"]), [])
            anchor = candidates[0] if len(candidates) == 1 else None
        if not anchor:
            continue
        source_lines = read_lines(ref["file"], files)
        target_lines = read_lines(anchor["file"], files)
        by_file[ref["file"]].append(
            {
                "source_line": ref["line"],
                "reference": ref["id"],
                "kind": ref["kind"],
                "source_text": ref["text"],
                "source_context": paragraph_at(source_lines, ref["line"]),
                "target_file": anchor["file"],
                "target_line": anchor["line"],
                "target_title": anchor["title"],
                "target_context": target_excerpt(target_lines, anchor["line"]),
                "validation_question": (
                    "Is this target the right same-volume canonical target for the source context, "
                    "or should the reference be kept, retargeted, removed, or localized?"
                ),
            }
        )

    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for file_path, refs in sorted(by_file.items()):
        packet = {
            "schema_version": "crossref-semantic-validation-packet/v1",
            "source_file": file_path,
            "volume": "vol1" if "/vol1/" in file_path else "vol2",
            "reference_count": len(refs),
            "instructions": [
                "Validate existing references only; do not hunt for missing references in this pass.",
                "Keep references that point to the canonical same-volume target and help the reader.",
                "Flag references that are mechanically valid but too broad, too narrow, redundant, or stylistically disruptive.",
                "Do not propose cross-volume Quarto references.",
            ],
            "references": refs,
        }
        path = packet_dir / packet_name(file_path)
        path.write_text(yaml.safe_dump(packet, sort_keys=False, allow_unicode=True), encoding="utf-8")
        summary_rows.append((file_path, len(refs), path.relative_to(ROOT)))

    summary = {
        "schema_version": "crossref-semantic-validation-summary/v1",
        "inventory": str(inventory_path.relative_to(ROOT) if inventory_path.is_relative_to(ROOT) else inventory_path),
        "packet_dir": str(packet_dir.relative_to(ROOT)),
        "source_files": len(summary_rows),
        "references": sum(count for _, count, _ in summary_rows),
        "packets": [
            {"source_file": file_path, "reference_count": count, "packet": str(path)}
            for file_path, count, path in summary_rows
        ],
    }
    (out_dir / "summary.yml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic validation packets for existing cross-references.")
    parser.add_argument(
        "--inventory",
        default="review/cross-references-round2-postedit/inventory.json",
        help="Mechanical audit inventory JSON, relative to repository root or absolute.",
    )
    parser.add_argument(
        "--out-dir",
        default="review/cross-references-semantic-validation-round1",
        help="Output directory, relative to repository root or absolute.",
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    build_packets(inventory_path, out_dir)
    print(f"Wrote semantic validation packets to {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
