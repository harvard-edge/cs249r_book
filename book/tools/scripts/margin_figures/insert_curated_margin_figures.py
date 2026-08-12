#!/usr/bin/env python3
"""Insert curated margin-figure SVG references into MLSysBook QMD files.

The source of truth is ``book/tools/audit/margin_figure_decisions.yml`` joined
with ``margin_figure_opportunities.yml``. The script is idempotent: if a QMD
already references the generated SVG filename, it leaves that candidate alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]

AUDIT_DIR = ROOT / "book/tools/audit"
OPPORTUNITIES = AUDIT_DIR / "margin_figure_opportunities.yml"
DECISIONS = AUDIT_DIR / "margin_figure_decisions.yml"


def curated_asset_name(candidate_id: str) -> str:
    return candidate_id.replace("-", "_")


def load_candidates() -> list[dict]:
    opportunities = yaml.safe_load(OPPORTUNITIES.read_text(encoding="utf-8"))["recommendations"]
    decisions = yaml.safe_load(DECISIONS.read_text(encoding="utf-8"))["decisions"]
    opp_by_id = {row["id"]: row for row in opportunities}
    rows = []
    for decision in decisions:
        if decision["decision"] not in {"must_add", "should_add", "revise_then_add"}:
            continue
        opp = opp_by_id[decision["id"]]
        rows.append({**opp, **decision, "opportunity": opp})
    return rows


def caption_from_purpose(purpose: str) -> str:
    caption = purpose.strip()
    verb_map = {
        "Show": "Shows",
        "Make": "Makes",
        "Map": "Maps",
        "Place": "Places",
        "Anchor": "Anchors",
        "Distinguish": "Distinguishes",
        "Use": "Uses",
    }
    for source, replacement in verb_map.items():
        caption = re.sub(rf"^{source}\b", replacement, caption)
    caption = caption.rstrip(".")
    words = caption.split()
    if len(words) > 18:
        caption = " ".join(words[:18]).rstrip(",;:")
    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption + "."


def alt_text(candidate: dict) -> str:
    idea = candidate.get("opportunity", {}).get("idea", "")
    purpose = candidate.get("purpose", "")
    text = f"Margin illustration showing {idea or purpose}".strip()
    text = re.sub(r"\s+", " ", text)
    return text[:240]


def block_for(candidate: dict) -> list[str]:
    asset = curated_asset_name(candidate["id"])
    caption = caption_from_purpose(candidate["purpose"])
    alt = alt_text(candidate).replace('"', "'")
    return [
        "::: {.column-margin}",
        f'![](images/svg/{asset}.svg){{width="100%" fig-alt="{alt}"}}',
        "",
        f"*{caption}*",
        ":::",
        "",
    ]


def inside_fence(lines: list[str], idx: int) -> bool:
    in_fence = False
    for line in lines[:idx]:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
    return in_fence


def skip_fence(lines: list[str], idx: int) -> int:
    if idx < len(lines) and lines[idx].lstrip().startswith("```"):
        idx += 1
    while idx < len(lines):
        if lines[idx].lstrip().startswith("```"):
            return idx + 1
        idx += 1
    return idx


def skip_div(lines: list[str], idx: int) -> int:
    if idx >= len(lines) or not lines[idx].lstrip().startswith(":::"):
        return idx
    idx += 1
    while idx < len(lines):
        if lines[idx].strip() == ":::":
            return idx + 1
        idx += 1
    return idx


def skip_table(lines: list[str], idx: int) -> int:
    while idx < len(lines) and lines[idx].lstrip().startswith("|"):
        idx += 1
    return idx


def skip_paragraph_or_list(lines: list[str], idx: int) -> int:
    if idx < len(lines) and lines[idx].lstrip().startswith("```"):
        return skip_fence(lines, idx)
    if idx < len(lines) and lines[idx].lstrip().startswith(":::"):
        return skip_div(lines, idx)
    if idx < len(lines) and lines[idx].lstrip().startswith("|"):
        return skip_table(lines, idx)
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            return idx
        if idx > 0 and stripped.startswith("#"):
            return idx
        if stripped.startswith(":::") or stripped.startswith("```"):
            return idx
        idx += 1
    return idx


def insertion_index(lines: list[str], line_no: int) -> int:
    idx = max(0, min(line_no - 1, len(lines)))
    if inside_fence(lines, idx):
        idx = skip_fence(lines, idx)
    elif idx < len(lines) and lines[idx].lstrip().startswith("```"):
        idx = skip_fence(lines, idx)

    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].lstrip().startswith("#"):
        idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

    idx = skip_paragraph_or_list(lines, idx)
    while idx < len(lines) and lines[idx].strip():
        idx = skip_paragraph_or_list(lines, idx)
    return idx


def insert_for_path(path: Path, candidates: list[dict]) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    inserted = 0
    for candidate in sorted(candidates, key=lambda row: row["line"], reverse=True):
        asset = curated_asset_name(candidate["id"])
        if f"images/svg/{asset}.svg" in text:
            continue
        idx = insertion_index(lines, candidate["line"])
        addition = block_for(candidate)
        if idx < len(lines) and lines[idx].strip() == "":
            lines[idx:idx] = [""] + addition
        else:
            lines[idx:idx] = [""] + addition
        text = "\n".join(lines) + "\n"
        inserted += 1
    if inserted:
        path.write_text(text, encoding="utf-8")
    return inserted


def main() -> None:
    by_path: dict[Path, list[dict]] = {}
    for candidate in load_candidates():
        by_path.setdefault(ROOT / candidate["path"], []).append(candidate)

    total = 0
    for path, rows in sorted(by_path.items()):
        total += insert_for_path(path, rows)
    print(f"inserted {total} curated margin figure blocks")


if __name__ == "__main__":
    main()
