#!/usr/bin/env python3
"""Inventory MLSysBook margin figures from rendered chapter source.

This scans the actual QMD ``.column-margin`` blocks. The output is meant to
answer the editorial question "what margin figure is placed where?" rather
than only restating the audit files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional metadata enrichment
    yaml = None


ROOT = Path(__file__).resolve().parents[4]
CONTENTS = ROOT / "book/quarto/contents"
AUDIT_DIR = ROOT / "book/tools/audit"
OPPORTUNITIES = AUDIT_DIR / "margin_figure_opportunities.yml"
DECISIONS = AUDIT_DIR / "margin_figure_decisions.yml"

COLUMN_START_RE = re.compile(r"^:{3,}\s*\{[^}]*\.column-margin\b")
DIV_CLOSE_RE = re.compile(r"^:{3,}\s*$")
IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((?P<src>images/(?P<kind>svg|png)/[^)]+)\)"
    r"(?:\{(?P<attrs>[^}]*)\})?"
)
ATTR_RE = re.compile(r"(?P<key>[A-Za-z0-9_-]+)\s*=\s*\"(?P<value>[^\"]*)\"")


@dataclass(frozen=True)
class MarginFigure:
    chapter: str
    qmd_path: str
    line: int
    asset: str
    asset_path: str
    exists: bool
    caption: str
    fig_alt: str
    candidate_id: str
    decision: str
    job: str
    device: str
    priority: str
    action: str


def attrs_from(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {match.group("key"): match.group("value") for match in ATTR_RE.finditer(raw)}


def clean_caption(raw: str) -> str:
    caption = raw.strip()
    if caption.startswith("*") and caption.endswith("*"):
        caption = caption.strip("*").strip()
    return re.sub(r"\s+", " ", caption)


def caption_after(block: list[tuple[int, str]], image_index: int) -> str:
    for _, line in block[image_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if IMAGE_RE.search(stripped):
            return ""
        if stripped.startswith(":::"):
            return ""
        return clean_caption(stripped)
    return ""


def iter_margin_blocks(qmd: Path) -> list[list[tuple[int, str]]]:
    blocks: list[list[tuple[int, str]]] = []
    in_margin = False
    current: list[tuple[int, str]] = []
    for line_no, line in enumerate(qmd.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not in_margin and COLUMN_START_RE.match(stripped):
            in_margin = True
            current = []
            continue
        if in_margin and DIV_CLOSE_RE.match(stripped):
            blocks.append(current)
            in_margin = False
            current = []
            continue
        if in_margin:
            current.append((line_no, line))
    return blocks


def load_metadata() -> tuple[dict[str, dict], dict[str, dict]]:
    if yaml is None:
        return {}, {}
    opportunities = yaml.safe_load(OPPORTUNITIES.read_text(encoding="utf-8")).get("recommendations", [])
    decisions = yaml.safe_load(DECISIONS.read_text(encoding="utf-8")).get("decisions", [])
    return (
        {row["id"]: row for row in opportunities if "id" in row},
        {row["id"]: row for row in decisions if "id" in row},
    )


def possible_candidate_ids(chapter: str, asset_stem: str) -> list[str]:
    slug = asset_stem.replace("_", "-")
    volume, chapter_name = chapter.split("/", 1)
    chapter_slug = chapter_name.replace("_", "-")
    candidates = [
        slug,
        f"{volume}-{chapter_slug}-{slug}",
        f"{volume}-{slug}",
        f"{chapter_slug}-{slug}",
    ]
    seen: set[str] = set()
    return [item for item in candidates if not (item in seen or seen.add(item))]


def match_metadata(chapter: str, asset_stem: str, opportunities: dict[str, dict], decisions: dict[str, dict]) -> tuple[str, dict, dict]:
    for candidate_id in possible_candidate_ids(chapter, asset_stem):
        if candidate_id in decisions or candidate_id in opportunities:
            return candidate_id, opportunities.get(candidate_id, {}), decisions.get(candidate_id, {})
    return "", {}, {}


def chapter_matches(chapter: str, filters: list[str]) -> bool:
    if not filters:
        return True
    short = chapter.split("/", 1)[-1]
    return any(chapter == item or short == item for item in filters)


def collect(volume: str, chapters: list[str], include_png: bool) -> list[MarginFigure]:
    opportunities, decisions = load_metadata()
    rows: list[MarginFigure] = []
    for qmd in sorted(CONTENTS.glob("vol*/*/*.qmd")):
        chapter = qmd.parent.relative_to(CONTENTS).as_posix()
        if volume != "all" and not chapter.startswith(f"{volume}/"):
            continue
        if not chapter_matches(chapter, chapters):
            continue

        for block in iter_margin_blocks(qmd):
            for idx, (line_no, line) in enumerate(block):
                match = IMAGE_RE.search(line.strip())
                if not match:
                    continue
                if match.group("kind") == "png" and not include_png:
                    continue
                src = match.group("src")
                asset_path = qmd.parent / src
                attrs = attrs_from(match.group("attrs"))
                candidate_id, opportunity, decision = match_metadata(
                    chapter, asset_path.stem, opportunities, decisions
                )
                rows.append(
                    MarginFigure(
                        chapter=chapter,
                        qmd_path=str(qmd.relative_to(ROOT)),
                        line=line_no,
                        asset=asset_path.name,
                        asset_path=str(asset_path.relative_to(ROOT)),
                        exists=asset_path.exists(),
                        caption=caption_after(block, idx),
                        fig_alt=attrs.get("fig-alt", ""),
                        candidate_id=candidate_id,
                        decision=decision.get("decision", ""),
                        job=decision.get("job", ""),
                        device=decision.get("device", opportunity.get("device", "")),
                        priority=opportunity.get("priority", ""),
                        action=opportunity.get("action", ""),
                    )
                )
    return rows


def write_json(rows: list[MarginFigure], out) -> None:
    json.dump([asdict(row) for row in rows], out, indent=2)
    out.write("\n")


def write_csv(rows: list[MarginFigure], out) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(MarginFigure.__dataclass_fields__)
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown(rows: list[MarginFigure], out) -> None:
    out.write("| Chapter | Line | Asset | Caption | Decision | Device | Exists |\n")
    out.write("| --- | ---: | --- | --- | --- | --- | --- |\n")
    for row in rows:
        line_ref = f"{row.qmd_path}:{row.line}"
        decision = row.decision or "untracked"
        device = row.device or "-"
        exists = "yes" if row.exists else "missing"
        out.write(
            f"| {markdown_cell(row.chapter)} | {line_ref} | {markdown_cell(row.asset)} | "
            f"{markdown_cell(row.caption)} | {markdown_cell(decision)} | "
            f"{markdown_cell(device)} | {exists} |\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", choices=("all", "vol1", "vol2"), default="all")
    parser.add_argument("--chapter", action="append", default=[], help="Filter to a chapter, e.g. vol2/inference or inference.")
    parser.add_argument("--include-png", action="store_true", help="Include non-SVG margin images such as chapter covers.")
    parser.add_argument("--missing-only", action="store_true", help="Only report references whose asset file is missing.")
    parser.add_argument("--untracked-only", action="store_true", help="Only report placements without audit/decision metadata.")
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--output", help="Write to this path instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect(args.volume, args.chapter, args.include_png)
    if args.missing_only:
        rows = [row for row in rows if not row.exists]
    if args.untracked_only:
        rows = [row for row in rows if not row.candidate_id]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as out:
            write_output(rows, args.format, out)
    else:
        write_output(rows, args.format, sys.stdout)
    return 0 if all(row.exists for row in rows) else 1


def write_output(rows: list[MarginFigure], fmt: str, out) -> None:
    if fmt == "json":
        write_json(rows, out)
    elif fmt == "csv":
        write_csv(rows, out)
    else:
        write_markdown(rows, out)


if __name__ == "__main__":
    raise SystemExit(main())
