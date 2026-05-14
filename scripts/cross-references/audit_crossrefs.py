#!/usr/bin/env python3
"""Audit Volume 1 and Volume 2 cross-references.

This script is deliberately conservative. It reports mechanical facts and
chapter-sized editorial cues; it does not rewrite prose or infer final targets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "book" / "quarto" / "contents").exists():
            return parent
    raise RuntimeError("Could not locate repository root containing book/quarto/contents")


ROOT = find_repo_root()
CONTENTS = ROOT / "book" / "quarto" / "contents"
OUT_DIR = ROOT / "review" / "cross-references"
INVENTORY_PATH = OUT_DIR / "inventory.json"
REPORT_PATH = OUT_DIR / "report.md"
PACKET_DIR = OUT_DIR / "chapter-packets"
CANONICAL_INDEX_PATH = OUT_DIR / "canonical-target-candidates.yml"
SCHEMA_PATH = OUT_DIR / "chapter-report-schema.yml"


def configure_output(out_dir: Path) -> None:
    global OUT_DIR, INVENTORY_PATH, REPORT_PATH, PACKET_DIR, CANONICAL_INDEX_PATH, SCHEMA_PATH

    OUT_DIR = out_dir
    INVENTORY_PATH = OUT_DIR / "inventory.json"
    REPORT_PATH = OUT_DIR / "report.md"
    PACKET_DIR = OUT_DIR / "chapter-packets"
    CANONICAL_INDEX_PATH = OUT_DIR / "canonical-target-candidates.yml"
    SCHEMA_PATH = OUT_DIR / "chapter-report-schema.yml"

ANCHOR_RE = re.compile(r"\{[^}]*#((?:sec|fig|tbl|eq|pri)-[A-Za-z0-9_-]+)(?=[\s}])")
QMD_REF_RE = re.compile(r"(?<![A-Za-z0-9_./-])@((?:sec|fig|tbl|eq)-[A-Za-z0-9_-]+)")
PRI_REF_RE = re.compile(r"\\ref\{(pri-[A-Za-z0-9_-]+)\}")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+\{#([A-Za-z0-9_-]+)\})?\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

MISSING_CUE_RE = re.compile(
    r"\b("
    r"introduced earlier|discussed earlier|as discussed|as shown|as described|"
    r"we saw|we introduced|will return to|later in this|full treatment|"
    r"canonical|taxonomy|framework|law|principle|archetype|lighthouse|"
    r"roadmap|playbook|derivation|appendix|glossary"
    r")\b",
    re.IGNORECASE,
)

SKIP_PARTS = {
    "references.qmd",
}


@dataclass
class Anchor:
    id: str
    kind: str
    volume: str
    file: str
    line: int
    title: str


@dataclass
class Reference:
    id: str
    kind: str
    volume: str
    file: str
    line: int
    text: str
    target_volume: str | None
    target_file: str | None
    status: str


@dataclass
class Cue:
    volume: str
    file: str
    line: int
    text: str
    reason: str


@dataclass
class DuplicateAnchor:
    id: str
    occurrences: list[Anchor]


def volume_for(path: Path) -> str | None:
    parts = path.parts
    if "vol1" in parts:
        return "vol1"
    if "vol2" in parts:
        return "vol2"
    return None


def kind_for(ref_id: str) -> str:
    return ref_id.split("-", 1)[0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_qmd_files() -> list[Path]:
    files: list[Path] = []
    for volume in ("vol1", "vol2"):
        base = CONTENTS / volume
        for path in sorted(base.rglob("*.qmd")):
            if path.name in SKIP_PARTS:
                continue
            if "_shelved" in path.as_posix():
                continue
            files.append(path)
    return files


def clean_heading_title(raw: str) -> str:
    return re.sub(r"\s+\{#[^}]+\}\s*$", "", raw).strip()


def is_rendered_skip_line(line: str, in_fence: bool) -> bool:
    stripped = line.strip()
    return (
        in_fence
        or stripped.startswith("<!--")
        or stripped.startswith("#|")
        or stripped.startswith("# │")
        or stripped.startswith("# |")
    )


def extract() -> tuple[dict[str, list[Anchor]], list[Reference], list[Cue], dict[str, dict], list[DuplicateAnchor]]:
    anchors: dict[str, list[Anchor]] = defaultdict(list)
    refs_raw: list[tuple[str, str, Path, int, str]] = []
    cues: list[Cue] = []
    file_stats: dict[str, dict] = {}

    files = iter_qmd_files()
    for path in files:
        volume = volume_for(path)
        if volume is None:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        file_key = rel(path)
        headings = 0
        in_fence = False

        for line_no, line in enumerate(lines, start=1):
            fence_match = FENCE_RE.match(line)
            heading = HEADING_RE.match(line)
            title = ""
            if heading:
                headings += 1
                title = clean_heading_title(heading.group(2))

            for match in ANCHOR_RE.finditer(line):
                anchor_id = match.group(1)
                anchors[anchor_id].append(
                    Anchor(
                        id=anchor_id,
                        kind=kind_for(anchor_id),
                        volume=volume,
                        file=file_key,
                        line=line_no,
                        title=title,
                    )
                )

            skip_rendered = is_rendered_skip_line(line, in_fence)
            if not skip_rendered:
                for match in QMD_REF_RE.finditer(line):
                    refs_raw.append((match.group(1), kind_for(match.group(1)), path, line_no, line.strip()))

                for match in PRI_REF_RE.finditer(line):
                    refs_raw.append((match.group(1), "pri", path, line_no, line.strip()))

            if not skip_rendered and MISSING_CUE_RE.search(line) and not QMD_REF_RE.search(line) and not PRI_REF_RE.search(line):
                stripped = line.strip()
                if stripped:
                    cues.append(
                        Cue(
                            volume=volume,
                            file=file_key,
                            line=line_no,
                            text=stripped[:240],
                            reason="cue phrase without nearby explicit cross-reference on the same line",
                        )
                    )

            if fence_match:
                in_fence = not in_fence

        file_stats[file_key] = {
            "volume": volume,
            "lines": len(lines),
            "headings": headings,
        }

    duplicates = [
        DuplicateAnchor(id=anchor_id, occurrences=items)
        for anchor_id, items in sorted(anchors.items())
        if len(items) > 1
    ]

    references: list[Reference] = []
    for ref_id, kind, path, line_no, line in refs_raw:
        source_volume = volume_for(path) or "unknown"
        candidates = anchors.get(ref_id, [])
        same_volume = [anchor for anchor in candidates if anchor.volume == source_volume]
        other_volume = [anchor for anchor in candidates if anchor.volume != source_volume]
        if not candidates:
            status = "unresolved"
            target_volume = None
            target_file = None
        elif len(same_volume) == 1:
            target = same_volume[0]
            status = "ok"
            target_volume = target.volume
            target_file = target.file
        elif len(same_volume) > 1:
            target = same_volume[0]
            status = "ambiguous"
            target_volume = target.volume
            target_file = target.file
        elif other_volume:
            target = other_volume[0]
            status = "cross-volume"
            target_volume = target.volume
            target_file = target.file
        else:
            status = "unresolved"
            target_volume = None
            target_file = None

        references.append(
            Reference(
                id=ref_id,
                kind=kind,
                volume=source_volume,
                file=rel(path),
                line=line_no,
                text=line[:240],
                target_volume=target_volume,
                target_file=target_file,
                status=status,
            )
        )

    return anchors, references, cues, file_stats, duplicates


def flatten_anchors(anchors: dict[str, list[Anchor]]) -> list[Anchor]:
    return [anchor for items in anchors.values() for anchor in items]


def yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_schema() -> None:
    SCHEMA_PATH.write_text(
        """# Schema for human or agent chapter cross-reference reports.
schema_version: "crossref-chapter-report/v1"
chapter_report:
  volume: "vol1|vol2"
  file: "book/quarto/contents/<volume>/<chapter>/<chapter>.qmd"
  reviewer: "agent-or-human-name"
  generated_from_packet: "review/cross-references/chapter-packets/<packet>.yml"
  status: "analysis-only|ready-for-edit|needs-human-decision"
  summary:
    existing_refs: 0
    proposed_keep: 0
    proposed_retarget: 0
    proposed_remove: 0
    proposed_add: 0
    proposed_localize: 0
    unresolved_mechanical: 0
    cross_volume_mechanical: 0
    confidence: "high|medium|low"
  canonical_targets_introduced:
    - anchor: "sec-..."
      kind: "sec|fig|tbl|eq|pri"
      title: "reader-facing title or caption label"
      line: 0
      concept_tags: ["taxonomy", "framework"]
      should_receive_incoming_refs: true
  findings:
    - id: "stable-decision-id"
      line: 0
      context: "short excerpt"
      action: "keep|add|remove|retarget|localize|needs-map-query"
      current_reference: "existing @sec/@fig/@tbl/@eq or null"
      recommended_target: "same-volume target id or null"
      priority: "blocker|normal|optional"
      confidence: "high|medium|low"
      needs_human_review: false
      rationale: "one sentence"
      proposed_edit: "exact prose edit or null"
  map_queries:
    - concept: "concept needing a canonical target"
      local_context: "short quoted/paraphrased context"
      constraints: "same volume only; prefer canonical definitions"
""",
        encoding="utf-8",
    )


def write_canonical_candidates(anchors: dict[str, list[Anchor]], references: list[Reference]) -> None:
    incoming = Counter(ref.id for ref in references if ref.status == "ok")
    candidate_words = re.compile(
        r"\b("
        r"introduction|taxonomy|framework|law|principle|archetype|lighthouse|"
        r"roadmap|playbook|summary|foundations|diagnostic|model|roofline|"
        r"stack|pipeline|workflow|invariant"
        r")\b",
        re.IGNORECASE,
    )

    by_volume: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in flatten_anchors(anchors):
        if anchor.kind == "sec" and (incoming[anchor.id] or candidate_words.search(anchor.title)):
            by_volume[anchor.volume].append(anchor)

    lines: list[str] = []
    lines.append("# Candidate canonical cross-reference targets.")
    lines.append("# This is an input to the map-agent pass, not a final editorial decision.")
    for volume in ("vol1", "vol2"):
        lines.append(f"{volume}:")
        for anchor in sorted(by_volume[volume], key=lambda item: (-incoming[item.id], item.file, item.line)):
            lines.append(f"  - anchor: {yaml_scalar(anchor.id)}")
            lines.append(f"    file: {yaml_scalar(anchor.file)}")
            lines.append(f"    line: {anchor.line}")
            lines.append(f"    title: {yaml_scalar(anchor.title)}")
            lines.append(f"    incoming_references: {incoming[anchor.id]}")
        if not by_volume[volume]:
            lines.append("  []")
    CANONICAL_INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chapter_packets(
    anchors: dict[str, list[Anchor]],
    references: list[Reference],
    cues: list[Cue],
    file_stats: dict[str, dict],
) -> None:
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    all_anchors = flatten_anchors(anchors)
    anchors_by_file: dict[str, list[Anchor]] = defaultdict(list)
    refs_by_file: dict[str, list[Reference]] = defaultdict(list)
    incoming_by_file: dict[str, list[Reference]] = defaultdict(list)
    cues_by_file: dict[str, list[Cue]] = defaultdict(list)

    for anchor in all_anchors:
        anchors_by_file[anchor.file].append(anchor)
    for ref in references:
        refs_by_file[ref.file].append(ref)
        if ref.target_file:
            incoming_by_file[ref.target_file].append(ref)
    for cue in cues:
        cues_by_file[cue.file].append(cue)

    for file_key in sorted(file_stats):
        stats = file_stats[file_key]
        packet_name = file_key.removeprefix("book/quarto/contents/").replace("/", "__").removesuffix(".qmd")
        path = PACKET_DIR / f"{packet_name}.yml"
        lines: list[str] = []
        lines.append(f"volume: {yaml_scalar(stats['volume'])}")
        lines.append(f"file: {yaml_scalar(file_key)}")
        lines.append(f"lines: {stats['lines']}")
        lines.append(f"headings: {stats['headings']}")
        lines.append("anchors:")
        for anchor in sorted(anchors_by_file[file_key], key=lambda item: item.line):
            lines.append(f"  - id: {yaml_scalar(anchor.id)}")
            lines.append(f"    kind: {yaml_scalar(anchor.kind)}")
            lines.append(f"    line: {anchor.line}")
            lines.append(f"    title: {yaml_scalar(anchor.title)}")
        if not anchors_by_file[file_key]:
            lines.append("  []")

        lines.append("outgoing_references:")
        for ref in sorted(refs_by_file[file_key], key=lambda item: item.line):
            lines.append(f"  - id: {yaml_scalar(ref.id)}")
            lines.append(f"    kind: {yaml_scalar(ref.kind)}")
            lines.append(f"    line: {ref.line}")
            lines.append(f"    status: {yaml_scalar(ref.status)}")
            lines.append(f"    target_volume: {yaml_scalar(ref.target_volume)}")
            lines.append(f"    target_file: {yaml_scalar(ref.target_file)}")
            lines.append(f"    text: {yaml_scalar(ref.text)}")
        if not refs_by_file[file_key]:
            lines.append("  []")

        lines.append("incoming_references:")
        for ref in sorted(incoming_by_file[file_key], key=lambda item: (item.file, item.line)):
            lines.append(f"  - source_file: {yaml_scalar(ref.file)}")
            lines.append(f"    source_line: {ref.line}")
            lines.append(f"    id: {yaml_scalar(ref.id)}")
            lines.append(f"    text: {yaml_scalar(ref.text)}")
        if not incoming_by_file[file_key]:
            lines.append("  []")

        lines.append("candidate_missing_pointer_cues:")
        for cue in sorted(cues_by_file[file_key], key=lambda item: item.line):
            lines.append(f"  - line: {cue.line}")
            lines.append(f"    reason: {yaml_scalar(cue.reason)}")
            lines.append(f"    text: {yaml_scalar(cue.text)}")
        if not cues_by_file[file_key]:
            lines.append("  []")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    anchors: dict[str, list[Anchor]],
    references: list[Reference],
    cues: list[Cue],
    file_stats: dict[str, dict],
    duplicates: list[DuplicateAnchor],
) -> str:
    all_anchors = flatten_anchors(anchors)
    by_volume_files = Counter(stats["volume"] for stats in file_stats.values())
    by_volume_anchors = Counter(anchor.volume for anchor in all_anchors)
    by_volume_refs = Counter(ref.volume for ref in references)
    by_status = Counter(ref.status for ref in references)
    by_kind = Counter(ref.kind for ref in references)

    per_file_refs: dict[str, Counter] = defaultdict(Counter)
    for ref in references:
        per_file_refs[ref.file][ref.status] += 1
        per_file_refs[ref.file]["total"] += 1

    per_file_cues = Counter(cue.file for cue in cues)

    lines: list[str] = []
    lines.append("# Cross-Reference Audit Report")
    lines.append("")
    lines.append("Generated by `review/cross-references/scripts/audit_crossrefs.py`.")
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append("| Metric | Volume 1 | Volume 2 | Total |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| QMD files | {by_volume_files['vol1']} | {by_volume_files['vol2']} | {sum(by_volume_files.values())} |"
    )
    lines.append(
        f"| Anchors | {by_volume_anchors['vol1']} | {by_volume_anchors['vol2']} | {sum(by_volume_anchors.values())} |"
    )
    lines.append(
        f"| References | {by_volume_refs['vol1']} | {by_volume_refs['vol2']} | {sum(by_volume_refs.values())} |"
    )
    lines.append("")
    lines.append("## Mechanical Findings")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for status in ("ok", "unresolved", "cross-volume", "ambiguous"):
        lines.append(f"| {status} | {by_status[status]} |")
    lines.append("")
    lines.append("| Kind | Count |")
    lines.append("|---|---:|")
    for kind, count in sorted(by_kind.items()):
        lines.append(f"| {kind} | {count} |")
    lines.append("")

    lines.append("## Duplicate Anchors")
    lines.append("")
    lines.append(
        "Duplicate IDs are allowed only when they occur once per separate volume and "
        "all references resolve to the same-volume copy. Same-volume duplicates need manual repair."
    )
    lines.append("")
    if not duplicates:
        lines.append("No duplicate anchors found.")
    else:
        lines.append("| Anchor | Occurrences |")
        lines.append("|---|---|")
        for duplicate in duplicates[:100]:
            occurrences = "; ".join(f"{item.file}:{item.line}" for item in duplicate.occurrences)
            lines.append(f"| `{duplicate.id}` | {occurrences} |")
        if len(duplicates) > 100:
            lines.append("")
            lines.append(f"Showing first 100 of {len(duplicates)} duplicate anchors.")
    lines.append("")

    flagged = [ref for ref in references if ref.status != "ok"]
    lines.append("## References Requiring Mechanical Review")
    lines.append("")
    if not flagged:
        lines.append("No unresolved or cross-volume references found.")
    else:
        lines.append("| Status | Source | Target | Target file | Text |")
        lines.append("|---|---|---|---|---|")
        for ref in flagged[:200]:
            source = f"{ref.file}:{ref.line}"
            target_file = ref.target_file or ""
            text = ref.text.replace("|", "\\|")
            lines.append(f"| {ref.status} | `{source}` | `{ref.id}` | `{target_file}` | {text} |")
        if len(flagged) > 200:
            lines.append(f"")
            lines.append(f"Showing first 200 of {len(flagged)} flagged references.")
    lines.append("")

    lines.append("## Chapter Work Packets")
    lines.append("")
    lines.append("| File | Lines | Headings | References | Flagged | Cue lines |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for file_key in sorted(file_stats):
        stats = file_stats[file_key]
        refs = per_file_refs[file_key]["total"]
        flagged_count = (
            per_file_refs[file_key]["unresolved"]
            + per_file_refs[file_key]["cross-volume"]
            + per_file_refs[file_key]["ambiguous"]
        )
        cues_count = per_file_cues[file_key]
        if refs or cues_count or stats["headings"]:
            lines.append(
                f"| `{file_key}` | {stats['lines']} | {stats['headings']} | {refs} | {flagged_count} | {cues_count} |"
            )
    lines.append("")

    lines.append("## Candidate Missing-Pointer Cues")
    lines.append("")
    lines.append(
        "These lines contain cue phrases but no explicit cross-reference on the same line. "
        "They are editorial prompts, not automatic failures."
    )
    lines.append("")
    if not cues:
        lines.append("No cue lines found.")
    else:
        lines.append("| Source | Reason | Text |")
        lines.append("|---|---|---|")
        for cue in cues[:300]:
            text = cue.text.replace("|", "\\|")
            lines.append(f"| `{cue.file}:{cue.line}` | {cue.reason} | {text} |")
        if len(cues) > 300:
            lines.append("")
            lines.append(f"Showing first 300 of {len(cues)} cue lines.")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Use this report to assign one chapter at a time. Mechanical findings should be "
        "fixed before editorial additions. Cue lines should be reviewed by an editor."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and packetize MLSysBook cross-references.")
    parser.add_argument(
        "--out-dir",
        default="review/cross-references",
        help="Output directory relative to the repository root, or an absolute path.",
    )
    parser.add_argument(
        "--no-packets",
        action="store_true",
        help="Do not write per-chapter YAML packets.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    configure_output(out_dir if out_dir.is_absolute() else ROOT / out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors, references, cues, file_stats, duplicates = extract()
    all_anchors = flatten_anchors(anchors)

    inventory = {
        "anchors": [asdict(anchor) for anchor in sorted(all_anchors, key=lambda a: (a.volume, a.file, a.line, a.id))],
        "references": [asdict(ref) for ref in references],
        "cues": [asdict(cue) for cue in cues],
        "duplicate_anchors": [
            {"id": duplicate.id, "occurrences": [asdict(anchor) for anchor in duplicate.occurrences]}
            for duplicate in duplicates
        ],
        "file_stats": file_stats,
    }
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(summarize(anchors, references, cues, file_stats, duplicates), encoding="utf-8")
    write_schema()
    write_canonical_candidates(anchors, references)
    if not args.no_packets:
        write_chapter_packets(anchors, references, cues, file_stats)

    flagged = [ref for ref in references if ref.status != "ok"]
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {INVENTORY_PATH.relative_to(ROOT)}")
    print(f"Wrote {SCHEMA_PATH.relative_to(ROOT)}")
    print(f"Wrote {CANONICAL_INDEX_PATH.relative_to(ROOT)}")
    if not args.no_packets:
        print(f"Wrote chapter packets to {PACKET_DIR.relative_to(ROOT)}")
    print(f"References: {len(references)}; anchors: {len(all_anchors)}; flagged: {len(flagged)}; cues: {len(cues)}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
