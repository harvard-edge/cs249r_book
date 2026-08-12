#!/usr/bin/env python3
"""Build review packets for curated glossary gap audits.

The packet output is deliberately an audit aid, not a glossary generator.
Concept maps are treated as candidate sources only; every missing candidate
still needs editorial classification before any glossary entry is added.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTENT_ROOT = REPO_ROOT / "book" / "quarto" / "contents"
OUTPUT_ROOT = REPO_ROOT / ".mlsysbook" / "glossary-pass" / "latest"

PRIMARY_CATEGORIES = (
    "primary_concepts",
    "secondary_concepts",
    "technical_terms",
)
REVIEW_CATEGORIES = (
    "methodologies",
    "formulas",
    "applications",
)
ALL_CATEGORIES = PRIMARY_CATEGORIES + REVIEW_CATEGORIES


@dataclass(frozen=True)
class GlossaryTerm:
    headword: str
    line: int


@dataclass(frozen=True)
class ConceptCandidate:
    volume: str
    chapter: str
    concept_file: Path
    source_file: Path
    category: str
    term: str
    priority: str
    glossary_status: str


def clean_headword(text: str) -> str:
    """Return a plain glossary headword from a QMD definition-list line."""
    text = text.strip()
    text = re.sub(r"^#+\s+", "", text)
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text)
    text = re.sub(r"^`(.*?)`$", r"\1", text)
    return text.strip()


def normalize(text: str) -> str:
    """Normalize terms for fuzzy equality across case, spaces, and hyphens."""
    text = text.casefold()
    text = text.replace("&", "and")
    text = re.sub(r"[‐‑‒–—−-]+", "", text)
    return re.sub(r"[^0-9a-z]+", "", text)


def term_variants(term: str) -> set[str]:
    """Return normalized forms that often represent the same lookup term."""
    variants = {term}
    for match in re.finditer(r"\(([^)]+)\)", term):
        variants.add(match.group(1))
    without_parens = re.sub(r"\s*\([^)]*\)", "", term).strip()
    if without_parens:
        variants.add(without_parens)

    # Handle "Long Form (ACR)" and "ACR (Long Form)" entries.
    words = re.findall(r"[A-Za-z]+", term)
    if words:
        acronym = "".join(word[0] for word in words if word[:1].isupper())
        if len(acronym) > 1:
            variants.add(acronym)

    normalized = {normalize(variant) for variant in variants}
    return {variant for variant in normalized if variant}


def extract_glossary_terms(path: Path) -> list[GlossaryTerm]:
    lines = path.read_text(encoding="utf-8").splitlines()
    terms: list[GlossaryTerm] = []
    in_code = False

    for index, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.lstrip().startswith(":"):
            continue

        previous_index = index - 1
        while previous_index >= 0 and not lines[previous_index].strip():
            previous_index -= 1
        if previous_index < 0:
            continue

        previous = lines[previous_index].strip()
        if not previous or previous.startswith("#") or previous.startswith("```"):
            continue

        headword = clean_headword(previous)
        if headword:
            terms.append(GlossaryTerm(headword=headword, line=previous_index + 1))

    return terms


def glossary_index(terms: Iterable[GlossaryTerm]) -> set[str]:
    index: set[str] = set()
    for term in terms:
        index.update(term_variants(term.headword))
    return index


def chapter_name(concept_path: Path, concept_map: dict[str, object]) -> str:
    source = str(concept_map.get("source") or concept_path.name)
    if source.endswith(".qmd"):
        return Path(source).stem
    return concept_path.parent.name


def source_path_for(concept_path: Path, concept_map: dict[str, object]) -> Path:
    source = concept_map.get("source")
    if isinstance(source, str) and source:
        return concept_path.parent / source
    fallback = concept_path.with_name(concept_path.name.replace("_concepts.yml", ".qmd"))
    return fallback


def collect_candidates(volume: str, glossary_terms: list[GlossaryTerm]) -> list[ConceptCandidate]:
    glossary = glossary_index(glossary_terms)
    candidates: list[ConceptCandidate] = []

    for concept_file in sorted((CONTENT_ROOT / volume).glob("**/*_concepts.yml")):
        data = yaml.safe_load(concept_file.read_text(encoding="utf-8")) or {}
        concept_map = data.get("concept_map") or {}
        if not isinstance(concept_map, dict):
            continue

        chapter = chapter_name(concept_file, concept_map)
        source_file = source_path_for(concept_file, concept_map)
        for category in ALL_CATEGORIES:
            terms = concept_map.get(category) or []
            if not isinstance(terms, list):
                continue
            for raw_term in terms:
                if not isinstance(raw_term, str) or not raw_term.strip():
                    continue
                variants = term_variants(raw_term)
                status = "covered" if variants & glossary else "missing"
                priority = "primary" if category in PRIMARY_CATEGORIES else "review"
                candidates.append(
                    ConceptCandidate(
                        volume=volume,
                        chapter=chapter,
                        concept_file=concept_file,
                        source_file=source_file,
                        category=category,
                        term=raw_term.strip(),
                        priority=priority,
                        glossary_status=status,
                    )
                )

    return candidates


def write_csv(path: Path, candidates: list[ConceptCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "volume",
                "chapter",
                "category",
                "priority",
                "glossary_status",
                "term",
                "decision",
                "rationale",
                "proposed_headword",
                "proposed_definition",
                "concept_file",
                "source_file",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "volume": candidate.volume,
                    "chapter": candidate.chapter,
                    "category": candidate.category,
                    "priority": candidate.priority,
                    "glossary_status": candidate.glossary_status,
                    "term": candidate.term,
                    "decision": "",
                    "rationale": "",
                    "proposed_headword": "",
                    "proposed_definition": "",
                    "concept_file": candidate.concept_file.relative_to(REPO_ROOT),
                    "source_file": candidate.source_file.relative_to(REPO_ROOT)
                    if candidate.source_file.is_relative_to(REPO_ROOT)
                    else candidate.source_file,
                }
            )


def markdown_table(candidates: list[ConceptCandidate]) -> str:
    rows = ["| term | category | priority | status |", "|---|---|---|---|"]
    for candidate in candidates:
        rows.append(
            "| "
            + " | ".join(
                [
                    candidate.term.replace("|", "\\|"),
                    candidate.category,
                    candidate.priority,
                    candidate.glossary_status,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def write_packet(path: Path, chapter: str, candidates: list[ConceptCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if candidates:
        first = candidates[0]
        source_file = first.source_file.relative_to(REPO_ROOT)
        concept_file = first.concept_file.relative_to(REPO_ROOT)
    else:
        source_file = Path("")
        concept_file = Path("")

    missing = [candidate for candidate in candidates if candidate.glossary_status == "missing"]
    body = f"""# Glossary Review Packet: {chapter}

Read the project glossary rules before classifying candidates.

Source chapter: `{source_file}`
Concept map: `{concept_file}`

Classify missing candidates only. Do not add every concept-map entry to the
glossary. Use one of these decisions:

- `add`
- `already covered`
- `too composite`
- `methodology/formula/application`
- `rename concept-map entry`
- `needs review`

For `add`, provide a stable headword and a one- to three-sentence definition.

## Missing Candidates

{markdown_table(missing)}
"""
    path.write_text(body, encoding="utf-8")


def write_summary(path: Path, volume: str, terms: list[GlossaryTerm], candidates: list[ConceptCandidate]) -> None:
    by_category: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        category_counts = by_category.setdefault(candidate.category, {"covered": 0, "missing": 0})
        category_counts[candidate.glossary_status] += 1

    summary = {
        "volume": volume,
        "glossary_headwords": len(terms),
        "concept_candidates": len(candidates),
        "missing_primary_candidates": sum(
            1
            for candidate in candidates
            if candidate.priority == "primary" and candidate.glossary_status == "missing"
        ),
        "missing_review_candidates": sum(
            1
            for candidate in candidates
            if candidate.priority == "review" and candidate.glossary_status == "missing"
        ),
        "by_category": by_category,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(volume: str, output_root: Path) -> None:
    glossary_path = CONTENT_ROOT / volume / "backmatter" / "glossary" / "glossary.qmd"
    terms = extract_glossary_terms(glossary_path)
    candidates = collect_candidates(volume, terms)

    volume_root = output_root / volume
    write_csv(volume_root / "concept_glossary_candidates.csv", candidates)
    write_summary(volume_root / "summary.json", volume, terms, candidates)

    by_chapter: dict[str, list[ConceptCandidate]] = {}
    for candidate in candidates:
        by_chapter.setdefault(candidate.chapter, []).append(candidate)

    packets_root = volume_root / "packets"
    for chapter, chapter_candidates in sorted(by_chapter.items()):
        write_packet(packets_root / f"{chapter}.md", chapter, chapter_candidates)

    print(f"Wrote {volume_root.relative_to(REPO_ROOT)}")
    print(f"  glossary headwords: {len(terms)}")
    print(f"  concept candidates: {len(candidates)}")
    print(
        "  missing primary candidates: "
        + str(
            sum(
                1
                for candidate in candidates
                if candidate.priority == "primary" and candidate.glossary_status == "missing"
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", choices=["vol1", "vol2", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volumes = ["vol1", "vol2"] if args.volume == "all" else [args.volume]
    for volume in volumes:
        run(volume, args.output_root)


if __name__ == "__main__":
    main()
