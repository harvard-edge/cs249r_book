#!/usr/bin/env python3
"""Extract concept taxonomy from preprocessed textbook chapters.

Uses Gemini CLI (gemini-3.1-pro-preview) to extract concepts and
prerequisite relationships from clean prose. Runs chapters in parallel.

Usage:
    python3 extract_taxonomy.py                       # Extract all chapters
    python3 extract_taxonomy.py vol1_nn_computation   # Extract one chapter
    python3 extract_taxonomy.py --merge               # Merge all → taxonomy.json
    python3 extract_taxonomy.py --workers 8           # Control parallelism
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INTERVIEWS_DIR = Path(__file__).parent
PROSE_DIR = INTERVIEWS_DIR / "_prose"
EXTRACTED_DIR = INTERVIEWS_DIR / "_extracted"
TAXONOMY_PATH = INTERVIEWS_DIR / "taxonomy.json"

MODEL = "gemini-3.1-pro-preview"

EXTRACTION_PROMPT = """You are an expert in ML systems education. Read this textbook chapter and extract every distinct technical concept taught.

For each concept, provide:
1. name: A clear, concise concept name
2. description: One sentence explaining what it is
3. prerequisites: List of other concept names the student must understand FIRST

Rules:
- Only extract real technical concepts (not generic terms like "understanding" or "overview")
- Focus on ML SYSTEMS concepts: hardware, memory, compute, networking, optimization, deployment
- Prerequisites must be concepts that appear in THIS chapter or are fundamental CS/math concepts
- Keep names concise (2-5 words)

Output as a JSON array (no markdown, no explanation, ONLY the JSON):
[{"name": "Concept Name", "description": "One sentence.", "prerequisites": ["Prereq 1", "Prereq 2"]}]

Chapter text:
"""


def extract_chapter(chapter_name: str, prose_path: Path) -> dict | None:
    """Extract concepts from one chapter using Gemini CLI."""
    prose = prose_path.read_text(encoding="utf-8")

    # For very large chapters, use first 100KB (~25K tokens)
    if len(prose) > 100_000:
        prose = prose[:100_000]

    prompt = EXTRACTION_PROMPT + prose

    try:
        result = subprocess.run(
            ["gemini", "-m", MODEL, "-p", prompt, "-o", "text"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            error = result.stderr[:200] if result.stderr else "unknown"
            print(f"    ✗ {chapter_name}: {error}")
            return None

        # Parse JSON from output (strip markdown fences if present)
        output = result.stdout.strip()
        output = re.sub(r"^```json\s*", "", output)
        output = re.sub(r"\s*```$", "", output)

        concepts = json.loads(output)

        # Determine volume and domain
        parts = chapter_name.split("_", 1)
        volume = 1 if parts[0] == "vol1" else 2
        domain = parts[1] if len(parts) > 1 else chapter_name

        return {
            "chapter": chapter_name,
            "domain": domain,
            "volume": volume,
            "model": MODEL,
            "concepts": concepts,
        }

    except subprocess.TimeoutExpired:
        print(f"    ✗ {chapter_name}: timeout (120s)")
        return None
    except json.JSONDecodeError as e:
        print(f"    ✗ {chapter_name}: JSON parse error: {e}")
        # Save raw output for debugging
        debug_path = EXTRACTED_DIR / f"{chapter_name}_raw.txt"
        debug_path.write_text(result.stdout)
        return None
    except Exception as e:
        print(f"    ✗ {chapter_name}: {e}")
        return None


def extract_all(
    chapter_filter: str | None = None, max_workers: int = 6
):
    """Extract taxonomy from all chapters in parallel."""
    EXTRACTED_DIR.mkdir(exist_ok=True)

    prose_files = sorted(PROSE_DIR.glob("*.txt"))
    if chapter_filter:
        prose_files = [
            f for f in prose_files if chapter_filter in f.stem
        ]

    # Skip conclusion/introduction (not concept-heavy)
    prose_files = [
        f
        for f in prose_files
        if "conclusion" not in f.stem
        and "introduction" not in f.stem
    ]

    if not prose_files:
        print(f"No prose files found in {PROSE_DIR}/")
        print("Run: python3 preprocess.py first")
        return

    # Check which are already extracted
    to_extract = []
    already_done = []
    for pf in prose_files:
        out_path = EXTRACTED_DIR / f"{pf.stem}.json"
        if out_path.exists() and not chapter_filter:
            already_done.append(pf.stem)
        else:
            to_extract.append(pf)

    print(f"═══ Extracting Taxonomy ({MODEL}) ═══")
    print(
        f"  Chapters: {len(prose_files)} total, "
        f"{len(already_done)} cached, "
        f"{len(to_extract)} to extract"
    )
    print(f"  Workers: {max_workers}")
    print()

    if already_done:
        for name in already_done:
            print(f"  SKIP {name} (cached)")

    # Extract in parallel
    results = {}
    if to_extract:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(extract_chapter, pf.stem, pf): pf.stem
                for pf in to_extract
            }

            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                if result:
                    out_path = EXTRACTED_DIR / f"{name}.json"
                    out_path.write_text(json.dumps(result, indent=2))
                    n_concepts = len(result["concepts"])
                    n_edges = sum(
                        len(c.get("prerequisites", []))
                        for c in result["concepts"]
                    )
                    print(
                        f"  ✓ {name}: {n_concepts} concepts, "
                        f"{n_edges} edges"
                    )
                    results[name] = result

    # Load all results (cached + new)
    all_results = []
    for f in sorted(EXTRACTED_DIR.glob("*.json")):
        if f.stem.endswith("_raw"):
            continue
        all_results.append(json.loads(f.read_text()))

    # Summary
    total_concepts = sum(
        len(r["concepts"]) for r in all_results
    )
    total_edges = sum(
        len(c.get("prerequisites", []))
        for r in all_results
        for c in r["concepts"]
    )
    print(f"\n═══ Extraction Complete ═══")
    print(f"  Chapters: {len(all_results)}")
    print(f"  Total concepts: {total_concepts}")
    print(f"  Total prerequisite edges: {total_edges}")

    return all_results


def merge_extractions() -> dict:
    """Merge per-chapter extractions into a unified taxonomy.json.

    Phase 1: Dedup concepts by name (case-insensitive)
    Phase 2: Union all prerequisite edges
    Phase 3: Assign concept IDs (kebab-case)
    """
    print("═══ Merging Taxonomy ═══\n")

    extractions = []
    for f in sorted(EXTRACTED_DIR.glob("*.json")):
        if f.stem.endswith("_raw"):
            continue
        extractions.append(json.loads(f.read_text()))

    if not extractions:
        print("No extractions found. Run extract first.")
        return {}

    # Phase 1: Collect all concepts, dedup by name
    raw_concepts: dict[str, dict] = {}
    for ext in extractions:
        chapter = ext["chapter"]
        volume = ext["volume"]
        domain = ext["domain"]
        for c in ext["concepts"]:
            key = c["name"].lower().strip()
            if key not in raw_concepts:
                raw_concepts[key] = {
                    "name": c["name"],
                    "id": re.sub(
                        r"[^a-z0-9]+", "-", key
                    ).strip("-"),
                    "description": c.get("description", ""),
                    "prerequisites_raw": set(),
                    "source_chapters": set(),
                    "source_domains": set(),
                    "volumes": set(),
                }
            raw_concepts[key]["source_chapters"].add(chapter)
            raw_concepts[key]["source_domains"].add(domain)
            raw_concepts[key]["volumes"].add(volume)
            for p in c.get("prerequisites", []):
                raw_concepts[key]["prerequisites_raw"].add(
                    p.lower().strip()
                )

    print(f"  Raw concepts: {len(raw_concepts)}")

    # Phase 2: Resolve prerequisites to concept IDs
    name_to_id = {
        key: data["id"] for key, data in raw_concepts.items()
    }

    concepts = []
    edges = []
    unresolved = 0
    for key, data in raw_concepts.items():
        resolved_prereqs = []
        for prereq_name in data["prerequisites_raw"]:
            if prereq_name in name_to_id:
                resolved_prereqs.append(name_to_id[prereq_name])
                edges.append(
                    {
                        "source": name_to_id[prereq_name],
                        "target": data["id"],
                        "type": "prerequisite",
                    }
                )
            else:
                unresolved += 1

        # Track mapping from volume/domain
        tracks = ["cloud", "edge", "mobile", "tinyml"]
        domain_track_map = {
            "edge_intelligence": ["edge"],
            "distributed_training": ["cloud"],
            "fleet_orchestration": ["cloud"],
            "network_fabrics": ["cloud"],
            "collective_communication": ["cloud"],
        }
        domains = data["source_domains"]
        if any(d in domain_track_map for d in domains):
            track_set = set()
            for d in domains:
                track_set.update(
                    domain_track_map.get(d, tracks)
                )
            tracks = sorted(track_set)

        concepts.append(
            {
                "id": data["id"],
                "name": data["name"],
                "description": data["description"],
                "prerequisites": resolved_prereqs,
                "tracks": tracks,
                "source_chapters": sorted(
                    data["source_chapters"]
                ),
                "source_domains": sorted(
                    data["source_domains"]
                ),
                "question_count": 0,
            }
        )

    # Deduplicate edges
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    taxonomy = {
        "version": "3.0",
        "extracted_from": "textbook",
        "extraction_model": MODEL,
        "extraction_date": __import__(
            "datetime"
        ).datetime.now().isoformat()[:10],
        "concepts": sorted(concepts, key=lambda c: c["id"]),
        "edges": unique_edges,
        "stats": {
            "total_concepts": len(concepts),
            "total_edges": len(unique_edges),
            "unresolved_prereqs": unresolved,
            "chapters_processed": len(extractions),
        },
    }

    TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2))
    print(f"  Unique concepts: {len(concepts)}")
    print(f"  Prerequisite edges: {len(unique_edges)}")
    print(f"  Unresolved prereqs: {unresolved}")
    print(f"  Saved to: {TAXONOMY_PATH}")

    return taxonomy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract taxonomy from textbook chapters"
    )
    parser.add_argument(
        "chapter", nargs="?", help="Chapter filter"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge extractions into taxonomy.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel workers (default: 6)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if cached",
    )
    args = parser.parse_args()

    if args.merge:
        merge_extractions()
    else:
        # Clear cache if force
        if args.force and EXTRACTED_DIR.exists():
            for f in EXTRACTED_DIR.glob("*.json"):
                f.unlink()
        extract_all(
            chapter_filter=args.chapter,
            max_workers=args.workers,
        )
