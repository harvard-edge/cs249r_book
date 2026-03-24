#!/usr/bin/env python3
"""vault.py — Unified CLI for the StaffML interview question vault.

Usage:
    python3 vault.py validate     # Schema check
    python3 vault.py stats        # Full statistics
    python3 vault.py gaps         # 3D coverage cube analysis
    python3 vault.py dedup        # Multi-stage dedup (exact + fuzzy + semantic)
    python3 vault.py search "q"   # Semantic search
    python3 vault.py chains       # Build depth chains + verify coherence
    python3 vault.py add <file>   # Validated insertion
    python3 vault.py export       # Generate .md files + sync to app
    python3 vault.py analyze      # Taxonomy gap analysis → _generation_plan.json
    python3 vault.py generate     # Generate questions from plan via Gemini
    python3 vault.py loop         # Auto-loop: analyze→generate→add until saturated
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path(__file__).parent
LEVELS_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6+"]
TRACKS = ["cloud", "edge", "mobile", "tinyml"]


def load_config() -> dict:
    """Load vault.yaml, falling back to defaults."""
    config_path = BASE / "vault.yaml"
    defaults = {
        "paths": {
            "corpus": "corpus.json",
            "chains": "chains.json",
            "taxonomy": "taxonomy_v2.json",
            "chroma_dir": "_chroma",
            "reviews_dir": "_reviews",
            "paper_dir": "paper",
            "staffml_export": "staffml/src/data/corpus.json",
        },
        "models": {
            "reviewer": "gemini-2.5-flash",
            "reviewer_deep": "claude-opus-4-6",
        },
        "review": {
            "chunk_size": 50,
            "max_parallel": 8,
            "auto_fix": False,
            "error_rate_threshold": 0.02,
        },
        "validation": {"dedup_threshold": 0.85, "fuzzy_threshold": 0.90},
        "chains": {"min_levels": 3, "backpopulate": True},
        "coverage": {"min_per_cell": 3},
        "release": {
            "steps": ["validate", "dedup", "chains", "stats", "figures", "export"],
            "require_zero_errors": True,
        },
    }
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                user = yaml.safe_load(f) or {}
            # Shallow merge per section
            for k, v in user.items():
                if isinstance(v, dict) and k in defaults:
                    defaults[k].update(v)
                else:
                    defaults[k] = v
        except ImportError:
            pass  # yaml not installed, use defaults
    return defaults


CONFIG = load_config()
CORPUS_PATH = BASE / CONFIG["paths"]["corpus"]
CHAINS_PATH = BASE / CONFIG["paths"]["chains"]
TAXONOMY_PATH = BASE / CONFIG["paths"]["taxonomy"]
STAFFML_DATA = BASE / CONFIG["paths"]["staffml_export"]


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text())


def save_corpus(corpus: list[dict]) -> None:
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))


# ─── VALIDATE ───────────────────────────────────────────────────
def cmd_validate(args):
    """Validate corpus.json against Pydantic schema."""
    from schema import validate_corpus

    corpus = load_corpus()
    print(f"Validating {len(corpus)} questions...\n")

    valid, errors, warnings = validate_corpus(corpus)

    if warnings:
        print(f"⚠️  {len(warnings)} warnings (duplicate titles — dedup candidates):\n")
        for w in warnings[:20]:
            print(f"  • {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
        print()

    if errors:
        print(f"❌ {len(errors)} validation errors:\n")
        for err in errors[:50]:
            print(f"  • {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        sys.exit(1)
    else:
        print(f"✅ All {len(valid)} questions pass schema validation.")


# ─── STATS ──────────────────────────────────────────────────────
def cmd_stats(args):
    """Print comprehensive corpus statistics."""
    corpus = load_corpus()
    total = len(corpus)

    print(f"═══ StaffML Vault Statistics ═══")
    print(f"  Total questions: {total}\n")

    # Track × Level matrix
    print("── Track × Level ──")
    matrix = defaultdict(lambda: defaultdict(int))
    for q in corpus:
        matrix[q["track"]][q["level"]] += 1

    header = f"  {'':12}" + "".join(f"{l:>6}" for l in LEVELS_ORDER) + f"{'Total':>7}"
    print(header)
    for t in TRACKS + ["global"]:
        row = [matrix[t][l] for l in LEVELS_ORDER]
        line = f"  {t:12}" + "".join(f"{v:6}" for v in row) + f"{sum(row):7}"
        print(line)
    totals = [sum(matrix[t][l] for t in TRACKS + ["global"]) for l in LEVELS_ORDER]
    print(f"  {'TOTAL':12}" + "".join(f"{v:6}" for v in totals) + f"{sum(totals):7}")

    # Competency area distribution
    print(f"\n── Competency Areas ──")
    areas = Counter(q.get("competency_area", "") for q in corpus)
    for a, cnt in areas.most_common():
        bar = "█" * (cnt // 15)
        print(f"  {a or '(empty)':16} {cnt:4}  {bar}")

    # Field coverage
    print(f"\n── Field Coverage ──")
    fields = {
        "competency_area": lambda q: q.get("competency_area", "").strip(),
        "napkin_math": lambda q: q.get("details", {}).get("napkin_math", "").strip(),
        "common_mistake": lambda q: q.get("details", {}).get("common_mistake", "").strip(),
        "realistic_solution": lambda q: q.get("details", {}).get("realistic_solution", "").strip(),
        "deep_dive_url": lambda q: q.get("details", {}).get("deep_dive_url", "").strip(),
        "MCQ options": lambda q: q.get("details", {}).get("options"),
        "bloom_level": lambda q: q.get("bloom_level", "").strip(),
        "canonical_topic": lambda q: q.get("canonical_topic", "").strip(),
    }
    for name, fn in fields.items():
        count = sum(1 for q in corpus if fn(q))
        pct = 100 * count / total
        status = "✅" if pct >= 99 else "⚠️" if pct >= 80 else "❌"
        print(f"  {status} {name:22} {count:4}/{total} ({pct:.1f}%)")

    # Question format breakdown
    print(f"\n── Question Format ──")
    format_counts = Counter()
    for q in corpus:
        s = q.get("scenario", "").lower()
        if any(w in s for w in ["calculate", "compute", "estimate", "how many", "how much"]):
            format_counts["calculation"] += 1
        if any(w in s for w in ["diagnose", "debug", "why is", "root cause", "fails"]):
            format_counts["diagnosis"] += 1
        if any(w in s for w in ["design", "architect", "propose", "how would you build"]):
            format_counts["design"] += 1
        if any(w in s for w in ["compare", "trade-off", "tradeoff", "versus", " vs "]):
            format_counts["tradeoff"] += 1
        if any(w in s for w in ["optimize", "improve", "reduce", "speed up"]):
            format_counts["optimization"] += 1
        if any(w in s for w in ["explain", "what is", "define", "describe"]):
            format_counts["conceptual"] += 1
    for fmt, cnt in format_counts.most_common():
        print(f"  {fmt:16} {cnt:4} ({100*cnt/total:.0f}%)")

    # Underfilled cells
    cube = defaultdict(int)
    for q in corpus:
        if q["track"] in TRACKS and q.get("competency_area"):
            cube[(q["track"], q["level"], q["competency_area"])] += 1
    underfilled = sum(1 for v in cube.values() if 0 < v < 3)
    empty = sum(1 for v in cube.values() if v == 0)
    print(f"\n── 3D Cube ──")
    print(f"  Empty cells: {empty}")
    print(f"  Underfilled (<3): {underfilled}")
    print(f"  Healthy (≥3): {sum(1 for v in cube.values() if v >= 3)}")

    # Chain stats
    chain_topics = defaultdict(set)
    topic_key = "canonical_topic" if corpus[0].get("canonical_topic") else "topic"
    for q in corpus:
        if q["track"] in TRACKS:
            chain_topics[(q["track"], q.get(topic_key, q["topic"]))].add(q["level"])
    chains_3 = sum(1 for lvls in chain_topics.values() if len(lvls) >= 3)
    chains_5 = sum(1 for lvls in chain_topics.values() if len(lvls) >= 5)
    print(f"\n── Depth Chains ──")
    print(f"  Chains (3+ levels): {chains_3}")
    print(f"  Chains (5+ levels): {chains_5}")

    # Unique topics
    topics = set(q.get(topic_key, q["topic"]) for q in corpus)
    singletons = sum(1 for t in topics if sum(1 for q in corpus if q.get(topic_key, q["topic"]) == t) == 1)
    print(f"  Unique topics: {len(topics)}")
    print(f"  Singleton topics: {singletons}")


# ─── GAPS ───────────────────────────────────────────────────────
def cmd_gaps(args):
    """Show underfilled cells in the 3D coverage cube."""
    corpus = load_corpus()
    areas = sorted(set(q["competency_area"] for q in corpus if q["competency_area"]))

    cube = defaultdict(int)
    for q in corpus:
        if q["track"] in TRACKS and q.get("competency_area"):
            cube[(q["track"], q["level"], q["competency_area"])] += 1

    min_target = args.min if hasattr(args, "min") and args.min else 3
    gaps = []
    for t in TRACKS:
        for l in LEVELS_ORDER:
            for a in areas:
                cnt = cube.get((t, l, a), 0)
                if cnt < min_target:
                    gaps.append((t, l, a, cnt, min_target - cnt))

    print(f"═══ Coverage Gaps (target ≥ {min_target} per cell) ═══\n")
    print(f"  Total underfilled cells: {len(gaps)}")
    print(f"  Questions needed: {sum(n for _, _, _, _, n in gaps)}\n")

    by_track = defaultdict(list)
    for t, l, a, have, need in gaps:
        by_track[t].append((l, a, have, need))

    for t in TRACKS:
        cells = by_track.get(t, [])
        if not cells:
            continue
        need_total = sum(n for _, _, _, n in cells)
        print(f"  {t}: {len(cells)} gaps, {need_total} questions needed")
        for l, a, have, need in cells:
            print(f"    {l}/{a}: have {have}, need {need}")
        print()


# ─── DEDUP ──────────────────────────────────────────────────────
def cmd_dedup(args):
    """Multi-stage deduplication check."""
    corpus = load_corpus()
    threshold = args.threshold if hasattr(args, "threshold") and args.threshold else 0.85

    print(f"═══ Dedup Check ({len(corpus)} questions) ═══\n")

    # Stage 1: Exact match
    print("── Stage 1: Exact Match ──")
    scenarios = defaultdict(list)
    for q in corpus:
        key = q["scenario"].lower().strip()
        scenarios[key].append(q["id"])
    exact_dupes = {k: v for k, v in scenarios.items() if len(v) > 1}
    print(f"  Exact duplicate groups: {len(exact_dupes)}")
    for key, ids in list(exact_dupes.items())[:5]:
        print(f"    {ids}")

    # Stage 2: Fuzzy match within (track, topic) groups
    print("\n── Stage 2: Fuzzy Match (>0.90) ──")
    groups = defaultdict(list)
    topic_key = "canonical_topic" if corpus[0].get("canonical_topic") else "topic"
    for q in corpus:
        groups[(q["track"], q.get(topic_key, q["topic"]))].append(q)

    fuzzy_pairs = []
    for group_key, qs in groups.items():
        if len(qs) < 2:
            continue
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                ratio = SequenceMatcher(
                    None,
                    qs[i]["scenario"].lower(),
                    qs[j]["scenario"].lower(),
                ).ratio()
                if ratio > 0.90:
                    fuzzy_pairs.append((qs[i]["id"], qs[j]["id"], ratio))

    print(f"  Fuzzy duplicate pairs: {len(fuzzy_pairs)}")
    for id1, id2, ratio in fuzzy_pairs[:10]:
        print(f"    {ratio:.2f}: {id1[:40]} <> {id2[:40]}")

    # Stage 3: Semantic match (requires ChromaDB)
    print("\n── Stage 3: Semantic Match ──")
    try:
        import chromadb

        chroma_dir = str(BASE / "_chroma")
        client = chromadb.PersistentClient(path=chroma_dir)

        # Check if collection exists and is up to date
        try:
            collection = client.get_collection("questions")
            if collection.count() == len(corpus):
                print(f"  ChromaDB collection up to date ({collection.count()} vectors)")
            else:
                print(f"  ChromaDB stale ({collection.count()} vs {len(corpus)}), rebuilding...")
                client.delete_collection("questions")
                raise ValueError("rebuild")
        except Exception:
            print(f"  Building ChromaDB index for {len(corpus)} questions...")
            collection = client.get_or_create_collection(
                "questions", metadata={"hnsw:space": "cosine"}
            )
            # Batch add
            batch_size = 500
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i : i + batch_size]
                collection.add(
                    ids=[q["id"] for q in batch],
                    documents=[q["scenario"] for q in batch],
                    metadatas=[{"track": q["track"], "level": q["level"]} for q in batch],
                )
            print(f"  Indexed {collection.count()} questions")

        # Query each question for neighbors
        semantic_pairs = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            results = collection.query(
                query_texts=[q["scenario"] for q in batch],
                n_results=3,
            )
            for j, (ids, dists) in enumerate(zip(results["ids"], results["distances"])):
                qid = batch[j]["id"]
                for k, (nid, dist) in enumerate(zip(ids, dists)):
                    sim = 1 - dist  # cosine distance to similarity
                    if nid != qid and sim > threshold:
                        pair = tuple(sorted([qid, nid]))
                        semantic_pairs.append((*pair, sim))

        # Deduplicate pairs
        seen = set()
        unique_pairs = []
        for a, b, sim in sorted(semantic_pairs, key=lambda x: -x[2]):
            if (a, b) not in seen:
                seen.add((a, b))
                unique_pairs.append((a, b, sim))

        print(f"  Semantic duplicate pairs (>{threshold}): {len(unique_pairs)}")
        for a, b, sim in unique_pairs[:10]:
            print(f"    {sim:.3f}: {a[:35]} <> {b[:35]}")

    except ImportError:
        print("  ⚠️ ChromaDB not installed. Run: pip3 install chromadb")
        print("  Skipping semantic dedup.")

    # Summary
    total = len(exact_dupes) + len(fuzzy_pairs) + len(unique_pairs if "unique_pairs" in dir() else [])
    print(f"\n═══ Total flagged: {total} ═══")


# ─── SEARCH ─────────────────────────────────────────────────────
def cmd_search(args):
    """Semantic search across questions."""
    query = args.query
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(BASE / "_chroma"))
        collection = client.get_collection("questions")
        results = collection.query(query_texts=[query], n_results=10)

        print(f"═══ Search: '{query}' ═══\n")
        for qid, dist, meta in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
            sim = 1 - dist
            print(f"  [{sim:.3f}] {meta['track']}/{meta['level']} — {qid[:60]}")
    except Exception as e:
        print(f"Search requires ChromaDB index. Run 'vault.py dedup' first.\n{e}")


# ─── CHAINS ─────────────────────────────────────────────────────
def cmd_chains(args):
    """Build depth chains and verify Bloom's coherence."""
    corpus = load_corpus()
    min_levels = args.min_levels if hasattr(args, "min_levels") and args.min_levels else 3

    topic_key = "canonical_topic" if corpus[0].get("canonical_topic") else "topic"

    # Group by (track, canonical_topic)
    groups = defaultdict(lambda: defaultdict(list))
    for q in corpus:
        if q["track"] in TRACKS:
            groups[(q["track"], q.get(topic_key, q["topic"]))][q["level"]].append(q)

    chains = []
    broken = []
    missing_foundation = []
    missing_depth = []

    for (track, topic), level_map in sorted(groups.items()):
        levels_present = sorted(level_map.keys(), key=lambda l: LEVELS_ORDER.index(l))
        if len(levels_present) < min_levels:
            continue

        # Pick best question per level (longest realistic_solution as proxy for quality)
        chain_questions = []
        for lvl in levels_present:
            best = max(level_map[lvl], key=lambda q: len(q["details"].get("realistic_solution", "")))
            chain_questions.append({
                "level": lvl,
                "id": best["id"],
                "title": best["title"],
                "bloom": best.get("bloom_level", ""),
            })

        chain = {
            "chain_id": f"{track}-{topic}",
            "track": track,
            "topic": topic,
            "competency_area": chain_questions[0].get("competency_area", "")
            if "competency_area" in chain_questions[0]
            else "",
            "levels": levels_present,
            "questions": chain_questions,
        }

        # Get competency_area from the questions
        areas = Counter()
        for lvl_qs in level_map.values():
            for q in lvl_qs:
                areas[q.get("competency_area", "")] += 1
        chain["competency_area"] = areas.most_common(1)[0][0] if areas else ""

        chains.append(chain)

        # Check for missing foundation/depth
        has_foundation = bool(set(levels_present) & {"L1", "L2"})
        has_advanced = bool(set(levels_present) & {"L5", "L6+"})
        if has_advanced and not has_foundation:
            missing_foundation.append(chain)
        if has_foundation and not has_advanced:
            missing_depth.append(chain)

    # Save chains
    chains_path = BASE / "chains.json"
    json.dump(chains, chains_path.open("w"), indent=2)

    print(f"═══ Depth Chains (min {min_levels} levels) ═══\n")
    print(f"  Total chains: {len(chains)}")
    print(f"  Missing foundation (L5+ but no L1/L2): {len(missing_foundation)}")
    print(f"  Missing depth (L1/L2 but no L4+): {len(missing_depth)}")
    print(f"\n  Saved to {chains_path}")

    # Show top 10 chains
    chains.sort(key=lambda ch: -len(ch["levels"]))
    print(f"\n── Top 10 Chains ──")
    for ch in chains[:10]:
        lvls = " → ".join(ch["levels"])
        print(f"  {ch['chain_id']}: {lvls} ({ch['competency_area']})")
        for q in ch["questions"]:
            print(f"    [{q['level']}] {q['title'][:55]}")
        print()

    if missing_foundation:
        print(f"\n── Missing Foundation (first 10) ──")
        for ch in missing_foundation[:10]:
            print(f"  {ch['chain_id']}: has {ch['levels']} but no L1/L2")

    if missing_depth:
        print(f"\n── Missing Depth (first 10) ──")
        for ch in missing_depth[:10]:
            print(f"  {ch['chain_id']}: has {ch['levels']} but no L4+")


# ─── ADD ────────────────────────────────────────────────────────
def cmd_add(args):
    """Add new questions from a JSON file."""
    from schema import validate_corpus

    new_qs = json.loads(Path(args.file).read_text())
    if isinstance(new_qs, dict):
        new_qs = [new_qs]

    print(f"Adding {len(new_qs)} questions...\n")

    # Validate
    valid, errors, _ = validate_corpus(new_qs)
    if errors:
        print(f"❌ {len(errors)} validation errors in input:")
        for err in errors[:20]:
            print(f"  • {err}")
        sys.exit(1)

    # Check for ID conflicts with existing corpus
    corpus = load_corpus()
    existing_ids = {q["id"] for q in corpus}
    conflicts = [q["id"] for q in new_qs if q["id"] in existing_ids]
    if conflicts:
        print(f"❌ {len(conflicts)} ID conflicts with existing corpus:")
        for cid in conflicts[:10]:
            print(f"  • {cid}")
        sys.exit(1)

    # Append and save
    corpus.extend(new_qs)
    save_corpus(corpus)
    print(f"✅ Added {len(new_qs)} questions. Total: {len(corpus)}")


# ─── EXPORT ─────────────────────────────────────────────────────
def cmd_export(args):
    """Generate markdown files and sync to StaffML app."""
    corpus = load_corpus()

    # Sync to StaffML app
    STAFFML_DATA.parent.mkdir(parents=True, exist_ok=True)
    STAFFML_DATA.write_text(json.dumps(corpus, indent=2))
    print(f"✅ Synced {len(corpus)} questions to {STAFFML_DATA}")

    # Generate README stats
    print(f"\n── Corpus Summary ──")
    for t in TRACKS:
        cnt = sum(1 for q in corpus if q["track"] == t)
        print(f"  {t}: {cnt}")
    print(f"  global: {sum(1 for q in corpus if q['track'] == 'global')}")
    print(f"  TOTAL: {len(corpus)}")


# ─── REVIEW ─────────────────────────────────────────────────────
def cmd_review(args):
    """Automated math review via LLM."""
    corpus = load_corpus()
    pub = [q for q in corpus if q.get("status", "published") == "published"]

    # Determine scope
    if hasattr(args, "scope") and args.scope:
        if args.scope.startswith("track:"):
            track = args.scope.split(":")[1]
            questions = [q for q in pub if q["track"] == track]
        else:
            questions = pub
    else:
        questions = pub

    print(f"═══ Math Review ({len(questions)} questions) ═══\n")

    model = CONFIG["models"]["reviewer"]
    chunk_size = CONFIG["review"]["chunk_size"]

    # Chunk questions
    chunks = []
    for i in range(0, len(questions), chunk_size):
        chunks.append(questions[i : i + chunk_size])

    print(f"  Model: {model}")
    print(f"  Chunks: {len(chunks)} × {chunk_size}")

    reviews_dir = BASE / CONFIG["paths"]["reviews_dir"]
    reviews_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_errors = []
    all_ok = 0

    for ci, chunk in enumerate(chunks):
        # Build review text
        review_text = ""
        for q in chunk:
            nm = q.get("details", {}).get("napkin_math", "")[:200]
            sol = q.get("details", {}).get("realistic_solution", "")[:200]
            review_text += f"### {q['title']} [{q['track']}/{q['level']}]\n"
            review_text += f"Scenario: {q['scenario'][:300]}\n"
            review_text += f"Napkin Math: {nm}\n\n"

        prompt = (
            "Review these ML interview questions for math errors. "
            "For each output ONE line: "
            'ERROR | title | error-type | description | fix, or OK | title. '
            "Check arithmetic, hardware specs, units, logical coherence.\n\n"
            + review_text
        )

        report_path = reviews_dir / f"review-{timestamp}-{ci:03d}.txt"

        try:
            result = subprocess.run(
                ["gemini", "-m", model, "-p", prompt, "-o", "text"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            report_path.write_text(result.stdout)

            for line in result.stdout.strip().split("\n"):
                if line.startswith("ERROR"):
                    all_errors.append(line)
                elif line.startswith("OK"):
                    all_ok += 1

            errors_in_chunk = sum(1 for l in result.stdout.split("\n") if l.startswith("ERROR"))
            oks_in_chunk = sum(1 for l in result.stdout.split("\n") if l.startswith("OK"))
            print(f"  [{ci+1}/{len(chunks)}] {errors_in_chunk}E {oks_in_chunk}OK")
        except Exception as e:
            print(f"  [{ci+1}/{len(chunks)}] FAILED: {e}")

    # Summary
    error_rate = len(all_errors) / len(questions) * 100 if questions else 0
    print(f"\n═══ Review Complete ═══")
    print(f"  Reviewed: {len(questions)}")
    print(f"  OK: {all_ok}")
    print(f"  Errors: {len(all_errors)} ({error_rate:.2f}%)")

    # Write structured report
    report = {
        "timestamp": timestamp,
        "model": model,
        "total_reviewed": len(questions),
        "errors": len(all_errors),
        "ok": all_ok,
        "error_rate_pct": round(error_rate, 2),
        "error_details": all_errors,
    }
    report_path = reviews_dir / f"review-{timestamp}-summary.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {report_path}")

    threshold = CONFIG["review"]["error_rate_threshold"] * 100
    if error_rate > threshold:
        print(f"\n  ⚠️ Error rate {error_rate:.2f}% exceeds threshold {threshold}%")
    else:
        print(f"\n  ✅ Error rate {error_rate:.2f}% within threshold {threshold}%")


# ─── FIGURES ────────────────────────────────────────────────────
def cmd_figures(args):
    """Regenerate paper figures from corpus data."""
    paper_dir = BASE / CONFIG["paths"]["paper_dir"]

    print("═══ Generating Figures ═══\n")

    # Step 1: Analyze
    print("  [1/2] Analyzing corpus...")
    result = subprocess.run(
        ["python3", "analyze_corpus.py"],
        cwd=paper_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ❌ analyze_corpus.py failed:\n{result.stderr[:300]}")
        return
    print(f"  {result.stdout.strip()}")

    # Step 2: Generate figures
    print("\n  [2/2] Generating figures...")
    result = subprocess.run(
        ["python3", "generate_figures.py"],
        cwd=paper_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ❌ generate_figures.py failed:\n{result.stderr[:300]}")
        return
    print(f"  {result.stdout.strip()}")

    print("\n✅ Figures regenerated.")


# ─── RELEASE ────────────────────────────────────────────────────
def cmd_release(args):
    """Full pipeline: validate → dedup → chains → stats → figures → export."""
    steps = CONFIG["release"]["steps"]

    if hasattr(args, "skip") and args.skip:
        skip_set = set(args.skip.split(","))
        steps = [s for s in steps if s not in skip_set]

    print("═══ StaffML Release Pipeline ═══")
    print(f"  Steps: {' → '.join(steps)}\n")

    results = {}

    for step in steps:
        print(f"{'─'*50}")
        print(f"  STEP: {step}")
        print(f"{'─'*50}")

        if step == "validate":
            from schema import validate_corpus

            corpus = load_corpus()
            valid, errors, warnings = validate_corpus(corpus)
            results["validate"] = {"errors": len(errors), "warnings": len(warnings)}
            if errors and CONFIG["release"]["require_zero_errors"]:
                print(f"  ❌ BLOCKED: {len(errors)} validation errors")
                for e in errors[:5]:
                    print(f"    • {e}")
                return
            print(f"  ✅ {len(valid)} questions pass ({len(warnings)} warnings)")

        elif step == "dedup":
            # Run dedup inline
            corpus = load_corpus()
            scenarios = defaultdict(list)
            for q in corpus:
                if q.get("status", "published") == "published":
                    key = q["scenario"].lower().strip()
                    scenarios[key].append(q["id"])
            exact = {k: v for k, v in scenarios.items() if len(v) > 1}
            results["dedup"] = {"exact_dupes": len(exact)}
            print(f"  Exact duplicates: {len(exact)}")
            if exact:
                for ids in list(exact.values())[:3]:
                    print(f"    {ids}")

        elif step == "chains":
            # Build chains + backpopulate
            import types

            chain_args = types.SimpleNamespace(min_levels=CONFIG["chains"]["min_levels"])
            cmd_chains(chain_args)

            if CONFIG["chains"]["backpopulate"]:
                # Backpopulate chain_ids
                corpus = load_corpus()
                chains = json.loads(CHAINS_PATH.read_text()) if CHAINS_PATH.exists() else []
                id_to_chains = defaultdict(list)
                for chain in chains:
                    for pos, cq in enumerate(chain["questions"]):
                        id_to_chains[cq["id"]].append((chain["chain_id"], pos))
                linked = 0
                for q in corpus:
                    if q["id"] in id_to_chains:
                        q["chain_ids"] = [cid for cid, _ in id_to_chains[q["id"]]]
                        q["chain_positions"] = {
                            cid: pos for cid, pos in id_to_chains[q["id"]]
                        }
                        linked += 1
                    else:
                        q["chain_ids"] = None
                        q["chain_positions"] = None
                save_corpus(corpus)
                print(f"  Backpopulated chain_ids for {linked} questions")
            results["chains"] = {"total": len(chains) if "chains" in dir() else 0}

        elif step == "stats":
            import types

            stats_args = types.SimpleNamespace()
            cmd_stats(stats_args)
            results["stats"] = {"done": True}

        elif step == "figures":
            import types

            fig_args = types.SimpleNamespace()
            cmd_figures(fig_args)
            results["figures"] = {"done": True}

        elif step == "export":
            import types

            export_args = types.SimpleNamespace()
            cmd_export(export_args)
            results["export"] = {"done": True}

        print()

    # Final summary
    print("═══ Release Summary ═══")
    for step, result in results.items():
        print(f"  {step}: {result}")
    print("\n✅ Release pipeline complete.")


# ─── ANALYZE ───────────────────────────────────────────────────
BLOOM_FOR_LEVEL = {
    "L1": "remember",
    "L2": "understand",
    "L3": "apply",
    "L4": "analyze",
    "L5": "evaluate",
    "L6+": "create",
}


def cmd_analyze(args):
    """Compare taxonomy vs corpus → find untested concepts → output generation plan."""
    corpus = load_corpus()
    taxonomy = json.loads(TAXONOMY_PATH.read_text())

    # Map questions to taxonomy concepts
    concept_qs = defaultdict(list)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc:
            concept_qs[tc].append(q)

    concepts = taxonomy["concepts"]
    total_concepts = len(concepts)

    # Find untested concepts (0 questions)
    untested = [c for c in concepts if c["id"] not in concept_qs]

    # Find partially tested (have some levels but not all)
    incomplete_chains = []
    for c in concepts:
        qs = concept_qs.get(c["id"], [])
        if qs:
            levels = set(q["level"] for q in qs)
            missing = [l for l in LEVELS_ORDER if l not in levels]
            if missing and len(levels) >= 2:
                incomplete_chains.append({"concept": c, "have": sorted(levels), "missing": missing})

    # Type balance per track
    type_balance = defaultdict(lambda: defaultdict(int))
    for q in corpus:
        type_balance[q["track"]][q["level"]] += 1

    # Saturation metrics
    tested_count = len(concept_qs)
    coverage = tested_count / total_concepts if total_concepts else 0

    # Build generation plan — prioritize untested concepts
    max_concepts = args.max_concepts if hasattr(args, "max_concepts") and args.max_concepts else 100
    plan = []

    # Priority 1: untested concepts, mid-range levels first (most useful)
    target_levels = ["L3", "L4", "L5", "L2", "L1", "L6+"]
    for c in untested[:max_concepts]:
        track = c["tracks"][0] if c["tracks"] else "cloud"
        for level in target_levels[:3]:  # L3, L4, L5
            plan.append({
                "concept_id": c["id"],
                "concept_name": c["name"],
                "description": c["description"],
                "prerequisites": c.get("prerequisites", []),
                "tracks": c["tracks"],
                "source_chapters": c.get("source_chapters", []),
                "target_track": track,
                "target_level": level,
                "bloom": BLOOM_FOR_LEVEL[level],
            })

    # Priority 2: incomplete chains — fill missing levels
    for item in incomplete_chains[:50]:
        c = item["concept"]
        track = c["tracks"][0] if c["tracks"] else "cloud"
        for level in item["missing"][:2]:  # fill up to 2 missing levels
            plan.append({
                "concept_id": c["id"],
                "concept_name": c["name"],
                "description": c["description"],
                "prerequisites": c.get("prerequisites", []),
                "tracks": c["tracks"],
                "source_chapters": c.get("source_chapters", []),
                "target_track": track,
                "target_level": level,
                "bloom": BLOOM_FOR_LEVEL[level],
            })

    # Save plan
    plan_path = BASE / "_generation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    print(f"═══ Taxonomy Gap Analysis ═══\n")
    print(f"  Total concepts:      {total_concepts}")
    print(f"  Tested concepts:     {tested_count} ({100*coverage:.0f}%)")
    print(f"  Untested concepts:   {len(untested)}")
    print(f"  Incomplete chains:   {len(incomplete_chains)}")
    print(f"  Coverage:            {coverage:.1%}")
    print(f"\n  Generation plan:     {len(plan)} questions")
    print(f"  Saved to:            {plan_path}")

    if untested:
        print(f"\n── Top 20 Untested Concepts ──")
        for c in untested[:20]:
            tracks = ", ".join(c["tracks"])
            print(f"  {c['id']:40} [{tracks}]")

    return {"coverage": coverage, "untested": len(untested), "plan_size": len(plan)}


# ─── GENERATE ──────────────────────────────────────────────────

def _build_prompt(item: dict) -> str:
    """Build a Gemini prompt for one question generation task."""
    tracks_str = ", ".join(item.get("tracks", ["cloud"]))
    prereqs_str = ", ".join(item.get("prerequisites", [])[:5]) or "none"
    chapters_str = ", ".join(item.get("source_chapters", [])[:3]) or "general"

    # Determine competency area from concept description
    desc = item.get("description", "")

    prompt = f"""You are an expert ML Systems interview question author for the StaffML platform.

Generate exactly ONE interview question as a JSON object. The question must test the concept described below.

## Concept
- **Name**: {item['concept_name']}
- **Description**: {desc}
- **Prerequisites**: {prereqs_str}
- **Source chapters**: {chapters_str}

## Target
- **Track**: {item['target_track']}
- **Level**: {item['target_level']} (Bloom's: {item['bloom']})
- **Tracks this concept appears in**: {tracks_str}

## Level Guidelines
- L1 (Remember): Recall facts, definitions, hardware specs
- L2 (Understand): Explain concepts, compare approaches
- L3 (Apply): Calculate, estimate, use formulas with real numbers
- L4 (Analyze): Diagnose bottlenecks, debug failures, root-cause analysis
- L5 (Evaluate): Compare trade-offs, justify design decisions with quantitative reasoning
- L6+ (Create): Design systems, architect solutions, propose novel approaches

## Requirements
1. scenario: A concrete, real-world scenario (min 50 chars). Use specific hardware specs and numbers.
2. title: Short descriptive title (3-10 words)
3. common_mistake: What most candidates get wrong (min 20 chars)
4. realistic_solution: The correct approach with specific numbers (min 20 chars)
5. napkin_math: Back-of-envelope calculation showing the key insight
6. deep_dive_url: A real URL to a relevant technical resource (paper, docs, blog)
7. competency_area: One of: compute, memory, latency, precision, power, architecture, optimization, parallelism, networking, deployment, reliability, data, cross-cutting

## Output Format
Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "id": "{item['target_track']}-{item['concept_id']}-{item['target_level'].lower()}-0",
  "track": "{item['target_track']}",
  "scope": "Foundations",
  "level": "{item['target_level']}",
  "title": "...",
  "topic": "{item['concept_id']}",
  "competency_area": "...",
  "scenario": "...",
  "details": {{
    "common_mistake": "...",
    "realistic_solution": "...",
    "napkin_math": "...",
    "deep_dive_title": "...",
    "deep_dive_url": "..."
  }},
  "bloom_level": "{item['bloom']}",
  "canonical_topic": "{item['concept_id']}",
  "taxonomy_concept": "{item['concept_id']}"
}}"""
    return prompt


def _generate_one(item: dict, model: str, timeout: int) -> dict | None:
    """Generate one question via Gemini CLI."""
    prompt = _build_prompt(item)
    try:
        result = subprocess.run(
            ["gemini", "-m", model, "-p", prompt, "-o", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  ✗ {item['concept_id']}/{item['target_level']}: gemini error")
            return None

        # Parse JSON from output — strip markdown fences if present
        text = result.stdout.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        q = json.loads(text)

        # Ensure required fields
        if not q.get("scenario") or len(q["scenario"]) < 30:
            print(f"  ✗ {item['concept_id']}/{item['target_level']}: scenario too short")
            return None

        # Ensure taxonomy_concept is set
        q["taxonomy_concept"] = item["concept_id"]
        q["status"] = "published"
        q["version"] = 1
        q["created_at"] = datetime.now().isoformat()

        return q

    except json.JSONDecodeError as e:
        print(f"  ✗ {item['concept_id']}/{item['target_level']}: JSON parse error: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ✗ {item['concept_id']}/{item['target_level']}: timeout")
        return None
    except Exception as e:
        print(f"  ✗ {item['concept_id']}/{item['target_level']}: {e}")
        return None


def cmd_generate(args):
    """Generate questions from _generation_plan.json via Gemini CLI."""
    plan_path = BASE / "_generation_plan.json"
    if not plan_path.exists():
        print("❌ No _generation_plan.json found. Run 'vault.py analyze' first.")
        sys.exit(1)

    plan = json.loads(plan_path.read_text())
    count = args.count if hasattr(args, "count") and args.count else len(plan)
    plan = plan[:count]

    model = CONFIG["models"]["generator"]
    timeout = CONFIG.get("generation", {}).get("timeout_seconds", 180)
    workers = args.workers if hasattr(args, "workers") and args.workers else CONFIG.get("generation", {}).get("max_parallel", 4)

    print(f"═══ Question Generation ═══\n")
    print(f"  Plan items:    {len(plan)}")
    print(f"  Model:         {model}")
    print(f"  Workers:       {workers}")
    print(f"  Timeout:       {timeout}s\n")

    # Generate in parallel
    results = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_one, item, model, timeout): item for item in plan}
        for i, future in enumerate(as_completed(futures)):
            item = futures[future]
            q = future.result()
            if q:
                results.append(q)
                print(f"  ✓ [{len(results)}/{len(plan)}] {item['concept_id']}/{item['target_level']}")
            else:
                failed += 1

    if not results:
        print("\n❌ No questions generated successfully.")
        return {"generated": 0, "failed": failed}

    # Deduplicate IDs (append suffix if collision)
    seen_ids = set()
    for q in results:
        base_id = q["id"]
        suffix = 0
        while q["id"] in seen_ids:
            suffix += 1
            q["id"] = f"{base_id[:-1]}{suffix}"
        seen_ids.add(q["id"])

    # Save batch
    gen_dir = BASE / "_generated"
    gen_dir.mkdir(exist_ok=True)
    batch_file = gen_dir / f"batch-{datetime.now():%Y%m%d-%H%M}.json"
    batch_file.write_text(json.dumps(results, indent=2))

    print(f"\n  Generated: {len(results)}/{len(plan)} ({failed} failed)")
    print(f"  Saved to:  {batch_file}")

    # Auto-add if requested
    if hasattr(args, "auto") and args.auto:
        print(f"\n── Auto-adding to corpus ──")
        _add_batch(results)

    return {"generated": len(results), "failed": failed, "batch_file": str(batch_file)}


def _add_batch(new_qs: list[dict]) -> int:
    """Validate and add a batch of questions to corpus. Returns count added."""
    from schema import validate_corpus

    valid, errors, _ = validate_corpus(new_qs)
    if errors:
        print(f"  ⚠️ {len(errors)} validation errors, filtering to valid only")
        # Keep only valid questions
        valid_ids = {q.id for q in valid}
        new_qs = [q for q in new_qs if q.get("id") in {v.id for v in valid}]
        if not new_qs:
            print(f"  ❌ No valid questions to add")
            return 0

    corpus = load_corpus()
    existing_ids = {q["id"] for q in corpus}

    # Filter out ID conflicts
    to_add = [q for q in new_qs if q["id"] not in existing_ids]
    skipped = len(new_qs) - len(to_add)
    if skipped:
        print(f"  Skipped {skipped} (ID conflict)")

    if to_add:
        corpus.extend(to_add)
        save_corpus(corpus)
        print(f"  ✅ Added {len(to_add)} questions. Total: {len(corpus)}")

    return len(to_add)


# ─── LOOP ──────────────────────────────────────────────────────
def cmd_loop(args):
    """Run analyze→generate→validate→add until saturated."""
    import types

    max_rounds = args.max_rounds if hasattr(args, "max_rounds") and args.max_rounds else 20
    batch_size = args.batch_size if hasattr(args, "batch_size") and args.batch_size else 20
    release_every = 5

    print(f"═══ StaffML Generation Loop ═══")
    print(f"  Max rounds:   {max_rounds}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Release every: {release_every} rounds\n")

    log_entries = []

    for round_num in range(max_rounds):
        round_start = time.time()
        print(f"\n{'═'*50}")
        print(f"  ROUND {round_num + 1}/{max_rounds}")
        print(f"{'═'*50}\n")

        # Step 1: Analyze (plan for more concepts than batch_size so we always have work)
        analyze_args = types.SimpleNamespace(max_concepts=500)
        result = cmd_analyze(analyze_args)

        if result["coverage"] >= 0.95:
            print(f"\n🎯 COVERAGE SATURATED at {result['coverage']:.1%}")
            break

        if result["plan_size"] == 0:
            print(f"\n✅ No more gaps to fill")
            break

        # Step 2: Generate
        gen_args = types.SimpleNamespace(
            count=batch_size,
            workers=CONFIG.get("generation", {}).get("max_parallel", 4),
            auto=False,
        )
        gen_result = cmd_generate(gen_args)

        if gen_result["generated"] == 0:
            print(f"\n❌ Generation produced 0 questions, stopping")
            break

        # Step 3: Validate + Add
        batch_file = gen_result.get("batch_file")
        if batch_file:
            new_qs = json.loads(Path(batch_file).read_text())
            added = _add_batch(new_qs)
        else:
            added = 0

        # Yield check
        yield_rate = added / gen_result["generated"] if gen_result["generated"] else 0
        elapsed = time.time() - round_start

        entry = {
            "round": round_num + 1,
            "coverage": result["coverage"],
            "generated": gen_result["generated"],
            "added": added,
            "yield_rate": yield_rate,
            "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
            "corpus_size": len(load_corpus()),
        }
        log_entries.append(entry)

        # Write progress file for monitoring
        progress_path = BASE / "_loop_progress.json"
        progress_path.write_text(json.dumps(log_entries, indent=2))

        print(f"\n── Round {round_num + 1} Summary ──")
        print(f"  Coverage:  {result['coverage']:.1%}")
        print(f"  Generated: {gen_result['generated']}")
        print(f"  Added:     {added}")
        print(f"  Yield:     {yield_rate:.0%}")
        print(f"  Time:      {elapsed:.0f}s")

        if yield_rate < 0.20 and round_num > 0:
            print(f"\n⚠️ YIELD SATURATED: {yield_rate:.0%} < 20%")
            break

        # Release every N rounds
        if (round_num + 1) % release_every == 0:
            print(f"\n── Intermediate Release ──")
            release_args = types.SimpleNamespace(skip="figures")
            cmd_release(release_args)

    # Final release
    print(f"\n{'═'*50}")
    print(f"  FINAL RELEASE")
    print(f"{'═'*50}\n")
    release_args = types.SimpleNamespace(skip="figures")
    cmd_release(release_args)

    # Write loop log
    log_path = BASE / f"vault_loop_{datetime.now():%Y%m%d-%H%M}.json"
    log_path.write_text(json.dumps(log_entries, indent=2))

    print(f"\n═══ Loop Complete ═══")
    print(f"  Rounds:        {len(log_entries)}")
    total_added = sum(e["added"] for e in log_entries)
    print(f"  Total added:   {total_added}")
    if log_entries:
        print(f"  Final coverage: {log_entries[-1]['coverage']:.1%}")
    print(f"  Log:           {log_path}")


# ─── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="StaffML Vault CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate", help="Schema validation")
    sub.add_parser("stats", help="Full statistics")

    gaps_p = sub.add_parser("gaps", help="Coverage gap analysis")
    gaps_p.add_argument("--min", type=int, default=3, help="Min questions per cell")

    dedup_p = sub.add_parser("dedup", help="Deduplication check")
    dedup_p.add_argument("--threshold", type=float, default=0.85, help="Semantic similarity threshold")

    search_p = sub.add_parser("search", help="Semantic search")
    search_p.add_argument("query", help="Search query")

    chains_p = sub.add_parser("chains", help="Build depth chains")
    chains_p.add_argument("--min-levels", type=int, default=3, help="Min levels per chain")

    add_p = sub.add_parser("add", help="Add questions from file")
    add_p.add_argument("file", help="JSON file with new questions")

    sub.add_parser("export", help="Generate markdown + sync to app")

    review_p = sub.add_parser("review", help="Automated math review")
    review_p.add_argument(
        "--scope",
        default="all",
        help="Scope: all, track:cloud, track:edge, etc.",
    )

    sub.add_parser("figures", help="Regenerate paper figures")

    release_p = sub.add_parser("release", help="Full release pipeline")
    release_p.add_argument(
        "--skip", default="", help="Comma-separated steps to skip"
    )

    analyze_p = sub.add_parser("analyze", help="Taxonomy gap analysis")
    analyze_p.add_argument("--max-concepts", type=int, default=100, help="Max concepts to plan for")

    gen_p = sub.add_parser("generate", help="Generate questions from plan")
    gen_p.add_argument("--count", type=int, default=0, help="Max questions to generate (0=all)")
    gen_p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    gen_p.add_argument("--auto", action="store_true", help="Auto-add passing questions")

    loop_p = sub.add_parser("loop", help="Auto-loop: analyze→generate→add")
    loop_p.add_argument("--max-rounds", type=int, default=20, help="Max generation rounds")
    loop_p.add_argument("--batch-size", type=int, default=20, help="Questions per round")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "validate": cmd_validate,
        "stats": cmd_stats,
        "gaps": cmd_gaps,
        "dedup": cmd_dedup,
        "search": cmd_search,
        "chains": cmd_chains,
        "add": cmd_add,
        "export": cmd_export,
        "review": cmd_review,
        "figures": cmd_figures,
        "release": cmd_release,
        "analyze": cmd_analyze,
        "generate": cmd_generate,
        "loop": cmd_loop,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
