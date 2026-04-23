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
        "resources (≥1)": lambda q: q.get("details", {}).get("resources") or None,
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


# ─── PLAN-ALL ──────────────────────────────────────────────────
def cmd_plan_all(args):
    """Generate per-(level, track) plan files for parallel generation."""
    corpus = load_corpus()
    taxonomy = json.loads(TAXONOMY_PATH.read_text())

    # Build existing coverage map: concept → set of (track, level)
    concept_coverage = defaultdict(set)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc:
            concept_coverage[tc].add((q["track"], q["level"]))

    plans_dir = BASE / "_plans"
    plans_dir.mkdir(exist_ok=True)

    # Priority mode: focus on highest-value work
    priority = args.priority if hasattr(args, "priority") and args.priority else "all"

    # Build per-concept, per-track level sets
    concept_track_levels = defaultdict(lambda: defaultdict(set))
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc:
            concept_track_levels[tc][q["track"]].add(q["level"])

    total_items = 0
    plan_files = []

    for level in ["L1", "L2", "L3", "L4", "L5", "L6+"]:
        for track in TRACKS:
            items = []
            for c in taxonomy["concepts"]:
                if track not in c["tracks"]:
                    continue
                if (track, level) in concept_coverage.get(c["id"], set()):
                    continue  # already covered

                # Apply priority filter
                has_any_in_track = bool(concept_track_levels.get(c["id"], {}).get(track))
                has_any_anywhere = bool(concept_coverage.get(c["id"]))

                if priority == "chains":
                    # P1: Only add L1/L2/L6+ for concepts with existing L3-L5 in THIS track
                    if level in ["L1", "L2", "L6+"] and has_any_in_track:
                        pass  # include
                    elif level in ["L3", "L4", "L5"] and not has_any_anywhere:
                        pass  # P2: untested concepts
                    else:
                        continue
                elif priority == "expand":
                    # P3: Only cross-track expansion
                    if has_any_anywhere and not has_any_in_track:
                        pass  # include
                    else:
                        continue
                # priority == "all" includes everything

                items.append({
                    "concept_id": c["id"],
                    "concept_name": c["name"],
                    "description": c["description"],
                    "prerequisites": c.get("prerequisites", [])[:5],
                    "tracks": c["tracks"],
                    "source_chapters": c.get("source_chapters", [])[:3],
                    "target_track": track,
                    "target_level": level,
                    "bloom": BLOOM_FOR_LEVEL[level],
                })

            if not items:
                continue

            plan_file = plans_dir / f"plan-{level.lower()}-{track}.json"
            plan_file.write_text(json.dumps(items, indent=2))
            plan_files.append({"file": str(plan_file.name), "level": level, "track": track, "count": len(items)})
            total_items += len(items)

    # Write manifest
    manifest = {"total_questions": total_items, "plans": plan_files}
    manifest_path = plans_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"═══ Plan-All: Parallel Generation Plans ═══\n")
    print(f"  Total questions:  {total_items}")
    print(f"  Plan files:       {len(plan_files)}")
    print(f"  Output dir:       {plans_dir}\n")

    print(f"  {'Level':6} {'Track':8} {'Count':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*6}")
    for p in plan_files:
        print(f"  {p['level']:6} {p['track']:8} {p['count']:6}")

    return manifest


def cmd_generate_batch(args):
    """Generate questions from a specific plan file. Writes to _generated/."""
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"❌ Plan file not found: {plan_path}")
        sys.exit(1)

    plan = json.loads(plan_path.read_text())
    if not plan:
        print(f"⚠️ Empty plan: {plan_path}")
        return {"generated": 0, "failed": 0}

    model = CONFIG["models"]["generator"]
    timeout = CONFIG.get("generation", {}).get("timeout_seconds", 180)
    workers = args.workers if hasattr(args, "workers") and args.workers else 4

    # Derive output name from plan filename
    stem = plan_path.stem.replace("plan-", "batch-")
    gen_dir = BASE / "_generated"
    gen_dir.mkdir(exist_ok=True)
    batch_file = gen_dir / f"{stem}-{datetime.now():%Y%m%d-%H%M}.json"

    print(f"═══ Batch Generate: {plan_path.name} ═══")
    print(f"  Items:   {len(plan)}")
    print(f"  Model:   {model}")
    print(f"  Workers: {workers}\n")

    results = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_one, item, model, timeout): item for item in plan}
        for future in as_completed(futures):
            item = futures[future]
            q = future.result()
            if q:
                results.append(q)
                if len(results) % 10 == 0:
                    print(f"  ... {len(results)}/{len(plan)} generated")
            else:
                failed += 1

    # Deduplicate IDs
    seen_ids = set()
    for q in results:
        base_id = q["id"]
        suffix = 0
        while q["id"] in seen_ids:
            suffix += 1
            q["id"] = f"{base_id[:-1]}{suffix}"
        seen_ids.add(q["id"])

    batch_file.write_text(json.dumps(results, indent=2))
    print(f"\n  Generated: {len(results)}/{len(plan)} ({failed} failed)")
    print(f"  Saved to:  {batch_file}")

    return {"generated": len(results), "failed": failed, "batch_file": str(batch_file)}


def cmd_merge(args):
    """Merge all _generated/batch-*.json files into corpus. Validates + dedup."""
    from schema import validate_corpus

    gen_dir = BASE / "_generated"
    pattern = args.pattern if hasattr(args, "pattern") and args.pattern else "batch-*.json"
    batch_files = sorted(gen_dir.glob(pattern))

    if not batch_files:
        print(f"❌ No batch files matching {gen_dir / pattern}")
        sys.exit(1)

    # Load all batches
    all_new = []
    for bf in batch_files:
        qs = json.loads(bf.read_text())
        all_new.extend(qs)
    print(f"═══ Merge: {len(batch_files)} files, {len(all_new)} questions ═══\n")

    # Validate
    valid, errors, _ = validate_corpus(all_new)
    valid_ids = {q.id for q in valid}
    passing = [q for q in all_new if q.get("id") in valid_ids]
    print(f"  Valid:   {len(passing)}/{len(all_new)}")
    if errors:
        print(f"  Errors:  {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")

    # Check ID conflicts with existing corpus
    corpus = load_corpus()
    existing_ids = {q["id"] for q in corpus}
    to_add = [q for q in passing if q["id"] not in existing_ids]
    skipped = len(passing) - len(to_add)

    # Also dedup within the new batch itself
    seen = set()
    unique = []
    for q in to_add:
        if q["id"] not in seen:
            seen.add(q["id"])
            unique.append(q)
    intra_dupes = len(to_add) - len(unique)

    print(f"  ID conflicts: {skipped}")
    print(f"  Intra-dupes:  {intra_dupes}")
    print(f"  To add:       {len(unique)}")

    if unique and not (hasattr(args, "dry_run") and args.dry_run):
        corpus.extend(unique)
        save_corpus(corpus)
        print(f"\n  ✅ Added {len(unique)} questions. Total: {len(corpus)}")
    elif hasattr(args, "dry_run") and args.dry_run:
        print(f"\n  🔍 Dry run — no changes made")

    return {"added": len(unique), "valid": len(passing), "total": len(all_new)}


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
6. resources (optional): 0–3 author-style references. Each is {{"name": "<label>", "url": "https://..."}}. Prefer seminal papers (arXiv), framework docs (PyTorch, HuggingFace), or engineering blogs. Do NOT include mlsysbook.ai links — book linking is deferred. Omit or empty-list when no relevant reference exists.
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
    "resources": []
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


# ─── TAXONOMY-CHECK ────────────────────────────────────────────
def _tarjan_sccs(adj: dict, nodes: set) -> list[list[str]]:
    """Tarjan's SCC algorithm. Returns SCCs with >1 node (cycles)."""
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    # Use explicit stack to avoid RecursionError on deep graphs
    sys.setrecursionlimit(max(10000, len(nodes) * 2))
    for node in sorted(nodes):
        if node not in index:
            strongconnect(node)

    return sccs


def cmd_taxonomy_check(args):
    """Read-only diagnostic of taxonomy health."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    corpus = load_corpus()

    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}

    p0_count = 0  # Critical issues
    warn_count = 0  # Warnings

    print("═══ Taxonomy Health Check ═══\n")

    # ── 1. Self-references ─────────────────────────────────────
    self_refs = [c["id"] for c in concepts if c["id"] in c.get("prerequisites", [])]
    if self_refs:
        p0_count += len(self_refs)
        print(f"  ❌ Self-references: {len(self_refs)}")
        for s in self_refs:
            print(f"     {s}")
    else:
        print("  ✅ Self-references: 0")

    # ── 2. Dangling prereq refs (concept.prerequisites → missing ID) ──
    dangling_prereqs = set()
    dangling_detail = defaultdict(list)
    for c in concepts:
        for p in c.get("prerequisites", []):
            if p not in all_ids:
                dangling_prereqs.add(p)
                dangling_detail[p].append(c["id"])
    if dangling_prereqs:
        p0_count += len(dangling_prereqs)
        print(f"  ❌ Dangling prereq refs: {len(dangling_prereqs)}")
        for d in sorted(dangling_prereqs)[:15]:
            refs = ", ".join(dangling_detail[d][:3])
            print(f"     {d}  (referenced by: {refs})")
        if len(dangling_prereqs) > 15:
            print(f"     ... and {len(dangling_prereqs) - 15} more")
    else:
        print("  ✅ Dangling prereq refs: 0")

    # ── 3. Dangling edge refs ──────────────────────────────────
    dangling_edges = [e for e in edges if e["source"] not in all_ids or e["target"] not in all_ids]
    if dangling_edges:
        p0_count += len(dangling_edges)
        print(f"  ❌ Dangling edges: {len(dangling_edges)}")
        for e in dangling_edges[:5]:
            print(f"     {e['source']} -> {e['target']}")
    else:
        print("  ✅ Dangling edges: 0")

    # ── 4. Bidirectional edges (A→B and B→A) ───────────────────
    edge_set = set()
    bidir = []
    for e in edges:
        pair = (e["source"], e["target"])
        reverse = (e["target"], e["source"])
        if reverse in edge_set:
            bidir.append(pair)
        edge_set.add(pair)
    if bidir:
        p0_count += len(bidir)
        print(f"  ❌ Bidirectional edge pairs: {len(bidir)}")
        for b in bidir:
            print(f"     {b[0]} ↔ {b[1]}")
    else:
        print("  ✅ Bidirectional edges: 0")

    # ── 5. Cycles (Tarjan's SCC) ──────────────────────────────
    adj = defaultdict(list)
    for e in edges:
        if e["source"] in all_ids and e["target"] in all_ids:
            adj[e["source"]].append(e["target"])
    sccs = _tarjan_sccs(adj, all_ids)
    if sccs:
        p0_count += len(sccs)
        print(f"  ❌ Cycles (SCCs): {len(sccs)}")
        for scc in sccs:
            print(f"     {' → '.join(scc)}")
    else:
        print("  ✅ Cycles: 0")

    # ── 6. Stale taxonomy_concept in corpus ────────────────────
    stale = []
    stale_ids = defaultdict(int)
    unmapped = []
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc and tc not in all_ids:
            stale.append(q["id"])
            stale_ids[tc] += 1
        elif not tc:
            unmapped.append(q["id"])
    if stale:
        p0_count += 1  # Count as one P0 issue
        print(f"  ❌ Stale taxonomy_concept mappings: {len(stale)} questions → {len(stale_ids)} invalid IDs")
        for tc_id, cnt in sorted(stale_ids.items(), key=lambda x: -x[1])[:10]:
            print(f"     {tc_id}: {cnt} Qs")
        if len(stale_ids) > 10:
            print(f"     ... and {len(stale_ids) - 10} more invalid IDs")
    else:
        print("  ✅ Stale taxonomy_concept mappings: 0")

    if unmapped:
        warn_count += 1
        print(f"  ⚠️  Unmapped questions (no taxonomy_concept): {len(unmapped)}")
    else:
        print("  ✅ Unmapped questions: 0")

    # ── 7. Over-represented concepts (>30 Qs) ─────────────────
    concept_qcount = defaultdict(int)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc and tc in all_ids:
            concept_qcount[tc] += 1
    over = [(cid, cnt) for cid, cnt in concept_qcount.items() if cnt > 30]
    over.sort(key=lambda x: -x[1])
    if over:
        warn_count += len(over)
        print(f"  ⚠️  Over-represented concepts (>30 Qs): {len(over)}")
        for cid, cnt in over[:10]:
            print(f"     {cid}: {cnt} Qs")
        if len(over) > 10:
            print(f"     ... and {len(over) - 10} more")
    else:
        print("  ✅ Over-represented concepts: 0")

    # ── 8. Graph shape ─────────────────────────────────────────
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for e in edges:
        if e["source"] in all_ids and e["target"] in all_ids:
            incoming[e["target"]] += 1
            outgoing[e["source"]] += 1
    roots = [cid for cid in all_ids if incoming[cid] == 0]
    leaves = [cid for cid in all_ids if outgoing[cid] == 0]

    # Compute depth via BFS from roots
    depth = {r: 0 for r in roots}
    queue = list(roots)
    max_depth = 0
    visited = set(roots)
    while queue:
        node = queue.pop(0)
        for e in edges:
            if e["source"] == node and e["target"] in all_ids and e["target"] not in visited:
                depth[e["target"]] = depth[node] + 1
                max_depth = max(max_depth, depth[e["target"]])
                visited.add(e["target"])
                queue.append(e["target"])

    # Connected components (undirected)
    undirected = defaultdict(set)
    for e in edges:
        if e["source"] in all_ids and e["target"] in all_ids:
            undirected[e["source"]].add(e["target"])
            undirected[e["target"]].add(e["source"])
    comp_visited = set()
    components = 0
    for node in all_ids:
        if node not in comp_visited:
            components += 1
            stack = [node]
            while stack:
                n = stack.pop()
                if n not in comp_visited:
                    comp_visited.add(n)
                    stack.extend(undirected[n] - comp_visited)

    print(f"\n── Graph Shape ──")
    print(f"  Concepts:     {len(all_ids)}")
    print(f"  Edges:        {len(edges)}")
    print(f"  Roots:        {len(roots)} ({len(roots)/len(all_ids):.1%})")
    print(f"  Leaves:       {len(leaves)} ({len(leaves)/len(all_ids):.1%})")
    print(f"  Intermediates:{len(all_ids) - len(roots) - len(leaves)}")
    print(f"  Max depth:    {max_depth}")
    print(f"  Components:   {components}")

    # ── 9. Question coverage ──────────────────────────────────
    tested = sum(1 for cid in all_ids if concept_qcount.get(cid, 0) > 0)
    print(f"\n── Question Coverage ──")
    print(f"  Tested concepts:   {tested}/{len(all_ids)} ({tested/len(all_ids):.1%})")
    print(f"  Untested concepts: {len(all_ids) - tested}")
    print(f"  Mapped questions:  {len(corpus) - len(unmapped) - len(stale)}")
    print(f"  Stale mappings:    {len(stale)}")
    print(f"  Unmapped:          {len(unmapped)}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'═' * 40}")
    if p0_count == 0:
        print(f"  ✅ P0 issues: 0 — taxonomy is clean!")
    else:
        print(f"  ❌ P0 issues: {p0_count} — run `vault.py taxonomy-fix` to repair")
    if warn_count > 0:
        print(f"  ⚠️  Warnings: {warn_count}")
    print(f"{'═' * 40}")

    return {"p0": p0_count, "warnings": warn_count}


# ─── TAXONOMY-FIX ──────────────────────────────────────────────
def cmd_taxonomy_fix(args):
    """Automated repair of P0 taxonomy issues."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    corpus = load_corpus()

    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}
    fixes = []

    print("═══ Taxonomy Fix ═══\n")

    # ── 1. Remove self-referential prerequisite edges ──────────
    for c in concepts:
        prereqs = c.get("prerequisites", [])
        if c["id"] in prereqs:
            c["prerequisites"] = [p for p in prereqs if p != c["id"]]
            fixes.append(f"Removed self-ref: {c['id']}")
    # Also remove self-ref edges
    orig_edge_count = len(edges)
    edges = [e for e in edges if e["source"] != e["target"]]
    removed_self = orig_edge_count - len(edges)
    if removed_self:
        fixes.append(f"Removed {removed_self} self-referential edge(s)")
    print(f"  [1] Self-references removed: {len([f for f in fixes if 'self-ref' in f.lower()])}")

    # ── 2. Remove dangling prereq refs ─────────────────────────
    dangling_removed = 0
    for c in concepts:
        prereqs = c.get("prerequisites", [])
        valid = [p for p in prereqs if p in all_ids]
        removed = len(prereqs) - len(valid)
        if removed:
            c["prerequisites"] = valid
            dangling_removed += removed
    if dangling_removed:
        fixes.append(f"Removed {dangling_removed} dangling prereq refs from concepts")
    print(f"  [2] Dangling prereq refs removed: {dangling_removed}")

    # ── 3. Remove dangling edges ───────────────────────────────
    before = len(edges)
    edges = [e for e in edges if e["source"] in all_ids and e["target"] in all_ids]
    dangling_edge_removed = before - len(edges)
    if dangling_edge_removed:
        fixes.append(f"Removed {dangling_edge_removed} dangling edge(s)")
    print(f"  [3] Dangling edges removed: {dangling_edge_removed}")

    # ── 4. Break bidirectional edges ───────────────────────────
    # Strategy: for each A↔B pair, keep the pedagogically correct direction.
    # Heuristic: keep edge where source has fewer prerequisites (more foundational).
    prereq_count = {}
    for c in concepts:
        prereq_count[c["id"]] = len(c.get("prerequisites", []))

    edge_set = set()
    bidir_pairs = []
    for e in edges:
        pair = (e["source"], e["target"])
        reverse = (e["target"], e["source"])
        if reverse in edge_set:
            bidir_pairs.append(pair)
        edge_set.add(pair)

    removed_bidir = set()
    for a, b in bidir_pairs:
        # Keep direction: more foundational (fewer prereqs) → more advanced
        a_prereqs = prereq_count.get(a, 0)
        b_prereqs = prereq_count.get(b, 0)
        if a_prereqs <= b_prereqs:
            # A is more foundational, keep A→B, remove B→A
            removed_bidir.add((b, a))
            fixes.append(f"Broke bidir: kept {a} → {b}, removed {b} → {a}")
        else:
            removed_bidir.add((a, b))
            fixes.append(f"Broke bidir: kept {b} → {a}, removed {a} → {b}")
    edges = [e for e in edges if (e["source"], e["target"]) not in removed_bidir]
    print(f"  [4] Bidirectional edges broken: {len(bidir_pairs)}")

    # ── 5. Break remaining cycles ──────────────────────────────
    # Rebuild adjacency and find SCCs. For each cycle, remove the edge
    # whose target has the fewest prerequisites (weakest dependency).
    cycle_breaks = 0
    for _pass in range(10):  # Max 10 passes
        adj = defaultdict(list)
        for e in edges:
            adj[e["source"]].append(e["target"])
        sccs = _tarjan_sccs(adj, all_ids)
        if not sccs:
            break
        for scc in sccs:
            # Find the weakest edge in this SCC to remove
            scc_set = set(scc)
            scc_edges = [(e["source"], e["target"]) for e in edges
                         if e["source"] in scc_set and e["target"] in scc_set]
            if not scc_edges:
                continue
            # Remove edge whose target has fewest prereqs (least dependent)
            weakest = min(scc_edges, key=lambda x: prereq_count.get(x[1], 0))
            edges = [e for e in edges if not (e["source"] == weakest[0] and e["target"] == weakest[1])]
            cycle_breaks += 1
            fixes.append(f"Broke cycle: removed {weakest[0]} → {weakest[1]}")
    print(f"  [5] Cycle-breaking edges removed: {cycle_breaks}")

    # ── 6. Fuzzy-match stale taxonomy_concept IDs ──────────────
    stale_qs = []
    stale_ids = set()
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc and tc not in all_ids:
            stale_qs.append(q)
            stale_ids.add(tc)

    # Build fuzzy match candidates
    remapped = 0
    failed_remap = 0
    remap_log = []
    for stale_id in sorted(stale_ids):
        # Try exact substring matches first
        best_match = None
        best_score = 0.0
        for valid_id in all_ids:
            score = SequenceMatcher(None, stale_id, valid_id).ratio()
            if score > best_score:
                best_score = score
                best_match = valid_id

        if best_score >= 0.70:
            # Apply remap
            count = 0
            for q in corpus:
                if q.get("taxonomy_concept") == stale_id:
                    q["taxonomy_concept"] = best_match
                    count += 1
            remapped += count
            remap_log.append(f"  {stale_id} → {best_match} ({count} Qs, score={best_score:.2f})")
        else:
            # Clear the invalid mapping rather than leave it broken
            count = 0
            for q in corpus:
                if q.get("taxonomy_concept") == stale_id:
                    q["taxonomy_concept"] = ""
                    count += 1
            failed_remap += count
            remap_log.append(f"  {stale_id} → CLEARED ({count} Qs, best={best_match} score={best_score:.2f})")

    print(f"  [6] Stale mappings remapped: {remapped}")
    print(f"      Stale mappings cleared:  {failed_remap}")
    if hasattr(args, "verbose") and args.verbose and remap_log:
        for line in remap_log[:20]:
            print(f"     {line}")

    # ── 7. Sync concept.prerequisites with edges ──────────────
    # Ensure edges and concept.prerequisites are consistent
    edge_prereqs = defaultdict(set)
    for e in edges:
        edge_prereqs[e["target"]].add(e["source"])

    synced = 0
    for c in concepts:
        new_prereqs = sorted(edge_prereqs.get(c["id"], set()))
        if c.get("prerequisites", []) != new_prereqs:
            c["prerequisites"] = new_prereqs
            synced += 1
    if synced:
        fixes.append(f"Synced prerequisites for {synced} concepts to match edges")
    print(f"  [7] Concepts re-synced with edges: {synced}")

    # ── Save ───────────────────────────────────────────────────
    taxonomy["edges"] = edges
    taxonomy["stats"]["total_edges"] = len(edges)
    TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2))
    save_corpus(corpus)

    print(f"\n  Total fixes: {len(fixes)}")
    print(f"  Saved: {TAXONOMY_PATH.name}, {CORPUS_PATH.name}")

    # ── Verify ─────────────────────────────────────────────────
    print(f"\n── Verify ──")
    # Quick re-check
    result = cmd_taxonomy_check(args)
    return result


# ─── TAXONOMY-IMPROVE ──────────────────────────────────────────

IMPROVE_ROUNDS = {
    "toc-validate": {
        "desc": "Check textbook TOC against taxonomy for missing concepts",
        "prompt_template": """You are an expert ML Systems curriculum designer.

TASK: Compare this textbook chapter's table of contents against the existing taxonomy concepts and identify MISSING concepts.

CHAPTER: {chapter}
CHAPTER SECTIONS:
{sections}

EXISTING CONCEPTS for this chapter:
{existing}

RULES:
1. Only propose concepts that represent distinct, testable engineering knowledge
2. Each concept must have a unique kebab-case ID
3. Each concept must have at least one prerequisite from existing concepts
4. Do NOT propose concepts that are just rewordings of existing ones
5. Focus on concepts a Staff ML engineer would need to know

Return a JSON array of proposed new concepts. Each object must have:
- "id": kebab-case unique identifier
- "name": Human-readable name
- "description": 1-2 sentence description
- "prerequisites": array of existing concept IDs this depends on
- "tracks": array from ["cloud", "edge", "mobile", "tinyml"]
- "source_chapters": ["{chapter}"]
- "rationale": Why this concept is missing and important

If no concepts are missing, return an empty array: []

Return ONLY the JSON array, no markdown fences or other text.""",
    },
    "split-overloaded": {
        "desc": "Break up concepts with >50 questions into sub-concepts",
        "prompt_template": """You are an expert ML Systems taxonomy designer.

TASK: This concept has {question_count} questions, which is too many for a single concept.
Break it into 3-7 more specific sub-concepts.

CONCEPT: {concept_name} ({concept_id})
DESCRIPTION: {description}
CURRENT PREREQUISITES: {prerequisites}

SAMPLE QUESTION TITLES (showing breadth):
{sample_titles}

RULES:
1. Each sub-concept should cover a distinct facet of the parent concept
2. Sub-concepts should have prerequisite edges between them where appropriate
3. Existing questions should clearly map to exactly one sub-concept
4. Use the parent concept ID as a prefix: "{concept_id}-<suffix>"
5. The parent concept becomes a "hub" that points to all sub-concepts

Return a JSON array of proposed sub-concepts. Each object must have:
- "id": kebab-case ID (prefixed with parent)
- "name": Human-readable name
- "description": 1-2 sentence description
- "prerequisites": array of concept IDs (can include other sub-concepts and existing concepts)
- "tracks": {tracks}
- "source_chapters": {source_chapters}
- "parent_id": "{concept_id}"
- "question_filter": keywords/patterns to match questions belonging to this sub-concept

Return ONLY the JSON array, no markdown fences or other text.""",
    },
    "validate-edges": {
        "desc": "Validate every prerequisite edge in the taxonomy",
        "prompt_template": """You are an expert ML Systems pedagogy reviewer.

TASK: Evaluate whether these prerequisite edges are correct.
A prerequisite edge A → B means "you must understand A before learning B."

EDGES TO VALIDATE:
{edges}

For each edge, return:
- "edge": "source → target"
- "valid": true/false
- "reason": brief explanation
- "fix": null if valid, or {{"action": "remove"}} or {{"action": "reverse"}}

Return ONLY a JSON array of validation results.""",
    },
    "deepen-prereqs": {
        "desc": "Propose prerequisite chains for leaf concepts with 0 prereqs",
        "prompt_template": """You are an expert ML Systems curriculum designer.

TASK: These concepts have NO prerequisites, meaning they appear as root/leaf nodes with no incoming edges.
For each, propose 1-3 prerequisite concepts that already exist in the taxonomy.

CONCEPTS WITHOUT PREREQUISITES:
{concepts}

AVAILABLE CONCEPTS (full list):
{all_concepts}

RULES:
1. Only propose prerequisites from the AVAILABLE CONCEPTS list
2. A prerequisite must genuinely be needed to understand the target concept
3. Don't create circular dependencies

Return a JSON array. Each object must have:
- "concept_id": the concept that needs prerequisites
- "proposed_prereqs": array of existing concept IDs
- "rationale": why each prerequisite is needed

Return ONLY the JSON array, no markdown fences or other text.""",
    },
}


def _call_gemini(prompt: str, model: str, timeout: int = 180) -> str | None:
    """Call Gemini CLI and return output text, or None on failure."""
    try:
        result = subprocess.run(
            ["gemini", "-m", model, "-p", prompt, "-o", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        text = result.stdout.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
    except (subprocess.TimeoutExpired, Exception):
        return None


def _parse_json_response(text: str) -> list | None:
    """Parse JSON array from Gemini response."""
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


def cmd_taxonomy_improve(args):
    """Run a taxonomy improvement round using Gemini."""
    round_name = args.round
    if round_name not in IMPROVE_ROUNDS:
        print(f"❌ Unknown round: {round_name}")
        print(f"   Available: {', '.join(IMPROVE_ROUNDS.keys())}")
        sys.exit(1)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    corpus = load_corpus()
    concepts = taxonomy["concepts"]
    all_ids = {c["id"] for c in concepts}
    concept_map = {c["id"]: c for c in concepts}
    model = CONFIG["models"].get("extractor", "gemini-2.5-flash")
    count = args.count if hasattr(args, "count") and args.count else 0

    round_info = IMPROVE_ROUNDS[round_name]
    print(f"═══ Taxonomy Improve: {round_name} ═══")
    print(f"  {round_info['desc']}")
    print(f"  Model: {model}")

    proposals = []

    if round_name == "toc-validate":
        proposals = _improve_toc_validate(taxonomy, corpus, model, count)
    elif round_name == "split-overloaded":
        proposals = _improve_split_overloaded(taxonomy, corpus, model, count)
    elif round_name == "validate-edges":
        proposals = _improve_validate_edges(taxonomy, model, count)
    elif round_name == "deepen-prereqs":
        proposals = _improve_deepen_prereqs(taxonomy, model, count)

    if not proposals:
        print("\n  No proposals generated.")
        return

    # Save proposals
    prop_dir = BASE / "_taxonomy_proposals"
    prop_dir.mkdir(exist_ok=True)
    prop_file = prop_dir / f"{round_name}-{datetime.now():%Y%m%d-%H%M}.json"
    prop_file.write_text(json.dumps(proposals, indent=2))

    print(f"\n  Proposals: {len(proposals)}")
    print(f"  Saved to:  {prop_file}")
    print(f"\n  Review proposals, then run: vault.py taxonomy-apply {prop_file}")


def _improve_toc_validate(taxonomy, corpus, model, count):
    """Round: toc-validate — check textbook TOC against taxonomy."""
    concepts = taxonomy["concepts"]
    # Group concepts by chapter
    by_chapter = defaultdict(list)
    for c in concepts:
        for ch in c.get("source_chapters", []):
            by_chapter[ch].append(c)

    chapters = sorted(by_chapter.keys())
    if count:
        chapters = chapters[:count]

    # Try to find textbook QMD files for section headings
    book_dir = BASE.parent / "book" / "quarto" / "contents"
    all_proposals = []

    for ch in chapters:
        print(f"\n  Checking: {ch}")

        # Find the QMD file for this chapter
        sections_text = "(section headers not available — analyze based on concept names)"
        ch_short = ch.replace("vol1_", "").replace("vol2_", "")
        vol = "vol1" if "vol1" in ch else "vol2"

        # Search for the QMD file
        qmd_candidates = list((book_dir / vol).glob(f"**/{ch_short}*/*.qmd")) + \
                         list((book_dir / vol).glob(f"**/{ch_short}*.qmd"))
        if qmd_candidates:
            # Extract ## and ### headings
            qmd_text = qmd_candidates[0].read_text(errors='ignore')
            headings = [line.strip() for line in qmd_text.split('\n')
                       if line.strip().startswith('## ') or line.strip().startswith('### ')]
            if headings:
                sections_text = '\n'.join(headings[:30])

        existing_text = '\n'.join(f"- {c['id']}: {c['name']} — {c['description'][:80]}"
                                  for c in by_chapter[ch])

        prompt = IMPROVE_ROUNDS["toc-validate"]["prompt_template"].format(
            chapter=ch,
            sections=sections_text,
            existing=existing_text,
        )

        text = _call_gemini(prompt, model)
        result = _parse_json_response(text)

        if result:
            for p in result:
                p["round"] = "toc-validate"
                p["source_chapter"] = ch
            all_proposals.extend(result)
            print(f"    → {len(result)} proposals")
        else:
            print(f"    → 0 proposals (parse failed)")

    return all_proposals


def _improve_split_overloaded(taxonomy, corpus, model, count):
    """Round: split-overloaded — break up concepts with too many questions."""
    concepts = taxonomy["concepts"]
    concept_qs = defaultdict(list)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc:
            concept_qs[tc].append(q)

    # Find overloaded concepts (>50 Qs)
    overloaded = [(c, len(concept_qs[c["id"]])) for c in concepts
                  if len(concept_qs.get(c["id"], [])) > 50]
    overloaded.sort(key=lambda x: -x[1])

    if count:
        overloaded = overloaded[:count]

    all_proposals = []
    for c, qcount in overloaded:
        print(f"\n  Splitting: {c['id']} ({qcount} Qs)")

        # Sample question titles for context
        qs = concept_qs[c["id"]]
        titles = sorted(set(q["title"] for q in qs))
        sample = titles[:20]

        prompt = IMPROVE_ROUNDS["split-overloaded"]["prompt_template"].format(
            question_count=qcount,
            concept_name=c["name"],
            concept_id=c["id"],
            description=c["description"],
            prerequisites=json.dumps(c.get("prerequisites", [])),
            sample_titles='\n'.join(f"- {t}" for t in sample),
            tracks=json.dumps(c.get("tracks", [])),
            source_chapters=json.dumps(c.get("source_chapters", [])),
        )

        text = _call_gemini(prompt, model, timeout=240)
        result = _parse_json_response(text)

        if result:
            for p in result:
                p["round"] = "split-overloaded"
            all_proposals.extend(result)
            print(f"    → {len(result)} sub-concepts proposed")
        else:
            print(f"    → parse failed")

    return all_proposals


def _improve_validate_edges(taxonomy, model, count):
    """Round: validate-edges — check if prerequisite edges are correct."""
    edges = taxonomy["edges"]
    concept_map = {c["id"]: c for c in taxonomy["concepts"]}

    # Batch edges into groups of 20 for efficiency
    edge_texts = []
    for e in edges:
        src = concept_map.get(e["source"], {})
        tgt = concept_map.get(e["target"], {})
        edge_texts.append(f"- {e['source']} ({src.get('name', '?')}) → {e['target']} ({tgt.get('name', '?')})")

    if count:
        edge_texts = edge_texts[:count]

    batch_size = 20
    all_proposals = []

    for i in range(0, len(edge_texts), batch_size):
        batch = edge_texts[i:i + batch_size]
        print(f"\n  Validating edges {i+1}-{i+len(batch)} of {len(edge_texts)}")

        prompt = IMPROVE_ROUNDS["validate-edges"]["prompt_template"].format(
            edges='\n'.join(batch),
        )

        text = _call_gemini(prompt, model)
        result = _parse_json_response(text)

        if result:
            for p in result:
                p["round"] = "validate-edges"
            all_proposals.extend(result)
            invalid = sum(1 for p in result if not p.get("valid", True))
            print(f"    → {len(result)} validated, {invalid} invalid")
        else:
            print(f"    → parse failed")

    return all_proposals


def _improve_deepen_prereqs(taxonomy, model, count):
    """Round: deepen-prereqs — add prereq chains to orphaned concepts."""
    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}

    # Find concepts with no incoming edges (no prereqs via edges)
    has_incoming = set()
    for e in edges:
        if e["target"] in all_ids:
            has_incoming.add(e["target"])

    orphans = [c for c in concepts if c["id"] not in has_incoming and c["id"] in all_ids]

    if count:
        orphans = orphans[:count]

    print(f"\n  Orphaned concepts (no prereqs): {len(orphans)}")

    if not orphans:
        return []

    # Batch into groups of 15
    batch_size = 15
    all_proposals = []
    all_concept_names = '\n'.join(f"- {c['id']}: {c['name']}" for c in concepts)

    for i in range(0, len(orphans), batch_size):
        batch = orphans[i:i + batch_size]
        print(f"\n  Processing batch {i+1}-{i+len(batch)}")

        concepts_text = '\n'.join(f"- {c['id']}: {c['name']} — {c['description'][:80]}"
                                   for c in batch)

        prompt = IMPROVE_ROUNDS["deepen-prereqs"]["prompt_template"].format(
            concepts=concepts_text,
            all_concepts=all_concept_names,
        )

        text = _call_gemini(prompt, model, timeout=240)
        result = _parse_json_response(text)

        if result:
            for p in result:
                p["round"] = "deepen-prereqs"
            all_proposals.extend(result)
            print(f"    → {len(result)} proposals")
        else:
            print(f"    → parse failed")

    return all_proposals


def cmd_taxonomy_apply(args):
    """Apply accepted proposals from a taxonomy-improve round."""
    prop_path = Path(args.file)
    if not prop_path.exists():
        print(f"❌ File not found: {prop_path}")
        sys.exit(1)

    proposals = json.loads(prop_path.read_text())
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}

    round_name = proposals[0].get("round", "unknown") if proposals else "unknown"
    print(f"═══ Taxonomy Apply: {round_name} ═══\n")
    print(f"  Proposals: {len(proposals)}")

    added_concepts = 0
    added_edges = 0
    removed_edges = 0
    reversed_edges = 0

    if round_name in ("toc-validate", "split-overloaded"):
        # Proposals are new concepts to add
        for p in proposals:
            cid = p.get("id", "")
            if not cid or cid in all_ids:
                continue

            new_concept = {
                "id": cid,
                "name": p.get("name", cid),
                "description": p.get("description", ""),
                "prerequisites": [pid for pid in p.get("prerequisites", []) if pid in all_ids],
                "tracks": p.get("tracks", []),
                "source_chapters": p.get("source_chapters", []),
                "question_count": 0,
            }
            concepts.append(new_concept)
            all_ids.add(cid)
            added_concepts += 1

            # Add edges from prerequisites
            for pid in new_concept["prerequisites"]:
                edges.append({"source": pid, "target": cid, "type": "prerequisite"})
                added_edges += 1

    elif round_name == "validate-edges":
        # Proposals are edge validations
        for p in proposals:
            if p.get("valid", True):
                continue
            fix = p.get("fix", {})
            if not fix:
                continue

            # Parse edge string: "source → target"
            edge_str = p.get("edge", "")
            parts = edge_str.split(" → ")
            if len(parts) != 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()

            if fix.get("action") == "remove":
                edges = [e for e in edges if not (e["source"] == src and e["target"] == tgt)]
                removed_edges += 1
            elif fix.get("action") == "reverse":
                edges = [e for e in edges if not (e["source"] == src and e["target"] == tgt)]
                edges.append({"source": tgt, "target": src, "type": "prerequisite"})
                reversed_edges += 1

    elif round_name == "deepen-prereqs":
        # Proposals are new edges for existing concepts
        for p in proposals:
            cid = p.get("concept_id", "")
            if cid not in all_ids:
                continue
            for pid in p.get("proposed_prereqs", []):
                if pid in all_ids and pid != cid:
                    # Check if edge already exists
                    exists = any(e["source"] == pid and e["target"] == cid for e in edges)
                    if not exists:
                        edges.append({"source": pid, "target": cid, "type": "prerequisite"})
                        added_edges += 1

    # Save
    taxonomy["concepts"] = concepts
    taxonomy["edges"] = edges
    taxonomy["stats"]["total_concepts"] = len(concepts)
    taxonomy["stats"]["total_edges"] = len(edges)
    TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2))

    print(f"  Added concepts: {added_concepts}")
    print(f"  Added edges:    {added_edges}")
    print(f"  Removed edges:  {removed_edges}")
    print(f"  Reversed edges: {reversed_edges}")
    print(f"  Saved: {TAXONOMY_PATH.name}")

    # Run check to verify
    print(f"\n── Verify ──")
    cmd_taxonomy_check(args)


# ─── TAXONOMY-SYNC ─────────────────────────────────────────────
def cmd_taxonomy_sync(args):
    """Export enriched taxonomy to staffml/src/data/taxonomy.json."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    corpus = load_corpus()

    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}

    # Verify cycle-free
    adj = defaultdict(list)
    for e in edges:
        if e["source"] in all_ids and e["target"] in all_ids:
            adj[e["source"]].append(e["target"])
    sccs = _tarjan_sccs(adj, all_ids)
    if sccs:
        print(f"❌ Cannot sync: {len(sccs)} cycles detected. Run taxonomy-fix first.")
        sys.exit(1)

    # Compute question counts and level distribution per concept
    concept_qs = defaultdict(list)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc and tc in all_ids:
            concept_qs[tc].append(q)

    # Build prereq/dependent maps from edges
    prereq_map = defaultdict(list)
    dep_map = defaultdict(list)
    for e in edges:
        if e["source"] in all_ids and e["target"] in all_ids:
            prereq_map[e["target"]].append(e["source"])
            dep_map[e["source"]].append(e["target"])

    # Assign role: foundational, competency, contextual
    def assign_role(c):
        qcount = len(concept_qs.get(c["id"], []))
        has_deps = len(dep_map.get(c["id"], [])) > 0
        if qcount == 0 and has_deps:
            return "foundational"
        elif qcount > 0:
            return "competency"
        else:
            return "contextual"

    # Enrich concepts
    enriched = []
    for c in concepts:
        qs = concept_qs.get(c["id"], [])
        level_dist = defaultdict(int)
        for q in qs:
            level_dist[q["level"]] += 1

        enriched.append({
            "id": c["id"],
            "name": c["name"],
            "description": c["description"],
            "tracks": c.get("tracks", []),
            "source_chapters": c.get("source_chapters", []),
            "prerequisites": sorted(prereq_map.get(c["id"], [])),
            "dependents": sorted(dep_map.get(c["id"], [])),
            "question_count": len(qs),
            "level_distribution": dict(level_dist),
            "role": assign_role(c),
        })

    # Build output
    out = {
        "version": taxonomy.get("version", "3.2"),
        "synced_at": datetime.now().isoformat(),
        "total_concepts": len(enriched),
        "total_edges": len(edges),
        "total_questions": len(corpus),
        "concepts": enriched,
        "edges": [{"source": e["source"], "target": e["target"]} for e in edges
                  if e["source"] in all_ids and e["target"] in all_ids],
    }

    # Write to staffml
    out_path = BASE / "staffml" / "src" / "data" / "taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # Stats
    roles = defaultdict(int)
    for c in enriched:
        roles[c["role"]] += 1
    tested = sum(1 for c in enriched if c["question_count"] > 0)

    print(f"═══ Taxonomy Sync ═══\n")
    print(f"  Concepts:      {len(enriched)}")
    print(f"  Edges:         {len(out['edges'])}")
    print(f"  Tested:        {tested}/{len(enriched)} ({tested/len(enriched):.1%})")
    print(f"  Roles:         foundational={roles['foundational']}, competency={roles['competency']}, contextual={roles['contextual']}")
    print(f"  Output:        {out_path}")


# ─── COMPETENCY-MODEL ──────────────────────────────────────────
def cmd_competency_model(args):
    """Extract competency clusters from the taxonomy graph via Gemini."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    corpus = load_corpus()
    concepts = taxonomy["concepts"]
    edges = taxonomy["edges"]
    all_ids = {c["id"] for c in concepts}
    model = CONFIG["models"].get("extractor", "gemini-2.5-flash")

    # Build concept summaries
    concept_qs = defaultdict(int)
    for q in corpus:
        tc = q.get("taxonomy_concept", "")
        if tc and tc in all_ids:
            concept_qs[tc] += 1

    concept_list = '\n'.join(
        f"- {c['id']}: {c['name']} ({concept_qs.get(c['id'], 0)} Qs, "
        f"prereqs: {c.get('prerequisites', [])[:3]}, "
        f"tracks: {c.get('tracks', [])})"
        for c in concepts
    )

    prompt = f"""You are an expert ML Systems competency framework designer.

TASK: Analyze this taxonomy of {len(concepts)} ML Systems concepts and identify 10-15 competency clusters.

A competency cluster is a group of 5-30 concepts that together enable a demonstrable engineering ability.
Each competency should be something a Staff ML engineer could be evaluated on.

TAXONOMY CONCEPTS:
{concept_list}

RULES:
1. Each competency must have 5-30 concept IDs from the list above
2. Competencies should cover the full breadth of the taxonomy
3. Every concept should belong to at least one competency
4. Competencies should be at the "ability" level, not the "knowledge" level
   - Good: "Size and configure accelerator fleets for given workloads"
   - Bad: "Know about GPU memory hierarchy"
5. Include 2-3 example assessment tasks for each competency

Return a JSON array of competency objects. Each must have:
- "id": kebab-case identifier
- "name": Short human-readable name (3-5 words)
- "description": One sentence describing the engineering ability
- "concept_ids": Array of concept IDs from the taxonomy
- "example_tasks": Array of 2-3 concrete assessment tasks

Return ONLY the JSON array, no markdown fences or other text."""

    print(f"═══ Competency Model Extraction ═══\n")
    print(f"  Concepts: {len(concepts)}")
    print(f"  Model:    {model}")
    print(f"  Extracting competency clusters...\n")

    text = _call_gemini(prompt, model, timeout=300)
    result = _parse_json_response(text)

    if not result:
        print("  ❌ Failed to parse Gemini response")
        sys.exit(1)

    # Save competencies
    comp_path = BASE / "competencies.json"
    output = {
        "version": "1.0",
        "extracted_at": datetime.now().isoformat(),
        "model": model,
        "total_competencies": len(result),
        "competencies": result,
    }
    comp_path.write_text(json.dumps(output, indent=2))

    # Stats
    all_concept_ids = set()
    for comp in result:
        all_concept_ids.update(comp.get("concept_ids", []))

    coverage = len(all_concept_ids & all_ids) / len(all_ids) if all_ids else 0

    print(f"  Competencies:      {len(result)}")
    for comp in result:
        print(f"    {comp['id']:35} {len(comp.get('concept_ids', []))} concepts")
    print(f"\n  Concept coverage:  {len(all_concept_ids & all_ids)}/{len(all_ids)} ({coverage:.1%})")
    print(f"  Saved to:          {comp_path}")


# ─── GRAPH (deprecated) ────────────────────────────────────────
# NOTE: cmd_graph removed — replaced by staffml/src/app/taxonomy/ (Phase 2+4)
def cmd_graph(args):
    """Placeholder — graph visualization moved to StaffML app."""
    print("⚠️  The `graph` command has been removed.")
    print("   Use the StaffML Taxonomy Explorer instead:")
    print("   cd staffml && npm run dev → http://localhost:3000/taxonomy")
    sys.exit(0)


# ─── CLI ────────────────────────────────────────────────────────
def cmd_facets(args):
    """Show per-axis coverage audit for v5.3 taxonomy fields."""
    corpus = load_corpus()
    total = len(corpus)

    axes = {
        "reasoning_competency": {"label": "Axis 3: Reasoning Competency", "values": set()},
        "knowledge_area": {"label": "Axis 4: Knowledge Area", "values": set()},
        "reasoning_mode": {"label": "Axis 5: Reasoning Mode", "values": set()},
        "concept_tags": {"label": "Axis 6: Concept Tags", "values": set()},
        "primary_concept": {"label": "Primary Concept (preserved)", "values": set()},
    }

    populated = {k: 0 for k in axes}

    for q in corpus:
        for field in axes:
            val = q.get(field)
            if val is not None:
                if isinstance(val, list):
                    if len(val) > 0:
                        populated[field] += 1
                        axes[field]["values"].update(val)
                elif val:
                    populated[field] += 1
                    axes[field]["values"].add(val)

    print(f"\n{'='*60}")
    print(f"  v5.3 FACETED CLASSIFICATION AUDIT ({total} questions)")
    print(f"{'='*60}\n")

    for field, info in axes.items():
        pct = (populated[field] / total * 100) if total else 0
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {info['label']}")
        print(f"    Populated: {populated[field]:,}/{total:,} ({pct:.1f}%)")
        print(f"    Distinct:  {len(info['values']):,} unique values")
        print(f"    [{bar}]")
        print()


def cmd_reclassify_stats(args):
    """Comprehensive distribution report for v5.3 taxonomy axes."""
    from schema import (
        VALID_REASONING_COMPETENCIES,
        VALID_KNOWLEDGE_AREAS,
        VALID_REASONING_MODES,
    )

    corpus = load_corpus()
    total = len(corpus)

    print(f"\n{'='*70}")
    print(f"  v5.3 RECLASSIFICATION STATISTICS ({total} questions)")
    print(f"{'='*70}\n")

    # Axis 3: Reasoning Competency distribution
    rc_counts = Counter(q.get("reasoning_competency") for q in corpus if q.get("reasoning_competency"))
    print("  AXIS 3: REASONING COMPETENCY")
    print("  " + "-" * 50)
    for rc in sorted(VALID_REASONING_COMPETENCIES):
        count = rc_counts.get(rc, 0)
        pct = count / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        print(f"    {rc:>5}: {count:>5} ({pct:5.1f}%) {bar}")
    print()

    # Axis 4: Knowledge Area distribution
    ka_counts = Counter(q.get("knowledge_area") for q in corpus if q.get("knowledge_area"))
    print("  AXIS 4: KNOWLEDGE AREA")
    print("  " + "-" * 50)
    for ka in sorted(VALID_KNOWLEDGE_AREAS):
        count = ka_counts.get(ka, 0)
        pct = count / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        flag = " ⚠ LOW" if count < 50 else ""
        print(f"    {ka:>3}: {count:>5} ({pct:5.1f}%) {bar}{flag}")
    print()

    # Axis 5: Reasoning Mode distribution
    rm_counts = Counter(q.get("reasoning_mode") for q in corpus if q.get("reasoning_mode"))
    print("  AXIS 5: REASONING MODE")
    print("  " + "-" * 50)
    for mode in sorted(VALID_REASONING_MODES):
        count = rm_counts.get(mode, 0)
        pct = count / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        print(f"    {mode:>30}: {count:>5} ({pct:5.1f}%) {bar}")
    print()

    # Axis 6: Concept Tags — top 20
    tag_counts = Counter()
    for q in corpus:
        for tag in q.get("concept_tags") or []:
            tag_counts[tag] += 1
    print(f"  AXIS 6: CONCEPT TAGS (top 20 of {len(tag_counts)} unique)")
    print("  " + "-" * 50)
    for tag, count in tag_counts.most_common(20):
        pct = count / total * 100 if total else 0
        flag = " ⚠ HIGH" if count > 300 else ""
        print(f"    {tag:>40}: {count:>5} ({pct:5.1f}%){flag}")
    print()

    # Cross-axis: Level vs Reasoning Mode
    print("  CROSS-AXIS: Level × Reasoning Mode")
    print("  " + "-" * 50)
    level_mode = defaultdict(Counter)
    for q in corpus:
        lvl = q.get("level", "?")
        mode = q.get("reasoning_mode", "?")
        if mode != "?":
            level_mode[lvl][mode] += 1
    for lvl in LEVELS_ORDER:
        if level_mode[lvl]:
            top_mode = level_mode[lvl].most_common(1)[0]
            print(f"    {lvl}: top mode = {top_mode[0]} ({top_mode[1]}x)")
    print()


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

    planall_p = sub.add_parser("plan-all", help="Generate per-(level,track) plan files")
    planall_p.add_argument("--priority", choices=["all", "chains", "expand"], default="all",
                           help="chains=L1/L2/L6++untested, expand=cross-track, all=everything")

    genbatch_p = sub.add_parser("generate-batch", help="Generate from a specific plan file")
    genbatch_p.add_argument("plan", help="Path to plan JSON file")
    genbatch_p.add_argument("--workers", type=int, default=4, help="Parallel workers")

    merge_p = sub.add_parser("merge", help="Merge _generated/ batches into corpus")
    merge_p.add_argument("--pattern", default="batch-*.json", help="Glob pattern for batch files")
    merge_p.add_argument("--dry-run", action="store_true", help="Validate without adding")

    graph_p = sub.add_parser("graph", help="Interactive taxonomy visualization")
    graph_p.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    gen_p = sub.add_parser("generate", help="Generate questions from plan")
    gen_p.add_argument("--count", type=int, default=0, help="Max questions to generate (0=all)")
    gen_p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    gen_p.add_argument("--auto", action="store_true", help="Auto-add passing questions")

    loop_p = sub.add_parser("loop", help="Auto-loop: analyze→generate→add")
    loop_p.add_argument("--max-rounds", type=int, default=20, help="Max generation rounds")
    loop_p.add_argument("--batch-size", type=int, default=20, help="Questions per round")

    sub.add_parser("taxonomy-check", help="Read-only taxonomy health diagnostic")

    tax_fix_p = sub.add_parser("taxonomy-fix", help="Automated taxonomy repair")
    tax_fix_p.add_argument("--verbose", action="store_true", help="Show remap details")

    sub.add_parser("taxonomy-sync", help="Export enriched taxonomy to staffml app")

    improve_p = sub.add_parser("taxonomy-improve", help="Run taxonomy improvement round via Gemini")
    improve_p.add_argument("--round", required=True, choices=list(IMPROVE_ROUNDS.keys()),
                           help="Improvement round to run")
    improve_p.add_argument("--count", type=int, default=0, help="Limit items to process (0=all)")

    apply_p = sub.add_parser("taxonomy-apply", help="Apply accepted taxonomy proposals")
    apply_p.add_argument("file", help="Path to proposals JSON file")

    sub.add_parser("competency-model", help="Extract competency clusters from taxonomy graph")

    sub.add_parser("facets", help="Per-axis coverage audit for v5.3 taxonomy fields")
    sub.add_parser("reclassify-stats", help="Comprehensive distribution report for v5.3 axes")

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
        "plan-all": cmd_plan_all,
        "generate-batch": cmd_generate_batch,
        "merge": cmd_merge,
        "graph": cmd_graph,
        "taxonomy-check": cmd_taxonomy_check,
        "taxonomy-fix": cmd_taxonomy_fix,
        "taxonomy-sync": cmd_taxonomy_sync,
        "taxonomy-improve": cmd_taxonomy_improve,
        "taxonomy-apply": cmd_taxonomy_apply,
        "competency-model": cmd_competency_model,
        "facets": cmd_facets,
        "reclassify-stats": cmd_reclassify_stats,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
