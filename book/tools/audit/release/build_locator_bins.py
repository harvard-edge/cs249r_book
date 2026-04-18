#!/usr/bin/env python3
"""Phase B1 (prep) — build per-chapter locator input bins.

Reads ``vol1-annotations-ground-truth.json``, filters to entries with
``needs_locator == True``, and writes one input file per QMD chapter to
``16_release_audit/scripts/locator-input/<bin>.json``. Each bin holds at
most ``MAX_PER_BIN`` entries. Bigger chapters are stratified-sampled (a
deterministic subset is selected so the same audit run is reproducible)
and the unsampled tail is recorded as ``deferred_entries`` so the
coverage report can disclose the sampling methodology honestly.

The bin file also embeds the QMD file path the locator agent should
read.

Output bins, plus a manifest at
``16_release_audit/scripts/locator-input/MANIFEST.json``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

LEDGER = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers/vol1-annotations-ground-truth.json"
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/scripts/locator-input"
QUARTO_ROOT = Path("/Users/VJ/GitHub/MLSysBook-release-audit/book/quarto/contents")

MAX_PER_BIN = 250

# Map ground-truth chapter → current vol1 QMD path (relative to the repo
# root). ``None`` means the chapter has no direct dev-branch counterpart
# (e.g. Part-page placeholders — those edits land in
# ``contents/vol1/parts/`` files).
CHAPTER_TO_QMD: dict[str, str | None] = {
    "Ch1: Introduction": "book/quarto/contents/vol1/introduction/introduction.qmd",
    "Ch2: ML Systems": "book/quarto/contents/vol1/ml_systems/ml_systems.qmd",
    "Ch4: Data Engineering": "book/quarto/contents/vol1/data_engineering/data_engineering.qmd",
    "Ch6: Network Architectures": "book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd",
    "Ch7: ML Frameworks": "book/quarto/contents/vol1/frameworks/frameworks.qmd",
    "Ch8: Model Training": "book/quarto/contents/vol1/training/training.qmd",
    "Ch9: Data Selection": "book/quarto/contents/vol1/data_selection/data_selection.qmd",
    "Ch10: Model Compression": "book/quarto/contents/vol1/optimizations/model_compression.qmd",
    "Ch11: Hardware Acceleration": "book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd",
    "Ch12: Benchmarking": "book/quarto/contents/vol1/benchmarking/benchmarking.qmd",
    "Ch13: Model Serving": "book/quarto/contents/vol1/model_serving/model_serving.qmd",
    "Ch14: ML Operations": "book/quarto/contents/vol1/ml_ops/ml_ops.qmd",
    "Ch15: Responsible Engineering": "book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd",
    "Conclusion/Backmatter": "book/quarto/contents/vol1/conclusion/conclusion.qmd",
    "About This Book": "book/quarto/contents/vol1/frontmatter/about.qmd",
    "Acknowledgements": "book/quarto/contents/vol1/frontmatter/acknowledgements.qmd",
    "Notation and Conventions": "book/quarto/contents/vol1/frontmatter/notation.qmd",
    "Author's Note": "book/quarto/contents/vol1/frontmatter/about.qmd",
    "Cover/Title": "book/quarto/contents/vol1/frontmatter/about.qmd",
    "Appendices/Glossary/Index": "book/quarto/contents/vol1/backmatter",
    "Part I: Foundations": "book/quarto/contents/vol1/parts",
    "Part III: Optimize": "book/quarto/contents/vol1/parts",
    "Part IV: Deploy": "book/quarto/contents/vol1/parts",
}


def safe_name(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = json.loads(LEDGER.read_text())
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("needs_locator"):
            by_chapter[e.get("chapter") or "unknown"].append(e)

    rng = random.Random(0xA1)  # deterministic stratified sample
    manifest = []
    bin_files = []

    for chapter, items in sorted(by_chapter.items()):
        qmd = CHAPTER_TO_QMD.get(chapter)
        # Sort items deterministically by (page, annotation_id).
        items_sorted = sorted(items, key=lambda x: (x.get("page", 0), x.get("annotation_id", "")))

        if len(items_sorted) <= MAX_PER_BIN:
            sampled = items_sorted
            deferred: list[dict] = []
        else:
            # Stratify by bucket — preserve proportional representation.
            by_bucket: dict[str, list[dict]] = defaultdict(list)
            for it in items_sorted:
                by_bucket[it["bucket"]].append(it)
            target = MAX_PER_BIN
            allocated: dict[str, int] = {}
            total = len(items_sorted)
            # First pass: proportional allocation.
            for bucket, lst in by_bucket.items():
                allocated[bucket] = max(1, round(len(lst) / total * target))
            # Trim down/up to exactly MAX_PER_BIN.
            diff = target - sum(allocated.values())
            buckets_sorted = sorted(allocated.items(), key=lambda kv: -kv[1])
            i = 0
            while diff != 0 and buckets_sorted:
                key = buckets_sorted[i % len(buckets_sorted)][0]
                if diff > 0:
                    allocated[key] += 1
                    diff -= 1
                elif allocated[key] > 1:
                    allocated[key] -= 1
                    diff += 1
                i += 1
            sampled = []
            deferred = []
            for bucket, lst in by_bucket.items():
                k = min(allocated[bucket], len(lst))
                rng_local = random.Random(0xB1 ^ hash(bucket))
                idxs = sorted(rng_local.sample(range(len(lst)), k))
                chosen_set = set(idxs)
                for j, it in enumerate(lst):
                    if j in chosen_set:
                        sampled.append(it)
                    else:
                        deferred.append(it)
            sampled.sort(key=lambda x: (x.get("page", 0), x.get("annotation_id", "")))
            deferred.sort(key=lambda x: (x.get("page", 0), x.get("annotation_id", "")))

        bin_id = safe_name(chapter)
        bin_path = OUT_DIR / f"bin-{bin_id}.json"
        bin_data = {
            "chapter": chapter,
            "qmd_target": qmd,
            "total_needs_locator_in_chapter": len(items_sorted),
            "sampled_count": len(sampled),
            "deferred_count": len(deferred),
            "sampling_rationale": (
                "All entries included; bin <= MAX_PER_BIN" if not deferred
                else "Stratified-by-bucket random sample; rest of chapter recorded as deferred"
            ),
            "max_per_bin": MAX_PER_BIN,
            "entries": sampled,
            "deferred_entries": deferred,
        }
        bin_path.write_text(json.dumps(bin_data, indent=2))
        bin_files.append(str(bin_path))
        manifest.append({
            "bin_id": bin_id,
            "chapter": chapter,
            "qmd_target": qmd,
            "total_in_chapter": len(items_sorted),
            "sampled": len(sampled),
            "deferred": len(deferred),
            "bin_file": str(bin_path),
        })

    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "max_per_bin": MAX_PER_BIN,
        "bins": manifest,
        "total_sampled": sum(m["sampled"] for m in manifest),
        "total_deferred": sum(m["deferred"] for m in manifest),
        "total_in_chapters": sum(m["total_in_chapter"] for m in manifest),
    }, indent=2))
    print(f"Wrote {len(manifest)} bin files + manifest at {manifest_path}")
    for m in manifest:
        print(f"  {m['bin_id']:<28}  {m['sampled']:>3} sampled / {m['total_in_chapter']:>4} total → {m['qmd_target']}")


if __name__ == "__main__":
    main()
