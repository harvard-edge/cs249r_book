#!/usr/bin/env python3
"""Generate vault-manifest.json from corpus and taxonomy data.

Run after any corpus/taxonomy update to version the vault.
Usage: python3 interviews/staffml/scripts/generate-manifest.py [--bump-minor | --bump-patch]
"""

import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

STAFFML_DATA = Path(__file__).parent.parent / "src" / "data"
MANIFEST_PATH = STAFFML_DATA / "vault-manifest.json"


def main():
    # Load corpus
    with open(STAFFML_DATA / "corpus.json") as f:
        corpus = json.load(f)

    # Load taxonomy
    with open(STAFFML_DATA / "taxonomy.json") as f:
        taxonomy = json.load(f)

    # Content hash — deterministic, based on sorted question IDs + levels
    fingerprint = "|".join(
        f"{q['id']}:{q.get('level', '')}:{q.get('chain_ids', '')}"
        for q in sorted(corpus, key=lambda q: q["id"])
    )
    content_hash = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

    # Chain stats
    chains: dict[str, list[str]] = {}
    for q in corpus:
        cid = q.get("chain_ids", "")
        if cid:
            chains.setdefault(cid, []).append(q["id"])

    # Level distribution
    levels: dict[str, int] = {}
    for q in corpus:
        lvl = q.get("level", "unknown")
        levels[lvl] = levels.get(lvl, 0) + 1

    # Track distribution
    tracks: dict[str, int] = {}
    for q in corpus:
        t = q.get("track", "unknown")
        tracks[t] = tracks.get(t, 0) + 1

    # Competency area distribution
    areas: dict[str, int] = {}
    for q in corpus:
        a = q.get("competency_area", "unknown")
        areas[a] = areas.get(a, 0) + 1

    # Taxonomy stats
    concepts = taxonomy.get("concepts", [])

    # Version handling
    prev_manifest = None
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            prev_manifest = json.load(f)

    prev_version = prev_manifest.get("version", "0.1.0") if prev_manifest else "0.1.0"
    parts = prev_version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if "--bump-minor" in sys.argv:
        minor += 1
        patch = 0
    elif "--bump-patch" in sys.argv:
        patch += 1
    elif prev_manifest and prev_manifest.get("contentHash") != content_hash:
        # Auto-bump patch if content changed
        patch += 1

    version = f"{major}.{minor}.{patch}"

    # Build changelog entry
    changelog_entry = None
    if prev_manifest:
        prev_count = prev_manifest.get("questionCount", 0)
        diff = len(corpus) - prev_count
        if diff != 0 or prev_manifest.get("contentHash") != content_hash:
            changelog_entry = {
                "version": version,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "questionsDelta": diff,
                "previousHash": prev_manifest.get("contentHash"),
            }

    # Build manifest
    manifest = {
        "version": version,
        "buildDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contentHash": content_hash,
        "questionCount": len(corpus),
        "chainCount": len(chains),
        "conceptCount": len(concepts),
        "trackDistribution": dict(sorted(tracks.items())),
        "levelDistribution": dict(
            sorted(levels.items(), key=lambda x: x[0])
        ),
        "areaCount": len(areas),
        "taxonomyVersion": taxonomy.get("version", "unknown"),
    }

    if changelog_entry:
        # Append to existing changelog
        prev_changelog = prev_manifest.get("changelog", []) if prev_manifest else []
        manifest["changelog"] = [changelog_entry] + prev_changelog[:9]  # keep last 10
    elif prev_manifest and "changelog" in prev_manifest:
        manifest["changelog"] = prev_manifest["changelog"]

    # Write
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Vault manifest v{version} written to {MANIFEST_PATH}")
    print(f"  Questions: {len(corpus)}")
    print(f"  Chains:    {len(chains)}")
    print(f"  Concepts:  {len(concepts)}")
    print(f"  Hash:      {content_hash}")
    if changelog_entry:
        print(f"  Delta:     {changelog_entry['questionsDelta']:+d} questions")


if __name__ == "__main__":
    main()
