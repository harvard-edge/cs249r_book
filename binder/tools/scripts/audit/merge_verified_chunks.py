#!/usr/bin/env python3
"""
merge_verified_chunks.py — Merges subagent verification results into the manifest
and generates actionable staging patches.
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def main():
    chunk_dir = REPO_ROOT / "review/bib-verification/chunks"
    manifest_path = REPO_ROOT / "review/bib-verification/manifest.json"
    staging_dir = REPO_ROOT / "review/bib-verification/staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    result_files = sorted(chunk_dir.glob("chunk_*_results.json"))
    print(f"Found {len(result_files)} chunk result file(s).")

    updated_count = 0
    metadata_fixes = []

    for rf in result_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                results = json.load(f)
            for item in results:
                key = item.get("key")
                if not key or key not in manifest:
                    continue

                status = item.get("status")
                manifest[key]["subagent_status"] = status
                manifest[key]["subagent_evidence"] = item.get("evidence")
                manifest[key]["canonical_doi"] = item.get("canonical_doi")
                manifest[key]["canonical_url"] = item.get("canonical_url")
                manifest[key]["canonical_title"] = item.get("canonical_title")
                manifest[key]["canonical_authors"] = item.get("canonical_authors")
                manifest[key]["canonical_year"] = item.get("canonical_year")

                if status == "verified":
                    manifest[key]["status"] = "verified"
                    manifest[key]["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    updated_count += 1
                elif status == "metadata_fix":
                    manifest[key]["status"] = "metadata_fix"
                    manifest[key]["recommended_fix"] = item.get("recommended_fix", {})
                    metadata_fixes.append(item)
                    updated_count += 1
                elif status == "invalid":
                    manifest[key]["status"] = "invalid"
                    updated_count += 1
        except Exception as e:
            print(f"Error reading {rf}: {e}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated_count} entries in manifest.")
    print(f"Found {len(metadata_fixes)} metadata fixes.")

    if metadata_fixes:
        fixes_path = staging_dir / "suggested_fixes.json"
        with open(fixes_path, "w", encoding="utf-8") as f:
            json.dump(metadata_fixes, f, indent=2, ensure_ascii=False)
        print(f"Wrote fixes to {fixes_path}")


if __name__ == "__main__":
    main()
