"""Apply consensus progressive disclosure edits from an audit dossier to target chapter files."""

import argparse
import json
import os
import shutil
import sys
from typing import List


def apply_dossier_edits(dossier_json_path: str, dry_run: bool = True):
    if not os.path.exists(dossier_json_path):
        raise FileNotFoundError(f"Dossier JSON not found: {dossier_json_path}")

    with open(dossier_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_file = data.get("audited_file")
    if not target_file or not os.path.exists(target_file):
        raise FileNotFoundError(f"Target chapter file not found: {target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        file_content = f.read()

    topics = data.get("topics", [])
    print("=" * 70)
    print(f"🔧 PEDAGOGICAL EDIT APPLICATOR: {target_file}")
    print(f"Mode: {'DRY RUN (Preview Only)' if dry_run else 'APPLYING LIVE'}")
    print("=" * 70)

    applied_count = 0
    updated_content = file_content

    for topic in topics:
        orig = topic.get("original_text", "").strip()
        rewrite = topic.get("proposed_rewrite", "").strip()
        priority = topic.get("priority", "")
        rationale = topic.get("rationale", "")

        if not orig or not rewrite or orig == rewrite:
            continue

        print(f"\n[{topic.get('topic_id')}] Lines {topic.get('line_range')} ({priority}):")
        print(f"Rationale: {rationale}")
        print("-" * 40)
        print(f"  - Current:  {orig}")
        print(f"  + Proposed: {rewrite}")

        if orig in updated_content:
            if not dry_run:
                updated_content = updated_content.replace(orig, rewrite, 1)
                print("  ✓ Match found: Will replace.")
            else:
                print("  [Dry Run] Valid match found in source text.")
            applied_count += 1
        else:
            print("  ⚠️  Exact substring match not found in source text (may require fuzzy/line-range patch).")

    if not dry_run and applied_count > 0:
        # Create a backup
        backup_path = f"{target_file}.bak"
        shutil.copyfile(target_file, backup_path)
        print(f"\nCreated backup at {backup_path}")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"✅ Successfully applied {applied_count} edit(s) to {target_file}!")
    else:
        print(f"\nFinished preview: {applied_count} applicable progressive disclosure edit(s) identified.")


def main():
    parser = argparse.ArgumentParser(description="Apply progressive disclosure edits from an audit dossier.")
    parser.add_argument("dossier", type=str, help="Path to the audit dossier JSON")
    parser.add_argument("--apply", action="store_true", help="Apply edits to file (defaults to dry-run preview)")
    args = parser.parse_args()

    apply_dossier_edits(args.dossier, dry_run=not args.apply)


if __name__ == "__main__":
    main()
