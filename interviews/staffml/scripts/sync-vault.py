#!/usr/bin/env python3
"""Sync vault data → StaffML app data directory.

Copies corpus.json and taxonomy.json from the vault (source of truth)
to the StaffML app's data directory, then regenerates the vault manifest.

Usage: python3 interviews/staffml/scripts/sync-vault.py [--dry-run]
"""

import json
import shutil
import sys
from pathlib import Path

VAULT_DIR = Path(__file__).parent.parent.parent / "vault"
STAFFML_DATA = Path(__file__).parent.parent / "src" / "data"

FILES_TO_SYNC = [
    ("corpus.json", "Question corpus"),
    ("taxonomy.json", "Taxonomy graph"),
]

DRY_RUN = "--dry-run" in sys.argv


def main() -> None:
    print("🔄 Syncing vault → StaffML data...\n")

    if not VAULT_DIR.exists():
        print(f"❌ Vault directory not found: {VAULT_DIR}")
        sys.exit(1)

    if not STAFFML_DATA.exists():
        print(f"❌ StaffML data directory not found: {STAFFML_DATA}")
        sys.exit(1)

    synced = 0
    for filename, label in FILES_TO_SYNC:
        src = VAULT_DIR / filename
        dst = STAFFML_DATA / filename

        if not src.exists():
            print(f"  ❌ {label}: source not found ({src})")
            sys.exit(1)

        # Compare sizes
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size if dst.exists() else 0

        # Compare question counts for corpus
        diff_info = ""
        if filename == "corpus.json":
            with open(src) as f:
                src_data = json.load(f)
            src_count = len(src_data)
            if dst.exists():
                with open(dst) as f:
                    dst_data = json.load(f)
                dst_count = len(dst_data)
                delta = src_count - dst_count
                if delta != 0:
                    diff_info = f" ({delta:+d} questions, {dst_count} → {src_count})"
                else:
                    diff_info = f" ({src_count} questions, no change)"
            else:
                diff_info = f" ({src_count} questions, new)"

        if filename == "taxonomy.json":
            with open(src) as f:
                src_data = json.load(f)
            src_concepts = len(src_data.get("concepts", []))
            if dst.exists():
                with open(dst) as f:
                    dst_data = json.load(f)
                dst_concepts = len(dst_data.get("concepts", []))
                delta = src_concepts - dst_concepts
                if delta != 0:
                    diff_info = f" ({delta:+d} concepts, {dst_concepts} → {src_concepts})"
                else:
                    diff_info = f" ({src_concepts} concepts, no change)"
            else:
                diff_info = f" ({src_concepts} concepts, new)"

        if DRY_RUN:
            print(f"  📋 {label}: would copy{diff_info}")
        else:
            shutil.copy2(src, dst)
            print(f"  ✅ {label}: synced{diff_info}")
            synced += 1

    # Regenerate manifest
    if not DRY_RUN:
        print("\n📦 Regenerating vault manifest...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "generate-manifest.py")],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
        else:
            print(f"  ❌ Manifest generation failed:\n{result.stderr}")
            sys.exit(1)
    else:
        print("\n📋 Would regenerate vault manifest")

    print(f"\n{'🎯 Sync complete' if not DRY_RUN else '📋 Dry run complete'} — {synced} files synced")


if __name__ == "__main__":
    main()
