#!/usr/bin/env python3
# ┌─── DEPRECATED ────────────────────────────────────────────────────────────┐
# │ Pre-YAML-migration script. Replaced by:                                   │
# │     vault build --legacy-json                                             │
# │ See ./DEPRECATED.md for the full map.                                     │
# └───────────────────────────────────────────────────────────────────────────┘
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

    # Run the export script which handles filtering, field stripping, and all data files
    export_script = VAULT_DIR / "scripts" / "export_to_staffml.py"
    if export_script.exists():
        print("  Running vault export script...")
        if DRY_RUN:
            print("  📋 Would run export_to_staffml.py")
        else:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(export_script)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
            else:
                print(f"  ❌ Export failed:\n{result.stderr}")
                sys.exit(1)
    else:
        # Fallback: direct copy (legacy behavior)
        print("  ⚠️  export_to_staffml.py not found, falling back to direct copy")
        for filename, label in FILES_TO_SYNC:
            src = VAULT_DIR / filename
            dst = STAFFML_DATA / filename
            if not src.exists():
                print(f"  ❌ {label}: source not found ({src})")
                sys.exit(1)
            if not DRY_RUN:
                shutil.copy2(src, dst)
                print(f"  ✅ {label}: synced")

    # Show what we got
    corpus_path = STAFFML_DATA / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path) as f:
            count = len(json.load(f))
        print(f"\n  📊 Corpus: {count} questions")

    taxonomy_path = STAFFML_DATA / "taxonomy.json"
    if taxonomy_path.exists():
        with open(taxonomy_path) as f:
            concepts = len(json.load(f).get("concepts", []))
        print(f"  📊 Taxonomy: {concepts} topics")

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

    print(f"\n{'🎯 Sync complete' if not DRY_RUN else '📋 Dry run complete'}")


if __name__ == "__main__":
    main()
