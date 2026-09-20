#!/usr/bin/env python3
"""
Rebuild all 5 variant PDFs and store them safely in books/vol3/pdfs/
and books/_build/pdf-vol3/chapters/02_processor/.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
PDFS_DIR = REPO_ROOT / "books" / "vol3" / "pdfs"
PDFS_DIR.mkdir(parents=True, exist_ok=True)
BUILD_PDF_DIR = REPO_ROOT / "books" / "_build" / "pdf-vol3" / "chapters" / "02_processor"
TARGET_QMD = REPO_ROOT / "books" / "vol3" / "02_processor" / "02_processor.qmd"
BACKUP_QMD = REPO_ROOT / "drafts" / "vol3" / "02_processor_canonical_backup.qmd"

VARIANTS = ["v3", "v4", "v5", "v6", "v7"]

def main():
    print("=" * 80)
    print("Compiling all 5 Variant PDFs into persistent directory: books/vol3/pdfs/")
    print("=" * 80)

    try:
        for tag in VARIANTS:
            draft_path = REPO_ROOT / "drafts" / "vol3" / f"ch02_processor_{tag}" / "assembled_draft.qmd"
            if not draft_path.exists():
                print(f"❌ Draft not found for {tag}: {draft_path}")
                continue

            words = len(draft_path.read_text(encoding="utf-8").split())
            print(f"\n>>> [Variant {tag.upper()}] Staging {draft_path.name} ({words:,} words)...")
            shutil.copy(draft_path, TARGET_QMD)

            print(f">>> [Variant {tag.upper()}] Running binder build pdf...")
            start = time.time()
            res = subprocess.run(
                ["./binder/binder", "build", "pdf", "02_processor", "--vol3", "--skip-validate"],
                cwd=str(REPO_ROOT),
                check=False
            )
            elapsed = time.time() - start

            if res.returncode != 0:
                print(f"❌ [Variant {tag.upper()}] Build failed (took {elapsed:.1f}s)")
                continue

            built_pdf = BUILD_PDF_DIR / "02_processor.pdf"
            if not built_pdf.exists():
                print(f"❌ [Variant {tag.upper()}] Output PDF missing at {built_pdf}")
                continue

            dest_pdf = PDFS_DIR / f"02_processor_{tag}.pdf"
            shutil.copy(built_pdf, dest_pdf)
            size_mb = dest_pdf.stat().st_size / (1024 * 1024)
            print(f"✅ [Variant {tag.upper()}] Saved to {dest_pdf.relative_to(REPO_ROOT)} ({size_mb:.2f} MB, {elapsed:.1f}s)")

        # Copy all into BUILD_PDF_DIR as well
        print("\n>>> Synchronizing all variant PDFs to books/_build/pdf-vol3/chapters/02_processor/...")
        for tag in VARIANTS:
            src = PDFS_DIR / f"02_processor_{tag}.pdf"
            if src.exists():
                shutil.copy(src, BUILD_PDF_DIR / f"02_processor_{tag}.pdf")
                print(f"  ✓ Copied {src.name} to {BUILD_PDF_DIR.relative_to(REPO_ROOT)}")

    finally:
        if BACKUP_QMD.exists():
            shutil.copy(BACKUP_QMD, TARGET_QMD)
            print(f"\n✅ Restored canonical Chapter 02 from {BACKUP_QMD.name}")

    print("\n" + "=" * 80)
    print("ALL VARIANT PDFS CURRENT STATUS:")
    for tag in VARIANTS:
        dest = PDFS_DIR / f"02_processor_{tag}.pdf"
        if dest.exists():
            mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✅ [02_processor_{tag}.pdf] {mb:.2f} MB ({dest})")
        else:
            print(f"  ❌ [02_processor_{tag}.pdf] Missing")
    print("=" * 80)

if __name__ == "__main__":
    main()
