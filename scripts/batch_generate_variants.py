#!/usr/bin/env python3
"""
Batch Generator for Chapter 02 Variations (V3 to V7).

Orchestrates multi-variant generation, assembly, Quarto PDF compilation,
and verification across the 5 distinct pedagogical outline stances:
- V3: The Quantitative Architect (Hennessy & Patterson)
- V4: The Systems Principles Architect (Saltzer & Kaashoek)
- V5: The Pragmatic Systems Engineer (Ousterhout Clean Abstraction)
- V6: The Dependability & Invariants Stance (Dijkstra / Schneider)
- V7: The Systems Builder & Kernel Hacker (MIT 6.828 / CS 162 Lab)
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTLINES_DIR = REPO_ROOT / "books" / "vol3" / "outlines"
BUILD_PDF_DIR = REPO_ROOT / "books" / "_build" / "pdf-vol3" / "chapters" / "02_processor"
TARGET_QMD = REPO_ROOT / "books" / "vol3" / "02_processor" / "02_processor.qmd"

VARIANTS = [
    {
        "tag": "v3",
        "name": "Quantitative Architecture (Hennessy & Patterson)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V3_QUANTITATIVE.md",
    },
    {
        "tag": "v4",
        "name": "Systems Principles (Saltzer & Kaashoek)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V4_PRINCIPLES.md",
    },
    {
        "tag": "v5",
        "name": "Pragmatic Systems (Ousterhout Clean Abstraction)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V5_PRAGMATIC.md",
    },
    {
        "tag": "v6",
        "name": "Dependability & Invariants (Dijkstra & Schneider)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V6_DEPENDABILITY.md",
    },
    {
        "tag": "v7",
        "name": "Systems Builder & Hacker (MIT 6.828 / CS 162)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V7_BUILDER.md",
    },
    {
        "tag": "v8",
        "name": "Distributed Systems & Unreliable Service (Lamport / Gray / Vogels)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V8_DISTRIBUTED.md",
    },
    {
        "tag": "v9",
        "name": "Compiler & Language Runtime (Dragon Book & LLVM)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V9_COMPILER.md",
    },
    {
        "tag": "v10",
        "name": "AI Systems Engineering Apprenticeship (CS Undergrad Bridge)",
        "outline": OUTLINES_DIR / "MASTER_OUTLINE_V10_APPRENTICE.md",
    },
]


def run_command_live(cmd: list[str], desc: str) -> bool:
    print(f"\n>>> [{desc}] Running: {' '.join(cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"❌ FAILED: {desc} (code {proc.returncode}, took {elapsed:.1f}s)")
        return False
    print(f"✅ SUCCESS: {desc} (took {elapsed:.1f}s)")
    return True


def build_variant(var_info: dict, chapter_num: str = "02", force: bool = False, model: str = "gemini-3.8-flash-high") -> dict:
    tag = var_info["tag"]
    name = var_info["name"]
    outline_path = var_info["outline"]

    print("\n" + "=" * 80)
    print(f"STARTING VARIANT {tag.upper()}: {name}")
    print(f"Outline: {outline_path.name}")
    print("=" * 80)

    # 1. Run generation
    gen_cmd = [
        sys.executable,
        "scripts/generate_chapter_from_v2.py",
        "--chapter", chapter_num,
        "--outline", str(outline_path),
        "--variant", tag,
        "--model", model,
        "--run-all",
    ]
    if force:
        gen_cmd.append("--force")

    if not run_command_live(gen_cmd, f"Generate Chapter {chapter_num} Variant {tag}"):
        return {"tag": tag, "name": name, "status": "GENERATION_FAILED"}

    # 2. Assemble and Validate
    asm_cmd = [
        sys.executable,
        "scripts/generate_chapter_from_v2.py",
        "--chapter", chapter_num,
        "--outline", str(outline_path),
        "--variant", tag,
        "--assemble",
        "--validate",
    ]
    if not run_command_live(asm_cmd, f"Assemble & Validate Chapter {chapter_num} Variant {tag}"):
        return {"tag": tag, "name": name, "status": "ASSEMBLY_FAILED"}

    # 3. Stage to books/vol3/02_processor/02_processor.qmd and build PDF
    assembled_draft = REPO_ROOT / "drafts" / "vol3" / f"ch{chapter_num}_processor_{tag}" / "assembled_draft.qmd"
    if not assembled_draft.exists():
        print(f"❌ Assembled draft not found: {assembled_draft}")
        return {"tag": tag, "name": name, "status": "DRAFT_NOT_FOUND"}

    # Read word count
    words = len(assembled_draft.read_text(encoding="utf-8").split())

    # Backup existing canonical if not already backed up
    backup_qmd = REPO_ROOT / "drafts" / "vol3" / "02_processor_canonical_backup.qmd"
    if not backup_qmd.exists() and TARGET_QMD.exists():
        shutil.copy(TARGET_QMD, backup_qmd)
        print(f"Backed up canonical Chapter 02 to {backup_qmd.name}")

    # Copy assembled variant to target QMD
    shutil.copy(assembled_draft, TARGET_QMD)
    print(f"Staged {assembled_draft.name} ({words:,}w) -> {TARGET_QMD.name}")

    # 4. Build PDF
    pdf_cmd = [
        "./binder/binder", "build", "pdf", f"{chapter_num}_processor", "--vol3", "--skip-validate"
    ]
    if not run_command_live(pdf_cmd, f"Build PDF for Variant {tag}"):
        return {"tag": tag, "name": name, "status": "PDF_BUILD_FAILED", "words": words}

    canonical_pdf = BUILD_PDF_DIR / "02_processor.pdf"
    variant_pdf = BUILD_PDF_DIR / f"02_processor_{tag}.pdf"

    if not canonical_pdf.exists():
        print(f"❌ Output PDF not found: {canonical_pdf}")
        return {"tag": tag, "name": name, "status": "PDF_FILE_MISSING", "words": words}

    shutil.copy(canonical_pdf, variant_pdf)
    pdf_size_mb = variant_pdf.stat().st_size / (1024 * 1024)
    print(f"✅ Created Variant PDF: {variant_pdf.name} ({pdf_size_mb:.2f} MB)")

    # 5. Inspection image rendering skipped per user request
    verify_dir = Path(f"/tmp/verify_{tag}")

    return {
        "tag": tag,
        "name": name,
        "status": "COMPLETED",
        "words": words,
        "pdf_path": str(variant_pdf),
        "pdf_size_mb": round(pdf_size_mb, 2),
        "verify_dir": str(verify_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Generate Chapter Variations.")
    parser.add_argument("--variants", nargs="+", default=["v3", "v4", "v5", "v6", "v7"], help="Variants to run (e.g. v3 v4).")
    parser.add_argument("--chapter", default="02", help="Chapter number (default: 02).")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing steps.")
    parser.add_argument("--model", default="gemini-3.8-flash-high", help="Model override.")
    args = parser.parse_args()

    results = []
    selected_variants = [v for v in VARIANTS if v["tag"] in args.variants]

    print("=" * 80)
    print(f"MLSysBook Vol 3 - Multi-Variant Batch Chapter Generator")
    print(f"Target Chapter: {args.chapter}")
    print(f"Variants Scheduled ({len(selected_variants)}): {[v['tag'] for v in selected_variants]}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Ensure backup exists before batch run
    backup_qmd = REPO_ROOT / "drafts" / "vol3" / "02_processor_canonical_backup.qmd"
    if not backup_qmd.exists() and TARGET_QMD.exists():
        shutil.copy(TARGET_QMD, backup_qmd)
        print(f"Initial backup of canonical Chapter 02 to {backup_qmd.name}")

    try:
        for var in selected_variants:
            res = build_variant(var, chapter_num=args.chapter, force=args.force, model=args.model)
            results.append(res)
    finally:
        # Restore canonical after batch execution
        if backup_qmd.exists():
            shutil.copy(backup_qmd, TARGET_QMD)
            print(f"Restored canonical Chapter 02 from {backup_qmd.name}")

    print("\n" + "=" * 80)
    print("BATCH EXECUTION SUMMARY")
    print("=" * 80)
    for r in results:
        status_icon = "✅" if r.get("status") == "COMPLETED" else "❌"
        words_str = f"{r.get('words', 0):,}w" if "words" in r else "N/A"
        size_str = f"{r.get('pdf_size_mb', 0)} MB" if "pdf_size_mb" in r else "N/A"
        print(f"{status_icon} [{r['tag'].upper()}] {r['name']:<50} | {r['status']} | {words_str} | {size_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
