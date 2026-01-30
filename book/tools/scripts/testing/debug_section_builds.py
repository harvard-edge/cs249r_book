#!/usr/bin/env python3
"""
Build a chapter section-by-section to isolate which section causes a build failure.

Uses section_splitter.py to split a chapter into ## sections, then progressively
builds with more sections included until the build breaks. This pinpoints exactly
which section introduces the error.

Usage:
    python3 debug_section_builds.py --chapter dl_primer --vol1
    python3 debug_section_builds.py --chapter dl_primer --vol1 --binary-search
    python3 debug_section_builds.py --chapter dl_primer --vol1 --format html
    python3 debug_section_builds.py --chapter dl_primer --vol1 -v

Log files are written to: book/tools/scripts/testing/logs/<volume>/<chapter>/section_debug/
"""

import subprocess
import sys
import time
import signal
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add content scripts to path for section_splitter
SCRIPT_DIR = Path(__file__).resolve().parent
CONTENT_SCRIPTS_DIR = SCRIPT_DIR.parents[0] / "content"
sys.path.insert(0, str(CONTENT_SCRIPTS_DIR))

from section_splitter import split_chapter, ChapterStructure


def find_chapter_qmd(chapter: str, volume: str) -> Path:
    """Locate the .qmd file for a chapter."""
    contents_dir = SCRIPT_DIR.parents[2] / "quarto" / "contents" / volume
    # Try direct match: contents/vol1/chapter/chapter.qmd
    for subdir in contents_dir.iterdir():
        if subdir.is_dir():
            qmd = subdir / f"{chapter}.qmd"
            if qmd.exists():
                return qmd
    raise FileNotFoundError(f"Chapter '{chapter}' not found in {contents_dir}")


def assemble_progressive_content(chapter: ChapterStructure, up_to_section: int) -> str:
    """
    Build chapter content including sections 0..up_to_section.

    Args:
        chapter: Parsed chapter structure
        up_to_section: Include sections with index <= this value (-1 = preamble only)

    Returns:
        Assembled .qmd content string
    """
    parts = []

    # Always include frontmatter
    if chapter.frontmatter:
        parts.append(chapter.frontmatter)

    # Always include pre-content (# title, intro text before first ##)
    if chapter.pre_content:
        parts.append(chapter.pre_content)

    # Progressively add sections
    for section in chapter.sections:
        if section.index <= up_to_section:
            parts.append(section.content)

    return "\n".join(parts)


def build_and_check(
    chapter_name: str,
    volume: str,
    format_type: str,
    log_file: Path,
) -> tuple[bool, float, str]:
    """
    Run a build and return (success, duration, error_snippet).
    """
    book_dir = SCRIPT_DIR.parents[2] / "quarto"

    cmd = [
        "./binder",
        format_type,
        chapter_name,
        f"--{volume}",
        "-v",
    ]

    # Determine output file
    if format_type == "pdf":
        output_path = book_dir / "_build" / f"pdf-{volume}" / "Machine-Learning-Systems.pdf"
    elif format_type == "epub":
        output_path = book_dir / "_build" / f"epub-{volume}" / "Machine-Learning-Systems.epub"
    elif format_type == "html":
        output_path = book_dir / "_build" / f"html-{volume}" / "index.html"
    else:
        output_path = None

    # Delete previous output to ensure clean test
    if output_path and output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=book_dir.parent,  # Run from book/ directory
            capture_output=True,
            text=True,
            timeout=600,
        )
        duration = time.time() - start

        # Write full log
        full_output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        full_output += f"=== TIMESTAMP ===\n{datetime.now().isoformat()}\n\n"
        full_output += f"=== STDOUT ===\n{result.stdout}\n\n"
        full_output += f"=== STDERR ===\n{result.stderr}\n\n"
        full_output += f"=== EXIT CODE ===\n{result.returncode}\n"
        full_output += f"=== DURATION ===\n{duration:.1f}s\n"
        log_file.write_text(full_output)

        # Check if output file was created
        if output_path and output_path.exists():
            return True, duration, ""
        else:
            combined = result.stdout + result.stderr
            error_lines = combined.strip().split("\n")[-20:]
            return False, duration, "\n".join(error_lines)

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        log_file.write_text(f"TIMEOUT after {duration:.0f}s\nCommand: {' '.join(cmd)}")
        return False, duration, "TIMEOUT: Build exceeded 10 minutes"
    except Exception as e:
        duration = time.time() - start
        log_file.write_text(f"EXCEPTION: {e}\nCommand: {' '.join(cmd)}")
        return False, duration, f"EXCEPTION: {e}"


def run_linear_search(
    chapter_name: str,
    volume: str,
    format_type: str,
    chapter: ChapterStructure,
    qmd_path: Path,
    log_dir: Path,
    verbose: bool,
) -> int | None:
    """
    Progressively add sections until build fails. Returns failing section index or None.
    """
    total_steps = len(chapter.sections) + 1  # +1 for preamble-only step
    failing_section = None

    for step in range(total_steps):
        section_idx = step - 1  # -1 = preamble only, 0 = first section, etc.

        if section_idx < 0:
            label = "Preamble only"
            safe_name = "preamble"
        else:
            sec = chapter.sections[section_idx]
            label = sec.title
            safe_name = f"section_{section_idx:02d}"

        log_file = log_dir / f"step_{step:02d}_{safe_name}.log"

        print(f"  [{step + 1}/{total_steps}] {'+ ' if step > 0 else ''}{label}...", end=" ", flush=True)

        # Write progressive content to the chapter file
        content = assemble_progressive_content(chapter, section_idx)
        qmd_path.write_text(content, encoding="utf-8")

        success, duration, error = build_and_check(chapter_name, volume, format_type, log_file)

        if success:
            print(f"PASS ({duration:.1f}s)")
        else:
            print(f"FAIL ({duration:.1f}s)")
            if verbose and error:
                for line in error.strip().split("\n")[-5:]:
                    print(f"         {line}")
            failing_section = section_idx
            break

    return failing_section


def run_binary_search(
    chapter_name: str,
    volume: str,
    format_type: str,
    chapter: ChapterStructure,
    qmd_path: Path,
    log_dir: Path,
    verbose: bool,
) -> int | None:
    """
    Binary search for the failing section. Returns failing section index or None.
    """
    num_sections = len(chapter.sections)

    # First, test preamble only
    print(f"  [pre] Preamble only...", end=" ", flush=True)
    content = assemble_progressive_content(chapter, -1)
    qmd_path.write_text(content, encoding="utf-8")
    log_file = log_dir / "binary_preamble.log"
    success, duration, _ = build_and_check(chapter_name, volume, format_type, log_file)
    if not success:
        print(f"FAIL ({duration:.1f}s)")
        print("\n  Preamble itself fails to build. Cannot isolate section.")
        return -1
    print(f"PASS ({duration:.1f}s)")

    # Then test full chapter to confirm it actually fails
    print(f"  [full] All {num_sections} sections...", end=" ", flush=True)
    content = assemble_progressive_content(chapter, num_sections - 1)
    qmd_path.write_text(content, encoding="utf-8")
    log_file = log_dir / "binary_full.log"
    success, duration, _ = build_and_check(chapter_name, volume, format_type, log_file)
    if success:
        print(f"PASS ({duration:.1f}s)")
        print("\n  Full chapter builds successfully. No failure to isolate.")
        return None
    print(f"FAIL ({duration:.1f}s)")

    # Binary search between 0 and num_sections-1
    lo, hi = 0, num_sections - 1
    build_count = 2  # Already did preamble + full

    while lo < hi:
        mid = (lo + hi) // 2
        sec = chapter.sections[mid]
        build_count += 1

        print(f"  [bisect {build_count}] Up to section {mid}: \"{sec.title}\"...", end=" ", flush=True)

        content = assemble_progressive_content(chapter, mid)
        qmd_path.write_text(content, encoding="utf-8")
        log_file = log_dir / f"binary_step_{build_count:02d}_upto_{mid}.log"
        success, duration, _ = build_and_check(chapter_name, volume, format_type, log_file)

        if success:
            print(f"PASS ({duration:.1f}s)")
            lo = mid + 1
        else:
            print(f"FAIL ({duration:.1f}s)")
            hi = mid

    return lo


def main():
    parser = argparse.ArgumentParser(
        description="Build a chapter section-by-section to isolate build failures"
    )
    parser.add_argument("--chapter", required=True, help="Chapter name (e.g., dl_primer)")
    parser.add_argument("--vol1", action="store_true", help="Volume 1")
    parser.add_argument("--vol2", action="store_true", help="Volume 2")
    parser.add_argument("--format", default="pdf", choices=["pdf", "html", "epub"],
                        help="Build format (default: pdf)")
    parser.add_argument("--binary-search", action="store_true",
                        help="Use binary search instead of linear (fewer builds)")
    parser.add_argument("--log-dir", type=str, help="Custom log directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show error snippets on failure")
    args = parser.parse_args()

    if not args.vol1 and not args.vol2:
        print("Please specify --vol1 or --vol2")
        sys.exit(1)

    volume = "vol1" if args.vol1 else "vol2"
    chapter_name = args.chapter
    format_type = args.format
    search_mode = "binary" if args.binary_search else "linear"

    # Find chapter file
    try:
        qmd_path = find_chapter_qmd(chapter_name, volume)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set up log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = SCRIPT_DIR / "logs" / volume / chapter_name / "section_debug"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Parse the chapter into sections
    print("=" * 70)
    print(f"  SECTION-BY-SECTION BUILD DEBUG ({chapter_name}, {volume.upper()}, {format_type.upper()})")
    print("=" * 70)
    print(f"  File: {qmd_path}")
    print(f"  Mode: {search_mode} search")
    print(f"  Logs: {log_dir}")
    print()

    print("  Splitting chapter into sections...")
    chapter = split_chapter(str(qmd_path))
    num_sections = len(chapter.sections)
    print(f"  Found {num_sections} sections:")
    for i, sec in enumerate(chapter.sections):
        lines = f"L{sec.start_line}-{sec.end_line}"
        print(f"    {i:2d}. {sec.title:<50s} ({lines})")
    print()

    # Back up the original file
    backup_path = qmd_path.with_suffix(".qmd.debug_backup")
    shutil.copy2(qmd_path, backup_path)

    # Ensure we always restore the original file
    def restore_original(signum=None, frame=None):
        if backup_path.exists():
            shutil.copy2(backup_path, qmd_path)
            backup_path.unlink()
        if signum is not None:
            print("\n  Interrupted. Original file restored.")
            sys.exit(1)

    signal.signal(signal.SIGINT, restore_original)
    signal.signal(signal.SIGTERM, restore_original)

    try:
        if args.binary_search:
            failing_idx = run_binary_search(
                chapter_name, volume, format_type, chapter, qmd_path, log_dir, args.verbose
            )
        else:
            failing_idx = run_linear_search(
                chapter_name, volume, format_type, chapter, qmd_path, log_dir, args.verbose
            )
    finally:
        restore_original()

    # Report results
    print()
    print("=" * 70)
    if failing_idx is None:
        print("  RESULT: All sections build successfully")
    elif failing_idx == -1:
        print("  RESULT: Preamble itself fails (issue is before first ## section)")
        print(f"  Check: YAML frontmatter and content before first ## heading")
    else:
        sec = chapter.sections[failing_idx]
        print(f"  RESULT: Build breaks at section {failing_idx}: \"{sec.title}\"")
        print(f"  Lines: {sec.start_line}-{sec.end_line} in {qmd_path.name}")
        if sec.section_id:
            print(f"  ID: #{sec.section_id}")
        print(f"  Logs: {log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
