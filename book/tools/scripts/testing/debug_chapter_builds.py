#!/usr/bin/env python3
"""
Build chapters one-by-one to identify which ones have build errors.

Usage:
    python debug_chapter_builds.py --vol1          # Test all vol1 chapters
    python debug_chapter_builds.py --vol2          # Test all vol2 chapters
    python debug_chapter_builds.py --vol1 --format html  # Test HTML instead of PDF
    python debug_chapter_builds.py --vol1 -v       # Verbose output + save build artifacts
    
Log files are written to: book/tools/scripts/testing/logs/<volume>/<chapter>/
Build artifacts (index.tex, index.log) are saved to each chapter's folder.
"""

import subprocess
import sys
import time
import re
import shutil
from pathlib import Path
from datetime import datetime


def get_chapters_from_config(volume: str) -> list[str]:
    """
    Read chapter list from the PDF config file.
    Extracts chapter names from both commented and uncommented lines.
    """
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parents[2] / "quarto" / "config"
    
    config_file = config_dir / f"_quarto-pdf-{volume}.yml"
    
    if not config_file.exists():
        print(f"Warning: Config file not found: {config_file}")
        return []
    
    content = config_file.read_text()
    
    # Pattern to match chapter paths like:
    # - contents/vol1/introduction/introduction.qmd
    # - contents/vol1/optimizations/model_compression.qmd
    # - contents/vol1/backmatter/appendix_math_foundations.qmd
    # # - contents/vol1/introduction/introduction.qmd
    # Captures the .qmd filename (without extension) as the chapter name
    pattern = rf'#?\s*-\s*contents/{volume}/[^/]+/([^/]+)\.qmd'
    
    # Exclude these non-chapter files
    exclude = {"index", "references", "glossary", "foreword", "about", 
               "acknowledgements", "foundations_principles", "build_principles",
               "optimize_principles", "deploy_principles"}
    
    chapters = []
    for match in re.finditer(pattern, content):
        chapter = match.group(1)
        if chapter not in chapters and chapter not in exclude:
            chapters.append(chapter)
    
    return chapters


def get_all_chapters_from_directory(volume: str) -> list[str]:
    """
    Fallback: Get all chapter directories from the filesystem.
    """
    script_dir = Path(__file__).resolve().parent
    contents_dir = script_dir.parents[2] / "quarto" / "contents" / volume
    
    if not contents_dir.exists():
        return []
    
    # Exclude special directories
    exclude = {"parts", "frontmatter", "backmatter", "index"}
    
    chapters = []
    for item in sorted(contents_dir.iterdir()):
        if item.is_dir() and item.name not in exclude:
            # Check if it has a matching .qmd file
            qmd_file = item / f"{item.name}.qmd"
            if qmd_file.exists():
                chapters.append(item.name)
    
    return chapters


def build_chapter(
    chapter: str, 
    volume: str, 
    format_type: str, 
    log_dir: Path,
) -> tuple[bool, float, str, int, int, Path]:
    """
    Build a single chapter and return (success, duration, error_output, warning_count, error_count, chapter_log_dir).
    Full output is written to chapter-specific log folder.
    """
    book_dir = Path(__file__).resolve().parents[3] / "quarto"
    
    # Create chapter-specific log directory
    chapter_log_dir = log_dir / volume / chapter
    chapter_log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "./binder",
        format_type,
        chapter,
        f"--{volume}",
        "-v",  # Always use verbose for full output capture (including "Output created:" line)
    ]
    
    log_file = chapter_log_dir / f"build_{format_type}.log"
    
    # Determine the output file path and delete it before building
    # This ensures we're checking if THIS build created the file, not a previous one
    # Note: The PDF filename can vary (e.g., "Machine-Learning-Systems.pdf" or 
    # "Introduction-to-Machine-Learning-Systems.pdf"), so we use glob to find any PDF
    output_dir = book_dir / "_build" / f"{format_type}-{volume}"
    if format_type == "pdf":
        output_pattern = "*.pdf"
    elif format_type == "epub":
        output_pattern = "*.epub"
    elif format_type == "html":
        output_pattern = "index.html"
    else:
        output_pattern = None
    
    # Delete any existing output files (to ensure clean test)
    if output_pattern and output_dir.exists():
        for f in output_dir.glob(output_pattern):
            try:
                f.unlink()
            except Exception:
                pass  # Ignore deletion errors
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=book_dir.parent,  # Run from book/ directory
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per chapter
        )
        duration = time.time() - start
        
        # Combine stdout and stderr
        full_output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        full_output += f"=== TIMESTAMP ===\n{datetime.now().isoformat()}\n\n"
        full_output += f"=== STDOUT ===\n{result.stdout}\n\n"
        full_output += f"=== STDERR ===\n{result.stderr}\n\n"
        full_output += f"=== EXIT CODE ===\n{result.returncode}\n"
        full_output += f"=== DURATION ===\n{duration:.1f}s\n"
        
        # Write full output to log file
        log_file.write_text(full_output)
        
        # Copy build artifacts (tex, pdf, logs) to chapter log folder
        _copy_build_artifacts(book_dir, chapter_log_dir, format_type, volume)
        
        # Count warnings and errors
        combined = result.stdout + result.stderr
        warning_count = combined.lower().count('warning')
        error_count = combined.lower().count('error')
        
        # Check for specific problematic patterns
        has_fenced_div_warning = ':::' in combined and 'fenced div' in combined.lower()
        has_tikz_error = 'Gscale@box' in combined or 'Emergency stop' in combined
        has_duplicate_footnote = 'Duplicate note reference' in combined
        
        # Ultimate test: did the output file get created?
        # Ignore exit code - only check if any matching output file exists
        output_files = list(output_dir.glob(output_pattern)) if output_pattern and output_dir.exists() else []
        if output_files:
            return True, duration, "", warning_count, error_count, chapter_log_dir
        else:
            # Build failed - capture last 50 lines of output for error summary
            error_lines = combined.strip().split('\n')
            error_summary = '\n'.join(error_lines[-50:])
            file_type = format_type.upper()
            return False, duration, f"{file_type} not created in {output_dir}\n\n{error_summary}", warning_count, error_count + 1, chapter_log_dir
            
    except subprocess.TimeoutExpired:
        log_file.write_text(f"TIMEOUT: Build exceeded 10 minutes\nCommand: {' '.join(cmd)}")
        return False, 600, "TIMEOUT: Build exceeded 10 minutes", 0, 1, chapter_log_dir
    except Exception as e:
        log_file.write_text(f"EXCEPTION: {str(e)}\nCommand: {' '.join(cmd)}")
        return False, time.time() - start, f"EXCEPTION: {str(e)}", 0, 1, chapter_log_dir


def _copy_build_artifacts(book_dir: Path, chapter_log_dir: Path, format_type: str, volume: str) -> None:
    """
    Copy build artifacts from build directory to chapter log folder.
    For PDF: copies Machine-Learning-Systems.tex and Machine-Learning-Systems.pdf
    """
    # Create artifacts subdirectory
    artifacts_dir = chapter_log_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    if format_type == "pdf":
        # Primary artifact: any .tex file in quarto/ directory (name varies)
        for tex_file in book_dir.glob("*.tex"):
            try:
                dst = artifacts_dir / tex_file.name
                shutil.copy2(tex_file, dst)
                copied_files.append(tex_file.name)
            except Exception as e:
                (artifacts_dir / f"{tex_file.name}.error").write_text(f"Failed to copy: {e}")
        
        # Copy any generated PDF (name varies)
        pdf_dir = book_dir / "_build" / f"pdf-{volume}"
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                try:
                    dst = artifacts_dir / pdf_file.name
                    shutil.copy2(pdf_file, dst)
                    copied_files.append(pdf_file.name)
                except Exception as e:
                    (artifacts_dir / f"{pdf_file.name}.error").write_text(f"Failed to copy: {e}")
        
        # Also copy any .log files from quarto/ (LaTeX logs)
        for log_file in book_dir.glob("*.log"):
            try:
                dst = artifacts_dir / log_file.name
                if not dst.exists():
                    shutil.copy2(log_file, dst)
                    copied_files.append(log_file.name)
            except Exception:
                pass
    
    elif format_type == "html":
        # For HTML, check _build/html-* directories
        for html_build in book_dir.glob("_build/html-*"):
            index_html = html_build / "index.html"
            if index_html.exists():
                try:
                    dst = artifacts_dir / "index.html"
                    shutil.copy2(index_html, dst)
                    copied_files.append("index.html")
                except Exception:
                    pass
                break
    
    elif format_type == "epub":
        # For EPUB, check _build/epub-* directories
        for epub_build in book_dir.glob("_build/epub-*"):
            for epub_file in epub_build.glob("*.epub"):
                try:
                    dst = artifacts_dir / epub_file.name
                    shutil.copy2(epub_file, dst)
                    copied_files.append(epub_file.name)
                except Exception:
                    pass
                break
    
    if copied_files:
        # Write a summary file
        (artifacts_dir / "_artifacts_copied.txt").write_text(
            f"Copied {len(copied_files)} build artifacts from {book_dir}\n"
            f"Files: {', '.join(copied_files)}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
        )


def analyze_log_for_issues(log_file: Path) -> dict:
    """Analyze a log file for specific issues."""
    if not log_file.exists():
        return {}
    
    content = log_file.read_text()
    issues = {
        "fenced_div_warnings": content.count("fenced div"),
        "duplicate_footnotes": content.count("Duplicate note reference"),
        "tikz_errors": 1 if ("Gscale@box" in content or "Emergency stop" in content) else 0,
        "latex_errors": content.count("! "),
        "total_warnings": content.lower().count("[warning]") + content.lower().count("warning:"),
    }
    return issues


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build chapters one-by-one to find errors")
    parser.add_argument("--vol1", action="store_true", help="Test Volume 1 chapters")
    parser.add_argument("--vol2", action="store_true", help="Test Volume 2 chapters")
    parser.add_argument("--format", default="pdf", choices=["pdf", "html", "epub"], 
                        help="Build format (default: pdf)")
    parser.add_argument("--start-from", type=str, help="Start from this chapter (skip earlier ones)")
    parser.add_argument("--only", type=str, help="Only test these chapters (comma-separated)")
    parser.add_argument("--log-dir", type=str, help="Custom log directory")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Show extra status info (verbose build output is always captured in logs)")
    args = parser.parse_args()
    
    if not args.vol1 and not args.vol2:
        print("Please specify --vol1 or --vol2")
        sys.exit(1)
    
    # Create log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine chapters to test
    if args.vol1:
        volume = "vol1"
    else:
        volume = "vol2"
    
    # Get chapters from config file or fallback to directory scan
    chapters = get_chapters_from_config(volume)
    if not chapters:
        print(f"  Falling back to directory scan for {volume}...")
        chapters = get_all_chapters_from_directory(volume)
    
    if not chapters:
        print(f"Error: No chapters found for {volume}")
        sys.exit(1)
    
    # Filter chapters if --only specified
    # Note: --only bypasses the normal exclusion list, allowing any chapter to be built
    if args.only:
        only_chapters = [c.strip() for c in args.only.split(',')]
        # Use the specified chapters directly (bypass exclusion filter)
        chapters = only_chapters
    
    # Skip chapters if --start-from specified
    if args.start_from:
        try:
            start_idx = chapters.index(args.start_from)
            chapters = chapters[start_idx:]
        except ValueError:
            print(f"Warning: Chapter '{args.start_from}' not found, testing all chapters")
    
    print("=" * 70)
    print(f"  CHAPTER-BY-CHAPTER BUILD TEST ({volume.upper()}, {args.format.upper()})")
    print("=" * 70)
    print(f"  Testing {len(chapters)} chapters...")
    print(f"  Log directory: {log_dir}")
    print(f"  Verbose build output: always captured in logs")
    print(f"  Build artifacts: always saved (Machine-Learning-Systems.tex, etc.)")
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for i, chapter in enumerate(chapters, 1):
        print(f"[{i}/{len(chapters)}] Building {chapter}...", end=" ", flush=True)
        
        success, duration, error, warn_count, err_count, chapter_log_dir = build_chapter(
            chapter, volume, args.format, log_dir
        )
        
        log_file = chapter_log_dir / f"build_{args.format}.log"
        issues = analyze_log_for_issues(log_file)
        
        # Check if artifacts were saved
        artifacts_dir = chapter_log_dir / "artifacts"
        has_artifacts = artifacts_dir.exists() and any(artifacts_dir.iterdir())
        
        status_parts = []
        if issues.get("fenced_div_warnings", 0) > 0:
            status_parts.append(f"⚠️ {issues['fenced_div_warnings']} fenced-div")
        if issues.get("duplicate_footnotes", 0) > 0:
            status_parts.append(f"⚠️ {issues['duplicate_footnotes']} dup-footnotes")
        if issues.get("tikz_errors", 0) > 0:
            status_parts.append("🔴 TikZ error")
        if has_artifacts:
            status_parts.append("📁 artifacts")
        
        extra_info = f" [{', '.join(status_parts)}]" if status_parts else ""
        
        if success:
            print(f"✅ PASSED ({duration:.1f}s){extra_info}")
            passed += 1
            results.append((chapter, "PASS", duration, "", issues, chapter_log_dir))
        else:
            print(f"❌ FAILED ({duration:.1f}s){extra_info}")
            failed += 1
            results.append((chapter, "FAIL", duration, error, issues, chapter_log_dir))
    
    # Print summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total: {len(chapters)} chapters")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Log directory: {log_dir}")
    print()
    
    # Show chapters with specific issues
    chapters_with_issues = []
    for chapter, status, duration, error, issues, chapter_log_dir in results:
        if any(v > 0 for v in issues.values()):
            chapters_with_issues.append((chapter, issues, chapter_log_dir))
    
    if chapters_with_issues:
        print("  CHAPTERS WITH WARNINGS/ISSUES:")
        print("  " + "-" * 50)
        for chapter, issues, chapter_log_dir in chapters_with_issues:
            issue_strs = []
            if issues.get("fenced_div_warnings", 0) > 0:
                issue_strs.append(f"fenced-div: {issues['fenced_div_warnings']}")
            if issues.get("duplicate_footnotes", 0) > 0:
                issue_strs.append(f"dup-footnotes: {issues['duplicate_footnotes']}")
            if issues.get("tikz_errors", 0) > 0:
                issue_strs.append("TikZ error")
            if issues.get("latex_errors", 0) > 0:
                issue_strs.append(f"LaTeX errors: {issues['latex_errors']}")
            print(f"    {chapter}: {', '.join(issue_strs)}")
            print(f"      Log: {chapter_log_dir / f'build_{args.format}.log'}")
            artifacts_dir = chapter_log_dir / "artifacts"
            if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
                print(f"      Artifacts: {artifacts_dir}")
        print()
    
    if failed > 0:
        print("  FAILED CHAPTERS:")
        print("  " + "-" * 40)
        for chapter, status, duration, error, issues, chapter_log_dir in results:
            if status == "FAIL":
                print(f"    ❌ {chapter}")
                print(f"       Log: {chapter_log_dir / f'build_{args.format}.log'}")
                artifacts_dir = chapter_log_dir / "artifacts"
                if artifacts_dir.exists():
                    tex_files = list(artifacts_dir.glob("**/*.tex"))
                    log_files = list(artifacts_dir.glob("**/*.log"))
                    if tex_files or log_files:
                        print(f"       Artifacts: {len(tex_files)} .tex, {len(log_files)} .log files in {artifacts_dir}")
        print()
        
        # Show brief error details for failed chapters
        print("  ERROR SNIPPETS (see log files for full output):")
        print("  " + "-" * 40)
        for chapter, status, duration, error, issues, chapter_log_dir in results:
            if status == "FAIL" and error:
                print(f"\n  === {chapter} ===")
                # Show last 15 lines of error
                error_lines = error.strip().split('\n')[-15:]
                for line in error_lines:
                    print(f"    {line}")
    
    print()
    print("=" * 70)
    print(f"  Full logs available in: {log_dir}/{volume}/<chapter>/")
    print(f"  Build artifacts saved in: <chapter>/artifacts/")
    print("=" * 70)
    
    # Exit with error code if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
