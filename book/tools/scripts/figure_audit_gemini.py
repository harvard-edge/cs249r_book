#!/usr/bin/env python3
"""
Figure-Narrative Audit via Gemini CLI (Multimodal).

Architecture:
  1. `binder info figures` → CSV with fig-id, caption, alt-text, chapter, file
  2. Rendered HTML (_build/) → actual image paths keyed by fig-id
  3. QMD source → surrounding prose context for each figure
  4. Gemini CLI → multimodal review comparing image vs. text

Prerequisites:
    ./book/binder build html --vol1 -v
    ./book/binder build html --vol2 -v

Usage:
    # Full audit (both volumes, parallel):
    python3 book/tools/scripts/figure_audit_gemini.py

    # Single volume:
    python3 book/tools/scripts/figure_audit_gemini.py --vol1

    # Single chapter:
    python3 book/tools/scripts/figure_audit_gemini.py --chapter vol1/introduction

    # Dry-run (extract + write prompts only):
    python3 book/tools/scripts/figure_audit_gemini.py --dry-run

    # Custom parallelism and model:
    python3 book/tools/scripts/figure_audit_gemini.py --parallel 4 --model gemini-2.5-flash
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
BINDER = REPO_ROOT / "book" / "binder"
BUILD_DIR = REPO_ROOT / "book" / "quarto" / "_build"
CONTENTS_DIR = REPO_ROOT / "book" / "quarto" / "contents"
OUTPUT_DIR = REPO_ROOT / ".claude" / "_reviews" / "figure_audit"
DEFAULT_MODEL = "gemini-2.5-pro"
BATCH_SIZE = 16  # larger batches = fewer API calls = less rate limiting
PROSE_CONTEXT_LINES = 15
RATE_LIMIT_DELAY = 15  # seconds between API calls to stay under 5 RPM


# ── Data ───────────────────────────────────────────────────────────────

@dataclass
class Figure:
    fig_id: str
    chapter: str
    label: str
    caption: str
    alt_text: str
    source_type: str       # div | image | code-cell
    qmd_file: str
    image_path: str = ""   # absolute path to rendered image
    prose_context: str = ""
    fig_number: str = ""   # e.g. "Figure 14" from PDF
    page: str = ""


# ── Step 1: Extract figures via binder ─────────────────────────────────

def extract_figures_via_binder(vol_flag: str) -> list[Figure]:
    """Run `binder info figures` and parse the CSV output."""
    tmp_csv = OUTPUT_DIR / f"_tmp_{vol_flag.strip('-')}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(BINDER),
        "info", "figures",
        vol_flag,
        "--format", "csv",
        "--output", str(tmp_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR running binder: {result.stderr[:200]}")
        return []

    figures = []
    with open(tmp_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            figures.append(Figure(
                fig_id=row.get("id", ""),
                chapter=row.get("chapter", ""),
                label=row.get("label", ""),
                caption=row.get("caption", ""),
                alt_text=row.get("alt_text", ""),
                source_type=row.get("source", ""),
                qmd_file=row.get("file", ""),
                fig_number=row.get("fig_number", ""),
                page=row.get("page", ""),
            ))
    tmp_csv.unlink(missing_ok=True)
    return figures


# ── Step 2: Build image index ─────────────────────────────────────────

# Staging area for images — outside _build/ so Gemini CLI can read them
STAGE_DIR = REPO_ROOT / "book" / "quarto" / "_figure_audit_images"


def build_image_index() -> dict[str, str]:
    """
    Two-pass approach to get accessible image paths for every figure:

    Pass 1: Extract SOURCE image paths from QMD files (not gitignored).
            Works for static images (![](images/png/xxx.png)).

    Pass 2: For TikZ/Python figures that only exist in _build/, copy
            them to a staging directory outside _build/ so Gemini can read them.
    """
    index = {}

    # Pass 1: Source images from QMD files
    QUARTO_DIR = REPO_ROOT / "book" / "quarto"
    for qmd_file in (QUARTO_DIR / "contents").rglob("*.qmd"):
        lines = qmd_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            fig_match = re.match(r'^:{3,}\s*\{#(fig-[\w-]+)', line)
            if fig_match:
                fig_id = fig_match.group(1)
                # Look for ![](images/...) in the next few lines
                for k in range(i + 1, min(i + 8, len(lines))):
                    img_match = re.search(r'!\[.*?\]\((images/[^)]+)\)', lines[k])
                    if img_match:
                        img_rel = img_match.group(1)
                        img_abs = (qmd_file.parent / img_rel).resolve()
                        if img_abs.exists():
                            index[fig_id] = str(img_abs)
                        break

    # Pass 2: Rendered HTML for TikZ/Python figures not found in Pass 1
    missing_ids = set()
    html_index = {}  # fig-id → absolute path in _build/

    for html_file in BUILD_DIR.rglob("*.html"):
        if html_file.name == "figure_review.html":
            continue
        try:
            content = html_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in re.finditer(
            r'<div\s+id="(fig-[\w-]+)"[^>]*class="[^"]*quarto-float[^"]*"[^>]*>'
            r'.*?<img\s+src="([^"]+)"',
            content, re.DOTALL,
        ):
            fig_id = m.group(1)
            img_src = m.group(2)
            img_abs = (html_file.parent / img_src).resolve()
            if img_abs.exists() and fig_id not in html_index:
                html_index[fig_id] = img_abs

    # Copy missing figures to staging dir
    need_staging = {fid: p for fid, p in html_index.items() if fid not in index}
    if need_staging:
        STAGE_DIR.mkdir(parents=True, exist_ok=True)
        for fig_id, src_path in need_staging.items():
            dest = STAGE_DIR / f"{fig_id}{src_path.suffix}"
            if not dest.exists() or dest.stat().st_mtime < src_path.stat().st_mtime:
                import shutil
                shutil.copy2(src_path, dest)
            index[fig_id] = str(dest)

    return index


# ── Step 3: Extract prose context from QMD ─────────────────────────────

def get_prose_context(qmd_file: str, fig_id: str) -> str:
    """
    Get ALL prose where the figure is referenced in the chapter.

    Finds every line that contains @fig-xxx (the cross-reference) and
    collects the surrounding paragraph for each occurrence. Also includes
    the prose immediately before the figure definition itself.
    """
    # binder outputs paths relative to book/quarto/ (e.g. contents/vol1/...)
    path = REPO_ROOT / "book" / "quarto" / qmd_file
    if not path.exists():
        path = REPO_ROOT / qmd_file
    if not path.exists():
        return ""

    lines = path.read_text(encoding="utf-8").splitlines()
    prose_chunks = []

    # Pattern 1: Find every @fig-xxx cross-reference in body prose
    ref_pattern = f"@{fig_id}"
    for i, line in enumerate(lines):
        if ref_pattern in line and not line.strip().startswith(":::"):
            # Grab the paragraph: expand up and down to blank lines
            start = i
            while start > 0 and lines[start - 1].strip():
                start -= 1
            end = i
            while end < len(lines) - 1 and lines[end + 1].strip():
                end += 1
            para = " ".join(
                l.strip() for l in lines[start:end + 1]
                if l.strip() and not l.strip().startswith("```")
            )
            if para:
                prose_chunks.append(f"[Ref at line {i+1}]: {para}")

    # Pattern 2: Prose immediately before the figure div definition
    for i, line in enumerate(lines):
        if f"#{fig_id}" in line and line.strip().startswith(":::"):
            start = max(0, i - PROSE_CONTEXT_LINES)
            before = []
            for p in range(start, i):
                s = lines[p].strip()
                if s and not s.startswith(":::") and not s.startswith("```"):
                    before.append(s)
            if before:
                prose_chunks.append(f"[Before definition]: {' '.join(before)}")
            break

    text = "\n\n".join(prose_chunks)
    return text[-3000:] if len(text) > 3000 else text


# ── Step 4: Gemini prompt + execution ──────────────────────────────────

def build_prompt(batch_label: str, figures: list[Figure]) -> str:
    """Build audit prompt for a batch of figures."""
    tasks = []
    for idx, fig in enumerate(figures, 1):
        tasks.append(f"""
### Figure {idx}: `{fig.fig_id}` ({fig.qmd_file})

**Image file**: {fig.image_path}
**Current caption**: {fig.caption}
**Current alt-text**: {fig.alt_text}
**Surrounding prose**: {fig.prose_context[:1500]}

-> Use the read_file tool to open `{fig.image_path}`.
-> Then answer:

1. **WHAT THE FIGURE ACTUALLY SHOWS**: Read every label, number, axis title, legend entry,
   and annotation visible in the image. List them precisely.

2. **FACTUAL ERRORS**: Compare the labels/numbers you just listed against THREE sources:
   the caption, the alt-text, AND the surrounding prose. Report ONLY factual mismatches:

   CAPTION errors:
   - Caption says "4x speedup" but figure shows 2.5x
   - Caption says "FP32 to INT8" but figure only shows integer operations
   - Caption references "red circles" but the data series uses blue circles

   ALT-TEXT errors:
   - Alt-text says "96 images" but figure has 40
   - Alt-text says "stacked bars" but they are grouped bars
   - Alt-text describes elements (arrows, panels, labels) that do not exist

   PROSE errors (check the surrounding prose carefully):
   - Prose says "as shown in the figure, latency drops by 75%" but figure shows 50%
   - Prose says "the red line indicates..." but the figure uses blue
   - Prose says "three phases" but the figure shows four
   - Prose cites a specific number (percentage, speedup, count) that the figure contradicts
   - Prose describes the figure's structure incorrectly (e.g., "left panel" when there is no panel)

   Do NOT flag: writing style, word choice, phrasing preferences, or prose polish.
   Only flag things where the TEXT IS WRONG about what the FIGURE SHOWS.

3. **VERDICT**:
   - PASS = no factual errors between figure and text
   - ERROR = caption, alt-text, or prose makes a factual claim the figure contradicts
   - FIGURE_ISSUE = the figure itself appears wrong/corrupted/placeholder (red flag for author)

4. **FIX** (only if ERROR or FIGURE_ISSUE):

   IMPORTANT: We fix the TEXT to match the FIGURE. We do NOT change figures.
   The figure is the source of truth. Adjust captions, alt-text, and prose to
   accurately describe what the figure actually shows.

   If the figure itself is wrong (e.g., shows wrong data, is a placeholder, is
   corrupted), mark as FIGURE_ISSUE — the author must update the figure.

   When writing fixes, follow these MIT Press rules:
   - Caption format: **Bold Title**: explanation in sentence case
   - Alt-text: max 250 characters, objective description only, no interpretation
   - Alt-text concept terms lowercase: "machine learning", "iron law", "transformer",
     "memory wall" (NOT capitalized). Keep proper nouns capitalized: ImageNet, BERT,
     GPT-4, AlexNet, ResNet, PyTorch
   - Write "vs." with period
   - Spell out one through nine; digits for 10+; always digits with units (3 GB, 7 ms)
   - Spell out "percent" in captions (not %)
   - No fig-cap="" or fig-alt="" wrapper — just the raw text

   For each error provide:
   - For caption: the corrected caption (full replacement)
   - For alt-text: the corrected alt-text (full replacement, max 250 chars)
   - For prose: quote the wrong sentence, then write the corrected version
   - If PASS: "No fix needed."
""")

    return f"""# Figure-Narrative Audit: {batch_label}

You are a fact-checker auditing a textbook before publication. Your ONLY job is to find
cases where the text (caption, alt-text, or prose) makes a FACTUAL CLAIM that contradicts
what the figure actually shows.

You are NOT a copy editor. Do NOT flag:
- Writing style or word choice
- Phrasing that could be "better"
- Capitalization or formatting
- Prose that is technically correct but could be clearer

You ARE a fact-checker. DO flag:
- Caption says "4x speedup" but figure shows 2.5x
- Caption says "FP32" but figure only shows integer operations
- Alt-text says "96 images" but figure has 40
- Alt-text says "stacked bars" but they are grouped side-by-side
- Caption says "red circles" but the data uses blue circles
- Prose says "reduces by 75%" but figure shows a 50% reduction
- Alt-text describes arrows, panels, or labels that do not exist in the image
- Caption references a specific model/dataset not shown in the figure

Think of github issue #1318: a caption claimed "Moving from FP32 to INT8 reduces inference
time by up to 4 times" but the actual bar chart showed reductions of only 1.1x to 2.5x
depending on the model. THAT is the kind of error you are looking for.

For SVGs, you will see XML markup — extract labels, text elements, and structure from it.

---
{"".join(tasks)}
---

## SUMMARY

| fig-id | Verdict | Issue | Fix |
|--------|---------|-------|-----|
(one row per figure)
"""


def run_batch(prompt: str, model: str) -> tuple[str, int]:
    """Run one Gemini CLI call. Returns (stdout, effective_rc)."""
    cmd = [
        "gemini", "-m", model, "--yolo",
        "-p", "Follow the instructions provided on stdin.",
    ]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=600, cwd=str(REPO_ROOT),
        )
        has_content = any(k in result.stdout for k in ("VERDICT", "SUMMARY", "PASS"))
        return (result.stdout, 0 if has_content else result.returncode)
    except subprocess.TimeoutExpired:
        return ("[TIMEOUT]", 1)
    except Exception as e:
        return (f"[ERROR: {e}]", 1)


def audit_chapter(
    chapter_key: str,
    figures: list[Figure],
    model: str,
    dry_run: bool,
) -> tuple[str, str, int]:
    """Audit all figures in a chapter, batching as needed."""
    if not figures:
        return (chapter_key, "No figures.", 0)

    valid = [f for f in figures if f.image_path]
    if not valid:
        return (chapter_key, "No rendered images found.", 1)

    safe_key = chapter_key.replace("/", "_")
    prompt_dir = OUTPUT_DIR / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    batches = [valid[i:i+BATCH_SIZE] for i in range(0, len(valid), BATCH_SIZE)]

    if dry_run:
        for bi, batch in enumerate(batches):
            p = build_prompt(f"{chapter_key} (batch {bi+1}/{len(batches)})", batch)
            (prompt_dir / f"{safe_key}_b{bi+1}.md").write_text(p, encoding="utf-8")
        return (chapter_key, f"[DRY] {len(valid)} figs, {len(batches)} batches", 0)

    import time
    outputs, worst = [], 0
    for bi, batch in enumerate(batches):
        # Rate limit: wait between calls to stay under 5 RPM
        if bi > 0:
            time.sleep(RATE_LIMIT_DELAY)

        label = f"{chapter_key} (batch {bi+1}/{len(batches)})"
        prompt = build_prompt(label, batch)
        (prompt_dir / f"{safe_key}_b{bi+1}.md").write_text(prompt, encoding="utf-8")
        out, rc = run_batch(prompt, model)
        outputs.append(out)
        worst = max(worst, rc)

    combined = "\n\n".join(outputs)
    (OUTPUT_DIR / f"{safe_key}_audit.md").write_text(
        f"# Figure Audit: {chapter_key}\n\n"
        f"Model: {model}\nFigures: {len(valid)}\nBatches: {len(batches)}\n\n---\n\n{combined}",
        encoding="utf-8",
    )
    return (chapter_key, combined, worst)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multimodal figure audit via Gemini CLI")
    parser.add_argument("--vol1", action="store_true")
    parser.add_argument("--vol2", action="store_true")
    parser.add_argument("--chapter", help="Single chapter (e.g. vol1/introduction)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Parallel chapters (default: 1, safest for rate limits)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    args = parser.parse_args()

    # Default: both volumes
    if not args.vol1 and not args.vol2 and not args.chapter:
        args.vol1 = args.vol2 = True

    # ── Step 1: Extract via binder ──
    print("Step 1: Extracting figures via binder...\n")
    all_figures: list[Figure] = []
    if args.vol1 or (args.chapter and args.chapter.startswith("vol1")):
        figs = extract_figures_via_binder("--vol1")
        print(f"  Vol1: {len(figs)} figures")
        all_figures.extend(figs)
    if args.vol2 or (args.chapter and args.chapter.startswith("vol2")):
        figs = extract_figures_via_binder("--vol2")
        print(f"  Vol2: {len(figs)} figures")
        all_figures.extend(figs)

    # ── Step 2: Build image index ──
    print("\nStep 2: Building image index from rendered HTML...")
    image_index = build_image_index()
    print(f"  {len(image_index)} fig-id -> image mappings\n")

    # ── Step 3: Enrich figures ──
    print("Step 3: Enriching with image paths and prose context...")
    for fig in all_figures:
        fig.image_path = image_index.get(fig.fig_id, "")
        fig.prose_context = get_prose_context(fig.qmd_file, fig.fig_id)

    matched = sum(1 for f in all_figures if f.image_path)
    print(f"  {matched}/{len(all_figures)} figures matched to rendered images\n")

    # ── Group by chapter ──
    chapters: dict[str, list[Figure]] = {}
    for fig in all_figures:
        # Derive chapter key from qmd_file path
        parts = Path(fig.qmd_file).parts
        # e.g. contents/vol1/introduction/introduction.qmd -> vol1/introduction
        vol_idx = next((i for i, p in enumerate(parts) if p in ("vol1", "vol2")), None)
        if vol_idx is not None and vol_idx + 1 < len(parts):
            key = f"{parts[vol_idx]}/{parts[vol_idx+1]}"
        else:
            key = fig.chapter
        chapters.setdefault(key, []).append(fig)

    # Filter to single chapter if requested
    if args.chapter:
        if args.chapter in chapters:
            chapters = {args.chapter: chapters[args.chapter]}
        else:
            print(f"Chapter '{args.chapter}' not found. Available: {sorted(chapters.keys())}")
            sys.exit(1)

    # Print summary
    for key in sorted(chapters):
        figs = chapters[key]
        n_img = sum(1 for f in figs if f.image_path)
        n_miss = len(figs) - n_img
        status = f"{n_img}/{len(figs)} with images"
        if n_miss:
            status += f" ({n_miss} missing)"
        print(f"  {key}: {status}")

    # ── Step 4: Run Gemini audit ──
    active = {k: v for k, v in chapters.items() if any(f.image_path for f in v)}
    parallel = min(args.parallel, len(active))
    total = sum(len(v) for v in active.values())

    print(f"\nStep 4: Auditing {total} figures across {len(active)} chapters "
          f"(parallel={parallel}, model={args.model})")
    if args.dry_run:
        print("  [DRY RUN — writing prompts only]\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(audit_chapter, k, v, args.model, args.dry_run): k
            for k, v in sorted(active.items())
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                ch, out, rc = future.result()
                status = "done" if rc == 0 else f"error (rc={rc})"
                print(f"  {ch}: {status}")
                results[ch] = (out, rc)
            except Exception as e:
                print(f"  {key}: {e}")
                results[key] = (str(e), 1)

    # ── Summary ──
    summary_path = OUTPUT_DIR / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Figure-Narrative Audit Summary\n\nModel: {args.model}\n\n")
        f.write("| Chapter | Figures | Status |\n|---------|---------|--------|\n")
        for ch in sorted(results):
            n = len(chapters.get(ch, []))
            _, rc = results[ch]
            f.write(f"| {ch} | {n} | {'Done' if rc == 0 else 'Error'} |\n")

    print(f"\nSummary: {summary_path}")
    print(f"Reports: {OUTPUT_DIR}/")
    print(f"\nTo generate review dashboard:")
    print(f"  python3 book/tools/scripts/figure_audit_review.py")


if __name__ == "__main__":
    main()
