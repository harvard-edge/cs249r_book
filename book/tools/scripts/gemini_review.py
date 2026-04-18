#!/usr/bin/env python3
"""
Gemini chapter review pipeline — dispatches parallel reviews of rendered HTML.

Usage:
    python3 gemini_review.py --vol1              # Review vol1 chapters
    python3 gemini_review.py --vol2              # Review vol2 chapters
    python3 gemini_review.py --vol1 --vol2       # Both volumes
    python3 gemini_review.py --chapter introduction --vol1  # Single chapter

Prerequisites: gemini CLI installed, API key configured.
Output: _build/review_{timestamp}/ with per-chapter markdown reports.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

MODEL = "gemini-3.1-pro-preview"
MAX_PARALLEL = 1
STAGGER_SECONDS = 3
MAX_CHARS = 50000  # Truncate HTML text to fit context

BOOK_DIR = Path(__file__).resolve().parents[2] / "quarto"

# Content chapters only (no frontmatter, parts, backmatter)
VOL1_CHAPTERS = [
    "introduction", "ml_systems", "ml_workflow", "data_engineering",
    "nn_computation", "nn_architectures", "frameworks", "training",
    "data_selection", "model_compression", "hw_acceleration", "benchmarking",
    "model_serving", "ml_ops", "responsible_engr", "conclusion",
]

VOL2_CHAPTERS = [
    "introduction", "compute_infrastructure", "distributed_training",
    "collective_communication", "network_fabrics", "data_storage",
    "inference", "performance_engineering", "edge_intelligence",
    "fleet_orchestration", "ops_scale", "fault_tolerance",
    "security_privacy", "robust_ai", "responsible_ai", "sustainable_ai",
    "conclusion",
]

REVIEW_PROMPT = r"""You are an independent reviewer for an MIT Press textbook on Machine Learning Systems. You are reviewing the RENDERED HTML output of a single chapter.

Your job is to find ERRORS and ISSUES. Be thorough but precise — only flag real problems, not style preferences.

## Check these categories (mapped to MIT Press copyedit passes):

### Rendering Issues (CRITICAL — these block delivery)
- Raw LaTeX showing instead of rendered math (e.g., \frac{}, \times, $$...$$)
- Broken cross-references showing as "??" or raw labels like @sec-xxx, @fig-xxx, @tbl-xxx
- Missing or broken figures (alt text visible but no image, or [Figure ?])
- Broken code blocks (raw markdown fences showing)
- Malformed tables (pipes showing, misaligned columns)
- Raw inline Python expressions like `{python} ClassName.attr`

### MIT Press Style Violations (MAJOR)
- "%" symbol in body prose (should be spelled "percent"; OK in tables/equations/code)
- Spaced em dashes " — " (should be closed "—"; no spaces)
- Capitalized concept terms that should be lowercase: "Iron Law" → "iron law", "Degradation Equation" → "degradation equation", "Verification Gap" → "verification gap", "Bitter Lesson" → "bitter lesson" (exception: sentence start, bold definitions, H1/H2 headings)
- "Acknowledgements" instead of "Acknowledgments" (American spelling)
- Contractions in body prose: "can't", "don't", "it's", "we'll" etc.
- Abbreviations NOT expanded on first use in the chapter (e.g., bare "CNN" without prior "convolutional neural network (CNN)")
- "e.g." or "i.e." in running text outside parentheses (should be "for example" / "that is")

### Content Issues (MINOR)
- Placeholder text or TODO markers
- Obviously wrong numbers or broken computed values
- Broken bibliography entries (missing fields in rendered references)

## Output format

For each issue found:

**[SEVERITY] Location: description**
- SEVERITY: CRITICAL / MAJOR / MINOR
- Location: section heading or nearby text snippet
- Brief description of the issue

If the chapter is clean, say:
**NO ISSUES FOUND** — Checked: math rendering, cross-refs, figures, tables, code blocks, em dashes, percent, capitalization, abbreviations, contractions, latin abbreviations.

Review the following chapter:
"""


class HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping scripts/styles/nav."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "svg", "path"}

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def extract_text(html_path: Path) -> str:
    """Extract visible text from an HTML file."""
    extractor = HTMLTextExtractor()
    extractor.feed(html_path.read_text(encoding="utf-8"))
    text = extractor.get_text()
    # Truncate to fit in context
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[... truncated for review ...]"
    return text


def find_html(vol: str, chapter: str) -> Path | None:
    """Find the rendered HTML file for a chapter."""
    build_dir = BOOK_DIR / f"_build/html-{vol}/contents/{vol}"
    # Try common patterns
    candidates = list(build_dir.rglob(f"{chapter}.html"))
    if candidates:
        return candidates[0]
    return None


def review_chapter(vol: str, chapter: str, output_dir: Path, model: str = MODEL) -> dict:
    """Review a single chapter with Gemini."""
    html_path = find_html(vol, chapter)
    if html_path is None:
        return {"vol": vol, "chapter": chapter, "status": "SKIP", "reason": "HTML not found"}

    text = extract_text(html_path)
    if len(text.strip()) < 100:
        return {"vol": vol, "chapter": chapter, "status": "SKIP", "reason": "Empty or too short"}

    output_file = output_dir / f"{vol}_{chapter}.md"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["gemini", "-m", model, "-p", REVIEW_PROMPT, "-o", "text"],
                input=text,
                capture_output=True,
                text=True,
                timeout=180,
            )
            # Filter out stderr noise from stdout (gemini CLI mixes them)
            stdout = result.stdout
            # Remove ERROR lines from gemini CLI internals
            clean_lines = [
                line for line in stdout.split("\n")
                if not line.startswith("ERROR: Failed to fetch")
                and not line.strip().startswith("at ")
                and not line.strip().startswith("at async")
                and not "Gaxios" in line
                and not "googleapis" in line
            ]
            clean_output = "\n".join(clean_lines).strip()

            if clean_output and len(clean_output) > 20:
                output_file.write_text(clean_output, encoding="utf-8")
                return {"vol": vol, "chapter": chapter, "status": "OK", "output": output_file}
            elif attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))  # backoff
                continue
            else:
                output_file.write_text(f"ERROR after {max_retries} retries: empty response", encoding="utf-8")
                return {"vol": vol, "chapter": chapter, "status": "ERROR", "reason": "empty response"}
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return {"vol": vol, "chapter": chapter, "status": "TIMEOUT"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return {"vol": vol, "chapter": chapter, "status": "ERROR", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Gemini chapter review pipeline")
    parser.add_argument("--vol1", action="store_true", help="Review vol1 chapters")
    parser.add_argument("--vol2", action="store_true", help="Review vol2 chapters")
    parser.add_argument("--chapter", type=str, help="Review a single chapter")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    if not args.vol1 and not args.vol2:
        parser.error("Specify --vol1 and/or --vol2")

    model = args.model
    max_parallel = args.max_parallel

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = BOOK_DIR / f"_build/review_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    if args.vol1:
        chapters = [args.chapter] if args.chapter else VOL1_CHAPTERS
        tasks.extend([("vol1", ch) for ch in chapters])
    if args.vol2:
        chapters = [args.chapter] if args.chapter else VOL2_CHAPTERS
        tasks.extend([("vol2", ch) for ch in chapters])

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Gemini Review Pipeline — {model}")
    print(f"║  {len(tasks)} chapters, max {max_parallel} parallel")
    print(f"║  Output: {output_dir}")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for i, (vol, chapter) in enumerate(tasks):
            if i > 0 and i % max_parallel == 0:
                time.sleep(STAGGER_SECONDS)
            future = executor.submit(review_chapter, vol, chapter, output_dir, model)
            futures[future] = (vol, chapter)

        for future in as_completed(futures):
            vol, chapter = futures[future]
            result = future.result()
            results.append(result)
            status = result["status"]
            icon = {"OK": "✓", "SKIP": "⏭", "ERROR": "✗", "TIMEOUT": "⏰"}[status]
            print(f"  {icon} {vol}/{chapter}: {status}")

    # Write summary
    summary_path = output_dir / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(f"# Gemini Review Summary — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL}\n\n")

        for r in sorted(results, key=lambda x: (x["vol"], x["chapter"])):
            if r["status"] == "OK":
                report = (output_dir / f"{r['vol']}_{r['chapter']}.md").read_text()
                if "NO ISSUES FOUND" in report:
                    f.write(f"- ✓ **{r['vol']}/{r['chapter']}**: clean\n")
                else:
                    issue_count = report.count("**[")
                    f.write(f"- ⚠ **{r['vol']}/{r['chapter']}**: {issue_count} issues\n")
            else:
                f.write(f"- ✗ **{r['vol']}/{r['chapter']}**: {r['status']}\n")

    print(f"\n{'═' * 50}")
    print(f"Summary: {summary_path}")
    print(f"Reports: {output_dir}/")

    # Print summary
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
