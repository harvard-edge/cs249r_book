#!/usr/bin/env python3
"""Build a per-chapter audit/improve context package for a sub-agent.

For each chapter we want to drive to A-grade, this script produces a
single self-contained Markdown document containing:

- Chapter identity (vol, name, position in reading order)
- The list of prior chapters already read (so the agent knows what
  vocabulary it may assume)
- Prior-vocabulary terms (JSON)
- A per-section bundle: the section's prose, the current quiz questions
  for that section, and any audit issues gpt-5.4 flagged against them

The output is written to
``book/tools/scripts/genai/quiz_refresh/_audit/contexts/{vol}_{chapter}.md``
and is the single input a sub-agent needs to do an audit + improve pass
targeting A-grade output per §16 of the quiz-generation spec.

Usage
-----
    # One chapter:
    python3 build_audit_context.py vol1 training

    # All 33 chapters:
    python3 build_audit_context.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_prior_vocab import READING_ORDER, build as build_prior_vocab_for  # noqa: E402

REPO_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=HERE
    ).decode().strip()
)
CONTENTS_DIR = REPO_ROOT / "book" / "quarto" / "contents"
OUTPUT_DIR = HERE / "_audit" / "contexts"


def qmd_path_for(vol: str, chapter: str) -> Path:
    """Return the chapter's main ``.qmd``. Matches the same logic used
    in ``generate_quizzes.py`` for handling outliers like
    ``optimizations/model_compression.qmd``."""
    chapter_dir = CONTENTS_DIR / vol / chapter
    direct = chapter_dir / f"{chapter}.qmd"
    if direct.is_file():
        return direct
    candidates = sorted(
        p for p in chapter_dir.glob("*.qmd") if not p.name.startswith("_")
    )
    if not candidates:
        raise FileNotFoundError(f"no .qmd found in {chapter_dir}")
    return candidates[0]


def quiz_json_path_for(vol: str, chapter: str) -> Path:
    qmd = qmd_path_for(vol, chapter)
    stem = qmd.stem
    return CONTENTS_DIR / vol / chapter / f"{stem}_quizzes.json"


# The existing gpt-5.4 audit files live under two possible filenames
# because of the pre-fix naming collision. Try vol-prefixed first, then
# bare chapter name.
def audit_json_path_for(vol: str, chapter: str) -> Path | None:
    audit_dir = HERE / "_audit"
    for name in (f"{vol}_{chapter}_audit.json", f"{chapter}_audit.json"):
        p = audit_dir / name
        if p.is_file():
            # If the bare-name file is from a different volume, skip it.
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                declared = data.get("metadata", {}).get("chapter", "")
                if declared and declared != f"{vol}/{chapter}":
                    continue
            except json.JSONDecodeError:
                continue
            return p
    return None


# Regex for ``## Section Title {#sec-anchor-id}`` lines (exactly two
# hashes — not ### or deeper). We split the QMD into sections keyed by
# section_id.
_H2_LINE = re.compile(r"^##\s+(?P<title>.+?)\s*\{#(?P<id>sec-[^\s}]+)\}\s*$")
# Any heading line (## through ######). We use this to know when a
# section's text has ended — the next heading at ANY level below H1 is
# still a boundary for our purposes since quizzes are only ever at H2.
# Wait — a section CAN contain H3/H4 subheadings which are part of its
# text. We only end on the next H2. So match only `^## ` as the
# boundary.
_H2_BOUNDARY = re.compile(r"^##\s+(?!#)")


def extract_sections(qmd_text: str) -> list[dict]:
    """Return a list of ``{"id", "title", "text"}`` dicts, one per ``##``
    section that carries a ``{#sec-...}`` anchor. Sections without an
    anchor are skipped (they cannot carry a quiz anyway).

    The text of a section begins at its ``##`` heading line and ends at
    the next ``##`` heading (not at deeper headings inside it).
    """
    lines = qmd_text.splitlines()
    # First pass: locate every H2 start line and its anchor metadata.
    # We need both anchored and unanchored H2s so we know where a
    # section ends.
    h2_indices: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"^##\s+(?!#)", line):
            h2_indices.append(i)
    h2_indices.append(len(lines))  # sentinel

    sections: list[dict] = []
    for j in range(len(h2_indices) - 1):
        start = h2_indices[j]
        end = h2_indices[j + 1]
        head = lines[start]
        m = _H2_LINE.match(head)
        if not m:
            continue  # H2 without explicit anchor — not quizzed
        sec_id = "#" + m.group("id")
        title = m.group("title").strip()
        text = "\n".join(lines[start:end]).rstrip()
        sections.append({"id": sec_id, "title": title, "text": text})
    return sections


def reading_position(vol: str, chapter: str) -> int:
    for i, (v, c) in enumerate(READING_ORDER):
        if v == vol and c == chapter:
            return i + 1
    raise ValueError(f"{vol}/{chapter} not in READING_ORDER")


def prior_chapter_list(position: int) -> list[str]:
    """List of ``vol/chapter`` strings for chapters 1..position-1."""
    return [f"{v}/{c}" for v, c in READING_ORDER[: position - 1]]


def build_context(vol: str, chapter: str) -> str:
    position = reading_position(vol, chapter)
    total = len(READING_ORDER)
    qmd = qmd_path_for(vol, chapter)
    quiz_json = quiz_json_path_for(vol, chapter)
    audit_json = audit_json_path_for(vol, chapter)

    qmd_text = qmd.read_text(encoding="utf-8")
    quiz_data = json.loads(quiz_json.read_text(encoding="utf-8"))
    audit_data: dict | None = None
    if audit_json is not None:
        try:
            audit_data = json.loads(audit_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit_data = None

    sections = extract_sections(qmd_text)
    section_text_by_id = {s["id"]: s for s in sections}

    # Prior vocabulary
    if position > 1:
        vocab = build_prior_vocab_for(vol, chapter)
        prior_terms = [
            {"term": t["term"], "first_seen": t.get("first_seen", "")}
            for t in vocab.get("terms", [])
        ]
    else:
        prior_terms = []

    # Prior chapters list
    prior = prior_chapter_list(position)

    # Index audit issues by section_id
    audit_issues_by_section: dict[str, list[dict]] = {}
    audit_overall = None
    audit_buildup = None
    audit_distribution = None
    if audit_data is not None:
        audit_overall = audit_data.get("overall")
        audit_buildup = audit_data.get("build_up")
        audit_distribution = audit_data.get("distribution")
        for issue in audit_data.get("per_question_issues", []) or []:
            sid = issue.get("section_id")
            if not sid:
                continue
            audit_issues_by_section.setdefault(sid, []).append(issue)

    # Build the Markdown document
    out: list[str] = []
    out.append(f"# Audit/Improve context — `{vol}/{chapter}`")
    out.append("")
    out.append(f"**Position in reading order**: chapter {position} of "
               f"{total} ({position / total * 100:.0f}% through the book)")
    out.append("")

    # --- Prior chapters -------------------------------------------------
    out.append("## Prior chapters (already read by the reader)")
    out.append("")
    if prior:
        for i, p in enumerate(prior, start=1):
            out.append(f"{i}. `{p}`")
    else:
        out.append("_None — this is chapter 1._")
    out.append("")

    # --- Prior vocabulary ----------------------------------------------
    out.append(f"## Prior vocabulary ({len(prior_terms)} terms)")
    out.append("")
    out.append("The reader has already encountered these terms in earlier "
               "chapters and does **not** need them redefined. Questions "
               "whose entire point is defining one of these are "
               "`build_up_violation` — rewrite them to test application "
               "instead.")
    out.append("")
    out.append("```json")
    out.append(json.dumps(prior_terms, indent=2))
    out.append("```")
    out.append("")

    # --- Overall audit signal ------------------------------------------
    out.append("## Chapter-level audit signal (from prior gpt-5.4 audit)")
    out.append("")
    if audit_data is None:
        out.append("_No prior audit available for this chapter._ "
                   "Assess from scratch against §16.")
    else:
        if audit_overall:
            out.append(f"- **Overall grade (gpt-5.4)**: "
                       f"{audit_overall.get('quality_grade', '?')}")
            out.append(f"- **Summary**: {audit_overall.get('summary', '').strip()}")
        if audit_buildup:
            out.append("- **Build-up assessment**: "
                       + audit_buildup.get("chapter_level_assessment", "").strip())
        if audit_distribution:
            mix = audit_distribution.get("type_mix") or {}
            out.append(f"- **Type mix**: {mix}")
            out.append(f"- **Distribution note**: "
                       + audit_distribution.get("type_mix_assessment", "").strip())
    out.append("")

    # --- Per-section bundles -------------------------------------------
    out.append("## Per-section bundles")
    out.append("")
    out.append("Each section block below contains: (a) the section's "
               "prose from the chapter QMD, (b) the current quiz "
               "questions for that section, and (c) any audit issues "
               "gpt-5.4 flagged against those questions. Your task is "
               "to rewrite each question to A-grade per §16 of the "
               "canonical spec.")
    out.append("")

    quiz_sections = quiz_data.get("sections", []) or []
    for qs in quiz_sections:
        sid = qs.get("section_id")
        if not sid:
            continue
        sec_title = qs.get("section_title", "(untitled)")
        out.append("---")
        out.append("")
        out.append(f"### Section `{sid}` — {sec_title}")
        out.append("")

        # Section prose
        text_entry = section_text_by_id.get(sid)
        if text_entry is None:
            out.append("_Section text not found in QMD (anchor may not "
                       "match an H2 heading). Skipping prose embed._")
            out.append("")
        else:
            out.append("**Section prose** (the text the quizzes must test):")
            out.append("")
            out.append("```qmd")
            out.append(text_entry["text"])
            out.append("```")
            out.append("")

        # Current questions
        quiz_info = qs.get("quiz_data", {})
        out.append("**Current quiz (generated by gpt-5.4)**:")
        out.append("")
        out.append("```json")
        out.append(json.dumps(quiz_info, indent=2))
        out.append("```")
        out.append("")

        # Audit issues for this section
        issues = audit_issues_by_section.get(sid, [])
        out.append(f"**Audit issues flagged by gpt-5.4** ({len(issues)}):")
        out.append("")
        if issues:
            for iss in issues:
                out.append(
                    f"- q{iss.get('question_index', '?')} — "
                    f"**{iss.get('issue_type', 'other')}** "
                    f"({iss.get('severity', '?')}): "
                    f"{iss.get('description', '').strip()}"
                )
                fix = iss.get("suggested_fix", "").strip()
                if fix:
                    out.append(f"  - Suggested fix: {fix}")
        else:
            out.append("_No per-question issues flagged for this section._")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vol", nargs="?", help='"vol1" or "vol2"')
    ap.add_argument("chapter", nargs="?", help="chapter directory name")
    ap.add_argument("--all", action="store_true",
                    help="build context for every chapter in READING_ORDER")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, str]]
    if args.all:
        targets = list(READING_ORDER)
    else:
        if not args.vol or not args.chapter:
            ap.error("provide vol and chapter, or --all")
        targets = [(args.vol, args.chapter)]

    errors: list[str] = []
    for vol, chap in targets:
        try:
            ctx = build_context(vol, chap)
        except FileNotFoundError as e:
            errors.append(f"{vol}/{chap}: {e}")
            continue
        out_path = OUTPUT_DIR / f"{vol}_{chap}.md"
        out_path.write_text(ctx, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")

    if errors:
        print("\nerrors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
