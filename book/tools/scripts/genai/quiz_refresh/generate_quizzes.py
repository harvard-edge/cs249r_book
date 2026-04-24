#!/usr/bin/env python3
"""Quiz regeneration runner for Vol1 and Vol2 of the ML Systems textbook.

Reads the canonical spec at ``.claude/rules/quiz-generation.md``, reads a
chapter's prose, and produces ``{chapter}_quizzes.json.new`` at the
canonical path. Every decision about taxonomy, format, quality bar, and
schema lives in the spec — this script is plumbing.

Usage
-----
    # Single chapter:
    python3 generate_quizzes.py --chapter vol1/training

    # All chapters in reading order, 6-way parallel:
    python3 generate_quizzes.py --all --workers 6

    # Dry-run (no API calls, prints what would happen):
    python3 generate_quizzes.py --all --dry-run

Environment
-----------
    ANTHROPIC_API_KEY   required unless --dry-run

Outputs
-------
- ``book/quarto/contents/{vol}/{chapter}/{chapter}_quizzes.json.new`` — the
  regenerated file. Rename to drop ``.new`` after human review.
- ``book/tools/scripts/genai/quiz_refresh/_reviews/{chapter}_memo.md`` — a
  short summary of coverage, type mix, and flags for spot-check.

Each chapter's output is validated with ``validate_quiz_json.py``. A
non-zero validator exit is surfaced as a hard failure; warnings are
reported but do not block.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

# Local scaffolding
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_anchors import extract as extract_anchors_for  # noqa: E402
from build_prior_vocab import build as build_prior_vocab_for, READING_ORDER  # noqa: E402

# Repo layout (derived from git, so the script works from any worktree)
REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE)
    .decode()
    .strip()
)
SPEC_PATH = REPO_ROOT / ".claude" / "rules" / "quiz-generation.md"
CONTENTS_DIR = REPO_ROOT / "book" / "quarto" / "contents"
REVIEWS_DIR = HERE / "_reviews"
VALIDATOR = HERE / "validate_quiz_json.py"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def qmd_path_for(vol: str, chapter: str) -> Path:
    """Return the chapter's main ``.qmd``. Handles the ``optimizations``
    outlier (``model_compression.qmd``) via glob fallback."""
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


def canonical_json_new_path(vol: str, chapter: str) -> Path:
    return CONTENTS_DIR / vol / chapter / f"{chapter}_quizzes.json.new"


def memo_path(chapter: str) -> Path:
    return REVIEWS_DIR / f"{chapter}_memo.md"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def load_spec() -> str:
    if not SPEC_PATH.is_file():
        sys.exit(
            f"error: canonical spec not found at {SPEC_PATH}\n"
            f"       sync it from AIConfigs/projects/MLSysBook/.claude/rules/"
        )
    return SPEC_PATH.read_text(encoding="utf-8")


def build_user_prompt(vol: str, chapter: str, qmd_path: Path) -> str:
    """Assemble the per-chapter user prompt.

    Structure: position -> anchor map -> prior vocabulary -> full chapter
    text. All four inputs the spec mentions in §13. Everything fits in
    Sonnet's 200K context for every chapter we have.
    """
    position = next(
        (i + 1 for i, (v, c) in enumerate(READING_ORDER) if v == vol and c == chapter),
        None,
    )
    total_chapters = len(READING_ORDER)
    anchor_data = extract_anchors_for(qmd_path)
    prior_vocab_data = build_prior_vocab_for(vol, chapter)
    chapter_text = qmd_path.read_text(encoding="utf-8")

    sections = [a for a in anchor_data["anchors"] if a["level"] == "section"]
    subsections = [a for a in anchor_data["anchors"] if a["level"] == "subsection"]

    prior_terms_brief = (
        f"{prior_vocab_data['prior_term_count']} terms assumed as known "
        f"(from chapters 1..{position - 1 if position else 0})"
        if position and position > 1
        else "none (this is the first chapter)"
    )

    return f"""You are generating quizzes for chapter **{vol}/{chapter}** \
(position {position} of {total_chapters} in the reading order).

## Anchor map — the authoritative list of ``section_id`` / ``parent_section_id`` values

You MUST use these exact identifiers. {len(sections)} section-level (``##``) \
and {len(subsections)} subsection-level (``###``) anchors are available.

```json
{json.dumps(anchor_data["anchors"], indent=2)}
```

## Prior vocabulary — {prior_terms_brief}

Terms below were first defined in earlier chapters. Per the spec §8, you \
MAY use these freely in questions and answers without re-defining them. \
You MUST NOT write questions that test the *definition* of any of these \
terms; those definition-tests exist in the earlier chapters already. You \
MAY still test the *application* of a prior-chapter term in this \
chapter's context.

```json
{json.dumps([
    {"term": t["term"], "first_seen": t["first_seen"]}
    for t in prior_vocab_data["terms"]
], indent=2)}
```

## Chapter source — `{qmd_path.relative_to(REPO_ROOT)}`

Read the full text below before drafting questions. Find the Learning \
Objectives callout and use it to constrain what your questions should \
assess. Note which `##` sections and `###` subsections are substantive \
enough to warrant a quiz vs. administrative/recap (per spec §3).

```qmd
{chapter_text}
```

## Your task

For every `##` and `###` anchor listed above, decide per the spec §3 \
whether a quiz is needed. For quizzed entries, write 4–6 questions \
(section) or 2–3 questions (subsection) per the spec §1, following the \
five-type taxonomy (§4), per-type answer format (§5), quality bar (§6), \
difficulty progression (§7), knowledge-boundary rules (§8), and \
anti-patterns (§9) — including the anti-shuffle-bug rules (§10).

Distribute MCQ correct answers evenly across A, B, C, D as you \
construct each question. Explain MCQ distractors by their CONTENT, \
NEVER by their letter.

Return the output as a single JSON object matching the schema in §11. \
The JSON object must be the entire response; no prose before or after. \
Set `metadata.total_sections`, `metadata.total_subsections`, and \
`metadata.total_quizzes` to EXACTLY match your actual entry counts — \
the validator cross-checks these.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_claude(
    system_prompt: str, user_prompt: str, model: str, api_key: str
) -> dict[str, Any]:
    """Single API call. Extracts a top-level JSON object from the response.

    Raises on non-JSON output. The caller is responsible for retrying."""
    from anthropic import Anthropic  # type: ignore

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    ).strip()
    # Strip markdown fences if present — Claude sometimes wraps JSON in ```json ... ```
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:400]
        raise RuntimeError(
            f"model returned non-JSON output: {e}\n---\n{snippet}\n---"
        ) from e


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def finalize_metadata(data: dict[str, Any], vol: str, chapter: str, model: str) -> None:
    """Fix metadata counts to match actual entries (defense against the
    off-by-one mistake the Wave 1 pilot made)."""
    sections = data.get("sections", []) or []
    n_sec = sum(1 for s in sections if s.get("level") == "section")
    n_sub = sum(1 for s in sections if s.get("level") == "subsection")
    meta = data.setdefault("metadata", {})
    meta["source_file"] = str(
        (CONTENTS_DIR / vol / chapter / f"{chapter}.qmd").relative_to(REPO_ROOT)
    )
    meta["schema_version"] = 2
    meta["generated_by"] = "quiz-refresh/generate_quizzes.py"
    meta["generated_on"] = date.today().isoformat()
    meta["model"] = model
    meta["total_sections"] = n_sec
    meta["total_subsections"] = n_sub
    meta["total_quizzes"] = n_sec + n_sub


def run_validator(json_path: Path, qmd_path: Path) -> tuple[int, str]:
    """Run the schema+anchor validator. Returns (exit_code, combined_output)."""
    proc = subprocess.run(
        ["python3", str(VALIDATOR), str(json_path), str(qmd_path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def write_memo(
    chapter: str, vol: str, data: dict[str, Any], validator_out: str
) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sections = data.get("sections", []) or []
    secs = [s for s in sections if s.get("level") == "section"]
    subs = [s for s in sections if s.get("level") == "subsection"]
    quizzed = [s for s in sections if s.get("quiz_data", {}).get("quiz_needed")]
    skipped = [s for s in sections if not s.get("quiz_data", {}).get("quiz_needed")]
    all_qs = [
        q
        for s in sections
        for q in s.get("quiz_data", {}).get("questions") or []
    ]
    from collections import Counter

    type_mix = Counter(q.get("question_type", "?") for q in all_qs)

    lines = [
        f"# {chapter} quiz-refresh memo",
        "",
        f"- **Chapter**: {vol}/{chapter}",
        f"- **Generated on**: {data.get('metadata', {}).get('generated_on')}",
        f"- **Model**: {data.get('metadata', {}).get('model')}",
        f"- **Anchors processed**: {len(secs)} sections + {len(subs)} subsections = {len(sections)} entries",
        f"- **Quizzes written**: {len(quizzed)} active, {len(skipped)} skipped (quiz_needed=false)",
        f"- **Questions written**: {len(all_qs)} total ({', '.join(f'{n} {t}' for t, n in type_mix.most_common())})",
        "",
        "## Skipped entries (`quiz_needed: false`)",
        "",
    ]
    if skipped:
        for s in skipped:
            reason = (
                s.get("quiz_data", {})
                .get("rationale", {})
                .get("ranking_explanation", "(no reason given)")
            )
            lines.append(f"- `{s['section_id']}` — {reason}")
    else:
        lines.append("_None._")
    lines += [
        "",
        "## Validator output",
        "",
        "```",
        validator_out or "(no output)",
        "```",
        "",
    ]
    mp = memo_path(chapter)
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate_for_chapter(
    vol: str,
    chapter: str,
    system_prompt: str,
    model: str,
    api_key: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Run the full pipeline for one chapter. Returns a summary dict."""
    t0 = time.time()
    result: dict[str, Any] = {"vol": vol, "chapter": chapter}
    try:
        qmd = qmd_path_for(vol, chapter)
        user_prompt = build_user_prompt(vol, chapter, qmd)
        result["prompt_chars"] = len(user_prompt)
        if dry_run:
            result["status"] = "dry-run"
            result["elapsed_s"] = round(time.time() - t0, 1)
            return result
        assert api_key is not None
        data = call_claude(system_prompt, user_prompt, model, api_key)
        finalize_metadata(data, vol, chapter, model)
        out = canonical_json_new_path(vol, chapter)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        rc, vout = run_validator(out, qmd)
        memo = write_memo(chapter, vol, data, vout)
        result.update(
            {
                "status": "ok" if rc == 0 else "validator-failed",
                "validator_rc": rc,
                "output": str(out.relative_to(REPO_ROOT)),
                "memo": str(memo.relative_to(REPO_ROOT)),
                "sections": data.get("metadata", {}).get("total_sections"),
                "subsections": data.get("metadata", {}).get("total_subsections"),
                "questions": sum(
                    len(s.get("quiz_data", {}).get("questions") or [])
                    for s in data.get("sections", []) or []
                ),
            }
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", help="single chapter in 'vol1/training' form")
    g.add_argument(
        "--all",
        action="store_true",
        help="process every chapter in READING_ORDER",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model name (default {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallelism for --all (default 4)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build prompts but do not call the API",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.dry_run and not api_key:
        sys.exit("error: ANTHROPIC_API_KEY not set (or pass --dry-run)")

    system_prompt = load_spec()
    if args.chapter:
        parts = args.chapter.split("/", 1)
        if len(parts) != 2:
            sys.exit(f"error: --chapter must be 'vol/name', got {args.chapter!r}")
        chapters: list[tuple[str, str]] = [(parts[0], parts[1])]
    else:
        chapters = list(READING_ORDER)

    print(f"spec: {SPEC_PATH.relative_to(REPO_ROOT)} ({len(system_prompt)} chars)")
    print(f"chapters: {len(chapters)}")
    print(f"model: {args.model}")
    print(f"workers: {args.workers}")
    print(f"dry-run: {args.dry_run}")
    print()

    results: list[dict[str, Any]] = []
    if args.workers <= 1 or len(chapters) == 1:
        for vol, chap in chapters:
            print(f"→ {vol}/{chap}", flush=True)
            r = generate_for_chapter(
                vol, chap, system_prompt, args.model, api_key, args.dry_run
            )
            results.append(r)
            print(f"   {r.get('status')}  {r.get('elapsed_s')}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    generate_for_chapter,
                    vol,
                    chap,
                    system_prompt,
                    args.model,
                    api_key,
                    args.dry_run,
                ): (vol, chap)
                for vol, chap in chapters
            }
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(
                    f"→ {r['vol']}/{r['chapter']}  {r.get('status'):<18} "
                    f"{r.get('elapsed_s')}s  "
                    f"Qs={r.get('questions', '-')}  "
                    f"err={r.get('error', '')}",
                    flush=True,
                )

    print("\n=== SUMMARY ===")
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(
        1 for r in results if r.get("status") not in ("ok", "dry-run")
    )
    dry = sum(1 for r in results if r.get("status") == "dry-run")
    total_qs = sum(r.get("questions", 0) or 0 for r in results)
    print(f"  ok: {ok}")
    print(f"  failed: {failed}")
    print(f"  dry-run: {dry}")
    print(f"  total questions: {total_qs}")
    if failed:
        print("\nfailed chapters:")
        for r in results:
            if r.get("status") not in ("ok", "dry-run"):
                print(f"  {r['vol']}/{r['chapter']}: {r.get('error', r.get('status'))}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
