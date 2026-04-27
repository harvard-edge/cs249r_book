#!/usr/bin/env python3
"""Quiz regeneration runner for Vol1 and Vol2 of the ML Systems textbook.

Reads the canonical spec at ``.claude/docs/shared/quiz-generation.md``, reads a
chapter's prose, and produces ``{chapter}_quizzes.json.new`` at the
canonical path. Every decision about taxonomy, format, quality bar, and
schema lives in the spec — this script is plumbing.

Usage
-----
    # Single chapter (default provider is OpenAI with gpt-4o):
    python3 generate_quizzes.py --chapter vol1/training

    # Anthropic (Opus) instead:
    python3 generate_quizzes.py --chapter vol1/training --provider anthropic

    # All chapters in reading order, 4-way parallel:
    python3 generate_quizzes.py --all --workers 4

    # Override model explicitly:
    python3 generate_quizzes.py --all --provider openai --model gpt-4.1

    # Dry-run (no API calls, prints what would happen):
    python3 generate_quizzes.py --all --dry-run

Environment
-----------
    OPENAI_API_KEY      required when --provider openai (the default)
    ANTHROPIC_API_KEY   required when --provider anthropic

Outputs
-------
- ``book/quarto/contents/{vol}/{chapter}/{chapter}_quizzes.json.new`` — the
  regenerated file. Rename to drop ``.new`` after human review.
- ``book/tools/scripts/quizzes/_reviews/{chapter}_memo.md`` — a
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

# Per-provider defaults. --provider chooses which path; --model overrides
# the default for the chosen provider.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        # gpt-5.4 (March 2026) — current top-line GPT for chat workloads.
        # Use --model gpt-5.4-pro for the higher-tier variant; --model
        # gpt-5.4-mini for a cheaper pass. Quiz regeneration is an
        # infrequent batch job so we default to the best mainstream
        # model rather than the cheapest.
        "model": "gpt-5.4",
        "env": "OPENAI_API_KEY",
    },
    "anthropic": {
        # Opus 4.7 — the most capable Anthropic model for this task.
        "model": "claude-opus-4-7",
        "env": "ANTHROPIC_API_KEY",
    },
}
DEFAULT_PROVIDER = "openai"
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
    """Return the canonical ``.new`` staging path. The stem matches the
    chapter's ``.qmd`` filename (not the directory name), so outlier
    chapters like ``vol1/optimizations/`` (whose ``.qmd`` is
    ``model_compression.qmd``) get the correct ``model_compression_quizzes.json``
    output that the Lua filter expects per the chapter's ``quiz:`` YAML key."""
    qmd = qmd_path_for(vol, chapter)
    stem = qmd.stem  # e.g. "model_compression" for vol1/optimizations
    return CONTENTS_DIR / vol / chapter / f"{stem}_quizzes.json.new"


def memo_path(vol: str, chapter: str) -> Path:
    # Prefix with ``vol`` so that ``vol1/conclusion`` and ``vol2/conclusion``
    # (and same-named ``introduction`` chapters) do not overwrite each
    # other's memo on disk. Pre-fix filename was ``{chapter}_memo.md``.
    return REVIEWS_DIR / f"{vol}_{chapter}_memo.md"


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


def build_audit_prompt(vol: str, chapter: str, qmd_path: Path) -> str:
    """Assemble the per-chapter prompt for AUDIT mode.

    Audit reads the existing canonical quiz JSON and produces a
    structured assessment: per-question issues, build-up compliance
    (does it use prior vocab vs. redefine it), distribution quality,
    and an overall grade. Used to identify what still needs improvement
    after generate + improve passes.
    """
    position = next(
        (i + 1 for i, (v, c) in enumerate(READING_ORDER) if v == vol and c == chapter),
        None,
    )
    total_chapters = len(READING_ORDER)
    existing_json_path = canonical_json_new_path(vol, chapter).with_suffix("")
    if not existing_json_path.is_file():
        raise FileNotFoundError(
            f"audit mode requires an existing canonical quiz JSON at {existing_json_path}"
        )
    existing_data = json.loads(existing_json_path.read_text(encoding="utf-8"))
    chapter_text = qmd_path.read_text(encoding="utf-8")
    prior_vocab_data = build_prior_vocab_for(vol, chapter)
    prior_terms = [
        {"term": t["term"], "first_seen": t["first_seen"]}
        for t in prior_vocab_data["terms"]
    ]
    prior_terms_brief = (
        f"{len(prior_terms)} terms from chapters 1..{position - 1}"
        if position and position > 1
        else "(this is chapter 1 — no prior chapters)"
    )

    return f"""You are auditing the quizzes for chapter **{vol}/{chapter}** \
(position {position} of {total_chapters} in the reading order).

This is an **audit pass** — you do NOT change the quiz file. You PRODUCE \
a structured assessment that surfaces questions still needing \
improvement after a generate + improve cycle. The auditor catches what \
the generator and improver missed.

## Audit dimensions (assess each)

### 1. Per-question quality

For each question, check it against the spec's six-point quality bar \
(grounded, reasoning over recall, concrete numbers, explanatory \
answer, Bloom's-verb LO, plausible distractors). Specifically hunt for:

- MCQ distractors that are obviously wrong (no informed reader would \
  pick) — they reduce a 4-option MCQ to a 3-option test.
- FILL questions that test pure term recall instead of conceptual \
  inference.
- TF questions whose answer is obvious from grammar alone.
- Templated/formulaic stems ("Which best captures…", "What is the \
  best interpretation…") used too often.
- LOs that tautologically restate the question stem.
- Questions that test memorization of a date/name/quote rather than \
  systems reasoning.
- Answers that miss explaining WHY a plausible distractor is wrong.

### 2. Build-up compliance (the critical audit)

For each question, check the prior-chapter vocabulary list. Flag:

- **Re-definition violation (HIGH)**: question whose ENTIRE point is \
  defining or testing the definition of a term first established in a \
  previous chapter. (Application of a prior-chapter term in this \
  chapter's context is fine; testing the definition itself is not.)
- **Missed-build-up opportunity (MEDIUM)**: question that re-explains \
  a prior-chapter concept in the question text or answer when it \
  should have assumed reader familiarity, padding the question \
  unnecessarily.
- **Forward-reference (HIGH)**: question that requires knowledge from \
  a LATER chapter (you may need to flag based on whether a term \
  appears in the chapter text or in prior_vocab — if neither, it's \
  forward-referenced).

Also assess at the chapter level:
- Does the chapter appropriately *leverage* prior-chapter vocabulary, \
  or does it feel like it was written in isolation?
- Is the difficulty appropriate for chapter position {position}/33? \
  (Chapters 1–5 foundational; 6–10 intermediate; 11–15 advanced; \
  16+ specialized.)

### 3. Distribution

- Type-mix balance per chapter against the spec target \
  (MCQ 40% / SHORT 30% / TF 13% / FILL 8% / ORDER 9%).
- Per-section question count distribution (should be 4–6, with 2–3 \
  for tier-2 minimal quizzes).

## Prior vocabulary — {prior_terms_brief}

```json
{json.dumps(prior_terms, indent=2)}
```

## Chapter source — `{qmd_path.relative_to(REPO_ROOT)}`

```qmd
{chapter_text}
```

## Quiz JSON to audit

```json
{json.dumps(existing_data, indent=2)}
```

## Required output format

Return a single JSON object with this exact structure (no prose \
before or after):

```jsonc
{{
  "metadata": {{
    "chapter": "{vol}/{chapter}",
    "position": {position},
    "total_chapters": {total_chapters},
    "questions_audited": <int>,
    "audit_date": "YYYY-MM-DD"
  }},
  "overall": {{
    "quality_grade": "A|B|C|D|F",  // A=excellent, B=good with minor issues, C=acceptable but multiple issues, D=substantial issues, F=needs regeneration
    "summary": "1–3 sentence overall assessment"
  }},
  "build_up": {{
    "prior_terms_appropriately_used": [
      // list of prior-chapter terms that THIS chapter's questions
      // correctly leverage (use without redefining)
    ],
    "redefinition_violations": [
      // questions that wrongly redefine prior-chapter terms
      {{"section_id": "...", "question_index": <int>, "term": "...",
        "explanation": "..."}}
    ],
    "missed_buildup_opportunities": [
      // questions that re-explain a prior-chapter concept unnecessarily
      {{"section_id": "...", "question_index": <int>, "concept": "...",
        "explanation": "..."}}
    ],
    "forward_reference_violations": [
      // questions requiring later-chapter knowledge
      {{"section_id": "...", "question_index": <int>, "concept": "...",
        "explanation": "..."}}
    ],
    "chapter_level_assessment": "1–2 sentences on whether the chapter feels appropriately built on prior context"
  }},
  "distribution": {{
    "type_mix": {{"MCQ": <count>, "SHORT": <count>, "TF": <count>, "FILL": <count>, "ORDER": <count>}},
    "type_mix_assessment": "matches target | over-MCQ | under-FILL | etc.",
    "section_count_outliers": [
      // sections with question counts outside the 4–6 / 2–3 windows
      {{"section_id": "...", "count": <int>, "expected": "4–6"}}
    ]
  }},
  "per_question_issues": [
    // List ONLY genuinely problematic questions. Do not list strong
    // questions. Aim for the ones that would benefit from another
    // pass. Empty list is fine if the chapter is clean.
    {{
      "section_id": "...",
      "question_index": <int>,
      "issue_type": "throwaway_distractor|trivia_fill|easy_tf|vague_lo|tautological_lo|missing_explanation|build_up_violation|forward_reference|recall_only|other",
      "severity": "high|medium|low",
      "description": "1–2 sentences explaining the issue",
      "suggested_fix": "1–2 sentences proposing a specific edit"
    }}
  ],
  "recommended_action": "merge_as_is | minor_fixes_recommended | targeted_improvements_needed | requires_regeneration"
}}
```

Be honest in your assessment. The point is to surface real issues, \
not to hand out gold stars. A chapter with grade B and 5 minor \
issues is more useful than a chapter with grade A and 0 issues if \
the issues are real.
"""


def build_improve_prompt(vol: str, chapter: str, qmd_path: Path) -> str:
    """Assemble the per-chapter prompt for the IMPROVE mode.

    The model receives: the chapter text (for ground truth), the prior
    vocabulary (for building-up context — terms from chapters 1..N-1 may
    be assumed without re-defining), and the **existing** quiz JSON. It
    returns a surgically-improved JSON preserving section_ids, question
    counts per section, and question types — only the content of
    individual questions/choices/answers/learning-objectives may change.
    """
    position = next(
        (i + 1 for i, (v, c) in enumerate(READING_ORDER) if v == vol and c == chapter),
        None,
    )
    total_chapters = len(READING_ORDER)
    existing_json_path = canonical_json_new_path(vol, chapter).with_suffix("")
    if not existing_json_path.is_file():
        raise FileNotFoundError(
            f"improve mode requires an existing canonical quiz JSON at {existing_json_path}"
        )
    existing_data = json.loads(existing_json_path.read_text(encoding="utf-8"))
    chapter_text = qmd_path.read_text(encoding="utf-8")
    prior_vocab_data = build_prior_vocab_for(vol, chapter)

    # Compact prior vocab to term + source only (definitions bloat the prompt)
    prior_terms = [
        {"term": t["term"], "first_seen": t["first_seen"]}
        for t in prior_vocab_data["terms"]
    ]

    prior_terms_brief = (
        f"{len(prior_terms)} terms from chapters 1..{position - 1} "
        "are listed below — you MUST assume the reader knows them."
        if position and position > 1
        else "This is the first chapter — no prior vocabulary to build on."
    )

    return f"""You are improving the existing quizzes for chapter **{vol}/{chapter}** \
(position {position} of {total_chapters}).

This is an **improvement pass**, not regeneration. Most questions are \
already strong. Your job is targeted surgical fixes on the weakest \
questions based on a review rubric. Preserve structure: keep the same \
section_ids, keep the same number of questions per section, keep the \
same question_type on each question. Only the **content** of the \
question, choices, answer, and learning_objective may change.

## Review rubric — five weaknesses to hunt

Scan every question through these five lenses. Apply fixes where any \
apply. Leave untouched questions untouched; preserving good content \
is as important as fixing weak content.

1. **Throwaway MCQ distractors.** Any distractor that no informed \
   reader would consider — e.g., "CPUs stopped supporting \
   floating-point arithmetic", "checkpointing is impossible on TPUs", \
   "lower tokens per second means less information can leak". Replace \
   with a plausible misconception a CS/EE grad student might actually \
   hold (confusing memory bandwidth with compute throughput, \
   attributing a memory-bound symptom to a compute cause, assuming \
   linear scaling under Amdahl's law, etc.).

2. **FILL questions that are bare term recall.** "Detecting ___ \
   prevents silent propagation" → answer is "NaN" is a vocabulary \
   test. Either rewrite so the blank is a concept students must infer \
   from context (not just name) or keep it FILL but make the sentence \
   force real reasoning, not recognition. (FILL remains FILL — do not \
   change the type.)

3. **TF questions whose answer is grammatically obvious.** \
   "True or False: synthetic data can fully replace real data" — no \
   grad student would say True. Rewrite to address a misconception \
   that's actually plausible: something the student would likely get \
   wrong without having read the section.

4. **Templated stems.** If too many MCQs start "Which statement best \
   captures...", "What is the best interpretation...", "Which change \
   best describes..." — vary the stems where variation improves \
   clarity. Scenario-rooted stems ("A team profiles X and sees Y...") \
   are consistently stronger than abstract "which best" stems.

5. **Learning objectives that tautologically restate the question.** \
   A question "What is the definition of AI Engineering?" paired with \
   LO "Identify the definition of AI Engineering" is redundant. \
   Rewrite the LO as a concrete testable outcome using a Bloom's verb \
   (Apply, Calculate, Identify, Compare, Explain, Analyze, Evaluate, \
   Design, Classify, Justify, Distinguish, Interpret, Predict, \
   Sequence).

## Critical constraints

- **Prior-vocabulary contract** (this is the "building up" principle): \
  terms below were established in earlier chapters. You MAY use them \
  in questions and answers without re-defining them — the reader has \
  met them. You MUST NOT write questions that test definitions of \
  prior-chapter terms (those tests exist in earlier chapters). You MAY \
  test their *application* in this chapter's context.

- **No forward-reference:** do not introduce concepts from later \
  chapters. Only concepts in THIS chapter or in prior_vocab are valid.

- **Preserve section_id strings exactly** — the Lua filter matches by \
  id and a change would orphan the quiz.

- **Preserve question count per section** (if a section has 5 questions, \
  return 5 questions).

- **Preserve question_type** — do not change an MCQ to a SHORT, do not \
  change a FILL to an MCQ, etc.

- **Anti-shuffle-bug rules stay in force** (spec §5, §10): \
  distractor explanations refer to content, not letters.

## Prior vocabulary — {prior_terms_brief}

```json
{json.dumps(prior_terms, indent=2)}
```

## Chapter source — `{qmd_path.relative_to(REPO_ROOT)}`

(Use this as the ground truth for factual correctness; quiz content \
must be answerable from material already established up to and \
including this chapter.)

```qmd
{chapter_text}
```

## Existing quiz JSON — the file you are improving

```json
{json.dumps(existing_data, indent=2)}
```

## Your task

Return the **improved JSON** — the complete structure, identical \
shape to the input, with surgical content fixes applied per the \
rubric above. Your output MUST be a single JSON object, no prose \
before or after. `metadata.total_sections`, \
`metadata.sections_with_quizzes`, and \
`metadata.sections_without_quizzes` must match the actual entry \
counts. Every `section_id` must match the input exactly.

In addition, update `metadata` with one extra field: \
`improved_by: "quizzes/improve-mode"`, and bump the \
`generated_on` date to today.
"""


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

## Anchor map

You MUST use these exact identifiers. The chapter has \
{len(sections)} section-level (``##``) anchors and {len(subsections)} \
subsection-level (``###``) anchors.

**Important**: per the spec §1, ONLY ``##`` anchors are quiz candidates. \
The ``###`` anchors listed here are the **scope** the parent section's \
quiz must cover, not standalone quiz entries. Never emit a quiz entry \
whose ``section_id`` is a ``###`` anchor; the validator will reject it.

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
assess. Determine per spec §3 which ``##`` sections warrant a full quiz \
(Tier 1, 4–6 questions), a minimal quiz (Tier 2, 2–3 questions), or no \
quiz at all (Tier 3, `quiz_needed: false`).

```qmd
{chapter_text}
```

## Your task

For every ``##`` anchor listed above (and ONLY for ``##`` anchors), emit \
exactly one entry in the output's ``sections`` array. The entry's \
``section_id`` MUST be a ``##`` anchor; the entry's questions MUST draw \
on the entire ``##`` section including every ``###`` subsection nested \
under it (per spec §1 and §8). Sample questions across the full span of \
the section — not just its opening paragraph — so one quiz surveys the \
whole argumentative arc.

Follow the five-type taxonomy (§4), per-type answer format (§5), quality \
bar (§6), difficulty progression (§7), knowledge-boundary rules (§8), \
and anti-patterns (§9) — including the anti-shuffle-bug rules (§10).

Distribute MCQ correct answers evenly across A, B, C, D as you \
construct each question. Explain MCQ distractors by their CONTENT, \
NEVER by their letter.

Return the output as a single JSON object matching the schema in §11. \
The JSON object must be the entire response; no prose before or after. \
Set ``metadata.total_sections``, ``metadata.sections_with_quizzes``, and \
``metadata.sections_without_quizzes`` to EXACTLY match your actual \
entry counts — the validator cross-checks these.
"""


# ---------------------------------------------------------------------------
# LLM calls — one function per provider, dispatched by call_model()
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    """Remove a surrounding ``` or ```json fence if the model wrapped its
    JSON response in one. Some models ignore JSON-object format hints and
    still emit fenced markdown."""
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence + optional language tag on its own line
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


def call_anthropic(
    system_prompt: str, user_prompt: str, model: str, api_key: str
) -> dict[str, Any]:
    """Call the Anthropic Messages API. Returns the parsed JSON object."""
    from anthropic import Anthropic  # type: ignore

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = _strip_json_fence(
        "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:400]
        raise RuntimeError(
            f"anthropic returned non-JSON output: {e}\n---\n{snippet}\n---"
        ) from e


def call_openai(
    system_prompt: str, user_prompt: str, model: str, api_key: str
) -> dict[str, Any]:
    """Call the OpenAI Chat Completions API. Returns the parsed JSON object.

    Uses ``response_format={'type': 'json_object'}`` to coerce JSON output.
    The system prompt instructs the model to return JSON; this flag enforces
    it at the API level.
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    # GPT-5 family (and newer) deprecated ``max_tokens`` in favor of
    # ``max_completion_tokens``. Older gpt-4x models accept both. Use the
    # new name for forward compatibility.
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = _strip_json_fence(resp.choices[0].message.content or "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:400]
        raise RuntimeError(
            f"openai returned non-JSON output: {e}\n---\n{snippet}\n---"
        ) from e


def call_model(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """Provider-agnostic dispatch. Raises on unsupported provider."""
    if provider == "openai":
        return call_openai(system_prompt, user_prompt, model, api_key)
    if provider == "anthropic":
        return call_anthropic(system_prompt, user_prompt, model, api_key)
    raise ValueError(f"unsupported provider: {provider!r}")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def finalize_metadata(data: dict[str, Any], vol: str, chapter: str, model: str) -> None:
    """Fix metadata counts to match actual entries (defense against model
    off-by-one reporting). ``total_sections`` = total ``##`` entries;
    ``sections_with_quizzes`` = entries with ``quiz_needed: true``."""
    sections = data.get("sections", []) or []
    total = len(sections)
    with_quiz = sum(1 for s in sections if (s.get("quiz_data") or {}).get("quiz_needed"))
    meta = data.setdefault("metadata", {})
    meta["source_file"] = str(
        (CONTENTS_DIR / vol / chapter / f"{chapter}.qmd").relative_to(REPO_ROOT)
    )
    meta["schema_version"] = 2
    meta["generated_by"] = "quizzes/generate_quizzes.py"
    meta["generated_on"] = date.today().isoformat()
    meta["model"] = model
    meta["total_sections"] = total
    meta["sections_with_quizzes"] = with_quiz
    meta["sections_without_quizzes"] = total - with_quiz
    # Remove any stale v2-draft fields that no longer belong
    for stale in ("total_subsections", "total_quizzes"):
        meta.pop(stale, None)


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
        f"# {chapter} quizzes memo",
        "",
        f"- **Chapter**: {vol}/{chapter}",
        f"- **Generated on**: {data.get('metadata', {}).get('generated_on')}",
        f"- **Model**: {data.get('metadata', {}).get('model')}",
        f"- **`##` sections processed**: {len(sections)} entries",
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
    mp = memo_path(vol, chapter)
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate_for_chapter(
    vol: str,
    chapter: str,
    system_prompt: str,
    provider: str,
    model: str,
    api_key: str | None,
    dry_run: bool,
    mode: str = "generate",
) -> dict[str, Any]:
    """Run the full pipeline for one chapter. Returns a summary dict.

    ``mode="generate"`` does fresh generation from scratch.
    ``mode="improve"`` reads the existing canonical JSON and asks the
    model to apply surgical quality fixes, preserving structure.
    """
    t0 = time.time()
    result: dict[str, Any] = {"vol": vol, "chapter": chapter, "mode": mode}
    try:
        qmd = qmd_path_for(vol, chapter)
        if mode == "improve":
            user_prompt = build_improve_prompt(vol, chapter, qmd)
        elif mode == "audit":
            user_prompt = build_audit_prompt(vol, chapter, qmd)
        else:
            user_prompt = build_user_prompt(vol, chapter, qmd)
        result["prompt_chars"] = len(user_prompt)
        if dry_run:
            result["status"] = "dry-run"
            result["elapsed_s"] = round(time.time() - t0, 1)
            return result
        assert api_key is not None
        data = call_model(provider, system_prompt, user_prompt, model, api_key)
        finalize_metadata(data, vol, chapter, model)
        if mode == "improve":
            out = canonical_json_new_path(vol, chapter).with_suffix(".improved")
        elif mode == "audit":
            audit_dir = HERE / "_audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            # Prefix with ``vol`` so ``vol1/introduction`` and
            # ``vol2/introduction`` (and the two ``conclusion`` chapters)
            # do not overwrite each other's audit on disk. Pre-fix
            # filename was ``{chapter}_audit.json``.
            out = audit_dir / f"{vol}_{chapter}_audit.json"
        else:
            out = canonical_json_new_path(vol, chapter)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if mode == "audit":
            # Audit JSONs do not need to pass the quiz validator (they
            # have a different schema). Skip validation/memo for audits.
            result.update({
                "status": "ok",
                "output": str(out.relative_to(REPO_ROOT)),
                "grade": data.get("overall", {}).get("quality_grade"),
                "issues": len(data.get("per_question_issues") or []),
                "recommended_action": data.get("recommended_action"),
            })
            result["elapsed_s"] = round(time.time() - t0, 1)
            return result
        rc, vout = run_validator(out, qmd)
        memo = write_memo(chapter, vol, data, vout)
        result.update(
            {
                "status": "ok" if rc == 0 else "validator-failed",
                "validator_rc": rc,
                "output": str(out.relative_to(REPO_ROOT)),
                "memo": str(memo.relative_to(REPO_ROOT)),
                "sections": data.get("metadata", {}).get("total_sections"),
                "quizzed": data.get("metadata", {}).get("sections_with_quizzes"),
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
        "--provider",
        choices=sorted(PROVIDER_DEFAULTS.keys()),
        default=DEFAULT_PROVIDER,
        help=(
            f"LLM provider (default {DEFAULT_PROVIDER}). Each provider has a "
            f"default model: "
            + ", ".join(
                f"{k}={v['model']}" for k, v in PROVIDER_DEFAULTS.items()
            )
            + "."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "override the provider's default model. If omitted, uses "
            "PROVIDER_DEFAULTS[--provider]['model']."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallelism for --all (default 4)",
    )
    p.add_argument(
        "--mode",
        choices=["generate", "improve", "audit"],
        default="generate",
        help=(
            "generate=fresh generation (default). improve=read existing "
            "canonical JSON and apply surgical quality fixes per the "
            "improve-mode rubric. improve mode writes to "
            "{chapter}_quizzes.json.improved for human review."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build prompts but do not call the API",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip any chapter whose canonical ``{chapter}_quizzes.json`` "
            "already exists. Useful for resuming after a partial --all run."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    provider = args.provider
    defaults = PROVIDER_DEFAULTS[provider]
    model = args.model or defaults["model"]
    env_var = defaults["env"]
    api_key = os.environ.get(env_var)
    if not args.dry_run and not api_key:
        sys.exit(
            f"error: {env_var} not set (required for --provider {provider}); "
            "pass --dry-run to skip the API call"
        )

    system_prompt = load_spec()
    if args.chapter:
        parts = args.chapter.split("/", 1)
        if len(parts) != 2:
            sys.exit(f"error: --chapter must be 'vol/name', got {args.chapter!r}")
        chapters: list[tuple[str, str]] = [(parts[0], parts[1])]
    else:
        chapters = list(READING_ORDER)

    if args.skip_existing:
        # Skip a chapter only if its canonical ``_quizzes.json`` was already
        # produced by THIS pipeline (metadata.generated_by names
        # ``generate_quizzes.py`` regardless of the directory it was run
        # from — the legacy ``quiz-refresh/...`` and the renamed
        # ``quizzes/...`` descriptors both qualify). Chapters with stale
        # pre-refresh canonicals still get regenerated.
        def already_refreshed(vol: str, chap: str) -> bool:
            path = canonical_json_new_path(vol, chap).with_suffix("")
            if not path.exists():
                return False
            try:
                meta = json.loads(path.read_text()).get("metadata", {}) or {}
            except Exception:
                return False
            return "generate_quizzes" in (meta.get("generated_by") or "")

        before = len(chapters)
        chapters = [(v, c) for v, c in chapters if not already_refreshed(v, c)]
        print(f"--skip-existing: {before - len(chapters)} refreshed skipped, {len(chapters)} to run")

    print(f"spec: {SPEC_PATH.relative_to(REPO_ROOT)} ({len(system_prompt)} chars)")
    print(f"chapters: {len(chapters)}")
    print(f"provider: {provider}")
    print(f"model: {model}")
    print(f"workers: {args.workers}")
    print(f"dry-run: {args.dry_run}")
    print()

    results: list[dict[str, Any]] = []
    if args.workers <= 1 or len(chapters) == 1:
        for vol, chap in chapters:
            print(f"→ {vol}/{chap}", flush=True)
            r = generate_for_chapter(
                vol, chap, system_prompt, provider, model, api_key, args.dry_run, args.mode
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
                    provider,
                    model,
                    api_key,
                    args.dry_run,
                    args.mode,
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
