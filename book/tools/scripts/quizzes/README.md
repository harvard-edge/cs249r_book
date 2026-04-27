# Quizzes — runner

This directory regenerates `{chapter}_quizzes.json` from chapter prose for
every chapter in Vol1 and Vol2. It is the **plumbing** half of the
spec-plus-runner pattern. The **spec** — what "good" looks like — lives at:

    .claude/docs/shared/quiz-generation.md

Every decision about taxonomy, per-type answer formats, quality bar,
anti-patterns, schema, and difficulty progression is in the spec. When
the rules change, they change there; this directory does not redefine
any of those rules.

## Scripts

| Script | Role |
|---|---|
| `generate_quizzes.py` | The runner. Reads the spec, calls the Claude API, writes `{chapter}_quizzes.json.new` and a per-chapter memo. Supports `--chapter vol1/training` (single) or `--all` (parallel fan-out). |
| `extract_anchors.py`  | Parses `##` + `###` `{#sec-…}` anchors from a `.qmd`. |
| `build_prior_vocab.py` | Unions every prior chapter's glossary into the context for a given chapter. |
| `validate_quiz_json.py` | Schema + anchor + metadata-count + anti-shuffle-bug validator. Runs after every generator pass. |

## Usage

```bash
# Prerequisite: ANTHROPIC_API_KEY in your environment.

# Regenerate one chapter:
python3 book/tools/scripts/quizzes/generate_quizzes.py \
    --chapter vol1/training

# Regenerate all 33 in parallel (4 workers):
python3 book/tools/scripts/quizzes/generate_quizzes.py \
    --all --workers 4

# Dry-run (no API calls; prints prompt assembly status):
python3 book/tools/scripts/quizzes/generate_quizzes.py \
    --all --dry-run
```

## Output layout

For each chapter the generator writes two artifacts:

1. `book/quarto/contents/{vol}/{chapter}/{chapter}_quizzes.json.new` —
   the regenerated JSON. Rename (drop `.new`) after human review.
2. `book/tools/scripts/quizzes/_reviews/{chapter}_memo.md` —
   short summary of coverage, question counts, type mix, and any
   `quiz_needed: false` entries with their rationale.

The canonical `.json` at the non-`.new` path is overwritten only after a
human reviews the diff and renames manually. The generator never
overwrites without the staging suffix.

## Legacy reference

`book/tools/scripts/quizzes/_legacy/quizzes_reference.py` is a
read-only snapshot of the legacy `quizzes.py` (removed in `2bc58dba3d`).
Kept as a reference for the SYSTEM_PROMPT, per-type criteria, and the
known-buggy post-shuffle logic that this new pipeline deliberately does
not replicate. Do not execute it; it is for historical inspection only.

## Pattern

This directory is the first implementation of a broader pattern for
derivative artifacts in the book: **spec + lean runner + validator +
human gate**. When you need to regenerate glossaries, concepts, index
entries, or similar derived content from prose, follow the same shape:

- Put the canonical rules in `.claude/rules/{artifact}-generation.md`.
- Write a lean runner in `book/tools/scripts/{artifact}/generate.py`.
- Put the validator beside it.
- Optionally add an interactive persona at
  `.claude/agents/{artifact}-refresh-agent.md`.

The runner and the validator are small enough to audit. The rules file
is the IP.
