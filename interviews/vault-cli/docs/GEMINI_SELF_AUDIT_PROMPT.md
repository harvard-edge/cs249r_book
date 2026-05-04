# Self-running Gemini audit prompt

A single prompt that lets Gemini CLI walk the corpus and audit it directly,
without the Python `audit_corpus_batched.py` wrapper. Use when the wrapper is
flaky (rate limits, exit 55, etc.) or when you want Gemini to checkpoint
results to disk as it goes.

## How to run

```bash
cd /Users/VJ/GitHub/MLSysBook-yaml-audit
gemini -m gemini-3.1-pro-preview --yolo --skip-trust \
  -p "$(cat interviews/vault-cli/docs/GEMINI_SELF_AUDIT_PROMPT.md | sed -n '/^## PROMPT BEGIN/,/^## PROMPT END/p' | sed '1d;$d')" \
  < /dev/null
```

Or paste the prompt body interactively into a fresh `gemini` session.

The audit results are written to `interviews/vault/_pipeline/runs/gemini-self-audit/01_audit.jsonl` (one JSON record per line, appended). Resumable: re-run picks up where it left off by skipping qids already in the file.

## PROMPT BEGIN
You are auditing the StaffML ML-systems interview corpus. Each item is a YAML
file under `interviews/vault/questions/<track>/<area>/<id>.yaml`. Audit only
files where `status: published`.

OUTPUT TARGET (write here, append, one JSON object per line):
  `interviews/vault/_pipeline/runs/gemini-self-audit/01_audit.jsonl`
Create the directory if it doesn't exist. If the file already exists, read it
first, collect the set of qids already audited, and SKIP those — this lets
the run resume after an interruption.

WORK PLAN
1. List published YAML files under `interviews/vault/questions/`. Track them
   in lexical order (sorted by track, then area, then qid).
2. For each unaudited published file:
   a. Read the YAML. Extract: id, track, level, zone, topic, competency_area,
      title, scenario, question (if present), and the entire `details` block
      (realistic_solution, common_mistake, napkin_math, options, correct_index).
   b. Run the five gates below.
   c. Append a single JSON record to the output file (with a trailing newline).
3. Every 25 questions, print a one-line progress update to stdout:
   `progress: <N>/<TOTAL> · pass=<P> fail=<F> · current=<qid>`.
4. When done, print a summary block: per-gate pass/fail counts, per-track
   totals, top 10 failure rationales by frequency.

THE FIVE GATES

  Gate A — format_compliance
    common_mistake (when non-empty) must contain in order:
      "**The Pitfall:**"  "**The Rationale:**"  "**The Consequence:**"
    napkin_math (when non-empty) must contain in order:
      "**Assumptions" (or "**Assumptions & Constraints:**")
      "**Calculations:**"
      "**Conclusion" (or "**Conclusion & Interpretation:**")
    Verdict: pass | fail · with `format_issues: [<missing markers>]` on fail.

  Gate B — level_fit
    The `level` field claims a Bloom-mapped depth (L1=Remember .. L6+=Create
    Staff-level). Read the question + scenario + realistic_solution and judge
    whether the claimed level matches what the question actually demands.
    Verdict: pass | fail
    On fail: `level_fit_rationale` (1-2 sentences), `suggested_level` (e.g. "L3").

  Gate C — coherence
    Reject (verdict=fail) on any of:
      1. PHYSICAL ABSURDITY: hardware/software numbers violate real-world
         bounds (e.g., NPU wake-up >50ms, smartphone pulling 50W, latency
         >5× off realistic for the named hardware).
      2. VENDOR-NAME FABRICATION: hardware/framework/benchmark names that
         don't exist or are misattributed (e.g., "Coral Edge TPU XL" — no XL
         variant). Treat ambiguous-but-plausible as ok; flag clearly invented.
      3. SCENARIO/QUESTION/SOLUTION MISMATCH: question doesn't follow from
         scenario, realistic_solution doesn't actually answer the question,
         or numbers contradict across fields.
      4. ARITHMETIC IN SCENARIO: scenario contains a stated calculation that
         is wrong on its face (this is separate from gate D's napkin math).
    Verdict: pass | fail · `coherence_failure_mode` (one of: physical-absurdity,
    vendor-fabrication, mismatch, scenario-arithmetic, none) · `coherence_rationale`.

  Gate D — math_correct
    Independently re-derive the napkin_math calculations. Are the assumptions
    sound? Do the unit conversions check out? Does the conclusion follow?
    Verdict: pass | fail · `math_errors: [<short error list>]` on fail.

  Gate E — title_quality
    Title (≤120 chars, plaintext, no LaTeX, no markdown, no underscores).
    Verdicts:
      good        — descriptive, concrete, names the operative concept
      generic     — too vague to retrieve ("Cloud Q1", "Memory Question")
      placeholder — clearly an unfilled placeholder ("TODO", "draft", "x")
    On non-good: `title_suggestion` if you can produce a short concrete one.

OUTPUT JSON SHAPE (one per line in `01_audit.jsonl`)

  {
    "qid": "cloud-2297",
    "track": "cloud",
    "format_compliance": "pass" | "fail",
    "format_issues": [],
    "level_fit": "pass" | "fail",
    "level_fit_rationale": "...",
    "suggested_level": "L4" | null,
    "coherence": "pass" | "fail",
    "coherence_failure_mode": "none" | "physical-absurdity" | ...,
    "coherence_rationale": "...",
    "math_correct": "pass" | "fail",
    "math_errors": [],
    "title_quality": "good" | "generic" | "placeholder",
    "title_suggestion": null
  }

CRITICAL RULES

  - Append only. Do not rewrite the file. Each batch you complete should be
    durable on disk so a kill-9 mid-run loses at most one item.
  - Do not modify any YAML. This is read-only audit; corrections are a
    downstream task.
  - Skip non-published statuses. Do not audit drafts, flagged, deleted,
    or archived.
  - Process at least 200 items per session. Print progress every 25.
  - If you encounter a YAML you can't parse, write a record with
    `qid: "<filename-stem>"` and all gates `error`, plus `_reason: "..."`.
  - If you hit a tool / network error, write what you have so far, then
    print `STOPPING: <reason>` and exit cleanly. Do not crash.

START NOW. First action: read the existing
`interviews/vault/_pipeline/runs/gemini-self-audit/01_audit.jsonl` (or note
that it doesn't exist), then list published YAMLs.
## PROMPT END

## Notes

- Gemini's `--yolo --skip-trust` are required: the first lets it use file
  tools without prompting, the second bypasses the workspace-trust gate that
  silently breaks `--yolo` in unfamiliar `cwd`s.
- Gemini's tool-use latency is ~1-3 seconds per file read on the local FS.
  9,446 published YAMLs × 5s = ~13 hours wall-clock if Gemini reads serially.
  Recommend running on a slice (one track at a time) and concatenating
  results.
- To slice by track, change the WORK PLAN line to:
  `1. List published YAML files under interviews/vault/questions/cloud/.`
- To resume across sessions, the JSONL append + skip-already-audited
  contract makes this safe: just re-run the same prompt.
