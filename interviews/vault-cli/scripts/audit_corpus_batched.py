#!/usr/bin/env python3
"""Full-corpus audit of StaffML published questions, batched per Gemini call.

The single call audits 30-40 questions for ALL judge dimensions at once
(format compliance, level fit, scenario coherence, math correctness,
title quality), and optionally proposes corrections.

Cost (full corpus, 9,446 published questions):
  - audit-only:        ~315 calls (1.3 days at the 250/day Gemini cap)
  - --propose-fixes:   ~+50%  (denser per-batch output)

Design rationale: see CORPUS_HARDENING_PLAN.md Phase 3. The earlier
audit_corpus.py (deleted in Phase 0) used 1 call per (gate × question),
which would have cost ~3,000 calls / ~12 days. Batching is
~10× cheaper for the same corpus coverage.

Usage:

    # Plan the run; no Gemini calls:
    python3 audit_corpus_batched.py --dry-run

    # Full corpus, audit-only (default):
    python3 audit_corpus_batched.py --all

    # Subset by track:
    python3 audit_corpus_batched.py --tracks cloud,edge

    # Same, but ALSO ask Gemini to propose fixes for any failures
    # (humans review via apply_corrections.py — never auto-apply):
    python3 audit_corpus_batched.py --all --propose-fixes

    # Cap calls (resume rest tomorrow):
    python3 audit_corpus_batched.py --all --max-calls 100

Outputs (under interviews/vault/_pipeline/runs/<UTC-timestamp>/):
    00_config.json                — what was run, with what flags
    01_audit.json                 — list of per-question rows
    02_proposed_corrections.json  — only with --propose-fixes
    AUDIT_REPORT.md               — synthesis (one extra Gemini call)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Reuse the shared infrastructure rather than re-defining marker
# constants and Gemini-call wrappers locally.
from _batching import DEFAULT_WRAPPER_CHARS, MAX_PROMPT_CHARS, pack_batches
from _judges import (
    COMMON_MISTAKE_MARKERS,
    FAILURE_MODE_TAXONOMY,
    GEMINI_MODEL,
    call_gemini_judge,
    gate_format,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
PIPELINE_DIR = VAULT_DIR / "_pipeline"
RUNS_DIR = PIPELINE_DIR / "runs"

# Tuning. Default 30 questions/call leaves headroom for the prompt
# wrapper and per-question response. Drop lower if seeing JSON parse
# failures (Gemini sometimes truncates very-long outputs).
DEFAULT_BATCH_SIZE = 30
PROPOSE_FIXES_BATCH_SIZE = 20  # per-question response is ~3× larger
DEFAULT_MAX_CALLS = 250
INTER_CALL_DELAY_S = 4

# Per-question payload truncation. Each candidate body is bounded so a
# single huge scenario doesn't blow the prompt budget.
SCENARIO_CHAR_BUDGET = 1_500
SOLUTION_CHAR_BUDGET = 800
COMMON_MISTAKE_CHAR_BUDGET = 800
NAPKIN_MATH_CHAR_BUDGET = 1_500


# ─── corpus loading ───────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            d = yaml.safe_load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def load_published_corpus(tracks: set[str] | None = None) -> dict[str, dict]:
    """qid -> full body for every status=published YAML.

    If ``tracks`` is provided, only return questions from those tracks.
    """
    out: dict[str, dict] = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        d = load_yaml(path)
        if not d or d.get("status") != "published":
            continue
        if tracks and d.get("track") not in tracks:
            continue
        qid = d.get("id")
        if qid:
            out[qid] = d
    return out


# ─── payload construction ─────────────────────────────────────────────────


def _truncate(s: str | None, budget: int) -> str:
    if not s:
        return ""
    if len(s) <= budget:
        return s
    return s[:budget] + " [...truncated]"


def candidate_payload(q: dict) -> dict:
    """Compact projection of a question for the audit prompt.

    Includes every field the judge needs: classification, full content,
    format markers (so the judge can check format compliance directly
    without the host-side regex pre-pass).
    """
    d = q.get("details") or {}
    return {
        "qid": q.get("id"),
        "track": q.get("track"),
        "level": q.get("level"),
        "zone": q.get("zone"),
        "topic": q.get("topic"),
        "competency_area": q.get("competency_area"),
        "bloom_level": q.get("bloom_level"),
        "title": q.get("title"),
        "scenario": _truncate(q.get("scenario"), SCENARIO_CHAR_BUDGET),
        "question": q.get("question") or "",
        "realistic_solution": _truncate(d.get("realistic_solution"),
                                         SOLUTION_CHAR_BUDGET),
        "common_mistake": _truncate(d.get("common_mistake"),
                                     COMMON_MISTAKE_CHAR_BUDGET),
        "napkin_math": _truncate(d.get("napkin_math"),
                                  NAPKIN_MATH_CHAR_BUDGET),
    }


# ─── prompt construction ─────────────────────────────────────────────────


LEVEL_CALIBRATION = """LEVEL ↔ BLOOM CALIBRATION:
  L1 = remember     — recall a fact, definition, ratio
  L2 = understand   — explain a concept, identify a category
  L3 = apply        — execute a calculation given the inputs
  L4 = analyze      — decompose, root-cause, pick from competing trade-offs
  L5 = evaluate     — judge a design quantitatively, weigh alternatives
  L6+ = create      — synthesize a new design under unusual constraints

LEVEL-FIT FAILURE MODES (return level_fit=fail on any of):
  - "Level inflation": L3+ stamped on what's actually a recall or
    simple-multiplication problem (every input given upfront, only
    mechanical computation). The 2026-05-01 audit caught edge-2537
    failing this — claimed L3 but was a one-step multiplication.
  - "Verb mismatch": question's actual verb is more than one Bloom step
    away from the level field's expected verb.
  - "No real judgement required": L4+ must require decomposition,
    root-cause reasoning, or trade-off decisions — not just
    mechanical computation.
"""


FORMAT_REQUIREMENTS = f"""FORMAT-MARKER REQUIREMENTS (only if the field is non-empty):
  common_mistake (when present) MUST contain all 3 of these literal markers:
    {', '.join(repr(m) for m in COMMON_MISTAKE_MARKERS)}
  napkin_math (when present) MUST contain ALL of these (prefix-match):
    '**Assumptions' (accepts 'Assumptions:' or 'Assumptions & Constraints:'),
    '**Calculations:**',
    '**Conclusion' (accepts 'Conclusion:' or 'Conclusion & Interpretation:').
  Do NOT flag absence of the field itself; only flag PRESENT-AND-MALFORMED.
"""


PROPOSE_FIXES_INSTRUCTIONS = """ADDITIONALLY, for any candidate that fails ANY gate, populate
``suggested_corrections``. Per the project's correction policy
(CORPUS_HARDENING_PLAN.md §10):

  - title: rewrite ONLY when title_quality is "placeholder" or "malformed".
    Keep ≤ 120 chars, descriptive, no trailing period, no LaTeX.

  - common_mistake / napkin_math: rewrite to add the missing markers when
    format_compliance=fail. PRESERVE the author's content; just structure
    it under the canonical markers. Do not invent new content.

  - realistic_solution + napkin_math: when math_correct=fail, REWRITE BOTH
    AS A UNIT — the solution typically depends on the napkin_math number,
    so fixing the math without fixing the solution leaves the question
    internally inconsistent.

  - level: when level_fit=fail, RELABEL DOWN to the actual level the
    question demands. Do NOT attempt to rewrite the question to match a
    higher level — that's a separate authoring task. Output the recommended
    level value as one of: L1, L2, L3, L4, L5, L6+.

If a particular sub-field doesn't need correcting, OMIT it from
suggested_corrections rather than echoing the original value.
"""


def build_audit_prompt(batch: list[dict], *, propose_fixes: bool) -> str:
    """Assemble the per-batch audit prompt."""
    payloads = [candidate_payload(q) for q in batch]

    schema = """{
  "results": [
    {
      "qid": "<id>",
      "format_compliance":      "pass" | "fail",
      "format_issues":          ["<specific marker missing>", ...],
      "level_fit":              "pass" | "fail" | "skip",
      "level_fit_rationale":    "<one sentence pointing to the SPECIFIC failure mode if fail>",
      "coherence":              "pass" | "fail",
      "coherence_failure_mode": "physical_absurdity" | "vendor_fabrication" | "mismatch" | "arithmetic" | "none",
      "coherence_rationale":    "<one sentence pointing to the SPECIFIC issue if fail>",
      "math_correct":           "pass" | "fail" | "no_math",
      "math_errors":            ["<specific issue: claims X = Y but X = Z>", ...],
      "title_quality":          "good" | "placeholder" | "malformed"
"""
    if propose_fixes:
        schema += """,
      "suggested_corrections": {
        "title":              "<rewritten title — only if title_quality is placeholder/malformed>",
        "common_mistake":     "<rewritten with markers — only if format_compliance=fail or coherence=fail>",
        "napkin_math":        "<rewritten with markers — only if format_compliance=fail or math_correct=fail>",
        "realistic_solution": "<rewritten — only if math_correct=fail>",
        "level":              "<L1..L6+ — only if level_fit=fail; relabel DOWN to the actual level>"
      }
"""
    schema += """    },
    ...
  ]
}"""

    parts = [
        "You are independently auditing the StaffML ML-systems interview corpus.",
        "Each candidate below is a published question. Audit them against the",
        "failure-mode taxonomies and return per-question verdicts.",
        "",
        FAILURE_MODE_TAXONOMY,
        "",
        LEVEL_CALIBRATION,
        "",
        FORMAT_REQUIREMENTS,
    ]
    if propose_fixes:
        parts += ["", PROPOSE_FIXES_INSTRUCTIONS]

    parts += [
        "",
        "For each candidate, return STRICT JSON with no prose or fences:",
        "",
        schema,
        "",
        "Return ONE entry per candidate in the input order. n entries for n candidates.",
        "",
        f"CANDIDATES (n={len(payloads)}):",
        json.dumps(payloads, indent=2),
    ]
    return "\n".join(parts)


# ─── result handling ─────────────────────────────────────────────────────


def normalize_response(resp: dict | None, batch: list[dict]) -> list[dict]:
    """Convert a raw Gemini response into a list of one row per qid.

    Always returns len(batch) rows. Missing-from-response qids get an
    "error" placeholder so the per-batch persistence still has a row
    for every candidate.
    """
    expected_qids = [q.get("id") for q in batch]
    by_qid: dict[str, dict] = {}
    if resp and isinstance(resp.get("results"), list):
        for r in resp["results"]:
            if isinstance(r, dict) and r.get("qid"):
                by_qid[r["qid"]] = r

    rows: list[dict] = []
    for qid in expected_qids:
        if qid in by_qid:
            rows.append(by_qid[qid])
        else:
            rows.append({
                "qid": qid,
                "format_compliance": "error",
                "level_fit": "error",
                "coherence": "error",
                "math_correct": "error",
                "title_quality": "error",
                "_reason": "missing from Gemini response",
            })
    return rows


def cross_check_format(rows: list[dict], batch: list[dict]) -> list[dict]:
    """Compare host-side regex format check vs. Gemini's verdict.

    Adds ``format_regex`` and ``format_agree`` fields. Detects cases
    where Gemini misses a marker the regex catches (or vice versa) —
    useful as a sanity check on Gemini's adherence to the strict
    marker requirement.
    """
    qid_to_q = {q.get("id"): q for q in batch}
    for row in rows:
        q = qid_to_q.get(row.get("qid"))
        if not q:
            continue
        regex_result = gate_format(q)
        row["format_regex"] = regex_result["verdict"]
        row["format_regex_issues"] = regex_result["issues"]
        gem_v = row.get("format_compliance")
        row["format_agree"] = (gem_v == regex_result["verdict"])
    return rows


# ─── run loop ────────────────────────────────────────────────────────────


def run_audit(
    *,
    targets: list[dict],
    outdir: Path,
    propose_fixes: bool,
    batch_size: int,
    max_calls: int,
    dry_run: bool,
) -> dict:
    """Drive the per-batch audit loop. Persist after each call."""
    batches = pack_batches(
        targets,
        payload_for=candidate_payload,
        max_chars=MAX_PROMPT_CHARS,
        wrapper_chars=DEFAULT_WRAPPER_CHARS,
        max_items_per_batch=batch_size,
    )
    n_batches_target = len(batches)
    capped = min(n_batches_target, max_calls)

    print(f"  packed: {len(targets)} questions → {n_batches_target} batches "
          f"(target {batch_size}/batch)")
    print(f"  cap: {max_calls} calls; will run {capped} batch(es) this invocation")
    if dry_run:
        sizes = [len(b) for b in batches[:5]]
        chars = [
            sum(len(json.dumps(candidate_payload(q))) for q in b)
            for b in batches[:5]
        ]
        print(f"  first 5 batch sizes: {sizes}; payload chars: {chars}")
        return {"batches_planned": n_batches_target, "calls_planned": capped,
                "batches_run": 0, "rows": []}

    audit_path = outdir / "01_audit.json"
    rows: list[dict] = []
    if audit_path.exists():
        # Resume: load whatever's already persisted.
        try:
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(existing.get("rows"), list):
                rows = existing["rows"]
        except json.JSONDecodeError:
            pass
    seen_qids = {r.get("qid") for r in rows if r.get("qid")}

    started = time.time()
    calls_made = 0

    for i, batch in enumerate(batches, start=1):
        # Skip batches whose first qid is already in `seen_qids` (a
        # prior partial run completed it).
        batch_qids = [q.get("id") for q in batch]
        if all(q in seen_qids for q in batch_qids):
            print(f"  [{i:3d}/{n_batches_target}] skip — already audited")
            continue

        if calls_made >= max_calls:
            print(f"  [{i:3d}/{n_batches_target}] HALT — call cap reached")
            break

        prompt = build_audit_prompt(batch, propose_fixes=propose_fixes)
        prompt_chars = len(prompt)
        print(f"  [{i:3d}/{n_batches_target}] {len(batch)} questions, "
              f"{prompt_chars // 1000}K char prompt")

        resp = call_gemini_judge(prompt)
        calls_made += 1

        new_rows = normalize_response(resp, batch)
        new_rows = cross_check_format(new_rows, batch)

        # Drop any prior rows for these qids (resume can revise) then append.
        rows = [r for r in rows if r.get("qid") not in set(batch_qids)]
        rows.extend(new_rows)
        seen_qids.update(batch_qids)

        # Persist after every call so a Ctrl-C / timeout doesn't lose work.
        outdir.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": GEMINI_MODEL,
            "propose_fixes": propose_fixes,
            "batches_run_so_far": calls_made,
            "rows_so_far": len(rows),
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")

        if i < n_batches_target and calls_made < max_calls:
            time.sleep(INTER_CALL_DELAY_S)

    elapsed = time.time() - started
    print(f"\n  elapsed {elapsed:.1f}s  calls={calls_made}  rows={len(rows)}")

    return {
        "batches_planned": n_batches_target,
        "calls_planned": capped,
        "batches_run": calls_made,
        "rows": rows,
    }


# ─── main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--all", action="store_true",
                     help="audit the full published corpus (default)")
    src.add_argument("--tracks", type=str, default=None,
                     help="comma-separated track filter (e.g. cloud,edge)")
    src.add_argument("--qids", type=str, default=None,
                     help="comma-separated explicit qid list")

    ap.add_argument("--propose-fixes", action="store_true",
                    help="ALSO ask Gemini to propose corrections for each "
                         "failure (humans review separately; never "
                         "auto-applied)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"questions per batch (default "
                         f"{DEFAULT_BATCH_SIZE} audit-only, "
                         f"{PROPOSE_FIXES_BATCH_SIZE} with --propose-fixes)")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                    help=f"cap on Gemini calls this invocation "
                         f"(default {DEFAULT_MAX_CALLS}). Resume by re-running.")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir (default _pipeline/runs/<UTC-timestamp>/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without making Gemini calls")
    args = ap.parse_args()

    # Resolve target set.
    tracks = None
    if args.tracks:
        tracks = {t.strip() for t in args.tracks.split(",") if t.strip()}

    print("loading published corpus ...")
    corpus = load_published_corpus(tracks=tracks)
    print(f"  {len(corpus)} candidates")

    if args.qids:
        qid_set = {q.strip() for q in args.qids.split(",") if q.strip()}
        targets = [corpus[q] for q in qid_set if q in corpus]
        missing = qid_set - set(corpus.keys())
        if missing:
            print(f"  WARN: {len(missing)} qid(s) not in corpus: "
                  f"{sorted(missing)[:5]}", file=sys.stderr)
    else:
        # Stable order: sorted by qid for deterministic batching across runs.
        targets = [corpus[q] for q in sorted(corpus.keys())]

    if not targets:
        print("no targets", file=sys.stderr)
        return 1

    batch_size = args.batch_size or (
        PROPOSE_FIXES_BATCH_SIZE if args.propose_fixes else DEFAULT_BATCH_SIZE
    )

    # Set up the run dir.
    if args.output:
        outdir = args.output
    else:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        outdir = RUNS_DIR / ts
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": GEMINI_MODEL,
        "args": {
            "all": args.all,
            "tracks": sorted(tracks) if tracks else None,
            "qids_count": len(args.qids.split(",")) if args.qids else None,
            "propose_fixes": args.propose_fixes,
            "batch_size": batch_size,
            "max_calls": args.max_calls,
            "dry_run": args.dry_run,
        },
        "candidate_count": len(targets),
    }
    (outdir / "00_config.json").write_text(json.dumps(config, indent=2) + "\n",
                                            encoding="utf-8")

    print(f"\nrun dir: {outdir.relative_to(REPO_ROOT)}")
    summary = run_audit(
        targets=targets,
        outdir=outdir,
        propose_fixes=args.propose_fixes,
        batch_size=batch_size,
        max_calls=args.max_calls,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    # Cheap synthesis: per-gate counts. (The Gemini-synthesis call that
    # produces AUDIT_REPORT.md is run by a separate command —
    # `vault audit synthesize` in CORPUS_HARDENING_PLAN.md Phase 8 — so
    # the audit run is separable from the report generation.)
    rows = summary.get("rows", [])
    counts = {
        "format_compliance": Counter(r.get("format_compliance") for r in rows),
        "level_fit":         Counter(r.get("level_fit") for r in rows),
        "coherence":         Counter(r.get("coherence") for r in rows),
        "math_correct":      Counter(r.get("math_correct") for r in rows),
        "title_quality":     Counter(r.get("title_quality") for r in rows),
    }
    by_track: dict[str, Counter] = defaultdict(Counter)
    qid_to_track = {q.get("id"): q.get("track") for q in targets}
    for r in rows:
        track = qid_to_track.get(r.get("qid"), "?")
        by_track[track]["total"] += 1
        for gate in ("format_compliance", "level_fit", "coherence",
                      "math_correct", "title_quality"):
            if r.get(gate) == "fail":
                by_track[track][gate] += 1

    print("\nsummary by gate:")
    for gate, counter in counts.items():
        print(f"  {gate:20s} {dict(counter)}")
    print("\nsummary by track:")
    for track in sorted(by_track):
        print(f"  {track:7s} {dict(by_track[track])}")

    print(f"\nwrote {(outdir / '01_audit.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
