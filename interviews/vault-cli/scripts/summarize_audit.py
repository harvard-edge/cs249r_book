#!/usr/bin/env python3
"""Summarize an audit_corpus_batched run into an AUDIT_FINDINGS markdown.

Reads a 01_audit.json file produced by audit_corpus_batched.py and
emits a human-readable triage doc with:

  - per-gate pass/fail/error counts
  - per-track per-gate failure rate matrix
  - top failure categories (level_fit failure modes, coherence failure
    modes)
  - lists of qids requiring human attention, grouped by priority
  - recommended next-step actions per category

CORPUS_HARDENING_PLAN.md Phase 4.

Output is markdown intended to live at:

    interviews/vault-cli/docs/AUDIT_FINDINGS_<YYYY-MM-DD>.md

Usage:

    python3 interviews/vault-cli/scripts/summarize_audit.py \\
        --input interviews/vault/_pipeline/runs/full-corpus-20260503/01_audit.json \\
        --output interviews/vault-cli/docs/AUDIT_FINDINGS_2026-05-03.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"


# ─── corpus + helpers ────────────────────────────────────────────────────


def load_track_index() -> dict[str, str]:
    """qid → track lookup. Used to bucket findings by track without
    requiring the audit JSON to carry the track field on every row."""
    out: dict[str, str] = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        try:
            with path.open(encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("id") and d.get("track"):
            out[d["id"]] = d["track"]
    return out


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{100 * n / total:.1f}%"


def truncate_words(text: str, max_chars: int) -> str:
    """Truncate at the last word boundary before max_chars to avoid
    mid-word truncations like 'claimin' (which would otherwise trip
    spell-checkers). Adds an ellipsis when truncated."""
    if not text or len(text) <= max_chars:
        return text or ""
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:  # don't cut too aggressively
        cut = cut[:last_space]
    return cut + "..."


# ─── markdown builders ───────────────────────────────────────────────────


def build_executive_summary(rows: list[dict], track_index: dict[str, str]) -> str:
    n_total = len(rows)
    by_gate = {
        gate: Counter(r.get(gate) for r in rows)
        for gate in ("format_compliance", "level_fit", "coherence",
                      "math_correct", "title_quality")
    }
    out = ["## Executive summary", ""]
    out += [
        f"- **{n_total}** questions audited",
        f"- **{by_gate['format_compliance'].get('error', 0)}** errored "
        f"(no Gemini response — should be retried)",
        "",
    ]

    out += ["| gate | pass | fail | other |", "|---|---:|---:|---:|"]
    for gate in ("format_compliance", "level_fit", "coherence",
                  "math_correct"):
        c = by_gate[gate]
        other_total = sum(v for k, v in c.items()
                          if k not in ("pass", "fail"))
        out.append(f"| {gate} | {c.get('pass', 0)} | {c.get('fail', 0)} | "
                   f"{other_total} |")
    title = by_gate["title_quality"]
    out.append(f"| title_quality | {title.get('good', 0)} good | "
               f"{title.get('placeholder', 0)} placeholder + "
               f"{title.get('malformed', 0)} malformed | "
               f"{title.get('error', 0)} error |")

    return "\n".join(out) + "\n"


def build_per_track_matrix(rows: list[dict], track_index: dict[str, str]) -> str:
    by_track: dict[str, Counter] = defaultdict(Counter)
    track_totals: Counter = Counter()
    for r in rows:
        qid = r.get("qid")
        track = track_index.get(qid, "?")
        track_totals[track] += 1
        for gate in ("format_compliance", "level_fit", "coherence",
                      "math_correct"):
            if r.get(gate) == "fail":
                by_track[track][gate] += 1
            elif r.get(gate) == "error":
                by_track[track][f"{gate}_err"] += 1

    out = ["## Per-track failure rates", ""]
    out += ["| track | total | format | level_fit | coherence | math |",
            "|---|---:|---:|---:|---:|---:|"]
    for track in sorted(track_totals):
        n = track_totals[track]
        c = by_track[track]
        out.append(
            f"| {track} | {n} | "
            f"{c['format_compliance']} ({pct(c['format_compliance'], n)}) | "
            f"{c['level_fit']} ({pct(c['level_fit'], n)}) | "
            f"{c['coherence']} ({pct(c['coherence'], n)}) | "
            f"{c['math_correct']} ({pct(c['math_correct'], n)}) |"
        )
    return "\n".join(out) + "\n"


def build_failure_modes(rows: list[dict]) -> str:
    coherence_modes = Counter(
        r.get("coherence_failure_mode")
        for r in rows
        if r.get("coherence") == "fail"
    )
    out = ["## Coherence failure-mode breakdown", "",
           "When coherence=fail, Gemini classifies the failure into one of "
           "four modes. The distribution tells us where to focus targeted "
           "fixes.", "",
           "| failure_mode | count |", "|---|---:|"]
    for mode, n in coherence_modes.most_common():
        if mode is None or mode == "none":
            continue
        out.append(f"| {mode or '(unspecified)'} | {n} |")
    return "\n".join(out) + "\n"


def build_priority_lists(
    rows: list[dict],
    track_index: dict[str, str],
    *,
    qid_limit: int = 25,
) -> str:
    """Lists of qids needing the most-attention review, grouped by
    severity. Capped at qid_limit per category to keep the doc readable.
    """
    out = ["## Priority lists for human review", ""]

    # Math errors are the highest-stakes — wrong arithmetic published
    # under a verified question is the worst outcome.
    math_fails = [r for r in rows if r.get("math_correct") == "fail"]
    out += [f"### Math errors — {len(math_fails)} questions",
            "",
            "**Highest priority.** Each requires per-instance human review "
            "(per CORPUS_HARDENING_PLAN.md §10 Q2). When fixing, rewrite "
            "BOTH napkin_math AND realistic_solution as a unit.", ""]
    if not math_fails:
        out.append("(none)")
    else:
        out.append("| qid | track | first error |")
        out.append("|---|---|---|")
        for r in math_fails[:qid_limit]:
            qid = r.get("qid")
            errs = r.get("math_errors") or []
            err = truncate_words(errs[0] if errs else "", 80)
            out.append(f"| `{qid}` | {track_index.get(qid, '?')} | {err} |")
        if len(math_fails) > qid_limit:
            out.append(f"\n_…and {len(math_fails) - qid_limit} more._")
    out.append("")

    # Coherence failures by mode (vendor fabrication, physical absurdity).
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("coherence") == "fail":
            by_mode[r.get("coherence_failure_mode") or "?"].append(r)

    for mode_label, mode_key in [
        ("Vendor fabrication", "vendor_fabrication"),
        ("Physical absurdity", "physical_absurdity"),
        ("Scenario/solution mismatch", "mismatch"),
    ]:
        items = by_mode.get(mode_key, [])
        out += [f"### {mode_label} — {len(items)} questions", ""]
        if not items:
            out.append("(none)")
        else:
            out.append("| qid | track | rationale |")
            out.append("|---|---|---|")
            for r in items[:qid_limit]:
                qid = r.get("qid")
                rat = truncate_words(r.get("coherence_rationale") or "", 100)
                out.append(f"| `{qid}` | {track_index.get(qid, '?')} | "
                           f"{rat} |")
            if len(items) > qid_limit:
                out.append(f"\n_…and {len(items) - qid_limit} more._")
        out.append("")

    # Level inflation
    level_fails = [r for r in rows if r.get("level_fit") == "fail"]
    out += [f"### Level inflation — {len(level_fails)} questions",
            "",
            "Per CORPUS_HARDENING_PLAN.md §10 Q3, default disposition is "
            "**relabel down** to the actual cognitive level. Rewriting "
            "the question up is a separate authoring task, not a "
            "Phase-5 concern.", ""]
    if level_fails:
        out.append("| qid | track | claimed level | rationale |")
        out.append("|---|---|---|---|")
        for r in level_fails[:qid_limit]:
            qid = r.get("qid")
            rat = truncate_words(r.get("level_fit_rationale") or "", 80)
            out.append(f"| `{qid}` | {track_index.get(qid, '?')} | ? | "
                       f"{rat} |")
        if len(level_fails) > qid_limit:
            out.append(f"\n_…and {len(level_fails) - qid_limit} more._")
    out.append("")

    # Placeholder titles
    placeholders = [r for r in rows if r.get("title_quality") == "placeholder"]
    out += [f"### Placeholder titles — {len(placeholders)} questions",
            "",
            "Phase 7 will batch these for Gemini-proposed replacements "
            "(~1 call per 5 placeholders). All require human review of "
            "the proposed title.", ""]
    if placeholders:
        out.append(f"qids (first {min(qid_limit, len(placeholders))}): "
                   + ", ".join(f"`{r['qid']}`"
                               for r in placeholders[:qid_limit]))
        if len(placeholders) > qid_limit:
            out.append(f"\n_…and {len(placeholders) - qid_limit} more._")
    out.append("")

    return "\n".join(out) + "\n"


def build_format_disagreements(rows: list[dict]) -> str:
    """Sanity check: where do Gemini and the regex disagree on format?
    Each disagreement is a hint that one of the prompts has a bug."""
    disagreements = [r for r in rows
                     if r.get("format_agree") is False]
    out = ["## Format-compliance: regex vs. Gemini", "",
           f"Of the rows where both verdicts are present, "
           f"**{len(disagreements)}** disagree. Regex is the source of "
           f"truth (it's mechanical); disagreements indicate Gemini "
           f"missed a marker the regex caught (or vice versa).", ""]
    if disagreements:
        out.append("| qid | gemini | regex | regex_issues |")
        out.append("|---|---|---|---|")
        for r in disagreements[:20]:
            issues = (r.get("format_regex_issues") or [None])[0]
            issues = truncate_words(issues or "", 80)
            out.append(f"| `{r.get('qid')}` | {r.get('format_compliance')} | "
                       f"{r.get('format_regex')} | {issues} |")
        if len(disagreements) > 20:
            out.append(f"\n_…and {len(disagreements) - 20} more._")
    out.append("")
    return "\n".join(out) + "\n"


def build_recommendations(rows: list[dict]) -> str:
    n_format = sum(1 for r in rows
                   if r.get("format_compliance") == "fail"
                   or r.get("format_regex") == "fail")
    n_math = sum(1 for r in rows if r.get("math_correct") == "fail")
    n_level = sum(1 for r in rows if r.get("level_fit") == "fail")
    n_coherence = sum(1 for r in rows if r.get("coherence") == "fail")
    n_placeholder = sum(1 for r in rows
                         if r.get("title_quality") == "placeholder")
    n_error = sum(1 for r in rows if r.get("format_compliance") == "error")

    out = ["## Recommendations", ""]
    out += [
        "1. **Resume to clear errored batches.** "
        f"{n_error} rows show `format_compliance: error` — these batches "
        "didn't get a Gemini response. Re-running "
        "`audit_corpus_batched.py --output <same-dir>` retries them.",
        "",
        "2. **Run `--propose-fixes` on the format-fail subset.** "
        f"{n_format} rows fail format. Most are mechanical marker "
        "additions — `apply_corrections.py --auto-accept-format` will "
        "auto-apply low-risk fixes.",
        "",
        "3. **Per-instance review for math errors.** "
        f"{n_math} rows have math errors. Each needs a human to verify "
        "the proposed fix. Use `apply_corrections.py "
        "--filter-gate math_correct`.",
        "",
        "4. **Per-instance review for coherence failures** "
        f"({n_coherence} rows). Categorize by failure_mode; "
        "vendor-fabrication failures often need a question rewrite, "
        "physical-absurdity failures often need a number adjustment.",
        "",
        "5. **Relabel down for level-fit failures** "
        f"({n_level} rows). Default disposition is L_claimed → "
        "L_actual. `apply_corrections.py --filter-gate level_fit` walks "
        "them.",
        "",
        f"6. **Bulk-fix placeholder titles** ({n_placeholder} rows) "
        "via Phase 7's title-only --propose-fixes pass.",
        "",
        "Once the corpus is clean, Phase 6 lifts the format gate into "
        "`vault check --strict`'s structural tier and tightens the "
        "LinkML schema with pattern constraints — making the new state "
        "of cleanliness load-bearing.",
    ]
    return "\n".join(out) + "\n"


# ─── main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="01_audit.json from audit_corpus_batched")
    ap.add_argument("--output", type=Path, default=None,
                    help="output markdown (default: docs/AUDIT_FINDINGS_<date>.md)")
    ap.add_argument("--qid-limit", type=int, default=25,
                    help="max qids per priority list (default 25)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    # Resolve so relative_to(REPO_ROOT) works.
    args.input = args.input.resolve()

    audit = json.loads(args.input.read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    if not rows:
        print("no rows in input", file=sys.stderr)
        return 1

    print(f"loaded {len(rows)} audit rows from {args.input}")

    print("loading track index from corpus ...")
    track_index = load_track_index()
    print(f"  {len(track_index)} qid → track mappings")

    today = date.today().isoformat()
    if args.output:
        outpath = args.output
    else:
        outpath = (REPO_ROOT / "interviews" / "vault-cli" / "docs"
                    / f"AUDIT_FINDINGS_{today}.md")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        f"# Corpus audit findings — {today}",
        "",
        f"**Source:** `{args.input.relative_to(REPO_ROOT)}`",
        f"**Generated:** {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"**Audit model:** `{audit.get('model', '?')}`",
        "**Triggered by:** CORPUS_HARDENING_PLAN.md Phase 4 finalization",
        "",
        "---",
        "",
    ]
    parts.append(build_executive_summary(rows, track_index))
    parts.append("---\n")
    parts.append(build_per_track_matrix(rows, track_index))
    parts.append("---\n")
    parts.append(build_failure_modes(rows))
    parts.append("---\n")
    parts.append(build_priority_lists(rows, track_index,
                                       qid_limit=args.qid_limit))
    parts.append("---\n")
    parts.append(build_format_disagreements(rows))
    parts.append("---\n")
    parts.append(build_recommendations(rows))

    outpath.write_text("\n".join(parts), encoding="utf-8")
    try:
        display = outpath.resolve().relative_to(REPO_ROOT)
    except ValueError:
        display = outpath
    print(f"\nwrote {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
