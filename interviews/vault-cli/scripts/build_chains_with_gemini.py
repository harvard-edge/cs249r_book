#!/usr/bin/env python3
"""Build pedagogical chains within (track, topic) buckets via Gemini CLI.

For each bucket of published questions, prompts gemini-3.1-pro-preview to
identify natural chains (groups of 2-6 questions progressing through Bloom
levels, where one builds on another). Output is validated against the
chain schema and written to a staging file for human review before
replacing chains.json.

Design decisions:
  - Sidecar architecture: chains.json is the authoritative registry.
    This script writes a *new* staging chains.json — never edits YAMLs.
  - Adaptive batching: packs multiple small buckets per call to use
    Gemini's 1M context efficiently without maxing it (target ~80K
    input tokens per call). Aim: full corpus in ≤90 calls (250/day cap).
  - Validation: every chain is checked structurally — all member ids
    exist in input, level non-decreasing, 2 ≤ size ≤ 6, single-topic.

Usage:
    python3 build_chains_with_gemini.py --dry-run        # preview batching plan
    python3 build_chains_with_gemini.py --bucket cloud:kv-cache  # one bucket
    python3 build_chains_with_gemini.py --all            # full corpus
    python3 build_chains_with_gemini.py --output proposed_chains.json --all

Modes:
    --mode strict (default): Δ ∈ {1, 2} between consecutive members. This is
        the cleanest pedagogical shape and what we want for primary chains.
    --mode lenient: Δ ∈ {1, 2, 3}. Used for second-pass coverage on buckets
        the strict pass missed; resulting chains are tagged tier=secondary.
        Earlier revisions of lenient mode also allowed Δ=0 for
        "shared scenario, different angle" pairs; that constraint did not
        bind in practice (audit found 54/55 Δ=0 chains had no shared
        scenario), so Δ=0 was removed 2026-05-02.

Bucket scoping:
    --buckets-from <chain-coverage.json>: limit the run to the
        ``uncovered_buckets`` list in a coverage report (output of
        diagnose_chain_coverage.py). Use with --mode lenient for the
        Phase 1.4 second-pass sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parents[2] / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
# AI-pipeline intermediate artifacts live under _pipeline/ (gitignored).
# See interviews/CLAUDE.md for the convention.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT_OUTPUT = PIPELINE_DIR / "chains.proposed.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
TOKENS_PER_CHAR = 0.25
MAX_INPUT_CHARS_PER_CALL = 320_000   # ~80K tokens — safely under 1M
MAX_QUESTIONS_PER_CALL = 250         # Gemini quality degrades on huge tasks
SCENARIO_CHAR_BUDGET = 280           # truncate per question for prompt budget
LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


def load_corpus() -> dict[str, dict]:
    """Load all published question YAMLs."""
    corpus = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
            if d.get("status") not in ("published", None):
                continue
            corpus[d["id"]] = d
        except Exception:
            continue
    return corpus


def bucket_corpus(corpus: dict[str, dict]) -> dict[tuple[str, str], list[str]]:
    """(track, topic) -> sorted list of qids."""
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, d in corpus.items():
        by_bucket[(d.get("track"), d.get("topic"))].append(qid)
    for k in by_bucket:
        by_bucket[k].sort()
    return dict(by_bucket)


def question_payload(corpus: dict[str, dict], qid: str) -> dict:
    """Compact JSON payload for one question — input to Gemini."""
    d = corpus[qid]
    scenario = (d.get("scenario") or "")[:SCENARIO_CHAR_BUDGET]
    return {
        "id": qid,
        "level": d.get("level"),
        "title": d.get("title", ""),
        "question": d.get("question") or "",
        "scenario": scenario,
        "competency_area": d.get("competency_area"),
    }


def estimate_chars(buckets: list[tuple[tuple[str, str], list[str]]],
                   corpus: dict[str, dict]) -> int:
    """Roughly estimate the prompt size of these buckets."""
    n = 0
    for _, qids in buckets:
        for qid in qids:
            d = corpus[qid]
            n += len(d.get("title", "")) + min(len(d.get("scenario","")), SCENARIO_CHAR_BUDGET) + len(d.get("question","") or "") + 40
    return n


def plan_batches(buckets: dict[tuple[str, str], list[str]],
                 corpus: dict[str, dict]) -> list[list[tuple[tuple[str, str], list[str]]]]:
    """Pack buckets into batches under MAX_INPUT_CHARS_PER_CALL each."""
    items = sorted(buckets.items(), key=lambda x: -len(x[1]))  # big first
    batches: list[list[tuple[tuple[str, str], list[str]]]] = []
    cur: list[tuple[tuple[str, str], list[str]]] = []
    cur_chars = 0
    cur_count = 0
    for k, qids in items:
        item_chars = estimate_chars([(k, qids)], corpus)
        if (cur_chars + item_chars > MAX_INPUT_CHARS_PER_CALL
            or cur_count + len(qids) > MAX_QUESTIONS_PER_CALL) and cur:
            batches.append(cur)
            cur = []
            cur_chars = 0
            cur_count = 0
        cur.append((k, qids))
        cur_chars += item_chars
        cur_count += len(qids)
    if cur:
        batches.append(cur)
    return batches


STRICT_PROMPT_TEMPLATE = """You are an expert ML systems educator helping curate pedagogical chains
of interview questions. A "chain" is a sequence of 2-6 questions within a
SINGLE topic that progress through Bloom levels (L1 → L2 → ... up to L6+),
where each question naturally builds on its predecessor — same scenario or
concept, increasing in cognitive demand.

You will be given several BUCKETS, each containing all published questions
for one (track, topic) pair. For each bucket, identify the BEST natural
chains. A bucket may yield 0 chains (no good progressions), 1 chain (one
arc through the topic), or several chains (multiple distinct arcs).

LEVEL PROGRESSION RULES (HARD):
  - Each consecutive pair of members MUST satisfy: cand_level - prev_level ∈ {{1, 2}}
  - PREFER strict +1 progression (L1→L2→L3→L4→L5→L6+) — this is the cleanest
    pedagogical shape and should account for the majority of chains
  - ACCEPT a +2 jump (e.g., L1→L3 or L3→L5) ONLY when no Δ=1 candidate is
    available within the bucket and the conceptual progression is genuinely
    natural — i.e., the harder question still meaningfully builds on the
    easier one even with one Bloom step skipped
  - REJECT Δ=0 (same-level pairs) — same Bloom level isn't a progression
  - REJECT Δ ≥ 3 (e.g., L1→L4) and any backward step — too large to be a
    coherent single-step pedagogical move

OTHER CONSTRAINTS:
  - 2 ≤ chain size ≤ 6 members
  - All members from the SAME (track, topic) bucket
  - A question MAY appear in UP TO 2 different chains if and only if:
      (a) The question is L1 or L2 (a foundational anchor)
      (b) The two chains diverge into genuinely distinct sub-progressions
          AFTER this anchor — not the same arc viewed twice
      (c) Each chain is individually coherent and pedagogically valuable
    Default to 1 chain per question; multi-membership is the exception.
  - Prefer chains where Q[i+1] genuinely builds on Q[i] (shared scenario,
    sequential reasoning) over loosely related same-topic questions
  - Don't force chains — if questions are unrelated, return 0 chains for
    that bucket. Quality over coverage.

GAP DETECTION (free signal — emit alongside chains):
For each bucket, also identify "missing-rung" gaps: pedagogical arcs that
WOULD form a clean strict +1 chain if the bucket had a question at a
specific Bloom level it currently lacks. Example: bucket has L1, L3, L5
on the same scenario thread → propose a missing-L2 and missing-L4
question that would link them. These gaps drive future authoring; we
don't act on them in this pass.

Return STRICT JSON in this exact shape, no prose:
{{
  "buckets": [
    {{
      "track": "<track>",
      "topic": "<topic>",
      "chains": [
        {{
          "questions": ["<qid1>", "<qid2>", ...],
          "rationale": "<one sentence — what does this chain teach?>"
        }}
      ],
      "gaps": [
        {{
          "missing_level": "L<N>",
          "between": ["<qid_lower>", "<qid_higher>"],
          "rationale": "<what concept the missing question should cover to bridge these two>"
        }}
      ]
    }}
  ]
}}

INPUT (buckets to process):
{buckets_json}
"""


# Lenient prompt for the second-pass coverage sweep (Phase 1.4 of
# CHAIN_ROADMAP.md). Same structural envelope as STRICT, but with relaxed
# Δ rules so we can wring at least one chain out of buckets the strict pass
# rejected. Chains produced under this prompt are tagged tier=secondary.
LENIENT_PROMPT_TEMPLATE = """You are an expert ML systems educator helping curate pedagogical chains
of interview questions. A "chain" is a sequence of 2-6 questions within a
SINGLE topic that progress through Bloom levels (L1 → L2 → ... up to L6+),
where each question naturally builds on its predecessor — same scenario or
concept, increasing in cognitive demand.

You will be given several BUCKETS, each containing all published questions
for one (track, topic) pair. These are buckets a stricter first pass was
unable to chain — your job is to find at least one coherent progression
per bucket if any pedagogical clustering exists at all. Only return zero
chains for a bucket when its questions are genuinely unrelated even on
the loosest reading.

LEVEL PROGRESSION RULES (LENIENT MODE):
  - Each consecutive pair of members satisfies: cand_level - prev_level ∈ {{1, 2, 3}}
  - STRONGLY PREFER strict +1 progression where it exists
  - +2 jumps acceptable when no Δ=1 candidate is available
  - +3 jumps allowed only when no smaller intermediate exists in the bucket
  - REJECT Δ=0 (same-level pair). Earlier versions of this prompt allowed
    Δ=0 for "shared scenario / different angle" pairs, but in practice
    that constraint did not bind — Gemini routinely produced Δ=0 chains
    that were just two unrelated same-level same-topic questions.
    If two same-level questions share a scenario thread, model them as
    siblings (separate registry entries pointing at the anchor), not
    as a chain.
  - REJECT any backward step (Δ < 0)

OTHER CONSTRAINTS:
  - 2 ≤ chain size ≤ 6 members
  - All members from the SAME (track, topic) bucket
  - A question MAY appear in UP TO 2 different chains if and only if:
      (a) The question is L1 or L2 (a foundational anchor)
      (b) The two chains diverge into genuinely distinct sub-progressions
          AFTER this anchor — not the same arc viewed twice
      (c) Each chain is individually coherent and pedagogically valuable
    Default to 1 chain per question; multi-membership is the exception.
  - Prefer chains where Q[i+1] genuinely builds on Q[i] (shared scenario,
    sequential reasoning) over loosely related same-topic questions
  - Quality still matters — but err on the side of producing at least one
    chain per bucket rather than rejecting the bucket entirely

GAP DETECTION (free signal — emit alongside chains):
For each bucket, also identify "missing-rung" gaps: pedagogical arcs that
WOULD form a clean strict +1 chain if the bucket had a question at a
specific Bloom level it currently lacks. Example: bucket has L1, L3, L5
on the same scenario thread → propose a missing-L2 and missing-L4
question that would link them. These gaps drive future authoring; we
don't act on them in this pass.

Return STRICT JSON in this exact shape, no prose:
{{
  "buckets": [
    {{
      "track": "<track>",
      "topic": "<topic>",
      "chains": [
        {{
          "questions": ["<qid1>", "<qid2>", ...],
          "rationale": "<one sentence — what does this chain teach?>"
        }}
      ],
      "gaps": [
        {{
          "missing_level": "L<N>",
          "between": ["<qid_lower>", "<qid_higher>"],
          "rationale": "<what concept the missing question should cover to bridge these two>"
        }}
      ]
    }}
  ]
}}

INPUT (buckets to process):
{buckets_json}
"""


# Map mode -> prompt template + accepted Δ set. Single source of truth so
# build_prompt and validate_chain stay in lockstep when modes are added.
MODE_CONFIG = {
    "strict": {
        "prompt_template": STRICT_PROMPT_TEMPLATE,
        "allowed_deltas": frozenset({1, 2}),
    },
    "lenient": {
        "prompt_template": LENIENT_PROMPT_TEMPLATE,
        "allowed_deltas": frozenset({1, 2, 3}),
    },
}

# Backwards-compatible alias for any external readers — strict was the
# original (and only) prompt before Phase 1.2.
PROMPT_TEMPLATE = STRICT_PROMPT_TEMPLATE


def build_prompt(batch: list[tuple[tuple[str, str], list[str]]],
                 corpus: dict[str, dict],
                 mode: str = "strict") -> str:
    payload = []
    for (track, topic), qids in batch:
        payload.append({
            "track": track,
            "topic": topic,
            "questions": [question_payload(corpus, qid) for qid in qids],
        })
    template = MODE_CONFIG[mode]["prompt_template"]
    return template.format(buckets_json=json.dumps(payload, indent=2))


def call_gemini(prompt: str, model: str = GEMINI_MODEL, timeout: int = 600) -> dict | None:
    """Run gemini -p '...' --yolo and parse JSON response.

    Gemini CLI sometimes exits non-zero even when stdout contains a valid
    JSON response (e.g., YOLO-mode info messages, transient 429s that the
    CLI internally retries past). We try to parse stdout regardless and
    only treat unparsable output as failure.
    """
    try:
        result = subprocess.run(
            ["gemini", "-m", model, "-p", prompt, "--yolo"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    out = (result.stdout or "").strip()
    # Strip code fences if present
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:].lstrip()
    # Find first { ... } block
    i = out.find("{")
    j = out.rfind("}")
    if i == -1 or j == -1:
        # No JSON in stdout — surface stderr so the operator sees what happened
        if result.returncode != 0:
            print(f"  gemini exit {result.returncode}, no JSON: {(result.stderr or '')[:200]}",
                  file=sys.stderr)
        return None
    try:
        return json.loads(out[i:j+1])
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}", file=sys.stderr)
        return None


def validate_chain(
    chain: dict,
    bucket_qids: set[str],
    corpus: dict[str, dict],
    mode: str = "strict",
) -> tuple[bool, str]:
    """Structural validation of a Gemini-proposed chain.

    Δ-rule depends on mode:
      strict  → Δ ∈ {1, 2}     (clean +1 progression, +2 if no intermediate)
      lenient → Δ ∈ {1, 2, 3}  (Δ=3 last-resort when no smaller rung exists)
    Both modes reject backward steps, Δ=0 (same-level edges), and require
    the chain to be single-topic.
    """
    if mode not in MODE_CONFIG:
        return False, f"unknown mode {mode!r}"
    allowed_deltas = MODE_CONFIG[mode]["allowed_deltas"]

    qs = chain.get("questions", [])
    if len(qs) < 2 or len(qs) > 6:
        return False, f"size {len(qs)} out of [2, 6]"
    seen = set()
    levels = []
    topics = set()
    for qid in qs:
        if qid not in bucket_qids:
            return False, f"qid {qid} not in bucket"
        if qid in seen:
            return False, f"qid {qid} duplicated"
        seen.add(qid)
        d = corpus[qid]
        levels.append(LEVEL_RANK.get(d.get("level"), 0))
        topics.add(d.get("topic"))
    deltas = [levels[i+1] - levels[i] for i in range(len(levels)-1)]
    bad_deltas = [d for d in deltas if d not in allowed_deltas]
    if bad_deltas:
        return False, (
            f"levels {levels} have Δ={deltas} "
            f"(need each Δ ∈ {sorted(allowed_deltas)} under mode={mode!r})"
        )
    if len(topics) != 1:
        return False, f"multi-topic: {topics}"
    return True, ""


def process_batch(batch: list[tuple[tuple[str, str], list[str]]],
                  corpus: dict[str, dict],
                  call_idx: int,
                  mode: str = "strict") -> tuple[list[dict], list[dict]]:
    """Call Gemini on this batch. Returns (validated_chains, raw_gaps).

    In lenient mode, accepted chains carry tier="secondary"; strict-mode
    chains are emitted without a tier field (primary tagging is backfilled
    in the merge step — see merge_chain_passes.py / Phase 1.5).
    """
    prompt = build_prompt(batch, corpus, mode=mode)
    n_questions = sum(len(qids) for _, qids in batch)
    print(f"  [call {call_idx}] {len(batch)} buckets, {n_questions} questions, "
          f"{len(prompt)//1000}K char prompt (mode={mode})")
    response = call_gemini(prompt)
    if response is None:
        print(f"  [call {call_idx}] no response")
        return [], []

    out_chains: list[dict] = []
    out_gaps: list[dict] = []
    chain_seq = 0
    chain_id_suffix = "-secondary" if mode == "lenient" else ""
    for bucket_resp in response.get("buckets", []):
        track = bucket_resp.get("track")
        topic = bucket_resp.get("topic")
        bucket_qids = set()
        for (t, p), qids in batch:
            if t == track and p == topic:
                bucket_qids = set(qids)
                break
        if not bucket_qids:
            print(f"  [call {call_idx}] response references unknown bucket ({track},{topic})")
            continue
        for ch in bucket_resp.get("chains", []):
            ok, why = validate_chain(ch, bucket_qids, corpus, mode=mode)
            if not ok:
                print(f"  [call {call_idx}] dropped invalid chain in {track}/{topic}: {why}")
                continue
            chain_seq += 1
            chain_id = f"{track}-chain-auto{chain_id_suffix}-{call_idx:03d}-{chain_seq:02d}"
            entry = {
                "chain_id": chain_id,
                "track": track,
                "topic": topic,
                "competency_area": corpus[ch["questions"][0]].get("competency_area"),
                "levels": [corpus[qid].get("level") for qid in ch["questions"]],
                "questions": [
                    {
                        "level": corpus[qid].get("level"),
                        "id": qid,
                        "title": corpus[qid].get("title", ""),
                        "bloom": corpus[qid].get("bloom_level"),
                    }
                    for qid in ch["questions"]
                ],
                "rationale": ch.get("rationale", ""),
                "_origin": "gemini-3.1-pro-preview",
            }
            if mode == "lenient":
                entry["tier"] = "secondary"
            out_chains.append(entry)
        # Capture gap recommendations as-is (not validated structurally —
        # they describe questions that DON'T exist yet). We store them for
        # a follow-up authoring pass.
        for gap in bucket_resp.get("gaps", []) or []:
            gap_record = {
                "track": track,
                "topic": topic,
                "missing_level": gap.get("missing_level"),
                "between": gap.get("between") or [],
                "rationale": gap.get("rationale", ""),
                "_origin": "gemini-3.1-pro-preview",
                "_source_call": call_idx,
            }
            out_gaps.append(gap_record)
    print(f"  [call {call_idx}] accepted {len(out_chains)} chain(s), "
          f"{len(out_gaps)} gap(s)")
    return out_chains, out_gaps


def load_buckets_filter(path: Path) -> list[tuple[str, str]]:
    """Read uncovered_buckets from a chain-coverage.json report.

    Output of diagnose_chain_coverage.py — we use the ``uncovered_buckets``
    array (≥3 questions, 0 chains) as the input set for Phase 1.4.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("uncovered_buckets") or []
    return [(b["track"], b["topic"]) for b in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Process every bucket")
    ap.add_argument("--bucket", help="Process one bucket: <track>:<topic>")
    ap.add_argument(
        "--buckets-from",
        type=Path,
        help="Restrict to uncovered_buckets in a chain-coverage.json report "
             "(output of diagnose_chain_coverage.py). Pair with --mode lenient.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Show plan, don't call Gemini")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument(
        "--mode",
        choices=sorted(MODE_CONFIG.keys()),
        default="strict",
        help="strict (default): Δ ∈ {1,2}; lenient: Δ ∈ {1,2,3}, "
             "tags chains tier=secondary",
    )
    ap.add_argument("--max-calls", type=int, default=200,
                    help="Daily cap (Gemini Pro is 250 calls/day; reserve some buffer)")
    args = ap.parse_args()

    corpus = load_corpus()
    buckets = bucket_corpus(corpus)
    print(f"corpus: {len(corpus)} published questions in {len(buckets)} (track, topic) buckets")
    print(f"mode: {args.mode}")

    selectors = [bool(args.all), bool(args.bucket), bool(args.buckets_from)]
    if sum(selectors) > 1:
        ap.error("--all, --bucket, and --buckets-from are mutually exclusive")
    if not any(selectors):
        ap.error("specify --all, --bucket <track>:<topic>, or --buckets-from <path>")

    if args.bucket:
        track, topic = args.bucket.split(":", 1)
        if (track, topic) not in buckets:
            print(f"unknown bucket: {args.bucket}")
            return 1
        buckets = {(track, topic): buckets[(track, topic)]}
    elif args.buckets_from:
        wanted = load_buckets_filter(args.buckets_from)
        missing = [b for b in wanted if b not in buckets]
        if missing:
            print(f"WARNING: {len(missing)} buckets in coverage report not found in corpus "
                  f"(skipping): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        buckets = {b: buckets[b] for b in wanted if b in buckets}
        print(f"buckets-from filter: {len(buckets)} buckets selected from "
              f"{args.buckets_from.name}")

    batches = plan_batches(buckets, corpus)
    sizes = [sum(len(qids) for _, qids in b) for b in batches]
    print(f"\nbatching plan: {len(batches)} calls")
    print(f"  questions/call — min {min(sizes)}, mean {sum(sizes)//len(sizes)}, max {max(sizes)}")
    print(f"  daily cap: {args.max_calls}; budget OK: {len(batches) <= args.max_calls}")
    if args.dry_run:
        return 0

    if len(batches) > args.max_calls:
        print(f"\nWARNING: {len(batches)} batches exceeds max-calls {args.max_calls}")
        return 1

    all_chains: list[dict] = []
    all_gaps: list[dict] = []
    gaps_path = Path(args.output).with_name(
        Path(args.output).stem.replace("chains.proposed", "gaps.proposed") + ".json"
    )
    inter_call_delay_s = 8  # backoff: avoid Gemini-side 429 from rapid-fire calls
    for i, batch in enumerate(batches, start=1):
        if i > 1:
            time.sleep(inter_call_delay_s)
        chains, gaps = process_batch(batch, corpus, i, mode=args.mode)
        all_chains.extend(chains)
        all_gaps.extend(gaps)
        Path(args.output).write_text(json.dumps(all_chains, indent=2) + "\n")
        gaps_path.write_text(json.dumps(all_gaps, indent=2) + "\n")

    print(f"\nDONE: {len(all_chains)} chains accepted across {len(batches)} calls; "
          f"{len(all_gaps)} corpus gaps identified for future authoring")
    print(f"output: {args.output}")
    print(f"gaps:   {gaps_path}")
    print("review the staging file before replacing interviews/vault/chains.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
