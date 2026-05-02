#!/usr/bin/env python3
"""Independent audit of the Phase 1-3 chain work via gemini-3.1-pro-preview.

Designed to be a complementary check on the output of the chain-build,
tier-classification, and gap-detection pipeline — running an
independent Gemini pass over the artifacts that human review would
otherwise have to spot-check by eye.

Total call budget: ~50-60 calls (well under the 250/day Pro cap).
Per-call target: ~80K input tokens (roughly 320K chars), the sweet
spot where Gemini's attention stays sharp without burning context on
ground that won't be used.

Categories audited:

  1. drafts        — All 4 Phase 3 promoted drafts (independent quality
                     gate vs the validate_drafts.py judges; ~2 calls).
  2. secondary     — 100-chain sample of tier=secondary chains
                     (pedagogical coherence; ~10 calls).
  3. delta_zero    — All Δ=0 chains (highest-risk lenient additions:
                     verifies "shared scenario" claim; ~6 calls).
  4. primary       — 100-chain sample of tier=primary chains
                     (regression check on strict-pass quality; ~10 calls).
  5. gaps          — 50-gap sample with the two between-questions in full
                     (real bridge vs hallucination; ~10 calls).
  6. synthesis     — 1 wrap-up call that reads category outputs and
                     emits AUDIT_REPORT.md.

  (NOTE: an originally-planned tier_compare category was dropped —
  the lenient sweep was scoped to uncovered buckets, so 0 buckets
  carry both primary and secondary chains. Per-tier quality is
  inferred from categories 2 and 4 by the synthesis call.)

Outputs:
  interviews/vault/_pipeline/runs/<UTC-timestamp>/
    config.json            — what was run, with what samples
    01_drafts.json         — per-call traces (prompt, response, parsed verdict)
    02_secondary.json
    03_delta_zero.json
    04_primary.json
    05_gaps.json
    06_tier_compare.json
    07_synthesis.json
    AUDIT_REPORT.md        — human-readable rollup

Modes:
  --dry-run                # plan + show batching, don't call Gemini
  --only <category>        # run a single category (debugging)
  --skip <category,...>    # skip listed categories (debugging)

Findings only — this script never edits chains.json or any question
YAML. Issues are surfaced for human review.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"
# AI-pipeline staging artifacts live under _pipeline/ (gitignored).
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
GAPS_PRIMARY_PATH = PIPELINE_DIR / "gaps.proposed.json"
GAPS_LENIENT_PATH = PIPELINE_DIR / "gaps.proposed.lenient.json"
AUDIT_RUNS = PIPELINE_DIR / "runs"
SCORECARD = PIPELINE_DIR / "draft-validation-scorecard.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
INTER_CALL_DELAY_S = 4
MAX_PROMPT_CHARS = 320_000   # ~80K input tokens, attention-sweet spot
SCENARIO_CHAR_BUDGET = 350   # truncate per-question for prompt budget
LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}

CATEGORIES = ["drafts", "secondary", "delta_zero", "primary", "gaps",
              "synthesis"]

# RNG seed for reproducible sampling — flip via --seed for a different draw.
RNG_SEED = 42


# ─── corpus + chain helpers ───────────────────────────────────────────────


def load_corpus() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        try:
            with path.open(encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("id"):
            out[d["id"]] = d
    return out


def load_chains() -> list[dict]:
    return json.loads(CHAINS_PATH.read_text(encoding="utf-8"))


def question_payload(q: dict, *, terse: bool = False) -> dict[str, Any]:
    """Compact view of a question for prompt context. terse=True for
    cases where we have a lot of questions to fit in one call."""
    out = {
        "id": q.get("id"),
        "level": q.get("level"),
        "title": q.get("title"),
        "scenario": (q.get("scenario") or "")[:SCENARIO_CHAR_BUDGET],
        "question": q.get("question"),
    }
    if not terse:
        details = q.get("details") or {}
        out["realistic_solution"] = details.get("realistic_solution")
    return out


def chain_payload(c: dict, corpus: dict[str, dict], *, terse: bool = False) -> dict[str, Any]:
    qids = [m["id"] for m in c.get("questions", []) if m.get("id")]
    return {
        "chain_id": c["chain_id"],
        "track": c["track"],
        "topic": c["topic"],
        "tier": c.get("tier", "primary"),
        "rationale": c.get("rationale"),
        "members": [question_payload(corpus[q], terse=terse)
                    for q in qids if q in corpus],
    }


# ─── Gemini call ──────────────────────────────────────────────────────────


def call_gemini(prompt: str, *, timeout: int = 600) -> dict | None:
    try:
        result = subprocess.run(
            ["gemini", "-m", GEMINI_MODEL, "-p", prompt, "--yolo"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    out = (result.stdout or "").strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:].lstrip()
    i = out.find("{")
    j = out.rfind("}")
    if i == -1 or j == -1:
        if result.returncode != 0:
            print(f"  gemini exit {result.returncode}: {(result.stderr or '')[:200]}",
                  file=sys.stderr)
        return None
    try:
        return json.loads(out[i:j+1])
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}", file=sys.stderr)
        return None


# ─── batching ─────────────────────────────────────────────────────────────


def batch_chains(items: list[dict], corpus: dict[str, dict],
                 max_chars: int = MAX_PROMPT_CHARS,
                 wrapper_chars: int = 4_000) -> list[list[dict]]:
    """Pack ~80K-token batches of full chain payloads."""
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_chars = wrapper_chars
    for c in items:
        payload_chars = len(json.dumps(chain_payload(c, corpus)))
        if cur and cur_chars + payload_chars > max_chars:
            batches.append(cur)
            cur = []
            cur_chars = wrapper_chars
        cur.append(c)
        cur_chars += payload_chars
    if cur:
        batches.append(cur)
    return batches


# ─── category 1: drafts audit ─────────────────────────────────────────────


def audit_drafts(corpus: dict[str, dict], outdir: Path) -> dict:
    if not SCORECARD.exists():
        return {"skipped": True, "reason": "no scorecard"}
    sc = json.loads(SCORECARD.read_text(encoding="utf-8"))
    drafts_qids = [r["draft_id"] for r in sc.get("rows", [])]
    drafts = [corpus[q] for q in drafts_qids if q in corpus]

    # Pack 2 drafts per call so each gets a substantial context window
    # for the same-bucket exemplars and the full draft body.
    batches = [drafts[i:i+2] for i in range(0, len(drafts), 2)]
    rows: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        payload: list[dict] = []
        for d in batch:
            same_bucket = [
                question_payload(q) for q in corpus.values()
                if q.get("track") == d.get("track")
                and q.get("topic") == d.get("topic")
                and q.get("level") == d.get("level")
                and q.get("id") != d.get("id")
                and q.get("status") == "published"
            ][:5]
            payload.append({
                "draft": question_payload(d),
                "draft_id": d.get("id"),
                "same_level_neighbours": same_bucket,
                "tags": d.get("tags") or [],
                "authors": d.get("authors"),
                "human_reviewed": d.get("human_reviewed"),
            })
        prompt = f"""You are an ML systems interview-question reviewer running an
INDEPENDENT QUALITY CHECK on LLM-authored draft questions. They have already
passed: Pydantic schema, BAAI/bge-small-en-v1.5 cosine vs in-bucket
neighbours (<0.92), and three Gemini-judge gates (level_fit, coherence,
bridge). Your job is to surface failure modes those gates routinely miss:

  - vendor-name fabrication (made-up hardware / benchmarks / model names)
  - subtle cognitive-load drift (the level field claims L4 but the question
    is actually L2-shaped)
  - factually wrong but internally-consistent answers
  - low-quality scenarios that read as ML cosplay rather than a real situation

For each candidate, return STRICT JSON (no prose, no fences) of the shape:

{{
  "drafts": [
    {{
      "draft_id": "<id>",
      "verdict": "accept" | "edit" | "reject",
      "fabrication_check": "yes" | "no" | "unclear",  // any made-up vendor/benchmark/model?
      "level_match": "yes" | "no" | "unclear",         // does cognitive load match the level field?
      "answer_correctness": "yes" | "no" | "unclear",  // is realistic_solution correct?
      "scenario_realism": "yes" | "no" | "unclear",
      "rationale": "<one or two sentences with the SPECIFIC issue if not accept>"
    }}
  ]
}}

INPUT:
{json.dumps(payload, indent=2)}
"""
        print(f"  [drafts] call {i}/{len(batches)} — {len(prompt)//1000}K char prompt")
        resp = call_gemini(prompt)
        rows.append({"call_idx": i, "draft_ids": [d["id"] for d in batch],
                     "prompt_chars": len(prompt), "response": resp})
        with (outdir / "01_drafts.json").open("w") as f:
            json.dump(rows, f, indent=2)
        if i < len(batches):
            time.sleep(INTER_CALL_DELAY_S)
    return {"calls": len(rows), "rows": rows}


# ─── category 2/3/4: chain sample audits (shared shape) ───────────────────


def audit_chain_sample(
    chains: list[dict],
    corpus: dict[str, dict],
    *,
    label: str,
    outname: str,
    outdir: Path,
    instructions: str,
    extra_fields: str = "",
) -> dict:
    if not chains:
        return {"skipped": True, "reason": f"{label}: no chains"}
    batches = batch_chains(chains, corpus)
    rows: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        payload = [chain_payload(c, corpus) for c in batch]
        prompt = f"""You are an ML systems interview question reviewer auditing
chains (pedagogical sequences of 2-6 questions through Bloom levels) for
quality. {instructions}

For each chain, return STRICT JSON of the shape:

{{
  "chains": [
    {{
      "chain_id": "<id>",
      "verdict": "good" | "weak" | "bad",
      "progression": "yes" | "no" | "unclear",  // is each step a real progression?
      "topic_unity": "yes" | "no" | "unclear",  // does the chain stay on one topic?
      "duplicate_pair": "yes" | "no" | "unclear",  // any pair too similar to be a real chain step?
      {extra_fields}
      "rationale": "<one sentence pointing to the SPECIFIC issue if not good>"
    }}
  ]
}}

Return ONLY the JSON, no prose, no fences.

INPUT:
{json.dumps(payload, indent=2)}
"""
        print(f"  [{label}] call {i}/{len(batches)} — {len(payload)} chains, "
              f"{len(prompt)//1000}K char prompt")
        resp = call_gemini(prompt)
        rows.append({"call_idx": i,
                     "chain_ids": [c["chain_id"] for c in batch],
                     "prompt_chars": len(prompt), "response": resp})
        with (outdir / outname).open("w") as f:
            json.dump(rows, f, indent=2)
        if i < len(batches):
            time.sleep(INTER_CALL_DELAY_S)
    return {"calls": len(rows), "rows": rows}


# ─── category 5: gaps audit ───────────────────────────────────────────────


def audit_gaps(corpus: dict[str, dict], outdir: Path,
                limit: int, rng: random.Random) -> dict:
    gaps_all = []
    for p in [GAPS_PRIMARY_PATH, GAPS_LENIENT_PATH]:
        if p.exists():
            gaps_all.extend(json.loads(p.read_text(encoding="utf-8")))
    sampled = rng.sample(gaps_all, min(limit, len(gaps_all)))

    # Pack ~5 gaps per call (each gap brings 2 full anchor questions).
    batch_size = 5
    batches = [sampled[i:i+batch_size] for i in range(0, len(sampled), batch_size)]
    rows: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        payload = []
        for g in batch:
            anchors = [corpus.get(q) for q in (g.get("between") or [])]
            if any(a is None for a in anchors):
                continue
            payload.append({
                "track": g.get("track"),
                "topic": g.get("topic"),
                "missing_level": g.get("missing_level"),
                "between_anchors": [question_payload(a) for a in anchors],
                "rationale": g.get("rationale"),
            })
        if not payload:
            continue
        prompt = f"""You are reviewing GAP DETECTION output: each entry claims
that a chain bucket is missing a question at a specific Bloom level
between two existing anchor questions. Judge whether the gap is REAL
(the two anchors share a scenario thread and a true bridge would chain
them) or HALLUCINATED (the anchors are too unrelated for a bridge to
make sense, or the missing-level is wrong).

For each gap, return STRICT JSON:

{{
  "gaps": [
    {{
      "track": "<>",
      "topic": "<>",
      "missing_level": "<>",
      "verdict": "real" | "hallucinated" | "unclear",
      "anchors_share_scenario": "yes" | "no" | "unclear",
      "level_makes_sense": "yes" | "no",
      "rationale": "<one sentence>"
    }}
  ]
}}

Return ONLY JSON, no prose.

INPUT:
{json.dumps(payload, indent=2)}
"""
        print(f"  [gaps] call {i}/{len(batches)} — {len(payload)} gaps, "
              f"{len(prompt)//1000}K char prompt")
        resp = call_gemini(prompt)
        rows.append({"call_idx": i, "gap_count": len(payload),
                     "prompt_chars": len(prompt), "response": resp})
        with (outdir / "05_gaps.json").open("w") as f:
            json.dump(rows, f, indent=2)
        if i < len(batches):
            time.sleep(INTER_CALL_DELAY_S)
    return {"calls": len(rows), "rows": rows}


# ─── category 6: tier comparison ──────────────────────────────────────────


def audit_tier_compare(chains: list[dict], corpus: dict[str, dict],
                        outdir: Path, limit: int, rng: random.Random) -> dict:
    """Find buckets that have BOTH primary and secondary chains, send one
    pair per call to Gemini for side-by-side judgement."""
    by_bucket: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"primary": [], "secondary": []})
    for c in chains:
        tier = c.get("tier", "primary")
        if tier not in ("primary", "secondary"):
            continue
        by_bucket[(c["track"], c["topic"])][tier].append(c)

    candidates = [
        (bucket, lists) for bucket, lists in by_bucket.items()
        if lists["primary"] and lists["secondary"]
    ]
    if not candidates:
        return {"skipped": True, "reason": "no buckets with both tiers"}

    sampled = rng.sample(candidates, min(limit, len(candidates)))
    rows: list[dict] = []
    for i, (bucket, lists) in enumerate(sampled, start=1):
        # one primary + one secondary
        p = rng.choice(lists["primary"])
        s = rng.choice(lists["secondary"])
        payload = {
            "bucket": {"track": bucket[0], "topic": bucket[1]},
            "primary_chain": chain_payload(p, corpus),
            "secondary_chain": chain_payload(s, corpus),
        }
        prompt = f"""You are judging the tier-classification of two chains
in the same (track, topic) bucket. The PRIMARY chain came from a strict
Bloom-progression sweep (Δ ∈ {{1,2}}); the SECONDARY chain came from a
lenient second-pass that allowed Δ ∈ {{0,1,2,3}}. Judge whether the
classification is plausible: the primary should look like a cleaner,
more canonical pedagogical sequence than the secondary.

Return STRICT JSON:

{{
  "primary_genuinely_stronger": "yes" | "no" | "unclear",
  "primary_quality":   "good" | "weak" | "bad",
  "secondary_quality": "good" | "weak" | "bad",
  "tier_inversion": "yes" | "no",  // is secondary actually better than primary?
  "rationale": "<one or two sentences>"
}}

Return ONLY JSON.

INPUT:
{json.dumps(payload, indent=2)}
"""
        print(f"  [tier_compare] call {i}/{len(sampled)} — bucket={bucket[0]}/{bucket[1]}")
        resp = call_gemini(prompt)
        rows.append({"call_idx": i, "bucket": list(bucket),
                     "primary_chain_id": p["chain_id"],
                     "secondary_chain_id": s["chain_id"],
                     "prompt_chars": len(prompt), "response": resp})
        with (outdir / "06_tier_compare.json").open("w") as f:
            json.dump(rows, f, indent=2)
        if i < len(sampled):
            time.sleep(INTER_CALL_DELAY_S)
    return {"calls": len(rows), "rows": rows}


# ─── category 7: synthesis ────────────────────────────────────────────────


def synthesise(outdir: Path) -> dict:
    """Single call that reads category outputs and emits AUDIT_REPORT.md."""
    summary = {}
    for fname in sorted(outdir.glob("0?_*.json")):
        if fname.name.startswith("07_"):
            continue
        try:
            data = json.loads(fname.read_text())
        except Exception:
            continue
        # Extract per-call response verdicts compactly so the synthesis
        # call doesn't have to re-read full chain payloads.
        flat = []
        for row in data if isinstance(data, list) else []:
            r = (row.get("response") or {})
            for key in ("drafts", "chains", "gaps"):
                for entry in (r.get(key) or []):
                    flat.append({"category": fname.stem, **entry})
            for key in ("primary_genuinely_stronger", "tier_inversion"):
                if key in r:
                    flat.append({"category": fname.stem, **r})
                    break
        summary[fname.stem] = flat

    summary_chars = sum(len(json.dumps(v)) for v in summary.values())
    prompt = f"""You are writing an AUDIT REPORT for an ML systems interview
question pipeline. The pipeline built 879 chains, 4 LLM-authored drafts,
and detected 407 chain gaps. You have the per-category Gemini judge
results below. Produce STRICT JSON of the shape:

{{
  "summary": "<2-3 sentence overall verdict>",
  "headline_findings": ["<finding 1>", "<finding 2>", ...],   // top 3-5 issues worth a human's attention
  "per_category": {{
    "drafts":       {{"pass_rate": <0..1>, "key_issue": "<...>" }},
    "secondary":    {{"pass_rate": <0..1>, "key_issue": "<...>" }},
    "delta_zero":   {{"pass_rate": <0..1>, "key_issue": "<...>" }},
    "primary":      {{"pass_rate": <0..1>, "key_issue": "<...>" }},
    "gaps":         {{"pass_rate": <0..1>, "key_issue": "<...>" }}
  }},
  "tier_quality_delta": "<does secondary look systematically weaker than primary? one sentence>",
  "recommendations": ["<actionable recommendation 1>", ...]
}}

Return ONLY JSON, no prose, no fences.

CATEGORY RESULTS (already-distilled per-call verdicts; total {summary_chars} chars):
{json.dumps(summary, indent=2)}
"""
    print(f"  [synthesis] 1 call — {len(prompt)//1000}K char prompt")
    resp = call_gemini(prompt)
    out = {"call_idx": 1, "prompt_chars": len(prompt), "response": resp,
           "summary_input": summary}
    with (outdir / "07_synthesis.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


def write_report(outdir: Path) -> Path:
    """Generate AUDIT_REPORT.md from per-category outputs + synthesis."""
    syn_path = outdir / "07_synthesis.json"
    syn = json.loads(syn_path.read_text()) if syn_path.exists() else {}
    s = (syn.get("response") or {})

    lines: list[str] = [
        f"# Vault chain pipeline — independent audit report",
        f"",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"**Auditor:** {GEMINI_MODEL} (independent of the pipeline's own judges)",
        f"**Audit run dir:** `{outdir.relative_to(REPO_ROOT)}`",
        f"",
        "---",
        f"",
        f"## Summary",
        f"",
        s.get("summary", "*(synthesis call failed; see per-category JSON)*"),
        f"",
        f"## Headline findings",
        f"",
    ]
    for f in s.get("headline_findings", []) or []:
        lines.append(f"- {f}")
    if not s.get("headline_findings"):
        lines.append("*(no synthesis findings; see per-category JSON)*")
    lines.extend(["", "## Per-category", ""])
    for cat in ("drafts", "secondary", "delta_zero", "primary", "gaps"):
        cat_data = (s.get("per_category") or {}).get(cat) or {}
        rate = cat_data.get("pass_rate")
        rate_str = f"{rate*100:.0f}%" if isinstance(rate, (int, float)) else "n/a"
        lines.append(f"### {cat}")
        lines.append(f"")
        lines.append(f"- pass rate: **{rate_str}**")
        lines.append(f"- key issue: {cat_data.get('key_issue', '*(none reported)*')}")
        lines.append("")
    if s.get("tier_quality_delta"):
        lines.append(f"### Tier quality delta (primary vs secondary)\n")
        lines.append(s["tier_quality_delta"])
        lines.append("")
    lines.append("## Recommendations\n")
    for r in s.get("recommendations", []) or []:
        lines.append(f"- {r}")
    lines.extend([
        "",
        "---",
        "",
        f"Per-call traces are in `{outdir.relative_to(REPO_ROOT)}/`. "
        "Each `0N_*.json` file contains the prompt-char count, the IDs in "
        "scope, and the raw Gemini response. Use these for ground-truth "
        "follow-up — the synthesis above is one model's compression of "
        "the underlying judgements.",
    ])

    report = outdir.parent / "AUDIT_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ─── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan and batching, don't call Gemini")
    ap.add_argument("--only", choices=CATEGORIES, default=None,
                    help="run a single category (debugging)")
    ap.add_argument("--skip", default="",
                    help="comma-separated categories to skip")
    ap.add_argument("--secondary-sample", type=int, default=100)
    ap.add_argument("--primary-sample", type=int, default=100)
    ap.add_argument("--gap-sample", type=int, default=50)
    ap.add_argument("--bucket-pairs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    skipped = set(c.strip() for c in args.skip.split(",") if c.strip())
    rng = random.Random(args.seed)

    print("loading corpus + chains…")
    corpus = load_corpus()
    chains = load_chains()
    print(f"  corpus={len(corpus)}, chains={len(chains)} "
          f"({sum(1 for c in chains if c.get('tier') == 'primary')} primary / "
          f"{sum(1 for c in chains if c.get('tier') == 'secondary')} secondary)")

    # Δ=0 chains: any consecutive pair with same level
    delta_zero_chains = [
        c for c in chains
        if any(LEVEL_RANK.get(c["questions"][i+1]["level"], 0)
                 - LEVEL_RANK.get(c["questions"][i]["level"], 0) == 0
               for i in range(len(c["questions"])-1))
    ]
    primary_chains = [c for c in chains if c.get("tier", "primary") == "primary"]
    secondary_chains = [c for c in chains if c.get("tier") == "secondary"]
    primary_sample = rng.sample(primary_chains,
                                 min(args.primary_sample, len(primary_chains)))
    secondary_sample = rng.sample(secondary_chains,
                                   min(args.secondary_sample, len(secondary_chains)))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = AUDIT_RUNS / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    plan = {
        "drafts":       2,
        "secondary":    len(batch_chains(secondary_sample, corpus)),
        "delta_zero":   len(batch_chains(delta_zero_chains, corpus)),
        "primary":      len(batch_chains(primary_sample, corpus)),
        "gaps":         (args.gap_sample + 4) // 5,  # 5/call
        "synthesis":    1,
    }
    print(f"\nbatching plan ({sum(plan.values())} total calls):")
    for k, v in plan.items():
        marker = "*" if (args.only and args.only != k) or k in skipped else " "
        print(f"  {marker} {k:14s} {v} call(s)")

    config = {
        "timestamp": timestamp,
        "seed": args.seed,
        "samples": {
            "secondary": len(secondary_sample),
            "primary":   len(primary_sample),
            "delta_zero": len(delta_zero_chains),
            "gap_sample": args.gap_sample,
            "bucket_pairs": args.bucket_pairs,
        },
        "plan": plan,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    if args.dry_run:
        print(f"\n--dry-run set; wrote {outdir / 'config.json'}")
        return 0

    def should(cat: str) -> bool:
        if args.only and args.only != cat:
            return False
        return cat not in skipped

    if should("drafts"):
        print("\n[1] drafts audit")
        audit_drafts(corpus, outdir)
        time.sleep(INTER_CALL_DELAY_S)
    if should("secondary"):
        print("\n[2] secondary chain sample audit")
        audit_chain_sample(
            secondary_sample, corpus,
            label="secondary", outname="02_secondary.json", outdir=outdir,
            instructions="These chains came from a LENIENT second-pass coverage "
                         "build (Δ ∈ {0,1,2,3}). Be especially attentive to "
                         "consecutive-pair quality, since the lenient sweep is "
                         "where weak chains are likeliest to slip through.",
        )
        time.sleep(INTER_CALL_DELAY_S)
    if should("delta_zero"):
        print("\n[3] Δ=0 chain audit")
        audit_chain_sample(
            delta_zero_chains, corpus,
            label="delta_zero", outname="03_delta_zero.json", outdir=outdir,
            instructions="These chains contain at least one same-level (Δ=0) "
                         "consecutive pair. The lenient prompt allowed Δ=0 ONLY "
                         "when both questions share a scenario thread. Verify "
                         "that claim per-pair.",
            extra_fields='"shared_scenario_for_d0_pair": "yes" | "no" | "unclear",',
        )
        time.sleep(INTER_CALL_DELAY_S)
    if should("primary"):
        print("\n[4] primary chain sample audit")
        audit_chain_sample(
            primary_sample, corpus,
            label="primary", outname="04_primary.json", outdir=outdir,
            instructions="These chains came from the STRICT first-pass build "
                         "(Δ ∈ {1,2}). This is a regression check on strict-pass "
                         "quality; failures here suggest the original chain "
                         "rebuild itself has issues, not just the lenient sweep.",
        )
        time.sleep(INTER_CALL_DELAY_S)
    if should("gaps"):
        print("\n[5] gap detection audit")
        audit_gaps(corpus, outdir, args.gap_sample, rng)
        time.sleep(INTER_CALL_DELAY_S)
    if should("synthesis"):
        print("\n[6] synthesis")
        synthesise(outdir)
        report = write_report(outdir)
        print(f"\nwrote {report.relative_to(REPO_ROOT)}")

    print(f"\nDONE. Audit run dir: {outdir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
