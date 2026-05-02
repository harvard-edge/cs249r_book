#!/usr/bin/env python3
"""Author a candidate question to fill a chain gap (Phase 3.a).

Reads a gap entry (from gaps.proposed.json / gaps.proposed.lenient.json)
that names two existing questions and a missing Bloom level between
them, then prompts Gemini-3.1-pro-preview to draft a bridging question
that fits the (track, topic, target-level) slot.

Inputs per gap entry:
  {
    "track": "edge",
    "topic": "memory-mapped-inference",
    "missing_level": "L3",
    "between": ["edge-0220", "edge-0224"],
    "rationale": "..."
  }

Outputs per accepted draft:
  interviews/vault/questions/<track>/<area>/<auto-id>.yaml.draft
    — full question YAML with stamped authoring metadata. The .draft
      suffix is intentional: vault check / vault build only load *.yaml,
      so drafts ride along in the tree without affecting the release set
      until they are promoted (renamed to .yaml) by a follow-up step.

Usage:
  python3 generate_question_for_gap.py --gap-index 0
  python3 generate_question_for_gap.py --gaps-from interviews/vault/gaps.proposed.json --limit 5
  python3 generate_question_for_gap.py --gaps-from <path> --limit 30 --output-dir <dir>

This is the Phase 3.a tool. Validation (originality / level-fit /
coherence / bridge) is a separate concern handled by validate_drafts.py.
The only validation done here is structural Pydantic-schema acceptance,
which is the gate that prevents writing a malformed YAML to disk.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
ID_REGISTRY = VAULT_DIR / "id-registry.yaml"
# AI-pipeline staging lives under _pipeline/ (gitignored).
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT_GAPS = PIPELINE_DIR / "gaps.proposed.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
INTER_CALL_DELAY_S = 6  # be polite to the Gemini CLI's rate limiter

# Imported lazily so the file is still readable as a script even if the
# vault_cli package isn't editable-installed in the current interpreter.
try:
    from vault_cli.models import Question
except ImportError:  # pragma: no cover
    Question = None  # type: ignore


# ─── corpus + registry helpers ────────────────────────────────────────────


def load_corpus_index() -> dict[str, dict]:
    """qid → full YAML dict for every published question.

    We need full bodies (scenario + details) for the between-questions and
    exemplars; the corpus.json summary doesn't carry them.
    """
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


def next_ids_per_track(corpus: dict[str, dict], existing_drafts: list[Path]) -> dict[str, int]:
    """Return per-track next-available numeric suffix.

    Considers BOTH committed YAMLs in the corpus AND any .yaml.draft files
    written in earlier runs of this script — so a batch generating 30 drafts
    gets 30 distinct IDs even before any of them is promoted into the
    id-registry.
    """
    max_for_track: dict[str, int] = {}
    pat = re.compile(r"^([a-z]+)-(\d+)$")
    for qid in corpus:
        m = pat.match(qid)
        if not m:
            continue
        track, num = m.group(1), int(m.group(2))
        if num > max_for_track.get(track, -1):
            max_for_track[track] = num
    for draft in existing_drafts:
        # filename like edge-2545.yaml.draft
        stem = draft.name.split(".")[0]
        m = pat.match(stem)
        if m:
            track, num = m.group(1), int(m.group(2))
            if num > max_for_track.get(track, -1):
                max_for_track[track] = num
    return {t: n + 1 for t, n in max_for_track.items()}


# ─── prompt construction ──────────────────────────────────────────────────


SCHEMA_SUMMARY = """SCHEMA SUMMARY (Pydantic Question, v1.0):
  REQUIRED FIELDS:
    schema_version: "1.0"
    id: "<track>-<NNNN>"            # provided externally, do NOT invent
    track:           one of [cloud, edge, mobile, tinyml, global]
    level:           one of [L1, L2, L3, L4, L5, L6+]
    zone:            one of [analyze, design, diagnosis, evaluation, fluency,
                             implement, mastery, optimization, realization,
                             recall, specification]
    topic:           closed enum (87 topics; use the one in the gap input)
    competency_area: one of [architecture, compute, cross-cutting, data,
                             deployment, latency, memory, networking,
                             optimization, parallelism, power, precision,
                             reliability]
    bloom_level:     one of [remember, understand, apply, analyze,
                             evaluate, create]  # informs cognitive demand
    title:            ≤ 120 chars, descriptive, no trailing period
    scenario:         1-3 sentences setting up a concrete situation
    question:         the explicit interrogative the candidate must answer
    details.realistic_solution: 1-3 sentence high-quality answer
    details.common_mistake:     "**The Pitfall:** ...\\n**The Rationale:** ...\\n**The Consequence:** ..."
    details.napkin_math:        OPTIONAL but recommended for L3+
    status:          MUST be "draft" (this is a candidate for review)
    provenance:      MUST be "llm-draft"
    requires_explanation: false (default)
    expected_time_minutes: integer, ≥ 0  (typical: 5-15)

LEVEL ↔ BLOOM ROUGH MAPPING:
    L1 → remember          L2 → understand         L3 → apply / analyze
    L4 → analyze           L5 → evaluate           L6+ → create

  STRICT JSON OUTPUT FORMAT (no prose, no fences, no extra fields):
  {
    "title":  "<title>",
    "scenario": "<scenario>",
    "question": "<question>",
    "zone":  "<zone>",
    "bloom_level": "<bloom>",
    "phase":   "training | inference | both",
    "expected_time_minutes": <int>,
    "tags": ["<tag>", ...],
    "details": {
      "realistic_solution": "<1-3 sentence answer>",
      "common_mistake": "**The Pitfall:** ...\\n**The Rationale:** ...\\n**The Consequence:** ...",
      "napkin_math": "**Assumptions & Constraints:** ...\\n\\n**Calculations:** ...\\n\\n**Conclusion:** ..."
    }
  }
"""


def question_payload(q: dict[str, Any]) -> dict[str, Any]:
    """Compact view of an existing question to feed Gemini as context."""
    d = q.get("details") or {}
    return {
        "id": q.get("id"),
        "level": q.get("level"),
        "zone": q.get("zone"),
        "bloom_level": q.get("bloom_level"),
        "title": q.get("title"),
        "scenario": q.get("scenario"),
        "question": q.get("question"),
        "realistic_solution": d.get("realistic_solution"),
    }


def find_exemplars(
    corpus: dict[str, dict],
    track: str,
    topic: str,
    target_level: str,
    skip_ids: set[str],
    limit: int = 3,
) -> list[dict]:
    """Pick up to `limit` published questions in the same (track, topic) at
    the target level. Used as style-and-cognitive-load exemplars for the
    drafted question.
    """
    pool = [
        q for q in corpus.values()
        if q.get("track") == track
        and q.get("topic") == topic
        and q.get("level") == target_level
        and q.get("status") == "published"
        and q.get("id") not in skip_ids
    ]
    pool.sort(key=lambda q: q.get("id", ""))
    return pool[:limit]


def build_prompt(gap: dict, between: list[dict], exemplars: list[dict]) -> str:
    parts = [
        "You are an ML systems interview question author. Draft ONE candidate",
        "question that fills the missing rung in a pedagogical chain.",
        "",
        SCHEMA_SUMMARY,
        "",
        f"GAP TO FILL:",
        f"  track:           {gap['track']}",
        f"  topic:           {gap['topic']}",
        f"  target level:    {gap['missing_level']}",
        f"  bridge between:  {gap['between']}",
        f"  rationale:       {gap.get('rationale', '')}",
        "",
        "BETWEEN-QUESTIONS (these MUST flank the new question pedagogically):",
        json.dumps([question_payload(q) for q in between], indent=2),
        "",
        "EXEMPLARS at the target level in the same (track, topic) — match",
        "their voice and cognitive load (NOT their content):",
        json.dumps([question_payload(q) for q in exemplars], indent=2) if exemplars
        else "  (no in-bucket exemplars at this level — use the between-questions' style)",
        "",
        "AUTHORING RULES:",
        "  - The new question MUST chain naturally between the two between-questions:",
        "    Q[lower].level < new.level < Q[higher].level (or equal-level edges where",
        "    one between-question is exactly at target_level — re-read the gap).",
        "  - Same scenario/concept thread as the bridge — do NOT introduce a",
        "    new system topic.",
        "  - Cognitive load matches target Bloom: e.g. L3 (apply) asks the",
        "    candidate to perform a calculation; L4 (analyze) asks for",
        "    decomposition or root-cause; L5 (evaluate) asks for a",
        "    trade-off judgment with quantitative basis.",
        "  - realistic_solution is a high-quality, concise answer — NOT a",
        "    rubric. common_mistake follows the **Pitfall / Rationale /",
        "    Consequence** format. napkin_math has the **Assumptions /",
        "    Calculations / Conclusion** format.",
        "  - Avoid duplicating any title or scenario in the between or",
        "    exemplar inputs.",
        "  - Output ONLY the JSON object specified in the schema summary.",
    ]
    return "\n".join(parts)


# ─── Gemini call ──────────────────────────────────────────────────────────


def call_gemini(prompt: str, model: str = GEMINI_MODEL, timeout: int = 600) -> dict | None:
    try:
        result = subprocess.run(
            ["gemini", "-m", model, "-p", prompt, "--yolo"],
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


# ─── draft assembly + validation ──────────────────────────────────────────


def assemble_draft(
    gap: dict,
    response: dict,
    qid: str,
) -> dict[str, Any]:
    """Build the full YAML body from Gemini's response + gap-derived fields."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    details_in = response.get("details") or {}
    return {
        "schema_version": "1.0",
        "id": qid,
        "track": gap["track"],
        "level": gap["missing_level"],
        "zone": response.get("zone") or "analyze",
        "topic": gap["topic"],
        # competency_area must come from the bridge — the gap entry doesn't
        # carry it, so we inherit from the between-question. assemble_draft
        # is called with this already resolved by main(); see _competency.
        "competency_area": gap.get("_competency_area"),
        "bloom_level": response.get("bloom_level"),
        "phase": response.get("phase") or "both",
        "title": response.get("title", "").strip(),
        "scenario": response.get("scenario", "").strip(),
        "question": response.get("question", "").strip(),
        "details": {
            "realistic_solution": (details_in.get("realistic_solution") or "").strip(),
            "common_mistake": (details_in.get("common_mistake") or "").strip() or None,
            "napkin_math": (details_in.get("napkin_math") or "").strip() or None,
        },
        "status": "draft",
        "provenance": "llm-draft",
        "requires_explanation": False,
        "expected_time_minutes": int(response.get("expected_time_minutes") or 10),
        "tags": response.get("tags") or None,
        "_authoring": {
            "origin": GEMINI_MODEL,
            "tool": "generate_question_for_gap.py",
            "generated_at": now,
            "gap": {
                "between": gap["between"],
                "missing_level": gap["missing_level"],
                "rationale": gap.get("rationale"),
            },
        },
    }


def schema_validate(draft: dict[str, Any]) -> tuple[bool, str]:
    """Run the draft through Pydantic Question. Returns (ok, error_text)."""
    if Question is None:
        return False, "vault_cli not importable; install with `pip install -e interviews/vault-cli/`"
    # Strip our private metadata; the Pydantic model will accept extra by
    # config, but we don't want it to surface as a validation surprise.
    body = {k: v for k, v in draft.items() if not k.startswith("_")}
    # Drop None-valued optional details so Pydantic gets a clean dict.
    if isinstance(body.get("details"), dict):
        body["details"] = {k: v for k, v in body["details"].items() if v is not None}
    try:
        Question.model_validate(body)
        return True, ""
    except Exception as e:  # pydantic ValidationError stringifies usefully
        return False, str(e)


def write_draft(draft: dict[str, Any], output_dir: Path) -> Path:
    track = draft["track"]
    area = draft["competency_area"]
    qid = draft["id"]
    target_dir = output_dir / track / area
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{qid}.yaml.draft"
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(draft, f, sort_keys=False, allow_unicode=True, width=100)
    return target


# ─── main ─────────────────────────────────────────────────────────────────


def resolve_competency_area(gap: dict, corpus: dict[str, dict]) -> str | None:
    """Inherit competency_area from the between-questions.

    All published questions in the same (track, topic) bucket should agree on
    competency_area (it's a topic-level invariant). We pick from the first
    between question; if they disagree, prefer the lower-level one (since the
    gap is bridging upward from it) and warn the caller.
    """
    for qid in gap.get("between", []):
        q = corpus.get(qid)
        if q and q.get("competency_area"):
            return q["competency_area"]
    return None


def process_gap(
    gap: dict,
    corpus: dict[str, dict],
    next_ids: dict[str, int],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Returns a one-row report describing the outcome."""
    track = gap.get("track")
    if not track or track not in next_ids:
        next_ids[track] = 0
    seq = next_ids[track]
    qid = f"{track}-{seq:04d}"
    next_ids[track] = seq + 1

    between = [corpus[q] for q in gap.get("between", []) if q in corpus]
    if len(between) < 1:
        return {"qid": qid, "ok": False, "why": "no between-questions found in corpus",
                "gap": gap}

    competency = resolve_competency_area(gap, corpus)
    if not competency:
        return {"qid": qid, "ok": False, "why": "could not resolve competency_area",
                "gap": gap}

    exemplars = find_exemplars(
        corpus,
        track=track,
        topic=gap["topic"],
        target_level=gap["missing_level"],
        skip_ids=set(gap.get("between", [])),
        limit=3,
    )

    prompt = build_prompt(gap, between, exemplars)
    if dry_run:
        return {"qid": qid, "ok": True, "dry_run": True,
                "prompt_chars": len(prompt),
                "exemplars": [e["id"] for e in exemplars]}

    response = call_gemini(prompt)
    if response is None:
        return {"qid": qid, "ok": False, "why": "no/unparsable Gemini response", "gap": gap}

    gap_with_area = dict(gap)
    gap_with_area["_competency_area"] = competency
    draft = assemble_draft(gap_with_area, response, qid)

    ok, why = schema_validate(draft)
    if not ok:
        return {"qid": qid, "ok": False, "why": f"schema: {why[:300]}",
                "gap": gap, "draft": draft}

    target = write_draft(draft, output_dir)
    return {"qid": qid, "ok": True,
            "path": str(target.relative_to(REPO_ROOT)),
            "title": draft["title"],
            "level": draft["level"],
            "competency_area": draft["competency_area"]}


def select_gaps(args: argparse.Namespace) -> list[dict]:
    if args.gap_index is not None:
        all_gaps = json.loads(Path(args.gaps_from or DEFAULT_GAPS).read_text(encoding="utf-8"))
        return [all_gaps[args.gap_index]]
    gaps_path = Path(args.gaps_from or DEFAULT_GAPS)
    all_gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
    return all_gaps[: args.limit] if args.limit else all_gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaps-from", type=Path,
                    help=f"path to gaps JSON (default {DEFAULT_GAPS})")
    ap.add_argument("--gap-index", type=int,
                    help="process a single gap entry by 0-based index")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N gaps from the file")
    ap.add_argument("--output-dir", type=Path, default=QUESTIONS_DIR,
                    help=f"target tree (default {QUESTIONS_DIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve gaps + build prompts, but don't call Gemini")
    args = ap.parse_args()

    corpus = load_corpus_index()
    existing_drafts = list(args.output_dir.rglob("*.yaml.draft"))
    next_ids = next_ids_per_track(corpus, existing_drafts)
    print(f"corpus: {len(corpus)} questions; "
          f"existing drafts: {len(existing_drafts)}")
    print(f"next-id allocator: {dict(sorted(next_ids.items()))}")

    gaps = select_gaps(args)
    print(f"processing {len(gaps)} gap(s)")

    results: list[dict[str, Any]] = []
    for i, gap in enumerate(gaps):
        print(f"\n[{i+1}/{len(gaps)}] {gap.get('track')}/{gap.get('topic')} "
              f"L?→{gap.get('missing_level')} between={gap.get('between')}")
        if i > 0 and not args.dry_run:
            time.sleep(INTER_CALL_DELAY_S)
        r = process_gap(gap, corpus, next_ids, args.output_dir, dry_run=args.dry_run)
        results.append(r)
        if r.get("ok"):
            print(f"  ✓ {r['qid']}: {r.get('path') or '(dry-run)'}")
        else:
            print(f"  ✗ {r['qid']}: {r.get('why')}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDONE: {n_ok}/{len(results)} draft(s) written successfully")
    return 0 if n_ok > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
