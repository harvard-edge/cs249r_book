#!/usr/bin/env python3
"""Validate Gemini-authored draft questions (Phase 3.b).

For each ``*.yaml.draft`` under interviews/vault/questions/, run a
multi-gate scorecard:

  1. schema      — Pydantic Question model (same gate as published)
  2. originality — cosine vs nearest neighbour in the same (track, topic);
                   reject if any neighbour exceeds the threshold (default 0.92)
  3. level_fit   — Gemini-judge: "does this question's cognitive load match
                   level=<L>?", calibrated against ≤5 existing L-level
                   questions in the same topic.
  4. coherence   — Gemini-judge: "are scenario / question /
                   realistic_solution mutually consistent?"
  5. bridge      — Gemini-judge: "does this question pedagogically chain
                   between <between[0]> and <between[1]> from the gap?"

A draft passes when **all** gates return "yes" (or skipped). Output:

  - per-draft scorecard rows in interviews/vault/draft-validation-scorecard.json
  - stdout summary: pass/fail counts + per-gate failure reasons

Use case: pilot run lands ~30 drafts in the tree; this script tells the
human reviewer which to look at first (passes) vs which to discard
(failed bridge / failed coherence).

The originality gate needs an embedding model. By default it loads
BAAI/bge-small-en-v1.5 (the same model used for the corpus's
embeddings.npz) so cosine values are directly comparable. Pass
``--no-originality`` to skip if the model load is undesirable.

The LLM-judge gates need ``gemini`` on PATH (gemini-3.1-pro-preview).
Pass ``--no-llm-judge`` to skip those gates and only run schema +
originality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
EMBEDDINGS_PATH = VAULT_DIR / "embeddings.npz"
DEFAULT_OUTPUT = VAULT_DIR / "draft-validation-scorecard.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
ORIGINALITY_THRESHOLD = 0.92  # cosine; >= this is "too duplicative"
LEVEL_FIT_EXEMPLAR_LIMIT = 5

try:
    from vault_cli.models import Question
except ImportError:
    Question = None  # type: ignore


# ─── corpus / drafts ──────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            d = yaml.safe_load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def load_corpus_index() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        d = load_yaml(path)
        if d and d.get("id"):
            out[d["id"]] = d
    return out


def find_drafts(scope: Path | None = None) -> list[Path]:
    root = scope or QUESTIONS_DIR
    return sorted(root.rglob("*.yaml.draft"))


def question_payload(q: dict[str, Any]) -> dict[str, Any]:
    d = q.get("details") or {}
    return {
        "id": q.get("id"),
        "level": q.get("level"),
        "title": q.get("title"),
        "scenario": q.get("scenario"),
        "question": q.get("question"),
        "realistic_solution": d.get("realistic_solution"),
    }


# ─── Gate 1: schema ───────────────────────────────────────────────────────


def gate_schema(draft: dict[str, Any]) -> tuple[bool, str]:
    if Question is None:
        return False, "vault_cli not importable; pip install -e interviews/vault-cli/"
    body = {k: v for k, v in draft.items() if not k.startswith("_")}
    if isinstance(body.get("details"), dict):
        body["details"] = {k: v for k, v in body["details"].items() if v is not None}
    try:
        Question.model_validate(body)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


# ─── Gate 2: originality (cosine vs neighbours) ───────────────────────────


_embed_state: dict[str, Any] = {}


def _load_embedding_model_and_corpus():
    """Lazy: load BAAI/bge-small-en-v1.5 + corpus vectors once per run."""
    if "model" in _embed_state:
        return _embed_state
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"missing {EMBEDDINGS_PATH} — needed for originality gate")
    npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    model_name = str(npz["model_name"])
    model = SentenceTransformer(model_name)
    _embed_state.update({
        "model": model,
        "model_name": model_name,
        "vectors": npz["vectors"],          # (N, dim) L2-normalised
        "qids": [str(x) for x in npz["qids"]],
        "qid_to_row": {str(q): i for i, q in enumerate(npz["qids"])},
    })
    return _embed_state


def gate_originality(
    draft: dict[str, Any],
    corpus: dict[str, dict],
    threshold: float = ORIGINALITY_THRESHOLD,
) -> tuple[bool, str, dict[str, Any]]:
    """Return (ok, reason, detail).

    detail carries the top-1 neighbour qid + cosine, useful for the human
    reviewer to spot-check against.
    """
    import numpy as np
    state = _load_embedding_model_and_corpus()
    model = state["model"]
    vectors = state["vectors"]
    qid_to_row = state["qid_to_row"]

    # Embed the draft (concat title + scenario + question — what the v1
    # corpus embedding script also used for its rows).
    text = "\n".join([
        draft.get("title", "") or "",
        draft.get("scenario", "") or "",
        draft.get("question", "") or "",
    ])
    vec = model.encode([text], normalize_embeddings=True)[0]

    # Restrict comparisons to the same (track, topic) bucket — that's
    # where duplicates would actually matter.
    track = draft.get("track")
    topic = draft.get("topic")
    bucket_qids = [
        qid for qid, q in corpus.items()
        if q.get("track") == track and q.get("topic") == topic
        and qid in qid_to_row
    ]
    if not bucket_qids:
        return True, "", {"note": "no in-bucket corpus neighbours; skipping"}

    rows = np.array([qid_to_row[q] for q in bucket_qids], dtype=np.int64)
    # cosine = dot product since both sides are L2-normalised
    sims = vectors[rows] @ vec  # (len(rows),)
    top = int(np.argmax(sims))
    top_qid = bucket_qids[top]
    top_cos = float(sims[top])

    detail = {"top_neighbour": top_qid, "cosine": round(top_cos, 4),
              "threshold": threshold, "bucket_size": len(bucket_qids)}
    if top_cos >= threshold:
        return False, f"too similar to {top_qid} (cosine={top_cos:.3f} >= {threshold})", detail
    return True, "", detail


# ─── Gate 3-5: Gemini judges ──────────────────────────────────────────────


def call_gemini_judge(prompt: str, timeout: int = 240) -> dict | None:
    """Single judge call; expects strict-JSON {"verdict": "yes|no", "rationale": "..."}."""
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
        return None
    try:
        return json.loads(out[i:j+1])
    except json.JSONDecodeError:
        return None


def _judge_block(draft: dict[str, Any]) -> str:
    return json.dumps(question_payload(draft), indent=2)


def gate_level_fit(draft: dict, corpus: dict[str, dict]) -> tuple[bool, str, dict]:
    target_level = draft.get("level")
    track = draft.get("track")
    topic = draft.get("topic")
    exemplars = sorted(
        [q for q in corpus.values()
         if q.get("track") == track and q.get("topic") == topic
         and q.get("level") == target_level
         and q.get("status") == "published"],
        key=lambda q: q.get("id", ""),
    )[:LEVEL_FIT_EXEMPLAR_LIMIT]

    if not exemplars:
        return True, "", {"note": f"no published L={target_level} exemplars in bucket; skipping"}

    prompt = f"""You are calibrating cognitive load. Given an EXAMPLE PAIR of
existing published interview questions at level={target_level} for
track={track}, topic={topic}, judge whether the CANDIDATE question
matches that level's typical cognitive demand.

Bloom mapping: L1=remember, L2=understand, L3=apply, L4=analyze,
L5=evaluate, L6+=create.

EXEMPLARS at level={target_level}:
{json.dumps([question_payload(q) for q in exemplars], indent=2)}

CANDIDATE:
{_judge_block(draft)}

Return STRICT JSON with no prose or fences:
{{"verdict": "yes" | "no", "rationale": "<one sentence>"}}
"""
    resp = call_gemini_judge(prompt)
    if resp is None:
        return False, "no judge response", {}
    verdict = (resp.get("verdict") or "").strip().lower()
    if verdict == "yes":
        return True, "", {"rationale": resp.get("rationale", "")}
    return False, f"level_fit=no: {resp.get('rationale', '')}", {"rationale": resp.get("rationale")}


def gate_coherence(draft: dict) -> tuple[bool, str, dict]:
    prompt = f"""Judge whether the scenario, question, and realistic_solution
are MUTUALLY CONSISTENT. Specifically:
  - Does the question logically follow from the scenario?
  - Does the realistic_solution actually answer the question (not adjacent)?
  - Are the numbers / system parameters internally consistent across all
    three fields (no contradictions)?

CANDIDATE:
{_judge_block(draft)}

Return STRICT JSON with no prose or fences:
{{"verdict": "yes" | "no", "rationale": "<one sentence>"}}
"""
    resp = call_gemini_judge(prompt)
    if resp is None:
        return False, "no judge response", {}
    verdict = (resp.get("verdict") or "").strip().lower()
    if verdict == "yes":
        return True, "", {"rationale": resp.get("rationale", "")}
    return False, f"coherence=no: {resp.get('rationale', '')}", {"rationale": resp.get("rationale")}


def gate_bridge(draft: dict, corpus: dict[str, dict]) -> tuple[bool, str, dict]:
    auth = draft.get("_authoring") or {}
    gap = auth.get("gap") or {}
    between_ids = gap.get("between") or []
    between = [corpus.get(q) for q in between_ids if corpus.get(q)]
    if len(between) < 2:
        # Without two between-questions we can't judge a bridge meaningfully.
        return True, "", {"note": "fewer than 2 between-questions in corpus; skipping"}

    prompt = f"""Judge whether the CANDIDATE question pedagogically chains
between the two BETWEEN-questions. Specifically:
  - Is the candidate's cognitive load above between[0]'s level and at or
    below between[1]'s level (Bloom progression direction)?
  - Does the candidate share scenario/concept thread with the between-
    questions (not introducing a new system)?
  - Would inserting the candidate between the two existing questions
    produce a coherent +1 (or +2 last-resort) progression chain?

BETWEEN[0] (lower):
{json.dumps(question_payload(between[0]), indent=2)}

BETWEEN[1] (higher):
{json.dumps(question_payload(between[1]), indent=2)}

CANDIDATE:
{_judge_block(draft)}

Return STRICT JSON with no prose or fences:
{{"verdict": "yes" | "no", "rationale": "<one sentence>"}}
"""
    resp = call_gemini_judge(prompt)
    if resp is None:
        return False, "no judge response", {}
    verdict = (resp.get("verdict") or "").strip().lower()
    if verdict == "yes":
        return True, "", {"rationale": resp.get("rationale", "")}
    return False, f"bridge=no: {resp.get('rationale', '')}", {"rationale": resp.get("rationale")}


# ─── runner ───────────────────────────────────────────────────────────────


def evaluate_draft(
    draft_path: Path,
    corpus: dict[str, dict],
    args: argparse.Namespace,
) -> dict[str, Any]:
    draft = load_yaml(draft_path)
    if not draft:
        return {"path": str(draft_path), "verdict": "fail",
                "errors": ["could not load YAML"]}

    try:
        rel_path = str(draft_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = str(draft_path)
    rec: dict[str, Any] = {
        "path": rel_path,
        "draft_id": draft.get("id"),
        "track": draft.get("track"),
        "topic": draft.get("topic"),
        "level": draft.get("level"),
    }

    # Gate 1 — schema (mandatory)
    ok, why = gate_schema(draft)
    rec["schema_ok"] = ok
    if not ok:
        rec["schema_error"] = why
        rec["verdict"] = "fail"
        return rec  # downstream gates assume a structurally valid YAML

    # Gate 2 — originality
    if args.no_originality:
        rec["originality"] = "skipped"
    else:
        try:
            ok, why, detail = gate_originality(draft, corpus, threshold=args.threshold)
            rec["originality"] = "pass" if ok else "fail"
            rec["originality_detail"] = detail
            if not ok:
                rec["originality_reason"] = why
        except Exception as e:
            rec["originality"] = "error"
            rec["originality_reason"] = str(e)[:200]

    # Gates 3-5 — Gemini judges
    if args.no_llm_judge:
        rec["level_fit"] = "skipped"
        rec["coherence"] = "skipped"
        rec["bridge"]    = "skipped"
    else:
        for name, gate in [("level_fit", gate_level_fit),
                           ("coherence", gate_coherence),
                           ("bridge", gate_bridge)]:
            try:
                if name == "coherence":
                    ok, why, detail = gate(draft)
                else:
                    ok, why, detail = gate(draft, corpus)
            except Exception as e:
                rec[name] = "error"
                rec[f"{name}_reason"] = str(e)[:200]
                continue
            rec[name] = "pass" if ok else "fail"
            rec[f"{name}_detail"] = detail
            if not ok:
                rec[f"{name}_reason"] = why
            time.sleep(args.judge_delay)  # be polite between calls

    # Final verdict: pass iff every non-skipped gate is pass.
    gate_results = [
        rec.get("originality"),
        rec.get("level_fit"),
        rec.get("coherence"),
        rec.get("bridge"),
    ]
    has_fail = any(r == "fail" for r in gate_results)
    has_error = any(r == "error" for r in gate_results)
    rec["verdict"] = "fail" if has_fail else ("error" if has_error else "pass")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", type=Path, default=None,
                    help=f"directory tree to scan for *.yaml.draft "
                         f"(default {QUESTIONS_DIR})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"scorecard JSON (default {DEFAULT_OUTPUT})")
    ap.add_argument("--no-originality", action="store_true",
                    help="skip the embedding-based originality gate")
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="skip the Gemini-judge gates (level_fit, coherence, bridge)")
    ap.add_argument("--threshold", type=float, default=ORIGINALITY_THRESHOLD,
                    help=f"originality cosine cutoff (default {ORIGINALITY_THRESHOLD})")
    ap.add_argument("--judge-delay", type=float, default=4.0,
                    help="seconds between Gemini judge calls (default 4.0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N drafts")
    args = ap.parse_args()

    drafts = find_drafts(args.scope)
    if args.limit:
        drafts = drafts[: args.limit]
    if not drafts:
        print(f"no *.yaml.draft files found under {args.scope or QUESTIONS_DIR}")
        return 0

    corpus = load_corpus_index()
    print(f"corpus: {len(corpus)} published+draft questions; "
          f"drafts to evaluate: {len(drafts)}")

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(drafts, start=1):
        try:
            display = p.relative_to(REPO_ROOT)
        except ValueError:
            display = p
        print(f"\n[{i}/{len(drafts)}] {display}")
        rec = evaluate_draft(p, corpus, args)
        gate_summary = ", ".join(
            f"{g}={rec.get(g, '-')}"
            for g in ("originality", "level_fit", "coherence", "bridge")
        )
        print(f"  verdict={rec.get('verdict'):4s}  {gate_summary}")
        if rec.get("verdict") == "fail":
            for k in ("schema_error", "originality_reason",
                      "level_fit_reason", "coherence_reason", "bridge_reason"):
                if k in rec:
                    print(f"    {k}: {str(rec[k])[:200]}")
        rows.append(rec)

    try:
        out_display = args.output.relative_to(REPO_ROOT)
    except ValueError:
        out_display = args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "originality_threshold": args.threshold,
        "drafts_evaluated": len(rows),
        "passes": sum(1 for r in rows if r.get("verdict") == "pass"),
        "fails":  sum(1 for r in rows if r.get("verdict") == "fail"),
        "errors": sum(1 for r in rows if r.get("verdict") == "error"),
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {out_display}")
    n_pass = sum(1 for r in rows if r.get("verdict") == "pass")
    n_fail = sum(1 for r in rows if r.get("verdict") == "fail")
    n_err  = sum(1 for r in rows if r.get("verdict") == "error")
    print(f"summary: pass={n_pass}  fail={n_fail}  error={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
