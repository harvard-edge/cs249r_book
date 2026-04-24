#!/usr/bin/env python3
"""Backfill the `question` YAML field across the StaffML corpus with OpenAI.

This is the OpenAI/GPT port of ``gemini_backfill_question.py``. It keeps the
same corpus walk, idempotency behavior, batching, thread-pool parallelism, and
text-splice YAML insertion routine so existing folded scalars are not reflowed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"

# Keep the prompt short — the job is mechanical. The "quality bar" is
# simply: extract or synthesize the single interrogative sentence that
# matches what the realistic_solution is answering.
BACKFILL_PROMPT = """You are editing the StaffML ML-systems interview corpus.

Each of the {num_questions} questions below has a scenario and a
realistic_solution, but no explicit one-sentence ask. Your job is to
produce that missing sentence — the exact question a human interviewer
would speak after reading the scenario aloud.

Rules:
  1. ONE sentence, ending with `?`. Max 200 characters.
  2. The sentence must match what the `realistic_solution` is answering.
     If the solution argues "pipeline parallelism beats tensor parallelism
     here because…", the question should ask which parallelism strategy
     to choose. Do not invent asks that the solution does not answer.
     Use the full item context: title, track, level, zone, topic, scenario,
     realistic_solution, common_mistake, napkin_math, and MCQ options when
     present. The scenario alone is not enough.
  3. If the scenario already contains a `?` sentence that captures the
     ask, copy that interrogative verbatim (still as ONE sentence).
  4. Use concrete language. Reference the scenario's concrete numbers
     or setup when helpful ("at the 33 ms frame deadline…"). Avoid
     meta-phrasing like "Based on the above" or "According to the
     scenario".
  5. No quotation marks around the sentence. Plain text only.

Output STRICT JSON only — an array, one object per input, in the same
order:

[
  {{
    "id": "edge-0546",
    "question": "Which parallelism strategy — tensor or pipeline — would you choose for this model, and why?"
  }},
  ...
]

Do not include any prose outside the JSON array.

## Questions to process

{questions_json}
"""


@dataclass
class Candidate:
    path: Path
    id: str
    track: str
    level: str
    zone: str
    topic: str
    competency_area: str
    bloom_level: str
    phase: str | None
    title: str
    scenario: str
    realistic_solution: str
    common_mistake: str
    napkin_math: str | None
    options: list[str] | None
    correct_index: int | None


def load_candidates(tracks: set[str] | None, limit: int | None) -> list[Candidate]:
    """Find every YAML that's missing a `question` field."""
    out: list[Candidate] = []
    paths = sorted(QUESTIONS_DIR.glob("*/*.yaml"))
    for p in paths:
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            print(f"  [skip] YAML error in {p}: {e}", file=sys.stderr)
            continue
        track = d.get("track", "")
        if tracks and track not in tracks:
            continue
        # Idempotency: skip if already filled.
        if (d.get("question") or "").strip():
            continue
        scenario = (d.get("scenario") or "").strip()
        if not scenario:
            continue
        details = d.get("details") or {}
        out.append(
            Candidate(
                path=p,
                id=d.get("id", p.stem),
                track=track,
                level=d.get("level", ""),
                zone=d.get("zone", ""),
                topic=d.get("topic", ""),
                competency_area=d.get("competency_area", ""),
                bloom_level=d.get("bloom_level", ""),
                phase=d.get("phase"),
                title=(d.get("title") or "").strip(),
                scenario=scenario,
                realistic_solution=(details.get("realistic_solution") or "").strip(),
                common_mistake=(details.get("common_mistake") or "").strip(),
                napkin_math=(details.get("napkin_math") or None),
                options=details.get("options"),
                correct_index=details.get("correct_index"),
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def slim_for_prompt(c: Candidate) -> dict:
    """Thin the payload sent to GPT so batch prompts stay bounded."""

    def cap(s: str, n: int) -> str:
        return s if len(s) <= n else (s[: n - 1] + "…")

    slim = {
        "id": c.id,
        "title": c.title,
        "track": c.track,
        "level": c.level,
        "zone": c.zone,
        "topic": c.topic,
        "competency_area": c.competency_area,
        "bloom_level": c.bloom_level,
        "phase": c.phase,
        "scenario": cap(c.scenario, 2000),
        "realistic_solution": cap(c.realistic_solution, 1500),
    }
    # common_mistake and napkin_math are only sometimes load-bearing for
    # the ask — include a short version so the model sees them.
    if c.common_mistake:
        slim["common_mistake"] = cap(c.common_mistake, 500)
    if c.napkin_math:
        slim["napkin_math"] = cap(c.napkin_math, 500)
    if c.options:
        slim["options"] = [cap(str(option), 240) for option in c.options]
        slim["correct_index"] = c.correct_index
    return slim


def parse_response(raw: str) -> list[dict]:
    """Parse strict JSON, accepting an object wrapper if JSON mode adds one."""
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "questions", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if "id" in data and "question" in data:
            return [data]
    raise TypeError(f"response not a JSON list or wrapper object: {type(data)}")


def generate_batch(
    batch_idx: int,
    batch: list[Candidate],
    model: str,
    output_dir: Path,
    provider: str,
) -> dict:
    """Send one batch to a model provider, return a dict of {id: question}."""
    prompt = BACKFILL_PROMPT.format(
        num_questions=len(batch),
        questions_json=json.dumps([slim_for_prompt(c) for c in batch], ensure_ascii=False, indent=2),
    )

    raw = ""
    try:
        if provider == "openai":
            from openai import OpenAI  # type: ignore[import-not-found]

            if not os.environ.get("OPENAI_API_KEY"):
                return {"batch": batch_idx, "ok": False, "error": "OPENAI_API_KEY not set"}
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "staffml_question_backfill",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "question": {"type": "string"},
                                        },
                                        "required": ["id", "question"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["items"],
                            "additionalProperties": False,
                        },
                    },
                },
                reasoning_effort="high",
                verbosity="high",
            )
            raw = response.choices[0].message.content or ""
        elif provider == "gemini":
            from google import genai  # type: ignore[import-not-found]

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return {"batch": batch_idx, "ok": False, "error": "GEMINI_API_KEY not set"}
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = response.text or ""
        else:
            return {"batch": batch_idx, "ok": False, "error": f"unknown provider: {provider}"}
        (output_dir / f"raw_{batch_idx:04d}.json").write_text(raw, encoding="utf-8")
        items = parse_response(raw)
        mapping: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            qid = (item.get("id") or "").strip()
            q = (item.get("question") or "").strip()
            if qid and q:
                mapping[qid] = q
        # Apply: write each `question:` back into the YAML file.
        applied = 0
        for c in batch:
            q = mapping.get(c.id)
            if not q:
                continue
            if insert_question_field(c.path, q):
                applied += 1
        return {
            "batch": batch_idx,
            "ok": True,
            "size": len(batch),
            "applied": applied,
            "missing_ids": [c.id for c in batch if c.id not in mapping],
        }
    except json.JSONDecodeError as e:
        (output_dir / f"raw_{batch_idx:04d}.err.txt").write_text(raw, encoding="utf-8")
        return {"batch": batch_idx, "ok": False, "error": f"JSON parse: {e}"}
    except Exception as e:
        return {"batch": batch_idx, "ok": False, "error": str(e)}


def insert_question_field(path: Path, question: str) -> bool:
    """Insert `question:` after the `scenario:` block, preserving the
    author's YAML formatting (which is block-style with folded
    scalars). We operate on text rather than yaml.dump() round-tripping
    so we don't reformat the rest of the file.

    Returns True if the file was modified, False if the field was
    already present (idempotent guard) or the scenario block couldn't
    be located.
    """
    text = path.read_text(encoding="utf-8")
    # Already has it? Bail out defensively.
    if re.search(r"^question:\s", text, flags=re.MULTILINE):
        return False
    # Find the end of the scenario block. scenario is always followed
    # by either `details:` at column 0 or another top-level key. Find
    # the first such key AFTER `scenario:`.
    m = re.search(r"^scenario:", text, flags=re.MULTILINE)
    if not m:
        return False
    # Find the next top-level key (starts at column 0, word chars + colon).
    tail = text[m.end():]
    next_key = re.search(r"^[A-Za-z_][A-Za-z0-9_]*:", tail, flags=re.MULTILINE)
    if not next_key:
        return False
    insertion_offset = m.end() + next_key.start()
    # Escape any YAML-hostile characters in the question. The schema caps
    # it to a short interrogative, but a `:` or leading `-` would confuse
    # the parser. Double-quote defensively and escape internal quotes.
    escaped = question.replace("\\", "\\\\").replace('"', '\\"')
    insertion = f'question: "{escaped}"\n'
    new_text = text[:insertion_offset] + insertion + text[insertion_offset:]
    # Sanity-check that the result still parses as YAML before writing.
    try:
        yaml.safe_load(new_text)
    except yaml.YAMLError:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Model provider. Default is OpenAI; Gemini is useful for quota-limited resumes.",
    )
    parser.add_argument(
        "--tracks",
        default="",
        help="Comma-separated track filter (e.g. 'edge,mobile'). Empty = all.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N candidates (dry-run aid).")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to park OpenAI raw responses (default: _validation_results/question_backfill_<TS>/).",
    )
    args = parser.parse_args()

    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()} or None
    cands = load_candidates(tracks=tracks, limit=args.limit or None)
    print(f"Found {len(cands)} candidates (missing `question`).")
    if not cands:
        print("Nothing to do.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"_validation_results/question_backfill_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")
    print(f"Provider:   {args.provider}")
    print(f"Model:      {args.model}")
    print(f"Workers:    {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Chunk into batches.
    batches = [cands[i : i + args.batch_size] for i in range(0, len(cands), args.batch_size)]
    print(f"Submitting {len(batches)} batches...")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_batch, i, b, args.model, output_dir, args.provider): i
            for i, b in enumerate(batches)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r.get("ok"):
                missing = r.get("missing_ids") or []
                suffix = ""
                if missing:
                    shown = ", ".join(missing[:3])
                    suffix = f" (missing: {shown}{'…' if len(missing) > 3 else ''})"
                print(f"  batch {r['batch']:4d}: {r['applied']}/{r['size']} applied{suffix}")
            else:
                print(f"  batch {r['batch']:4d}: ERROR — {r.get('error')}")

    # Summary
    elapsed = time.time() - t0
    applied = sum(r.get("applied", 0) for r in results)
    failed_batches = sum(1 for r in results if not r.get("ok"))
    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Candidates:      {len(cands)}")
    print(f"  Batches:         {len(batches)}")
    print(f"  Applied:         {applied}")
    print(f"  Failed batches:  {failed_batches}")
    print(f"  Elapsed:         {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output:          {output_dir}")
    (output_dir / "summary.json").write_text(
        json.dumps({"results": results, "applied": applied}, indent=2),
        encoding="utf-8",
    )
    return 0 if failed_batches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
