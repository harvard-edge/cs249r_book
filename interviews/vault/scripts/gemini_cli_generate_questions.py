#!/usr/bin/env python3
"""Gemini 3.1 Pro driven, BATCHED question generation.

Why batched: the gemini-3.1-pro-preview API has a 250-call/day cap, and
each call has a massive context window. Generating one question per call
wastes the budget. This runner packs N cells per call (default 12) and
asks Gemini to return a JSON array of N questions in one shot, dropping
the per-call cost dramatically.

Cell selection is balanced across tracks × competency areas × zones ×
levels by round-robin so a single batch covers diverse coverage gaps,
not just one corner of the corpus.

The runner is review-first by design: every emitted draft is
``status: draft`` with ``provenance: llm-draft``. Promotion to published
requires a separate verification pass.

Usage:

    # Plan a balanced batch (no API calls):
    python3 gemini_cli_generate_questions.py --auto-balance --total 60 --dry-run

    # Generate 60 questions in 5 calls of 12 cells each:
    python3 gemini_cli_generate_questions.py --auto-balance --total 60 --batch-size 12

    # Generate visual-eligible questions:
    python3 gemini_cli_generate_questions.py --auto-balance --total 20 --batch-size 5 --visual

    # Single-call targeted generation:
    python3 gemini_cli_generate_questions.py --target cloud:queueing-theory:specification:L5 --target tinyml:duty-cycling:analyze:L4 --batch-size 2

Architecture: see interviews/vault/visuals/ARCHITECTURE.md.
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

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
VISUALS_DIR = VAULT_DIR / "visuals"
SCHEMA_DIR = VAULT_DIR / "schema"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OUTPUT_DIR = VAULT_DIR / "_validation_results" / "gemini_generation"

# Same constants reference used by gemini_cli_math_review.py — keeps
# generation and validation consistent.
HARDWARE_REFERENCE = """
Reference constants to use unless the question explicitly states otherwise:
- H100 SXM: 80 GB HBM3, 3.35 TB/s HBM bandwidth, 989 TFLOP/s FP16 tensor, 700 W.
- A100 80GB SXM: ~2.0 TB/s HBM2e bandwidth, 312 TFLOP/s FP16 tensor, 400 W.
- MI300X: 192 GB HBM3, 5.3 TB/s bandwidth, ~1307 TFLOP/s FP16 sparse peak.
- Jetson AGX Orin: up to 275 TOPS INT8, ~204.8 GB/s LPDDR5, 15-60 W modes.
- Hailo-8: 26 TOPS INT8, ~2.5 W accelerator power.
- Apple A17 Pro Neural Engine: roughly 35 TOPS.
- Snapdragon 8 Gen 3 Hexagon NPU: roughly 45 TOPS.
- Cortex-M4: 80-240 MHz, KB-scale SRAM.
- 1 byte = 8 bits. 1 GB/s = 1000 MB/s for napkin math unless question says GiB.
- FP16/BF16 weights: 2 bytes/parameter. INT8: 1 byte/parameter. INT4: 0.5 byte/parameter.
- KV cache: 2 x layers x KV heads x head dim x sequence length x batch x bytes.
- Ring AllReduce lower-bound byte term: 2(N-1)/N x payload / bandwidth.
- Power energy: Wh = W x hours; kWh cost = kWh x price.
""".strip()

# Topics suitable for visual-archetype generation (per audit_visual_questions.py).
VISUAL_ARCHETYPES: dict[str, dict[str, str]] = {
    "collective-communication": {"kind": "dot", "archetype": "ring/tree topology"},
    "pipeline-parallelism": {"kind": "matplotlib", "archetype": "bubble Gantt"},
    "kv-cache-management": {"kind": "matplotlib", "archetype": "page layout bar chart"},
    "queueing-theory": {"kind": "matplotlib", "archetype": "hockey-stick curve"},
    "data-pipeline-engineering": {"kind": "matplotlib", "archetype": "throughput stages"},
    "memory-hierarchy-design": {"kind": "matplotlib", "archetype": "tier bandwidth bars"},
    "interconnect-topology": {"kind": "dot", "archetype": "topology placement"},
    "network-bandwidth-bottlenecks": {"kind": "dot", "archetype": "fanout diagram"},
    "duty-cycling": {"kind": "matplotlib", "archetype": "sleep/wake timeline"},
    "fault-tolerance-checkpointing": {"kind": "matplotlib", "archetype": "RPO/RTO timeline"},
}

# Diverse balance dimensions for --auto-balance.
TRACKS = ["cloud", "edge", "mobile", "tinyml"]
COMPETENCY_AREA_TOPIC = {
    "compute": "compute-cost-estimation",
    "memory": "memory-hierarchy-design",
    "networking": "network-bandwidth-bottlenecks",
    "parallelism": "pipeline-parallelism",
    "latency": "queueing-theory",
    "deployment": "model-serving-infrastructure",
    "data": "data-pipeline-engineering",
    "power": "duty-cycling",
    "precision": "quantization-fundamentals",
    "reliability": "fault-tolerance-checkpointing",
    "optimization": "communication-computation-overlap",
    "architecture": "kv-cache-management",
    "cross-cutting": "distributed-training-economics",
}
ZONES_BY_LEVEL: dict[str, list[str]] = {
    "L3": ["fluency", "analyze", "diagnosis"],
    "L4": ["diagnosis", "analyze", "evaluation"],
    "L5": ["specification", "evaluation", "realization"],
    "L6+": ["mastery", "specification"],
}
TARGET_LEVELS = ["L3", "L4", "L5", "L6+"]
# Track applicability: TinyML cannot meaningfully test KV cache, etc.
TRACK_TOPIC_BLOCKLIST = {
    "tinyml": {"kv-cache-management", "pipeline-parallelism",
               "interconnect-topology", "data-pipeline-engineering",
               "distributed-training-economics"},
    "mobile": {"pipeline-parallelism", "interconnect-topology"},
}

BATCH_PROMPT = """You are an expert ML systems interview-question author.

Below is a JSON array of CELLS to fill — each cell specifies one question's
target. Generate ONE question per cell and return a JSON array of EXACTLY
{n_cells} objects, in the same order. No commentary, no markdown.

Each output object must include this schema:

{{
  "cell_index": <int — matches the input cell's index>,
  "title": "<short title, 5-10 words>",
  "scenario": "<1-3 sentence interview scenario; ground in named hardware>",
  "question": "<one-sentence task; what the candidate must compute or decide>",
  "realistic_solution": "<2-4 sentence answer with the math worked through>",
  "common_mistake": "<1-2 sentence misconception this catches>",
  "napkin_math": "<compact calculation; must contain a digit>",
  "expected_time_minutes": <integer 4-15>,
  "competency_area": "<from the cell's competency_area>"{visual_field_spec}
}}

Hardware reference:
{hardware_reference}

Cells (each cell is one question to generate):

{cells_json}

Rules:
- One question per cell, in array order, with cell_index matching.
- Use real hardware specs from the reference. Pick the most natural
  hardware for the track if not specified.
- Match Bloom level for the cell.
- Diverse scenarios: do not duplicate canonical questions (KV-cache for
  Llama-70B, Ring AllReduce on 4 ranks).
- napkin_math compact and machine-checkable; must contain a digit.
- Each question stands alone — no cross-references between cells.
{visual_rules}
Return only the JSON array.
"""

VISUAL_FIELD_SPEC = """,
  "visual": {{
    "kind": "<dot | matplotlib>",
    "alt": "<200-char objective description of the diagram>",
    "caption": "<one-line caption>"
  }},
  "visual_source": "<the diagram source code as a single string>"
"""

VISUAL_RULES = """- For cells with with_visual=true, include the visual + visual_source fields.
  - DOT: a complete `digraph {{...}}` with semantic palette
    (compute fill #cfe2f3 / stroke #4a90c4, data fill #d4edda / stroke
    #3d9e5a, accent fill #fdebd0 / stroke #c87b2a). Node count ≤ 16.
  - matplotlib: a Python script that reads os.environ["VISUAL_OUT_PATH"]
    and `savefig(out, format="svg", bbox_inches="tight")`. matplotlib
    + numpy only. No seaborn. Include `import os`.
- For cells with with_visual=false, OMIT visual + visual_source entirely.
"""


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

def parse_target(spec: str) -> dict[str, Any]:
    """Parse `track:topic:zone:level` into a cell dict."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"--target must be track:topic:zone:level (got {spec!r})")
    return dict(zip(["track", "topic", "zone", "level"], parts))


def bloom_for_level(level: str) -> str:
    return {
        "L1": "remember", "L2": "understand", "L3": "apply",
        "L4": "analyze", "L5": "evaluate", "L6+": "create",
    }.get(level, "apply")


def auto_balance(total: int, want_visual: bool, seed: int = 0) -> list[dict[str, Any]]:
    """Pick `total` cells balanced across tracks × competency areas × zones × levels.

    Round-robin so a batch covers diverse coverage gaps, not one corner.
    """
    cells: list[dict[str, Any]] = []
    rotation = 0
    for track_i in range(len(TRACKS)):
        for area_i, (area, topic) in enumerate(COMPETENCY_AREA_TOPIC.items()):
            track = TRACKS[(track_i + seed) % len(TRACKS)]
            if topic in TRACK_TOPIC_BLOCKLIST.get(track, set()):
                continue
            level = TARGET_LEVELS[rotation % len(TARGET_LEVELS)]
            zones_for_level = ZONES_BY_LEVEL[level]
            zone = zones_for_level[rotation % len(zones_for_level)]
            with_visual = bool(want_visual and topic in VISUAL_ARCHETYPES)
            cells.append({
                "track": track, "topic": topic, "zone": zone,
                "level": level, "with_visual": with_visual,
                "competency_area": area,
            })
            rotation += 1
            if len(cells) >= total:
                return cells
    return cells


# ---------------------------------------------------------------------------
# ID minting
# ---------------------------------------------------------------------------

def next_id_for_track(track: str, used: set[str]) -> str:
    """Return the next available `<track>-NNNN` id, considering reserved IDs."""
    max_n = 0
    for p in QUESTIONS_DIR.glob(f"{track}/*.yaml"):
        m = re.match(rf"{re.escape(track)}-(\d+)$", p.stem)
        if m:
            max_n = max(max_n, int(m.group(1)))
    while True:
        max_n += 1
        candidate = f"{track}-{max_n:04d}"
        if candidate not in used:
            used.add(candidate)
            return candidate


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str, timeout: int = 300) -> str:
    result = subprocess.run(
        ["gemini", "-m", model, "--prompt", prompt],
        input="", capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI failed: {result.stderr.strip()}")
    return result.stdout


def extract_json_array(raw: str) -> list[Any]:
    """Best-effort JSON-array extraction tolerant to markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
        raw = re.sub(r"\n```\s*$", "", raw)
    start = raw.find("[")
    if start < 0:
        # Fallback: maybe a single object, wrap into array
        obj_start = raw.find("{")
        if obj_start < 0:
            raise ValueError("No JSON array or object found in response")
        return [json.loads(raw[obj_start:])]
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("Unbalanced JSON array in response")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

def build_yaml(cell: dict[str, Any], parsed: dict[str, Any], qid: str) -> dict[str, Any]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "id": qid,
        "track": cell["track"],
        "level": cell["level"],
        "zone": cell["zone"],
        "topic": cell["topic"],
        "competency_area": parsed.get("competency_area",
                                       cell.get("competency_area", "cross-cutting")),
        "bloom_level": bloom_for_level(cell["level"]),
        "phase": parsed.get("phase", "both"),
        "title": str(parsed["title"]).strip(),
        "scenario": str(parsed["scenario"]).strip(),
        "question": str(parsed["question"]).strip(),
    }
    if cell.get("with_visual") and "visual" in parsed:
        v = parsed["visual"]
        doc["visual"] = {
            "kind": "svg",
            "path": f"{qid}.svg",
            "alt": str(v.get("alt", "")).strip(),
            "caption": str(v.get("caption", "")).strip(),
        }
    doc["details"] = {
        "realistic_solution": str(parsed["realistic_solution"]).strip(),
        "common_mistake": str(parsed["common_mistake"]).strip(),
        "napkin_math": str(parsed["napkin_math"]).strip(),
    }
    doc.update({
        "status": "draft",
        "provenance": "llm-draft",
        "expected_time_minutes": int(parsed.get("expected_time_minutes", 8)),
        "requires_explanation": True,
        "validated": False,
        "math_verified": False,
        "human_reviewed": {"status": "not-reviewed", "by": None,
                            "date": None, "notes": None},
        "tags": ["gemini-generated", f"target-{cell['zone']}-{cell['level']}"],
        "created_at": today,
    })
    return doc


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_batch(cells: list[dict[str, Any]], model: str,
                   dry_run: bool, output_dir: Path) -> dict[str, Any]:
    """Submit ONE batched call to Gemini covering all cells, parse the
    array response, and write one YAML + (optional) visual source per cell."""
    n = len(cells)
    has_visual = any(c.get("with_visual") for c in cells)

    cells_for_prompt = []
    for i, c in enumerate(cells):
        entry = {
            "index": i,
            "track": c["track"],
            "topic": c["topic"],
            "zone": c["zone"],
            "level": c["level"],
            "bloom": bloom_for_level(c["level"]),
            "with_visual": bool(c.get("with_visual")),
        }
        if entry["with_visual"]:
            arch = VISUAL_ARCHETYPES.get(c["topic"], {})
            entry["visual_format"] = arch.get("kind", "matplotlib")
            entry["visual_archetype"] = arch.get("archetype", "diagram")
        cells_for_prompt.append(entry)

    prompt = BATCH_PROMPT.format(
        n_cells=n,
        cells_json=json.dumps(cells_for_prompt, indent=2),
        hardware_reference=HARDWARE_REFERENCE,
        visual_field_spec=VISUAL_FIELD_SPEC if has_visual else "",
        visual_rules=VISUAL_RULES if has_visual else "",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"prompt_batch_{int(time.time())}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    if dry_run:
        return {"status": "dry-run", "n_cells": n, "prompt_path": str(prompt_path)}

    print(f"  → calling Gemini for batch of {n} cells "
          f"({'visual' if has_visual else 'text-only'}) ...", flush=True)
    raw = call_gemini(prompt, model)
    raw_path = output_dir / f"raw_batch_{int(time.time())}.txt"
    raw_path.write_text(raw, encoding="utf-8")

    try:
        items = extract_json_array(raw)
    except Exception as exc:
        return {"status": "parse-error", "error": str(exc),
                "raw_path": str(raw_path), "n_cells": n}

    if len(items) != n:
        print(f"  ! WARNING: expected {n} items, got {len(items)}; "
              "matching by cell_index where possible", flush=True)

    used_ids: set[str] = set()
    written = []
    failures = []
    by_index = {it.get("cell_index", i): it for i, it in enumerate(items)}

    for i, cell in enumerate(cells):
        parsed = by_index.get(i)
        if parsed is None:
            failures.append({"index": i, "error": "missing in response"})
            continue
        try:
            qid = next_id_for_track(cell["track"], used_ids)
            doc = build_yaml(cell, parsed, qid)
            yaml_path = QUESTIONS_DIR / cell["track"] / f"{qid}.yaml"
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            # Visual source artifact if present
            if cell.get("with_visual") and parsed.get("visual_source"):
                arch = VISUAL_ARCHETYPES.get(cell["topic"], {})
                ext = {"dot": "dot", "matplotlib": "py"}.get(
                    arch.get("kind", "matplotlib"), "py")
                track_visuals = VISUALS_DIR / cell["track"]
                track_visuals.mkdir(parents=True, exist_ok=True)
                (track_visuals / f"{qid}.{ext}").write_text(
                    str(parsed["visual_source"]), encoding="utf-8")
            written.append({"index": i, "id": qid, "path": str(yaml_path)})
        except Exception as exc:
            failures.append({"index": i, "error": str(exc)})

    return {"status": "ok", "n_cells": n, "written": written,
            "failures": failures, "raw_path": str(raw_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-model-override", action="store_true",
                        help="Allow a model other than gemini-3.1-pro-preview.")
    parser.add_argument("--target", action="append", default=[],
                        help="track:topic:zone:level (repeatable)")
    parser.add_argument("--auto-balance", action="store_true",
                        help="Pick balanced cells across tracks × topics × zones × levels.")
    parser.add_argument("--total", type=int, default=12,
                        help="Total cells to generate (auto-balance only).")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Cells per Gemini call (max ~25 for visual, ~50 text-only).")
    parser.add_argument("--max-calls", type=int, default=10,
                        help="Hard cap on API calls per run.")
    parser.add_argument("--visual", action="store_true",
                        help="Request visual archetypes for visual-eligible topics.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Rotation seed for auto-balance.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL and not args.allow_model_override:
        sys.exit(
            f"Refusing to run with --model={args.model!r}; default is "
            f"{DEFAULT_MODEL!r}. Use --allow-model-override to bypass.")

    cells: list[dict[str, Any]] = []
    if args.auto_balance:
        cells = auto_balance(args.total, args.visual, seed=args.seed)
    for spec in args.target:
        cell = parse_target(spec)
        cell["with_visual"] = args.visual and cell["topic"] in VISUAL_ARCHETYPES
        cell["competency_area"] = "cross-cutting"
        cells.append(cell)

    if not cells:
        sys.exit("No cells to generate. Use --auto-balance or --target.")

    print(f"Plan: {len(cells)} cells, batch size {args.batch_size} → "
          f"{(len(cells) + args.batch_size - 1) // args.batch_size} call(s)")
    output_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    summary = {"model": args.model, "n_cells_requested": len(cells),
               "batches": [], "calls_used": 0}

    batches = [cells[i:i + args.batch_size]
               for i in range(0, len(cells), args.batch_size)]
    for bi, batch in enumerate(batches):
        if bi >= args.max_calls:
            print(f"\nReached --max-calls={args.max_calls}; stopping.")
            break
        if bi:
            time.sleep(args.sleep)
        print(f"\n[batch {bi+1}/{min(len(batches), args.max_calls)}] "
              f"{len(batch)} cells")
        result = generate_batch(batch, args.model, args.dry_run, output_dir)
        summary["batches"].append(result)
        summary["calls_used"] += 0 if args.dry_run else 1
        if result["status"] == "ok":
            print(f"  ← wrote {len(result['written'])} questions; "
                  f"failures={len(result['failures'])}")
        else:
            print(f"  ← {result['status']}: {result.get('error', '')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    total_written = sum(len(b.get("written", [])) for b in summary["batches"])
    print(f"\nTotal: {total_written} questions written across "
          f"{summary['calls_used']} API call(s). Summary: {output_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
