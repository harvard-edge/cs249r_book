#!/usr/bin/env python3
"""Gemini 3.1 Pro driven question generation for weak coverage cells.

Reduces single-model bias by letting Gemini draft the question while
Claude (or another reviewer model, via gemini_cli_math_review.py) verifies
the math. Pairs naturally with portfolio_balance_loop.py to target the
under-represented track × zone × level × topic cells.

The runner is review-first by design: every emitted draft is
``status: draft`` with ``provenance: gemini-3.1-pro-preview``. Promotion
to published requires a separate verification pass.

Usage:

    # Dry-run a generation plan (no API calls):
    python3 gemini_cli_generate_questions.py --target cloud:queueing-theory:specification:L5 --dry-run

    # Generate 5 questions targeting weak cells from portfolio balance loop:
    python3 gemini_cli_generate_questions.py --from-portfolio-loop --limit 5

    # Generate visual-enabled questions (DOT or matplotlib source):
    python3 gemini_cli_generate_questions.py --target cloud:pipeline-parallelism:diagnosis:L4 --visual --limit 1

Architecture: see interviews/vault/visuals/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
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

# Pre-curated archetypes from audit_visual_questions.py — the topics where
# a visual genuinely earns its place.
VISUAL_ARCHETYPES: dict[str, dict[str, str]] = {
    "collective-communication": {"kind": "dot", "archetype": "ring-tree-topology"},
    "pipeline-parallelism": {"kind": "matplotlib", "archetype": "bubble-gantt"},
    "kv-cache-management": {"kind": "svg", "archetype": "page-layout"},
    "queueing-theory": {"kind": "matplotlib", "archetype": "hockey-stick-curve"},
    "data-pipeline-engineering": {"kind": "matplotlib", "archetype": "throughput-stages"},
    "memory-hierarchy-design": {"kind": "svg", "archetype": "hierarchy-data-path"},
    "interconnect-topology": {"kind": "dot", "archetype": "topology-placement"},
    "network-bandwidth-bottlenecks": {"kind": "dot", "archetype": "fanout-budget"},
    "duty-cycling": {"kind": "matplotlib", "archetype": "sleep-wake-timeline"},
    "fault-tolerance-checkpointing": {"kind": "matplotlib", "archetype": "rpo-rto-timeline"},
}

# JSON schema the model must return. We keep it tight so deserialization
# is reliable; richer fields are filled in by the runner after parsing.
GENERATION_PROMPT = """You are an expert ML systems interview-question author.

Generate ONE candidate question targeting the cell:
  track    = {track}
  topic    = {topic}
  zone     = {zone}
  level    = {level}
  bloom    = {bloom_hint}

The question must require napkin math grounded in the hardware constants below
unless the cell is genuinely a recall cell.

{hardware_reference}

Return STRICT JSON with this exact schema (no markdown, no commentary):

{{
  "title": "<short title, 5-10 words>",
  "scenario": "<1-3 sentence interview scenario; ground in named hardware>",
  "question": "<one-sentence task; what the candidate must compute or decide>",
  "realistic_solution": "<2-4 sentence answer with the math worked through>",
  "common_mistake": "<1-2 sentence misconception this question catches>",
  "napkin_math": "<the calculation in compact form; must contain at least one digit>",
  "expected_time_minutes": <integer 4-15>,
  "competency_area": "<one of: parallelism, networking, latency, memory, compute, deployment, data, power, precision, reliability, optimization, architecture, cross-cutting>"
}}

Rules:
- Use real hardware specs from the reference. If the cell does not name
  hardware, pick the most natural choice for the track.
- Match Bloom level: L1 = recall fact, L2 = explain, L3 = apply formula,
  L4 = analyze tradeoff, L5 = evaluate or specify, L6+ = synthesize/architect.
- Match zone: realization = sizing, specification = under SLO, mastery =
  cross-cutting tradeoffs, diagnosis = root-cause from symptoms.
- Avoid duplicating known canonical questions (KV-cache sizing for Llama-70B,
  Ring AllReduce for 4 ranks). Pick a fresh angle.
- Keep napkin_math compact and machine-checkable.
"""

VISUAL_GENERATION_ADDENDUM = """

The question must include a visual diagram. Visual archetype: {archetype}
Visual format: {kind}

Add the following extra JSON fields:

  "visual": {{
    "kind": "{kind}",
    "alt": "<200-character objective description of the diagram>",
    "caption": "<one-line caption shown below the diagram>"
  }},
  "visual_source": "<the diagram source code: {source_lang}>"

For DOT, return a syntactically valid Graphviz `digraph` body that lays out
the named topology. Use semantic palette: compute fill #cfe2f3 / stroke
#4a90c4, data fill #d4edda / stroke #3d9e5a, accent fill #fdebd0 / stroke
#c87b2a. Keep node count ≤ 16 for legibility.

For matplotlib, return a Python script that reads
``os.environ["VISUAL_OUT_PATH"]`` and ``savefig(out, format="svg",
bbox_inches="tight")``. Use only matplotlib + numpy. No seaborn.
"""


# ---------------------------------------------------------------------------
# Cell discovery
# ---------------------------------------------------------------------------

def parse_target(spec: str) -> dict[str, str]:
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


# ---------------------------------------------------------------------------
# ID minting
# ---------------------------------------------------------------------------

def next_id_for_track(track: str) -> str:
    """Return the next available `<track>-NNNN` id by scanning the corpus."""
    max_n = 0
    for p in QUESTIONS_DIR.glob(f"{track}/*.yaml"):
        m = re.match(rf"{re.escape(track)}-(\d+)$", p.stem)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{track}-{max_n + 1:04d}"


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str, timeout: int = 120) -> str:
    """Invoke the gemini CLI and return the raw stdout.

    The CLI accepts --prompt for non-interactive use. We pass an empty
    stdin to keep it from blocking.
    """
    result = subprocess.run(
        ["gemini", "-m", model, "--prompt", prompt],
        input="", capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI failed: {result.stderr.strip()}")
    return result.stdout


def extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction tolerant to markdown fences."""
    raw = raw.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
        raw = re.sub(r"\n```\s*$", "", raw)
    # Find first balanced JSON object
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("Unbalanced JSON in response")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

def build_yaml(cell: dict[str, str], parsed: dict[str, Any], qid: str,
               include_visual: bool, archetype_info: dict[str, str] | None
               ) -> dict[str, Any]:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "id": qid,
        "track": cell["track"],
        "level": cell["level"],
        "zone": cell["zone"],
        "topic": cell["topic"],
        "competency_area": parsed.get("competency_area", "cross-cutting"),
        "bloom_level": bloom_for_level(cell["level"]),
        "phase": parsed.get("phase", "both"),
        "title": parsed["title"].strip(),
        "scenario": parsed["scenario"].strip(),
        "question": parsed["question"].strip(),
    }
    if include_visual and "visual" in parsed:
        v = parsed["visual"]
        suffix = "dot" if archetype_info["kind"] == "dot" else "py" if archetype_info["kind"] == "matplotlib" else "svg"
        doc["visual"] = {
            "kind": archetype_info["kind"],
            "source": f"{qid}.{suffix}",
            "rendered": f"{qid}.svg",
            "alt": v.get("alt", "").strip(),
            "caption": v.get("caption", "").strip(),
        }
    doc["details"] = {
        "realistic_solution": parsed["realistic_solution"].strip(),
        "common_mistake": parsed["common_mistake"].strip(),
        "napkin_math": parsed["napkin_math"].strip(),
    }
    doc.update({
        "status": "draft",
        "provenance": "gemini-3.1-pro-preview",
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
# Generation loop
# ---------------------------------------------------------------------------

def generate_one(cell: dict[str, str], visual: bool, model: str,
                 dry_run: bool, output_dir: Path) -> dict[str, Any]:
    archetype_info = None
    prompt = GENERATION_PROMPT.format(
        track=cell["track"], topic=cell["topic"], zone=cell["zone"],
        level=cell["level"], bloom_hint=bloom_for_level(cell["level"]),
        hardware_reference=HARDWARE_REFERENCE,
    )
    if visual:
        archetype_info = VISUAL_ARCHETYPES.get(cell["topic"])
        if not archetype_info:
            return {"status": "skip", "reason":
                    f"no visual archetype for topic={cell['topic']}"}
        source_lang = {
            "dot": "Graphviz DOT digraph syntax",
            "matplotlib": "Python matplotlib script",
            "svg": "raw SVG (only when requested)",
        }[archetype_info["kind"]]
        prompt += VISUAL_GENERATION_ADDENDUM.format(
            archetype=archetype_info["archetype"],
            kind=archetype_info["kind"],
            source_lang=source_lang,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_slug = "_".join(cell.values())
    prompt_path = output_dir / f"prompt_{cell_slug}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    if dry_run:
        return {"status": "dry-run", "prompt_path": str(prompt_path)}

    print(f"  → calling Gemini for {cell_slug} ...", flush=True)
    raw = call_gemini(prompt, model)
    raw_path = output_dir / f"raw_{cell_slug}.txt"
    raw_path.write_text(raw, encoding="utf-8")

    try:
        parsed = extract_json(raw)
    except Exception as exc:
        return {"status": "parse-error", "error": str(exc),
                "raw_path": str(raw_path)}

    qid = next_id_for_track(cell["track"])
    yaml_path = QUESTIONS_DIR / cell["track"] / f"{qid}.yaml"
    if yaml_path.exists():
        return {"status": "id-collision", "id": qid}

    doc = build_yaml(cell, parsed, qid, visual, archetype_info)

    # Visual source
    if visual:
        source_text = parsed.get("visual_source", "").strip()
        if source_text:
            track_visuals = VISUALS_DIR / cell["track"]
            track_visuals.mkdir(parents=True, exist_ok=True)
            suffix = doc["visual"]["source"].rsplit(".", 1)[-1]
            (track_visuals / doc["visual"]["source"]).write_text(
                source_text, encoding="utf-8")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"status": "generated", "id": qid, "yaml_path": str(yaml_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-model-override", action="store_true",
                        help="Allow a model other than gemini-3.1-pro-preview.")
    parser.add_argument("--target", action="append", default=[],
                        help="track:topic:zone:level (repeatable)")
    parser.add_argument("--from-portfolio-loop", action="store_true",
                        help="Pull weak cells from portfolio_balance_loop.py output.")
    parser.add_argument("--visual", action="store_true",
                        help="Request a visual diagram (only valid for archetype topics).")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between API calls.")
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL and not args.allow_model_override:
        sys.exit(
            f"Refusing to run with --model={args.model!r}; default is "
            f"{DEFAULT_MODEL!r}. Use --allow-model-override to bypass.")

    cells = [parse_target(t) for t in args.target]

    if args.from_portfolio_loop:
        # Best-effort: read latest portfolio_loop output
        loop_dir = VAULT_DIR / "_validation_results" / "portfolio_loop"
        if loop_dir.exists():
            latest = sorted(loop_dir.iterdir())[-1] if list(loop_dir.iterdir()) else None
            if latest:
                report = latest / "weakest_cells.json"
                if report.exists():
                    weak = json.loads(report.read_text())[: args.limit]
                    cells.extend(weak)

    if not cells:
        print("No targets specified. Use --target or --from-portfolio-loop.",
              file=sys.stderr)
        return 1

    cells = cells[: args.limit]
    print(f"Generating {len(cells)} candidate(s) using {args.model} "
          f"(visual={args.visual}, dry_run={args.dry_run})")
    output_dir = args.output_dir / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary = {"model": args.model, "cells": [], "results": []}

    for i, cell in enumerate(cells):
        if i:
            time.sleep(args.sleep)
        print(f"\n[{i+1}/{len(cells)}] cell = {cell}")
        result = generate_one(cell, args.visual, args.model,
                              args.dry_run, output_dir)
        print(f"  ← {result['status']}: {result.get('id') or result.get('reason') or result.get('error') or ''}")
        summary["cells"].append(cell)
        summary["results"].append(result)

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    counts: dict[str, int] = {}
    for r in summary["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Status counts: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
