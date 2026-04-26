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
  "competency_area": "<EXACTLY one of: deployment, parallelism, networking, latency, memory, compute, data, power, precision, reliability, optimization, architecture, cross-cutting>",
  "bloom_level": "<EXACTLY one of: remember, understand, apply, analyze, evaluate, create — must match the cell's `valid_blooms` set>"{visual_field_spec}
}}

Hardware reference:
{hardware_reference}

Cells (each cell is one question to generate):

{cells_json}

Rules:
- One question per cell, in array order, with cell_index matching.
- Use real hardware specs from the reference. Pick the most natural
  hardware for the track if not specified.
- bloom_level MUST be in the cell's `valid_blooms` set. The zone × bloom
  matrix is enforced server-side: zone=recall admits remember/understand
  only; zone=evaluation admits analyze/evaluate; zone=mastery admits
  analyze/evaluate/create. A mismatch causes the YAML to be rejected.
- competency_area is a CLOSED enum (13 values). Do NOT use the topic name
  ("queueing-theory") or zone name ("evaluation") here. Use the closest
  canonical area for the topic (e.g. queueing-theory → latency,
  duty-cycling → power, fault-tolerance-checkpointing → reliability).
- For L5/L6+ cells especially: AVOID trivial framings (raw division of
  payload by bandwidth; "compute X / Y"). The judge rejects these as
  "too shallow for Staff level." Instead, require integration across
  systems (e.g. memory + compute trade-off, parallelism strategy choice
  with synchronization cost), a non-obvious failure mode, or a
  quantitative argument that holds under specific hardware constraints.
- Diverse scenarios: do not duplicate canonical questions (KV-cache for
  Llama-70B, Ring AllReduce on 4 ranks).
- napkin_math compact and machine-checkable; must contain a digit.
- Each question stands alone — no cross-references between cells.
{variant_rules}{visual_rules}
Return only the JSON array.
"""

PARALLELISM_RULES = """- PARALLELISM-VARIANT: these cells target a parallelism gap (pipeline-
  parallelism, collective-communication, kv-cache-management,
  interconnect-topology). The judge has rejected prior parallelism
  drafts as too shallow. To pass:
  • FORBID single-step bandwidth division (e.g. "payload / bandwidth").
    If your napkin_math is one division, reframe.
  • REQUIRE concrete topology — name an actual interconnect appropriate
    to the track:
       - cloud / edge multi-accel: NVLink, NVSwitch, InfiniBand HDR/NDR,
         RoCE v2, PCIe Gen4/5, Hailo-8 fabric
       - mobile / multi-device: BLE, Wi-Fi 6/6E, 5G NR, LoRa, Thread
       - tinyml / multi-MCU: SPI, I2C, UART, CAN, LoRa, ESP-NOW,
         dual-core M7+M4, RP2040 dual-core
  • REQUIRE a synchronization or pipeline-bubble cost — every parallelism
    question must quantify ONE of: bubble fraction in a Gantt-style
    pipeline, AllReduce ring/tree time with the 2(N-1)/N factor,
    consensus latency in a distributed protocol, KV-cache fragmentation
    overhead, or synchronization barrier wait under skew.
  • REQUIRE a non-obvious failure mode in `common_mistake` — naming the
    wrong topology (e.g. ring on shared CSMA medium), confusing
    bandwidth-bound vs latency-bound, missing the all-to-all term in
    AllReduce, treating MoE expert imbalance as load-balancing rather
    than memory pressure.
  • For tinyml multi-MCU: ground in real numbers — Cortex-M4 SPI at
    5-25 MHz, LoRa at 5-50 kbps, ESP-NOW at 1 Mbps. Sub-millisecond
    parallelism budgets are realistic; sub-microsecond is not.
  • FORBID the canonical "Llama-70B KV cache on 4× H100" framing — pick
    a different model size, batch composition, or hardware mix.
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

# Inverse of COMPETENCY_AREA_TOPIC + extras for topics not in that map.
# Used to set the per-cell `competency_area` default when --target is
# explicit (otherwise everything would default to "cross-cutting" and
# the practice page's area filter would mis-bucket them).
TOPIC_TO_AREA: dict[str, str] = {
    "compute-cost-estimation":          "compute",
    "memory-hierarchy-design":          "memory",
    "network-bandwidth-bottlenecks":    "networking",
    "pipeline-parallelism":             "parallelism",
    "queueing-theory":                  "latency",
    "model-serving-infrastructure":     "deployment",
    "data-pipeline-engineering":        "data",
    "duty-cycling":                     "power",
    "quantization-fundamentals":        "precision",
    "fault-tolerance-checkpointing":    "reliability",
    "communication-computation-overlap": "optimization",
    "kv-cache-management":              "architecture",
    "distributed-training-economics":   "cross-cutting",
    "collective-communication":         "networking",
    "interconnect-topology":            "networking",
}


def parse_target(spec: str) -> dict[str, Any]:
    """Parse `track:topic:zone:level` into a cell dict."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"--target must be track:topic:zone:level (got {spec!r})")
    return dict(zip(["track", "topic", "zone", "level"], parts))


def bloom_for_zone_level(zone: str, level: str) -> str:
    """Pick a bloom_level compatible with both zone (ZONE_BLOOM_AFFINITY)
    and level (level-coarse mapping). v0.1.2: the prior `bloom_for_level`
    helper ignored zone, which produced YAMLs that violated the new
    `_zone_bloom_compatible` Pydantic validator (e.g. zone=recall +
    bloom=analyze at L4). This version picks the level-preferred bloom
    if it's valid for the zone, otherwise the zone's nearest admissible
    bloom.
    """
    sys.path.insert(0, str(SCHEMA_DIR))
    try:
        from enums import ZONE_BLOOM_AFFINITY  # type: ignore
    except ImportError:
        ZONE_BLOOM_AFFINITY = {}  # fallback during initial bootstrap
    level_pref = {
        "L1": "remember", "L2": "understand", "L3": "apply",
        "L4": "analyze", "L5": "evaluate", "L6+": "create",
    }.get(level, "apply")
    admits = ZONE_BLOOM_AFFINITY.get(zone, set())
    if not admits or level_pref in admits:
        return level_pref
    # Prefer the highest admissible bloom ≤ level preference.
    bloom_rank = ["remember", "understand", "apply",
                  "analyze", "evaluate", "create"]
    pref_rank = bloom_rank.index(level_pref)
    for b in reversed(bloom_rank[: pref_rank + 1]):
        if b in admits:
            return b
    # Fallback: first admissible.
    for b in bloom_rank:
        if b in admits:
            return b
    return level_pref


# Backward-compat alias used elsewhere in the script.
def bloom_for_level(level: str) -> str:
    return bloom_for_zone_level(zone="", level=level)


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
        "bloom_level": parsed.get(
            "bloom_level",
            bloom_for_zone_level(cell.get("zone", ""), cell["level"]),
        ),
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
                   dry_run: bool, output_dir: Path,
                   prompt_variant: str = "default") -> dict[str, Any]:
    """Submit ONE batched call to Gemini covering all cells, parse the
    array response, and write one YAML + (optional) visual source per cell.

    `prompt_variant` selects an additional rules block injected into
    BATCH_PROMPT. Recognized: "default" (no extra rules), "parallelism"
    (anti-shallow rules for parallelism cells per Phase D.2).
    """
    n = len(cells)
    has_visual = any(c.get("with_visual") for c in cells)

    sys.path.insert(0, str(SCHEMA_DIR))
    try:
        from enums import ZONE_BLOOM_AFFINITY  # type: ignore
    except ImportError:
        ZONE_BLOOM_AFFINITY = {}

    cells_for_prompt = []
    for i, c in enumerate(cells):
        valid_blooms = sorted(ZONE_BLOOM_AFFINITY.get(c["zone"], set())) or [
            bloom_for_level(c["level"])
        ]
        entry = {
            "index": i,
            "track": c["track"],
            "topic": c["topic"],
            "zone": c["zone"],
            "level": c["level"],
            "preferred_bloom": bloom_for_zone_level(c["zone"], c["level"]),
            "valid_blooms": valid_blooms,
            "competency_area": c.get("competency_area", "cross-cutting"),
            "with_visual": bool(c.get("with_visual")),
        }
        if entry["with_visual"]:
            arch = VISUAL_ARCHETYPES.get(c["topic"], {})
            entry["visual_format"] = arch.get("kind", "matplotlib")
            entry["visual_archetype"] = arch.get("archetype", "diagram")
        cells_for_prompt.append(entry)

    variant_rules_text = ""
    if prompt_variant == "parallelism":
        variant_rules_text = PARALLELISM_RULES

    prompt = BATCH_PROMPT.format(
        n_cells=n,
        cells_json=json.dumps(cells_for_prompt, indent=2),
        hardware_reference=HARDWARE_REFERENCE,
        visual_field_spec=VISUAL_FIELD_SPEC if has_visual else "",
        visual_rules=VISUAL_RULES if has_visual else "",
        variant_rules=variant_rules_text,
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

    # B.4: validate-at-write. Round-trip every generated doc through
    # Pydantic before disk write. If validation fails (e.g. zone-bloom
    # mismatch, malformed competency_area, illegal visual.path), record
    # the failure and skip — never persist a self-contradicting draft.
    try:
        sys.path.insert(0, str(VAULT_DIR.parent / "vault-cli" / "src"))
        from vault_cli.models import Question  # type: ignore
        _validate = lambda d: Question.model_validate(d)
    except ImportError:
        _validate = lambda d: None  # bootstrap fallback

    # E.1: track validate-at-write failures separately so a single retry
    # call can fix them. `validate_failures` carries (cell_index,
    # error_message) tuples — retried as one batch at the end.
    validate_failures: list[tuple[int, str]] = []

    def _try_write(cell: dict[str, Any], parsed: dict[str, Any],
                   index: int) -> bool:
        """Validate then write one cell. Returns True on success."""
        try:
            qid = next_id_for_track(cell["track"], used_ids)
            doc = build_yaml(cell, parsed, qid)
            try:
                _validate(doc)
            except Exception as ve:
                validate_failures.append((index, str(ve)))
                return False
            yaml_path = QUESTIONS_DIR / cell["track"] / f"{qid}.yaml"
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            if cell.get("with_visual") and parsed.get("visual_source"):
                arch = VISUAL_ARCHETYPES.get(cell["topic"], {})
                ext = {"dot": "dot", "matplotlib": "py"}.get(
                    arch.get("kind", "matplotlib"), "py")
                track_visuals = VISUALS_DIR / cell["track"]
                track_visuals.mkdir(parents=True, exist_ok=True)
                (track_visuals / f"{qid}.{ext}").write_text(
                    str(parsed["visual_source"]), encoding="utf-8")
            written.append({"index": index, "id": qid, "path": str(yaml_path)})
            return True
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
            return False

    for i, cell in enumerate(cells):
        parsed = by_index.get(i)
        if parsed is None:
            failures.append({"index": i, "error": "missing in response"})
            continue
        _try_write(cell, parsed, i)

    # E.1 retry pass: if any validate-at-write rejections, send ONE more
    # Gemini call with the failure context — Gemini is good at fixing
    # constraint violations once it sees the specific error. Capped at
    # one retry to avoid runaway API spend.
    retry_calls_used = 0
    if validate_failures and not dry_run:
        retry_cells = [cells[idx] for idx, _ in validate_failures]
        retry_errors = "\n".join(
            f"  - cell_index={idx}: {err[:300]}"
            for idx, err in validate_failures
        )
        # Re-run cells_for_prompt construction for just the failed cells
        retry_cells_for_prompt = []
        for new_i, (orig_idx, _) in enumerate(validate_failures):
            c = cells[orig_idx]
            valid_blooms = sorted(
                ZONE_BLOOM_AFFINITY.get(c["zone"], set())
            ) or [bloom_for_level(c["level"])]
            retry_cells_for_prompt.append({
                "index": new_i,
                "track": c["track"],
                "topic": c["topic"],
                "zone": c["zone"],
                "level": c["level"],
                "preferred_bloom": bloom_for_zone_level(c["zone"], c["level"]),
                "valid_blooms": valid_blooms,
                "competency_area": c.get("competency_area", "cross-cutting"),
                "with_visual": bool(c.get("with_visual")),
            })
        retry_prompt = (
            "Your previous JSON had these validate-at-write violations:\n"
            f"{retry_errors}\n\n"
            "Re-emit ONLY the failed items as a JSON array, fixed. "
            "Use the same schema as the original prompt. The CRITICAL "
            "constraints to satisfy this time:\n"
            "- competency_area MUST be one of {deployment, parallelism, "
            "networking, latency, memory, compute, data, power, precision, "
            "reliability, optimization, architecture, cross-cutting}.\n"
            "- bloom_level MUST be in the cell's valid_blooms.\n"
            "- Match cell_index to your output array index 0..N-1.\n\n"
            "Cells to re-generate:\n\n"
            f"{json.dumps(retry_cells_for_prompt, indent=2)}\n\n"
            "Return only the JSON array."
        )
        retry_path = output_dir / f"prompt_retry_{int(time.time())}.txt"
        retry_path.write_text(retry_prompt, encoding="utf-8")
        print(f"  → retry call: {len(validate_failures)} validate-at-write "
              f"failures, sending one corrective call ...", flush=True)
        try:
            retry_raw = call_gemini(retry_prompt, model)
            retry_calls_used = 1
            (output_dir / f"raw_retry_{int(time.time())}.txt").write_text(
                retry_raw, encoding="utf-8")
            retry_items = extract_json_array(retry_raw)
            retry_by_index = {
                it.get("cell_index", i): it
                for i, it in enumerate(retry_items)
            }
            # Wipe validate_failures so successful retries won't
            # double-count; failures that retry-fail get re-appended
            # by _try_write.
            recovered = 0
            for new_i, (orig_idx, _) in enumerate(validate_failures):
                parsed_retry = retry_by_index.get(new_i)
                if parsed_retry is None:
                    continue
                # Reset validate_failures entry so a retry-pass
                # failure reappends; on success it just doesn't.
                if _try_write(cells[orig_idx], parsed_retry, orig_idx):
                    recovered += 1
            print(f"  → retry recovered {recovered}/"
                  f"{len(validate_failures)} items", flush=True)
        except Exception as retry_exc:
            print(f"  ! retry call failed: {retry_exc}", flush=True)

    # Surface validate_failures that didn't recover into the failures list
    # (so downstream summary sees them).
    written_indices = {w["index"] for w in written}
    for orig_idx, err in validate_failures:
        if orig_idx not in written_indices:
            failures.append({
                "index": orig_idx,
                "error": f"validate-at-write rejected (after retry): {err[:200]}",
            })

    return {"status": "ok", "n_cells": n, "written": written,
            "failures": failures, "raw_path": str(raw_path),
            "retry_calls_used": retry_calls_used}


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
    parser.add_argument("--total", type=int, default=30,
                        help="Total cells to generate (auto-balance only).")
    parser.add_argument("--batch-size", type=int, default=30,
                        help=("Cells per Gemini call. Defaults raised from 12 to "
                              "30 on 2026-04-25 — Gemini Pro's 1M context easily "
                              "handles 30 cells × ~2.5 KB prompt fragment, and "
                              "the 250-call/day cap rewards larger batches. "
                              "Use 20 for visual-bearing batches."))
    parser.add_argument("--max-calls", type=int, default=20,
                        help="Hard cap on API calls per run.")
    parser.add_argument("--visual", action="store_true",
                        help="Request visual archetypes for visual-eligible topics.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Rotation seed for auto-balance.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--prompt-variant", default="default",
                        choices=["default", "parallelism"],
                        help="Inject extra rules block: 'default' uses the "
                             "stock prompt; 'parallelism' adds anti-shallow "
                             "rules for parallelism-topic cells (Phase D.2).")
    parser.add_argument("--targets-from", type=Path, default=None,
                        help="Read --target strings from a file (one per "
                             "line). Useful with hand-authored target lists.")
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL and not args.allow_model_override:
        sys.exit(
            f"Refusing to run with --model={args.model!r}; default is "
            f"{DEFAULT_MODEL!r}. Use --allow-model-override to bypass.")

    cells: list[dict[str, Any]] = []
    if args.auto_balance:
        cells = auto_balance(args.total, args.visual, seed=args.seed)

    # Targets can come from --target flags AND/OR a --targets-from file.
    target_specs = list(args.target)
    if args.targets_from:
        for line in args.targets_from.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                target_specs.append(line)
    for spec in target_specs:
        cell = parse_target(spec)
        cell["with_visual"] = args.visual and cell["topic"] in VISUAL_ARCHETYPES
        # Use the canonical topic→area mapping when known; fall back to
        # cross-cutting only for unknown topics.
        cell["competency_area"] = TOPIC_TO_AREA.get(cell["topic"], "cross-cutting")
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
        result = generate_batch(batch, args.model, args.dry_run, output_dir,
                                prompt_variant=args.prompt_variant)
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
