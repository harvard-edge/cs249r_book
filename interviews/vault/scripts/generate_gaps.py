#!/usr/bin/env python3
"""
Gap-fill generator for underfilled StaffML corpus cells.

Reads corpus.json, identifies all (track, level, competency_area) cells with
fewer than 3 questions, then generates the missing questions using either:
  - Gemini 2.5 Flash via the `gemini` CLI (default)
  - Claude Opus 4.6 via the Anthropic API (if ANTHROPIC_API_KEY is set)

Outputs are:
  1. Appended to the correct source markdown file
  2. Written as JSON to _generated_gaps.json for corpus rebuild

Usage:
    python3 generate_gaps.py                     # Gemini Flash (default)
    python3 generate_gaps.py --model opus        # Claude Opus 4.6
    python3 generate_gaps.py --dry-run           # Show plan, don't generate
    python3 generate_gaps.py --workers 4         # Control parallelism
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "corpus.json"
NUMBERS_PATH = BASE_DIR / "NUMBERS.md"
OUTPUT_JSON = BASE_DIR / "_generated_gaps.json"
TARGET_PER_CELL = 3

# Badge metadata for markdown rendering
LEVEL_META = {
    "L1": {"label": "L1_Foundation", "color": "brightgreen", "alt": "Level 1"},
    "L2": {"label": "L2_Analytical", "color": "blue", "alt": "Level 2"},
    "L3": {"label": "L3_Junior", "color": "brightgreen", "alt": "Level 1"},
    "L4": {"label": "L4_Mid", "color": "blue", "alt": "Level 2"},
    "L5": {"label": "L5_Senior", "color": "yellow", "alt": "Level 3"},
    "L6+": {"label": "L6%2B_Staff", "color": "red", "alt": "Level 4"},
}

# Map competency area to representative topic tags for prompting
AREA_TO_TAGS = {
    "compute": ["roofline", "arithmetic-intensity", "compute-bound", "memory-bound"],
    "memory": ["memory-hierarchy", "kv-cache", "activation-memory", "memory-bandwidth"],
    "precision": ["quantization", "mixed-precision", "calibration"],
    "architecture": ["scaling-laws", "attention", "transformers", "depthwise-separable", "early-exit"],
    "latency": ["latency", "throughput", "ttft", "batching", "real-time"],
    "power": ["power", "thermal", "tops-w", "duty-cycle", "battery", "cooling"],
    "optimization": ["pruning", "distillation", "operator-fusion", "flash-attention", "compilation"],
    "parallelism": ["data-parallelism", "tensor-parallelism", "pipeline-parallelism", "fsdp"],
    "networking": ["interconnect", "network-topology", "congestion", "bus-protocol", "wireless"],
    "deployment": ["serving", "deployment", "rollout", "ota", "firmware"],
    "reliability": ["monitoring", "drift", "fault-tolerance", "watchdog", "checkpoint"],
    "data": ["data-pipeline", "data-quality", "training-serving-skew", "sensor-pipeline", "streaming-data"],
    "cross-cutting": ["economics", "tco", "cost-per-query", "security", "privacy"],
}

# Map track to preferred file for each competency area
# Picks the most thematically appropriate file for new questions
TRACK_FILE_MAP = {
    "cloud": {
        "compute":       "01_single_machine.md",
        "memory":        "01_single_machine.md",
        "precision":     "01_single_machine.md",
        "architecture":  "01_single_machine.md",
        "latency":       "03_serving_stack.md",
        "power":         "04_production_ops.md",
        "optimization":  "01_single_machine.md",
        "parallelism":   "02_distributed_systems.md",
        "networking":    "02_distributed_systems.md",
        "deployment":    "03_serving_stack.md",
        "reliability":   "04_production_ops.md",
        "data":          "01_single_machine.md",
        "cross-cutting": "04_production_ops.md",
    },
    "edge": {
        "compute":       "01_hardware_platform.md",
        "memory":        "01_hardware_platform.md",
        "precision":     "01_hardware_platform.md",
        "architecture":  "01_hardware_platform.md",
        "latency":       "02_realtime_pipeline.md",
        "power":         "01_hardware_platform.md",
        "optimization":  "02_realtime_pipeline.md",
        "parallelism":   "01_hardware_platform.md",
        "networking":    "03_deployed_system.md",
        "deployment":    "03_deployed_system.md",
        "reliability":   "03_deployed_system.md",
        "data":          "02_realtime_pipeline.md",
        "cross-cutting": "03_deployed_system.md",
    },
    "mobile": {
        "compute":       "01_device_hardware.md",
        "memory":        "01_device_hardware.md",
        "precision":     "01_device_hardware.md",
        "architecture":  "01_device_hardware.md",
        "latency":       "02_app_experience.md",
        "power":         "01_device_hardware.md",
        "optimization":  "01_device_hardware.md",
        "parallelism":   "01_device_hardware.md",
        "networking":    "01_device_hardware.md",
        "deployment":    "03_ship_and_update.md",
        "reliability":   "03_ship_and_update.md",
        "data":          "02_app_experience.md",
        "cross-cutting": "03_ship_and_update.md",
    },
    "tinyml": {
        "compute":       "01_microcontroller.md",
        "memory":        "01_microcontroller.md",
        "precision":     "01_microcontroller.md",
        "architecture":  "01_microcontroller.md",
        "latency":       "02_sensing_pipeline.md",
        "power":         "01_microcontroller.md",
        "optimization":  "01_microcontroller.md",
        "parallelism":   "01_microcontroller.md",
        "networking":    "03_deployed_device.md",
        "deployment":    "03_deployed_device.md",
        "reliability":   "03_deployed_device.md",
        "data":          "02_sensing_pipeline.md",
        "cross-cutting": "03_deployed_device.md",
    },
}

# Track-specific hardware context for prompt tuning
TRACK_CONTEXT = {
    "cloud": (
        "Cloud track: NVIDIA H100/A100 GPUs, HBM3 memory, NVLink/InfiniBand "
        "interconnects, data center power/cooling, multi-GPU servers. "
        "Focus on large model training and high-throughput serving."
    ),
    "edge": (
        "Edge track: NVIDIA Jetson Orin, Qualcomm RB5, Google Coral TPU, "
        "multi-core ARM CPUs with GPU/NPU accelerators, 10-30W power envelopes. "
        "Focus on real-time inference for robotics, autonomous vehicles, drones."
    ),
    "mobile": (
        "Mobile track: Apple A-series/M-series, Qualcomm Snapdragon with Hexagon NPU, "
        "Samsung Exynos, MediaTek Dimensity. 3-5W thermal envelope, battery life critical. "
        "Focus on on-device ML for apps: camera, NLP, recommender systems."
    ),
    "tinyml": (
        "TinyML track: ARM Cortex-M0 to M7 MCUs, 256KB-2MB SRAM, 1-16MB Flash, "
        "no OS, bare-metal C. Power budget: microwatts to milliwatts. "
        "Focus on keyword spotting, anomaly detection, sensor fusion on MCUs."
    ),
}

# Bloom's level cognitive descriptors
BLOOM_DESCRIPTORS = {
    "L1": "Remember — pure recall of facts, specs, ratios. Direct questions.",
    "L2": "Understand — single-variable calculations, explain why, compare two things.",
    "L3": "Apply — use a formula in a new situation, diagnose a described scenario.",
    "L4": "Analyze — multi-step debugging, identify root cause from symptoms.",
    "L5": "Evaluate — compare two architectures, justify a design choice with trade-offs.",
    "L6+": "Create — design a system from scratch, propose a novel solution to a constraint.",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GapCell:
    """A single underfilled cell in the corpus matrix."""
    track: str
    level: str
    area: str
    current_count: int
    needed: int  # questions to generate

    @property
    def key(self) -> str:
        return f"{self.track}/{self.level}/{self.area}"


# ---------------------------------------------------------------------------
# Corpus analysis
# ---------------------------------------------------------------------------

def find_gaps(corpus_path: Path = CORPUS_PATH, target: int = TARGET_PER_CELL) -> list[GapCell]:
    """Identify all cells with fewer than `target` questions."""
    with open(corpus_path) as f:
        corpus = json.load(f)

    cells = Counter()
    for q in corpus:
        track = q.get("track", "")
        level = q.get("level", "")
        area = q.get("competency_area", "")
        if track and track != "global" and level and area:
            cells[(track, level, area)] += 1

    gaps = []
    for (track, level, area), count in sorted(cells.items()):
        if count < target:
            gaps.append(GapCell(
                track=track,
                level=level,
                area=area,
                current_count=count,
                needed=target - count,
            ))
    return gaps


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _load_hardware_context() -> str:
    """Load NUMBERS.md for hardware constants."""
    if NUMBERS_PATH.exists():
        return NUMBERS_PATH.read_text(encoding="utf-8")
    return "(Hardware constants not available — use conservative estimates)"


def build_prompt(gap: GapCell, hardware_ctx: str) -> str:
    """Build a targeted generation prompt for one gap cell."""
    tags = AREA_TO_TAGS.get(gap.area, [gap.area])
    tag_str = ", ".join(tags)
    track_ctx = TRACK_CONTEXT.get(gap.track, "")
    bloom_desc = BLOOM_DESCRIPTORS.get(gap.level, "")

    # Pick a few suggested topic tags for variety
    import random
    suggested_tags = random.sample(tags, min(2, len(tags)))
    suggested_str = " or ".join(f"`{t}`" for t in suggested_tags)

    return f"""You are a world-class ML systems interview question writer for the StaffML platform.
Your questions are used by engineers preparing for Staff/Principal ML Systems Engineer roles.
Every question must be grounded in real hardware physics and quantitative reasoning.

## ABSOLUTE REQUIREMENTS
1. Every scenario must be REALISTIC — something that actually happens in production
2. Every napkin math section must use REAL hardware specs from the constants below
3. Every common mistake must be a REAL misconception that engineers actually have
4. Every calculation must be arithmetically correct — you will be verified
5. The question must test the SPECIFIC competency area: **{gap.area}**
6. Use topic tags from: {tag_str}

## HARDWARE CONSTANTS (Source of Truth)
{hardware_ctx}

## TRACK CONTEXT
{track_ctx}

## COGNITIVE LEVEL
**{gap.level}**: {bloom_desc}

## YOUR TASK
Generate exactly {gap.needed} interview question(s) for:
- **Track:** {gap.track}
- **Competency area:** {gap.area}
- **Target level:** {gap.level}
- **Suggested topic tags:** {suggested_str}

Each question must be DISTINCT — test different aspects of {gap.area} at the {gap.level} level.
Do NOT duplicate concepts already common in the corpus (roofline basics, simple memory calculations).
Focus on {gap.track}-specific scenarios that a {gap.level}-level engineer would face.

## OUTPUT FORMAT
Return a JSON array. Each object must have these fields:
```json
[
  {{
    "level": "{gap.level}",
    "title": "The [Evocative Title]",
    "topic": "kebab-case-topic-from-taxonomy",
    "track": "{gap.track}",
    "scenario": "The full interviewer question text...",
    "common_mistake": "What engineers typically get wrong and why...",
    "realistic_solution": "The correct answer with full explanation...",
    "napkin_math": "Step-by-step quantitative reasoning with real numbers...",
    "resources": []
  }}
]
```

IMPORTANT:
- `napkin_math` is REQUIRED for ALL levels
- Return ONLY valid JSON, no markdown wrapping, no ```json fences
- Each question must be self-contained and test {gap.area} specifically
"""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str = "gemini-3.1-pro-preview") -> str:
    """Call Gemini via the locally-authenticated CLI."""
    result = subprocess.run(
        ["gemini", "--model", model, "-"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI failed (exit {result.returncode}): {result.stderr[:500]}")
    return result.stdout


def call_opus(prompt: str, api_key: str) -> str:
    """Call Claude Opus 4.6 via the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("[ERROR] pip install anthropic required for Opus backend", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Parsing LLM output
# ---------------------------------------------------------------------------

def parse_llm_output(raw: str) -> list[dict]:
    """Parse LLM response into a list of question dicts."""
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3].strip()
    # Also strip ```json prefix
    if text.startswith("json\n"):
        text = text[4:].strip()

    # Try to find JSON array in the output
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed: {e}", file=sys.stderr)
        print(f"  [DEBUG] First 300 chars: {text[:300]}", file=sys.stderr)
        return []

    if isinstance(questions, dict):
        questions = [questions]

    return questions


# ---------------------------------------------------------------------------
# Markdown rendering (matches render.py format exactly)
# ---------------------------------------------------------------------------

def render_question_md(q: dict) -> str:
    """Render a question dict to the exact markdown <details> format."""
    level = q.get("level", "L3")
    meta = LEVEL_META.get(level, LEVEL_META["L3"])
    badge_label = meta["label"]
    badge_color = meta["color"]
    badge_alt = meta["alt"]

    title = q.get("title", "Untitled")
    topic = q.get("topic", "unknown")
    scenario = q.get("scenario", "")
    common_mistake = q.get("common_mistake", "")
    realistic_solution = q.get("realistic_solution", "")
    napkin_math = q.get("napkin_math", "")
    resources = q.get("resources") or []

    topic_tags = " ".join(f"<code>{t.strip()}</code>" for t in topic.split(","))

    inner = []
    inner.append(f"  **Common Mistake:** {common_mistake}")
    inner.append("")
    inner.append(f"  **Realistic Solution:** {realistic_solution}")

    if napkin_math:
        inner.append("")
        inner.append(f"  > **Napkin Math:** {napkin_math}")

    if resources:
        inner.append("")
        inner.append("  📖 **Resources:**")
        for r in resources:
            name = (r.get("name") or "").strip()
            url = (r.get("url") or "").strip()
            if name and url:
                inner.append(f"  - [{name}]({url})")

    inner_content = "\n".join(inner)

    return f"""<details>
<summary><b><img src="https://img.shields.io/badge/Level-{badge_label}-{badge_color}?style=flat-square" alt="{badge_alt}" align="center"> {title}</b> · {topic_tags}</summary>

- **Interviewer:** "{scenario}"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

{inner_content}
  </details>
</details>"""


# ---------------------------------------------------------------------------
# File appending
# ---------------------------------------------------------------------------

def get_target_file(gap: GapCell) -> Path:
    """Determine which markdown file to append to for a given gap."""
    track_map = TRACK_FILE_MAP.get(gap.track, {})
    filename = track_map.get(gap.area, "01_" + gap.track + ".md")
    return BASE_DIR / gap.track / filename


def append_to_markdown(gap: GapCell, questions: list[dict]) -> Path:
    """Append rendered questions to the target markdown file."""
    target = get_target_file(gap)
    if not target.exists():
        print(f"  [WARN] Target file not found: {target}", file=sys.stderr)
        return target

    md_blocks = []
    for q in questions:
        md_blocks.append(render_question_md(q))

    separator = f"\n\n<!-- === Generated: {gap.key} === -->\n\n"
    content = separator + "\n\n".join(md_blocks) + "\n"

    with open(target, "a", encoding="utf-8") as f:
        f.write(content)

    return target


# ---------------------------------------------------------------------------
# Worker function for parallel generation
# ---------------------------------------------------------------------------

def process_gap(
    gap: GapCell,
    hardware_ctx: str,
    backend: str,
    api_key: Optional[str],
    model: str,
) -> dict:
    """Generate questions for one gap cell. Returns a result dict."""
    result = {
        "cell": gap.key,
        "needed": gap.needed,
        "generated": 0,
        "questions": [],
        "file": str(get_target_file(gap)),
        "error": None,
    }

    try:
        prompt = build_prompt(gap, hardware_ctx)

        if backend == "opus":
            raw = call_opus(prompt, api_key)
        else:
            raw = call_gemini(prompt, model=model)

        questions = parse_llm_output(raw)

        if not questions:
            result["error"] = "No questions parsed from LLM output"
            return result

        # Ensure each question has the right track/level
        for q in questions:
            q["track"] = gap.track
            q["level"] = gap.level
            q.setdefault("competency_area", gap.area)

        # Append to markdown
        target = append_to_markdown(gap, questions)
        result["generated"] = len(questions)
        result["questions"] = questions
        result["file"] = str(target)

        print(f"  [OK] {gap.key}: generated {len(questions)}/{gap.needed} -> {target.name}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERR] {gap.key}: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fill underfilled StaffML corpus cells")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--model", choices=["flash", "opus"], default="flash",
                        help="LLM backend: flash=Gemini 2.5 Flash, opus=Claude Opus 4.6")
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-preview",
                        help="Specific Gemini model name (default: gemini-3.1-pro-preview)")
    parser.add_argument("--target", type=int, default=TARGET_PER_CELL,
                        help="Target questions per cell (default: 3)")
    args = parser.parse_args()

    target = args.target

    # --- Find gaps ---
    print("=" * 60)
    print("StaffML Gap-Fill Generator")
    print("=" * 60)

    gaps = find_gaps(target=target)
    total_needed = sum(g.needed for g in gaps)

    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Underfilled cells: {len(gaps)}")
    print(f"Total questions to generate: {total_needed}")
    print(f"Backend: {'Claude Opus 4.6 (Anthropic API)' if args.model == 'opus' else f'Gemini ({args.gemini_model})'}")
    print(f"Workers: {args.workers}")
    print()

    # --- Breakdown by track ---
    by_track = defaultdict(list)
    for g in gaps:
        by_track[g.track].append(g)

    for track in sorted(by_track):
        track_gaps = by_track[track]
        track_needed = sum(g.needed for g in track_gaps)
        print(f"  {track}: {len(track_gaps)} cells, {track_needed} questions needed")
        for g in track_gaps:
            print(f"    {g.level}/{g.area}: {g.current_count} -> {target} (need {g.needed})")
    print()

    # --- Estimate time ---
    # ~15s per Gemini Flash call, ~30s per Opus call
    time_per_call = 30 if args.model == "opus" else 15
    batches = (len(gaps) + args.workers - 1) // args.workers
    est_seconds = batches * time_per_call
    est_minutes = est_seconds / 60
    print(f"Estimated time: ~{est_minutes:.1f} minutes ({batches} batches x {time_per_call}s)")
    print()

    if args.dry_run:
        print("[DRY RUN] No questions generated.")
        return

    # --- Validate backend ---
    api_key = None
    if args.model == "opus":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[ERROR] ANTHROPIC_API_KEY not set. Required for Opus backend.", file=sys.stderr)
            sys.exit(1)

    # --- Load hardware context ---
    hardware_ctx = _load_hardware_context()

    # --- Generate in parallel ---
    print("Generating...")
    start_time = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_gap, gap, hardware_ctx, args.model, api_key, args.gemini_model
            ): gap
            for gap in gaps
        }

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    elapsed = time.time() - start_time

    # --- Save JSON output ---
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Summary ---
    total_generated = sum(r["generated"] for r in all_results)
    errors = [r for r in all_results if r["error"]]

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Cells processed: {len(all_results)}")
    print(f"  Questions generated: {total_generated}/{total_needed}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_JSON}")
    print()

    if errors:
        print("ERRORS:")
        for r in errors:
            print(f"  {r['cell']}: {r['error']}")
        print()

    # --- Files modified ---
    modified = set(r["file"] for r in all_results if r["generated"] > 0)
    if modified:
        print("Files modified:")
        for f in sorted(modified):
            print(f"  {f}")
        print()
        print("Next steps:")
        print("  1. Review the generated questions in each file")
        print("  2. Run: python3 build_corpus.py   (rebuild corpus.json)")
        print("  3. Verify: python3 -m engine validate")


if __name__ == "__main__":
    main()
