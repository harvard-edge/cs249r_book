#!/usr/bin/env python3
"""Parallel zone gap filler — generates questions targeting specific topic×zone holes.

Uses gemini CLI in parallel to fill the ikigai coverage gaps identified by
the coverage analysis. Generates questions with explicit topic, zone, track,
and level targeting.

Usage:
    python3 fill_zone_gaps.py --zone analyze --budget 50 --workers 8
    python3 fill_zone_gaps.py --zone realization --budget 30
    python3 fill_zone_gaps.py --zone mastery --budget 30
    python3 fill_zone_gaps.py --auto --budget 100         # Auto-fill worst gaps
    python3 fill_zone_gaps.py --dry-run                   # Show plan only
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
from datetime import datetime
from pathlib import Path

import yaml

VAULT = Path(__file__).resolve().parent.parent
CORPUS_PATH = VAULT / "corpus.json"
TAXONOMY_PATH = VAULT / "schema" / "taxonomy_data.yaml"
ZONES_PATH = VAULT / "schema" / "zones.py"

MODEL = "gemini-2.5-flash"

# Zone descriptions for prompting
ZONE_PROMPTS = {
    "recall": "Test pure RECALL: ask the candidate to retrieve a fact, definition, or hardware specification from memory. The answer is a single fact, not a calculation or analysis. Example: 'What is the HBM bandwidth of an H100?'",
    "analyze": "Test pure ANALYZE: give the candidate a system behavior and ask them to explain WHY it happens through tradeoff reasoning. No recall of specs needed (provide them). No design or calculation needed. Example: 'Given that GPU X has these specs [provided], why does utilization collapse at batch size 1?'",
    "design": "Test pure DESIGN: ask the candidate to architect a system or make architecture decisions from requirements. Provide all necessary specs. No calculation needed. Example: 'Architect a KV-cache paging system for variable-length requests.'",
    "implement": "Test pure IMPLEMENT: ask the candidate to write a formula, produce a concrete number, or build something specific. Provide context so they don't need to recall specs. Example: 'Given these specs [provided], write the formula for KV-cache memory.'",
    "diagnosis": "Test DIAGNOSIS (Recall + Analyze): present a system symptom and ask the candidate to identify the root cause. They must recall hardware specs AND reason about what explains the symptom. Example: 'MFU is 40% on H100 during decode — diagnose it.'",
    "specification": "Test SPECIFICATION (Recall + Design): give requirements and ask the candidate to design a system that meets them. They must recall hardware capabilities AND translate constraints into architecture. Example: 'Design a serving stack with P99 < 100ms for a 70B model.'",
    "fluency": "Test FLUENCY (Recall + Implement): ask the candidate to do napkin math from memory. They must recall the formula and specs AND execute the arithmetic. Example: 'How much VRAM does Llama-70B need in FP16? Show the math.'",
    "evaluation": "Test EVALUATION (Analyze + Design): present two or more architectural alternatives and ask which is better for a given scenario. Provide specs. The candidate must analyze tradeoffs AND make a design decision. Example: 'MoE vs dense for this latency budget — which and why?'",
    "realization": "Test REALIZATION (Design + Implement): the candidate has already decided on an architecture — now ask them to size it concretely. They must turn the design into specific numbers. Example: 'You chose pipeline parallelism. Size the micro-batches and estimate bubble overhead.'",
    "optimization": "Test OPTIMIZATION (Analyze + Implement): present a system bottleneck and ask the candidate to both diagnose it AND quantify the fix. Example: 'Inference is memory-bound at 30% MFU. Walk through the optimization ladder with estimated speedup at each step.'",
    "mastery": "Test MASTERY (all four skills): present a complex scenario requiring the candidate to recall specs, analyze the problem, design a solution, AND size it with napkin math. Example: 'Your serving cluster TTFT spiked to 4s. Diagnose, design a fix, size the resources, estimate the cost.'",
}

# Level descriptions
LEVEL_PROMPTS = {
    "L1": "L1 (Remember/Intern): Retrieve a stored fact or definition.",
    "L2": "L2 (Understand/Intern): Explain a concept or relationship.",
    "L3": "L3 (Apply/New Grad): Apply a formula or procedure to a specific scenario.",
    "L4": "L4 (Analyze/Mid-level): Break down a system to identify components and interactions.",
    "L5": "L5 (Evaluate/Senior): Judge tradeoffs and make system-level decisions.",
    "L6+": "L6+ (Create/Staff+): Synthesize a novel architecture from first principles.",
}

# Hardware reference constants
HARDWARE_REF = """
Hardware Reference (use these exact specs in questions):
- H100: 80 GB HBM3, 3.35 TB/s bandwidth, 989 TFLOPS FP16, 700W TDP, ridge point ~295 FLOPs/Byte
- A100: 80 GB HBM2e, 2.0 TB/s bandwidth, 312 TFLOPS FP16, 400W TDP, ridge point ~156 FLOPs/Byte
- Jetson Orin: 32 GB LPDDR5, 275 TOPS INT8, 60W TDP
- Cortex-M4: 168 MHz, 256 KB SRAM, no FPU, ~100 MFLOPS
- Cortex-M7: 480 MHz, 512 KB SRAM, FPU, ~300 MFLOPS
- NVLink 4.0: 900 GB/s per GPU
- PCIe Gen5 x16: 64 GB/s bidirectional, 32 GB/s unidirectional
- InfiniBand NDR: 400 Gb/s = 50 GB/s per port
"""


def load_gaps(target_zone: str | None = None) -> list[dict]:
    """Identify topic×zone gaps and return generation jobs."""
    corpus = json.load(open(CORPUS_PATH))
    pub = [q for q in corpus if q.get("status", "published") == "published"]
    taxonomy = yaml.safe_load(open(TAXONOMY_PATH).read())

    # Build coverage matrix
    tz = defaultdict(int)
    for q in pub:
        tz[(q.get("topic", ""), q.get("zone", ""))] += 1

    topics = taxonomy["topics"]
    zones = list(ZONE_PROMPTS.keys())

    jobs = []
    for t in topics:
        tid = t["id"]
        area = t["area"]
        tracks = t.get("tracks", ["cloud", "edge", "mobile", "tinyml"])
        description = t.get("description", "")

        for z in zones:
            if target_zone and z != target_zone:
                continue
            count = tz.get((tid, z), 0)
            if count >= 3:  # Already have enough
                continue

            # Pick appropriate level for this zone
            zone_levels = {
                "recall": ["L1", "L2"],
                "analyze": ["L3", "L4"],
                "design": ["L5", "L6+"],
                "implement": ["L3", "L4"],
                "diagnosis": ["L3", "L4"],
                "specification": ["L4", "L5"],
                "fluency": ["L2", "L3"],
                "evaluation": ["L4", "L5"],
                "realization": ["L5", "L6+"],
                "optimization": ["L4", "L5"],
                "mastery": ["L6+"],
            }
            level = zone_levels[z][0]  # Pick the primary level
            track = tracks[0] if tracks else "cloud"

            deficit = max(0, 3 - count)
            jobs.append({
                "topic": tid,
                "topic_name": t["name"],
                "topic_description": description,
                "area": area,
                "zone": z,
                "track": track,
                "level": level,
                "current": count,
                "deficit": deficit,
            })

    # Sort: biggest gaps first (0 count before 1 before 2)
    jobs.sort(key=lambda j: (j["current"], j["zone"], j["topic"]))
    return jobs


def build_prompt(job: dict) -> str:
    """Build the generation prompt for one question."""
    return f"""You are an expert ML systems engineer creating interview questions.

Generate exactly 1 interview question with the following classification:

TOPIC: {job["topic_name"]}
  Description: {job["topic_description"]}
  Competency Area: {job["area"]}

ZONE: {job["zone"]}
  {ZONE_PROMPTS[job["zone"]]}

TRACK: {job["track"]}
LEVEL: {LEVEL_PROMPTS[job["level"]]}

{HARDWARE_REF}

Output EXACTLY this JSON format (no markdown fences, no explanation, ONLY the JSON):
{{
  "title": "Short descriptive title (5-10 words)",
  "track": "{job["track"]}",
  "level": "{job["level"]}",
  "topic": "{job["topic"]}",
  "zone": "{job["zone"]}",
  "competency_area": "{job["area"]}",
  "bloom_level": "<remember|understand|apply|analyze|evaluate|create>",
  "scenario": "The interview scenario (3-5 sentences). Be specific with numbers and hardware specs.",
  "details": {{
    "common_mistake": "What candidates typically get wrong (2-3 sentences).",
    "realistic_solution": "The correct approach with reasoning (3-5 sentences).",
    "napkin_math": "Step-by-step calculation with real numbers. Show the formula and arithmetic."
  }}
}}

Rules:
- The scenario MUST require the specific cognitive skills of the {job["zone"]} zone
- Use REAL hardware specs from the reference above
- The napkin_math must contain actual calculations with numbers
- The title must be unique and descriptive
- The scenario must be realistic (something that happens in industry)
"""


def call_gemini(prompt: str, model: str = MODEL, retries: int = 2) -> str | None:
    """Call gemini CLI."""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ["gemini", "-m", model, "-o", "text"],
                input=prompt, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return None
            text = result.stdout.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return text.strip()
        except subprocess.TimeoutExpired:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:
            print(f"  Error: {e}")
            return None


def generate_one(job_idx: int, job: dict, model: str = MODEL) -> dict | None:
    """Generate one question for a specific gap."""
    prompt = build_prompt(job)
    text = call_gemini(prompt, model=model)
    if not text:
        return None

    try:
        q = json.loads(text)
        # Generate a unique ID
        q["id"] = f"{job['track']}-gen-{job['zone'][:4]}-{job_idx:04d}"
        q["scope"] = ""
        q["validated"] = False
        q["validation_status"] = "pending"
        q["validation_issues"] = []
        q["validation_model"] = None
        q["validation_date"] = None
        q["chain_ids"] = None
        q["chain_positions"] = None
        return q
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fill ikigai zone gaps")
    parser.add_argument("--zone", help="Target a specific zone (e.g., analyze, realization)")
    parser.add_argument("--auto", action="store_true", help="Auto-fill worst gaps across all zones")
    parser.add_argument("--budget", type=int, default=50, help="Max questions to generate (default: 50)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--model", default=MODEL, help=f"Model to use (default: {MODEL})")
    args = parser.parse_args()

    model = args.model

    if not args.zone and not args.auto:
        print("Usage: --zone <zone_name> or --auto")
        print(f"Zones: {', '.join(ZONE_PROMPTS.keys())}")
        sys.exit(1)

    target_zone = args.zone if not args.auto else None
    jobs = load_gaps(target_zone)

    if not jobs:
        print("No gaps found!")
        return

    # Limit to budget
    jobs = jobs[:args.budget]

    print(f"{'=' * 60}")
    print(f"StaffML Zone Gap Filler")
    print(f"{'=' * 60}")
    print(f"  Target zone: {args.zone or 'AUTO (worst gaps first)'}")
    print(f"  Budget: {args.budget}")
    print(f"  Workers: {args.workers}")
    print(f"  Model: {MODEL}")
    print(f"  Jobs: {len(jobs)}")
    print()

    # Show plan
    zone_counts = Counter(j["zone"] for j in jobs)
    print("  Generation plan by zone:")
    for z, cnt in zone_counts.most_common():
        print(f"    {z:15s} {cnt:>4d} questions")
    print()

    if args.dry_run:
        print("  [DRY RUN] No questions generated.")
        print(f"\n  Top 20 jobs:")
        for j in jobs[:20]:
            print(f"    {j['topic']:35s} × {j['zone']:15s} ({j['track']}/{j['level']}) current={j['current']}")
        return

    # Generate in parallel
    generated = []
    errors = 0

    print(f"  Generating {len(jobs)} questions with {args.workers} workers...\n")
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, i, job, model): (i, job)
            for i, job in enumerate(jobs)
        }

        for future in as_completed(futures):
            i, job = futures[future]
            try:
                q = future.result()
                if q:
                    generated.append(q)
                    done = len(generated) + errors
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(jobs)} @ {rate:.1f}/s] "
                          f"✓ {job['topic'][:30]:30s} × {job['zone']:12s}")
                else:
                    errors += 1
                    print(f"  ✗ Failed: {job['topic'][:30]} × {job['zone']}")
            except Exception as e:
                errors += 1
                print(f"  ✗ Error: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Generated: {len(generated)}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.0f}s ({len(generated)/elapsed:.1f} q/s)")

    if not generated:
        print("  No questions generated. Check gemini CLI access.")
        return

    # Append to corpus
    corpus = json.load(open(CORPUS_PATH))
    corpus.extend(generated)

    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n  Appended {len(generated)} questions to corpus.json")
    print(f"  New total: {len(corpus)} questions")

    # Show new zone distribution
    pub = [q for q in corpus if q.get("status", "published") == "published"]
    zones = Counter(q.get("zone", "") for q in pub)
    print(f"\n  Updated zone distribution:")
    for z, cnt in sorted(zones.items(), key=lambda x: -x[1]):
        print(f"    {z:15s} {cnt:>5d}")


if __name__ == "__main__":
    main()
