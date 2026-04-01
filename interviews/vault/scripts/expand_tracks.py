#!/usr/bin/env python3
"""Expand edge/mobile/TinyML tracks with platform-diverse questions.

Uses Opus 4.6 for generation (high quality), Gemini CLI for verification.
Cycles through hardware platforms to ensure vendor diversity.

Usage:
    python3 expand_tracks.py --track edge --budget 200 --workers 8
    python3 expand_tracks.py --track mobile --budget 200 --workers 8
    python3 expand_tracks.py --track tinyml --budget 100 --workers 8
    python3 expand_tracks.py --all --budget 500 --workers 8
    python3 expand_tracks.py --dry-run  # show plan only
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
from pathlib import Path

import yaml

VAULT = Path(__file__).resolve().parent.parent
CORPUS_PATH = VAULT / "corpus.json"
TAXONOMY_PATH = VAULT / "schema" / "taxonomy_data.yaml"

# ── Platform Specs by Track ──────────────────────────────────

PLATFORMS = {
    "cloud": [
        {
            "name": "NVIDIA H100",
            "specs": "80 GB HBM3, 495 TFLOPS FP16 dense (989 with 2:4 sparsity), 3.35 TB/s bandwidth, 700W TDP, ridge point ~148 FLOPs/Byte (dense)",
        },
        {
            "name": "Google TPU v5e",
            "specs": "16 GB HBM per chip, 197 TFLOPS BF16, 1.6 TB/s bandwidth, optimized for inference and small-to-medium training",
        },
        {
            "name": "AMD MI300X",
            "specs": "192 GB HBM3 (largest GPU memory), 1,300 TFLOPS FP16 (sparse), 5.3 TB/s bandwidth",
        },
        {
            "name": "NVIDIA A100",
            "specs": "80 GB HBM2e, 312 TFLOPS FP16, 2.0 TB/s bandwidth, 400W TDP",
        },
    ],
    "edge": [
        {
            "name": "NVIDIA Jetson Orin",
            "specs": "32 GB LPDDR5, 275 TOPS INT8, 60W TDP, Ampere GPU + DLA accelerator",
        },
        {
            "name": "Hailo-8",
            "specs": "26 TOPS INT8, 2.5W (dataflow accelerator), no local DRAM — streams from host memory",
        },
        {
            "name": "Google Coral Edge TPU",
            "specs": "4 TOPS INT8, 2W, USB/PCIe form factor, limited operator support (must quantize to INT8)",
        },
        {
            "name": "Qualcomm Cloud AI 100",
            "specs": "32 GB LPDDR4x, 400 TOPS INT8, 75W, on-premise inference accelerator",
        },
    ],
    "mobile": [
        {
            "name": "Apple A17 Pro Neural Engine",
            "specs": "35 TOPS, ~5W, 16-core Neural Engine, shared 8 GB unified memory with CPU/GPU",
        },
        {
            "name": "Qualcomm Hexagon NPU (Snapdragon 8 Gen 3)",
            "specs": "45 TOPS INT8, heterogeneous CPU+GPU+NPU, shared 12-16 GB LPDDR5X",
        },
        {
            "name": "Google Tensor G3",
            "specs": "On-device TPU, 7.5 TOPS, optimized for on-device LLM (Gemini Nano), 12 GB LPDDR5X",
        },
        {
            "name": "Samsung Exynos 2400 NPU",
            "specs": "34.7 TOPS, dual-core NPU, shared 12 GB LPDDR5X with CPU/GPU/ISP",
        },
    ],
    "tinyml": [
        {
            "name": "ARM Cortex-M4 (STM32F4)",
            "specs": "168 MHz, 256 KB SRAM, 1 MB flash, no FPU, CMSIS-NN SIMD (2x INT8 MACs/cycle)",
        },
        {
            "name": "ESP32-S3",
            "specs": "240 MHz dual-core Xtensa, 512 KB SRAM, 8 MB PSRAM, WiFi/BLE, vector extensions for INT8",
        },
        {
            "name": "ARM Cortex-M7 + Ethos-U55 (Corstone-300)",
            "specs": "480 MHz M7, 512 KB SRAM, Ethos-U55 NPU: 128-512 MAC/cycle, shares SRAM with M7",
        },
        {
            "name": "Nordic nRF5340",
            "specs": "128 MHz app core + 64 MHz net core, 256 KB SRAM, 1 MB flash, BLE 5.3, ~5mA active",
        },
    ],
}

# ── Zone Prompts ─────────────────────────────────────────────

ZONE_PROMPTS = {
    "recall": "Test pure RECALL: ask the candidate to retrieve a fact, definition, or hardware specification. The answer is a single fact, not a calculation.",
    "analyze": "Test pure ANALYZE: give the candidate a system behavior and ask them to explain WHY it happens through tradeoff reasoning. Provide all specs — no recall needed, no design needed.",
    "design": "Test pure DESIGN: ask the candidate to architect a system from requirements. Provide specs. No calculation needed.",
    "implement": "Test pure QUANTIFY: ask the candidate to write a formula or produce a number. Provide context.",
    "diagnosis": "Test DIAGNOSIS (Recall + Analyze): present a symptom and ask the candidate to identify the root cause using hardware knowledge.",
    "specification": "Test SPECIFICATION (Recall + Design): give requirements and ask the candidate to design a system that meets them.",
    "fluency": "Test FLUENCY (Recall + Quantify): ask napkin math from memory — recall the formula AND execute the arithmetic.",
    "evaluation": "Test EVALUATION (Analyze + Design): present alternatives and ask which is better. Provide specs.",
    "realization": "Test REALIZATION (Design + Quantify): candidate has chosen an architecture — now size it concretely.",
    "optimization": "Test OPTIMIZATION (Analyze + Quantify): present a bottleneck, diagnose it AND quantify the fix.",
    "mastery": "Test MASTERY (all four skills): complex scenario requiring recall, analysis, design, AND napkin math.",
}

LEVEL_PROMPTS = {
    "L1": "L1 (Remember): Retrieve a stored fact or definition.",
    "L2": "L2 (Understand): Explain a concept or relationship.",
    "L3": "L3 (Apply): Apply a formula to a specific scenario.",
    "L4": "L4 (Analyze): Break down a system to identify interactions.",
    "L5": "L5 (Evaluate): Judge tradeoffs and make system-level decisions.",
    "L6+": "L6+ (Create): Synthesize a novel architecture from first principles.",
}


def load_gaps(target_track: str | None = None) -> list[dict]:
    """Identify topic×zone×track gaps and return generation jobs."""
    corpus = json.load(open(CORPUS_PATH))
    pub = [q for q in corpus if q.get("status", "published") == "published"]
    taxonomy = yaml.safe_load(open(TAXONOMY_PATH).read())

    # Build coverage: (topic, zone, track) → count
    coverage = defaultdict(int)
    for q in pub:
        coverage[(q.get("topic", ""), q.get("zone", ""), q.get("track", ""))] += 1

    topics = taxonomy["topics"]
    zones = list(ZONE_PROMPTS.keys())

    jobs = []
    platform_idx = defaultdict(int)  # track → rotating index

    for t in topics:
        tid = t["id"]
        area = t["area"]
        tracks = t.get("tracks", ["cloud", "edge", "mobile", "tinyml"])
        description = t.get("description", "")

        for track in tracks:
            if target_track and track != target_track:
                continue
            if track in ("cloud", "global"):
                continue  # Only expand edge/mobile/tinyml

            for z in zones:
                count = coverage.get((tid, z, track), 0)
                if count >= 5:
                    continue

                # Pick level appropriate for zone
                zone_levels = {
                    "recall": "L1", "analyze": "L3", "design": "L5",
                    "implement": "L3", "diagnosis": "L4", "specification": "L4",
                    "fluency": "L3", "evaluation": "L5", "realization": "L5",
                    "optimization": "L4", "mastery": "L6+",
                }
                level = zone_levels[z]

                # Cycle through platforms
                plats = PLATFORMS[track]
                idx = platform_idx[track] % len(plats)
                platform = plats[idx]
                platform_idx[track] += 1

                jobs.append({
                    "topic": tid,
                    "topic_name": t["name"],
                    "topic_description": description,
                    "area": area,
                    "zone": z,
                    "track": track,
                    "level": level,
                    "platform": platform,
                    "current": count,
                    "deficit": max(0, 5 - count),
                })

    # Sort: zero-count first, then by track
    jobs.sort(key=lambda j: (j["current"], j["track"], j["zone"], j["topic"]))
    return jobs


def build_prompt(job: dict) -> str:
    """Build generation prompt with platform-specific context."""
    platform = job["platform"]
    return f"""You are an expert ML systems engineer creating interview questions for {job["track"]} deployment.

Generate exactly 1 interview question with the following classification:

TOPIC: {job["topic_name"]}
  Description: {job["topic_description"]}
  Competency Area: {job["area"]}

ZONE: {job["zone"]}
  {ZONE_PROMPTS[job["zone"]]}

TRACK: {job["track"]}
LEVEL: {LEVEL_PROMPTS[job["level"]]}

TARGET PLATFORM: {platform["name"]}
  Specs: {platform["specs"]}

The question MUST reference the specific platform ({platform["name"]}) and its characteristics.
Use the exact specs provided above in the napkin math.

Output EXACTLY this JSON format (no markdown fences, no explanation, ONLY the JSON):
{{
  "title": "Short descriptive title (5-10 words)",
  "track": "{job["track"]}",
  "level": "{job["level"]}",
  "topic": "{job["topic"]}",
  "zone": "{job["zone"]}",
  "competency_area": "{job["area"]}",
  "bloom_level": "<remember|understand|apply|analyze|evaluate|create>",
  "scenario": "The interview scenario (3-5 sentences). Reference {platform["name"]} specifically with real specs.",
  "details": {{
    "common_mistake": "What candidates typically get wrong (2-3 sentences).",
    "realistic_solution": "The correct approach with reasoning (3-5 sentences).",
    "napkin_math": "Step-by-step calculation with real numbers from {platform["name"]} specs."
  }}
}}

Rules:
- The scenario MUST mention {platform["name"]} by name
- Use REAL hardware specs from the platform description above
- The napkin_math must contain actual calculations with numbers
- Make the scenario realistic — something that happens in {job["track"]} deployment
"""


def call_opus(prompt: str, retries: int = 2) -> str | None:
    """Call Opus 4.6 via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    for attempt in range(retries + 1):
        try:
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return text.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Opus error: {e}")
                return None


def generate_one(job_idx: int, job: dict) -> dict | None:
    """Generate one question for a specific gap."""
    prompt = build_prompt(job)
    text = call_opus(prompt)
    if not text:
        return None

    try:
        q = json.loads(text)
        q["id"] = f"{job['track']}-exp-{job['zone'][:4]}-{job_idx:04d}"
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
    parser = argparse.ArgumentParser(description="Expand edge/mobile/TinyML tracks")
    parser.add_argument("--track", help="Target track (edge, mobile, tinyml)")
    parser.add_argument("--all", action="store_true", help="Expand all three tracks")
    parser.add_argument("--budget", type=int, default=200, help="Max questions per track")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    if not args.track and not args.all:
        print("Usage: --track <edge|mobile|tinyml> or --all")
        sys.exit(1)

    tracks_to_expand = (
        ["edge", "mobile", "tinyml"] if args.all
        else [args.track]
    )

    for track in tracks_to_expand:
        jobs = load_gaps(track)
        if not jobs:
            print(f"\n{track}: No gaps found!")
            continue

        jobs = jobs[:args.budget]

        print(f"\n{'=' * 60}")
        print(f"Track: {track.upper()}")
        print(f"{'=' * 60}")
        print(f"  Jobs: {len(jobs)}")
        print(f"  Workers: {args.workers}")

        # Show plan by zone
        zone_counts = Counter(j["zone"] for j in jobs)
        print(f"\n  By zone:")
        for z, cnt in zone_counts.most_common():
            print(f"    {z:15s} {cnt:>4d}")

        # Show platform distribution
        plat_counts = Counter(j["platform"]["name"] for j in jobs)
        print(f"\n  By platform:")
        for p, cnt in plat_counts.most_common():
            print(f"    {p:30s} {cnt:>4d}")

        if args.dry_run:
            continue

        # Generate
        generated = []
        errors = 0
        start = time.time()

        print(f"\n  Generating with Opus 4.6...\n")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_one, i, job): (i, job)
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
                        if done % 10 == 0:
                            print(f"  [{done}/{len(jobs)} @ {rate:.1f}/s] "
                                  f"{job['platform']['name'][:20]:20s} × {job['zone']:12s}")
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1

        elapsed = time.time() - start
        print(f"\n  Generated: {len(generated)}")
        print(f"  Errors: {errors}")
        print(f"  Time: {elapsed:.0f}s")

        if generated:
            corpus = json.load(open(CORPUS_PATH))
            corpus.extend(generated)
            with open(CORPUS_PATH, "w") as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  Appended to corpus.json (new total: {len(corpus)})")


if __name__ == "__main__":
    main()
