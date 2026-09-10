#!/usr/bin/env python3
"""Fast parallel batch generator — fills ALL topic×track×zone gaps.

Reads pre-computed jobs from /tmp/staffml_jobs.json and spawns
gemini CLI calls in parallel, writing results to a temp file.

Usage:
    python3 generate_batch.py --workers 40 --output /tmp/batch_all.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PLATFORMS = {
    "cloud": [
        ("NVIDIA H100", "80 GB HBM3, 495 TFLOPS FP16 dense, 3.35 TB/s, 700W"),
        ("Google TPU v5e", "16 GB HBM, 197 TFLOPS BF16, 1.6 TB/s"),
        ("AMD MI300X", "192 GB HBM3, 1300 TFLOPS FP16 sparse, 5.3 TB/s"),
        ("NVIDIA A100", "80 GB HBM2e, 312 TFLOPS FP16, 2.0 TB/s, 400W"),
    ],
    "edge": [
        ("NVIDIA Jetson Orin", "32 GB LPDDR5, 275 TOPS INT8, 60W"),
        ("Hailo-8", "26 TOPS INT8, 2.5W dataflow accelerator"),
        ("Google Coral Edge TPU", "4 TOPS INT8, 2W, INT8 only"),
        ("Qualcomm Cloud AI 100", "32 GB LPDDR4x, 400 TOPS INT8, 75W"),
    ],
    "mobile": [
        ("Apple A17 Pro", "35 TOPS Neural Engine, ~5W, 8 GB unified"),
        ("Snapdragon 8 Gen 3 Hexagon NPU", "45 TOPS INT8, 12-16 GB LPDDR5X"),
        ("Google Tensor G3", "7.5 TOPS on-device TPU, 12 GB LPDDR5X"),
        ("Samsung Exynos 2400 NPU", "34.7 TOPS, 12 GB LPDDR5X"),
    ],
    "tinyml": [
        ("ARM Cortex-M4 STM32F4", "168 MHz, 256 KB SRAM, no FPU"),
        ("ESP32-S3", "240 MHz dual-core, 512 KB SRAM, 8 MB PSRAM"),
        ("Cortex-M7 + Ethos-U55", "480 MHz, 512 KB SRAM, 128-512 MAC/cycle NPU"),
        ("Nordic nRF5340", "128 MHz, 256 KB SRAM, 1 MB flash, BLE"),
    ],
    "global": [
        ("Generic", "Cross-platform principles"),
    ],
}

ZONE_HINT = {
    "recall": "retrieve a fact or spec",
    "analyze": "explain WHY a system behaves this way (provide specs)",
    "design": "architect a system from requirements",
    "implement": "produce a number or formula",
    "diagnosis": "identify root cause from symptoms",
    "specification": "design a system meeting quantitative constraints",
    "fluency": "do napkin math from memory",
    "evaluation": "compare two architectures (provide specs)",
    "realization": "size a chosen architecture concretely",
    "optimization": "diagnose bottleneck AND quantify the fix",
    "mastery": "recall specs + analyze + design + size (all four skills)",
}

LEVEL_MAP = {
    "recall": "L2", "analyze": "L4", "design": "L5", "implement": "L3",
    "diagnosis": "L4", "specification": "L5", "fluency": "L3",
    "evaluation": "L5", "realization": "L6+", "optimization": "L4", "mastery": "L6+",
}

MODEL = "gemini-3.1-pro-preview"


def gen_one(idx, job):
    plats = PLATFORMS.get(job["track"], PLATFORMS["global"])
    pname, pspecs = plats[idx % len(plats)]
    level = LEVEL_MAP.get(job["zone"], "L4")

    prompt = f"""Generate 1 ML systems interview question. Output ONLY valid JSON.
Topic: {job['name']} - {job['desc']}
Zone: {job['zone']} ({ZONE_HINT[job['zone']]})
Track: {job['track']} | Level: {level} | Area: {job['area']}
Platform: {pname} ({pspecs})
The scenario MUST reference {pname} with real specs.
JSON format: {{"title":"...","track":"{job['track']}","level":"{level}","topic":"{job['topic']}","zone":"{job['zone']}","competency_area":"{job['area']}","bloom_level":"analyze","scenario":"...","details":{{"common_mistake":"...","realistic_solution":"...","napkin_math":"..."}}}}"""

    try:
        r = subprocess.run(
            ["gemini", "-m", MODEL, "-o", "text"],
            input=prompt, capture_output=True, text=True, timeout=90,
        )
        if r.returncode != 0:
            return None
        text = r.stdout.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        q = json.loads(text.strip())
        q["id"] = f"{job['track']}-fill-{idx:05d}"
        q["scope"] = ""
        q["validated"] = False
        q["validation_status"] = "pending"
        q["validation_issues"] = []
        q["validation_model"] = None
        q["validation_date"] = None
        q["chain_ids"] = None
        q["chain_positions"] = None
        return q
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--output", default="/tmp/batch_all.json")
    parser.add_argument("--jobs", default="/tmp/staffml_jobs.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit jobs (0=all)")
    args = parser.parse_args()

    jobs = json.load(open(args.jobs))
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"Jobs: {len(jobs)}, Workers: {args.workers}, Model: {MODEL}")

    generated = []
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, i, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futs):
            q = f.result()
            if q:
                generated.append(q)
            else:
                errors += 1
            done = len(generated) + errors
            if done % 100 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(jobs)}] gen={len(generated)} err={errors} ({rate:.1f}/s)")

    json.dump(generated, open(args.output, "w"), indent=2, ensure_ascii=False)
    elapsed = time.time() - start
    print(f"\nDone: {len(generated)} generated, {errors} errors, {elapsed:.0f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
