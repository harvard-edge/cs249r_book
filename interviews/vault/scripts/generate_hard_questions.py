#!/usr/bin/env python3
"""Generate ~500 hard (L4-L6+) questions to fill level distribution gaps.

Uses Gemini API (gemini-3.1-pro-preview) to generate questions across all 35
knowledge areas, targeting deficit levels. Each batch includes examples from
the existing corpus and valid concept tags for that KA.

Usage:
    source ~/.zshrc  # ensure GEMINI_API_KEY is set
    python3 generate_hard_questions.py              # Full run (~500 Qs)
    python3 generate_hard_questions.py --dry-run    # Show plan without generating
    python3 generate_hard_questions.py --ka B2      # Only generate for one KA
    python3 generate_hard_questions.py --workers 5  # Customize parallelism
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Setup ────────────────────────────────────────────────────

BASE = Path(__file__).parent
CORPUS_PATH = BASE / "corpus.json"
TAGS_PATH = BASE / "concept_tags_vocabulary.json"
TAXONOMY_PATH = BASE / "TAXONOMY.md"

MODEL = "gemini-3.1-pro-preview"

# Target distribution for the full corpus (~5,279 after generation)
TARGET_TOTAL = 5280
TARGET_PCTS = {"L4": 0.25, "L5": 0.22, "L6+": 0.10}

# Mode distribution for generated questions (no concept-recall at L4+)
MODE_WEIGHTS = {
    "requirements-to-architecture": 0.30,
    "tradeoff-analysis": 0.25,
    "symptom-to-cause": 0.15,
    "failure-to-root-cause": 0.15,
    "napkin-math": 0.10,
    "optimization-task": 0.05,
}

# Track distribution for generated questions
TRACK_WEIGHTS = {"cloud": 0.55, "global": 0.20, "edge": 0.15, "mobile": 0.07, "tinyml": 0.03}

# Maximum questions to generate per API call
BATCH_GEN_SIZE = 10

# ─── Gemini Client ────────────────────────────────────────────

_client = None


def init_gemini():
    global _client
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Run: source ~/.zshrc")
        sys.exit(1)
    _client = genai.Client(api_key=api_key)
    # Quick test
    test = _client.models.generate_content(model=MODEL, contents="Say OK")
    if not test.text:
        print("ERROR: Gemini API test failed")
        sys.exit(1)
    print(f"  Gemini API ready ({MODEL})")


def call_gemini(prompt: str, retries: int = 2) -> str | None:
    """Call Gemini API with retries."""
    for attempt in range(retries + 1):
        try:
            response = _client.models.generate_content(
                model=MODEL, contents=prompt,
                config={"temperature": 0.8, "max_output_tokens": 16384},
            )
            text = response.text.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return text.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Gemini error: {e}")
                return None


# ─── KA Metadata ──────────────────────────────────────────────

# Knowledge area names and descriptions (from TAXONOMY.md)
KA_INFO = {
    "A1": ("ML Workflow", "The iterative development cycle: experiment, train, evaluate, deploy. Feedback loops, iteration speed."),
    "A2": ("Neural Computation", "From math to silicon: matrix multiply, activation functions, backpropagation, computational graphs, automatic differentiation."),
    "A3": ("Network Architectures", "CNN, RNN, Transformer, MoE, SSM. Inductive bias and hardware mapping. Attention complexity."),
    "A4": ("Training Dynamics", "Loss landscapes, optimizer state, learning rate schedules, gradient flow, convergence. Memory cost analysis."),
    "A5": ("Data Pipelines & Storage", "Data ingestion, preprocessing, storage formats, distributed file systems. I/O as bottleneck."),
    "A6": ("Feature Engineering", "Feature stores, feature computation at scale, real-time vs batch features, training-serving consistency."),
    "A7": ("Data Quality & Drift", "Distribution shift detection, data validation, schema evolution, label quality, training-serving skew."),
    "B1": ("Number Systems & Precision", "FP32, FP16, BF16, FP8, INT8, INT4. Representation range, precision loss, overflow/underflow."),
    "B2": ("Compute Arithmetic", "FLOPS, MACs, arithmetic intensity, roofline model. Compute-bound vs memory-bound classification."),
    "B3": ("Accelerator Design", "GPU, TPU, NPU, FPGA, custom ASICs. Tensor Cores, systolic arrays. Dataflow architectures."),
    "B4": ("Parallel Programming Models", "Thread/warp/block/grid hierarchy, memory hierarchy, occupancy analysis, memory coalescing, tiling."),
    "B5": ("Graph-Level Optimization", "Operator fusion, graph rewriting, memory planning, constant folding. Static vs dynamic graphs."),
    "B6": ("ML Compiler Infrastructure", "Multi-level IR, lowering passes, target-specific codegen, auto-tuning."),
    "B7": ("Accelerated Libraries & Runtimes", "Math libraries, inference runtimes, serving frameworks. Library defaults vs custom kernels."),
    "B8": ("Benchmarking & Profiling", "MLPerf, micro/macro/E2E benchmarks, profiling, roofline analysis, power measurement."),
    "C1": ("Memory Hierarchy", "Registers, L1, L2, DRAM, HBM, NVMe. Bandwidth at each level. Cache behavior, OOM diagnosis."),
    "C2": ("Data Movement & Bandwidth", "PCIe, NVLink, NVSwitch, CXL. Host-device transfers, peer-to-peer. Bandwidth asymmetry."),
    "C3": ("Network Interconnects", "InfiniBand, RoCE, Ethernet. Topology: fat-tree, torus, dragonfly. Bisection bandwidth, RDMA."),
    "C4": ("Collective Communication", "AllReduce, AllGather, ReduceScatter, AllToAll. Ring, tree, butterfly algorithms."),
    "C5": ("Distributed Training", "Data parallelism, tensor parallelism, pipeline parallelism, FSDP/ZeRO, 3D parallelism."),
    "C6": ("Fault Tolerance", "Checkpointing, failure detection, elastic training, MTBF/MTTR, Young-Daly formula."),
    "C7": ("Fleet Orchestration", "Multi-resource scheduling, DRF, locality-aware placement, gang scheduling, multi-tenancy."),
    "C8": ("Distributed Data Processing", "Distributed dataflow, data partitioning, shuffles, petabyte-scale deduplication."),
    "C9": ("High-Performance Storage", "Parallel filesystems, NVMe tiers, storage tiering, checkpoint I/O at scale."),
    "D1": ("Model Compression & Efficiency", "Pruning, knowledge distillation, low-rank factorization, NAS. Compression pipeline design."),
    "D2": ("Quantization", "PTQ, QAT, FP8 formats, calibration strategies, accuracy-efficiency Pareto curves."),
    "D3": ("Inference Optimization", "KV-cache management, continuous batching, paged attention, speculative decoding, prefill-decode."),
    "D4": ("Serving Systems", "Load balancing, autoscaling, batching strategies, SLA management, A/B testing, model routing."),
    "D5": ("Edge & Mobile AI", "Edge devices, mobile phones, NPU delegation, on-device inference, heterogeneous compute."),
    "D6": ("TinyML & Embedded", "Microcontrollers, 256KB SRAM, CMSIS-NN, TFLite Micro, fixed-point arithmetic, sensor fusion."),
    "D7": ("MLOps & Production", "CI/CD for ML, model versioning, safe deployment, drift detection, incident response, observability."),
    "E1": ("Power, Energy & Sustainability", "TDP, DVFS, power capping, perf/watt, carbon footprint, PUE, carbon-aware scheduling."),
    "E2": ("Security, Privacy & Robustness", "Threat models, differential privacy, TEEs, federated learning privacy, adversarial robustness."),
    "E3": ("Responsible AI", "Fairness metrics, disaggregated evaluation, bias auditing, EU AI Act, regulatory compliance."),
    "F1": ("Compound AI Systems", "RAG pipelines, agentic workflows, vector databases, cascading latency, multi-model orchestration."),
}

# RC descriptions for the prompt
RC_INFO = {
    "RC-1": "Resource Quantification",
    "RC-2": "Bottleneck Analysis & Decomposition",
    "RC-3": "Hardware-Compiler-Algorithm Co-Design",
    "RC-4": "Scaling Reasoning",
    "RC-5": "Representational Efficiency",
    "RC-6": "Latency Decomposition",
    "RC-7": "Fault & Reliability Reasoning",
    "RC-8": "System Design",
    "RC-9": "Optimization Methodology",
    "RC-10": "Cost-Efficiency Reasoning",
    "RC-11": "Locality Reasoning",
    "RC-12": "Observability & Debuggability",
    "RC-13": "Concurrency & Asynchrony",
}

# KA → primary RCs mapping
KA_RCS = {
    "A1": ["RC-9"], "A2": ["RC-1", "RC-3"], "A3": ["RC-3", "RC-1"],
    "A4": ["RC-1", "RC-9"], "A5": ["RC-8", "RC-11"], "A6": ["RC-8", "RC-6"],
    "A7": ["RC-12", "RC-7"], "B1": ["RC-5"], "B2": ["RC-1", "RC-2"],
    "B3": ["RC-3"], "B4": ["RC-3", "RC-9"], "B5": ["RC-9", "RC-3"],
    "B6": ["RC-3", "RC-9"], "B7": ["RC-3", "RC-9"], "B8": ["RC-9", "RC-2", "RC-12"],
    "C1": ["RC-1", "RC-11"], "C2": ["RC-2", "RC-11"], "C3": ["RC-4", "RC-2"],
    "C4": ["RC-4", "RC-6"], "C5": ["RC-4", "RC-8"], "C6": ["RC-7"],
    "C7": ["RC-8", "RC-10"], "C8": ["RC-4", "RC-8"], "C9": ["RC-2", "RC-11"],
    "D1": ["RC-5", "RC-2"], "D2": ["RC-5"], "D3": ["RC-6", "RC-9"],
    "D4": ["RC-6", "RC-8"], "D5": ["RC-3", "RC-10"], "D6": ["RC-3", "RC-11"],
    "D7": ["RC-12", "RC-7"], "E1": ["RC-10", "RC-1"], "E2": ["RC-7", "RC-8"],
    "E3": ["RC-8", "RC-10"], "F1": ["RC-8", "RC-6", "RC-11"],
}

# KA-based resource auto-fill was removed during the deep_dive → resources
# migration. Resources are now author-curated per question, not derived from
# hostname heuristics across a knowledge area.


# ─── Deficit Calculation ──────────────────────────────────────


def compute_deficit(corpus: list[dict]) -> list[dict]:
    """Compute per-KA deficit for L4, L5, L6+ levels."""
    ka_levels = defaultdict(Counter)
    for q in corpus:
        ka = q.get("knowledge_area", "?")
        level = q.get("level", "?")
        ka_levels[ka][level] += 1

    batches = []
    for ka in sorted(KA_INFO.keys()):
        counts = ka_levels.get(ka, Counter())
        total = sum(counts.values())
        if total == 0:
            continue

        for level, target_pct in TARGET_PCTS.items():
            current = counts.get(level, 0)
            target = round(total * target_pct)
            deficit = max(0, target - current)
            if deficit > 0:
                # Cap per-KA-level at 20 to avoid overwhelming any single area
                deficit = min(deficit, 20)
                batches.append({
                    "ka": ka,
                    "level": level,
                    "deficit": deficit,
                    "current": current,
                    "total": total,
                })

    return batches


def build_generation_plan(corpus: list[dict], target_ka: str | None = None) -> list[dict]:
    """Build the generation plan: list of (KA, level, count) batches."""
    deficits = compute_deficit(corpus)
    if target_ka:
        deficits = [d for d in deficits if d["ka"] == target_ka]

    # Scale to ~500 total
    total_deficit = sum(d["deficit"] for d in deficits)
    target = 500
    if total_deficit > target:
        scale = target / total_deficit
        for d in deficits:
            d["deficit"] = max(1, round(d["deficit"] * scale))

    # Recompute
    total = sum(d["deficit"] for d in deficits)
    print(f"  Generation plan: {total} questions across {len(deficits)} batches")
    return deficits


# ─── Prompt Builder ───────────────────────────────────────────


def build_generation_prompt(
    ka: str, level: str, count: int,
    ka_tags: list[str], example_qs: list[dict],
    mode: str, track: str, rc: str,
) -> str:
    """Build the Gemini prompt for generating questions."""
    ka_name, ka_desc = KA_INFO[ka]

    # Format example questions
    examples_text = ""
    if example_qs:
        examples_text = "\n\nEXAMPLE QUESTIONS AT THIS LEVEL:\n"
        for eq in example_qs[:3]:
            examples_text += f"""
---
Title: {eq['title']}
Level: {eq['level']} | Track: {eq['track']}
Scenario: {eq['scenario'][:300]}
Solution: {eq['details'].get('realistic_solution', '')[:300]}
Napkin Math: {eq['details'].get('napkin_math', '')[:200]}
"""

    tags_str = ", ".join(ka_tags) if ka_tags else "general concepts"

    return f"""You are generating Staff-level ML Systems interview questions for knowledge area {ka}: {ka_name}.

Area description: {ka_desc}
Reasoning competency: {rc} — {RC_INFO.get(rc, '')}

Generate {count} questions at level {level} for track "{track}".
Each question must test {mode} reasoning.

Valid concept tags for this area: {tags_str}
{examples_text}

Return a JSON array of {count} question objects. Each object must have these EXACT fields:
{{
  "id": "<{track}-{ka.lower()}-kebab-title-{{}}>",
  "track": "{track}",
  "scope": "single-system",
  "level": "{level}",
  "title": "<concise title, 5-10 words>",
  "topic": "<specific topic within {ka_name}>",
  "competency_area": "<one of: compute, memory, latency, precision, power, architecture, optimization, parallelism, networking, deployment, reliability, data, cross-cutting>",
  "canonical_topic": "<1-3 word canonical topic>",
  "bloom_level": "<analyze|evaluate|create>",
  "tags": [],
  "scenario": "<realistic scenario ≥100 chars with specific numbers: latencies in ms, memory in GB, throughput in QPS>",
  "details": {{
    "common_mistake": "<plausible wrong answer ≥30 chars>",
    "realistic_solution": "<detailed correct answer ≥250 chars with quantitative reasoning, step-by-step>",
    "napkin_math": "<step-by-step calculation with actual numbers>",
    "resources": []
  }},
  "status": "published",
  "version": 1,
  "reasoning_competency": "{rc}",
  "knowledge_area": "{ka}",
  "reasoning_mode": "{mode}",
  "concept_tags": ["<2-4 valid tag IDs from the list above>"],
  "primary_concept": "<most important concept tag>"
}}

CRITICAL RULES:
1. The "id" field must be unique and follow pattern: {track}-{ka.lower()}-kebab-title-NUMBER (use sequential numbers 1-{count})
2. scenario MUST be ≥100 chars with specific quantitative details (exact memory sizes, latencies, throughput numbers)
3. realistic_solution MUST be ≥250 chars with step-by-step quantitative reasoning
4. common_mistake MUST be ≥30 chars describing a plausible wrong approach
5. napkin_math MUST include actual arithmetic with units
6. concept_tags must be valid IDs from the tag list above (2-4 tags)
7. Level "{level}" means: {"L4 = analyze multi-component systems, identify non-obvious interactions" if level == "L4" else "L5 = design novel solutions, evaluate complex tradeoffs across the full stack" if level == "L5" else "L6+ = open-ended system design spanning multiple domains, production-critical decisions under ambiguity"}
8. NO concept-recall mode — every question must require analysis or design
9. Each question must be distinct and test a different aspect of {ka_name}

Return ONLY the JSON array, no explanation or markdown fences."""


# ─── Generation Pipeline ──────────────────────────────────────


def generate_batch(batch: dict, corpus: list[dict], ka_tags_map: dict[str, list[str]]) -> list[dict]:
    """Generate questions for one (KA, level) batch."""
    ka = batch["ka"]
    level = batch["level"]
    count = batch["deficit"]

    # Get concept tags for this KA
    ka_tags = ka_tags_map.get(ka, [])

    # Get example questions at this level from the same KA
    examples = [q for q in corpus if q.get("knowledge_area") == ka and q.get("level") == level]
    if not examples:
        # Fall back to adjacent levels
        examples = [q for q in corpus if q.get("knowledge_area") == ka and q.get("level") in ("L4", "L5")]
    random.shuffle(examples)
    examples = examples[:3]

    # Pick mode based on weights
    modes = list(MODE_WEIGHTS.keys())
    weights = list(MODE_WEIGHTS.values())
    mode = random.choices(modes, weights=weights, k=1)[0]

    # Pick track based on weights
    tracks = list(TRACK_WEIGHTS.keys())
    tweights = list(TRACK_WEIGHTS.values())
    track = random.choices(tracks, weights=tweights, k=1)[0]

    # Pick RC from KA mapping
    rcs = KA_RCS.get(ka, ["RC-8"])
    rc = random.choice(rcs)

    prompt = build_generation_prompt(ka, level, count, ka_tags, examples, mode, track, rc)
    text = call_gemini(prompt)

    if not text:
        return []

    # Parse JSON
    try:
        # Try direct parse
        questions = json.loads(text)
        if not isinstance(questions, list):
            questions = []
    except json.JSONDecodeError:
        # Try to extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    # Post-process and validate each question
    valid_questions = []
    for i, q in enumerate(questions):
        try:
            # Ensure required fields exist
            if not isinstance(q, dict):
                continue

            # Fix common Gemini issues
            # Ensure level is correct (Gemini sometimes outputs "L6" instead of "L6+")
            if q.get("level") == "L6":
                q["level"] = "L6+"

            # Ensure level matches what we asked for
            q["level"] = level

            # Ensure track matches
            q["track"] = track

            # Fix ID format
            title_slug = re.sub(r'[^a-z0-9]+', '-', q.get("title", f"q-{i}").lower()).strip('-')[:40]
            q["id"] = f"{track}-{ka.lower()}-{title_slug}-{i+1}"

            # Ensure knowledge_area and reasoning_competency are set
            q["knowledge_area"] = ka
            q["reasoning_competency"] = rc
            q["reasoning_mode"] = mode

            # Ensure details exists and has required fields
            details = q.get("details", {})
            if not isinstance(details, dict):
                details = {}

            # Validate minimum lengths
            scenario = q.get("scenario", "")
            solution = details.get("realistic_solution", "")
            mistake = details.get("common_mistake", "")

            if len(scenario) < 30 or len(solution) < 50 or len(mistake) < 10:
                continue

            # Resources default to [] when generator omits them; authors
            # curate post-generation. No auto-fill by knowledge area.
            if "resources" not in details:
                details["resources"] = []

            q["details"] = details

            # Ensure tags is a list
            if "tags" not in q or not isinstance(q.get("tags"), list):
                q["tags"] = []

            # Ensure concept_tags is valid
            if not isinstance(q.get("concept_tags"), list):
                q["concept_tags"] = ka_tags[:3] if ka_tags else []

            # Ensure other required fields
            q.setdefault("scope", "single-system" if level in ("L4", "L5") else "cross-cutting")
            q.setdefault("canonical_topic", q.get("topic", ""))
            q.setdefault("bloom_level", "analyze" if level == "L4" else "evaluate" if level == "L5" else "create")
            q.setdefault("status", "published")
            q.setdefault("version", 1)
            q.setdefault("created_at", "")
            q.setdefault("updated_at", "")
            q.setdefault("chain_ids", None)
            q.setdefault("chain_positions", None)

            valid_questions.append(q)
        except Exception as e:
            continue

    return valid_questions


# ─── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate hard questions via Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--ka", type=str, help="Only generate for one knowledge area")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load corpus
    corpus = json.load(open(CORPUS_PATH))
    print(f"Corpus: {len(corpus)} questions")

    # Load concept tags vocabulary
    tags_data = json.load(open(TAGS_PATH))
    ka_tags_map = defaultdict(list)
    for tag in tags_data.get("tags", []):
        ka_tags_map[tag["knowledge_area"]].append(tag["id"])

    # Build generation plan
    plan = build_generation_plan(corpus, target_ka=args.ka)

    if args.dry_run:
        print(f"\n{'KA':>4} | {'Level':>5} | {'Need':>4} | {'Current':>7} | {'Total':>5}")
        print("-" * 45)
        for batch in plan:
            print(f"{batch['ka']:>4} | {batch['level']:>5} | {batch['deficit']:>4} | {batch['current']:>7} | {batch['total']:>5}")
        total = sum(b["deficit"] for b in plan)
        print(f"\nTotal to generate: {total}")
        return

    # Initialize Gemini
    init_gemini()

    # Generate in parallel
    all_generated = []
    start = time.time()
    failed_batches = 0

    print(f"\n  Generating {sum(b['deficit'] for b in plan)} questions across {len(plan)} batches...")
    print(f"  Workers: {args.workers}")
    print()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_batch, batch, corpus, ka_tags_map): batch
            for batch in plan
        }

        for future in as_completed(futures):
            batch = futures[future]
            try:
                questions = future.result()
                if questions:
                    all_generated.extend(questions)
                    print(f"  {batch['ka']}/{batch['level']}: +{len(questions)} (wanted {batch['deficit']})")
                else:
                    failed_batches += 1
                    print(f"  {batch['ka']}/{batch['level']}: FAILED (wanted {batch['deficit']})")
            except Exception as e:
                failed_batches += 1
                print(f"  {batch['ka']}/{batch['level']}: EXCEPTION: {e}")

    elapsed = time.time() - start

    # ─── Dedup generated IDs ──────────────────────────────────
    existing_ids = {q["id"] for q in corpus}
    unique_generated = []
    seen_ids = set()
    for q in all_generated:
        # Ensure unique ID
        qid = q["id"]
        while qid in existing_ids or qid in seen_ids:
            qid = qid + "-g"
        q["id"] = qid
        seen_ids.add(qid)
        unique_generated.append(q)

    # ─── Append to corpus ─────────────────────────────────────
    corpus.extend(unique_generated)

    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # ─── Report ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Generated: {len(unique_generated)} questions in {elapsed:.0f}s")
    print(f"  Failed batches: {failed_batches}")
    print(f"  New corpus size: {len(corpus)}")
    print()

    # Level distribution
    level_dist = Counter(q["level"] for q in corpus)
    print(f"  Level distribution:")
    for level in ["L1", "L2", "L3", "L4", "L5", "L6+"]:
        count = level_dist.get(level, 0)
        pct = count / len(corpus) * 100
        print(f"    {level}: {count:>5} ({pct:.1f}%)")

    # Mode distribution of generated
    mode_dist = Counter(q.get("reasoning_mode", "?") for q in unique_generated)
    print(f"\n  Generated mode distribution:")
    for mode, count in mode_dist.most_common():
        print(f"    {mode}: {count}")


if __name__ == "__main__":
    main()
