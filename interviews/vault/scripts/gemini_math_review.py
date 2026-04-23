#!/usr/bin/env python3
"""
Gemini 3.1 Pro math review of StaffML corpus.

Chunks questions by track × topic cluster (~25-50 per chunk),
includes hardware reference sheet from mlsysim constants,
and calls gemini CLI for each chunk.

Usage:
    source ~/.zshrc_secrets
    PYTHONUNBUFFERED=1 python3 staffml/vault/scripts/gemini_math_review.py \
        --workers 8 --batch-size 30

Output:
    _validation_results/gemini_review_YYYYMMDD_HHMMSS/
"""

import json
import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Hardware Reference Sheet (from mlsysim/core/constants.py) ──────────────
HARDWARE_REFERENCE = """
## Hardware Reference Sheet (Source of Truth: mlsysim/core/constants.py)

### NVIDIA GPUs
| GPU | Memory | Type | Bandwidth | FP16 Tensor | TDP |
|-----|--------|------|-----------|-------------|-----|
| V100 | 32 GiB | HBM2 | 900 GB/s | 125 TFLOPS | 300W |
| A100 SXM | 80 GiB | HBM2e | 2039 GB/s (~2.0 TB/s) | 312 TFLOPS | 400W |
| H100 SXM | 80 GiB | HBM3 | 3.35 TB/s | 989 TFLOPS | 700W |
| H200 | 141 GB | HBM3e | 4.8 TB/s | 989 TFLOPS | 700W |
| B200 | 192 GiB | HBM3e | 8.0 TB/s | 2250 TFLOPS | 1000W |
| T4 | 16 GiB | GDDR6 | 320 GB/s | 65 TFLOPS | 70W |
| MI300X | 192 GiB | HBM3 | 5.3 TB/s | 1307 TFLOPS | 750W |

### Interconnects
| Link | Bandwidth | Notes |
|------|-----------|-------|
| NVLink V100 | 300 GB/s | 6 links × 50 GB/s |
| NVLink A100 | 600 GB/s | 12 links × 50 GB/s |
| NVLink H100 | 900 GB/s | 18 links × 50 GB/s |
| NVLink B200 | 1800 GB/s | 72 links × 25 GB/s |
| PCIe Gen3 x16 | 15.75 GB/s | After encoding overhead |
| PCIe Gen4 x16 | 32 GB/s | Bidirectional |
| PCIe Gen5 x16 | 64 GB/s | Bidirectional |
| IB HDR | 200 Gbps = 25 GB/s | Per port |
| IB NDR | 400 Gbps = 50 GB/s | Per port |
| IB XDR | 800 Gbps = 100 GB/s | Per port |

### Energy (Horowitz 2014, 45nm)
| Operation | Energy |
|-----------|--------|
| FP32 multiply | 3.7 pJ |
| FP16 multiply | 1.1 pJ |
| INT8 multiply | 0.2 pJ |
| DRAM access | 640 pJ |
| L1 SRAM access | 0.5 pJ |
| L2 SRAM access | 2.0 pJ |
| Register access | 0.01 pJ |

### Key Formulas
- **Params to memory**: 1B params = 2 GB (FP16), 4 GB (FP32), 1 GB (INT8), 0.5 GB (INT4)
- **Training memory (Adam FP32)**: 16 bytes/param (4 weights + 4 grads + 4 momentum + 4 variance)
- **Training memory (mixed precision)**: 2 (FP16 weights) + 2 (FP16 grads) + 4 (FP32 master) + 4 (m) + 4 (v) = 16 bytes/param
- **KV cache**: 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element
- **Ridge point**: peak_FLOPS / peak_bandwidth (ops/byte)
- **AllReduce ring**: 2(N-1)/N × data_size / bandwidth
- **Matmul FLOPs**: (M,K)×(K,N) = 2×M×K×N
- **Conv2D FLOPs**: 2 × K² × Cin × Cout × Hout × Wout
- **DW-separable FLOPs**: K² × C × H × W (depthwise) + Cin × Cout × H × W (pointwise)
- **Frame budget**: 1000ms / FPS
- **Energy**: Power × Time (Joules = Watts × seconds)
- **Battery**: Wh = mAh × V / 1000

### Model Specs (for verification)
- GPT-3: 175B params, 96 layers, 96 heads, d=12288
- LLaMA-2 70B: 70B params, 80 layers, 64 heads (8 KV heads GQA), d=8192
- LLaMA-3 8B: 8B params, 32 layers, 32 heads (8 KV heads GQA), d=4096
- BERT-Base: 110M params, 12 layers, 12 heads, d=768
- ResNet-50: 25.6M params, 4.1 GFLOPs
- MobileNetV2: 3.5M params, 0.3 GFLOPs

### Edge/Mobile Hardware
- Jetson AGX Orin: 275 TOPS INT8, 204.8 GB/s, 32/64 GB LPDDR5, 15-60W
- Coral Edge TPU: 4 TOPS INT8
- Hailo-8: 26 TOPS INT8, ~2.5W
- Apple A17 Pro ANE: ~35 TOPS
- Snapdragon 8 Gen 3 Hexagon: ~45 TOPS
- LPDDR5: ~51.2 GB/s | LPDDR5X: ~68 GB/s

### TinyML Hardware
- Cortex-M4: up to 240 MHz, ~1.25 DMIPS/MHz, FPU+DSP
- Cortex-M7: up to 600 MHz, double-precision FPU
- ESP32-S3: 240 MHz dual-core, 512 KB SRAM, WiFi+BLE
- STM32H7: up to 480 MHz, 1 MB SRAM, 2 MB Flash
- nRF52840: 64 MHz, 256 KB SRAM, 1 MB Flash, BLE 5.0
- RP2040: 133 MHz dual M0+, 264 KB SRAM
""".strip()

REVIEW_PROMPT_TEMPLATE = """You are an expert ML Systems engineer reviewing interview questions for mathematical accuracy.

{hardware_reference}

## Your Task

Review each question below. For EACH question:
1. **Verify all arithmetic** — recompute every calculation from scratch
2. **Check hardware specs** — compare against the reference sheet above
3. **Verify unit conversions** — especially Gbps↔GB/s, GB↔GiB, mAh↔Wh
4. **Check logical consistency** — does the solution actually answer the scenario?
5. **Check MCQ answers** — if options exist, does correct_index point to the right answer?

## Output Format

For each question, output ONE line:
- If correct: `OK | <question_id>`
- If wrong: `ERROR | <question_id> | <what's wrong> | <correct value>`
- If uncertain: `WARN | <question_id> | <concern>`

Be thorough. Check every number. These questions are used in real job interviews.

---

## Questions to Review ({num_questions} questions, {track} track, topics: {topics})

{questions_json}
"""


def chunk_corpus(corpus_path: str, batch_size: int) -> list[dict]:
    """Chunk corpus by track × topic cluster."""
    with open(corpus_path) as f:
        data = json.load(f)

    # Group by track × canonical_topic
    groups = defaultdict(list)
    for q in data:
        track = q.get("track", "unknown")
        topic = q.get("canonical_topic", "misc")
        groups[(track, topic)].append(q)

    # Build chunks of ~batch_size, merging small groups
    chunks = []
    current_chunk = []
    current_track = None
    current_topics = []

    for (track, topic), questions in sorted(groups.items()):
        if current_track and current_track != track and current_chunk:
            # Flush on track change
            chunks.append({
                "track": current_track,
                "topics": current_topics,
                "questions": current_chunk,
            })
            current_chunk = []
            current_topics = []

        current_track = track

        if len(current_chunk) + len(questions) > batch_size and current_chunk:
            chunks.append({
                "track": current_track,
                "topics": current_topics,
                "questions": current_chunk,
            })
            current_chunk = []
            current_topics = []

        current_chunk.extend(questions)
        current_topics.append(topic)

    if current_chunk:
        chunks.append({
            "track": current_track,
            "topics": current_topics,
            "questions": current_chunk,
        })

    return chunks


def build_prompt(chunk: dict) -> str:
    """Build the review prompt for a chunk."""
    # Slim down questions to only the fields Gemini needs
    slim_qs = []
    for q in chunk["questions"]:
        slim = {
            "id": q.get("id", ""),
            "track": q.get("track", ""),
            "level": q.get("level", ""),
            "scenario": q.get("scenario", ""),
            "napkin_math": q.get("details", {}).get("napkin_math", ""),
            "realistic_solution": q.get("details", {}).get("realistic_solution", ""),
            "common_mistake": q.get("details", {}).get("common_mistake", ""),
        }
        # Include options if MCQ
        opts = q.get("details", {}).get("options")
        if opts:
            slim["options"] = opts
            slim["correct_index"] = q.get("details", {}).get("correct_index")
        slim_qs.append(slim)

    return REVIEW_PROMPT_TEMPLATE.format(
        hardware_reference=HARDWARE_REFERENCE,
        num_questions=len(slim_qs),
        track=chunk["track"],
        topics=", ".join(chunk["topics"][:10]),
        questions_json=json.dumps(slim_qs, indent=2, ensure_ascii=False),
    )


def get_genai_client(model: str):
    """Initialize the Google GenAI client."""
    from google import genai
    import os
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    return client


def review_chunk(chunk_idx: int, chunk: dict, output_dir: Path, model: str) -> dict:
    """Call Gemini API to review a chunk."""
    prompt = build_prompt(chunk)
    prompt_path = output_dir / f"prompt_{chunk_idx:03d}.txt"
    result_path = output_dir / f"result_{chunk_idx:03d}.txt"

    prompt_path.write_text(prompt, encoding="utf-8")

    try:
        from google import genai
        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        output = response.text
        result_path.write_text(output, encoding="utf-8")

        # Parse results
        errors = []
        warns = []
        oks = 0
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("ERROR"):
                errors.append(line)
            elif line.startswith("WARN"):
                warns.append(line)
            elif line.startswith("OK"):
                oks += 1

        return {
            "chunk": chunk_idx,
            "track": chunk["track"],
            "topics": chunk["topics"][:5],
            "num_questions": len(chunk["questions"]),
            "oks": oks,
            "errors": errors,
            "warns": warns,
        }
    except subprocess.TimeoutExpired:
        return {
            "chunk": chunk_idx,
            "track": chunk["track"],
            "error": "TIMEOUT",
        }
    except Exception as e:
        return {
            "chunk": chunk_idx,
            "track": chunk["track"],
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Gemini math review of StaffML corpus")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--dry-run", action="store_true", help="Just create chunks, don't call Gemini")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"_validation_results/gemini_review_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunking corpus from {args.corpus}...")
    chunks = chunk_corpus(args.corpus, args.batch_size)
    print(f"Created {len(chunks)} chunks")

    # Summary
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i:3d}: {chunk['track']:8s} | {len(chunk['questions']):3d} Qs | {', '.join(chunk['topics'][:3])}")

    if args.dry_run:
        print("\nDry run — not calling Gemini. Prompts written to:", output_dir)
        for i, chunk in enumerate(chunks):
            prompt = build_prompt(chunk)
            (output_dir / f"prompt_{i:03d}.txt").write_text(prompt)
        return

    print(f"\nLaunching {args.workers} parallel workers with model={args.model}...")
    all_results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(review_chunk, i, chunk, output_dir, args.model): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            idx = result["chunk"]
            track = result["track"]
            n_err = len(result.get("errors", []))
            n_warn = len(result.get("warns", []))
            n_ok = result.get("oks", 0)
            print(f"  Chunk {idx:3d} ({track}): {n_ok} OK, {n_err} ERROR, {n_warn} WARN")

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print aggregate
    total_errors = sum(len(r.get("errors", [])) for r in all_results)
    total_warns = sum(len(r.get("warns", [])) for r in all_results)
    total_oks = sum(r.get("oks", 0) for r in all_results)
    total_qs = sum(r.get("num_questions", 0) for r in all_results)

    print(f"\n{'='*60}")
    print(f"REVIEW COMPLETE")
    print(f"{'='*60}")
    print(f"  Total questions: {total_qs}")
    print(f"  OK: {total_oks}")
    print(f"  ERRORS: {total_errors}")
    print(f"  WARNINGS: {total_warns}")
    print(f"  Results: {output_dir}")

    # Write all errors to a single file
    errors_path = output_dir / "all_errors.txt"
    with open(errors_path, "w") as f:
        for r in sorted(all_results, key=lambda x: x["chunk"]):
            for err in r.get("errors", []):
                f.write(f"{err}\n")
    print(f"  All errors: {errors_path}")


if __name__ == "__main__":
    main()
