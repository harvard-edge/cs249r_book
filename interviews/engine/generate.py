"""
Question generation engine using Gemini Pro.

This is the core generator. It takes a GenerationRequest and produces
validated Question objects using Google's Gemini 2.5 Pro model.

Theoretical basis:
- Automatic Item Generation (Gierl & Haladyna, 2013): template-based
  generation from cognitive models (item models)
- Evidence-Centered Design (Mislevy et al., 2003): every question starts
  with a claim about what competency it measures

The generator uses mlsysim as the single source of truth for all hardware
constants, ensuring napkin math in generated questions matches the textbook.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .schemas import Question, GenerationRequest
from .bloom import get_bloom_level, get_generation_prompt_context

# ---------------------------------------------------------------------------
# Hardware constants context (from mlsysim — the textbook's source of truth)
# ---------------------------------------------------------------------------

def _load_hardware_context() -> str:
    """Build a hardware constants reference from NUMBERS.md.

    We load NUMBERS.md rather than importing mlsysim directly because:
    1. The generation prompt needs human-readable text, not Pint objects
    2. NUMBERS.md is already formatted for the interview context
    3. It avoids complex import chains in the generation subprocess
    """
    numbers_path = Path(__file__).parent.parent / "NUMBERS.md"
    if numbers_path.exists():
        return numbers_path.read_text(encoding="utf-8")
    return "(Hardware constants not available — use conservative estimates)"


HARDWARE_CONTEXT = _load_hardware_context()


# ---------------------------------------------------------------------------
# Example questions (few-shot exemplars for the LLM)
# ---------------------------------------------------------------------------

EXEMPLAR_L1 = """\
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The HBM vs L1 Latency Gap</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "Roughly how much slower is accessing HBM3 memory compared to an L1 register read?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming the gap is small, like 10×. In reality, crossing the physical distance from the compute core to the HBM stacks is a massive latency event.

  **Realistic Solution:** ~300× slower. L1 registers are ~1ns, while HBM3 access is ~300ns.

  > **Napkin Math:** If an L1 read was 1 second, an HBM read would be 5 minutes.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)
  </details>
</details>"""

EXEMPLAR_L3 = """\
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Data Pipeline Stall</b> · <code>data-pipeline</code></summary>

- **Interviewer:** "You are training a vision model on high-resolution medical images. `nvidia-smi` shows GPU utilization fluctuating violently between 0% and 100%. What is the most likely bottleneck in your node?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is too small for the GPU" or "We need a bigger batch size." Both assume the GPU is the problem — it's actually starving.

  **Realistic Solution:** CPU Starvation (The Transformation Wall). The GPU finishes its math instantly, then sits at 0% waiting for the CPU to decode, crop, and augment the next batch of JPEGs. You must offload preprocessing to the GPU (like NVIDIA DALI) or increase your CPU worker count.

  > **Napkin Math:** An H100 can process a ResNet-50 forward pass in ~2ms. JPEG decoding + augmentation on CPU takes ~10-50ms per image. With a batch of 64 images and 8 CPU workers, preprocessing takes $64 \\times 30\\text{ms} / 8 = 240\\text{ms}$. The GPU finishes in 2ms and waits 238ms — that's 99% idle time.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>"""

EXEMPLAR_L5 = """\
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Input Chunking Pipeline Bubble</b> · <code>latency</code> <code>serving</code></summary>

- **Interviewer:** "Users submit 100,000-token documents to your summarization model. To avoid OOMing during the prefill phase, you implement 'Chunked Prefill' (breaking the document into 4,000-token blocks). The prefill now succeeds. But users notice that the Time-To-First-Token (TTFT) actually *increased* compared to when you ran the 100k prompt all at once on a bigger GPU. Why does chunking the prompt slow down the time to first token?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Chunking adds Python looping overhead." Python overhead is microseconds; the slowdown is seconds.

  **Realistic Solution:** You destroyed the **Parallel Math Advantage (Arithmetic Intensity)**. When you process 100,000 tokens in a single massive batch, you formulate a massive matrix multiplication. GPUs are incredibly efficient at this. When you chunk the prompt into 25 sequential blocks of 4,000 tokens, you force the GPU to do 25 separate, smaller matrix multiplications. The GPU must load the *entire* 140 GB model weights from HBM to the SRAM 25 separate times.

  > **Napkin Math:** 70B model, FP16 = 140 GB weights. Single 100k prefill: compute = 2 × 70B × 100k = 14 PFLOPS. At 989 TFLOPS (H100): ~14.2s. But the operation is compute-bound (arithmetic intensity = 100k tokens ≫ ridge point), so the GPU runs near peak. Read weights once = 140 GB / 3.35 TB/s = 42ms — negligible vs compute. Chunked into 25 × 4k blocks: each chunk = 2 × 70B × 4k = 560 TFLOPS. But now you read 140 GB weights 25 times = 3,500 GB memory traffic. At 3.35 TB/s = 1.04s just loading weights. Plus each chunk's arithmetic intensity drops 25×, pushing toward memory-bound. TTFT increases from seconds to much worse — dominated by repeated weight reads.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_generation_prompt(
    request: GenerationRequest,
    bloom_context: str,
    hardware_context: str,
    exemplars: list[str],
) -> str:
    """Build the full generation prompt for Gemini Pro.

    The prompt structure follows AIG methodology:
    1. Role and quality bar
    2. Hardware constants (source of truth)
    3. Bloom's level constraints
    4. Exemplar questions (few-shot)
    5. Generation instructions with specific concept
    """
    exemplar_text = "\n\n---\n\n".join(exemplars)

    return f"""You are a world-class ML systems interview question writer for the StaffML platform.
Your questions are used by engineers preparing for Staff/Principal ML Systems Engineer roles at
top tech companies. Every question must be grounded in real hardware physics and quantitative reasoning.

## ABSOLUTE REQUIREMENTS
1. Every scenario must be REALISTIC — something that actually happens in production
2. Every napkin math section must use REAL hardware specs from the constants below
3. Every common mistake must be a REAL misconception that engineers actually have
4. Every calculation must be arithmetically correct — you will be verified
5. The question must test the SPECIFIC concept requested, not a tangentially related one

## HARDWARE CONSTANTS (Source of Truth — from mlsysim)
Use ONLY these numbers for any hardware specifications in your napkin math:

{hardware_context}

## COGNITIVE LEVEL REQUIREMENTS
{bloom_context}

## EXEMPLAR QUESTIONS (Match this quality and format EXACTLY)

{exemplar_text}

## YOUR TASK
Generate {request.count} interview question(s) for:
- **Track:** {request.track}
- **Concept:** {request.concept}
- **Competency area:** {request.competency_area}
- **Target level:** {request.target_level}

{f"**Additional context from textbook chapter:**" + chr(10) + request.chapter_context if request.chapter_context else ""}

## OUTPUT FORMAT
Return a JSON array of question objects. Each object must have these fields:
```json
[
  {{
    "level": "{request.target_level}",
    "title": "The [Evocative Title]",
    "topic": "kebab-case-topic",
    "track": "{request.track}",
    "scenario": "The full interviewer question text...",
    "common_mistake": "What engineers typically get wrong and why...",
    "realistic_solution": "The correct answer with full explanation...",
    "napkin_math": "Step-by-step quantitative chain with real numbers...",
    "key_equation": "$\\\\text{{LaTeX equation}}$",
    "options": [
      {{"text": "Wrong answer exploiting misconception 1", "is_correct": false}},
      {{"text": "Wrong answer exploiting misconception 2", "is_correct": false}},
      {{"text": "The correct answer", "is_correct": true}},
      {{"text": "Wrong answer exploiting misconception 3", "is_correct": false}}
    ],
    "deep_dive_title": "Relevant Chapter Title",
    "deep_dive_url": "https://mlsysbook.ai/..."
  }}
]
```

IMPORTANT:
- `options` is REQUIRED for L1-L3, optional for L4+
- `napkin_math` is REQUIRED for ALL levels
- `key_equation` is optional but encouraged for L2+
- Return ONLY valid JSON, no markdown wrapping
- Each wrong option must exploit a SPECIFIC, NAMED misconception
"""


# ---------------------------------------------------------------------------
# Generation engine
# ---------------------------------------------------------------------------

def _select_exemplars(level: str) -> list[str]:
    """Select appropriate exemplar questions based on target level."""
    if level in ("L1", "L2"):
        return [EXEMPLAR_L1]
    elif level in ("L3", "L4"):
        return [EXEMPLAR_L1, EXEMPLAR_L3]
    else:
        return [EXEMPLAR_L3, EXEMPLAR_L5]


def _call_gemini_cli(prompt: str, model: str = "gemini-2.5-pro") -> str:
    """Call Gemini via the locally-authenticated CLI.

    Uses the `gemini` CLI tool which handles authentication via
    cached Google credentials — no API key env var needed.
    Prompt is piped via stdin (no temp files).
    """
    result = subprocess.run(
        ["gemini", "--model", model, "-"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=180,  # 3 min timeout for complex generation
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gemini CLI failed (exit {result.returncode}): {result.stderr[:500]}"
        )
    return result.stdout


def generate_questions(
    request: GenerationRequest,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> list[Question]:
    """Generate questions using Gemini Pro via CLI.

    Args:
        request: What to generate (track, concept, level, count)
        model_name: Gemini model to use (default: gemini-2.5-pro)
        temperature: Generation temperature (unused with CLI, kept for API compat)
        api_key: Unused — CLI uses cached credentials

    Returns:
        List of validated Question objects (may be fewer than requested
        if some fail validation)
    """
    # Build prompt
    bloom_context = get_generation_prompt_context(request.target_level)
    exemplars = _select_exemplars(request.target_level)
    prompt = build_generation_prompt(request, bloom_context, HARDWARE_CONTEXT, exemplars)

    # Call Gemini Pro via CLI
    raw_output = _call_gemini_cli(prompt, model=model_name)

    # Parse and validate
    raw_text = raw_output.strip()

    # Handle potential markdown wrapping
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    try:
        raw_questions = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse Gemini response as JSON: {e}", file=sys.stderr)
        print(f"[DEBUG] Raw response:\n{raw_text[:500]}", file=sys.stderr)
        return []

    if not isinstance(raw_questions, list):
        raw_questions = [raw_questions]

    # Validate each question through Pydantic
    validated: list[Question] = []
    for i, raw_q in enumerate(raw_questions):
        try:
            # Normalize options format
            if "options" in raw_q and raw_q["options"]:
                for opt in raw_q["options"]:
                    if isinstance(opt, str):
                        raw_q["options"] = None
                        break

            q = Question(**raw_q)
            validated.append(q)
        except ValidationError as e:
            print(
                f"[WARN] Question {i+1} failed validation: {e.error_count()} errors",
                file=sys.stderr,
            )
            for err in e.errors():
                print(f"  - {err['loc']}: {err['msg']}", file=sys.stderr)

    return validated
