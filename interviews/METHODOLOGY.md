# StaffML Taxonomy Methodology

## Concept Inclusion Framework

The StaffML taxonomy defines what concepts belong in the interview question bank. Every concept must pass **all three filters** to be included.

### The Three Filters

#### Filter 1: Systems Relevance

> "Does this concept have measurable systems implications?"

The concept must directly affect at least one of: compute, memory, latency, throughput, cost, power, or reliability. If you can't write a napkin-math question about it, it doesn't belong.

| Pass | Fail |
|------|------|
| GPTQ (4x memory reduction, 2x throughput gain) | "Attention Is All You Need" (research contribution) |
| Ring AllReduce (bandwidth-optimal gradient sync) | Adam optimizer (math, not systems) |
| KV cache (memory scales with sequence length) | Prompt engineering techniques |

#### Filter 2: Interview Currency

> "Would this appear in a Staff ML Systems interview at a top-5 AI lab in the next 2 years?"

The concept must be something a hiring committee at Google, Meta, OpenAI, Anthropic, or NVIDIA would expect a Staff-level candidate to reason about. This is a **2-year horizon**, not a 10-year horizon — interview questions can be refreshed as the field evolves.

| Pass | Fail |
|------|------|
| DeepSpeed ZeRO stages (active production use) | Caffe framework (historical) |
| LoRA fine-tuning (dominant approach in 2025-2026) | Theano (deprecated) |
| vLLM scheduling (standard serving stack) | TensorFlow 1.x sessions |

#### Filter 3: Reasoning Depth

> "Can you ask questions at 3+ Bloom's levels?"

The concept must support questions from recall through design. If you can only ask "What is X?" (L1), it's too shallow. A good concept supports:

- L1: Define it
- L3: Calculate its impact
- L5: Compare trade-offs against alternatives

| Pass | Fail |
|------|------|
| LoRA (define → calculate rank vs memory → compare to full fine-tuning → design adapter strategy) | CUDA toolkit version (only recall) |
| Continuous batching (define → estimate throughput gain → diagnose stalls → design scheduler) | nvidia-smi flags (only recall) |

### How This Differs From the Textbook

The textbook uses the **Six Patterson Questions**, which include a 10-year endurance test and reject tools/frameworks. StaffML intentionally differs:

| Dimension | Textbook (Patterson) | StaffML (This Framework) |
|-----------|---------------------|-------------------------|
| Horizon | 10 years | 2 years |
| Tools/frameworks | Rejected (teach principles) | Included if they embody systems trade-offs |
| Specificity | General principles only | Specific techniques with measurable impact |
| Goal | Educate students | Test working engineers |

### Refresh Policy

The taxonomy should be reviewed every 6 months. Concepts that no longer pass Filter 2 (interview currency) should be **archived, not deleted** — they may still have valid questions but shouldn't receive new generation priority.

---

## Taxonomy Structure

Each concept in `taxonomy.json` has:

```json
{
  "id": "kebab-case-id",
  "name": "Human Readable Name",
  "description": "One sentence: what it is and why it matters for systems.",
  "prerequisites": ["concept-ids-that-must-be-understood-first"],
  "tracks": ["cloud", "edge", "mobile", "tinyml"],
  "source_chapters": ["vol1_chapter_name"],
  "source_domains": ["domain_name"],
  "question_count": 0
}
```

**Naming rules**:
- `id`: kebab-case, no acronyms unless universally known (e.g., `kv-cache` not `key-value-cache`)
- `name`: Title case, spelled out (e.g., "Fully Sharded Data Parallel" not "FSDP")
- `description`: Must mention a measurable systems quantity (memory, latency, throughput, etc.)
- `prerequisites`: Only list concepts that are *necessary* to understand this one, not merely related
- `tracks`: Include all tracks where the concept applies, even if with different parameters

## Question Generation Standards

Generated questions must:
1. Include specific hardware numbers (not "a GPU" but "an A100 with 80 GB HBM3")
2. Have a napkin-math component with real arithmetic
3. Test systems reasoning, not trivia recall (except at L1)
4. Reference a real technical resource in `deep_dive_url`
5. Pass the Pydantic schema in `schema.py`

## Corpus Balance Targets

| Metric | Target |
|--------|--------|
| Concept coverage | >= 95% of taxonomy |
| Questions per concept | 3-50 (flag outliers) |
| Levels per concept (in primary track) | >= 3 for depth chains |
| Track balance | No track > 2x another (excluding global) |
| Field coverage | 100% for all required fields |
