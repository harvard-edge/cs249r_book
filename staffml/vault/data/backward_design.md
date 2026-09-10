# StaffML: Backward Design Framework

## The Question We're Answering

**"How do you know if someone is a Staff-level ML systems engineer?"**

Not: "How do you generate 10,000 interview questions?"
The questions are the LAST thing we design, not the first.

---

## Stage 1: Desired Results (What Must They Demonstrate?)

A Staff ML systems engineer must demonstrate **mechanical sympathy** —
the ability to reason quantitatively about the physical constraints of
ML infrastructure. This decomposes into:

### The Four Skills (atomic, non-decomposable)

| Skill | What it means | How you observe it |
|-------|--------------|-------------------|
| **Recall** | Retrieve hardware specs, formulas, and architectural facts from memory | "What is the HBM bandwidth of an H100?" |
| **Analyze** | Explain WHY a system behaves the way it does, using specs as evidence | "Why is this workload memory-bound?" |
| **Design** | Architect a system that meets requirements, justifying every choice | "Design a serving system for 10K QPS at P99 < 100ms" |
| **Quantify** | Produce a concrete number from specifications and formulas | "How much VRAM does this model need?" |

These four skills are the atoms. You cannot decompose "analyze" further
without leaving the ML systems domain.

### The Compound Competencies (pairwise combinations)

Real Staff work never exercises one skill in isolation. The compound
competencies are WHERE these skills intersect:

| Compound | Skills | What it tests | The "aha" |
|----------|--------|--------------|-----------|
| **Diagnosis** | Recall + Analyze | Identify root cause from symptoms | "The latency spiked because..." |
| **Specification** | Analyze + Design | Translate requirements into architecture | "Given these constraints, the right design is..." |
| **Fluency** | Recall + Quantify | Napkin math from memory | "Off the top of my head, that's ~40 GB" |
| **Evaluation** | Design + Quantify | Compare architectures with numbers | "Option A costs 2x more but serves 3x throughput" |
| **Realization** | Analyze + Quantify | Size a chosen architecture concretely | "This design needs 4 nodes because..." |
| **Optimization** | Recall + Design | Diagnose bottleneck AND propose fix | "The bottleneck is memory bandwidth; switching to INT8 gives 2x" |

### The Integration Competency

| Compound | Skills | What it tests |
|----------|--------|--------------|
| **Mastery** | All four | Full system reasoning under ambiguity |
| **Debug** | Recall + Analyze + Quantify | Fix a broken system with incomplete info |

**Key insight**: The 12 zones aren't arbitrary categories. They're the
complete set of pairwise (and higher) skill combinations from 4 atomic skills.
4-choose-1 + 4-choose-2 + 2 integration = 4 + 6 + 2 = 12.

---

## Stage 2: Acceptable Evidence (How Would You Know?)

Given the 12 competencies, what constitutes evidence that someone has them?

### The Evidence Must Be:

1. **Quantitative** — not "explain the roofline model" but "compute the
   ridge point for H100 and determine if this workload is memory-bound"

2. **Hardware-grounded** — reference real specs, not abstract "assume
   bandwidth is B"

3. **Scenario-based** — embedded in a realistic engineering context,
   not a textbook exercise

4. **Falsifiable** — there must be a wrong answer that reveals a
   specific misconception (the "common mistake")

### The Context Dimensions

Evidence varies along two independent axes:

**WHAT you're testing** (Topic × Competency Area):
- 86 topics spanning 13+ competency areas
- Each topic = a specific ML systems concept
- Each area = a cluster of related topics

**WHERE you're testing it** (Track):
- Cloud: TB-scale memory, TFLOPS compute, datacenter power
- Edge: GB-scale memory, TOPS compute, thermal envelope
- Mobile: GB unified memory, battery budget, app lifecycle
- TinyML: KB-scale SRAM, MHz clock, milliwatt power

**HOW HARD** (Level):
- L1-L2: Textbook knowledge, single-step reasoning
- L3-L4: Applied knowledge, multi-step with provided specs
- L5: Production experience, multi-factor tradeoff analysis
- L6+: System-of-systems, design under ambiguity

### The Applicability Filter

Not every (topic, track) pair produces valid evidence. The filter is
physics-based: if the concept has no physical substrate on that hardware
tier, no meaningful question exists.

**This is a research finding, not an assumption.** The applicability
matrix is derived from both physics reasoning AND empirical evidence
(topics that consistently fail to produce valid questions across 7,500+
generation attempts).

---

## Stage 3: Design the Assessment (Questions Come Last)

Only NOW do we design questions. Each question is fully determined by:

```
Question = f(topic, track, zone, level)
```

### The Capacity Bound

For a given (topic, track, zone), how many meaningfully distinct
questions can exist? This is bounded by:

- Number of distinct hardware platforms in that track (3-4)
- Number of distinct model architectures that apply (2-4)
- Number of distinct failure modes / bottlenecks (2-3)
- Number of distinct scale points (2-3)

The product gives theoretical capacity. Empirical capacity is lower
because not all combinations are interesting. We validate empirically
by generating until semantic similarity saturates.

### The Quality Criteria

Each question must satisfy:

1. **Specificity**: Tests exactly one (topic, zone) at one level
2. **Grounding**: References real hardware specs (not hypothetical)
3. **Falsifiability**: Has a common mistake that reveals a misconception
4. **Computability**: Answer includes concrete napkin math
5. **Distinctness**: Solution path differs from all other questions in the cell

### The Completeness Criterion

The corpus is COMPLETE when:
- Every applicable (topic, track, zone, level) cell has questions
  up to its empirical capacity
- No cell is grossly overfilled (capped at capacity)
- Distribution is balanced (σ/μ < 0.5 across topics)
- Every question passes quality verification

---

## The Backward Design Chain

```
Staff engineer competency (what they must do)
    ↓ decompose into
4 atomic skills × pairwise combinations = 12 zones (how we test)
    ↓ cross with
86 topics × 4 tracks (what and where we test)
    ↓ filter by
Physics applicability (what's meaningful)
    ↓ bound by
Empirical capacity (how many distinct questions exist)
    ↓ produces
~12,000-14,000 principled questions (the corpus)
    ↓ validated by
Expert review convergence + empirical saturation + inter-rater reliability
```

Each step is DERIVED from the one above. The question count is an
OUTPUT of the design, not an INPUT.

---

## How This Appears in the Paper

The paper presents this as a clean derivation:

**Section 2** (Competency Model): "We decompose Staff-level ML systems
competency into 4 atomic skills and show that their pairwise combinations
produce 12 cognitive zones..."

**Section 3** (Topic Taxonomy): "We identify 86 topics spanning 13
competency areas as the minimum spanning set..."

**Section 4** (Applicability): "Not every concept has a physical substrate
on every hardware tier. We derive an applicability matrix with physics-
grounded exclusions..."

**Section 5** (Capacity): "We empirically determine the number of
meaningfully distinct questions per cell through semantic similarity
analysis..."

**Section 6** (Coverage): "The intersection of topology × applicability ×
capacity yields a principled corpus of N questions..."

The reader sees: constraints → derivation → corpus.
NOT: we generated a bunch of questions and then rationalized the structure.

---

## What This Framework Gives Us

1. **Defensibility**: Every design decision traces back to "what must a
   Staff engineer demonstrate?"

2. **Completeness criterion**: We know WHEN we're done (all cells at
   capacity, distribution balanced)

3. **Prioritization**: Fill the most important gaps first (mastery and
   optimization zones, underfilled critical topics)

4. **Quality over quantity**: The framework tells us to STOP generating
   when capacity is reached, and shift to validation

5. **Reproducibility**: Another team could follow this methodology and
   arrive at a similar structure (different questions, same shape)
