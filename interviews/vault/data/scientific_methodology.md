# StaffML: Scientific Methodology for Corpus Construction

## Thesis

A physics-grounded interview corpus can be constructed systematically
through backward design, where every design decision is derived from
the competency being assessed, and every parameter is either justified
by first principles or validated empirically. The corpus size, structure,
and content are OUTPUTS of this methodology, not arbitrary choices.

---

## The Seven-Stage Methodology

### Stage 1: Competency Decomposition

**Question answered:** What must a Staff ML systems engineer demonstrate?

**Method:** Decompose "mechanical sympathy" (the ability to reason
quantitatively about hardware constraints) into atomic cognitive skills
through task analysis of real engineering work.

**Result:** Four atomic skills — recall, analyze, design, quantify.
These are irreducible: you cannot decompose "analyze" further without
leaving the ML systems domain.

**Validation:** The four skills map to observable interview behaviors:
- Recall → candidate retrieves a hardware spec without prompting
- Analyze → candidate explains WHY a system behaves as observed
- Design → candidate proposes an architecture meeting constraints
- Quantify → candidate produces a concrete number from specs

### Stage 2: Zone Construction

**Question answered:** How do these skills combine in real engineering tasks?

**Method:** Observe that real tasks never exercise one skill in isolation.
Enumerate all pairwise combinations: C(4,1) + C(4,2) + 1 = 4 + 6 + 1 = 11
zones. Each zone is a distinct cognitive act testable in an interview.

**Result:** 11 zones (recall, analyze, design, quantify, diagnosis,
specification, fluency, evaluation, realization, optimization, mastery).

**Justification:** This is a combinatorial lattice, not an arbitrary
taxonomy. The structure is complete — there are exactly C(4,1) + C(4,2) + 1
ways to combine 4 skills at the pairwise-and-above level.

**Distinction from Bloom:** Bloom's Revised Taxonomy is a hierarchy
(remember → create). Our model is a lattice — skills combine laterally,
not stack vertically. Bloom cannot represent "fluency" (recall ∩ quantify)
because it has no "quantify" dimension. This is a contribution.

### Stage 3: Topic Coverage

**Question answered:** What concepts must be tested?

**Method:** Curate concepts through three filters:
1. **Pedagogical role** — does this serve ML systems education?
2. **Endurance test** — will engineers need this in 5-10 years?
3. **Quantitative reasoning** — can it be tested with napkin math?

**Result:** 79 topics spanning 13 competency areas (compute, memory,
data, optimization, reliability, architecture, deployment, latency,
power, precision, networking, parallelism, cross-cutting), connected
by a prerequisite graph of 57 edges.

**Validation:** Expert review by 12 domain experts identified 7
additional topics with multi-reviewer consensus. Topics that fail
the endurance test (framework-specific APIs, ephemeral tools) are
excluded by design.

### Stage 4: Hardware Applicability Filter

**Question answered:** Where does each concept have physical meaning?

**Method:** For each (topic, track) pair, determine whether the concept
has a physical substrate on that hardware tier. If not, exclude with a
one-sentence physics justification.

**Result:** 233 applicable pairs out of 316 possible (79 × 4 tracks).
83 excluded pairs, each with an auditable physics reason.

**Examples of exclusions:**
- RDMA transport × TinyML: "MCUs use SPI/UART buses, not packet-switched
  networks with RDMA capability"
- Duty cycling × Cloud: "Datacenter GPUs run 24/7 at full utilization;
  duty cycling wastes CapEx"
- Pipeline parallelism × Mobile: "Single-SoC mobile devices cannot split
  model stages across chips"

**Validation:** Expert review corrected the matrix in both directions:
5 wrong exclusions restored (e.g., mixed-precision on TinyML — INT8/INT4
IS the MCU precision story), 2 wrong inclusions removed (e.g., 3D
parallelism on TinyML). The matrix is a living document.

**Scientific standard:** This establishes CONTENT VALIDITY (Messick, 1995)
— every question in the corpus references a concept that has physical
meaning on the targeted hardware tier.

### Stage 5: Capacity Bounding

**Question answered:** How many meaningfully distinct questions can exist
per (topic, track, zone) cell?

**Method:** Capacity is bounded by the number of distinct solution paths
available. Two questions are DISTINCT if they require different formulas,
different hardware parameters, or different reasoning chains.

**Capacity varies by:**
- Zone complexity (simple zones have fewer angles than compound zones)
- Difficulty level (L5/L6+ questions have more variation than L2)
- Topic scope (broad topics like roofline analysis support more scenarios
  than narrow topics like container orchestration)

**Current model (hypotheses, pending empirical validation):**

| Zone Type | L1-L2 | L3-L4 | L5-L6+ |
|-----------|:-----:|:-----:|:------:|
| Simple | 3 | 5 | 5 |
| Compound | 4 | 6 | 8 |
| Mastery | 5 | 7 | 10 |

**Planned validation:** Semantic similarity study — 3+ raters generate
questions for sample cells until saturation, measuring marginal information
gain. The knee of the saturation curve is the empirical capacity.

**Honest limitation:** Until the saturation study is complete, capacity
constants are calibrated estimates informed by expert enumeration
(one reviewer demonstrated 8 distinct questions in a cell previously
bounded at 4).

### Stage 6: Corpus Construction

**Question answered:** How are questions generated, validated, and curated?

**Method:** Three-phase construction:

**Phase A — Generation:**
Questions are generated by LLMs (Gemini 2.5 Flash, Claude Sonnet/Opus)
with structured prompts specifying exact (topic, track, zone, level)
targets and requiring:
- Real hardware specs (from a constants table, not hallucinated)
- Concrete napkin math with actual numbers
- A common mistake revealing a specific misconception
- A realistic solution with quantitative reasoning

**Phase B — Validation:**
Multi-model cross-validation: questions generated by Model A are
validated by Model B for:
- Mathematical correctness (napkin math produces the claimed answer)
- Hardware spec accuracy (numbers match published datasheets)
- Question quality (scenario is realistic, not contrived)

**Phase C — Curation:**
Quality infrastructure enforces 19 invariant checks:
- Schema compliance (all required fields present)
- Taxonomy consistency (all topics exist in the prerequisite graph)
- Uniqueness (no duplicate IDs, no duplicate titles within tracks)
- Distribution sanity (no topic exceeds 15% of corpus, all zones populated)
- Chain integrity (prerequisite chains are valid)

**Key principle:** Generation is cheap; validation is expensive.
The methodology prioritizes verification over volume.

### Stage 7: Expert Review Convergence

**Question answered:** How do we know the framework is complete?

**Method:** Structured review by domain experts with diverse perspectives
(academic, industry CEO, practitioner, framework developer, efficient AI
researcher, edge/mobile specialist). Each reviewer provides structured
feedback on:
1. Missing topics (max 3)
2. Wrong exclusions in the applicability matrix
3. Wrong inclusions
4. Zone model validity
5. Capacity model accuracy
6. Practical value for hiring
7. Single biggest gap

**Convergence criterion:** The framework is structurally complete when:
- No reviewer identifies a missing topic that 3+ others agree with
- Feedback shifts from "you're missing X" to "improve phrasing of Y"
- The north star document has not changed materially in the last round

**Result:** 12 reviewers across 2 rounds. Round 1 identified 15+
structural issues. After corrections, the north star stabilized.
Remaining feedback targets quality improvement (rubrics, timing,
vendor diversity), not structural gaps.

**Scientific standard:** This establishes FACE VALIDITY through
expert consensus and begins to establish CONTENT VALIDITY through
domain coverage review.

---

## What This Methodology Guarantees

1. **Traceability:** Every question traces back to a competency
   requirement through the chain: skill → zone → topic → track → question.

2. **Completeness criterion:** We know when we're done — all applicable
   cells at capacity, distribution balanced, expert feedback converged.

3. **Reproducibility:** Another team following this methodology would
   arrive at a similar structure (different questions, same shape).

4. **Falsifiability:** Each design decision can be challenged:
   - "Is 'quantify' really a distinct skill?" → test with inter-rater study
   - "Should RDMA be excluded from TinyML?" → challenge the physics reason
   - "Is capacity really 5 for recall?" → run the saturation study

5. **Improvability:** The methodology is a living process. Expert review
   already corrected the applicability matrix and proposed new topics.
   The framework improves without redesign.

---

## What This Methodology Does NOT Guarantee

1. **Construct validity of zones** — whether the 11 zones measure truly
   distinct cognitive competencies requires inter-rater reliability data
   (Cohen's κ > 0.7) that we do not yet have.

2. **Level calibration** — whether L3 questions are actually harder than
   L2 requires Item Response Theory analysis on candidate response data.

3. **Predictive validity** — whether high scores on StaffML predict
   job performance requires longitudinal outcome data.

4. **Question correctness** — LLM-generated questions may contain
   plausible-but-wrong math. Validation reduces but does not eliminate
   this risk. Expert verification of a random sample is needed.

These are acknowledged as future work, not swept under the rug.

---

## Summary: The Derivation Chain

```
What must a Staff engineer demonstrate?
    → 4 atomic skills (task analysis)

How do skills combine in real work?
    → 11 zones (combinatorial lattice, C(4,1)+C(4,2)+1)

What concepts must be tested?
    → 79-86 topics (curated through 3 filters)

Where does each concept have physical meaning?
    → 233 applicable pairs (physics filter, expert-corrected)

How many distinct questions per cell?
    → Variable capacity (empirically bounded)

How are questions constructed?
    → LLM generation → cross-model validation → 19-check QA

How do we know the framework is complete?
    → Expert review convergence (12 reviewers, structural stability)
```

**The corpus size is the LAST number computed, not the first.**
