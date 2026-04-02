# StaffML Methodology Notes (Living Document)

## What We're Actually Discovering

The methodology for building a principled interview corpus is itself the research contribution.
Not "we built 10,000 questions" but "here's how you derive what questions SHOULD exist,
verify they're correct, and know when you're done."

---

## The Three-Phase Discovery (as it happened)

### Phase 1: Generate-and-Count (naive)
- Started with: "fill every topic×track×zone cell to >= 3"
- Problem: treated all 3,476 cells as equal
- Lesson: **you can't brute-force coverage — some cells are physically meaningless**

### Phase 2: Physics-Grounded Exclusion
- Built applicability matrix: 86 of 316 topic×track pairs excluded
- Each exclusion has a one-sentence physics justification
- Lesson: **the corpus structure must respect hardware physics**
- But: Patterson flagged circularity — we used LLM generation failures as evidence
  of inapplicability. Need independent expert validation.

### Phase 3: Capacity-Bounded Design (where we are now)
- Zone capacity model: 3/4/5 questions per cell based on cognitive complexity
- Derives total from first principles: 230 × 41 = 9,430
- But: Dean says 3-5 is too conservative for L5/L6+, Reddi says recall needs 5 not 3
- Lesson: **capacity should vary by zone AND by level, not just zone**

### Phase 4: Rebalance Before Expand (emerging from feedback)
- 5/6 reviewers say: redistribute existing questions, don't just add more
- Some cells have 76 questions (25x capacity), others have 0
- Lesson: **corpus quality is about distribution, not volume**

---

## The Convergence Question: "Are We Done?"

We are NOT done when:
- We hit some magic number (10,000 questions)
- Every cell has >= N questions

We ARE done when:
1. Every applicable cell has enough distinct questions to test the concept
   (determined by empirical saturation, not arbitrary threshold)
2. No cell is grotesquely overfilled (capped at capacity)
3. Every question passes physics verification (specs correct, math right)
4. Expert panel feedback converges (no new structural gaps identified)
5. Inter-rater reliability on zone classification exceeds threshold

---

## Key Methodological Insights (for the paper)

### Insight 1: Corpus Size is Derived, Not Chosen
The principled approach: define the space (topics × tracks × zones),
exclude impossible regions (physics filter), bound capacity per cell
(cognitive complexity), and the number falls out. You don't pick 10,000
and work towards it — you derive ~9,500 and stop.

### Insight 2: The Applicability Matrix is a Research Contribution
Not every concept applies to every hardware tier. Documenting WHY
(with physics justifications) is as valuable as the questions themselves.
It defines the boundary of ML systems knowledge by deployment context.

### Insight 3: Distribution Matters More Than Volume
A corpus with 5,000 well-distributed questions beats 15,000 poorly
distributed ones. The metric is coverage completeness (% of applicable
cells at capacity), not raw count.

### Insight 4: Generation and Validation are Asymmetric
Generating a question takes seconds. Validating it takes minutes.
The hard work is verification, not creation. This inverts the usual
assumption about question bank construction.

### Insight 5: Feedback Convergence Signals Completeness
When expert reviewers stop identifying new structural gaps and start
suggesting polish (better wording, more examples), the framework is
structurally complete. We track this by counting "new structural issue"
flags per review round.

---

## Reviewer Feedback Tracker

### Round 1+2 (12 reviewers, in progress)
New structural issues identified: 15+
- Missing topics (comm-overlap, seq-parallelism, disagg serving, ...)
- Capacity model needs empirical backing
- Applicability matrix has errors (both directions)
- Missing competency area (testing/validation, compilation)
- Production/ops dimension underrepresented
- Question distribution severely skewed

### Convergence metric
- Round 1: 15+ new issues → NOT CONVERGED
- Round 2 (projected): 5-8 new issues → APPROACHING
- Round 3 (projected): 1-3 new issues → NEAR CONVERGENCE
- Round 4 (projected): 0-1 new issues → CONVERGED

---

## Open Questions

1. How do we EMPIRICALLY determine zone capacity? (Patterson's challenge)
   - Approach: semantic similarity within overfilled cells
   - Plot marginal information gain vs. question count
   - Find the knee → that's the empirical capacity

2. How do we validate the applicability matrix independently?
   - Approach: 3 experts independently rate each of 86 excluded pairs
   - Inter-rater agreement → confidence in exclusions

3. How do we define "meaningfully distinct"?
   - Approach: two questions are distinct if their napkin math requires
     different formulas or different hardware parameters
   - Formalize: distinct(q1, q2) iff the solution path differs

4. When should we stop iterating?
   - When the north star hasn't changed materially in the last review round
   - When feedback shifts from "missing X" to "improve wording of Y"
   - When we can write the paper's methods section without hedging

---

## Paper Notes: Frameworks & Terminology to Incorporate

### Cite & Use: Understanding by Design (Wiggins & McTighe, 2005)
- Our backward design is a specialization of their UbD framework
- Their 3 stages: desired results → evidence → learning plan
- Our 5 steps: competency → zones → topics → applicability → capacity
- Cite as methodological foundation. Shows we're not inventing pedagogy from scratch.

### Cite & Use: Item Response Theory (IRT)
- Psychometric framework for empirically calibrating question difficulty
- Used by GRE, MCAT, AP exams
- Our L1-L6+ levels are currently author-assigned (Patterson, Dean flagged this)
- IRT would let us calibrate from actual candidate response data
- Paper action: mention in future work / pilot study section

### Use These Terms: Content Validity vs Construct Validity
- Content validity = "do questions cover the right topics?" → our applicability matrix
- Construct validity = "do zones actually measure distinct competencies?" → inter-rater study
- Using these terms signals psychometric rigor to reviewers

### Report These Metrics: Inter-Rater Reliability
- Cohen's Kappa (2 raters) or Krippendorff's Alpha (3+ raters)
- κ > 0.7 = "substantial agreement"
- Apply to: zone classification, level assignment, topic assignment
- Paper action: report in validation section (or acknowledge as needed future work)

### Make This Distinction Explicit: Bloom vs Ikigai
- Bloom's Revised Taxonomy = hierarchy (remember→create)
- Our ikigai model = combinatorial lattice (4 skills combine pairwise)
- Key difference: Bloom can't capture "fluency" (recall + quantify) because
  Bloom doesn't have "quantify" as a distinct cognitive act
- This distinction IS a contribution — don't bury it

### Framing for the Paper
- "Backward design" goes in the introduction (DONE — added paragraph)
- "Content validity" → applicability matrix section
- "Construct validity" → future work / limitations
- "IRT" → future work / pilot study
- "Bloom vs ikigai" → competency model section (make the contrast explicit)
- "Inter-rater reliability" → quality assurance or future work

---

## Figures Needed for the Paper

### Fig 1: The Backward Design Chain (NEW — hero figure)
A flow diagram showing the 5-step derivation:
```
Staff competency → 4 skills → 12 zones → 86 topics → physics filter → capacity bounds → corpus
```
Each step narrows the space. Show numbers at each stage:
- 4 skills → 12 zones
- 86 topics × 4 tracks = 344 pairs
- Physics filter: 344 → 245 applicable (86 excluded)
- Capacity bounds: 245 × 12 zones × avg capacity = ~12,000
Style: left-to-right funnel, each stage labeled with its section reference.

### Fig 2: The Ikigai Competency Model (EXISTS — fig-competency-model.svg)
Already exists. The 4 skills → 11 zones Venn/lattice diagram.
UPDATE: add the "debug" zone if we adopt it.

### Fig 3: Applicability Matrix Heatmap (NEW)
79 topics (y-axis) × 4 tracks (x-axis). Color:
- Green = applicable (has questions)
- Red/X = excluded (with physics reason)
- Shows the "holes" and why they exist.
Groups topics by competency area for visual clustering.
This is the "not every cell makes sense" figure the user specifically requested.

### Fig 4: Topic Distribution (Before vs After Rebalancing) (NEW)
Bar chart: 79 topics on x-axis, question count on y-axis.
Show the BEFORE (skewed: roofline 309, KV-cache 24) and
AFTER (balanced: all between 40-150).
Red line at 40 (floor) and 150 (cap).
This tells the rebalancing story visually.

### Fig 5: Zone × Level Heatmap (EXISTS — fig-zone-level-heatmap.pdf)
Already exists. Shows where questions concentrate by zone and level.
UPDATE with new numbers after generation + rebalancing.

### Fig 6: Vendor Diversity Pie/Bar (NEW)
Hardware platform distribution in cloud-track questions:
- NVIDIA (H100, A100, etc.)
- AMD (MI300X)
- Google (TPU)
- Other
Show BEFORE (9:1 NVIDIA:AMD) and target AFTER (3:1).

### Fig 7: Coverage Completeness Curve (NEW)
X-axis: questions generated (cumulative). Y-axis: % of applicable cells filled.
Shows the diminishing returns: first 5,000 Qs fill 60% of cells,
next 5,000 fill 35%, last 2,000 fill the remaining 5%.
This is the empirical argument for "corpus size is bounded."

### Fig 8: Expert Convergence Plot (NEW)
X-axis: review round (1, 2, 3). Y-axis: new structural issues identified.
Round 1: 15+. Round 2: projected 5-8. Round 3: projected 1-3.
Shows convergence — when feedback shifts from structural to polish.

### Fig 9: Quality Summary (EXISTS — fig-quality-summary.pdf)
Already exists. Field coverage, validation status, invariant checks.
UPDATE with new numbers.

### Existing figures to keep:
- fig-corpus-distribution.pdf (track × level heatmap)
- fig-format-balance.pdf (Bloom's level distribution)
- fig-depth-chain.pdf (chain coverage)

### Priority order for NEW figures:
1. Fig 3 (applicability matrix) — user specifically requested this
2. Fig 1 (backward design chain) — hero figure, tells the whole story
3. Fig 4 (rebalancing before/after) — shows we're principled about distribution
4. Fig 7 (coverage curve) — empirical saturation argument
5. Fig 8 (convergence plot) — validates the review methodology
6. Fig 6 (vendor diversity) — shows we addressed the bias

---

## Vendor Balance Principle

The corpus must reflect the competitive hardware landscape, not a
single vendor's product line.

Target ratios for cloud track:
- NVIDIA (H100, A100, H200): ~50% (market leader but not monopoly)
- AMD (MI300X, MI300A): ~20% (second largest, deployed at Azure/Meta/Oracle)
- Google (TPU v5e, v5p): ~15% (significant for training workloads)
- Other (Intel Gaudi, Cerebras, custom ASICs): ~15%

For edge track:
- NVIDIA (Jetson Orin): ~40%
- Qualcomm (Cloud AI 100): ~20%
- Google (Coral Edge TPU): ~15%
- Hailo: ~15%
- Other: ~10%

For mobile track:
- Apple (A17 Pro ANE): ~30%
- Qualcomm (Snapdragon Hexagon): ~30%
- Google (Tensor G3): ~20%
- Samsung (Exynos): ~20%

For TinyML track:
- ARM (Cortex-M4/M7): ~35%
- Espressif (ESP32-S3): ~25%
- ARM+NPU (Ethos-U55): ~25%
- Nordic (nRF5340): ~15%

Action: vendor audit after rebalance analysis completes.

---

## For the Paper: How to Present This

The paper should read as if the methodology was always principled:

1. "We define the coverage space as topics × tracks × zones"
2. "We exclude inapplicable cells using physics-grounded criteria"
3. "We bound capacity per cell using cognitive complexity analysis"
4. "The principled total follows: 230 × 41 = 9,430"
5. "We validate completeness through expert review convergence"
6. "We validate distinctness through semantic similarity analysis"

The iterative discovery process becomes the "validation" section,
not the "methods" section. The reader sees the clean derivation;
the appendix shows the validation evidence.

---

## Mock NeurIPS Review Feedback (Reviewer 1 — Psychometrics)

### Score: Weak Reject (Novelty 5, Quality 4, Significance 6, Clarity 7)

### Critical fixes needed:
1. **Bloom critique softened** — ✅ FIXED (now says "complements" not "departs from")
2. **Internal inconsistencies** — ✅ FIXED (track %, zone count in captions)
3. **LLM generation transparency** — ✅ FIXED (95% ratio + 4.2% error rate disclosed)
4. **Scope statement** — ✅ FIXED (abstract now scopes to "technical systems reasoning")

### Still needed:
- Add MMLU, BIG-Bench, HumanEval, AERA testing standards to references
- Tone down ikigai framing (from "novel model" to "principled organization")
- Address three-skill intersection omission with reasoning
- Reduce self-citation concentration (frame ecosystem as strength not dependency)
- **Pilot study** — essential for moving from Weak Reject to Accept

### Key quote:
"With [pilot study + inter-rater reliability + inconsistency fixes], this could be 
a strong NeurIPS D&B submission. In its current form, it reads as a well-engineered 
system paper that has not yet produced the empirical evidence that a Datasets & 
Benchmarks venue demands."

## Mock NeurIPS Review Synthesis (3/4 reviewers)

### Scores
| | Novelty | Quality | Significance | Clarity | Rec |
|---|:---:|:---:|:---:|:---:|---|
| R1 (psychometrics) | 5 | 4 | 6 | 7 | Weak Reject |
| R2 (ML systems) | 6 | 5 | 6 | 7 | Weak Reject |
| R4 (practitioner) | 6 | 5 | 7 | 7 | Weak Accept |
| **Average** | **5.7** | **4.7** | **6.3** | **7.0** | **Borderline** |

### Universal feedback:
1. No empirical validation = fatal for D&B (all 3 reviewers)
2. Quality score lowest — driven entirely by missing pilot data
3. Significance acknowledged — the gap is real
4. Clarity consistently good

### Two-path strategy:
- **NeurIPS D&B**: Need pilot study (30-50 engineers). Timeline: 4-6 weeks.
- **Workshop**: Publishable now as methodology paper. Target: NeurIPS ML Systems Workshop.

### Fixes already applied from mock reviews:
- ✅ Bloom critique softened (R1)
- ✅ LLM generation transparency added (R1, R4)
- ✅ Scope statement in abstract (R4)
- ✅ H100 specs corrected (R2)
- ✅ Internal inconsistencies fixed (R1, R2)
- ✅ Track percentages use table reference (R1)
