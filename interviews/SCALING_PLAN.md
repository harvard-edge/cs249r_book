# StaffML Question Generation — Scaling Plan

## Current State (2026-03-22)
- **1,602 questions** across 4 tracks × 6 levels × 12 competency areas
- **66% weighted 3D coverage** (134 questions still needed in 101 cells)
- **Vault fill running** to close the remaining gaps
- **Engine**: fully built, expert-reviewed, 7-gate validation, Rich CLI, HTML reports

---

## Phase 1: Close the Cube (TODAY — running now)

**What**: `vault_fill.py run` generates 134 questions across 101 cells, 6 parallel Gemini calls.
**Then**: `vault_fill.py merge` inserts them into markdown files.
**Target**: 100% weighted 3D coverage, ~1,750 questions.

---

## Phase 2: Expert Taxonomy Review (TODO)

**What**: Get feedback on the 12-area, 101-tag taxonomy from:
- **Industry interviewers** (Google, Meta, Apple Staff+ loops): Is this what they actually test?
- **Hiring managers**: Are there competencies missing that they screen for?
- **Students/candidates**: Can they navigate the tags? Are the categories intuitive?
- **Academic perspective**: Does the taxonomy cover the CS curriculum adequately?

**Questions to ask**:
1. Are there interview topics that don't map to any of the 12 areas?
2. Are any areas too granular or too coarse for filtering?
3. Does the L1-L6+ progression match how companies actually level candidates?
4. Should "systems design" be its own competency (whiteboard architecture questions)?

**Action**: Launch expert agents (simulated Meta Staff interviewer, Google hiring manager, candidate perspective).

---

## Phase 3: Extreme Parallelization Architecture (NEXT)

### The Vision: Generate 1,000 questions in 30 minutes

**Current bottleneck**: Gemini CLI is single-threaded per call (~30s each). With 6 parallel, that's 6 questions/30s = 12 questions/min = ~720/hour.

**Scaling options (ordered by effort)**:

#### Option A: More parallel CLI processes (easy, today)
- Increase `--parallel` from 6 to 12-15
- Risk: Gemini rate limiting (free tier ~15 RPM, paid tier ~60 RPM)
- If you're on paid Gemini: 60 parallel calls = 60 questions/30s = 120/min = 7,200/hour
- **Action**: Just bump the parallel flag

#### Option B: Multi-model fan-out (medium, this week)
- Use Gemini for generation, Claude for validation (cross-model solver)
- Doubles throughput AND improves validation quality (Dean's recommendation)
- Architecture: Gemini generates → temp JSON → Claude validates → approved questions merge

#### Option C: Worker pool with API SDK (medium, if API key works)
- Switch from CLI to `google-genai` Python SDK with async
- `asyncio.gather()` on 20-50 concurrent requests
- ~50 questions/30s = 100/min = 6,000/hour
- **Action**: Fix API key, write async generator

#### Option D: Distributed generation (hard, future)
- Multiple machines each running vault_fill.py on different cell ranges
- Central merge server collects all temp JSONs
- Git-based merge: each machine writes to a branch, PR merge

### The Merge Architecture

```
Generation (parallel, no conflicts)
├── Cell 001: cloud/compute/L1 → _vault/cell_001.json
├── Cell 002: cloud/compute/L2 → _vault/cell_002.json
├── ...
└── Cell 101: tinyml/cross-cutting/L6+ → _vault/cell_101.json

Merge (sequential, handles file conflicts)
├── Read all _vault/*.json
├── Validate each question (Pydantic + quality suite)
├── Dedup against corpus (ChromaDB embedding check)
├── Insert into correct markdown section
├── Rebuild corpus.json
└── Clean _vault/
```

**Key insight**: Generation is embarrassingly parallel (no shared state). Merging is sequential (markdown file writes). Separate them completely.

---

## Phase 4: Quality at Scale (AFTER generation)

### Automated quality gates:
1. **Pydantic schema validation** — structure correct?
2. **Readability check** — appropriate for the level?
3. **Specificity check** — uses domain-specific terms?
4. **Napkin math depth** — enough calculation steps?
5. **Distractor quality** — non-trivial wrong answers?
6. **Corpus dedup** — not too similar to existing?
7. **Within-batch dedup** — not too similar to siblings?

### Human review workflow:
- Engine generates → auto-validate → flag bottom 20% for human review
- Human approves/rejects/edits → feedback loop to improve prompts
- Track quality score per model, per competency, per level over time

---

## Phase 5: The Product Loop (FUTURE)

```
GENERATE → VALIDATE → MERGE → SERVE → COLLECT → ANALYZE → REGENERATE
                                  ↓
                           StaffML App
                           (corpus.json)
                                  ↓
                          User Responses
                                  ↓
                       Difficulty Calibration
                       Discrimination Index
                       Concept Mastery Tracking
                                  ↓
                        Back to Generation Engine
                        (deprecate bad questions,
                         generate for weak concepts)
```

---

## Immediate TODOs

- [ ] Expert taxonomy review (4 perspectives: interviewer, hiring manager, candidate, academic)
- [ ] Close vault fill → merge → rebuild corpus
- [ ] Spot-check 50 generated questions manually
- [ ] Normalize tags in generated markdown to canonical taxonomy
- [ ] Update track README question counts
- [ ] Generate final HTML report with 3D cube at ~100% coverage
- [ ] Push to investigate async generation (Option C) for 10x throughput
