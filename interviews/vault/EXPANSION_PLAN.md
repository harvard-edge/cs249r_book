# StaffML Final Plan — Corpus Balancing + Paper Polish

## Goal

Bring the corpus to a balanced, validated, publication-ready state where:
- No track exceeds 42% of the corpus
- No zone has fewer than 200 questions
- No topic has fewer than 15 questions
- Validation rate exceeds 75%
- Chain coverage exceeds 60% for all tracks
- Platform diversity: each track references 3+ distinct hardware platforms

## Current State (April 1, 2026)

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Total published | 6,653 | 7,000+ | ~350 |
| Cloud share | 41.8% | ≤42% | OK |
| Edge share | 20.4% | ~20% | OK |
| Mobile share | 17.5% | ~17% | OK |
| TinyML share | 15.5% | ~15% | OK |
| Global share | 4.8% | ~6% | Need ~80 more |
| Validation rate | 53% | 75%+ | ~1,500 to verify |
| Chain coverage (edge) | 39% | 60%+ | Need chain building |
| Chain coverage (mobile) | 33% | 60%+ | Need chain building |
| Chain coverage (tinyml) | 33% | 60%+ | Need chain building |
| Realization zone | 145 (2.2%) | 200+ | ~55 more |
| Analyze zone | 268 (4.0%) | 300+ | ~32 more |
| Thin topics (<15 Qs) | 10 topics | 0 | ~100 Qs |

## Phase 1: Generation (fill content gaps)

### Campaign A: Thin Topics
**Target**: 10 topics with <15 questions each
**Method**: `expand_tracks.py` targeting specific topics across multiple tracks
**Budget**: 100 questions
**Workers**: 10 parallel gemini-3.1-pro-preview calls
**Est. time**: 8 min

Topics: queueing-theory (3), datacenter-efficiency (3), gradient-synchronization (4), rdma-transport (5), tail-latency (6), flash-attention (8), container-orchestration (9), scheduling-resource-management (10), model-tensor-parallelism (10), congestion-control (11)

### Campaign B: Realization Zone
**Target**: Bring from 145 → 200+ questions
**Method**: Targeted prompts for "you chose the architecture, now size it concretely"
**Budget**: 60 questions across cloud/edge/mobile
**Workers**: 10
**Est. time**: 5 min

### Campaign C: Global Track
**Target**: Bring from 319 → 400+ questions (cross-cutting principles)
**Method**: Generate track-agnostic questions testing fundamental laws
**Budget**: 80 questions
**Workers**: 10
**Est. time**: 6 min

### Campaign D: Mobile L5/L6+
**Target**: Bring mobile senior coverage from 28% → 32%
**Method**: L5/L6+ questions on mobile NPU delegation, battery optimization
**Budget**: 60 questions
**Workers**: 10
**Est. time**: 5 min

**Total generation: ~300 questions in ~20 min (run A-D in parallel)**

## Phase 2: Validation (verify all pending)

### Run validate_questions.py on ALL unvalidated questions
**Method**: `python3 scripts/validate_questions.py --batch-size 100 --workers 12`
**Scope**: ~2,000+ questions (existing pending + newly generated)
**Est. time**: 15-20 min (rate limited by Gemini API)

### Remove ERROR questions
After validation, remove questions with status=ERROR (typically factual errors or truncated text). Expected: 3-5% error rate → ~60-100 removals.

### Fix validation inconsistencies
Run `vault_invariants.py --fix` to sync validation_status ↔ validated fields.

## Phase 3: Chain Building

### Build depth chains for expanded tracks
For each topic×track combination with questions at 3+ Bloom's levels:
1. Group questions by (topic, track)
2. Sort by level (L1 → L6+)
3. Select one question per level to form a chain
4. Write chain to chains.json

**Script**: `python3 scripts/build_chains.py` (existing script or new)
**Target**: Chain coverage ≥60% for edge, mobile, tinyml

### Verify chain coherence
Each chain should progress from recall → apply → analyze → create with genuine cognitive escalation (not just longer scenarios).

## Phase 4: Dedup + Quality

### Title deduplication
Check for duplicate (track, level, title) combinations. Append topic qualifier to any duplicates.

### Truncation check
Remove any questions where scenario, napkin_math, or common_mistake ends mid-sentence (a known LLM generation artifact).

### Schema validation
Run full Pydantic validation: `python3 -c "from schema import validate_corpus; ..."`

### Invariant checks
`python3 scripts/vault_invariants.py` — must pass 19/19.

## Phase 5: Paper Update

### Rebuild stats and figures
```bash
cd interviews/paper
python3 analyze_corpus.py
python3 generate_figures.py
make svgs  # convert SVG→PDF
```

### Update paper macros
The \numquestions, \numtopics, etc. macros in paper.tex should be updated to match corpus_stats.json.

### Rebuild paper PDF
```bash
make paper  # pdflatex + bibtex
```

## Phase 6: Commit and Push

### Atomic commits
1. `feat(staffml): fill thin topics and weak zones (Campaign A-D)`
2. `feat(staffml): validate corpus with gemini-3.1-pro-preview`
3. `feat(staffml): build depth chains for expanded tracks`
4. `docs(staffml): update paper stats and figures`

## Execution Timeline

```
T+0     Launch Campaigns A-D in parallel (4 scripts, ~40 workers)
T+20    All generation complete. Merge batches into corpus.
T+22    Launch validation on all pending (12 workers)
T+40    Validation complete. Remove errors. Fix inconsistencies.
T+42    Build chains for edge/mobile/tinyml.
T+45    Dedup + truncation check + schema validation.
T+48    Invariant checks (19/19 must pass).
T+50    Rebuild paper stats + figures.
T+55    Commit and push.
T+60    Done.
```

## Verification Criteria (must ALL pass)

- [ ] No track exceeds 42%
- [ ] All zones have ≥200 questions
- [ ] All topics have ≥15 questions
- [ ] Validation rate ≥75%
- [ ] Chain coverage ≥55% for edge, mobile, tinyml
- [ ] 0 duplicate (track, level, title) combinations
- [ ] 19/19 invariant checks pass
- [ ] Paper builds with 0 undefined references
- [ ] Platform diversity: 3+ platforms per track in new questions

---

## Part B: Paper Polish (the REAL priority)

The corpus is large enough. The paper needs systematic polish. These items have been requested twice and must be addressed.

### B1. Figure 3 (prereq graph) — REDESIGN
The 79-node graph is unreadable at column width. Options:
1. Show ONLY one competency area (e.g., compute: 6 nodes, clear edges) as an example
2. Use a DOT graph with `rankdir=TB`, larger font, fewer nodes
3. Replace with a textual description + small illustrative subgraph

**Recommendation**: Show the compute area subgraph (6 topics, clean prerequisite chain: roofline → gpu-arch → systolic-dataflow, roofline → accelerator-comparison, etc.) as Figure 3, and put the full graph in the appendix or remove it entirely.

### B2. Related Work — Move earlier or weave in
The related work section dumps citations at the end. Instead:
- Move the LeetCode/Huyen comparison INTO the Introduction (it's already partially there)
- Move the Bloom's/IRT/ECD methodology citations into the sections where they're used
- Keep a SHORT "Positioning" section (0.5 page) that says "Here's where StaffML sits relative to prior art"

### B3. Numbers verification
Write a script that:
1. Reads corpus_stats.json
2. Generates a LaTeX `\input` file with all macros computed from data
3. paper.tex `\input`s this file instead of hand-typing macros

This ensures numbers are ALWAYS correct and ALWAYS from one source.

### B4. Figure sizing loop
For each figure:
1. Build paper
2. Screenshot each page
3. Check: is the figure legible? Is font ≥7pt? Is there whitespace balance?
4. If not, adjust figsize in generate_figures.py or width in paper.tex
5. Rebuild and re-check

### B5. Code listing polish
- Add `xleftmargin=3em` to prevent line numbers from sticking out
- Add `backgroundcolor=\color{lightgray}` for visual separation
- Reduce listing font to `\footnotesize`

### B6. Limitations — simplify
Remove:
- "Cloud-heavy corpus" (already fixed)
- "Quantify skill conflation" (already addressed in footnote)

Keep:
- No empirical validation (honest, important)
- Hardware constants require maintenance (fundamental)
- Hardware vendor concentration (NVIDIA-centric, important)
- Static questions (softened with chain-based progression)

### B7. Writing style pass
One final pass through the entire paper:
- Replace remaining contractions with full forms
- Mix up "X: Y" patterns with "X. Y" and "X, which Y"
- Ensure no section titles wrap (screenshot verify)
- All \citep instead of footnotes for references

## Part B Execution (in a fresh conversation)

```
Step 1: Fix Figure 3 (compute subgraph only)          — 10 min
Step 2: Write auto-macro script (numbers from data)   — 15 min
Step 3: Related work restructure                       — 20 min
Step 4: Writing style pass (contractions, colons)      — 15 min
Step 5: Limitations simplify                           — 5 min
Step 6: Code listing polish                            — 5 min
Step 7: Build → screenshot → fix loop (3 iterations)   — 30 min
Step 8: Final commit and push                          — 5 min
```

**Part B should be done in a FRESH conversation** to avoid context compression issues. The detailed feedback is saved at:
`.claude/_reviews/author-paper-feedback-detailed.md`

## Key Scripts

| Script | Purpose | Parallelism |
|--------|---------|-------------|
| `scripts/expand_tracks.py` | Generate questions with platform cycling | 10-12 workers |
| `scripts/fill_zone_gaps.py` | Generate zone-targeted questions | 8-10 workers |
| `scripts/validate_questions.py` | Gemini math/fact verification | 10-12 workers |
| `scripts/vault_invariants.py` | 19 structural checks | Sequential |
| `paper/analyze_corpus.py` | Compute stats from corpus | Sequential |
| `paper/generate_figures.py` | Matplotlib + seaborn figures | Sequential |
