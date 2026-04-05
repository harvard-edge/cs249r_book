# StaffML Iterative QA Workflow

A standardized, reproducible pipeline for maintaining question quality.

## The Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   MEASURE → DIAGNOSE → FIX → VALIDATE → MEASURE        │
│       ↑                                         │       │
│       └─────────────────────────────────────────┘       │
│                                                         │
│   Exit when: OK% > 95%, WARNs < 5%, ERRORs = 0         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Step 1: MEASURE (scorecard.py)

```bash
python3 staffml/vault/scripts/scorecard.py
```

Computes:
- OK / WARN / ERROR / Pending counts and percentages
- 6-axis classification completeness
- Bloom balance deviation per track
- Reasoning mode CV per track
- Chain coverage per track
- Short solutions / napkin math counts
- Taxonomy health (orphans, missing refs, overloaded concepts)
- Duplicate candidates (scenario similarity > 0.85)

Output: `_validation_results/scorecard_YYYYMMDD.json`

## Step 2: DIAGNOSE (auto)

From the scorecard, automatically prioritize:
1. ERRORs → must fix (highest priority)
2. Short napkin math (<100 chars) → incomplete stubs
3. Missing taxonomy axes → 6-axis classification gaps
4. Bloom balance gaps > 5% → generation targets
5. Chain coverage < 50% → chain building needed
6. Duplicates → dedup

## Step 3: FIX (parallel agents)

Three fix modes, chosen automatically based on diagnosis:

### Mode A: Math/Content Fix (for ERRORs and WARNs)
```bash
python3 staffml/vault/scripts/fix_warns.py --model gemini-3.1-pro-preview --workers 8 --batch-size 25
# Fallback: --model opus if Gemini quota hit
```
- Reads questions with WARN/ERROR status
- Sends to LLM with hardware reference sheet
- LLM diagnoses issue and returns corrected fields
- Applies fixes, sets status to OK

### Mode B: Generation (for Bloom/mode gaps)
```bash
python3 staffml/vault/scripts/generate_gaps.py --model gemini-3.1-pro-preview --workers 6
```
- Reads scorecard gap analysis
- Generates targeted questions for worst imbalances
- Validates immediately in a second pass
- Adds clean questions to corpus

### Mode C: Chain Building (for coverage gaps)
```bash
python3 staffml/vault/scripts/build_chains.py --model gemini-3.1-pro-preview --track edge
```
- Finds unchained questions with 3+ Bloom levels per topic
- LLM selects best question at each level
- Outputs chain JSON, backpopulates corpus

## Step 4: VALIDATE (separate pass)

```bash
python3 staffml/vault/scripts/gemini_math_review.py --batch-size 40 --workers 8
```
- Reviews ALL questions (or just newly fixed ones with `--status OK --since-date`)
- Chunks by track × topic for topical coherence
- Hardware reference sheet from mlsysim/core/constants.py
- Outputs OK/ERROR/WARN per question

## Step 5: MEASURE again → compare to previous scorecard

```bash
python3 staffml/vault/scripts/scorecard.py --compare _validation_results/scorecard_previous.json
```

## Automation: One Command

```bash
python3 staffml/vault/scripts/qa_loop.py --rounds 3 --target-ok 95
```

This script:
1. Runs scorecard
2. Diagnoses top issues
3. Launches appropriate fix mode (A/B/C) with available model
4. Validates fixes
5. Re-runs scorecard
6. If OK% < target, loops back to step 2
7. Commits when target reached or rounds exhausted

## Model Selection

| Model | Best for | Rate limit | Cost |
|-------|----------|------------|------|
| gemini-3.1-pro | Math review, generation | 250/day | Free tier |
| gemini-2.5-flash | Bulk validation | 1500/day | Free tier |
| claude-opus-4.6 | Deep review, complex fixes | No daily limit | API cost |

Strategy: Use Gemini for bulk, Opus for deep fixes, Flash for validation sweeps.

## Hardware Reference Source of Truth

All prompts include specs from `mlsysim/core/constants.py`:
- GPU: A100/H100/V100/T4 memory, bandwidth, TFLOPS
- Interconnect: NVLink, PCIe, InfiniBand
- Energy: Horowitz 2014 values
- Models: GPT-3, LLaMA-2/3, BERT params/layers/dims
- Edge: Jetson Orin, Coral, Hailo-8
- TinyML: Cortex-M4/M7, ESP32, STM32, nRF52840

Never hardcode specs in prompts — always derive from constants.py.

## Step 6: INVARIANT GATE (before commit)

```bash
python3 staffml/vault/scripts/vault_invariants.py
```

This runs 14 structural checks across corpus, taxonomy, and chains:
- No duplicate concept names or IDs
- All IDs are kebab-case (no Title Case stubs from LLM extraction)
- `question_count` matches actual corpus counts (auto-fixable with `--fix`)
- All `competency_area` and `level` values are canonical
- No orphan prerequisites or graph cycles
- All chain question IDs exist in corpus
- No duplicate question IDs

**The commit MUST NOT proceed if any check is FAIL.** Run `--fix` first
for auto-fixable issues (checks 4 and 9).

For pipeline scripts, use the gate module:
```python
from gate import InvariantGate
with InvariantGate():
    # ... modify corpus/taxonomy ...
# automatically blocks if new FAILs appear
```

## Commit Protocol

After each successful loop iteration:
```bash
# Verify invariants pass
python3 staffml/vault/scripts/vault_invariants.py
# Then commit
git add corpus.json chains.json staffml/src/data/corpus.json taxonomy.json
git commit -m "staffml: QA loop round N — OK X% → Y%, N fixes applied"
git push origin dev
```
