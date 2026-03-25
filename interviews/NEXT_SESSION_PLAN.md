# Next Session: Validation & Error Fixes

**Goal:** Validate all 5,198 questions via Gemini CLI, fix flagged errors, re-export, commit.

**Start by reading this file, then execute phases in order.**

---

## Current State (as of 2026-03-24 late evening)

### What's Done
- **5,198 questions** in corpus.json (4,779 original + 419 generated L4-L6+)
- **6-axis faceted classification** at 100% coverage (track, level, RC, KA, mode, tags)
- **paper.tex** fully updated (working notes removed, v5.3 scrubbed, numbers current)
- **4 PDF figures** and `corpus_stats.json` regenerated
- **Committed** at `078ab8f64` on `dev` branch
- **Schema validation passes** — all 5,198 questions clean

### What's Left
1. **Validate all 5,198 questions** via Gemini (math, facts, quality)
2. **Fix ~61 flagged errors** from partial validation (truncated text, KV-cache math)
3. **Re-export** to StaffML and recommit

### Why Validation Failed Last Session
- Gemini API key quota: 250 requests/day on `gemini-3.1-pro` — exhausted by generation + validation
- Small batch size (20 Qs/call) meant 260 API calls needed — way over quota
- CLI fallback was too slow with small batches due to subprocess overhead

### The Fix: Large Batches via Gemini CLI
- Gemini 3.1 Pro has **1M token context** and **65K token output**
- 200 questions per batch ≈ ~100K input tokens — well within limits
- 5,198 / 200 = **26 CLI calls** total (vs 260 before)
- 3 parallel workers × ~30s per call ≈ **~4-5 minutes** total

---

## Phase 1: Validate via Gemini CLI (5 min)

### 1.1 Update validate_questions.py for large CLI batches

The `--cli` flag already exists (added last session). The key change is using
`--batch-size 200` to leverage Gemini's massive context window.

```bash
python3 validate_questions.py --cli --batch-size 200 --workers 3
```

This will:
- Send 26 batches of 200 questions to `gemini-3.1-pro-preview` via CLI (OAuth creds)
- Parse JSON review responses
- Stamp each question with `validated`, `validation_status`, `validation_issues`
- Save results to `_validation_results/`

### 1.2 If CLI is still slow, use the API with a different model

The API has huge quota on Flash:
```bash
# Gemini 2.5 Flash: 10,000 RPD — plenty of headroom
source ~/.zshrc_secrets
python3 validate_questions.py --batch-size 200 --workers 5
```
But change `MODEL = "gemini-2.5-flash"` in the script first.

---

## Phase 2: Fix Flagged Errors (10 min)

### Known issues from partial validation (61 ERRORs):

**Category 1: Truncated text (~40 questions)**
Old corpus questions with truncated scenario/solution/napkin_math fields. IDs include:
- `cloud-llm-serving-*-121` through `-136` (serving latency questions)
- `cloud-continuous-batching-*-22`, `-26`, `-30`
- `cloud-kv-cache-*-21`, `-25`, `-29`, `-31`

Fix: Load these questions, check which fields are truncated, and either:
- Regenerate the truncated fields via Gemini
- Or delete questions that are too broken to salvage

**Category 2: KV-cache math errors (~5 questions)**
Questions that don't account for Grouped Query Attention in Llama-70B:
- `cloud-kv-cache-vram-accounting-the-kv-cache-vram-budget-21`
- `cloud-kv-cache-vram-accounting-the-kv-cache-vram-budget-25`
- `cloud-kv-cache-vram-the-kv-cache-memory-bomb-29`

Fix: Update KV-cache calculations to use 8 KV-heads (GQA) not 64 (MHA).
Formula: `KV = 2 × layers × kv_heads × head_dim × seq_len × bytes`
For Llama-70B: `2 × 80 × 8 × 128 × seq_len × 2` (not `× 64`)

### Approach
```python
# Script to find and fix truncated questions
import json
corpus = json.load(open("corpus.json"))

truncated = [q for q in corpus if
    len(q["scenario"]) > 50 and q["scenario"].rstrip()[-1] not in ".?!\"')"
    or len(q["details"].get("realistic_solution", "")) > 50
    and q["details"]["realistic_solution"].rstrip()[-1] not in ".?!\"')"]

print(f"Found {len(truncated)} potentially truncated questions")
```

---

## Phase 3: Re-export & Commit (2 min)

```bash
python3 vault.py validate          # Confirm 0 errors
python3 vault.py export            # Sync to StaffML
python3 vault.py facets            # Verify 100% coverage

git add interviews/corpus.json interviews/validate_questions.py \
  interviews/staffml/src/data/corpus.json
git commit -m "staffml: validate 5,198 questions, fix truncated text and KV-cache math"
```

---

## Technical Notes

- **Gemini CLI** uses OAuth creds cached at `~/.config/gemini/`. No API key needed.
- **API key** is in `~/.zshrc_secrets` (NOT `~/.zshrc` — the one in `.zshrc` is stale)
- **API quotas** (as of 2026-03-24):
  - `gemini-3.1-pro`: 250 RPD (API key), unlimited via CLI (OAuth)
  - `gemini-2.5-pro`: 1,000 RPD
  - `gemini-2.5-flash`: 10,000 RPD
- **Schema validation**: `from schema import validate_corpus` returns `(valid, errors, warnings)`
- **Question ID format**: `<track>-<ka-lowercase>-<kebab-title>-<number>`
- **Level "L6+" not "L6"**: The plus sign is required in the level field

## Quick Start

```
Read interviews/NEXT_SESSION_PLAN.md.
Execute Phases 1-3 in order.
Use Gemini CLI (--cli flag) with large batches (--batch-size 200).
Always use `source ~/.zshrc_secrets` before Python scripts that need GEMINI_API_KEY.
```
