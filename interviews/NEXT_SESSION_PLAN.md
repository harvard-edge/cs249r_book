# Next Session: WARN Fixes + Coverage Growth

**Goal:** Push production-ready from 63% → 80%+ by fixing WARNs and growing coverage.

**Start by reading this file, then execute phases in order.**

---

## Current State (as of 2026-03-25 early morning)

### What's Done (this session)
- **Three-model validation** (Flash → Pro → Opus) on all 5,228 questions
- **414 duplicates removed** (same-track, same-title)
- **366 content fixes** (math errors, truncated text) via 28 parallel Opus agents
- **1,169 taxonomy fixes** (KA, taxonomy_concept, reasoning_mode, level) via 5 Opus audits
- **144 mode reclassifications** (concept-recall → napkin-math)
- **30 new global questions** generated (L1/L3/L5)
- **ERRORs reduced 648 → 13** (98% reduction)
- Committed at `62d8c383d` on `dev`

### Current Numbers
- **4,802 active questions** (426 archived/deleted)
- **OK: 3,041 (63.3%)**
- **WARN: 1,742 (36.3%)**
- **ERROR: 13 (0.3%)**
- Schema validation: all 5,228 pass

### What's Left
1. **Fix ~1,742 WARN questions** — mostly incomplete napkin_math stubs
2. **Fix 13 remaining ERRORs** — math inconsistencies
3. **Grow global track** — still sparse (L3=12, needs more)
4. **Grow thin KAs** — 12 areas with <40 questions each
5. **Update paper.tex** figures and coverage section

---

## Phase 1: Fix WARNs in Waves (30 min)

The dominant WARN issue is incomplete `napkin_math` — stubs that start a calculation but never finish. These need Opus to complete them with physics-grounded math.

### Strategy: Batch by issue type, fix with parallel Opus agents

```bash
# 1. Extract WARN questions, categorize by issue
python3 -c "
import json
c = json.load(open('corpus.json'))
warns = [q for q in c if q.get('validation_status') == 'WARN' and q.get('status') != 'deleted']
print(f'Total WARNs: {len(warns)}')
"

# 2. Split into batches of 25, launch Opus agents to complete napkin_math
# Use mlsysim/core/formulas.py as ground truth for calculations
# Hardware specs: A100 2TB/s 312TFLOPS, H100 3.35TB/s 989TFLOPS
# KV-cache: 2 * L * kv_heads * head_dim * seq_len * bytes

# 3. Validate fixes with Flash (batch=10, workers=10)
source ~/.zshrc_secrets
PYTHONUNBUFFERED=1 python3 validate_questions.py --batch-size 10 --workers 10

# 4. Re-validate ERRORs with Pro for consensus
```

---

## Phase 2: Coverage Growth (20 min)

### Priority gaps:
- **Global track**: L3=12 (need 20+), still the weakest track
- **Thin KAs** (< 40 questions): A1, A2, A3, A4, A6, B4, B8, C4, C7, C8, C9, F1
- **L6+ questions**: Only 496 across corpus

Generate with Opus, validate with Flash + Pro.

---

## Phase 3: Paper Update (10 min)

After fixes:
```bash
python3 vault.py stats              # Regenerate corpus_stats.json
python3 vault.py export             # Sync to StaffML
# Update paper.tex \newcommand values
# Regenerate figures if needed
```

---

## Technical Notes

- **Validation command**: `source ~/.zshrc_secrets && PYTHONUNBUFFERED=1 python3 validate_questions.py --batch-size 10 --workers 10`
- **Model**: `gemini-2.5-flash` (10K RPD) for bulk, `gemini-2.5-pro` (1K RPD) for second opinions
- **Always** use `PYTHONUNBUFFERED=1` for background scripts
- **Physics formulas**: `/Users/VJ/GitHub/MLSysBook/mlsysim/core/formulas.py`
- **Hardware specs source of truth**: mlsysim constants
- **Small batches, fast feedback**: batch=10, workers=10, verify output within 15 seconds

## Quick Start

```
Read interviews/NEXT_SESSION_PLAN.md.
Execute Phases 1-3 in order.
Use small batches (10) with high parallelism (10 workers).
Launch parallel Opus agents for fixes.
Always validate with Flash, confirm with Pro.
```
