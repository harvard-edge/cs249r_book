# Phase 2 content audit — zone/level reclassifications

**Date:** 2026-04-21
**Triggered by:** schema v1.0 migration landing (PR #1426). That PR fixed the
deterministic migration defects; this Phase 2 pass reviewed the remaining
content-level zone/level classifications against the paper's §3.3 affinity
table.

## Method

1. `vault lint` flagged **1,606 zone-level affinity outliers** across the
   9,657-question corpus (questions whose `level` fell outside their
   `zone`'s natural range per paper Table 2).
2. Outliers were sorted by `(zone, level, id)` for coherence, then split
   into 11 shards of ~150 each (shard 11 has 106).
3. 11 parallel LLM agents reviewed their shards independently, applying
   the decision rubric at [`phase2_rubric.md`](./phase2_rubric.md). Each
   agent read each question's YAML directly (scenario + solution +
   mistake + napkin math), decided whether the current (zone, level)
   was defensible, and emitted a structured JSONL decision.
4. The 11 shard outputs were aggregated into
   [`phase2_aggregated.json`](./phase2_aggregated.json).
5. Reclassifications with `high` or `medium` confidence (813 total) were
   applied to the YAMLs. `low` confidence proposals (37) were preserved
   in the aggregated JSON but not applied — those represented only a
   mild preference.

## Results

| Decision | Count |
|---|---|
| `keep` (current labels defensible) | 756 |
| `reclassify` (proposed alternative) | 850 |
| &nbsp;&nbsp;of which `high` confidence | 401 |
| &nbsp;&nbsp;of which `medium` confidence | 412 |
| &nbsp;&nbsp;of which `low` confidence (not applied) | 37 |

**Top zone transitions applied:**
- `design` → `mastery`: 218 (compound diagnose+design+size at L6+)
- `evaluation` → `diagnosis`: 172 ("Diagnosing X" questions mis-zoned as evaluation)
- `diagnosis` → `evaluation`: 57 (quantitative tradeoff comparisons mis-zoned as diagnosis)
- `evaluation` → `analyze`: 38 (mechanism-explanation questions)
- `evaluation` → `fluency`: 36 (formula-application questions)
- `evaluation` → `design`: 35 (explicit architecture proposals)
- `recall` → `fluency`: 30 (recall + quantify computations)
- `realization` → `design`: 26 (realization was overused)

## Effect on outlier count

- **Before:** 1,606 affinity outliers
- **After:** 847 (-47%)

The residual 847 are cases where the reviewing agent decided the
current (technically-outside-affinity) label was still the best fit
given the content. Paper §3.3 states the affinity table is a soft
constraint, so residual outliers are expected and legitimate.

## Files preserved

- [`phase2_rubric.md`](./phase2_rubric.md) — the decision rubric given
  to each agent
- `shard_01.jsonl` ... `shard_11.jsonl` — per-shard decisions (1,606 total)
- [`phase2_aggregated.json`](./phase2_aggregated.json) — all 1,606
  decisions in one JSON array (input to the apply step)

## Reproducing

```
python3 -c "
import sys
sys.path.insert(0, 'interviews/vault-cli/src')
sys.path.insert(0, 'interviews/vault/schema')
from pathlib import Path
from vault_cli.loader import load_all
from enums import ZONE_LEVEL_AFFINITY
loaded, _ = load_all(Path('interviews/vault'))
outliers = [lq for lq in loaded
            if ZONE_LEVEL_AFFINITY.get(lq.question.zone)
            and lq.question.level not in ZONE_LEVEL_AFFINITY[lq.question.zone]]
print(f'affinity outliers: {len(outliers)}')
"
```

Should report `~847`.
