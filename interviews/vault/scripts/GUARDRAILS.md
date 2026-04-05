# StaffML Data Quality Guardrails

How to prevent taxonomy/corpus drift, duplicates, and normalization decay.

## The Problem

Issues recur because they enter through **four ingestion points**, each unguarded:

| Ingestion Point | Script | What Goes Wrong |
|---|---|---|
| **Taxonomy extraction** | `extract_taxonomy.py --merge` | LLM produces Title Case names → creates non-kebab IDs and name duplicates |
| **Question generation** | `generate.py`, `generate_gaps.py` | LLM uses free-text `competency_area` instead of canonical enum; `primary_concept` doesn't match taxonomy |
| **Gap filling** | `vault_fill.py`, `fill_gaps.sh` | Adds questions tagged to concepts that don't exist in taxonomy |
| **Manual edits** | Direct corpus.json edits | `question_count` goes stale; `L6` used instead of `L6+` |

## The Solution: Three Layers

### Layer 1: Invariant Checks (catch at commit time)

```bash
python3 scripts/vault_invariants.py          # 14 checks, exit 1 on FAIL
python3 scripts/vault_invariants.py --fix    # Auto-fix what's fixable
python3 scripts/vault_invariants.py --json   # Machine-readable for CI
```

**Run this after every pipeline operation.** It catches:
- Duplicate concept names/IDs (Check 1-2)
- Non-kebab-case IDs from LLM extraction (Check 3)
- Stale `question_count` (Check 4, auto-fixable)
- Corpus↔taxonomy concept drift (Check 5, 14)
- Orphan prerequisites (Check 6)
- Graph cycles (Check 7)
- Non-canonical competency areas (Check 8)
- Non-canonical levels like `L6` (Check 9, auto-fixable)
- Duplicate question IDs (Check 10)
- Broken chain references (Check 11)
- Duplicate titles (Check 12, warn)
- Disconnected singleton concepts (Check 13, warn)

### Layer 2: Pipeline Integration (catch at generation time)

Add invariant checks to the existing workflow scripts so issues are caught
immediately, not discovered later.

**In `extract_taxonomy.py --merge`:** After merging, run checks 1-3 and
reject the merge if new duplicates or non-kebab IDs are introduced.

**In `generate.py` and `generate_gaps.py`:** After generating a batch,
validate that all new questions have canonical `competency_area` and that
`primary_concept` exists in `taxonomy.json`. Reject non-conforming questions
before they enter the corpus.

**In `vault_fill.py`:** After filling, run the full invariant suite. If
new FAILs are introduced (compared to the pre-fill baseline), abort.

**In `scorecard.py`:** Add invariant check results to the scorecard output
so drift is visible in every health report.

### Layer 3: Extraction Hardening (prevent at source)

The root cause of most issues is the LLM extraction prompt producing
free-form concept names that don't match existing taxonomy entries.

**Fix the extraction prompt** to:
1. Include the current taxonomy concept list in the prompt context
2. Instruct the LLM to reuse existing concept IDs when the concept already exists
3. Force kebab-case output format for new concept IDs
4. Validate extraction output against the existing taxonomy before merging

**Fix the generation prompt** to:
1. Include `VALID_AREAS` from `schema.py` in the prompt
2. Include valid concept IDs from `taxonomy.json` in the prompt
3. Reject generated questions that use non-canonical values

## Integration Checklist

After implementing the remediation plan:

- [ ] `vault_invariants.py` passes with 0 FAIL
- [ ] `extract_taxonomy.py --merge` runs invariant checks 1-3 after merge
- [ ] `generate.py` validates competency_area against VALID_AREAS before writing
- [ ] `scorecard.py` includes invariant check summary
- [ ] WORKFLOW.md updated to include invariant check step
- [ ] CI workflow runs `vault_invariants.py --json` on vault data changes

## Suggested CI Workflow

```yaml
# .github/workflows/vault-validate.yml
name: Vault Data Validation
on:
  push:
    paths:
      - 'interviews/vault/corpus.json'
      - 'interviews/vault/taxonomy.json'
      - 'interviews/vault/chains.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install pydantic
      - name: Schema validation
        run: |
          cd interviews/vault
          python3 -c "
          import json
          from schema import validate_corpus
          corpus = json.load(open('corpus.json'))
          valid, errors, warnings = validate_corpus(corpus)
          print(f'Valid: {len(valid)}, Errors: {len(errors)}, Warnings: {len(warnings)}')
          if errors:
              for e in errors[:20]:
                  print(f'  ERROR: {e}')
              exit(1)
          "
      - name: Invariant checks
        run: |
          cd interviews/vault
          python3 scripts/vault_invariants.py --json > invariants.json
          python3 scripts/vault_invariants.py
```

## The Golden Rule

**Every script that writes to corpus.json or taxonomy.json must run
`vault_invariants.py` before and after, and abort if new FAILs appear.**
