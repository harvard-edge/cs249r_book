# Phase 6+ handoff — resume guide for the next session

**Status as of 2026-05-04:**
Phases 0-5 (autonomous portion) complete. **2,279 of 2,757 proposed
corrections applied + validated.** 478 outliers documented in
`PHASE_5_UNRESOLVED.md` for human review. Tree clean, validators green.

**Branch:** `yaml-audit` (106 commits ahead of `origin/dev`, 0 behind)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Active workplan:** `interviews/vault-cli/docs/CORPUS_HARDENING_PLAN.md`

---

## What's done

```diff
+ Phase 0  Cleanup deprecated scripts                                ✅
+ Phase 1  Provenance backfill (407 YAMLs)                          ✅
+ Phase 2  AUTHORING.md + vault new scaffold                        ✅
+ Phase 3  audit_corpus_batched.py + _judges.py + _batching.py      ✅
+ Phase 4  Full-corpus audit (9,446) + backfills + merge            ✅
+ Phase 5  autonomous mass-apply: 2,279 corrections applied         ✅
            └─ 2,075 low-risk (format/title/level) via mass_apply
            └─   204 math fixes via verify_math_corrections
+ Phase 8  cron workflow + vault audit CLI subcommand               ✅
```

---

## What remains

```yaml
Phase 5 cleanup:        478 unresolved corrections (see PHASE_5_UNRESOLVED.md)
                          - 75 math 'no'      ← highest priority
                          - 14 math 'unclear'
                          -168 relabel-up
                          -138 chain-block
                          - 13 math+level-block
                          - 70 already-applied (no action)
Phase 6:                schema tightening (LinkML pattern + Pydantic forbid + lift gate)
Phase 7:                title-quality verification re-audit
Phase 9:                paper.tech update + tag vault-1.0.0
```

Total estimated remaining: **~9 hours** of work, mostly:
- Phase 5 unresolved review: 4-6h human
- Phase 6 schema work: 2h code
- Phase 7 verification: 30min
- Phase 9 release: 1h

---

## How to resume

### Step 0 — sanity check the worktree

```bash
cd /Users/VJ/GitHub/MLSysBook-yaml-audit
git status                       # should be clean
git log --oneline -10            # confirm Phase 5 commits visible
git branch                       # * yaml-audit
vault check --strict             # 10,711 loaded, 0 invariant failures
pytest interviews/vault-cli/tests/ -q  # 84 passed
ruff check interviews/vault-cli  # clean
```

### Step 1 — disposition the 478 unresolved corrections

Read `interviews/vault-cli/docs/PHASE_5_UNRESOLVED.md` first for the
full breakdown. Suggested order, by priority:

#### Step 1a — math 'no' verdicts (75 questions, highest priority)

Independent Gemini check disputed Gemini's first proposed math fix.
For each question, two interpretations:
- First proposal was wrong (Gemini hallucinated)
- Second pass was overly strict (the fix is fine)

Read the `level_fit_rationale` + `coherence_rationale` + `math_errors`
in `01_audit.json` to understand the original failure, then look at
`suggested_corrections.napkin_math` in `01_audit.json`, then decide.

```bash
# Walk these interactively:
vault audit review \
    --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
    --filter-gate math_correct \
    --limit 25
```

#### Step 1b — relabel-up cases (168 questions)

Each is a question Gemini judged "deserves a HIGHER level than claimed."
Per §10 Q3, default policy is relabel-DOWN, but these go the other way.
Two paths:
- Accept the relabel-up (the question really IS bigger than its label)
- Rewrite the question DOWN to actually match the claimed level

Triage in batches by track + topic. Open issue for each chunk that
needs authoring follow-up.

#### Step 1c — chain-monotonicity blocks (138 + 13 = 151)

A level relabel was blocked because applying it would break the
chains.json non-decreasing-level invariant.

For each, you need to either:
1. Move the question OUT of the chain (`vault chain unlink ...`),
   then apply the relabel, OR
2. Restructure the chain itself (merge / split / reorder).

These are chain-team decisions; not pure Phase 5 work.

#### Step 1d — math 'unclear' (14 questions)

Same workflow as 1a but Gemini was less confident. Defaulted to skip.
Manually inspect.

#### Step 1e — already-applied (70 questions)

No action needed. The YAML's current state already matches the proposed
correction.

### Step 2 — Phase 6: tighten schema + lift format gate

Once corpus is clean (Step 1 done; or accept the residuals as known-deferred):

```yaml
files to edit:

  interviews/vault/schema/question_schema.yaml (LinkML, source of truth):
    Details.common_mistake:
      pattern: '(?s).*\*\*The Pitfall:\*\*.*\*\*The Rationale:\*\*.*\*\*The Consequence:\*\*.*'
    Details.napkin_math:
      pattern: '(?s).*\*\*Assumptions.*\*\*Calculations:\*\*.*\*\*Conclusion.*'
    Question.provenance:
      required: true   # was implicitly defaulted to "imported"

  interviews/vault-cli/src/vault_cli/models.py (Pydantic, derived):
    Details.model_config: ConfigDict(extra="forbid")   # was "allow"
    # Survey first; we already verified 0 unknown extras on Details
    # across 9,446 published YAMLs (2026-05-03 prep).
    # Question can stay extra="allow" for forward-compat
    # (audit-stamp fields like validation_status, math_status).

  interviews/vault-cli/src/vault_cli/validator.py:
    structural_tier:
      append _format_compliance() check
      (lift gate_format from validate_drafts.py / _judges.py
       to a published-corpus invariant)

run:
  vault codegen          # regenerate Pydantic / SQL DDL / TS types
  pytest                 # add tests covering new invariants
  vault check --strict   # 0 failures expected (corpus is clean)
```

Test plan: lift the LinkML pattern, run `vault codegen`, run pytest. If any test fails because a YAML doesn't match the new pattern, that YAML wasn't covered by Phase 5 — fix the YAML, not the schema.

### Step 3 — Phase 7: title verification

Phase 5 already applied 79 title corrections. To verify they took
correctly, run a small re-audit on those qids:

```bash
# Pull qid list from the disposition log:
python3 -c "
import json
d = json.loads(open('interviews/vault/_pipeline/runs/full-corpus-20260503-merged/02_mass_apply.json').read())
qids = [d['qid'] for d in d['dispositions']
        if d.get('result') == 'applied' and d.get('category') == 'title-only']
print(','.join(qids))
" > /tmp/title-fixed-qids.txt

# Re-audit those:
QIDS=$(cat /tmp/title-fixed-qids.txt)
vault audit run \
  --qids "$QIDS" \
  --workers 8 \
  --max-calls 5 \
  --output interviews/vault/_pipeline/runs/title-verify-20260504
```

Expect every title to come back `title_quality: good`.

### Step 4 — Phase 9: paper.tech + release

```bash
# Update paper.tech with post-Phase-5 corpus stats:
#   - 9,446 published, audit pass rates per gate, per-track tables
#   - Methodology paragraph naming gemini-3.1-pro-preview as audit model
#   - Citation of audit_corpus_batched.py + AUDIT_FINDINGS_<date>.md

vault export-paper
vault build --local-json   # release_hash should roll
vault publish 1.0.0
vault verify 1.0.0 --git-ref v1.0.0   # citation-grade round-trip

git tag vault-1.0.0
```

---

## Reference docs (in this worktree)

| doc | purpose |
|---|---|
| `interviews/vault-cli/docs/CORPUS_HARDENING_PLAN.md` | full 9-phase workplan (the spec) |
| `interviews/vault-cli/docs/PHASE_4_HANDOFF.md` | original Phase 4 handoff (now historical) |
| `interviews/vault-cli/docs/PHASE_5_UNRESOLVED.md` | the 478 unresolved corrections + per-category review workflow |
| `interviews/vault-cli/docs/PHASE_6_HANDOFF.md` | this doc — resume guide |
| `interviews/vault-cli/docs/AUDIT_FINDINGS_2026-05-03.md` | Phase 4 corpus snapshot before corrections |
| `interviews/vault/AUTHORING.md` | single-source authoring reference |

## Gemini-driven scripts (in vault-cli/scripts/)

| script | purpose |
|---|---|
| `audit_corpus_batched.py` | full-corpus audit with optional `--propose-fixes` |
| `apply_corrections.py` | INTERACTIVE accept/reject of proposed corrections |
| `mass_apply_corrections.py` | AUTONOMOUS apply of low-risk corrections (already used) |
| `verify_math_corrections.py` | independent Gemini verify-then-apply for math fixes (already used) |
| `summarize_audit.py` | generate AUDIT_FINDINGS markdown |
| `merge_audit_runs.py` | merge per-track audit dirs into one canonical |
| `_judges.py` / `_batching.py` | shared helpers |

## Pipeline data on disk (gitignored, ready for use)

```
interviews/vault/_pipeline/runs/
├── full-corpus-20260503/                  main audit (cloud + edge + global)
├── full-corpus-20260503-mobile/           parallel mobile run
├── full-corpus-20260503-tinyml/           parallel tinyml run
├── full-corpus-20260503-cloud-backfill/   cloud propose-fixes backfill
├── full-corpus-20260503-edge-backfill/    edge propose-fixes backfill
├── full-corpus-20260503-merged/           ← canonical merged dataset
│   ├── 01_audit.json                       9,446 rows, all gates verified
│   ├── 02_mass_apply.json                  per-qid disposition for low-risk apply
│   ├── 03_math_verification.json           per-qid Gemini verification for math fixes
│   └── 04_math_applied.json                per-qid disposition for math apply
└── (older runs from earlier in workflow)
```

The merged 01_audit.json is the input for ALL Phase 5 review and Phase 7
verification.

---

## Final commit log highlights from this session

```
79b4c3361  docs(vault-cli): PHASE_5_UNRESOLVED.md — list of corrections needing human review
f4d219ab2  fix(vault): apply 204 Gemini-verified math corrections (Phase 5 math leg)
04c69e6a5  feat(vault-cli): verify_math_corrections.py — Phase 5 math-fix verifier
15811ef4b  feat(vault-cli): mass_apply_corrections.py — Phase 5 low-risk auto-applier
e62e7e27b  fix(vault): apply 2,075 low-risk Gemini-proposed corrections (Phase 5 mass-apply)
9ee3c3430  docs(vault-cli): PHASE_4_HANDOFF — update post-backfill
87481ab6a  docs(vault-cli): refresh AUDIT_FINDINGS_2026-05-03 after Phase 4 backfill
2131696b8  fix(vault/cloud): move stray top-level options/correct_index into details
68012912f  feat(vault-cli): vault audit CLI subcommand — Phase 8
d2621cc9e  feat(vault-cli): merge_audit_runs.py + Phase 4 findings doc
2d9330da6  fix(vault-cli): isolate gemini CLI scratch files in temp dir
e7a2a27bf  feat(ci): staffml-audit-corpus-monthly.yml — recurring corpus audit workflow
3eaac3ca9  feat(vault-cli): summarize_audit.py — Phase 4 finalization helper
1722133fa  feat(vault-cli): apply_corrections.py — interactive accept/reject
1b58a9c50  feat(vault-cli): parallel audit_corpus_batched.py with submit-stagger
69cf6f0a5  feat(vault-cli): audit_corpus_batched.py — full-corpus batched audit
dd71c66ca  feat(vault-cli): _judges.py + _batching.py — shared infra
f691d6c14  feat(vault-cli): vault new scaffolds full Pitfall/Rationale/Consequence stubs
7500b9281  docs(vault): AUTHORING.md — single-source authoring reference
e8f0faa83  chore(vault): explicit provenance: imported on 407 published questions
56d3ed155  chore(vault): remove 18 deprecated scripts per CORPUS_HARDENING_PLAN.md Phase 0
36f2ef592  docs(vault-cli): CORPUS_HARDENING_PLAN.md — supersedes RELEASE_AUDIT_PLAN.md
```

---

## When ready to land on dev

```bash
cd /Users/VJ/GitHub/MLSysBook   # main worktree where dev is checked out
git pull origin dev
git merge --no-ff yaml-audit -m "Merge yaml-audit — Phase 0-5 of corpus hardening"
# (don't push until ready)
```

Or via PR if you'd rather review on GitHub.

The merge will land 2,279 corrected YAMLs + ~30 new tooling/doc files. No
schema changes (Phase 6 is gated on Phase 5 unresolved cleanup).

---

## Resume prompt for the next Claude Code session

```
Resume yaml-audit branch. Phase 5 autonomous portion is done; 2,279 of
2,757 corrections applied. Read interviews/vault-cli/docs/PHASE_6_HANDOFF.md
top to bottom — it's the self-contained resume guide.

Then start at Step 0 (sanity check), then Step 1 (disposition the 478
unresolved corrections in PHASE_5_UNRESOLVED.md), then Step 2 (Phase 6
schema tightening), Step 3 (Phase 7 title verification), Step 4 (Phase 9
release).

Total estimated remaining: ~9 hours, mostly Phase 5 review + Phase 6
schema work.
```
