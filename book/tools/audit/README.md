# book/tools/audit — Pass 15 Audit-Fix-Verify Loop

Automated editorial audit pipeline that scans textbook content against the
MIT Press round 1 style rules, applies safe fixes under strict safety
gates, and verifies the result.

**Status:** Phase A complete (infrastructure + 7 check categories).
**Plan:** `/Users/VJ/Desktop/MIT_Press_Feedback/15_audit_loop/PLAN.md`
**Rules:** `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/rules/book-prose-merged.md`

---

## The five-stage cycle

```
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌─────────┐
│ 1 SCAN │ → │ 2 PLAN │ → │ 3 FIX  │ → │ 4 VERIFY│ → │ 5 REPORT│
└────────┘   └────────┘   └────────┘   └────────┘   └─────────┘
 read-only    classify     script       3 checks      commit
              by lane      lane         must pass
```

**Verification is the load-bearing stage.** If any check fails, the
cycle rolls back and does NOT retry. See `verify.py` for the three
verification stages.

---

## File layout

```
book/tools/audit/
├── README.md                      — this file
├── __init__.py
├── protected_contexts.py          — LineWalker + inline span detection
├── ledger.py                      — Issue + Ledger JSON model
├── scan.py                        — SCAN stage + CLI
├── accept_list.py                 — persistent FP accept-list (Pass 16 Item A)
├── accepted_fps.json              — seeded from Pass 15 FINAL ledgers (75 entries)
├── fix_script_lane.py             — FIX stage (script lane) + 5 safety checks
├── verify.py                      — VERIFY stage (3 checks)
├── loop.py                        — orchestrator CLI
├── checks/
│   ├── __init__.py
│   ├── vs_period.py               — bare 'vs' → 'vs.'
│   ├── compound_prefix.py         — pre-/non- close-up (strict 6-term list)
│   ├── percent_symbol.py          — '%' → 'percent' in body prose
│   ├── lowercase_prose_references.py  — 'Chapter 12' → 'chapter 12'
│   ├── acknowledgements_spelling.py   — British → American
│   ├── binary_units.py            — GiB/TiB in prose (detection only)
│   └── h3_titlecase.py            — H3+ headings in title case (detection only)
└── subagent_prompts/              — (Phase B) prompts for judgment-required checks
```

### Persistent accept-list (Pass 16 Item A)

`accept_list.py` + `accepted_fps.json` together encode Pass 15's editorial
verdict on 75 h3-titlecase scanner false positives (proper-noun-heavy
headings, named principles, legislation, after-colon CMS 8.158 caps,
D·A·M/C³ taxonomy axes). After every scan, matching issues are flipped
from `open` to `accepted` and tagged with the §10.9 sub-rule that
justifies them. Match key is `(category, repo-relative file, exact
`before` line)` — if a heading is intentionally edited, its accept-list
entry stops matching and the issue correctly returns to `open` for
re-review.

```bash
# Default: accept-list applied, summary shows matched + stale counts
python3 book/tools/audit/scan.py --scope vol1 -v

# Reproduce pre-Pass-16 behavior (all 75 FPs report as open)
python3 book/tools/audit/scan.py --scope vol1 --no-accept-list -v

# Use a different accept-list file (e.g. a draft to iterate on)
python3 book/tools/audit/scan.py --scope vol1 --accept-list /tmp/draft.json
```

---

## CLI usage

All commands are from the repo root.

### Scan only (dry run)

```bash
python3 book/tools/audit/scan.py --scope vol2 --verbose
python3 book/tools/audit/scan.py --scope vol1 --output vol1-ledger.json --verbose
```

Produces `audit-ledger.json` (or the path given by `--output`).

### Fix one category (dry run)

```bash
python3 book/tools/audit/fix_script_lane.py \
    --ledger audit-ledger.json \
    --categories vs-period \
    --dry-run --verbose
```

### Run the full loop

```bash
# Dry run (scan + plan + report, no file changes)
python3 book/tools/audit/loop.py --scope vol2 --dry-run --verbose

# Apply, verify, but don't commit
python3 book/tools/audit/loop.py --scope vol2 \
    --categories vs-period,compound-prefix-closeup \
    --apply --verbose

# Apply, verify, and commit each iteration
python3 book/tools/audit/loop.py --scope vol2 \
    --categories vs-period,compound-prefix-closeup \
    --apply --commit-each-iteration --verbose

# Add quarto check (expensive) to verify stage
python3 book/tools/audit/loop.py --scope vol2 \
    --categories vs-period --apply --quarto-check --verbose
```

---

## Check categories

| Category | Rule | Lane | Notes |
|---|---|---|---|
| `vs-period` | book-prose-merged §10.10 | script | Proven from pass 10b |
| `compound-prefix-closeup` | §10.8 | script | Strict 6-term list, no extrapolation |
| `percent-symbol` | §10.2 | script | HTML attribute filter (width=N%) |
| `lowercase-prose-references` | §10.4 | script | Hand-written "Chapter 12" |
| `acknowledgements-spelling` | §10.7 | script | British → American |
| `binary-units-in-prose` | §1 | accept | Detection only; needs human |
| `h3-titlecase` | §10.9 | subagent | Per-heading judgment required |

**Phase A covers all 7 categories above as detection.** Phase B adds
parallel subagent dispatch for `h3-titlecase`.

---

## Validation anchors (Phase A baseline)

Scan times on a cold run from the repo root:

```
$ python3 book/tools/audit/scan.py --scope vol1 -v
Total: 629 issues across 34 files (0.4s)

$ python3 book/tools/audit/scan.py --scope vol2 -v
Total: 969 issues across 39 files (0.4s)
```

Per-category counts (baseline for regression detection):

| Category | vol1 | vol2 |
|---|---:|---:|
| vs-period | 0 | 16 |
| compound-prefix-closeup | 19 | 46 |
| percent-symbol | 1 | 160 |
| lowercase-prose-references | 0 | 0 |
| acknowledgements-spelling | 0 | 0 |
| binary-units-in-prose | 0 | 0 |
| h3-titlecase | 609 | 747 |

The `h3-titlecase: 609` matches the Pass 15 plan's expected ~611 (off
by 2, within tolerance). The `vs-period: 0` on vol1 confirms pass 10b's
work is intact.

### Post-Pass-16 anchor (2026-04-08)

After Pass 15's 847 editorial fixes and Pass 16 Item A's persistent
accept-list, the scanner reports the following steady state. Use these
as the regression anchor going forward.

| Category | vol1 open | vol1 accepted | vol2 open | vol2 accepted |
|---|---:|---:|---:|---:|
| vs-period | 0 | 0 | 0 | 0 |
| compound-prefix-closeup | 0 | 0 | 0 | 0 |
| percent-symbol | 0 | 0 | 0 | 0 |
| lowercase-prose-references | 0 | 0 | 0 | 0 |
| acknowledgements-spelling | 0 | 0 | 0 | 0 |
| binary-units-in-prose | 0 | 0 | 0 | 0 |
| h3-titlecase | **0** | 22 | **0** | 53 |

Reproduce with `python3 book/tools/audit/scan.py --scope vol1 -v`. If
you see a non-zero `open` count in any row, either (a) a real new
violation has been introduced that needs fixing, or (b) a previously
accepted heading has been edited and its accept-list entry needs
updating (a stale warning will identify which one).

---

## Safety invariants

The script lane runs **five checks before writing any file**. A failure
on any one causes immediate rollback:

1. **No null bytes** — leftover nulls from broken sentinel pipelines
2. **No leftover sentinels** — `⟦SENT0⟧`-style markers from stash/restore
3. **Byte delta matches expectation** — caught the discarded bulk run
4. **Quarto structural delta is zero** — fence/div/YAML counts unchanged
5. **No new issues introduced** — re-runs ALL check modules on the new text

Safety check #3 (byte delta) is the most important. If you close up N
occurrences of `pre-training` (-1 char each) and fix M bare `vs`
(+1 char each), the file delta must be exactly `-N + M`. Anything else
means the script touched content it shouldn't have — this is the exact
failure mode that the discarded bulk-edit run had.

---

## Stopping conditions (hard-coded)

Per Pass 15 plan section 2.5:

1. **Zero issues remaining** in active categories → exit 0 (success)
2. **No progress** in an iteration → exit 2 (stuck)
3. **Verification failure** → exit 3 (do not retry)
4. **Time budget exceeded** (default 30 min wall) → exit 4 (budget)
5. **Max iterations reached** → exit 4 (budget)
6. **Commit failure** → exit 5

---

## Adversarial test coverage

- `protected_contexts.py`: 14 adversarial tests covering every failure
  mode from the discarded bulk-edit run (bold definition, callout title,
  table header, sentence start, index entry, @-ref, citation, footnote
  ref, fig-cap attr, inline code, inline math).
- `compound_prefix.py`: 21 tests covering the strict 6-term list,
  domain-compound preservation, acronym/proper-noun continuation, and
  case preservation.
- `vs_period.py`: validated against pass 10b's claim (vol1 clean,
  vol2 still has 16 real hits).

Run the adversarial tests inline with each check module during
development. A pytest-based test harness is a Phase C deliverable.

---

## Do not

- **Do not run `--apply` against vol1 without explicit human approval
  per category.** Vol1 is the MIT Press deliverable.
- **Do not skip the verify stage** (`--dry-run` is the only exception).
- **Do not retry a failed verification.** Inspect with `git diff` and
  either commit or roll back manually.
- **Do not add a new category without a baseline count** — every check
  must be validated against a known-good state before being trusted.
- **Do not commit to `main` or `dev`.** Always commit to the feature
  branch (`feat/mitpress-vol1-copyedit-r1` at the time of writing).

See Pass 15 plan section 10 for the complete "do not" list.
