# binder/tools/audit — Editorial and Registry Audits

Detection tooling that scans textbook content against the house style rules
and the MLSys·im registry, plus the verification scripts that back the book
checks.

**Fmt / notation audit** (inline `{python}` numbers, spurious `.0`): see
[`fmt/README.md`](fmt/README.md).

### Registry migration build (constants → zoos)

End-to-end gate for the completed `mlsysim` registry migration:

```bash
python3 binder/tools/audit/check_registry_migration_build.py
```

Individual tools:

| Script | Purpose |
|--------|---------|
| `book_check_registry_sources.py` | Ban legacy flat constants / alias paths in QMD LEGO cells |
| `generate_appendix_constants.py --verify` | Execute appendix assumption-table cells against live registries |
| `refresh_mlsysim_constants_yamls.py --finalize` | Mark audit YAML inventory `should_change: false` |
| `audit_mlsysim_drift.py` | Detect hand-coded canonical constant drift |

---

## Style scan

```
binder/tools/audit/
├── README.md                      — this file
├── __init__.py
├── protected_contexts.py          — LineWalker + inline span detection
├── ledger.py                      — Issue + Ledger JSON model
├── scan.py                        — scanner CLI
├── accept_list.py                 — persistent false-positive accept-list
├── accepted_fps.json              — accepted false positives
├── verify.py                      — structural verification checks
├── math_render/                   — rendered HTML/PDF math-leak audits (see math_render/README.md)
└── checks/
    ├── __init__.py
    ├── vs_period.py               — bare 'vs' → 'vs.'
    ├── compound_prefix.py         — pre-/non- close-up (strict 6-term list)
    ├── percent_symbol.py          — '%' → 'percent' in body prose
    ├── lowercase_prose_references.py  — 'Chapter 12' → 'chapter 12'
    ├── acknowledgements_spelling.py   — British → American
    ├── binary_units.py            — GiB/TiB in prose (detection only)
    └── h3_titlecase.py            — H3+ headings in title case (detection only)
```

The scanner is read-only. It reports issues; an editor decides each fix in
context.

### Persistent accept-list

`accept_list.py` + `accepted_fps.json` together record editorial verdicts on
scanner false positives (proper-noun-heavy headings, named principles,
legislation, after-colon CMS 8.158 caps, D·A·M/C³ taxonomy axes). After every
scan, matching issues are flipped from `open` to `accepted` and tagged with the
sub-rule that justifies them. Match key is `(category, repo-relative file,
exact `before` line)` — if a heading is intentionally edited, its accept-list
entry stops matching and the issue correctly returns to `open` for re-review.

```bash
# Default: accept-list applied, summary shows matched + stale counts
python3 binder/tools/audit/scan.py --scope vol1 -v

# Report every issue, ignoring the accept-list
python3 binder/tools/audit/scan.py --scope vol1 --no-accept-list -v

# Use a different accept-list file (e.g. a draft to iterate on)
python3 binder/tools/audit/scan.py --scope vol1 --accept-list /tmp/draft.json
```

### Scan usage

All commands are from the repo root.

```bash
python3 binder/tools/audit/scan.py --scope vol2 --verbose
python3 binder/tools/audit/scan.py --scope vol1 --output vol1-ledger.json --verbose
```

Produces `audit-ledger.json` (or the path given by `--output`).

---

## Check categories

| Category | Rule | Notes |
|---|---|---|
| `vs-period` | §10.10 | Bare `vs` in prose |
| `compound-prefix-closeup` | §10.8 | Strict 6-term list, no extrapolation |
| `percent-symbol` | §10.2 | HTML attribute filter (width=N%) |
| `lowercase-prose-references` | §10.4 | Hand-written "Chapter 12" |
| `acknowledgements-spelling` | §10.7 | British → American |
| `binary-units-in-prose` | §1 | Detection only |
| `h3-titlecase` | §10.9 | Per-heading judgment required |

---

## Validation anchors

Scan times on a cold run from the repo root:

```
$ python3 binder/tools/audit/scan.py --scope vol1 -v
Total: 629 issues across 34 files (0.4s)

$ python3 binder/tools/audit/scan.py --scope vol2 -v
Total: 969 issues across 39 files (0.4s)
```

### Steady-state anchor (2026-04-08)

| Category | vol1 open | vol1 accepted | vol2 open | vol2 accepted |
|---|---:|---:|---:|---:|
| vs-period | 0 | 0 | 0 | 0 |
| compound-prefix-closeup | 0 | 0 | 0 | 0 |
| percent-symbol | 0 | 0 | 0 | 0 |
| lowercase-prose-references | 0 | 0 | 0 | 0 |
| acknowledgements-spelling | 0 | 0 | 0 | 0 |
| binary-units-in-prose | 0 | 0 | 0 | 0 |
| h3-titlecase | **61** | 0 | **29** | 0 |
| concept-term-capitalization | **19** | 0 | **62** | 0 |
| abbreviation-first-use | **163** | 0 | **111** | 0 |
| **TOTAL** | **243** | 0 | **202** | 0 |

Item D uses file-level exclusions for `glossary.qmd` (glossaries are
definitions, not first uses) and excludes the `SIFT` homonym (CV meaning vs
fault-tolerance meaning).

Reproduce with `python3 binder/tools/audit/scan.py --scope vol1 -v`.
Run the detector self-tests with:

```bash
PYTHONPATH=binder/tools python3 binder/tools/audit/checks/h3_titlecase.py
PYTHONPATH=binder/tools python3 binder/tools/audit/checks/concept_term_capitalization.py
PYTHONPATH=binder/tools python3 binder/tools/audit/checks/abbreviation_first_use.py
```

Expect `41/41 passed`, `32/32 passed`, and `17/17 passed`.

---

## Adversarial test coverage

- `protected_contexts.py`: 14 adversarial tests covering bold definitions,
  callout titles, table headers, sentence starts, index entries, @-refs,
  citations, footnote refs, fig-cap attributes, inline code, and inline math.
- `compound_prefix.py`: 21 tests covering the strict 6-term list,
  domain-compound preservation, acronym/proper-noun continuation, and
  case preservation.

Run the adversarial tests inline with each check module during development.
