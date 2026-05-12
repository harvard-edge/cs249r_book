# Cross-Reference Review Tooling

This folder contains the reusable cross-reference workflow for Volume 1 and Volume 2.
The goal is to make cross-references useful to readers without turning the
books into a dense web of authorial pointers.

Generated inventories, packets, reports, and merged decision queues default to
`review/cross-references/`. Keep reusable scripts and instructions here under
`scripts/cross-references/`; keep run outputs under `review/`.

## Scope

- Volume 1 and Volume 2 are separate books.
- Do not create Quarto cross-references across `contents/vol1/` and
  `contents/vol2/`.
- A volume may share frontmatter includes or bibliography infrastructure, but
  reader-facing section, figure, table, equation, and principle references must
  resolve inside the same volume.

## Editorial Rule

A cross-reference is a pedagogical signpost. Keep it only when it changes what
the reader can do next.

Use a cross-reference when:

- the current paragraph depends on a concept taught earlier in the same volume;
- the target is the canonical definition, taxonomy, law, table, or framework;
- a local reminder would otherwise become a repeated mini-explanation;
- the prose directly interprets a numbered figure, table, or equation;
- a forward pointer reduces confusion about where a derivation or full treatment
  will appear;
- two sections must be compared to understand a tradeoff.

Avoid a cross-reference when:

- the connection is merely thematic;
- the target is in the other volume;
- the term can be locally explained in a short phrase;
- the reference repeats within the same local discussion;
- the sentence already names the target concept clearly;
- the pointer exists mainly as author bookkeeping.

## Automation Strategy

The review has two layers:

1. **Mechanical audit**: extract anchors and references, detect unresolved
   targets, detect cross-volume targets, and summarize reference density.
2. **Editorial pass**: for each chapter, classify each reference as required,
   useful, redundant, wrong target, or missing.

The mechanical audit intentionally does not auto-insert references. It produces
chapter-sized work packets that an agent or human editor can evaluate.

Run:

```bash
python3 scripts/cross-references/audit_crossrefs.py
```

For a fresh round without overwriting earlier reports:

```bash
python3 scripts/cross-references/audit_crossrefs.py --out-dir review/cross-references-round2
```

Outputs:

- `review/cross-references/report.md`
- `review/cross-references/inventory.json`
- `review/cross-references/canonical-target-candidates.yml`
- `review/cross-references/chapter-report-schema.yml`
- `review/cross-references/chapter-packets/*.yml`

Merge completed reference-aware and blind-need reports with:

```bash
python3 scripts/cross-references/merge_crossref_reports.py --base-dir review/cross-references-round2
```

The chapter packets are designed for parallel analysis. Give each agent one
packet and the corresponding chapter file. Agents should return YAML that
conforms to `chapter-report-schema.yml`. Do not let analysis agents edit prose.
Apply edits only in a second pass after aggregating their YAML decisions.

## Recommended Review Order

Review each volume independently.

1. Introduction and part-opening principle chapters.
2. Core technical chapters in book order.
3. Appendices and glossary.
4. Final pass for over-referencing and repeated pointers.

Within a chapter:

1. Read the heading outline and existing references.
2. Identify canonical concepts introduced by the chapter.
3. Decide which incoming references should point to those concepts.
4. Decide which outgoing references are necessary for prerequisites or synthesis.
5. Remove cross-volume references or rewrite them as local reminders.
6. Render or run the mechanical audit after edits.

## Agent Unit Of Work

Assign one agent one chapter packet at a time. These assignments can run in
parallel because each packet is read-only analysis. The agent should return YAML
with:

- file path;
- candidate line;
- current reference, if any;
- target anchor;
- classification: `keep`, `retarget`, `remove`, `add`, or `localize`;
- one-sentence rationale;
- proposed prose edit.

Second-pass editing should be centralized: one editor applies accepted YAML
decisions, verifies the affected references, and reruns the audit.

Do not assign a whole volume to one pass unless the task is only mechanical
reporting.
