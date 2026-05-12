# Agent Playbook For Cross-References

Use this playbook when assigning a chapter-level cross-reference pass.

## Prompt Template

```text
You are reviewing cross-references in one MLSysBook chapter.

Scope:
- Work only in: <chapter path>
- Volume: <vol1 or vol2>
- Do not add references to the other volume.
- Do not rename anchors.
- Do not change principle references from \ref{pri-...} unless explicitly asked.

Inputs:
- review/cross-references/report.md
- review/cross-references/inventory.json
- review/cross-references/canonical-target-candidates.yml
- review/cross-references/chapter-packets/<this-chapter>.yml
- review/cross-references/chapter-report-schema.yml
- the target chapter file
- same-volume introduction, part opening, and glossary as needed

Task:
1. Classify existing @sec-/@fig-/@tbl-/@eq- references as keep, retarget,
   remove, or localize.
2. Identify missing references only where the reader needs a canonical same-volume
   definition, figure, table, equation, or later full treatment.
3. Prefer one local reminder plus one reference over repeated pointers.
4. Avoid references for thematic similarity.
5. Return YAML findings with line numbers and rationale.
6. Do not edit files. This is an analysis-only pass.

Output:
Return YAML conforming to `chapter-report-schema.yml`.
```

## Classification Rubric

`keep`: The reference is same-volume, resolves, and helps the reader navigate a
real prerequisite, canonical definition, figure/table/equation interpretation,
or synthesis.

`retarget`: The prose needs a reference, but the current target is too broad,
too narrow, unresolved, or not the canonical home.

`remove`: The reference is same-volume and resolves, but it is redundant,
thematic only, repeated too soon, or interrupts the paragraph.

`add`: The prose uses a named framework, law, taxonomy, archetype, nonlocal
equation, or specific figure/table result without pointing to the canonical
same-volume target.

`localize`: The prose points across volumes or assumes another volume. Replace
with a short local reminder, a same-volume appendix/glossary pointer, or no
pointer.

## Chapter Checklist

- Does the chapter point to its own canonical figures and tables where the prose
  interprets them?
- Does the first downstream use of a volume-level framework point to the local
  canonical section or table?
- Are there repeated references to the same target within one subsection?
- Are there references embedded in footnotes that should be body prose?
- Are forward references limited to genuine full treatments?
- Are all references inside the same volume?

## Parallel Workflow

Run one analysis agent per chapter packet. The analysis agents are read-only and
may run in parallel. Their output is a YAML report, not a patch.

After the reports are aggregated, a single edit pass applies accepted decisions:

1. reject weak or low-confidence additions;
2. fix mechanical blockers first;
3. localize cross-volume references;
4. add only high-value same-volume references;
5. rerun `python3 scripts/cross-references/audit_crossrefs.py`.

## Style Rules

Prefer integrated prose:

```text
The fleet law introduced in @sec-vol2-introduction-fleet-law separates compute,
communication, and coordination costs.
```

Avoid bookkeeping prose:

```text
As discussed previously in @sec-vol2-introduction-fleet-law, ...
```

Do not write manual chapter or section numbers. Use Quarto references.
