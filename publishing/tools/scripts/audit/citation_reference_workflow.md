# Citation Reference Validation Workflow

Use this workflow when the task is to verify that `@citekey` citations in
Quarto source actually support the local claim where they appear.

## Packet Builder

Generate one packet per chapter-sized audit unit:

```bash
python3 book/tools/scripts/build_citation_reference_packets.py \
  --out-dir review/citation-reference-validation
```

The packet builder scans the bibliography-scoped source trees used by
`check_bib_qmd_integrity.py`, skips code fences, YAML, raw HTML, inline code,
Quarto cross-references, CSS at-rules, and other non-citation `@` syntax, and
embeds the scoped BibTeX entry for every cited key.

By default, normal book packets are grouped by chapter directory, such as
`book/quarto/contents/vol2/inference/`, so one subagent can audit the chapter as
a coherent unit. Frontmatter pages, part openers, appendices, and glossary pages
are separate chapter-like packets. `_shelved*.qmd` files are skipped unless a
future audit intentionally opts into dormant material. For narrow rechecks, use
`--granularity qmd` to restore the old one-packet-per-QMD behavior.

Use `--all-qmd` only when the audit intentionally includes site, slide, kit, or
other unscoped Quarto files.

Generated outputs:

- `summary.json`: machine-readable packet inventory.
- `agent_manifest.md`: launch list and task prompt for chapter subagents.
- `packets/*.citation-packet.json`: one packet per audit unit.

## Agent Assignment

Assign exactly one packet to each audit agent. Each agent owns that chapter or
chapter-like audit unit from start to finish. The agent should:

1. Read the source context and the embedded BibTeX metadata.
2. Decide what local claim the citation is attached to.
3. Inspect the cited source when the metadata is not enough to judge support.
4. Judge whether the source supports that local claim, not the entire
   paragraph or chapter.
5. Return only actionable problems and a short coverage summary.
6. Prefer `needs_human` over guessing when the source cannot be inspected.

Source inspection is part of the task. Use DOI/publisher pages, arXiv,
OpenReview, official docs, project pages, standards documents, or other primary
sources. Search-result snippets are not evidence. Read enough of the paper or
source to support the judgment.

Use these statuses:

- `valid`: the cited source supports the local claim well enough.
- `unsupported`: the source does not back the claim.
- `overbroad`: the source supports a narrower or adjacent claim, and the local
  wording materially overclaims it.
- `misplaced`: the citation belongs on a different claim or sentence.
- `stale`: the source is superseded for a time-sensitive claim.
- `needs_human`: the source could not be inspected or the fit is ambiguous.

Do not flag low-confidence disagreements. Do not require a paper to support
unrelated nearby prose, and do not flag reasonable synthesis merely because the
paper uses different wording. Flag only citation usages where the source/claim
fit is actually wrong, too broad, stale, misplaced, or impossible to verify.

Recommended finding shape:

```json
{
  "key": "citekey",
  "source_file": "book/quarto/contents/vol2/inference/inference.qmd",
  "line": 123,
  "status": "unsupported",
  "claim": "short paraphrase of the local claim",
  "reason": "why the source does or does not support it",
  "suggested_fix": "replace citation | add source | narrow claim | move citation | remove citation",
  "evidence": "brief source evidence and URL/identifier, or what needs human review"
}
```

## Merge Results

After the chapter agents finish, merge their reports into:

```text
review/citation-reference-validation/report.md
```

Group findings by source file, keep line numbers clickable in review tools, and
do not edit `.qmd` prose until the citation findings have been reviewed.
