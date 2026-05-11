# Callout Style Todo

Scope: Volume 1 and Volume 2 `.qmd` files.

Source of truth: `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/rules/book-prose.md`.

## Open

- Audit and normalize bold functional lead-ins inside callouts and summary/checklist bullets.
  - Use sentence style for scaffold labels: `**The math**:`, `**The systems insight**:`, `**The systems conclusion**:`, `**The invariant**:`, `**The implication**:`.
  - Keep Title Case only when the bold lead-in itself contains a formal name, proper noun, acronym, eponym, product name, or first-definition term.
  - Keep the colon outside the bold span.
  - Ensure the text after the colon is a complete sentence with an initial capital.
- Preserve previous Title Case work for formal label contexts.
  - `.callout-principle` titles, H1/H2 headings, `\index{}` entries, first formal definitions, footnote term heads, table cells listing principles by name, and caption/table bold-title spans retain Title Case where appropriate.
- Review ordinary callout `title=` attributes only for sentence-style box-head consistency.
  - Do not lower formal `.callout-principle` titles.
  - Preserve proper nouns, acronyms, eponyms, and canonical archetype labels where required.
- Re-check candidate named concepts after the lead-in pass.
  - Confirm `silicon contract`, `Pareto frontier`, `generality tax`, `four pillars framework`, and related terms follow the formal-label vs. prose/display distinction.
- Verify all edits manually in context.
  - Use search only to flag candidates.
  - Do not apply regex/global replacements to QMD prose.
  - Commit after each major pass.
- Audit bare calculation-step labels in numbered lists.
  - Search by structure, not by remembered example terms: flag numbered/bulleted bold lead-ins with multiple capitalized words before the colon, then read each hit in context.
  - Use sentence style for ordinary labels such as `**Network bandwidth**:`, `**Transfer time**:`, `**Training cost**:`, and `**Detection latency**:`.
  - Preserve Title Case only for acronyms, product names, named methods, named datasets, and formal law/principle names.
  - Check whether repeated scaffold labels should drop leading `The` where appropriate: prefer `**Math**:`, `**Invariant**:`, `**Systems insight**:` only if the book adopts that pattern consistently.
