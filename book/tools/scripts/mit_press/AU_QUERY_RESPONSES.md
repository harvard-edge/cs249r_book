# AU Query Responses — MIT Press Copyedit Round 1

**Author**: Prof. Vijay Janapa Reddi, Harvard University
**Date**: April 2026
**ISBN**: 978-0-262-05888-9

---

## Category A: Resolved by Automated Passes

These queries are resolved by the systematic edits applied to the manuscript source files.

### Em Dashes (6 queries)
**Query**: "Close up space around the em dash"
**Response**: Fixed globally. All spaced em dashes (` — `) have been closed to `—` throughout the manuscript per MIT Press style. 213 instances corrected in Volume 1.

### Slash Spacing (3 queries)
**Query**: "Close up space around the slash"
**Response**: Fixed. Spaced slashes closed throughout.

### Number-Unit Spacing (2 queries)
**Query**: "Add a space between the number and 'ms'" / "and MB"
**Response**: Fixed. All number-unit pairs verified to have proper spacing per our existing validation rules.

### Bibliography — Publisher Names (30 queries)
**Query**: "Au: Please provide publisher name."
**Response**: Added. Publisher fields have been added to 84 bibliography entries based on known venue mappings (NeurIPS → Curran Associates, ICML → PMLR, ICLR → OpenReview.net, OSDI/NSDI → USENIX Association, etc.).

### Bibliography — Page Numbers (3 queries)
**Query**: "Au: Please provide page number."
**Response**: Many conference proceedings no longer use traditional page numbers. Where available from CrossRef, page numbers have been added. Remaining entries use DOI as the canonical locator per modern citation practice.

### Bibliography — Publisher Locations
**Query**: "For consistency, I have deleted the publisher location. Is it ok?"
**Response**: Yes. Deleting publisher locations for consistency is fine. Modern citation style does not require publisher city.

---

## Category B: Layout Notes (Typesetter Handles)

These are typesetting instructions that will be addressed during the page composition stage. They do not require changes to the manuscript source.

- "Can you make these two columns wider so there's not so much hyphenation?" (pp. 25, 27, 406, 484, 135)
- "Move this to the next page" (pp. 44, 333, 563, 707, 743, 897, 906)
- "Please avoid widowed lines" (pp. 73, 990)
- "This looks too close to the bottom of the page" (p. 765)
- "Should the three numbers here be left-aligned?" (p. 738) — Yes, left-align is fine.

**Response for all layout notes**: Noted. These will be addressed during typesetting.

---

## Category C: Footnotes (38 queries)

**Query**: "Is there a footnote missing here?" / "I don't see the footnotes defined in the margins for this chapter."

**Response**: All footnotes are present in the manuscript source files. Every chapter contains 55–82 footnote definitions. The issue is a rendering artifact in the double-spaced copyedit PDF: our book uses margin sidenotes (via a custom Quarto filter), and the double line spacing caused sidenote content to overflow the margins. In the production (single-spaced) PDF, all sidenotes render correctly in the margins.

Specifically verified:
- Ch 2 (ML Systems): 72 footnotes present
- Ch 6 (NN Architectures): 69 footnotes present
- Ch 7 (Frameworks): 55 footnotes present
- Ch 11 (HW Acceleration): 81 footnotes present
- Ch 12 (Benchmarking): 82 footnotes present
- Ch 13 (Model Serving): 66 footnotes present
- Ch 14 (ML Operations): 67 footnotes present

---

## Category D: Section Numbers (22 queries)

**Query**: "Please add the correct section number here."

**Response**: All 1,053 cross-references in the manuscript are valid and point to existing section IDs. The section numbers will render correctly in the production PDF build. The copyedit PDF may have shown section titles instead of numbers due to the build configuration.

We verified programmatically: zero broken cross-references across all Volume 1 QMD files.

---

## Category E: Decisions Required

### TOC Descriptions (pp. 15, 59)
**Query**: "These section descriptions don't match the section titles exactly. Is that OK?"
**Response**: Yes. TOC descriptions are intentionally concise summaries of the full section titles. This is deliberate for readability.

**Query**: "These don't match the part titles exactly. Is that OK?"
**Response**: Yes. Part descriptions in the TOC use abbreviated forms for space efficiency.

### Duplicate Term Definition (p. 106)
**Query**: "This term was already defined a couple of pages ago. OK to repeat?"
**Response**: Kept for reader convenience at the chapter boundary. Readers may enter at different points.

### Appendix Layout (p. 272)
**Query**: "Should this section be laid out like the fallacies and pitfalls in the chapters?"
**Response**: Yes, please lay out consistently with the chapter Fallacies and Pitfalls sections.

### Monospace in Headings (p. 583)
**Query**: "If possible in a heading, use monospace typeface"
**Response**: Yes, please use monospace for code-like terms in headings where technically feasible.

### Minus Signs (p. 908)
**Query**: "Use minus signs for all the negative numbers on this axis"
**Response**: Yes, please use proper minus signs (−, U+2212) rather than hyphens (-) for negative numbers.

### Von Neumann Footnote (p. 975)
**Query**: "Move the footnote to after 'Von Neumann'"
**Response**: Noted. This will be adjusted in the source.

### "Are these supposed to be words?" (p. 507)
**Query**: "Are these supposed to be words?"
**Response**: This likely refers to a rendering artifact in the copyedit PDF. The content renders correctly in the production build. If the specific location can be identified, we will verify.

---

## Category F: Abbreviation Expansions

All abbreviation expansions requested by the copy editor have been applied:
- CNN → convolutional neural network (CNN) on first use per chapter
- RNN → recurrent neural network (RNN) on first use per chapter
- i.i.d. → independent and identically distributed (i.i.d.) — already expanded in source

---

## Category G: Capitalization

All concept terms have been lowercased per MIT Press style:
- Iron Law → iron law (243 instances across the manuscript)
- Degradation Equation → degradation equation
- Verification Gap → verification gap
- Bitter Lesson → bitter lesson (except when citing Sutton's essay title)
- ML Node → ML node

Bold definitions at first introduction are preserved.

---

## Category H: Percent

All `%` symbols in body prose have been converted to "percent" (1,275 instances). The `%` symbol is retained in tables, equations, code blocks, and figure captions per MIT Press style.
