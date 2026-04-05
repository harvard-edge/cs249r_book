---
paths:
  - "book/**"
---

# Book Prose V2 — MIT Press Addendum

This document supplements and, where specified, overrides the base `book-prose.md`. All rules in `book-prose.md` remain in effect unless explicitly superseded here. These rules encode the copy editor's (Pamela Hunt) style decisions for the MIT Press publication and apply to both Volume 1 and Volume 2.

Reference: The Chicago Manual of Style, 17th ed.; Webster's New Collegiate Dictionary, 11th ed.

---

## 1. Percent (OVERRIDES any prior usage)

- **Spell out "percent"** in body prose: `94 percent accuracy`
- Use `%` symbol only inside: tables, equations (`$60\%$`), code blocks, figure captions
- When inline Python produces a value: `` `{python} val_str` percent `` (not `` `{python} val_str`% ``)
- `\%` in LaTeX math stays as-is

## 2. Em Dashes (CONFIRMS book-prose.md)

- Closed em dashes only: `word—word` (no spaces)
- Maximum one per paragraph (already in book-prose.md)
- Em-dash is not a colon (already in book-prose.md)

## 3. Capitalization of Concept Terms (NEW)

These concept terms must be **lowercase** in body prose (not proper nouns per MIT Press):

- iron law (not Iron Law)
- degradation equation (not Degradation Equation)
- verification gap (not Verification Gap)
- bitter lesson (the concept; "The Bitter Lesson" only when citing Sutton's essay title)
- ML node (not ML Node)
- data wall, compute wall, memory wall, power wall
- energy corollary
- machine learning operations (generic usage)
- transformer (neural network) — lowercase per style sheet
- four pillars framework

**Keep capitalized**: D-A-M (acronym), TinyML, AllReduce, FlashAttention, PagedAttention, CUDA, cuDNN, PyTorch, TensorFlow, MATLAB, ImageNet, BERT, GPT-4, Stable Diffusion, DALL-E, Tensor Core, TPU, Weights & Biases.

**Exceptions** (keep caps even for lowercase terms): sentence start, bold definitions (`**term**`), triple bold (`***term***`), H1/H2 headers, `\index{}` entries, callout titles, bold table headers.

## 4. References in Prose (OVERRIDES)

- "chapter 12" not "Chapter 12" (lowercase in body prose)
- "section 3.2" not "Section 3.2" (lowercase in body prose)
- "figure 1.1" not "Figure 1.1" (Quarto `@fig-` handles this)
- "table 1.1" not "Table 1.1" (Quarto `@tbl-` handles this)

## 5. Abbreviations (NEW — first use per chapter)

Every abbreviation must be **expanded on first use in each chapter**. Expansion resets at chapter boundaries. Pattern: `convolutional neural network (CNN)` then `CNN`.

See `.claude/rules/mit-press-style.md` Section 4 for the complete canonical forms dictionary.

Special cases:
- CUDA, cuDNN: no expansion needed (well-known)
- i.i.d.: always with periods; expand in every chapter on first use
- vs.: always abbreviated with period; never "versus"
- Adam: Adaptive Moment Estimation (Adam)

## 6. Heading Style (CONFIRMS/CLARIFIES)

- H1 and H2: **headline style** (capitalize principal words)
- H3 and below: **sentence style** (first word + proper nouns only)
- Figure captions and table titles: **sentence style**; no colon after figure/table number

## 7. Slashes (NEW)

No spaces around slashes: `training/inference` not `training / inference`.

## 8. Numbers in Prose (CONFIRMS/EXTENDS)

- Spell out one through nine in body prose
- Digits for 10 and above
- Always digits with units: "3 GB", "7 ms" (already in book-prose.md)
- All page/year ranges with full digits: 1992–1993, 175–185 (not 175–85)

## 9. Abbreviations in Running Text (NEW)

- Spell out "for example" instead of e.g. in running text (e.g. OK inside parentheses)
- Spell out "that is" instead of i.e. in running text (i.e. OK inside parentheses)
- Replace "etc." with "and so on" in running text (etc. OK inside parentheses and notes)
- Space between initials: B. F. Skinner
- US/UK OK as nouns (consistent usage)

## 10. Bibliography (NEW)

- Every `@inproceedings` must have a `publisher` field
- Every `@article` must have a `journal` field
- Include `pages` and `doi` when available
- Do NOT use em dashes for repeat author names
- Letter-by-letter alphabetical order
- Confirm all URLs are live
- Publisher locations (cities) removed for consistency

## 11. Spelling Dictionary (NEW — canonical forms)

Use first spelling in Webster's. Key entries:

AllReduce, backpropagation, bitter lesson (lowercase), break-even (adj), checkpointing, ClinAIOps, cloud ML, coin-cell battery, coreset, D-A-M taxonomy, data center (two words), dataset (one word), decision-making, dtype, earbud, e-commerce, edge ML, endpoint, engineer-month, FlashAttention, four pillars framework, front end (n), f-string, healthcare, im2col, iron law (lowercase), k-nearest neighbors, lifecycle, multi-chip (adj), nonzero, one-hot encoding, open-source (adj), PagedAttention, perceptron, pre- (closed up: pretrained, preprocessing), pytree, real time (n) / real-time (adj), round trip (n) / round-trip (adj), scatterplot, smartphone, smart speaker, smartwatch, softmax, space-time, speedup, time-series (adj), TinyML, total cost of ownership (TCO), trade-off (n), training-serving skew, transformer (lowercase), traveler (not traveller), vehicle-to-everything (V2X), vision transformer (ViT), wake-word detector, x-axis / y-axis, ZIP code.

## 12. Punctuation (CONFIRMS book-prose.md + additions)

- Serial (Oxford) comma: always (already in book-prose.md)
- Comma after e.g. and i.e.
- Double quotation marks; period/comma inside, colon/semicolon outside
- Single quotation marks for quote within a quote
- Spaced periods between ellipsis points; four points for end of sentence
- Periods after table and figure titles if complete sentences
- No ellipses at beginning or end of quotes
- Contractions OK per book-prose.md (forbidden in body prose)
