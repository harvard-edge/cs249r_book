# TinyTorch Lab Guide — Chapter Review Brief

> Internal working doc for the deep narrative-polish pass on the
> TinyTorch Lab Guide (PDF + website). One agent reviews one chapter,
> following this brief end-to-end. Read it twice before editing.

## 1. Framing — what this book is

TinyTorch is the **canonical lab book** for building a usable ML
framework from scratch. It exists in the same conceptual slot as:

- *Structure and Interpretation of Computer Programs* — for Lisp /
  programming abstraction
- *The C Programming Language* (K&R) — for C
- *Operating Systems: Three Easy Pieces* — for OS
- *Computer Architecture: A Quantitative Approach* (Hennessy &
  Patterson) — for hardware
- *Crafting Interpreters* — for language implementation

A professor hands TinyTorch to their TA and says *"this is the lab
guide for the course"*. A self-taught engineer downloads it on a
Friday night and works through it because it actually teaches them
how PyTorch is built. A researcher cites it because it's the cleanest
explanation of how an autograd engine actually works.

That is the bar. Every chapter must earn its place against that bar.

### What the book is NOT

- Not API documentation. The Quarto-rendered API stubs are a
  byproduct, not the point.
- Not a tutorial. Tutorials get you to "hello world" fast. We get
  the reader to *understanding*, which is slower and more valuable.
- Not a research paper. We don't justify with citations — we justify
  with code that works.
- Not a marketing site. The Welcome chapter is a doorway, not a
  pitch deck.

## 2. Voice & register

- **Direct.** "Subtract the max before exponentiating to prevent
  overflow." Not: "It is generally recommended that practitioners
  consider…".
- **Second person + active verbs.** *You* implement. *You* test.
  *You* compare. The reader is the agent.
- **Concrete > abstract.** "A 1B-parameter model needs 4GB just for
  the weights" beats "models can be memory-hungry".
- **One idea per paragraph.** If a paragraph has two `however`s, it
  is two paragraphs.
- **Cut hype.** "powerful", "revolutionary", "game-changing",
  "robust", "comprehensive", "world-class" → delete unless backed
  by a number on the next line.
- **No corporate softening.** "It is important to note that…" → just
  state the thing.

## 3. Pedagogical structure (Build-Use-Reflect)

Every module already follows this scaffold. Don't rearrange it. Do
make sure each piece is doing its job:

| Section            | Job                                           |
|--------------------|-----------------------------------------------|
| Eyebrow callout    | One sentence: what they'll have built.        |
| Why This Matters   | Real-world stakes. Numbers if possible.       |
| Mental Model       | The single picture they'll keep.              |
| What You'll Build  | Diagram + bulleted scope.                     |
| API Reference      | Signatures the implementer must satisfy.      |
| Implementation     | Step-by-step with rationale, not just code.   |
| Use it             | Apply to a real problem.                      |
| Reflect            | Tradeoffs, what production does differently.  |
| Check Your Understanding | Conceptual questions with answers.       |
| What's Next        | Bridge to the next chapter.                   |

If a chapter is missing a "What's Next" bridge, add one (1–2
sentences naming the next module and the question it answers).

## 4. Project conventions you MUST enforce

These are house rules. Every chapter must comply.

### 4.1 Markdown hygiene

- **No mid-document `---`.** Horizontal rules are forbidden in body
  content. The only `---` in a `.qmd` file is the YAML frontmatter
  fence at the very top. Use callouts, headings, or whitespace
  for separation instead.
- **Lists need a blank line above them.** A bullet list following a
  paragraph without a blank line gets pulled into the paragraph.
  Always:

  ```markdown
  Some intro sentence.

  - first bullet
  - second bullet
  ```

- **Definition-style entries: `**Term** — explanation.`**
  (em-dash). Used for short labeled paragraphs (audience labels,
  method tradeoffs, alternative options). Renders cleanly in both
  HTML and PDF without any divs.

### 4.2 Emoji

- **No emoji in PDF body prose.** XeLaTeX renders most color-emoji
  glyphs as zero-width invisible characters, leaving phantom spaces
  before punctuation.
- Emoji are allowed in:
  - Section titles in the website sidebar (controlled by `_quarto.yml`)
  - Callout titles (Quarto handles these properly)
  - Code-block comments inside fenced code (treated as monospace text)
- If you find emoji in body prose, either delete them or wrap the
  whole paragraph in `:::{.content-visible when-format="html"}`.

### 4.3 HTML divs / cards

- Inline `<div style="...">` blocks for colored card grids look great
  on the website but flatten to unstyled prose in PDF.
- Pattern: split into format-conditional blocks.

  ```markdown
  ::: {.content-visible when-format="html"}
  <div style="display: grid; ...">
    ...HTML cards...
  </div>
  :::

  ::: {.content-visible when-format="pdf"}
  **Label A** — description for PDF readers.

  **Label B** — description for PDF readers.
  :::
  ```

### 4.4 Cross-references

- Use Quarto cross-refs (`@sec-label`, `@fig-label`, `@tbl-label`)
  rather than hardcoded "see Chapter 4" prose. They auto-renumber.
- Every chapter heading (`# Chapter Title`) gets `{#sec-XX}` if it
  is referenced from elsewhere — don't invent new IDs without
  checking that nothing depends on the old ones.

### 4.5 Code blocks

- Triple-backtick fenced. Always specify language: ` ```python `,
  ` ```bash `, ` ```text ` for ASCII art.
- Long lines now wrap automatically (fvextra in `pdf/_quarto.yml`),
  but **prefer breaking lines yourself** at logical points — wrapped
  lines without an explicit break read as continuation arrows; lines
  the author broke read as deliberate.
- No `{python}` or `{r}` inline-execution shortcodes inside narrative
  prose unless the project actually evaluates them. We don't — the
  Quarto engine for these chapters is set to `markdown` (no kernel),
  so any `` `{python} foo` `` literal will render as the raw string
  `{python} foo` in both HTML and PDF. **If you see one, replace it
  with the literal value or with a static code span.**

### 4.6 Diagrams

- Don't touch the `<img>` paths or the SVGs — those are owned by the
  diagram pass. The Lua filter rewrites `.svg` → `.pdf` for PDF
  builds automatically.
- Do tighten figure captions: `fig-cap` should describe the takeaway,
  not the components. *"Inheritance and composition of neural network
  building blocks"* > *"Diagram showing Layer Base Class with three
  child classes"*.

### 4.7 Callouts

- `.callout-note` — neutral context, side discussion
- `.callout-tip` — pro tip, optimization, "what experts do"
- `.callout-warning` — bug-bait, common pitfall
- `.callout-caution` — destructive action, irreversible operation
- `.callout-important` — must-read, conceptual prerequisite
- Always give callouts a `title="..."` — untitled callouts read as
  generic boxes in PDF.
- Inside a callout, **paragraph spacing is collapsed**. If you need
  visible breathing room between two paragraphs in a callout, insert
  `:::{.content-visible when-format="pdf"}\medskip:::` between them.

## 5. What you should change

You have a budget of ~5–15 targeted edits per chapter. Spend it
where it matters most:

1. **Open the chapter strong.** First paragraph after the eyebrow
   callout must answer "why am I reading this?" in 1–2 sentences.
   No throat-clearing, no "In this chapter we will…".
2. **Tighten verbose prose.** Remove hedging adverbs, collapse
   triple-clause sentences into doubles, kill redundant phrases.
3. **Replace passive with active** when the agent is identifiable.
4. **Fix every house-rule violation** from §4.
5. **Verify each section's transition.** A new H2 should start with
   one sentence that connects from the previous section.
6. **Strengthen the close.** Final two paragraphs should land on a
   takeaway, not trail off into a list of features.
7. **Fix any literal `{python}` / `{r}` shortcodes** rendering as
   text (see §4.5).

## 6. What you should NOT change

- **Code semantics.** If you "improve" a code example to be more
  pythonic and the test suite breaks, you've shipped a regression.
  Touch code only to fix obvious bugs (typos, wrong variable names)
  or to add comments.
- **Section ordering.** The Build-Use-Reflect scaffold is intentional.
- **YAML frontmatter** unless you're adding a missing `fig-cap` or
  `fig-alt`.
- **Wholesale rewrites.** Voice continuity matters across 28
  chapters. We are polishing, not rebuilding.
- **Diagrams** (see §4.6).
- **Module / chapter numbers** in any reference like "Module 03" or
  "Chapter 7". The auto-numbering in PDF differs from the website
  module numbering — leave the manual references as-is unless they
  are demonstrably wrong.

## 7. Checklist (run before you save)

Before declaring the chapter done, verify:

- [ ] No mid-document `---` separators
- [ ] No emoji in body prose (PDF blank-glyph trap)
- [ ] All lists have a blank line above them
- [ ] All callouts have `title="..."`
- [ ] First paragraph earns the chapter's existence in 1–2 sentences
- [ ] "What's Next?" bridge present and concrete
- [ ] No literal `{python}`/`{r}` shortcodes in body text
- [ ] Inline HTML `<div>` cards have a PDF fallback or are wrapped
      in `:::{.content-visible when-format="html"}`
- [ ] Voice is direct, second-person, active where possible

## 8. Output

Edit the file in place. Do **not** commit (the orchestrator handles
commits). At the end of your turn, return a short report:

```
Chapter: <filename>
Lines changed: <approx>
Top 5 improvements:
  1. ...
  2. ...
  3. ...
  4. ...
  5. ...
House-rule fixes:
  - <count> mid-doc HRs removed
  - <count> emoji-in-prose removed
  - <count> bullet lists missing leading blank line
  - <count> divs given PDF fallback
  - <count> {python} shortcodes neutralized
Open questions for the orchestrator:
  - ...
```
