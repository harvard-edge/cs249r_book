# Float Exposition Standard

How a classic computer-science textbook integrates a float (figure, table, listing,
algorithm, equation) into its prose. This is the rubric the float-explanation eval grades against.

Precedent: Hennessy & Patterson (*Computer Architecture: A Quantitative Approach*; *Computer
Organization and Design*), Bryant & O'Hallaron (*CS:APP*), Cormen et al. (*CLRS*), Peterson &
Davie (*Computer Networks: A Systems Approach*). The convention below is what these texts share.

---

## The contract (every float, every type)

A float is an **exhibit**. The governing rule in technical exposition is: *a visual supplements
the prose, it never replaces it.* The running text must carry the argument on its own; the float
makes it concrete. Operationally, every float earns its place through three prose moves:

1. **Lead-in (Anticipate)** — before the float, the prose establishes the question, claim, or
   tension the float will resolve. The reader knows why it is coming.
2. **Citation (Cite)** — the prose references the float by its cross-reference and names what kind
   of object it is and, at a high level, what it contains.
3. **Lead-out (Interpret)** — the prose states the *takeaway*: what the reader should conclude,
   the "so what." This is the load-bearing move and the one most often missing.

Two tests decide whether the contract is met:

- **Removability test.** Delete the float. If the surrounding prose no longer teaches the concept
  or the argument breaks, the prose was leaning on the float. Fail.
- **Caption-independence rule.** The caption must be self-contained (a reader who looks only at the
  float understands it), but the caption, the figure's alt-text, in-figure labels, and code
  comments **do not count** toward the prose's job. Only running body prose counts.

No orphans (every float is cited) and floats are introduced before they appear (forward reference).

---

## Per-type level — how much the prose must carry

The contract is constant; the *depth* of the Interpret move scales with how opaque the float is on
its own. An equation means nothing without prose; a built figure can carry detail itself.

### 🔴 Equation — strictest
The float is symbols; the prose must teach it.
- **Prose must deliver:** what the equation expresses in words, the meaning of every symbol (a
  "where" clause is fine), units where physical, and the consequence or regime it implies. A worked
  numeric instance is the gold standard (the H&P "iron law" treatment).
- **Pass:** reader understands what the equation says and why it matters from the prose alone.
- **Finding:** a display equation dropped in with only "as shown in @eq-x," symbols never named in
  prose, no stated implication.
- **Not required:** re-deriving the equation step by step, or verbalizing every operator.

### 🔴 Algorithm — strict
- **Prose must deliver:** the algorithm's purpose, its key idea or invariant (the insight that makes
  it work), and the systems cost or the when-to-use. The pseudocode supplies the precise steps; the
  prose supplies the intuition. A small walkthrough is the CLRS standard.
- **Pass:** reader grasps what the algorithm achieves and why, without tracing the pseudocode.
- **Finding:** "Algorithm X shows the procedure" with no intuition, invariant, or cost in prose.
- **Not required:** restating each pseudocode line in sentences.

### 🟠 Table — high
The table holds the detail; the prose owes the conclusion.
- **Prose must deliver:** the takeaway the table encodes — the load-bearing contrast, the specific
  row(s) that matter, or the decision the table drives. H&P tables always carry a "the key result
  is…" sentence in the text.
- **Pass:** reader gets the point of the table from prose and could skip the cells.
- **Finding:** "Table X summarizes / lists / provides guidance on Y" where the actual insight lives
  only in the cells or the caption.
- **Not required:** narrating every cell or row.

### 🟠 Figure — high
- **Prose must deliver:** what the figure *demonstrates* (the relationship, mechanism, or trend) and
  why it matters. The prose tells the figure's story; CS:APP narrates the memory-mountain ridges in
  the text, it does not just point at them.
- **Pass:** reader gets the concept the figure shows from the prose, with the figure as reinforcement.
- **Finding:** "Figure X illustrates this," a pivot-away ("while Fig X shows the tradeoff, other…"),
  or the point living only in the caption / alt-text.
- **Not required:** naming colors, axes, or every visual glyph.

### 🟡 Listing (code) — medium
Visible code teaches by itself; the prose owes orientation, not narration.
- **Prose must deliver:** what the code *shows* — the mechanism it embodies and what the reader
  should notice (the key call, the transformation, the design choice). CS:APP references specific
  line numbers to anchor the point.
- **Pass:** reader knows what the listing demonstrates and what to look at before reading the code.
- **Finding:** "Listing X shows the implementation / a typical configuration" with no framing of the
  mechanism or the design choice that matters.
- **Not required:** a line-by-line walkthrough.

---

## Grade scale (used by the eval, per float, by its type's level)

- **✅ Meets standard** — body prose delivers the lead-out/takeaway at the type's required level;
  passes the removability test.
- **⚠️ Partial** — float is cited and set up, but the Interpret move is missing or thin: the prose
  points without delivering the takeaway, and the content lives in caption/cells/code instead.
- **🛑 Fails** — no body prose carries the float at all (bare pointer with nothing nearby), or the
  float is an orphan, or the prose pivots away without ever explaining it.

A finding is any ⚠️ or 🛑. Each finding gets a suggested rewrite that adds the missing move, in the
book's voice and prose rules (no em-dash/hyphen punctuation, at most one explanatory colon per
paragraph, no float-announcer colon, content leads and the reference rides along).
