# Float Exposition Audit — vol1/conclusion/conclusion.qmd

**Standard:** FLOAT_EXPOSITION_STANDARD.md (type-level grading)
**Audit date:** 2026-06-09
**Floats scanned:** 3 (1 fig, 2 tbl)

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| fig  | 🟠 high | 1 | 1 | 0 | 0 |
| tbl  | 🟠 high | 2 | 1 | 1 | 0 |
| **Total** | | **3** | **2** | **1** | **0** |

---

## Findings (⚠️ / 🛑 only)

### ⚠️ `tbl-thirteen-principles` (tbl 🟠) — def L247

**Ref sentence (L171):**
> "These thirteen quantitative invariants are not rules of thumb or best practices that evolve with fashion. They are constraints rooted in physics, information theory, and statistics. @Tbl-thirteen-principles collects all thirteen in one place, organized by the four Parts that revealed them. The first two columns identify each principle, the third locates where it was introduced, and the final two columns capture its mathematical essence and predictive power."

**Problem — lead-in is navigational only, not a tension/conclusion setup.**
The citation paragraph at L171 tells the reader what the columns contain (structural navigation: "the first two columns identify … the third locates … the final two capture") rather than establishing what the table's point is or what the reader should conclude from it. This is the pattern the standard flags: "Table X summarizes/lists/provides guidance on Y where the actual insight lives only in the cells." The genuine takeaway ("the thirteen invariants form an integrated framework unified by conservation of complexity … every invariant quantifies a specific consequence of where complexity currently resides") arrives only at L249, after the table. The lead-in does not prime the reader to look for that conclusion; it primes them to navigate the columns.

**Where the takeaway currently lives:** L249 (post-table payoff paragraph). The takeaway is there; it is just not anticipated in the lead-in.

**Missing move:** A lead-in sentence that states the conclusion the table encodes, so the reader enters the table knowing what to look for. The column-guide can remain as orientation, but it should follow a conclusion-setup, not replace it.

**Rule-compliant diff rewrite — replace the citation paragraph at L171:**

```diff
- Throughout this book, each Part introduced quantitative principles that govern ML system behavior. These thirteen quantitative invariants\index{Thirteen Quantitative Invariants!framework} are not rules of thumb or best practices that evolve with fashion. They are constraints rooted in physics, information theory, and statistics. @Tbl-thirteen-principles collects all thirteen in one place, organized by the four Parts that revealed them. The first two columns identify each principle, the third locates where it was introduced, and the final two columns capture its mathematical essence and predictive power.
+ Throughout this book, each Part introduced quantitative principles that govern ML system behavior. These thirteen quantitative invariants\index{Thirteen Quantitative Invariants!framework} are not rules of thumb or best practices that evolve with fashion. They are constraints rooted in physics, information theory, and statistics — and they are not independent: every one quantifies a specific consequence of where complexity currently resides in the system. @Tbl-thirteen-principles collects all thirteen in one place, organized by the four Parts that revealed them; the first two columns identify each principle, the third locates where it was introduced, and the final two columns capture its mathematical essence and what it predicts about real system behavior.
```

*(Note: the em-dash in the diff above is for diff-visual clarity only; the actual rewrite below avoids em-dash per house style.)*

**House-style rewrite (no em-dash, no hyphen-punctuation, content leads):**

> Throughout this book, each Part introduced quantitative principles that govern ML system behavior. These thirteen quantitative invariants\index{Thirteen Quantitative Invariants!framework} are not rules of thumb or best practices that evolve with fashion. They are constraints rooted in physics, information theory, and statistics, and they are not independent: every invariant quantifies a specific consequence of where complexity currently resides in the system. @Tbl-thirteen-principles collects all thirteen in one place, organized by the four Parts that revealed them. The first two columns identify each principle, the third locates where it was introduced, and the final two columns capture its mathematical essence and what it predicts about real system behavior.

**Effect:** the phrase "every invariant quantifies a specific consequence of where complexity currently resides" sets up the conservation-of-complexity conclusion that L249 then states explicitly. The reader enters the table knowing the interpretive lens, not just the column layout.

---

## Passing floats (for completeness)

### ✅ `tbl-lighthouse-journey-mobilenet` (tbl 🟠) — def L165

Lead-in (L153) cites and announces the seven-phase walkthrough. Lead-out (L167) delivers a concrete cascade chain as the takeaway: "Architecture choices (depthwise separable convolutions) enabled compression choices (INT8 quantization), which in turn enabled acceleration choices (mobile NPU deployment). Constraint propagation governs every ML system." The lead-out names the specific rows that matter and states the principle they illustrate. Passes removability.

### ✅ `fig-invariants-cycle` (fig 🟠) — def L285

Lead-in/citation (L283) narrates the figure's visual structure (four phases, central hub, arrows). Lead-out (L590) opens with "The critical insight the figure reveals is the Deploy-to-Foundations feedback arrow" and names the mechanism and its consequence (system must return to foundations when invariants 9–13 fire). The lead-out is explicit and strong. Passes removability.
