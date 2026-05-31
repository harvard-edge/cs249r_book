# LEGO → Prose Output Governance Plan

> **The mandate.** *Every value a reader sees in the book is produced by a LEGO cell
> and must be correct, consistently formatted, and faithfully rendered.* This plan is
> the single map for **everything that flows from a `{python}` LEGO export into prose** —
> not just `suffix=` cleanup. Each value-kind has one blessed formatter, one prose
> pattern, and at least one automated gate that fails if the contract breaks.
>
> Companions: **`MIGRATION.md`** (rollout board) · **`ASSESSMENT.md`** (per-chapter
> verification gauntlet) · **`README.md`** (spurious-`.0` workflow) · authoritative
> rules `.claude/rules/fmt.md`. Sources of truth: MIT Press style sheet
> (`~/Desktop/MIT_Press_Feedback/`), the LEGO verification plan
> (`~/Desktop/MLSysBook-LEGO-HTML-Verification-Plan.md`), `book/docs/LEGO_CELLS.md`.

The universal contract for any reader-visible value:

```
EXECUTE  raw = compute(...)             # the number
OUTPUT   raw_str = fmt_KIND(raw, ...)   # ONE typed formatter per value-kind
PROSE    ... `{python} Class.raw_str` ...   # ONE prose pattern per kind
VERIFY   guard ∧ equivalence ∧ prose-contract ∧ ref-in-HTML
```

---

## 1. The complete value-kind taxonomy (current corpus: 6,611 call sites)

| Kind | Blessed formatter | Glyph/unit owner | Prose pattern | MIT / rule | Sites today |
|---|---|---|---|---|---|
| Currency | `fmt_usd` | string (`$`) | `` `{python} cost_str` `` | Chicago | 271 |
| **Percent / share** (0–1) | `fmt_percent` | string via `style` | body: `style='prose'`→"71 percent"; table/eqn: `style='symbol'`→"71%" | **spell out in body** | 10 ✅ + **626** to migrate |
| Percentage points (Δ) | `fmt_pp` | string | "7 percentage points" / "7 pp" | — | (in 626) |
| **Multiplier / speedup** | `fmt_multiple` | **PROSE** (`$\times$`) | `` `{python} x_str`$\times$ `` | `math.md §6#14` | **107** to migrate |
| **Count / tally** (K/M/B) | `fmt_count` | string (scale) | `` `{python} n_str` `` | — | **75+** to migrate |
| Physical quantity | `fmt_qty` / `fmt_unit` | string (unit) | `` `{python} bw_str` `` | **abbreviate units** | 35 + **2,412 unit-suffix** to migrate |
| Dimensionless ratio | `fmt_ratio` | none | prose supplies meaning | — | — |
| Integer | `fmt_int` | none | `` `{python} layers_str` `` | — | 634 |
| Plain float (precision) | `fmt` (bare) | none | `` `{python} v_str` `` | no spurious `.0` | 2,169 |
| Scientific | `fmt_sci` / `sci_latex` | string | `` `{python} flops_str` `` | — | 32 |
| Math expression | `fmt_math` | LaTeX → MathJax | `` `{python} eq_math` `` | renders in MathJax | 96 |
| Fraction | `fmt_frac` | LaTeX | — | — | 8 |
| **Range** (min–max) | **`fmt_range`** *(to build)* | string | `` `{python} rng_str` `` | **en-dash, all digits** (`1992–1993`) | ~78 in MarkdownStr |
| Raw assembled string | `MarkdownStr` (last resort) | author | audited individually | must be guarded | 337 (see §3) |

Migration target: the **626 percent + 107 multiplier + 75 scale** semantic-suffix abuses
and the **78 range-like `MarkdownStr`** are the *dangerous* set (wrong-number risk); the
**2,412 unit suffixes** and remaining `MarkdownStr` are correctness-neutral cleanup.

---

## 2. Cross-cutting correctness (independent of kind)

| Concern | Guarantee | Where enforced |
|---|---|---|
| Spurious `.0` (`153.0 FLOP/byte`) | none in prose/HTML | `audit_prose.py`, `audit_html.py`, `fix_precision.py`, `spurious_zero.py` |
| `inf` / `nan` | rejected at format time | `_require_finite` in `fmt.py` |
| 0–1 ratio mis-scale (10,000%) | rejected | `fmt_percent` guard |
| Negative count / multiplier | rejected | `fmt_count` / `fmt_multiple` guards |
| Precision intent | explicit, fails loud | `fmt(..., precision=N)` `_check_fmt_precision` |
| Rounding ↔ prose match | no off-by-one vs displayed | manual rubric (ASSESSMENT §5 / coherence) |

## 3. The 337 `MarkdownStr` (the unguarded frontier)

Heuristic split: **78 range-like** → `fmt_range`; **39 multiplier** → `fmt_multiple`+prose;
**24 math/LaTeX** (legit — keep); **15 wrap a `fmt` call** (collapse to the inner typed
call); **181 "other"** → must be individually inspected and either typed or justified.
Every surviving `MarkdownStr` needs a one-line reason; a checker flags new unjustified ones.

## 4. Prose-reference integrity (the "in line with prose" half)

| Rule | Gate |
|---|---|
| Every `{python} ref` resolves to an export | `audit_prose.py` (`<MISSING:>`), `audit_lego_html.py` |
| Every export is referenced (no dead OUTPUT) | `binder check code --scope lego-dead-code` |
| Prose owns no duplicate glyph/unit the string already carries | **`fmt_prose_contract.py`** *(to build, §0a ASSESSMENT)* |
| Multiplier prose carries `$\times$` | `fmt_prose_contract.py` |
| Ref value actually appears in rendered HTML / MathJax | **`audit_lego_html.py`** (ground truth), `render_playwright_verify.py` (L4) |
| No hardcoded operands that should be exports | `binder check code --scope lego-prose-literals` |

## 5. MIT Press editorial style (prose + formatter output)

In-scope for formatters: **percent** (spell out body / `%` table), **ranges** (en-dash,
all digits), **units** (abbreviate). Orthogonal MIT passes already applied (em-dash
close-up, abbreviation first-use, capitalization, series comma, "vs.") — our edits must
**not undo** them. `binder check numbers` + `check punctuation` guard these.

## 6. LEGO structural contract (`LEGO_CELLS.md`)

Locality (cell ≤ ~50–100 lines from first ref) · span (exports used in one
callout/section) · coupling (no cross-cell reads) · LOAD/EXECUTE/GUARD/OUTPUT shape ·
registry-sourced values (`binder check registry --scope sources`) — verified via the
focal queue `lego_chapter_queue.json` + `lego_focal_verify.py`.

## 7. Semantic / logical coherence (manual + LLM)

Numbers tell a consistent story (setup → steps → conclusion); narrative names match
registry twins; units consistent (FLOPs vs FLOP/s). Rubric in ASSESSMENT §5;
`lego_prose_coherence.py` / `llm_chapter_signoff.py` assist.

---

## 8. Verification matrix — every concern maps to a catching gate

| Layer | Catches | Tool | Status |
|---|---|---|---|
| L0 static | suffix-semantics, canonical fmt, dead code, prose literals, registry | `binder check {math,code,registry}`, `fmt_semantic_suffix.py` | built (suffix gate opt-in until rollout done) |
| L1 exec+prose | exec failure, missing ref, spurious `.0`, **before/after value+prose diff** | `audit_prose.py`, **`assess_equiv.py`** | built |
| L2 build | chapter builds; spurious `.0` in HTML; tracebacks | `chapter_html_verify.py`, `audit_html.py`, `render_html.sh` | built (re-run post-merge) |
| **L3 ref↔HTML** | every `{python}` value present in rendered HTML | **`audit_lego_html.py`** | built — **ground truth** |
| L4 browser | MathJax math refs render | `render_playwright_verify.py` | optional |
| L5 orchestrate | full stack | `run_render_guarantee.py` | optional |
| NEW | Regime-2 composite proof; glyph duplication; ranges | visible-text normalizer, `fmt_prose_contract.py`, `fmt_range` | to build |

---

## 9. Workstreams (status)

| WS | Scope | Risk | Status |
|---|---|---|---|
| **WS1** Semantic-suffix → typed | 626 pct + 107 ×+ 75 scale | **high (wrong number)** | tooling in progress |
| **WS2** Precision / spurious `.0` | corpus-wide | med | existing tools; re-sweep |
| **WS3** `MarkdownStr` → `fmt_range`/typed | 337 (78 ranges) | med | needs `fmt_range` + inspection |
| **WS4** Unit-suffix → `fmt_qty` | 2,412 | low (cosmetic) | Phase 2 |
| **WS5** Prose-reference integrity | all refs | med | `audit_lego_html` + new prose-contract |
| **WS6** Semantic coherence | computational callouts | high (logic) | manual/LLM, per chapter |

**Definition of done (per chapter):** L0–L3 green · `assess_equiv diff` clean or adjudicated ·
prose-contract clean · `prose_read ✅`. **Book done:** all chapters done + `fmt_semantic_suffix`
flipped to a global pre-commit blocker.
