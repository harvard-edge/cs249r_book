# FMT Migration — Assessment Protocol

> **Zero-error mandate.** A formatter change may *never* alter a value the reader sees,
> except where we deliberately fix a pre-existing bug — and every such exception is
> proven and signed off. This document is the verification half of the migration; it
> is *more* rigorous than the rollout itself, by design. Every chapter passes the same
> gauntlet. Companion: `MIGRATION.md` (the rollout). Rules: `.claude/rules/fmt.md`.

---

## 0. The invariant we are protecting

For every value the reader sees, there is a chain:

```
EXECUTE: raw = compute(...)            # the number itself
OUTPUT:  raw_str = fmt_KIND(raw)       # the formatter (what we are changing)
PROSE:   ... {python} raw_str ...      # how it is used
```

A migration edits **only the OUTPUT formatter**. Therefore four things must hold,
and we prove each independently:

| # | Invariant | How it can break | Gate that catches it |
|---|---|---|---|
| **I1** | The **rendered string is byte-identical** before/after (unless an adjudicated fix) | codemod drops/keeps a `×100`, wrong precision, wrong glyph | G1 equivalence harness + G4 render-diff |
| **I2** | The **prose owns no duplicate/conflicting glyph** | prose said `…%` after a ref whose string now ends in `%` | G3 prose-contract checker |
| **I3** | The **guard never throws at render** | a ratio is actually 0–100, fed to `fmt_percent` | G2 render + G1 |
| **I4** | The **raw compute is untouched** | an edit strayed into EXECUTE | G0 diff-scope check |

**Rule of adjudication:** I1/I3 failures are *never* auto-resolved. Either the rewrite
is wrong (fix it) or we found a *pre-existing* bug (fix it, record it as an intended
change with a one-line justification in the chapter's sign-off). Silent magnitude
changes are forbidden.

---

## 0a. Who owns the glyph — and why I1 has TWO regimes

The single most error-prone fact in this migration: **each value-kind decides whether
the glyph/unit lives inside the formatted string or out in the prose.** Verified from
`fmt.py`:

| Kind | Formatter | Glyph/unit lives in… | Rendered string e.g. | Prose must… |
|---|---|---|---|---|
| usd | `fmt_usd` | **string** (`$`) | `\$15,000` | not prepend `$` |
| percent | `fmt_percent(style=)` | **string** (`prose`→` percent`, `symbol`→`%`) | `71.1 percent` / `71.1%` | not add `%`/`percent` |
| pp | `fmt_pp(style=)` | **string** | `7 percentage points` | not add the words |
| count | `fmt_count(scale=)` | **string** (`K`/`M`/`B`) | `5M` | not add the glyph |
| qty | `fmt_qty`/`fmt_unit` | **string** (unit) | `48 GB` | not repeat the unit |
| ratio | `fmt_ratio` | nothing (bare) | `3.2` | supply meaning |
| **multiple** | **`fmt_multiple`** | **PROSE** (LaTeX `$\times$`) | `6` | **add `$\times$`** |

`fmt_multiple` is the lone exception: `math.md §6 #14` forbids a literal `×` in the
string, so the multiplier glyph **must** be `$\times$` in prose. This splits invariant
**I1** into two regimes:

- **Regime 1 — string-preserving** (usd, percent, pp, count, qty): the glyph stays in
  the string. Equivalence = **`before_str == after_str`** (G1, byte-identical). The
  prose is *not* edited (only de-duplicated if it wrongly repeated the glyph).

- **Regime 2 — glyph-relocating** (multiplier only): the string deliberately changes
  (`"6×"` → `"6"`) and the glyph moves into prose (`{ref}` → `{ref}$\times$`). String
  equivalence is the *wrong* test — it will always "fail." The correct test is
  **composite rendered-prose equivalence**: normalize each side to *visible text*
  (`$\times$` → `×`, ` percent` → `%`, …) and require
  **`visible(old_str ⊕ old_prose_tail) == visible(new_str ⊕ new_prose_tail)`.**
  Regime 2 sites are **paired edits** — the OUTPUT line and every prose ref move
  together, in the same commit, or the chapter renders `"6 the inference cost"`.

**Consequence for the migration unit:** a site is not `(OUTPUT line)` but
`(OUTPUT line + all its `{python}` prose refs)`. The codemod and the harness operate on
that pair. This is the answer to *"is the output in line with how it's used in prose?"*:
for Regime 2 it is **only** verifiable as a pair.

---

## 1. The centerpiece: per-site **equivalence proof** (Gate G1)

This is what makes the migration provable instead of hopeful. Quarto executes a
chapter's code cells **sequentially in one shared namespace** — exactly like a script.
So we can reproduce the runtime *without* a full render:

1. Extract every code cell from the `.qmd` **in document order** (the audit extractor
   already does this).
2. `exec()` them into one namespace `ns` (LOAD + EXECUTE + GUARD + OUTPUT), so every
   `*_str` variable holds its **real rendered value** on the **real data**.
3. Snapshot `before = {name: str(val) for name in ns if name.endswith('_str')}`.
4. Apply the codemod to produce the candidate cell text; re-exec into a fresh `ns'`.
5. Snapshot `after`. **Diff by variable name** (the LHS `raw_str` never changes — only
   the RHS does, so names are a stable join key).

**Pass:** `before == after` for every `*_str` key. The rewrite is then *proven* a
pure refactor on live data — not an opinion.
**Fail / changed:** the site goes to the **adjudication queue** (§5). It is never
auto-committed.

This catches the `ckpt_memory_savings_pct` trap (see §6) deterministically: a naive
`fmt_percent(ckpt_memory_savings_pct)` either throws (guard) or yields a different
string — both surface here, before any render.

> The harness runs per cell and per site, so a 86-site chapter yields 86 individual
> PASS/CHANGED verdicts, each independently checkable.

---

## 2. The per-chapter gauntlet (G0 → G7)

Run **in order**; a chapter does not advance until the current gate is green. Each gate
emits a line into the chapter's assessment log (`/tmp/fmt_assess/<chapter>.log`).

| Gate | Name | What it does | Pass criterion |
|---|---|---|---|
| **G0** | Scope lock | Diff the proposed edit; assert it touches **only OUTPUT formatter lines**, never EXECUTE/GUARD/LOAD or compute | 0 non-OUTPUT hunks (any → reject) |
| **G1** | Equivalence proof | §1 harness, old vs new, per `*_str` | all keys identical, OR each diff queued |
| **G2** | Render | `binder` build of the single chapter to HTML | exit 0; **no guard exception** in log |
| **G3** | Prose contract | for each `{python} var`: value-kind of `var` (from its `fmt_KIND`) vs the glyph/word adjacent in prose | 0 duplicate/conflicting glyphs; 0 undefined refs; 0 unreferenced `*_str` |
| **G4** | Render-diff | extract rendered `*_str` tokens from HTML before/after; diff | empty diff, OR diff == the adjudicated set from G1 (exact match) |
| **G5** | Static gates | `binder check math` (canonical + suffix-semantics) + `fmt_semantic_suffix` + dead-LEGO | all green |
| **G6** | Ledger snapshot | write `<chapter>.values.json` = `{var: {kind, raw, rendered}}` | committed as the chapter's frozen truth |
| **G7** | Human sign-off | reviewer reads the adjudication queue + the G4 diff and signs | board row → DONE |

**Recheck discipline (the "recheck, recheck"):** G1 (pre-render, static) and G4
(post-render, dynamic) are **two independent measurements of the same invariant I1**
via different mechanisms (AST eval vs HTML extraction). They must *agree*. If G1 says
"identical" but G4 shows a diff, **stop** — that disagreement means our model of the
render is wrong, and we fix the harness before trusting any chapter.

---

## 3. The prose-contract check (Gate G3) in detail

For every exported `*_str`, we know its **value-kind** from the formatter on its RHS:

| RHS formatter | kind | Prose must NOT have, adjacent to the ref |
|---|---|---|
| `fmt_percent` / `fmt_pp` | percent | a trailing `%` or the word `percent`/`percentage points` |
| `fmt_usd` | usd | a leading `$` |
| `fmt_multiple` | multiple | a trailing `×`/`x` |
| `fmt_count` | count | a trailing `K`/`M`/`B`/`million`/`billion` |
| `fmt_qty`/`fmt_unit` | qty | a trailing duplicate of the unit it already carries |

The checker scans each `` `{python} var` `` occurrence, captures the ~24 chars on each
side, and flags any collision. It also enforces:
- **Every** `{python} var` resolves to an exported `*_str` (no undefined refs).
- **Every** exported `*_str` is referenced at least once (no dead OUTPUT — reuses the
  existing LEGO dead-code check).

This is invariant **I2**, and it is the answer to *"is the formatter output in line with
how it's used in the prose?"* — verified mechanically for every single reference.

---

## 4. The chapter value-ledger (Gate G6) — lock-in against the future

After a chapter passes, freeze `<chapter>.values.json`:

```json
{ "ckpt_memory_savings_pct_str": { "kind": "percent", "raw": 0.853, "rendered": "85.3%" } }
```

Commit it. Thereafter, CI re-runs the §1 harness on every PR and diffs against the
frozen ledger: **any future change to a rendered value is flagged automatically**,
forever. The migration doesn't just fix today — it installs a tripwire so the book
*stays* correct.

---

## 5. Adjudicating the queue (the only human-judgment step)

A site reaches the queue when G1 shows the new string ≠ old string, or the guard throws.
For each, the reviewer reads the EXECUTE block and classifies:

- **(a) Codemod error** — the rewrite was wrong (e.g. forgot to divide a 0–100 value).
  → fix the rewrite, re-run G1, expect identical.
- **(b) Pre-existing bug we just exposed** — the old output was actually wrong (e.g. a
  real double-count). → fix it, record `INTENDED CHANGE: <var> <old>→<new> because …`
  in the sign-off, and G4's diff must match *exactly* this set.
- **(c) Genuinely ambiguous source** — can't prove 0–1 vs 0–100 without domain knowledge.
  → normalize the **compute** to a ratio at the source, re-derive, re-run G1.

No site leaves the queue without landing in (a), (b), or (c) with evidence.

---

## 6. Two worked examples (real sites, `vol1/training/training.qmd`)

The harness (`assess_equiv.py`, built and validated) exec'd the whole chapter and
captured **453 `*_str` exports** on real data in <1s. Two of them:

```
TrainingScenarios.ckpt_memory_savings_pct_str      = '71.1%'   (Regime 1)
TrainingScenarios.adam_training_memory_multiplier_str = '6×'   (Regime 2)
```

### (A) Percent — Regime 1, string-preserving

```python
ckpt_memory_savings_pct = (1 - ckpt_memory_units / ckpt_layers) * 100   # EXECUTE: 0–100 scaled
ckpt_memory_savings_pct_str = fmt(ckpt_memory_savings_pct, ..., suffix='%')   # OUTPUT
```
- **Naive** `fmt_percent(ckpt_memory_savings_pct)` feeds `71.1` into a 0–1 guard →
  **throws** → caught at G1/G2. ✅
- **Correct** `fmt_percent(ckpt_memory_savings_pct/100, style='symbol')` → `'71.1%'`.
  Verified live: identical to the old string. G1 **PASS**, no prose edit. Prose
  (`…about {ref} activation-memory savings`) keeps no stray `%` (G3).

### (B) Multiplier — Regime 2, glyph-relocating (the subtle one)

```python
adam_training_memory_multiplier_str = fmt(adam_training_memory_multiplier, ..., suffix='×')  # '6×'
# prose: "Training memory cost is `{python} …multiplier_str` the inference memory cost"  → "…is 6× the…"
```
- **Rewrite** `fmt_multiple(adam_training_memory_multiplier)` → `'6'` (verified live).
  A string-identical test would call this a **failure** — and a migration that edited
  only the OUTPUT would ship **"…is 6 the inference memory cost"** (a real error).
- **Correct paired edit:** OUTPUT → `fmt_multiple(...)`; prose → `…is {ref}$\times$ the…`.
  Composite check: `visible('6×' ⊕ ' the')` == `visible('6' ⊕ '$\times$ the')` →
  both `"6× the"` → **PASS**. The literal `×` is also upgraded to proper `$\times$`
  (fixing a latent `math.md §6 #14` violation).

Example (A) shows the guard + Regime-1 proof; (B) shows why the migration unit is the
`(OUTPUT + prose)` pair and why we measure *visible composite*, not raw string.

---

## 7. What must exist for this protocol to run (build order)

1. **Equivalence harness** (`assess_equiv.py`) — §1. ✅ **built & validated** (453 exports
   captured live from `training.qmd`; `snapshot` + `diff` modes).
2. **Visible-text normalizer + composite prose check** — §0a Regime 2. *(Next.)*
3. **Prose-contract checker** (`fmt_prose_contract.py`) — §3 (I2).
4. **Codemod** (`codemod_fmt.py`) — provable rewrites + queue (feeds G0/G1); must emit
   **paired** OUTPUT+prose edits for Regime 2.
5. **`fmt_range`** — retire unguarded `MarkdownStr` ranges before they reach G1.
6. Wire G0–G7 into a single `binder`-driven per-chapter runner that writes the log + ledger.

Until the harness + composite check exist, **no chapter is migrated.** Tooling first,
then the loop.
