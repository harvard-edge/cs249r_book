# FMT plan of record

This is the current plan of record for finishing the `fmt_*` migration. It
supersedes ad hoc suffix cleanup: the target is semantic formatters plus a
separate render/prose audit that validates the substituted book text.

## 1. North star

Every computed value that reaches prose should be formatted exactly once in a
LEGO OUTPUT block. The formatter should know the value kind, validate the
domain, choose spacing and glyphs, and return the final `MarkdownStr` that
Quarto inserts inline.

`fmt()` remains the low-level numeric primitive. QMD prose and LEGO OUTPUT
blocks should use typed helpers whenever the value has a semantic kind.

## 2. Formatter taxonomy

Use the smallest set of helpers that reflects real value kinds:

| Value kind | Formatter | Owns |
|---|---|---|
| Plain scalar | `fmt` / `fmt_int` | finite and precision checks |
| Physical quantity | `fmt_qty` / `fmt_qty_int` | Pint conversion, unit suffix, currency refusal, explicit rounded-integer display |
| Duration | `fmt_time` | time-unit validation, non-negative duration, symbol/word style |
| Count | `fmt_count` | non-negative tally, optional scale, singular/plural label |
| Non-physical service rate | `fmt_rate` | allowlisted rate labels like `QPS`, `tokens/s`, `FPS` |
| Currency | `fmt_usd` | escaped dollar sign, scale, denominator |
| Percent share | `fmt_percent` | 0-1 ratio domain and percent style |
| Percentage-point delta | `fmt_pp` | point scale, noun/adjective form |
| Multiplier | `fmt_multiple` | non-negative factor, no glyph in string |
| Ratio | `fmt_ratio` | unitless ratio, non-negative by default |
| Range | typed range helpers / `fmt_range` | endpoint checks and one rendered range |
| Equation/math | `fmt_math`, `fmt_frac`, `sci_latex` | math wrapper or LaTeX expression |
| Label/sequence | `MarkdownStr` | escape hatch only for non-single-value output |

## 3. Structured arguments, not suffix strings

The migration target is not "better suffixes." It is structured formatter
arguments:

- `fmt_qty(bw, GB / second)`, not `fmt(bw.m_as(GB / second), suffix=" GB/s")`.
- `fmt_qty_int(q, GB)` only when rounded integer display is intentional; use
  `fmt_qty(q, GB, precision=0)` for values that should already be integer-like.
- `fmt_time(t, "second", style="symbol")`, not `fmt(t, suffix=" s")`.
- `fmt_count(n, label="GPU")`, not `fmt(n, suffix=" GPUs")`.
- `fmt_rate(qps, "QPS")`, not `fmt(qps, suffix=" QPS")`.
- `fmt_usd(cost, scale="M", per="month")`, not `fmt_usd(cost / MILLION, suffix="M/month")`.

Legacy `suffix=` can remain temporarily for migration compatibility, but QMD
uses are counted, audited, and eventually blocked.

### API ergonomics backlog

After the semantic suffix lanes are stable, add an API cleanup pass for
unit-bearing typed helpers. The preferred long-term call shape is keyworded:

- `fmt_time(t, unit="ms")`, not `fmt_time(t, "ms")`.
- `fmt_qty(q, unit=GB / second)`, not `fmt_qty(q, GB / second)`.
- `fmt_rate(qps, unit="QPS")`, not `fmt_rate(qps, "QPS")`.
- `fmt_time_range(lo, hi, unit="ms")` and `fmt_qty_range(lo, hi, unit=GB)`.

This is not part of the byte-identical suffix migration. Keep the current
positional forms available until the corpus is migrated, then add keyword
aliases, convert QMD call sites, and finally lint against new positional unit
arguments for semantic helpers. Required tests for that pass: keyword and
temporary positional compatibility, missing-unit and both-unit errors,
time-unit validation through `unit=`, rate allowlist validation through
`unit=`, and range endpoint checks through keyword units.

### Formatter output contract backlog

Every value that is exported from a LEGO OUTPUT block for inline prose should be
either produced by a typed formatter or explicitly wrapped as a `MarkdownStr`.
The typed formatters already return `MarkdownStr`; the open design work is a
static/prose gate that flags prose-bound plain strings or f-strings unless they
are intentional labels/sequences. This should reduce Quarto escaping surprises
without banning legitimate non-numeric `MarkdownStr` escape-hatch uses.

### Unit-label registry backlog

`fmt_qty(..., unit_label=...)` is useful during migration, but repeated labels
such as `unit_label="Gb/s"` are a smell. Add a central unit display-label
registry so stock units (`Gbps`, `GB/second`, `TB/second`, `TFLOP/second`,
etc.) render the book's house style automatically. Then migrate call sites like
`fmt_qty(link_bw, Gbps, unit_label="Gb/s")` to plain `fmt_qty(link_bw, Gbps)`,
and reserve `unit_label=` for one-off editorial labels that truly are not stock
units.

### Hardware display API backlog

Hardware/model twin objects know the authoritative stored units and should
probably expose display-oriented accessors for recurring specs: memory capacity,
memory bandwidth, interconnect bandwidth, TDP, and precision-specific peak
FLOP/s. A call site such as `Hardware.Cloud.H100.memory.capacity` should not
force prose authors to remember whether the canonical stored value is GiB or GB,
or whether an interconnect bandwidth should display in `GB/s`, `Gb/s`, or
`TB/s`. Candidate design: add methods/properties such as
`h.memory.capacity_str(...)`, `h.memory.bandwidth_str(...)`,
`h.interconnect.bandwidth_str(...)`, or a generic `fmt_hardware_spec(obj,
spec="memory_capacity")` that delegates to `fmt_qty` with stock unit labels.
This needs design review before implementation so it does not duplicate
formatter policy or hide important unit conversions.

For current `fmt_time(...)` migrations, the positional argument is still a unit,
not a label. Prefer full unit-name strings in source (`"millisecond"`,
`"second"`, `"hour"`) and let `style="symbol"` or `style="word"` decide whether
the rendered text is `ms`/`s`/`h` or `milliseconds`/`seconds`/`hours`. Short
spellings remain accepted for compatibility, but new QMD edits should not add
more mixed `"ms"`/`"second"` source style.

Hyphenated time noun modifiers use `fmt_time(..., style="word",
attributive=True)`, which renders a singular unit (`1-hour`, `24-hour`,
`15-minute`) and rejects `per=`. This is distinct from ordinary prose word style
(`1 hour`, `24 hours`) and from resource-time count labels such as `GPU-hour`.
For compact lower-bound notation that is intentionally written as a trailing
plus, use checked `fmt_time(..., marker="+")` so the plus cannot become an
unvalidated suffix.

### Scale-word and compound-scale status

The exact `scale_word` and `compound_scale` suffix buckets are migrated. The
public spelling is now direct:

- `fmt_count(n, scale="B")` renders compact glyph style, e.g. `70B`.
- `fmt_count(n, scale="billion", label="parameter")` renders word style, e.g.
  `70 billion parameters`.
- `fmt_count(n, scale="billion", attributive=True)` renders hyphenated noun
  modifiers, e.g. `7-billion`.
- `fmt_rate(tokens_per_hour, "tokens/hour", scale="million")` renders checked
  counted rates such as `45.2 million tokens/hour`.
- Word-scale FLOP prose keeps Pint validation through
  `fmt_qty(flops * flop, GFLOPs, unit_label="billion FLOPs")`.

The old `fmt_count(..., scale="B", scale_style="word")` compatibility path
still works, but new QMD should prefer `scale="billion"` because it is clearer
at the call site. The compound migration also found one correctness fix:
`32K tokens` from `32 * 1024` hid 768 tokens, so it now renders as
`32.8K tokens`.

## 4. Formatter defaults

Semantic formatters should own the default display policy for their value kind.
That includes comma grouping. The intended direction is:

- `fmt_usd`, `fmt_count`, and service-rate helpers such as `fmt_rate` default to
  grouped large numbers.
- compact physical/time quantities, percentages, percentage points, ratios, and
  multipliers default to no grouping unless a specific callsite benefits from it.
- explicit `commas=` in QMD should mean "this site intentionally overrides the
  formatter's semantic default," not "the formatter cannot decide."

During migration, keep explicit `commas=` when needed to prove byte-identical
output. After each semantic lane is stable, run a cleanup audit that removes
redundant `commas=` arguments and leaves only intentional overrides.

## 5. Count labels

`fmt_count` is the one intentionally string-bearing helper because the noun is
domain-specific. The label API is strict:

- `label` is singular: `label="GPU"` renders `1 GPU` and `2 GPUs`.
- `plural_label` is an explicit override for irregular or awkward plurals.
- Plural choice follows the raw count, not the scaled display. For example,
  `fmt_count(1_000_000, scale="M", label="parameter")` renders
  `1M parameters`.
- Labels must be clean count nouns, not units, rates, currency, percent, or
  multiplier expressions.
- Resource-time count nouns such as `GPU-hour`, `TPUv4-hour`, `PFLOP-day`,
  `person-year`, and `instance-second` use `fmt_count(..., label=...)` when the
  value is a tally of those resource-time buckets. Hyphenated attributive prose
  such as `100,000-hour` or `10-minute` is not the same thing and remains a
  separate wording/API decision.

If one noun becomes frequent and mistake-prone, add a thin wrapper such as
`fmt_gpus(n)` only after the audit shows that it removes real risk.

## 6. Pint boundary

For physical quantities, keep Pint attached until the formatter whenever
practical. `.m_as(...)` is fine for calculations, guards, ratios, or legacy
plain-scalar sites, but the preferred OUTPUT path is:

```python
value_str = fmt_qty(quantity, display_unit)
```

not:

```python
value_str = fmt(quantity.m_as(display_unit), suffix=" UNIT")
```

Plain scalar unit sites are not blindly fabricated into Quantities. They are
either refactored to carry a Quantity, migrated to a typed scalar helper such as
`fmt_time`/`fmt_rate`, or documented as an exception.

FLOP counts are already Pint quantities in this repo (`flop`, `KFLOPs`,
`MFLOPs`, `GFLOPs`, `PFLOPs`, ...), so exact FLOP-count suffixes should use
`fmt_qty(quantity, FLOP_unit)`. Do not add a separate `fmt_ops` wrapper unless a
later audit shows repeated call-site mistakes that the wrapper would prevent.
Word-scale phrases such as `billion FLOPs` and `trillion FLOPs` keep their
visible wording with `fmt_qty(..., unit_label="billion FLOPs")` after converting
through the corresponding Pint FLOP unit.

When Pint's compact label does not match the book's house style, use
`fmt_qty(..., unit_label="...")` rather than falling back to `suffix=`. The
value is still converted through `display_unit`, so dimensional mistakes are
caught while preserving labels such as `FLOP/byte`, `TFLOP/s per W`, or
`GB per day`.

## 7. Migration lanes

Work by semantic bucket, not by string replacement:

1. Formatter API and tests.
2. Currency scale/per/range oddities.
3. Count labels and non-physical rates.
4. Time values.
5. Clean Pint-backed physical quantities.
6. Plain-scalar unit sites, one chapter at a time.
7. Suspicious `MarkdownStr` numeric wrappers.
8. Final lock: make suffix semantics a global blocker.

Every batch must keep the static and semantic gates clean.

## 8. Validation policy

There are two audit scopes.

### Touched LEGO cell audit

For every LEGO cell touched on this branch, validate every exported
`_str`, `_math`, `_eq`, and `_frac` value:

- before value
- after value
- byte-identical vs intentional change
- rendered inline prose using the value
- spacing/unit/glyph result
- manual read of the surrounding sentence
- HTML render status
- PDF/tex status when required

Record this in `AUDIT_LEDGER.md` as work proceeds.

### Whole-book inline Python audit

At the end, evaluate every inline Python reference in the book, whether or not
the cell was touched. The final audit must confirm:

- every inline expression resolves
- rendered spacing is correct (`70B`, `70 GB`, `\$4.6M`, `\$0.09/GB`)
- units/glyphs are not duplicated
- percent vs percentage-point prose is correct
- no `nan`, `inf`, `??`, or unresolved refs leak through
- the substituted sentence reads correctly as prose

This audit is separate from source cleanup. Static checks are necessary but not
sufficient.

## 9. Done criteria

The migration is done only when:

- formatter tests pass
- `fmt_prose_contract.py` reports 0 violations
- `audit_prose_semantics.py` reports 0 findings
- `codemod_fmt.py queue` is empty or only documented exceptions remain
- every touched LEGO cell is recorded in `AUDIT_LEDGER.md`
- every changed chapter has render evidence
- the whole-book inline Python audit is complete
- suffix semantics is enabled as a blocker
