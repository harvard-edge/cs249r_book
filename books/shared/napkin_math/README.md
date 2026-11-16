# Napkin Math sheets

One US Letter landscape reference sheet per volume: the numbers a
student should know by heart and the quantities they should be able to
calculate, in the book's own notation. The name matches the book's "Napkin
Math" callouts.

| Volume | Module | Scope | Pages (each page answers one question) |
|---|---|---|---|
| I | `vol1.py` | One Node | Will it run? · Will it meet its target? · What does it cost? |
| II | `vol2.py` | The Fleet | Will it scale? · Will it survive and serve? · Can the overhead shrink? · Can it serve responsibly? |
| III | `vol3.py` | The Trajectory | Will it finish within budget? · Is it worth running? · Can it be cheaper and faster? · Can it be trusted and improved? |
| IV | `vol4.py` | The Action | Can it stop in time? · Has it earned permission? · Will the hardware hold up? · Can the evidence be trusted? |

## Build

From `books/shared`, in two steps (the first needs mlsysim, the second a
headless browser):

```sh
PYTHONPATH=../../mlsysim python3 -m napkin_math html vol1 vol2 vol3 vol4 --out _build/napkin-math
python3 -m napkin_math pdf vol1 vol2 vol3 vol4 --out _build/napkin-math   # needs playwright + Chromium
```

The `pdf` step prints a layout report and exits 1 if any item runs past its
page or any equation is wider than its column.

## Series conventions

Every sheet follows these, so the four read as one series.

**Numbers.** Every number is computed in the volume module from the mlsysim
registry or the volume's own scenarios. An input chosen only for an example
is tagged `assumed()` on the page. Constants from outside the registry go in
`EXTERNAL` with their source and should be rare. A printed example must
reproduce from the numbers printed on the sheet. Bounds round in the safe
direction.

**Notation.** Core quantities use the book's notation registry
(`mlsysim/mlsysim/notation/data/`), which all four volumes share, so a symbol
means the same thing on every sheet:

| Quantity | Symbol | Quantity | Symbol |
|---|---|---|---|
| parameters | *P* | bytes moved | *D*<sub>vol</sub> |
| training tokens (dataset size) | *D* | FLOPs (operations) | *O* |
| layers | *N*<sub>L</sub> | arithmetic intensity | *I* = *O* / *D*<sub>vol</sub> |
| hidden width | *d* | bandwidth | BW |
| KV heads | *H*<sub>KV</sub> | peak rate | *R*<sub>peak</sub> |
| head dimension | *d*<sub>head</sub> | fraction of peak sustained | *η*<sub>hw</sub> (MFU when counting model FLOPs) |
| sequence length / context | *S* | fixed overhead | *L*<sub>lat</sub> |
| batch | *B* | time | *T* |
| KV memory | *M*<sub>KV</sub> | accelerators | *N*<sub>GPU</sub> |
| arrival rate | *λ* | service time | *t*<sub>s</sub> |
| server utilization | *ρ* = *λ* *t*<sub>s</sub> | time in system | *T*<sub>lat</sub> |
| requests in system | *Q*<sub>req</sub> | quantization bits, scale | *b*, *s* |

A symbol has one meaning on a page. Quantities outside this table use the
registry's symbol when one exists; otherwise define them in the card that
uses them. The notation strip lists only symbols used on more than one card.

**Numbers to know.** The strip holds anchors a student memorizes and reuses
(specs, unit conversions, rules of thumb, physical constants), not results of
the sheet's own examples. About seven per sheet.

**Reference row.** Each sheet stands alone: its reference table carries the
model dimensions and hardware specs its cards rely on, so no sheet requires
another volume.

**Examples.** Every worked example follows one pattern so a student can trace
it: formula, then the numbers substituted with units, then the result, then a
"So" line (`Card.so`) saying what the result means or which lever it points
to. Print every input needed to reproduce the result, and compute each printed result from the rounded values printed beside it, so a student who redoes the arithmetic gets the printed digits. Define each symbol on
the page, near where it is used.

**Length.** Usefulness comes first. Pages 1 and 2 are the core quick
reference, sized to print front and back as one exam handout. Further pages
add the cards a student needs that the core cannot hold, each page answering
its own question. Prefer fewer, clearer cards to crowded ones.

**Color.** The accent is the volume's print accent (`core.THEME`): page
questions, results, equation panels, and the main chart line. The warning
color marks only a result that fails a limit, and `n(..., warn=True)` adds a
▲ so color is never the only cue. ASSUMED tags are neutral gray. Chart
categories use the accent, gray, and `cat3`, never the warning color.

**Headings.** Page titles are questions. Column titles are questions. Every page's
subtitle says how to read it.

**Shared conventions.** M/M/1: *T*<sub>lat</sub> = *t*<sub>s</sub> / (1 − *ρ*),
p99 of time in system ≈ ln(100) · *T*<sub>lat</sub>. GB = 10<sup>9</sup> B,
GiB = 2<sup>30</sup> B. Footer: units, the assumed legend, and the source of
the values.

**Layout.** Every column ends 0–30 px above the footer. Charts
where a picture teaches more than the formula; no chart that is only a
straight line.
