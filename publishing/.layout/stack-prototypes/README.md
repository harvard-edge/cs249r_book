# Margin-stack prototypes

Standalone LaTeX prototypes for the per-chapter margin stack, built so the
candidate forms can be compared before anything lands in the book.

| File | Shows |
|---|---|
| `vol4-stack-variants.tex` | Five candidate forms for the missing Volume IV stack |
| `vol3-stack-variants.tex` | The Volume III stack that ships today, plus two refinements |
| `vol2-stack-variants.tex` | The Volume II stack that ships today, plus a restored Data bar |
| `series-bracket.tex` | Two candidate series-level figures answering "why are there four volumes" |
| `all-stack-prototypes.pdf` | All four pages in one file |

`_proto-preamble.tex` mirrors the production geometry: 2.6 cm rung width, the
0.3 cm cross-cutting spine at x = 2.85–3.15, `\rmfamily` labels, and the four
volume accent colors read from `books/shared/tex/theme-colors-vol*.tex`.
The vol3-scale rung helper uses the same 4.8 pt label size as the real
`tex/agent-stack-vol3.tex`, which is why six rungs fit a column that only holds
four at vol1/vol2 scale.

Build one:

```bash
lualatex vol4-stack-variants.tex
```

## What the prototypes established

- **Volume IV has no margin stack at all.** It is the only volume without one;
  it uses a full-width `fig_locator.svg` float in all 17 chapters instead.
- **Four tiers is thin at vol1/vol2 rung height.** Adding each tier's cadence
  (0.1–1 Hz / 1–50 Hz / 1000 Hz / continuous) restores the density and is also
  the volume's own thesis, so variant B is the floor rather than variant A.
- **A part gutter does not port down to vol3's rung height.** vol2's gutter
  works because its rungs are 10 mm; at vol3's 6.2 mm only a bare numeral fits
  and the gutter stops reading as "Part".
- **Volume II's stack does not have the defect described from its doc comment.**
  The stale comment in `.layout/tables/vol2/tables.tex` documents a 9-argument
  stack with a Data bar; the macro the chapters actually call is the 8-argument
  `\mlfleetstackwithparts`, which already bands Assurance and Governance into
  Part IV. The one real difference from vol1 is the dropped Data bar.
- **vol3 and vol4 have no About-page `full` variant.** vol1 and vol2 both define
  one and use it with three sample chapters; that is the gap those two volumes
  share.
- **The literal image-4 bracket does not survive contact with four volumes.**
  The brackets overlap and jump the ladder. The volumes nest along a different
  axis: what a mistake is allowed to damage.

These are design artifacts, not book content. Nothing here is wired into any
Quarto config. When a form is chosen, it becomes a macro in
`books/shared/tex/` and this directory can be retired.
