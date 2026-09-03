# Mobile QA — Book HTML

The book uses Tufte-style margin notes and citations on the desktop layout
(`reference-location: margin`, `sidenote: true` in
[`config/_quarto-html-vol1.yml`](../quarto/config/_quarto-html-vol1.yml) and
[`_quarto-html-vol2.yml`](../quarto/config/_quarto-html-vol2.yml)). Quarto's
default behaviour is to collapse the margin column on narrow viewports and
re-flow margin notes inline, but this needs periodic verification on real
devices.

This file is the running checklist. Re-run after any change that touches
[`shared/styles/partials/_mobile.scss`](../../shared/styles/partials/_mobile.scss),
[`shared/styles/_book-only.scss`](../../shared/styles/_book-only.scss), the
Quarto version, or the chapter template.

## Devices to cover

Pick at least one per row. Test in **light** and **dark** mode each.

| Form factor              | Recommended device / emulator        | Width       |
|--------------------------|--------------------------------------|-------------|
| Small phone (portrait)   | iPhone SE / Pixel 5                  | 375–393px   |
| Modern phone (portrait)  | iPhone 15 / Pixel 8                  | 390–412px   |
| Phone (landscape)        | same as above, rotated               | 667–844px   |
| Small tablet (portrait)  | iPad mini / Galaxy Tab A9            | 744–800px   |
| Large tablet (landscape) | iPad Pro 11" / Surface               | 1180–1280px |

If you only have a desktop browser, use Chrome/Safari devtools' device
emulator covering the same widths.

## Pages to spot-check

Pick chapters that exercise the full feature set:

- A chapter with **lots of margin citations**: `vol1/intro/intro.qmd`
- A chapter with **lots of sidenotes**: any `vol1/<chapter>/*.qmd` with
  `[^margin]` references
- A chapter with **figures + captions in the margin**: `vol1/dl_primer/dl_primer.qmd`
- A chapter with **tables + code listings**: `vol1/training/training.qmd`
- A chapter with **callout-heavy content** (Definition, Example, Quiz):
  `vol1/frameworks/frameworks.qmd`

## Checklist

For each device/page combination:

- [ ] Page loads without horizontal scroll.
- [ ] Margin notes (`<aside class="column-margin">`) reflow inline below
      their anchor paragraph, not clipped or hidden.
- [ ] Margin citations render as inline footnote-style links and the
      backlink works.
- [ ] Figures span available width without overflow; captions wrap.
- [ ] Tables either scroll horizontally inside their container or collapse
      to a card layout — never push the page wider than the viewport.
- [ ] Code blocks scroll horizontally, syntax highlighting still applies.
- [ ] Sticky navbar collapses to a hamburger; menu opens; all 6 dropdowns
      still reachable; "Subscribe" / GitHub buttons accessible.
- [ ] Sidebar (chapter list) opens via the `≡` toggle and dismisses on
      tapping outside; scrolling inside the sidebar does not bleed into
      page scroll.
- [ ] Hypothesis annotation icon (right edge) is reachable but not
      covering tap targets.
- [ ] SocratiQ widget (bottom right) does not overlap the back-to-top
      button or footnote backlinks.
- [ ] Dark mode toggle switches the entire page (including margin notes,
      callouts, code blocks).
- [ ] Lightbox-enabled figures open full-screen on tap; close button is
      reachable.
- [ ] Quizzes (custom blocks) collapse readably; "Show answer" toggle
      works.
- [ ] Search bar opens, results selectable, no keyboard layout shift.
- [ ] Anchor links (`#section-id`) jump to the right offset (the sticky
      navbar must be accounted for).

## Known issues to watch for

- iOS Safari sometimes ignores `100vh` on the navbar; use `100dvh` if
  layout breaks at the bottom.
- Quarto's default margin-collapse breakpoint is around 991px. If the
  navbar `collapse-below: lg` change ([navbar-common.yml](../../shared/config/navbar-common.yml))
  introduces visual regressions on iPad portrait, revisit that decision.
- Hypothesis sidebar may obscure margin elements when expanded; current
  policy keeps it collapsed (`openSidebar: false`) — verify it stays
  collapsed on first paint on every device.

## How to record findings

Open a draft issue titled `Mobile QA <YYYY-MM-DD>`, paste the device matrix,
attach screenshots for any failing checkbox. Cross-reference any fixes back
to this file once merged.
