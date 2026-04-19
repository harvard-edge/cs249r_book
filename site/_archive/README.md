# Landing page archive

Historical landing-page iterations kept for reference and rollback. Quarto
ignores any directory beginning with `_`, so nothing in here is rendered or
deployed.

| File | Notes |
|------|-------|
| `index-v1.qmd` | First landing layout. Used `landing.css` only and class `mls-landing`. |
| `index-v2.qmd` | Second iteration. Used `landing.css` + `landing-v2.css` and class `mls-landing-v2`. |
| `landing-v2.css` | Stylesheet paired with `index-v2.qmd`. |
| `neural-bg-dark.js` | Dark-mode variant of the neural background canvas; never wired into the active landing page. |

The live landing page is `site/index.qmd`, which uses `landing.css` +
`landing-v3.css` and class `mls-landing-v3`. Selectors in `landing.css` that
were only used by v1/v2 may eventually be safe to drop after a careful audit
against `index.qmd`.

If a future iteration revives one of these layouts, restore the file and its
matching CSS class instead of starting from scratch.
