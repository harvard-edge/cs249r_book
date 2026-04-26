# Brand colours and where they live

Single source of truth: [`shared/styles/_brand.scss`](_brand.scss).
Theme files in [`themes/`](themes/) consume those tokens and expose semantic
variables (`$accent`, `$accent-dark`, `$callout-*`) that the rest of the SCSS
layer uses.

If you change the brand palette, update `_brand.scss` first, then verify the
non-SCSS sites listed below.

## Palette

| Token              | Value     | Used for                              |
|--------------------|-----------|---------------------------------------|
| `$brand-crimson`   | `#A51C30` | Volume I, Harvard SEAS, MLSys ecosystem accent |
| `$brand-crimson-dark` | `#E85D75` | Volume I dark-mode accent           |
| `$brand-eth-blue`  | `#1F407A` | Volume II                             |
| `$brand-eth-blue-dark` | `#6B9FD4` | Volume II dark-mode accent         |

## Hardcoded references the SCSS tokens cannot reach

These files cannot `@import` SCSS variables and so duplicate the colour values.
When rebranding, grep for the hex code and update each:

### HTML config (`<meta name="theme-color">`)

- [`book/quarto/config/_quarto-html-vol1.yml`](../../book/quarto/config/_quarto-html-vol1.yml) — `#A51C30`
- [`book/quarto/config/_quarto-html-vol2.yml`](../../book/quarto/config/_quarto-html-vol2.yml) — `#1F407A`
- [`site/_quarto.yml`](../../site/_quarto.yml) — `#A51C30`

### Hand-written CSS (not SCSS)

- [`book/quarto/assets/styles/epub.css`](../../book/quarto/assets/styles/epub.css), `epub-vol1.css`, `epub-vol2.css`
- [`site/landing.css`](../../site/landing.css), [`landing-v3.css`](../../site/landing-v3.css)
- [`site/about/about.css`](../../site/about/about.css), [`site/community/community.css`](../../site/community/community.css), [`site/newsletter/newsletter.css`](../../site/newsletter/newsletter.css)
- TinyTorch site: SCSS at [`tinytorch/quarto/assets/styles/style.scss`](../../tinytorch/quarto/assets/styles/style.scss) and [`dark-mode.scss`](../../tinytorch/quarto/assets/styles/dark-mode.scss)

### Inline CSS in `.qmd`

- [`instructors/index.qmd`](../../instructors/index.qmd) — uses `#A51C30` extensively in inline styles.

### Subsite SCSS that still hardcodes colors

These are pre-existing hardcoded copies (kept verbatim to avoid breaking
independent subsite builds; future cleanup should refactor each to
`@import` from this brand layer):

- [`instructors/assets/styles/style.scss`](../../instructors/assets/styles/style.scss)
- [`labs/assets/styles/style.scss`](../../labs/assets/styles/style.scss)
- [`kits/assets/styles/style.scss`](../../kits/assets/styles/style.scss)
- [`slides/assets/styles/style.scss`](../../slides/assets/styles/style.scss)
- [`mlsysim/docs/styles/style.scss`](../../mlsysim/docs/styles/style.scss)
- [`book/quarto/assets/styles/style-vol1.scss`](../../book/quarto/assets/styles/style-vol1.scss) · [`book/quarto/assets/styles/style-vol2.scss`](../../book/quarto/assets/styles/style-vol2.scss)

### Other surfaces

- SVG assets (`README/curriculum-map.svg`, `site/about/assets/images/ai-engineering-venn.svg`, `book/quarto/assets/images/icons/callouts/icon_callout_chapter_forward.svg`) embed the hex value directly.
- JS: [`site/neural-bg.js`](../../site/neural-bg.js) hardcodes `#A51C30` in its colour list.
- Python: [`mlsysim/mlsysim/viz/plots.py`](../../mlsysim/mlsysim/viz/plots.py) defines `crimson = "#A51C30"`.
- TSX: [`interviews/staffml/src/components/EcosystemBar.tsx`](../../interviews/staffml/src/components/EcosystemBar.tsx) uses `#a51c30`.
