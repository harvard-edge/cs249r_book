# Math Audit: vol1/frontmatter/acknowledgements.qmd

Source audited: `book/quarto/contents/vol1/frontmatter/acknowledgements.qmd`

## Summary

This frontmatter file contains no mathematical equations, derivations, formulas, algorithmic complexity claims, or quantitative results that require mathematical verification.

The audit found several numeric/unit consistency issues in prose, CSS, and HTML attributes. No source `.qmd` files were modified.

## Findings

### 1. Mixed numeric style in memory range

- Line 5: "sixteen to 256 KB device"
- Issue: The lower endpoint is spelled out while the upper endpoint is numeric, and the unit appears only at the end. For a technical memory range, both endpoints should use numerals and the unit should be unambiguous.
- Proposed correction: "16 KB to 256 KB device" or, if modifying the surrounding phrase, "devices with 16 KB to 256 KB of memory".

### 2. Invalid CSS percentage values

- Lines 22 and 39: `width: 100 percent;`
- Issue: CSS does not accept `percent` as a unit. Percentage values must use `%`.
- Proposed correction: `width: 100%;`

### 3. Invalid percentage values in HTML `width` attributes

- Lines 83-87, 90-94, 97-101, 104-108, 111-115, 118-122, 125-129, 132-136, 139-143, 146-150, 153-157, 160-164, 167-171, 174-178, 181-185, 188-192, 195-199, 202-206, 209-213, 216-219, 222-226, and 229: `width="20 percent"`
- Issue: HTML table-cell width values should not spell out `percent`. As written, these attributes are not valid percentage widths.
- Proposed correction: replace each `width="20 percent"` with `width="20%"`.

### 4. Pixel unit embedded in HTML `width` attributes

- Lines 83-87, 90-94, 97-101, 104-108, 111-115, 118-122, 125-129, 132-136, 139-143, 146-150, 153-157, 160-164, 167-171, 174-178, 181-185, 188-192, 195-199, 202-206, 209-213, 216-219, 222-226, and 229: image attributes use `width="100px;"`
- Issue: In HTML, the `width` attribute for `img` elements is a dimension attribute and should be a number of CSS pixels without `px` or a trailing semicolon. The current value is CSS-like syntax inside an HTML attribute.
- Proposed correction: replace each `width="100px;"` with `width="100"` or move sizing to CSS as `style="width: 100px;"`.

## Checked But No Issue

- Line 7: "over 100,000 students" is a plausible approximate count and uses a clear thousands separator.
- Line 9: "In 2023" is a date reference, not a mathematical claim.
- Line 32: `@media (max-width: 768px)` uses valid CSS units.
- Lines 50, 51, 53, 58, and 66: `10px`, `1px`, `100px`, and `5px` are valid CSS length values in declarations.
- Line 273: "ten" and "two or three in the morning" are narrative time references, not technical quantities requiring unit normalization.
