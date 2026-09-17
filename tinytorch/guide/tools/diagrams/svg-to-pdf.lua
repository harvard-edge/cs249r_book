-- svg-to-pdf.lua
--
-- For PDF output, swap `assets/images/diagrams/*.svg` references in qmd
-- files for the matching pre-converted `.pdf` siblings produced by
-- `make figures` (rsvg-convert, deterministic, version-controllable).
--
-- Why: by default Quarto would invoke its own SVG→PDF conversion at
-- render time. Funnelling that through the Makefile instead gives us
--   • one canonical conversion pipeline we control (rsvg-convert flags,
--     bounding-box trim, font handling)
--   • deterministic output independent of which Quarto/rsvg version
--     happens to be on the build host
--   • the option to commit the .pdf siblings if we ever need fully
--     reproducible builds without librsvg installed
--
-- Scope: only rewrites paths under `assets/images/diagrams/` so the
-- transformer SVG (and any future hand-authored vector asset that
-- doesn't have a Makefile-built .pdf companion) keeps relying on
-- Quarto's built-in conversion.
--
-- HTML output is unaffected — the website continues to serve the
-- original .svg files.

function Image(el)
  if quarto.doc.is_format("pdf") then
    if el.src:match("assets/images/diagrams/[^/]+%.svg$") then
      el.src = el.src:gsub("%.svg$", ".pdf")
    end
  end
  return el
end
