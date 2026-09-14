-- ============================================================================
-- fallacy-pitfall.lua
-- ============================================================================
--
-- Purpose
-- -------
-- This Lua filter handles short "Fallacy" / "Pitfall" blocks used near the end
-- of chapters.
--
-- In the QMD source, these blocks should be written as normal Markdown content
-- inside a Div with the class `.fallacy-pitfall`, for example:
--
--   ::: {.fallacy-pitfall}
--
--   **Fallacy:** *ML systems can be deployed once and left to run indefinitely.*
--
--   Engineers assume deployed systems maintain performance indefinitely, but ...
--
--   :::
--
-- Why this filter exists
-- ----------------------
-- The publisher requested a small amount of vertical space above each
-- "Fallacy" or "Pitfall" heading, similar to the spacing above a short
-- bulleted-list item. The publisher also requested that these paragraphs be
-- set without an initial paragraph indent.
--
-- PDF / LaTeX output
-- ------------------
-- For LaTeX/PDF output, this filter inserts a small vertical space before the
-- block and sets paragraph indentation to 0pt inside the block.
--
-- HTML / EPUB output
-- ------------------
-- For HTML and EPUB output, this filter leaves the content unchanged. The
-- `.fallacy-pitfall` class remains in the output, so spacing and indentation
-- can be controlled with CSS if needed.
--
-- Notes
-- -----
-- - The filter does not change the text content.
-- - The filter only affects Div blocks with class `.fallacy-pitfall`.
-- - The amount of PDF spacing can be adjusted by changing `4pt` below.
--
-- ============================================================================

local function has_class(el, class)
  for _, c in ipairs(el.classes) do
    if c == class then
      return true
    end
  end
  return false
end

function Div(el)
  if has_class(el, "fallacy-pitfall") then
    if FORMAT:match("latex") then
      table.insert(el.content, 1, pandoc.RawBlock("latex",
        "\\par\\Needspace{2.5\\baselineskip}\\addvspace{4pt}\\begingroup\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{2pt plus 0.5pt}%"
      ))

      table.insert(el.content, pandoc.RawBlock("latex",
        "\\par\\endgroup"
      ))
    end

    return el
  end
end