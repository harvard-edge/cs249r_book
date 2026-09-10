-- =============================================================================
-- TABLE CELL LINE BREAKS (PDF/LaTeX)
-- =============================================================================
-- Converts table cells containing <br> (LineBreak) to use LaTeX \makecell{}
-- so that line breaks render correctly in PDF. HTML/EPUB keep <br> as-is.
--
-- Without this filter, \newline from Pandoc's default conversion does not
-- produce visible line breaks in standard tabular columns (l, c, r).
-- \makecell{line1 \\ line2} works with any column type.
-- =============================================================================

local function is_latex_format()
  if quarto and quarto.doc and quarto.doc.is_format then
    return quarto.doc.is_format("latex") or
           quarto.doc.is_format("pdf") or
           quarto.doc.is_format("titlepage-pdf") or
           quarto.doc.is_format("beamer")
  end
  if FORMAT then
    return FORMAT:match("latex") or FORMAT:match("pdf") or FORMAT:match("beamer")
  end
  return false
end

-- Check if a cell (list of Blocks) contains LineBreak or <br> (RawInline)
local function cell_has_linebreak(cell)
  for _, block in ipairs(cell) do
    if block.content then
      for _, inline in ipairs(block.content) do
        if inline.t == "LineBreak" then
          return true
        end
        if inline.t == "RawInline" and inline.format == "html" then
          local raw = inline.text or ""
          if raw:match("^<br%s*/?>$") or raw == "<br>" then
            return true
          end
        end
      end
    end
  end
  return false
end

-- Replace RawInline <br> with LineBreak so pandoc.write produces \newline
local function normalize_br_in_block(block)
  if not block.content then return block end
  local new_content = pandoc.List()
  for _, inline in ipairs(block.content) do
    if inline.t == "RawInline" and inline.format == "html" then
      local raw = inline.text or ""
      if raw:match("^<br%s*/?>$") or raw == "<br>" then
        new_content:insert(pandoc.LineBreak())
      else
        new_content:insert(inline)
      end
    else
      new_content:insert(inline)
    end
  end
  return pandoc.Plain(new_content)
end

-- Convert cell content to LaTeX and wrap in \makecell, replacing \newline with \\
local function convert_cell_to_makecell(cell)
  -- Normalize <br> to LineBreak so pandoc.write produces \newline
  local normalized = pandoc.List()
  for _, block in ipairs(cell) do
    if block.content then
      normalized:insert(normalize_br_in_block(block))
    else
      normalized:insert(block)
    end
  end
  local doc = pandoc.Pandoc(normalized)
  local latex = pandoc.write(doc, "latex")
  -- Pandoc outputs \newline for LineBreak; \makecell needs \\
  latex = latex:gsub("\\newline", "\\\\")
  -- Remove trailing newline from pandoc.write
  latex = latex:gsub("\n$", "")
  return pandoc.RawBlock("latex", "\\makecell[tl]{" .. latex .. "}")
end

-- Process a single cell (Blocks list), return modified Blocks
local function process_cell(cell)
  if not cell_has_linebreak(cell) then
    return cell
  end
  return { convert_cell_to_makecell(cell) }
end

-- Table filter: use simple table for easy cell iteration
local function Table(tbl)
  if not is_latex_format() then
    return nil
  end

  local simple = pandoc.utils.to_simple_table(tbl)

  -- Process header cells
  for i, cell in ipairs(simple.headers) do
    simple.headers[i] = process_cell(cell)
  end

  -- Process body cells
  for i, row in ipairs(simple.rows) do
    for j, cell in ipairs(row) do
      simple.rows[i][j] = process_cell(cell)
    end
  end

  return pandoc.utils.from_simple_table(simple)
end

return { Table = Table }
