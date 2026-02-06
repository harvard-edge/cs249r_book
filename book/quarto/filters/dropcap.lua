-- =============================================================================
-- DROP CAP FILTER
-- =============================================================================
-- Automatically applies \lettrine{X}{rest} to the first paragraph of the
-- first numbered section in each chapter. PDF/LaTeX output only.
--
-- Logic:
--   1. Process document in order using Pandoc's Blocks filter
--   2. For each chapter (H1), find the first numbered H2, then the next Para
--   3. Apply \lettrine to that paragraph's first word
--   4. One drop cap per chapter
--
-- This keeps QMD files clean — no manual \lettrine calls needed.
-- =============================================================================

-- Robust format check
local function is_target_format()
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

-- Debug: print to stderr
local function debug(msg)
  io.stderr:write("[dropcap] " .. msg .. "\n")
end

if not is_target_format() then
  debug("Skipping - not a target format")
  return {}
end

debug("Filter active for format")

-- Check if a header has the .unnumbered class
local function is_unnumbered(el)
  if el.classes then
    for _, cls in ipairs(el.classes) do
      if cls == "unnumbered" then
        return true
      end
    end
  end
  return false
end

-- Apply lettrine to a paragraph, returns modified Para or nil if not applicable
local function apply_lettrine(el)
  if #el.content == 0 then
    debug("  -> empty paragraph, skipping")
    return nil
  end

  local first_str_index = nil
  local first_str_el = nil

  -- Find the first Str element, skipping RawInline, Spans, Spaces, etc.
  for i, inline in ipairs(el.content) do
    if inline.t == "Str" then
      first_str_index = i
      first_str_el = inline
      debug("  -> found Str at index " .. i .. ": '" .. inline.text .. "'")
      break
    elseif inline.t == "RawInline" or inline.t == "Span" or inline.t == "Space" or inline.t == "SoftBreak" then
      debug("  -> skipping " .. inline.t)
    else
      debug("  -> ABORT: unexpected element type " .. inline.t .. " before first Str")
      return nil
    end
  end

  if not first_str_index or not first_str_el then
    debug("  -> no Str found in paragraph")
    return nil
  end

  local text = first_str_el.text or ""
  if #text == 0 then
    debug("  -> first Str is empty")
    return nil
  end

  -- Extract first character
  local first_char = text:sub(1, 1)
  local rest_of_first_word = text:sub(2)

  debug("  -> APPLYING DROPCAP: '" .. first_char .. "' + '" .. rest_of_first_word .. "'")

  -- Build the lettrine command
  local lettrine_open = pandoc.RawInline('latex',
    '\\lettrine{' .. first_char .. '}{' .. rest_of_first_word .. '}')

  -- Construct new content list
  local new_content = pandoc.List()

  -- 1. Append skipped elements (but drop leading spaces)
  for i = 1, first_str_index - 1 do
    local inline = el.content[i]
    if inline.t ~= "Space" and inline.t ~= "SoftBreak" then
      new_content:insert(inline)
    end
  end

  -- 2. Append the lettrine
  new_content:insert(lettrine_open)

  -- 3. Append the rest of the paragraph
  for i = first_str_index + 1, #el.content do
    new_content:insert(el.content[i])
  end

  return pandoc.Para(new_content)
end

-- Main filter: process all blocks in document order
function Blocks(blocks)
  local new_blocks = pandoc.List()
  local state = "looking_for_chapter"  -- States: looking_for_chapter, looking_for_numbered_h2, looking_for_para, done
  local chapter_name = ""

  for _, block in ipairs(blocks) do
    local modified_block = block

    if block.t == "Header" then
      if block.level == 1 then
        -- New chapter: reset state
        chapter_name = pandoc.utils.stringify(block.content)
        debug("Found chapter: " .. chapter_name)
        state = "looking_for_numbered_h2"

      elseif block.level == 2 and state == "looking_for_numbered_h2" then
        local header_text = pandoc.utils.stringify(block.content)
        if is_unnumbered(block) then
          debug("Found H2 (unnumbered): " .. header_text .. " - skipping")
        else
          debug("Found H2 (numbered): " .. header_text .. " - will apply dropcap to next Para")
          state = "looking_for_para"
        end
      end

    elseif block.t == "Para" and state == "looking_for_para" then
      debug("Found Para after numbered H2, attempting dropcap...")
      local result = apply_lettrine(block)
      if result then
        modified_block = result
        state = "done"
        debug("Dropcap applied for chapter: " .. chapter_name)
      else
        debug("Could not apply dropcap to this Para, trying next...")
        -- Keep looking for a suitable paragraph
      end
    end

    new_blocks:insert(modified_block)
  end

  return new_blocks
end

return {
  { Blocks = Blocks }
}
