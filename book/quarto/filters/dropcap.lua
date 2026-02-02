-- =============================================================================
-- DROP CAP FILTER
-- =============================================================================
-- Automatically applies \lettrine{X}{rest} to the first paragraph of the
-- first numbered section in each chapter. PDF/LaTeX output only.
--
-- Logic:
--   1. Walk each chapter's blocks looking for the first numbered Header (level 2+)
--   2. After finding it, locate the next Para element
--   3. Extract the first letter and wrap it with \lettrine
--   4. Stop processing that chapter (one drop cap per chapter)
--
-- This keeps QMD files clean — no manual \lettrine calls needed.
-- =============================================================================

-- Robust format check
local function is_target_format()
  return quarto.doc.is_format("latex") or 
         quarto.doc.is_format("pdf") or 
         quarto.doc.is_format("titlepage-pdf") or
         quarto.doc.is_format("beamer")
end

if not is_target_format() then
  return {}
end

local found_numbered_header = false
local applied_dropcap = false

function Header(el)
  -- Reset state at chapter boundaries (level 1 headers)
  if el.level == 1 then
    found_numbered_header = false
    applied_dropcap = false
    return nil
  end

  -- Look for first numbered section header (level 2) that is NOT unnumbered
  if el.level == 2 then
    -- If we already applied a dropcap in this chapter, stop looking
    if applied_dropcap then
      found_numbered_header = false
      return nil
    end

    -- Check if this header has the .unnumbered class
    -- Use manual loop for maximum compatibility
    local is_unnumbered = false
    if el.classes then
      for _, cls in ipairs(el.classes) do
        if cls == "unnumbered" then
          is_unnumbered = true
          break
        end
      end
    end

    if not is_unnumbered then
      found_numbered_header = true
    else
      -- If we hit an unnumbered header (like Purpose), we are not ready yet
      found_numbered_header = false
    end
  end

  return nil
end

function Para(el)
  -- Only process if we found a numbered header and haven't applied dropcap yet
  if not found_numbered_header or applied_dropcap then
    return nil
  end

  -- Check that the paragraph starts with text content
  if #el.content == 0 then
    return nil
  end

  local first_str_index = nil
  local first_str_el = nil

  -- Find the first Str element, skipping RawInline (like \index), Spans, Spaces, etc.
  for i, inline in ipairs(el.content) do
    if inline.t == "Str" then
      first_str_index = i
      first_str_el = inline
      break
    elseif inline.t == "RawInline" or inline.t == "Span" or inline.t == "Space" or inline.t == "SoftBreak" then
      -- Skip these elements (indices, labels, spaces, line breaks)
    else
      -- If we encounter anything else (Image, Code, Strong, Emph, etc.) before the first Str,
      -- we abort because dropcap logic gets complicated or invalid.
      return nil
    end
  end

  if not first_str_index then
    return nil
  end

  local text = first_str_el.text

  -- Skip empty strings
  if #text == 0 then
    return nil
  end

  -- Extract first character (handle UTF-8 multibyte)
  local first_char = text:sub(1, 1)
  local rest_of_first_word = text:sub(2)

  -- Build the lettrine command: \lettrine{X}{rest_of_word}
  -- The second argument is the rest of the first word in small caps
  local lettrine_open = pandoc.RawInline('latex',
    '\\lettrine{' .. first_char .. '}{' .. rest_of_first_word .. '}')

  -- Construct new content list
  local new_content = pandoc.List()
  
  -- 1. Append all skipped elements (indices, etc.) BUT skip spaces/breaks to ensure flush left
  for i = 1, first_str_index - 1 do
    local inline = el.content[i]
    -- We keep RawInline (indices) and Spans (labels)
    -- We DROP Spaces and SoftBreaks that appear before the first letter
    -- to prevent indentation issues with the drop cap
    if inline.t ~= "Space" and inline.t ~= "SoftBreak" then
      new_content:insert(inline)
    end
  end
  
  -- 2. Append the lettrine element
  new_content:insert(lettrine_open)
  
  -- 3. Append the rest of the paragraph
  for i = first_str_index + 1, #el.content do
    new_content:insert(el.content[i])
  end

  applied_dropcap = true
  return pandoc.Para(new_content)
end

return {
  { Header = Header },
  { Para = Para }
}