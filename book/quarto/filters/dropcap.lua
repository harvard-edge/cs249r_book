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

-- Only apply to PDF/LaTeX output
if not (quarto.doc.is_format("latex") or quarto.doc.is_format("pdf")) then
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
  if el.level == 2 and not applied_dropcap and not found_numbered_header then
    -- Check if this header has the .unnumbered class
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
    end
  end

  return nil
end

function Para(el)
  -- Only process if we found a numbered header and haven't applied dropcap yet
  if not found_numbered_header or applied_dropcap then
    return nil
  end

  -- Check that the paragraph starts with text content (not an image, div, etc.)
  if #el.content == 0 then
    return nil
  end

  local first_inline = el.content[1]

  -- Only apply to paragraphs that start with a Str (regular text)
  if first_inline.t ~= "Str" then
    return nil
  end

  local text = first_inline.text

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

  -- Replace the first Str with the lettrine command
  -- Remove the original first inline and prepend the lettrine
  local new_content = pandoc.List({lettrine_open})
  for i = 2, #el.content do
    new_content:insert(el.content[i])
  end

  applied_dropcap = true
  return pandoc.Para(new_content)
end

-- Return the filter — Header must run before Para to set the flag
return {
  { Header = Header },
  { Para = Para }
}
