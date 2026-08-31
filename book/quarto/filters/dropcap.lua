-- =============================================================================
-- DROP CAP FILTER
-- =============================================================================
-- Automatically applies \lettrine{X}{rest} to a chapter's selected opening
-- paragraph. PDF/LaTeX output only.
--
-- Logic:
--   1. Process document in order using Pandoc's Blocks filter
--   2. For each chapter (H1), select either its opening prose or the first
--      paragraph of its first numbered H2
--   3. Apply \lettrine to that paragraph's first word
--   4. One drop cap per chapter
--
-- Modes:
--   dropcap: numbered-section  (default; first numbered H2)
--   dropcap: chapter-opening   (first eligible paragraph after H1)
--   dropcap: false             (disabled)
-- A chapter H1 may override the document default with
-- `dropcap="chapter-opening"`; unlike per-file YAML, H1 attributes survive
-- Quarto's full-book assembly.
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

local dropcap_mode = "numbered-section"

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

local function chapter_mode(header)
  local value = header.attributes and header.attributes.dropcap
  if value == "chapter-opening" or value == "numbered-section" then
    return value
  elseif value == "false" then
    return "disabled"
  end
  return dropcap_mode
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

-- Read frontmatter metadata
local function read_meta(meta)
  if meta.dropcap ~= nil then
    local value = pandoc.utils.stringify(meta.dropcap)
    if value == "false" then
      dropcap_mode = "disabled"
    elseif value == "chapter-opening" or value == "numbered-section" then
      dropcap_mode = value
    elseif value == "true" then
      dropcap_mode = "numbered-section"
    else
      debug("Unknown dropcap mode '" .. value .. "'; using numbered-section")
    end
  end
  if dropcap_mode == "disabled" then
    debug("Dropcap disabled via frontmatter (dropcap: false)")
  else
    debug("Dropcap mode: " .. dropcap_mode)
  end
end

-- Main filter: process all blocks in document order
local function process_blocks(blocks)
  if dropcap_mode == "disabled" then
    return blocks
  end
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
        local mode = chapter_mode(block)
        debug("Chapter dropcap mode: " .. mode)
        if mode == "chapter-opening" then
          state = "looking_for_para"
        elseif mode == "disabled" then
          state = "done"
        else
          state = "looking_for_numbered_h2"
        end

      elseif block.level == 2 then
        if state == "looking_for_para" then
          -- A rejected candidate must not leak into a later section.
          debug("Reached next H2 before applying dropcap - stopping search")
          state = "done"
        elseif state == "looking_for_numbered_h2" then
          local header_text = pandoc.utils.stringify(block.content)
          if is_unnumbered(block) then
            debug("Found H2 (unnumbered): " .. header_text .. " - skipping")
          else
            debug("Found H2 (numbered): " .. header_text .. " - will apply dropcap to next Para")
            state = "looking_for_para"
          end
        end
      end

    elseif block.t == "Para" and state == "looking_for_para" then
      debug("Found candidate Para, attempting dropcap...")
      local result = apply_lettrine(block)
      if result then
        modified_block = result
        state = "done"
        debug("Dropcap applied for chapter: " .. chapter_name)
      else
        debug("Could not apply dropcap to this Para; searching within the same section...")
      end
    end

    new_blocks:insert(modified_block)
  end

  return new_blocks
end

-- Return filter traversal: Meta first (to read frontmatter), then Blocks
return {
  { Meta = read_meta },
  { Blocks = process_blocks }
}
