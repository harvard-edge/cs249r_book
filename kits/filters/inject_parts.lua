-- ===============================================================================
-- PART STYLING LUA FILTER
-- ===============================================================================
--
-- This filter transforms \part{key:xxx} commands in QMD files into appropriate
-- LaTeX commands based on the key name, creating a structured book hierarchy.
--
-- ROUTING LOGIC:
-- 1. Book Divisions (frontmatter, main_content, backmatter, labs) → \division{title}
--    - Clean centered styling with geometric background
--    - No descriptions displayed
--
-- 2. Lab Platforms (arduino, xiao, grove, raspberry, shared) → \labdivision{title}
--    - Circuit-style neural network design with nodes and connections
--    - No descriptions displayed
--
-- 3. Numbered Parts (foundations, principles, etc.) → \part{title}
--    - Roman numeral styling (via \titleformat{\part})
--    - Includes part description
--
-- USAGE IN QMD FILES:
-- Simply use \part{key:foundations} and this filter will:
-- 1. Look up the key in part_summaries.yml
-- 2. Extract title and description
-- 3. Generate appropriate LaTeX command
-- 4. Include \setpartsummary{description} if needed
--
-- EXAMPLE TRANSFORMATIONS:
-- \part{key:foundations} → \part{Systems Foundations} + description
-- \part{key:labs} → \division{Labs} (clean geometric style)
-- \part{key:arduino} → \labdivision{Arduino Labs} (circuit-style neural network)
-- \part{key:frontmatter} → \division{Frontmatter} (clean geometric style)
-- ===============================================================================

-- 🔧 Normalize keys (lowercase, trim leading/trailing whitespace)
local function normalize(str)
  return str:lower():gsub("^%s+", ""):gsub("%s+$", "")
end

-- 🗝️ Extract key from LaTeX part command
local function extract_key_from_latex(content)
  -- Look for \part{key:xxx} pattern in content
  local key = content:match("\\part%{key:([^}]+)%}")
  return key
end

-- Helper function for formatted logging
local function log_info(message)
  io.stderr:write("📄 [Part Summary Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_success(message)
  io.stderr:write("✅ [Part Summary Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_warning(message)
  io.stderr:write("⚠️  [Part Summary Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_error(message)
  io.stderr:write("❌ [Part Summary Filter] " .. message .. "\n")
  io.stderr:flush()
end

-- Roman numeral conversion
local function to_roman(num)
  local values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
  local numerals = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
  local result = ""
  for i, value in ipairs(values) do
    while num >= value do
      result = result .. numerals[i]
      num = num - value
    end
  end
  return result
end

-- Dynamic part numbering system
local current_division = nil
local part_counter = 0

-- Reset counter when new division is encountered
local function reset_part_counter(division)
  if current_division ~= division then
    current_division = division
    part_counter = 0
    log_info("🔄 New division encountered: '" .. division .. "' - resetting part counter to 0")
  end
end

-- Get next part number for current division
local function get_next_part_number()
  part_counter = part_counter + 1
  return part_counter
end

-- Helper function to format title with part number
local function format_part_title(key, title, numbered)
  -- Always return just the clean title without any "Part X —" prefix
  -- The LaTeX template handles Roman numerals and "Part X" labels separately
  return title
end

-- 📄 Read summaries.yml into a Lua table
local function read_summaries(path)
  local summaries = {}

  local file = io.open(path, "r")
  if not file then
    log_warning("Failed to open part summaries file: " .. path)
    return summaries
  end

  local content = file:read("*all")
  file:close()

  -- Parse YAML manually for the new structure with parts array
  local entries_loaded = 0
  local current_key = nil
  local current_title = nil
  local current_type = nil
  local current_numbered = nil
  local description_lines = {}
  local in_parts_section = false
  local in_description = false

  for line in content:gmatch("[^\r\n]+") do
    if line:match('^parts:') then
      in_parts_section = true
    elseif in_parts_section and line:match('%s*%-%s*key:%s*"([^"]+)"') then
      if current_key and current_title then
        local description = table.concat(description_lines, " "):gsub("^%s+", ""):gsub("%s+$", "")
        summaries[normalize(current_key)] = {
          title = current_title,
          description = description,
          type = current_type or "part",
          numbered = current_numbered or false
        }
        entries_loaded = entries_loaded + 1
        log_info("📝 Loaded: '" .. current_key .. "' → '" .. current_title .. "' (type: " .. (current_type or "part") .. ", numbered: " .. tostring(current_numbered or false) .. ")")
      end
      current_key = line:match('%s*%-%s*key:%s*"([^"]+)"')
      current_title = nil
      current_type = nil
      current_numbered = nil
      description_lines = {}
      in_description = false
    elseif in_parts_section and current_key and line:match('%s*title:%s*"([^"]+)"') then
      current_title = line:match('%s*title:%s*"([^"]+)"')
    elseif in_parts_section and current_key and line:match('%s*type:%s*"([^"]+)"') then
      current_type = line:match('%s*type:%s*"([^"]+)"')
    elseif in_parts_section and current_key and line:match('%s*numbered:%s*(true|false)') then
      current_numbered = line:match('%s*numbered:%s*(true|false)') == "true"
    elseif in_parts_section and current_key and line:match('%s*description:%s*>?') then
      in_description = true
    elseif in_description and current_key then
      local desc_content = line:match('%s%s%s*(.+)')
      if desc_content then
        table.insert(description_lines, desc_content)
      elseif line:match('^%s*$') then
        -- Empty line, continue
      else
        -- End of description
        in_description = false
      end
    end
  end

  -- Handle the last entry
  if current_key and current_title then
    local description = table.concat(description_lines, " "):gsub("^%s+", ""):gsub("%s+$", "")
    summaries[normalize(current_key)] = {
      title = current_title,
      description = description,
      type = current_type or "part",
      numbered = current_numbered or false
    }
    entries_loaded = entries_loaded + 1
    log_info("📝 Loaded: '" .. current_key .. "' → '" .. current_title .. "' (type: " .. (current_type or "part") .. ", numbered: " .. tostring(current_numbered or false) .. ")")
  end

  log_success("Successfully loaded " .. entries_loaded .. " part summaries")
  return summaries
end

-- Load summaries from metadata
local has_part_summaries = false
local summaries = {}

-- Validation function to check all keys in the document
local function validate_all_keys()
  if not has_part_summaries then return end

  local used_keys = {}
  local invalid_keys = {}

  -- Collect all keys used in the document
  for key, _ in pairs(summaries) do
    used_keys[key] = true
  end

  -- Check if any keys are missing from part_summaries.yml
  for key, _ in pairs(used_keys) do
    if not summaries[key] then
      table.insert(invalid_keys, key)
    end
  end

  -- If there are invalid keys, report them all at once
  if #invalid_keys > 0 then
    log_error("❌ CRITICAL ERROR: Multiple undefined keys found:")
    for _, key in ipairs(invalid_keys) do
      log_error("   - '" .. key .. "' not found in part_summaries.yml")
    end
    log_error("🔍 Available keys: frontmatter, main_content, foundations, principles, optimization, deployment, trustworthy, futures, labs, arduino, xiao, grove, raspberry, shared, backmatter")
    log_error("🛑 Build stopped - fix all undefined keys before proceeding")
    error("Part summary filter failed: multiple undefined keys found. Please check your .qmd files and part_summaries.yml for consistency.")
  end
end

-- Pre-scan function to validate all keys before processing
local function prescan_document_keys(doc)
  if not has_part_summaries then return end

  log_info("🔍 Pre-scanning document for part keys...")

  local found_keys = {}
  local invalid_keys = {}
  local key_locations = {}

  -- Scan all RawBlocks for \part{key:xxx} patterns
  local function scan_blocks(blocks)
    for i, block in ipairs(blocks) do
      if block.t == "RawBlock" and block.format == "latex" then
        local key = extract_key_from_latex(block.text)
        if key then
          local normalized_key = normalize(key)
          found_keys[normalized_key] = true

          -- Check if key is valid
          if not summaries[normalized_key] then
            table.insert(invalid_keys, normalized_key)
            key_locations[normalized_key] = i
          end
        end
      end

      -- Recursively scan nested blocks
      if block.content then
        scan_blocks(block.content)
      end
    end
  end

  -- Scan the document
  scan_blocks(doc.blocks)

  -- Report findings
  if next(found_keys) then
    log_info("📋 Found keys in document:")
    for key, _ in pairs(found_keys) do
      if summaries[key] then
        log_info("   ✅ '" .. key .. "' - valid")
      else
        log_error("   ❌ '" .. key .. "' - INVALID (location: block " .. (key_locations[key] or "unknown") .. ")")
      end
    end
  else
    log_info("📋 No part keys found in document")
  end

  -- Report available keys for reference
  log_info("📚 Available keys in part_summaries.yml:")
  for key, _ in pairs(summaries) do
    log_info("   - '" .. key .. "'")
  end

  -- If there are invalid keys, stop the build
  if #invalid_keys > 0 then
    log_error("❌ CRITICAL ERROR: Invalid keys found during pre-scan:")
    for _, key in ipairs(invalid_keys) do
      log_error("   - '" .. key .. "' not found in part_summaries.yml")
    end
    log_error("🛑 Build stopped - fix all invalid keys before proceeding")
    log_error("💡 Check your .qmd files for \\part{key:" .. table.concat(invalid_keys, "} or \\part{key:") .. "} commands")
    error("Part summary filter failed: invalid keys found during pre-scan. Please check your .qmd files and part_summaries.yml for consistency.")
  else
    log_success("✅ Pre-scan validation passed - all keys are valid")
  end
end

-- Debug function to help identify the source of problematic keys
local function debug_key_source(key, el)
  log_error("🔍 DEBUG: Key '" .. key .. "' found in RawBlock")
  log_error("📍 RawBlock content: " .. (el.text or "nil"))
  log_error("📍 RawBlock format: " .. (el.format or "nil"))

  -- Try to extract more context about where this key came from
  if el.text then
    local context = string.sub(el.text, 1, 200) -- First 200 chars for context
    log_error("📍 Context: " .. context)
  end
end

-- 🏁 Main transformation function
-- This function intercepts \part{key:xxx} commands and transforms them
-- into appropriate LaTeX commands based on the routing logic above
function RawBlock(el)
  if not has_part_summaries then return nil end
  if el.format ~= "latex" then return nil end

  local key = extract_key_from_latex(el.text)
  if key then
    local normalized_key = normalize(key)
    if summaries[normalized_key] then
      local part_entry = summaries[normalized_key]
      local title = part_entry.title
      local description = part_entry.description
      local part_type = part_entry.type or "part"
      local numbered = part_entry.numbered or false
      local formatted_title = format_part_title(normalized_key, title, numbered)

      local setpartsummary_cmd = "\\setpartsummary{" .. description .. "}"
      local part_cmd

      -- ROUTING LOGIC: Transform based on type field from YAML

      -- 1. BOOK DIVISIONS: Major book structure sections
      if part_type == "division" then
        part_cmd = "\\division{" .. formatted_title .. "}"
        local toc_cmd = "\\addtocontents{toc}{\\par\\addvspace{12pt}\\noindent\\hfil\\bfseries\\color{crimson}" .. formatted_title .. "\\color{black}\\hfil\\par\\addvspace{6pt}}"
        local line_cmd = "\\addtocontents{toc}{\\par\\noindent\\hfil{\\color{crimson}\\rule{0.6\\textwidth}{0.5pt}}\\hfil\\par\\addvspace{6pt}}"
        log_info("🔄 Replacing key '" .. key .. "' with division: '" .. formatted_title .. "' (with TOC entry + crimson line)")
        return {
          pandoc.RawBlock("latex", toc_cmd),
          pandoc.RawBlock("latex", line_cmd),
          pandoc.RawBlock("latex", part_cmd)
        }

      -- 2. LAB PLATFORMS: Circuit-style neural network design
      elseif part_type == "lab" then
        part_cmd = "\\labdivision{" .. formatted_title .. "}"
        local toc_cmd = "\\addtocontents{toc}{\\par\\addvspace{12pt}\\noindent\\hfil\\bfseries\\color{crimson}" .. formatted_title .. "\\color{black}\\hfil\\par\\addvspace{6pt}}"
        log_info("🔄 Replacing key '" .. key .. "' with lab division: '" .. formatted_title .. "' (circuit style, clean TOC entry)")
        return {
          pandoc.RawBlock("latex", toc_cmd),
          pandoc.RawBlock("latex", part_cmd)
        }

      -- 3. NUMBERED PARTS: Main content sections (type: "part")
      elseif part_type == "part" then
        -- Reset counter if we're in a new division
        reset_part_counter(part_entry.division or "mainmatter")

        -- Get next part number and convert to Roman numeral
        local part_number = get_next_part_number()
        local roman_numeral = to_roman(part_number)

        part_cmd = "\\numberedpart{" .. formatted_title .. "}"  -- Use custom command instead
        local toc_cmd = "\\addtocontents{toc}{\\par\\addvspace{12pt}\\noindent\\hfil\\bfseries\\color{crimson}Part~" .. roman_numeral .. "~" .. formatted_title .. "\\color{black}\\hfil\\par\\addvspace{6pt}}"
        log_info("🔄 Replacing key '" .. key .. "' with numbered part: '" .. formatted_title .. "' (Part " .. roman_numeral .. ", division: " .. (part_entry.division or "mainmatter") .. ")")
        return {
          pandoc.RawBlock("latex", setpartsummary_cmd),
          pandoc.RawBlock("latex", toc_cmd),
          pandoc.RawBlock("latex", part_cmd)
        }
      end
    else
      -- Enhanced error reporting with more context
      log_error("❌ CRITICAL ERROR: UNDEFINED KEY '" .. key .. "' not found in part_summaries.yml")
      log_error("📍 Location: RawBlock processing")

      -- Add debug information to help identify the source
      debug_key_source(key, el)

      log_error("🔍 Available keys: frontmatter, main_content, foundations, principles, optimization, deployment, trustworthy, futures, labs, arduino, xiao, grove, raspberry, shared, backmatter")
      log_error("💡 Check your .qmd files for \\part{key:" .. key .. "} commands")
      log_error("🛑 Build stopped to prevent incorrect part titles.")

      -- Force immediate exit with detailed error
      local error_msg = string.format(
        "Part summary filter failed: undefined key '%s' in \\part{key:%s}. " ..
        "Available keys: frontmatter, main_content, foundations, principles, optimization, deployment, trustworthy, futures, labs, arduino, xiao, grove, raspberry, shared, backmatter. " ..
        "Please check your .qmd files and part_summaries.yml for consistency.",
        key, key
      )
      error(error_msg)
    end
  end
  return nil
end

-- Initialize the filter with Meta handler
function Meta(meta)
  if quarto.doc.is_format("pdf") or quarto.doc.is_format("titlepage-pdf") then
    local filter_metadata = meta["filter-metadata"]
    if filter_metadata and filter_metadata["part-summaries"] then
      local config = filter_metadata["part-summaries"]
      local file_path = pandoc.utils.stringify(config.file or "")
      local enabled = pandoc.utils.stringify(config.enabled or "true"):lower() == "true"

      if enabled and file_path ~= "" then
        log_info("🚀 Initializing Part Summary Filter")
        log_info("📂 Loading part summaries from: " .. file_path)

        -- Add error handling for file loading
        local success, result = pcall(read_summaries, file_path)
        if success then
          summaries = result
          -- Validate that summaries were loaded properly
          if type(summaries) == "table" and next(summaries) then
            has_part_summaries = true
            log_success("Part Summary Filter activated for PDF format")
          else
            log_error("❌ CRITICAL ERROR: part_summaries.yml is empty or invalid")
            log_error("📍 File path: " .. file_path)
            log_error("🛑 Build stopped - part_summaries.yml must contain valid entries")
            error("Part summary filter failed: part_summaries.yml is empty or contains no valid entries")
          end
        else
          log_error("❌ CRITICAL ERROR: Failed to load part_summaries.yml")
          log_error("📍 File path: " .. file_path)
          log_error("🔍 Error: " .. tostring(result))
          log_error("🛑 Build stopped - cannot proceed without part summaries")
          error("Part summary filter failed: cannot load part_summaries.yml from " .. file_path .. ". Error: " .. tostring(result))
        end
      else
        log_warning("Part Summary Filter disabled or no file specified")
      end
    else
      log_warning("Part Summary Filter metadata not found")
    end
  else
    log_info("Part Summary Filter skipped (not PDF format)")
  end
  return meta
end

-- Return the filter in the correct order
return {
  { Meta = Meta },
  { Pandoc = function(doc)
    -- Run pre-scan validation if part summaries are enabled
    if has_part_summaries then
      prescan_document_keys(doc)
    end
    return doc
  end },
  { RawBlock = RawBlock }
}
