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

-- Part ordering for Roman numerals (numbered parts only)
local numbered_part_order = {
  "foundations", "principles", "optimization", "deployment", "governance", "futures"
}

-- Get part number for a given key (numbered parts only)
local function get_part_number(key)
  for i, part_key in ipairs(numbered_part_order) do
    if part_key == key then
      return i
    end
  end
  return nil
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
      
      -- ROUTING LOGIC: Transform based on key name
      
      -- 1. BOOK DIVISIONS: Major book structure sections
      if normalized_key == "frontmatter" or normalized_key == "main_content" or normalized_key == "backmatter" or normalized_key == "labs" then
        part_cmd = "\\division{" .. formatted_title .. "}"
        log_info("🔄 Replacing key '" .. key .. "' with division: '" .. formatted_title .. "' (no description)")
        return {
          pandoc.RawBlock("latex", part_cmd)
        }
      
      -- 2. LAB PLATFORMS: Circuit-style neural network design
      elseif normalized_key == "arduino" or normalized_key == "xiao" or normalized_key == "grove" or normalized_key == "raspberry" or normalized_key == "shared" then
        part_cmd = "\\labdivision{" .. formatted_title .. "}"
        log_info("🔄 Replacing key '" .. key .. "' with lab division: '" .. formatted_title .. "' (circuit style, no description)")
        return {
          pandoc.RawBlock("latex", part_cmd)
        }
      
      -- 3. NUMBERED PARTS: Main content sections (foundations, principles, etc.)
      else
        part_cmd = "\\part{" .. formatted_title .. "}"
        log_info("🔄 Replacing key '" .. key .. "' with numbered part: '" .. formatted_title .. "' + description")
        return {
          pandoc.RawBlock("latex", setpartsummary_cmd),
          pandoc.RawBlock("latex", part_cmd)
        }
      end
    else
      log_error("UNDEFINED KEY: '" .. key .. "' not found in part_summaries.yml")
      log_error("Available keys: frontmatter, foundations, principles, optimization, deployment, governance, futures, labs, arduino, xiao, grove, raspberry, shared")
      log_error("Build stopped to prevent incorrect part titles.")
      error("Part summary filter failed: undefined key '" .. key .. "' in \\part{key:" .. key .. "}")
    end
  end
  return nil
end

-- Initialize the filter with Meta handler
function Meta(meta)
  if quarto.doc.is_format("pdf") then
    local filter_metadata = meta["filter-metadata"]
    if filter_metadata and filter_metadata["part-summaries"] then
      local config = filter_metadata["part-summaries"]
      local file_path = pandoc.utils.stringify(config.file or "")
      local enabled = pandoc.utils.stringify(config.enabled or "true"):lower() == "true"
      
      if enabled and file_path ~= "" then
        log_info("🚀 Initializing Part Summary Filter")
        log_info("📂 Loading part summaries from: " .. file_path)
        summaries = read_summaries(file_path)
        has_part_summaries = true
        log_success("Part Summary Filter activated for PDF format")
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
  { RawBlock = RawBlock }
}
