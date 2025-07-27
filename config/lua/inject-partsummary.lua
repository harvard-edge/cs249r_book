-- Helper function to convert number to Roman numerals
local function to_roman(num)
  local roman_numerals = {
    {1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"},
    {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"},
    {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}
  }
  
  local result = ""
  for _, pair in ipairs(roman_numerals) do
    local value, numeral = pair[1], pair[2]
    while num >= value do
      result = result .. numeral
      num = num - value
    end
  end
  return result
end

-- Part numbering mapping - defines the order and numbering of parts
local part_order = {
  "foundations",     -- Part I
  "principles",      -- Part II
  "optimization",    -- Part III
  "deployment",      -- Part IV
  "responsible",     -- Part V
  "futures",         -- Part VI
  "arduino",         -- Part VII
  "xiao",           -- Part VIII
  "grove",          -- Part IX
  "raspberry",      -- Part X
  "shared"          -- Part XI
}

-- Parts that should remain unnumbered (use \part*{} format)
local unnumbered_parts = {
  "frontmatter",
  "labs"
}

-- Helper function to check if a part should be unnumbered
local function is_unnumbered_part(key)
  for _, unnumbered_key in ipairs(unnumbered_parts) do
    if unnumbered_key == key then
      return true
    end
  end
  return false
end

-- Helper function to get part number for a key
local function get_part_number(key)
  for i, part_key in ipairs(part_order) do
    if part_key == key then
      return i
    end
  end
  return nil -- Key not found in ordered list
end

-- Helper function to format title with part number
local function format_part_title(key, title)
  -- If title already contains "Part", don't modify it
  if title:match("^Part ") then
    return title
  end
  
  -- Check if this should be unnumbered
  if is_unnumbered_part(key) then
    return title -- Just return the title as-is for unnumbered parts
  end
  
  -- For numbered parts, add Part X — prefix
  local part_num = get_part_number(key)
  if part_num then
    local roman = to_roman(part_num)
    return "Part " .. roman .. " — " .. title
  else
    -- Fallback: just return the title if key not in ordered list
    return title
  end
end

-- 🔧 Normalize keys (lowercase, trim leading/trailing whitespace)
local function normalize(str)
  return str:lower():gsub("^%s+", ""):gsub("%s+$", "")
end

-- 🗝️ Extract key from LaTeX part command
local function extract_key_from_latex(content)
  -- Look for \part*{key:xxx} pattern in content
  local key = content:match("\\part%*?%{key:([^}]+)%}")
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
  local in_parts_section = false
  
  for line in content:gmatch("[^\r\n]+") do
    -- Check if we're in the parts section
    if line:match('^parts:') then
      in_parts_section = true
    -- Match key line: - key: "frontmatter"
    elseif in_parts_section and line:match('%s*%-%s*key:%s*"([^"]+)"') then
      -- Save previous entry if exists
      if current_key and current_title then
        summaries[normalize(current_key)] = current_title
        entries_loaded = entries_loaded + 1
        log_info("📝 Loaded: '" .. current_key .. "' → '" .. current_title .. "'")
      end
      current_key = line:match('%s*%-%s*key:%s*"([^"]+)"')
      current_title = nil
    -- Match title line: title: "Frontmatter"
    elseif in_parts_section and current_key and line:match('%s*title:%s*"([^"]+)"') then
      current_title = line:match('%s*title:%s*"([^"]+)"')
    end
  end
  
  -- Save last entry
  if current_key and current_title then
    summaries[normalize(current_key)] = current_title
    entries_loaded = entries_loaded + 1
    log_info("📝 Loaded: '" .. current_key .. "' → '" .. current_title .. "'")
  end
  
  if entries_loaded > 0 then
    log_success("Loaded " .. entries_loaded .. " part titles from " .. path)
  else
    log_warning("No part titles found in " .. path)
  end
  
  return summaries
end

-- Helper function to get list of available keys for error reporting
local function get_available_keys()
  local keys = {}
  if summaries then
    for key, _ in pairs(summaries) do
      table.insert(keys, "'" .. key .. "'")
    end
    table.sort(keys)
  end
  return keys
end

-- ✅ Load summaries from metadata configuration
local summaries = {}
local has_part_summaries = false

-- Meta phase: read part summaries configuration from metadata
local function handle_meta(meta)
  -- Check if filter-metadata and part summaries are configured in _quarto.yml
  if not meta or not meta["filter-metadata"] or not meta["filter-metadata"]["part-summaries"] then
    log_info("No filter-metadata.part-summaries configuration in _quarto.yml - filter disabled")
    return meta
  end
  
  local part_config = meta["filter-metadata"]["part-summaries"]
  
  -- Check if enabled
  local enabled = part_config.enabled
  if enabled ~= nil then
    local enabled_str = pandoc.utils.stringify(enabled):lower()
    if enabled_str ~= "true" then
      log_info("Part summaries disabled in _quarto.yml")
      return meta
    end
  end

  -- Get file path from config
  local config_file = ""
  if part_config.file then
    config_file = pandoc.utils.stringify(part_config.file)
  end

  if config_file == "" then
    log_warning("Part summaries file not specified in _quarto.yml")
    return meta
  end

  log_info("🚀 Part Summary Injection Filter")
  log_info("📁 Loading part summaries from: " .. config_file)
  
  -- Load the summaries
  summaries = read_summaries(config_file)
  
  -- Check if we have any summaries
  if next(summaries) then
    has_part_summaries = true
  else
    log_warning("No part summaries loaded - filter will not process any headers")
  end
  
  return meta
end

-- 🧠 Replace \part*{key:xxx} with actual title from summaries
function RawBlock(el)
  -- Skip processing if we don't have any summaries loaded
  if not has_part_summaries then
    return nil
  end
  
  -- Only process LaTeX blocks
  if el.format ~= "latex" then
    return nil
  end
  
  local key = extract_key_from_latex(el.text)
  
  if key then
    local normalized_key = normalize(key)
    if summaries[normalized_key] then
      local title = summaries[normalized_key]
      local formatted_title = format_part_title(normalized_key, title)
      
      -- Determine if this should be numbered or unnumbered
      local new_latex
      if is_unnumbered_part(normalized_key) then
        -- Keep as unnumbered part
        new_latex = el.text:gsub("\\part%*?%{key:([^}]+)%}", "\\part*{" .. formatted_title .. "}")
        log_info("🔄 Replacing key '" .. key .. "' with unnumbered title '" .. formatted_title .. "'")
      else
        -- Change to numbered part
        new_latex = el.text:gsub("\\part%*?%{key:([^}]+)%}", "\\part{" .. formatted_title .. "}")
        log_info("🔄 Replacing key '" .. key .. "' with numbered title '" .. formatted_title .. "'")
      end
      
      return pandoc.RawBlock("latex", new_latex)
    else
      log_error("UNDEFINED KEY: '" .. key .. "' not found in part_summaries.yml")
      log_error("Available keys: frontmatter, foundations, principles, optimization, deployment, responsible, futures, labs, arduino, xiao, grove, raspberry, shared")
      log_error("Build stopped to prevent incorrect part titles.")
      error("Part summary filter failed: undefined key '" .. key .. "' in \\part*{key:" .. key .. "}")
    end
  end

  return nil
end

-- Keep the header function as a fallback (can be removed later)
function Header(el)
  return nil
end

-- Register the filter with meta handler
return {
  { Meta = handle_meta },
  { RawBlock = RawBlock }
}
