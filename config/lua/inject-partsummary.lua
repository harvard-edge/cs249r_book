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
      log_info("🔄 Replacing key '" .. key .. "' with title '" .. title .. "'")
      
      -- Replace the key with the actual title
      local new_latex = el.text:gsub("\\part%*?%{key:([^}]+)%}", "\\part*{" .. title .. "}")
      
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
