-- 🔧 Normalize keys (lowercase, trim leading/trailing whitespace)
local function normalize(str)
  return str:lower():gsub("^%s+", ""):gsub("%s+$", "")
end

-- 🗝️ Extract short key from part header title
local function extract_key(title)
  local normalized = normalize(title)
  
  -- Map full part titles to short keys
  local key_map = {
    ["part i — systems foundations"] = "foundations",
    ["part ii — design principles"] = "principles", 
    ["part iii — system optimization"] = "optimization",
    ["part iv — deployment and reliability"] = "deployment",
    ["part v — responsible ai"] = "responsible",
    ["part vi — impact and futures"] = "futures",
    ["part vii — laboratory exercises"] = "labs",
    ["part viii — arduino labs"] = "arduino",
    ["part ix — seeed xiao labs"] = "xiao",
    ["part x — grove vision labs"] = "grove",
    ["part xi — raspberry pi labs"] = "raspberry",
    ["part xii — shared labs"] = "shared"
  }
  
  return key_map[normalized]
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
  
  -- Parse YAML manually for the new structure
  local entries_loaded = 0
  local current_key = nil
  local description_lines = {}
  local in_description = false
  
  for line in content:gmatch("[^\r\n]+") do
    -- Match key line: - key: "part i — systems foundations"
    local key = line:match('%s*%-%s*key:%s*"([^"]+)"')
    if key then
      -- Save previous entry if exists
      if current_key and #description_lines > 0 then
        summaries[current_key] = table.concat(description_lines, " "):gsub("^%s+", ""):gsub("%s+$", "")
        entries_loaded = entries_loaded + 1
      end
      current_key = key
      description_lines = {}
      in_description = false
    -- Match description start: description: >
    elseif line:match('%s*description:%s*>?') then
      in_description = true
    -- Match description content (indented lines after description:)
    elseif in_description and current_key then
      local desc_content = line:match('%s%s%s*(.+)')
      if desc_content then
        table.insert(description_lines, desc_content)
      elseif line:match('^%s*$') then
        -- Empty line, continue
      elseif line:match('^%s*%-%s*key:') then
        -- Hit next key, handle this line again
        if current_key and #description_lines > 0 then
          summaries[current_key] = table.concat(description_lines, " "):gsub("^%s+", ""):gsub("%s+$", "")
          entries_loaded = entries_loaded + 1
        end
        -- Parse this key line
        local key = line:match('%s*%-%s*key:%s*"([^"]+)"')
        if key then
          current_key = key
          description_lines = {}
          in_description = false
        end
      end
    end
  end
  
  -- Save last entry
  if current_key and #description_lines > 0 then
    summaries[current_key] = table.concat(description_lines, " "):gsub("^%s+", ""):gsub("%s+$", "")
    entries_loaded = entries_loaded + 1
  end
  
  if entries_loaded > 0 then
    log_success("Loaded " .. entries_loaded .. " part summaries from " .. path)
  else
    log_warning("No part summaries found in " .. path)
  end
  
  return summaries
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

-- 🧠 Insert \setpartsummary before any heading whose title matches a summary key
function Header(el)
  -- Skip processing if we don't have any summaries loaded
  if not has_part_summaries then
    return nil
  end
  
  local title = pandoc.utils.stringify(el.content)
  local key = extract_key(title)

  if key and summaries[key] then
    local summary = summaries[key]
    log_info("💡 Injecting part summary for: '" .. title .. "' (key: " .. key .. ")")
    local latex = "\\setpartsummary{" .. summary .. "}"
    return {
      pandoc.RawBlock("latex", latex),
      el
    }
  end

  return nil
end

-- Register the filter with meta handler
return {
  { Meta = handle_meta },
  { Header = Header }
}
