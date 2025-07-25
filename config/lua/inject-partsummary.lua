-- 🔧 Normalize keys (lowercase, trim leading/trailing whitespace)
local function normalize(str)
  return str:lower():gsub("^%s+", ""):gsub("%s+$", "")
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
  local current_key = nil
  local buffer = {}

  local file = io.open(path, "r")
  if not file then 
    log_warning("Failed to open part summaries file: " .. path)
    return summaries 
  end

  local entries_loaded = 0
  for line in file:lines() do
    local key, val = line:match("^([%w %-%._]+):%s*>?%s*(.*)")
    if key then
      if current_key and #buffer > 0 then
        summaries[normalize(current_key)] = table.concat(buffer, " ")
        entries_loaded = entries_loaded + 1
        buffer = {}
      end
      current_key = key
      if val and val ~= "" then
        table.insert(buffer, val)
      end
    elseif current_key and line:match("^%s") then
      local trimmed = line:gsub("^%s+", "")
      table.insert(buffer, trimmed)
    end
  end

  if current_key and #buffer > 0 then
    summaries[normalize(current_key)] = table.concat(buffer, " ")
    entries_loaded = entries_loaded + 1
  end

  file:close()
  
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
  local key = normalize(title)
  local summary = summaries[key]

  if summary then
    log_info("💡 Injecting part summary for: '" .. title .. "'")
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
