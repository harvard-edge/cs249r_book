-- 🔧 Normalize keys (lowercase, trim leading/trailing whitespace)
local function normalize(str)
  return str:lower():gsub("^%s+", ""):gsub("%s+$", "")
end

-- 📄 Read summaries.yml into a Lua table
local function read_summaries(path)
  local summaries = {}
  local current_key = nil
  local buffer = {}

  local file = io.open(path, "r")
  if not file then return summaries end

  for line in file:lines() do
    local key, val = line:match("^([%w %-%._]+):%s*>?%s*(.*)")
    if key then
      if current_key and #buffer > 0 then
        summaries[normalize(current_key)] = table.concat(buffer, " ")
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
  end

  file:close()
  return summaries
end

-- ✅ Load summaries once at filter initialization
local summaries = read_summaries("summaries.yml")

-- 🧠 Insert \setpartsummary before any heading whose title matches a summary key
function Header(el)
  local title = pandoc.utils.stringify(el.content)
  local key = normalize(title)
  local summary = summaries[key]

  if summary then
    local latex = "\\setpartsummary{" .. summary .. "}"
    return {
      pandoc.RawBlock("latex", latex),
      el
    }
  end

  return nil
end
