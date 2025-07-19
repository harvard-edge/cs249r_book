-- inject_xrefs.lua
-- Cross-Reference Injection Filter for ML Systems Book
-- Reads cross-reference suggestions from JSON and injects callout blocks

local json = require("pandoc.json")
local utils = pandoc.utils

-- Global state: cross-reference suggestions organized by section ID
local xref_suggestions = {}
local current_document_file = "unknown"

-- Load cross-reference JSON data
local function load_xref_data(path)
  local f, err = io.open(path, "r")
  if not f then
    io.stderr:write("⚠️  [XREF] Cross-reference file not found: " .. path .. " (this is optional)\n")
    return nil
  end
  
  local content = f:read("*all")
  f:close()

  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    io.stderr:write("❌ [XREF] Failed to parse JSON from file: " .. path .. "\n")
    return nil
  end
  
  return data
end

-- Register cross-reference suggestions from JSON data
local function register_xref_suggestions(data, file_path)
  local suggestions_count = 0
  
  if not data.suggestions then
    io.stderr:write("⚠️  [XREF] No 'suggestions' field found in " .. file_path .. "\n")
    return suggestions_count
  end
  
  for _, suggestion in ipairs(data.suggestions) do
    -- Only process enabled suggestions
    if suggestion.enabled ~= false then
      local section_id = suggestion.source.section_id
      
      -- Ensure section_id starts with '#'
      if not section_id:match("^#") then
        section_id = "#" .. section_id
      end
      
      -- Store suggestion by section ID
      if not xref_suggestions[section_id] then
        xref_suggestions[section_id] = {}
      end
      
      table.insert(xref_suggestions[section_id], {
        target_id = suggestion.target.section_id,
        target_title = suggestion.target.enhanced_title or suggestion.target.section_title,
        connection_type = suggestion.target.connection_type or "related",
        similarity = suggestion.similarity
      })
      
      suggestions_count = suggestions_count + 1
    end
  end
  
  return suggestions_count
end

-- Create a cross-reference callout block
local function create_xref_callout(suggestions)
  local callout_content = {}
  
  -- Create callout header
  table.insert(callout_content, pandoc.Para({
    pandoc.Strong({ pandoc.Str("Foundation") }),
    pandoc.Str(": This builds on ")
  }))
  
  -- Build the reference list
  local ref_parts = {}
  
  for i, suggestion in ipairs(suggestions) do
    -- Add comma separator for multiple references
    if i > 1 then
      table.insert(ref_parts, pandoc.Str(", "))
    end
    
    -- Add connection type icon
    local icon = ""
    if suggestion.connection_type == "foundation" then
      icon = "🔗 "
    elseif suggestion.connection_type == "example" then
      icon = "📖 "
    else
      icon = ""
    end
    
    -- Add the reference
    if icon ~= "" then
      table.insert(ref_parts, pandoc.Str(icon))
    end
    
    table.insert(ref_parts, pandoc.Str(suggestion.target_title))
    table.insert(ref_parts, pandoc.Str(" ("))
    table.insert(ref_parts, pandoc.Str("@" .. suggestion.target_id))
    table.insert(ref_parts, pandoc.Str(")"))
  end
  
  -- Add period at the end
  table.insert(ref_parts, pandoc.Str("."))
  
  -- Create the paragraph with references
  local ref_para = pandoc.Para(ref_parts)
  table.insert(callout_content, ref_para)
  
  -- Create the callout div
  local callout_div = pandoc.Div(
    callout_content,
    {
      class = "callout-chapter-connection"
    }
  )
  
  return callout_div
end

-- Meta phase: load cross-reference data
local function handle_meta(meta)
  -- Try to get current document filename
  if PANDOC_DOCUMENT and PANDOC_DOCUMENT.meta and PANDOC_DOCUMENT.meta.filename then
    current_document_file = utils.stringify(PANDOC_DOCUMENT.meta.filename)
  elseif PANDOC_STATE and PANDOC_STATE.input_files and PANDOC_STATE.input_files[1] then
    current_document_file = PANDOC_STATE.input_files[1]
  end

  -- Get xref configuration from global metadata
  local xref_config = meta["xref-config"] or {}
  local file_path = xref_config["file-path"] or "cross_references.json"
  local auto_load = xref_config["auto-load"] ~= false -- default to true
  
  -- Convert Meta objects to strings if needed
  if type(file_path) == "table" then
    file_path = utils.stringify(file_path)
  end
  
  if type(auto_load) == "table" and auto_load.t == "MetaBool" then
    auto_load = auto_load.bool
  elseif type(auto_load) == "table" then
    auto_load = utils.stringify(auto_load) ~= "false"
  end

  -- Load cross-reference data if auto-load is enabled
  if auto_load then
    local data = load_xref_data(file_path)
    if data then
      local count = register_xref_suggestions(data, file_path)
      if count > 0 then
        io.stderr:write("✅ [XREF] Loaded " .. count .. " cross-reference suggestions from " .. file_path .. "\n")
      end
    end
  end
  
  return meta
end

-- Main document processing: inject cross-references
local function inject_xrefs(doc)
  local new_blocks = {}
  local blocks = doc.blocks
  
  for i = 1, #blocks do
    local block = blocks[i]
    
    -- Add the current block
    table.insert(new_blocks, block)
    
    -- Check if this is a level 2 header (##) with an identifier
    if block.t == "Header" and block.level == 2 and block.identifier then
      local section_id = "#" .. block.identifier
      local suggestions = xref_suggestions[section_id]
      
      if suggestions and #suggestions > 0 then
        -- Limit to maximum 2 suggestions for readability
        local limited_suggestions = {}
        for j = 1, math.min(2, #suggestions) do
          table.insert(limited_suggestions, suggestions[j])
        end
        
        -- Create and insert the callout
        local callout = create_xref_callout(limited_suggestions)
        table.insert(new_blocks, callout)
        
        -- Add some spacing
        table.insert(new_blocks, pandoc.Para({}))
      end
    end
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end

-- Register the filter
return {
  { Meta = handle_meta },
  { Pandoc = inject_xrefs }
} 