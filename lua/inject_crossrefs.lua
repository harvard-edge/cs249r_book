-- lua/inject_xrefs.lua

-- This script is a Pandoc Lua filter that injects cross-references
-- into a Quarto document based on a JSON file.

-- It reads a JSON file with an array of files, each containing sections with targets,
-- then injects a "Chapter Connection" callout box into the appropriate sections.
-- 
-- Cross-references are formatted in academic style with bold directional arrows:
-- **→** Forward Section Title (§\ref{sec-target-id}) **—** AI-generated explanation
-- **←** Background Section Title (§\ref{sec-target-id}) **—** AI-generated explanation
-- • General Section Title (§\ref{sec-target-id}) **—** AI-generated explanation (fallback)
-- 
-- Expected JSON format:
-- {
--   "cross_references": [
--     {
--       "file": "introduction.qmd",
--       "sections": [
--         {
--           "section_id": "sec-intro-overview",
--           "section_title": "Overview",
--           "targets": [
--             {
--               "target_section_id": "sec-training-basics",
--               "target_section_title": "Training Basics", 
--               "connection_type": "Preview",
--               "similarity": 0.72,
--               "explanation": "provides essential background on neural network mathematics"
--             }
--           ]
--         }
--       ]
--     }
--   ]
-- }

-- Initialize logging counters
local stats = {
  files_processed = 0,
  sections_found = 0,
  injections_made = 0,
  total_references = 0
}

-- Helper function for formatted logging
local function log_info(message)
  io.stderr:write("🔗 [Cross-Ref Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_success(message)
  io.stderr:write("✅ [Cross-Ref Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_warning(message)
  io.stderr:write("⚠️  [Cross-Ref Filter] " .. message .. "\n")
  io.stderr:flush()
end

-- Helper function to read file content
local function read_file(path)
  local file = io.open(path, "r")
  if not file then return nil end
  local content = file:read("*a")
  file:close()
  return content
end

-- Load and parse the cross-references JSON file
local function load_cross_references(meta)
  -- Check if cross-references are defined in _quarto.yml
  if not meta or not meta["cross-references"] then
    log_info("No cross-references configuration in _quarto.yml - filter disabled")
    return nil
  end
  
  local xref_config = meta["cross-references"]
  
  -- Check if enabled
  if xref_config.enabled then
    local enabled_str = pandoc.utils.stringify(xref_config.enabled):lower()
    if enabled_str ~= "true" then
      log_info("Cross-references disabled in _quarto.yml")
      return nil
    end
  end

  -- Get file path from config
  local config_file = ""
  if xref_config.file then
    config_file = pandoc.utils.stringify(xref_config.file)
  end

  -- Use the path from config as-is (Quarto has already resolved it correctly)
  local json_path = config_file

  -- Try to read the file from the specified path only
  local json_content = read_file(json_path)
  
  if not json_content then
    local error_msg = "❌ FATAL ERROR: Cross-references file not found: " .. json_path .. "\n" ..
                      "The file '" .. config_file .. "' is specified in _quarto.yml cross-references.file but does not exist.\n" ..
                      "Please ensure the file exists at the specified path or update your _quarto.yml configuration.\n" ..
                      "BUILD STOPPED."
    
    -- Write error to stderr and stop the build
    io.stderr:write("\n" .. error_msg .. "\n\n")
    io.stderr:flush()
    os.exit(1)
  end
  
  local ok, data = pcall(quarto.json.decode, json_content)
  if not ok then
    log_warning("Could not parse " .. json_path .. " - invalid JSON format")
    return nil
  end
  
  -- Count total references (handle both formats)
  local total_refs = 0
  local total_files = 0
  
  if data and data.cross_references then
    -- New format: array of file objects with sections and targets
    for _, file_data in ipairs(data.cross_references) do
      total_files = total_files + 1
      for _, section in ipairs(file_data.sections) do
        total_refs = total_refs + #section.targets
      end
    end
  elseif data and data.suggestions then
    -- Legacy format: flat array of suggestions
    total_refs = #data.suggestions
    total_files = 1 -- Single suggestions array
  end
  
  return data
end

-- Global variable to store the lookup table
local refs_by_source_id = {}

-- Global variable to track if current document has any cross-references
local has_cross_references = false

-- Helper function to check if document has any sections with cross-references
local function document_has_cross_references(doc, refs_lookup)
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.identifier and block.identifier ~= "" then
      if refs_lookup[block.identifier] then
        return true
      end
    end
  end
  return false
end

-- Initialize cross-references from metadata  
local function init_cross_references(meta)
  local xrefs_data = load_cross_references(meta)
  if not xrefs_data then
    return
  end

  -- Organize references by source section ID for quick lookup (silently)
  local total_refs_processed = 0
  
  if xrefs_data and xrefs_data.cross_references then
    -- New format: array of file objects with sections and targets
    for _, file_data in ipairs(xrefs_data.cross_references) do
      for _, section in ipairs(file_data.sections) do
        local source_section_id = section.section_id
        local source_section_title = section.section_title
        
        for _, target in ipairs(section.targets) do
          -- Convert to internal format
          local ref = {
            source_section_id = source_section_id,
            source_section_title = source_section_title,
            target_section_id = target.target_section_id,
            target_section_title = target.target_section_title,
            connection_type = target.connection_type,
            similarity = target.similarity,
            explanation = target.explanation or ""
          }
          
          if not refs_by_source_id[source_section_id] then
            refs_by_source_id[source_section_id] = {}
          end
          table.insert(refs_by_source_id[source_section_id], ref)
          total_refs_processed = total_refs_processed + 1
        end
      end
    end
  elseif xrefs_data and xrefs_data.suggestions then
    -- New format: flat array of suggestions
    log_info("Processing " .. #xrefs_data.suggestions .. " suggestions from xref.json")
    for _, suggestion in ipairs(xrefs_data.suggestions) do
      if suggestion.enabled then
        -- Convert new format to internal format
        local ref = {
          source_section_id = suggestion.source.section_id,
          source_section_title = suggestion.source.section_title,
          target_section_id = suggestion.target.section_id,
          target_section_title = suggestion.target.section_title,
          connection_type = suggestion.target.connection_type == "foundation" and "Foundation" or "Preview",
          similarity = suggestion.similarity,
          explanation = suggestion.explanation or ""
        }
        
        if not refs_by_source_id[ref.source_section_id] then
          refs_by_source_id[ref.source_section_id] = {}
        end
        table.insert(refs_by_source_id[ref.source_section_id], ref)
        total_refs_processed = total_refs_processed + 1
      end
    end
  end

  -- Count the sections properly (# doesn't work on tables with non-numeric keys)
  local section_count = 0
  for _ in pairs(refs_by_source_id) do
    section_count = section_count + 1
  end

  -- Store for later use
  stats.total_references = total_refs_processed
end

-- Function to create the connection box in academic style
local function create_connection_box(refs)
  -- Don't create a box if there are no valid references
  if #refs == 0 then
    return nil
  end

  -- Build content as academic-style bullet points
  local content_blocks = {}
  
  -- Add each reference with directional arrow and explanation
  for _, ref in ipairs(refs) do
         local arrow_content = ""
    
         -- Create the academic-style entry with bold directional arrows
     local arrow = ""
     if ref.connection_type == "Preview" then
       arrow = "→ "  -- Forward reference (material comes later)
     elseif ref.connection_type == "Background" then
       arrow = "← "  -- Backward reference (material from earlier)
     else
       arrow = "• "      -- Fallback bullet for unclear direction
     end
     
          if ref.explanation and ref.explanation ~= "" then
       -- With explanation: **→** Title (§\ref{sec-id}) **—** explanation
       arrow_content = arrow .. ref.target_section_title .. " (§\\ref{" .. ref.target_section_id .. "}) " .. ref.explanation
     else
       -- Without explanation: **→** Title (§\ref{sec-id}) or • Title (§\ref{sec-id})
       arrow_content = arrow .. ref.target_section_title .. " (§\\ref{" .. ref.target_section_id .. "})"
     end
     
     local display_type = ref.connection_type or "Unknown"
     if display_type ~= "Preview" and display_type ~= "Background" then
       display_type = display_type .. " [Fallback Bullet]"
     end
          local arrow_doc = pandoc.read(arrow_content, "markdown")
     if arrow_doc.blocks[1] then
       table.insert(content_blocks, arrow_doc.blocks[1])
    end
  end

  -- Create a simple div with callout-chapter-connection class
  -- This structure is exactly what margin-connections.lua expects
  
  local callout_div = pandoc.Div(
    content_blocks,
    pandoc.Attr("", {"callout", "callout-chapter-connection"}, {})
  )
  
  return callout_div
end

-- Process the entire document to inject cross-references
local function inject_cross_references(doc)
  -- Check if this document has any cross-references
  has_cross_references = document_has_cross_references(doc, refs_by_source_id)
  
  if not has_cross_references then
    -- No cross-references for this document, process silently
    return doc
  end
  
  -- Document has cross-references, show initialization info
  log_info("🚀 Cross-Reference Injection Filter")
  log_info("🔍 Document has cross-references - processing...")
  
  local new_blocks = {}
  
  for i, block in ipairs(doc.blocks) do
    table.insert(new_blocks, block)
    
    -- Look for headers with identifiers
    if block.t == "Header" and block.identifier and block.identifier ~= "" then
      local section_id = block.identifier
      stats.sections_found = stats.sections_found + 1
      
      -- Check if we have references for this section
      if refs_by_source_id[section_id] then
        local connection_box = create_connection_box(refs_by_source_id[section_id])
      
        if connection_box then
          -- Inject the connection box right after the header
          table.insert(new_blocks, connection_box)
          stats.injections_made = stats.injections_made + 1
        end
      end
    end
  end
  
  -- Final summary
  if stats.injections_made > 0 then
    log_success("📊 SUMMARY: " .. stats.injections_made .. " connection boxes injected")
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end

-- This is the main filter function called by Pandoc
return {
  -- Initialize cross-references when we first see metadata
  { Meta = function(meta)
    init_cross_references(meta)
    return meta
  end },
  
  -- Process the entire document to inject cross-references
  { Pandoc = inject_cross_references }
} 