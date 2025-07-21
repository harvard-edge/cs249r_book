-- lua/inject_xrefs.lua

-- This script is a Pandoc Lua filter that injects cross-references
-- into a Quarto document based on a JSON file.

-- It reads a JSON file with an array of files, each containing sections with targets,
-- then injects a "Chapter Connection" callout box into the appropriate sections.
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
--               "similarity": 0.72
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
  log_info("📁 Using cross-references file path: '" .. json_path .. "'")

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
  
  log_info("Loading cross-references from: " .. json_path)
  
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
  
  log_success("Loaded " .. total_refs .. " cross-references from " .. total_files .. " source(s)")
  stats.total_references = total_refs
  
  return data
end

-- Global variable to store the lookup table
local refs_by_source_id = {}

-- Global variable to track current file being processed
local current_file = nil

-- Helper function to get current file name from Pandoc environment
local function get_current_file()
  -- Try to get the input file from Pandoc's environment
  if PANDOC_STATE and PANDOC_STATE.input_files and #PANDOC_STATE.input_files > 0 then
    local full_path = PANDOC_STATE.input_files[1]
    return full_path:match("([^/\\]+)$") -- Extract just the filename
  end
  return nil
end

-- Helper function to detect if we're processing HTML output
local function is_html_output(meta)
  -- Check if this is HTML format processing
  if meta and meta.format then
    local format = pandoc.utils.stringify(meta.format):lower()
    return format:match("html")
  end
  return false
end

-- Initialize cross-references from metadata
local function init_cross_references(meta)
  log_info("🚀 Initializing Cross-Reference Injection Filter")
  log_info("================================================")

  -- Detect current processing context
  current_file = get_current_file()
  local is_html = is_html_output(meta)
  
  if is_html and current_file then
    log_info("📄 HTML mode - processing file: " .. current_file)
  else
    log_info("📚 Book mode - processing all files")
  end

  local xrefs_data = load_cross_references(meta)
  if not xrefs_data then
    log_info("No cross-references to inject - filter will pass through")
    return
  end

  -- Organize references by source section ID for quick lookup
  local total_refs_processed = 0
  local files_processed = 0
  
  if xrefs_data and xrefs_data.cross_references then
    -- New format: array of file objects with sections and targets
    for _, file_data in ipairs(xrefs_data.cross_references) do
      local filename = file_data.file
      
      -- In HTML mode, only process the current file
      if is_html and current_file and filename ~= current_file then
        -- Skip files that don't match current file in HTML mode
        goto continue
      end
      
      files_processed = files_processed + 1
      log_info("Processing file: " .. filename .. " (" .. #file_data.sections .. " sections)")
      
      for _, section in ipairs(file_data.sections) do
        local source_section_id = section.section_id
        local source_section_title = section.section_title
        
        -- In HTML mode, be less verbose about individual sections
        if not (is_html and current_file) then
          log_info("  Section: " .. source_section_id .. " (" .. #section.targets .. " targets)")
        end
        
        for _, target in ipairs(section.targets) do
          -- Convert to internal format
          local ref = {
            source_section_id = source_section_id,
            source_section_title = source_section_title,
            target_section_id = target.target_section_id,
            target_section_title = target.target_section_title,
            connection_type = target.connection_type,
            similarity = target.similarity
          }
          
          if not refs_by_source_id[source_section_id] then
            refs_by_source_id[source_section_id] = {}
          end
          table.insert(refs_by_source_id[source_section_id], ref)
          total_refs_processed = total_refs_processed + 1
        end
      end
      
      ::continue::
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
          similarity = suggestion.similarity
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

  log_info("Cross-reference lookup table built:")
  if is_html and current_file then
    log_info("  • File: " .. current_file)
    log_info("  • " .. section_count .. " sections with references")
    log_info("  • " .. total_refs_processed .. " references processed")
  else
    log_info("  • " .. files_processed .. " files processed")
    log_info("  • " .. section_count .. " sections with references") 
    log_info("  • " .. total_refs_processed .. " total references processed")
  end
  
  stats.total_references = total_refs_processed
end

-- Function to create the connection box as simple markdown-style content
local function create_connection_box(refs)
  local previews = {}
  local foundations = {}
  
  -- Sort references by type
  for _, ref in ipairs(refs) do
    if ref.connection_type == "Preview" then
      table.insert(previews, ref)
    elseif ref.connection_type == "Foundation" then
      table.insert(foundations, ref)
    end
  end
    
  -- Don't create a box if there are no valid references
  if #previews == 0 and #foundations == 0 then
    return nil
  end

  -- Create the content paragraphs
  local content_paras = {}
  
  -- Add preview references
  for _, ref in ipairs(previews) do
    -- Extract chapter name from target_section_id (e.g., sec-training-overview -> training)
    local chapter_name = "Related"
    if ref.target_section_id then
      local extracted = ref.target_section_id:match("^sec%-([^%-]+)")
      if extracted then
        chapter_name = extracted:gsub("_", " ")
        chapter_name = string.upper(string.sub(chapter_name, 1, 1)) .. string.sub(chapter_name, 2)
      end
    end
    
    local para_content = {
      pandoc.Str("→ "),
      pandoc.Strong({pandoc.Str(chapter_name .. ":")}),
      pandoc.Space(),
      pandoc.Str(ref.target_section_title)
    }
    table.insert(content_paras, pandoc.Para(para_content))
  end
  
  -- Add foundation references  
  for _, ref in ipairs(foundations) do
    -- Extract chapter name from target_section_id (e.g., sec-training-overview -> training)
    local chapter_name = "Related"
    if ref.target_section_id then
      local extracted = ref.target_section_id:match("^sec%-([^%-]+)")
      if extracted then
        chapter_name = extracted:gsub("_", " ")
        chapter_name = string.upper(string.sub(chapter_name, 1, 1)) .. string.sub(chapter_name, 2)
      end
    end
    
    local para_content = {
      pandoc.Str("↩ "),
      pandoc.Strong({pandoc.Str(chapter_name .. ":")}),
      pandoc.Space(),
      pandoc.Str(ref.target_section_title)
    }
    table.insert(content_paras, pandoc.Para(para_content))
  end

  -- Create the exact structure that Quarto's callout system produces
  local details_content_div = pandoc.Div(content_paras)
  
  local summary = pandoc.RawInline("html", "<summary><strong>Chapter connection</strong></summary>")
  local details = pandoc.RawBlock("html", 
    "<details class=\"callout-chapter-connection fbx-simplebox fbx-default\">" ..
    "<summary><strong>Chapter connection</strong></summary>" ..
    "<div>")
  
  local details_end = pandoc.RawBlock("html", "</div></details>")
  
  -- Generate unique ID for this callout
  local unique_id = "callout-chapter-connection*-auto-" .. tostring(math.random(1000, 9999))
  
  local inner_div = pandoc.Div(
    {details, table.unpack(content_paras), details_end},
    {
      id = unique_id,
      class = "callout-chapter-connection margin-chapter-connection"
    }
  )
  
  local callout_div = pandoc.Div({inner_div}, {class = "margin-container"})
  
  return callout_div
end

-- Process the entire document to inject cross-references
local function inject_cross_references(doc)
  local new_blocks = {}
  
  for i, block in ipairs(doc.blocks) do
    table.insert(new_blocks, block)
    
    -- Look for headers with identifiers
    if block.t == "Header" and block.identifier and block.identifier ~= "" then
      local section_id = block.identifier
      stats.sections_found = stats.sections_found + 1
      
      -- Debug: Show what sections we're finding (only in book mode)
      if not current_file and section_id:match("^sec-introduction-") then
        log_info("🔍 Found introduction section: " .. section_id)
      end
      
      -- Check if we have references for this section
      if refs_by_source_id[section_id] then
        local num_refs = #refs_by_source_id[section_id]
        log_info("📍 Processing section: " .. section_id .. " (" .. num_refs .. " references)")
        
        local connection_box = create_connection_box(refs_by_source_id[section_id])
      
        if connection_box then
          -- Inject the connection box right after the header
          table.insert(new_blocks, connection_box)
          
          stats.injections_made = stats.injections_made + 1
          log_success("✨ Injected connection box after: " .. section_id)
        end
              else
        -- Only log missing references in book mode for debugging
        if not current_file and section_id:match("^sec-introduction-") then
          log_info("❌ No references found for: " .. section_id)
        end
      end
    end
  end
  
  -- Final summary
  if stats.sections_found > 0 then
    log_info("================================================")
    if current_file then
      log_success("📊 INJECTION SUMMARY (" .. current_file .. "):")
    else
      log_success("📊 INJECTION SUMMARY:")
    end
    log_success("   • Sections processed: " .. stats.sections_found)
    log_success("   • Connection boxes injected: " .. stats.injections_made)
    if stats.injections_made > 0 then
      log_success("   • Total cross-references available: " .. stats.total_references)
      log_success("   • Injection rate: " .. string.format("%.1f%%", (stats.injections_made / stats.sections_found) * 100))
    end
    log_info("================================================")
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