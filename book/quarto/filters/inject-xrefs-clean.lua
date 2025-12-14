-- lua/inject-xrefs.lua

-- This script is a Pandoc Lua filter that injects cross-references
-- into a Quarto document based on individual chapter xrefs.json files.

-- Initialize logging counters
local stats = {
  files_processed = 0,
  sections_found = 0,
  injections_made = 0,
  total_references = 0
}

-- Helper function for formatted logging (consistent with other filters)
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

local function log_error(message)
  io.stderr:write("❌ [Cross-Ref Filter] " .. message .. "\n")
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

-- Helper function to detect chapter from headers
local function detect_chapter_from_headers(doc)
  -- Look for headers with IDs that match chapter patterns
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.identifier then
      -- Check if this looks like a chapter-level header
      local chapter = string.match(block.identifier, "^sec%-([^%-]+)%-")
      if chapter then
        return chapter
      end
    end
  end
  return nil
end

-- Get the current document's chapter name from the input file path
local function get_chapter_name(doc)
  -- First try to detect from document headers
  local chapter = detect_chapter_from_headers(doc)
  if chapter then
    log_info("Detected chapter from headers: " .. chapter)
    return chapter
  end

  -- Fallback to input file detection
  local input_file = quarto.doc.input_file
  if not input_file then
    log_warning("Could not determine input file")
    return nil
  end

  log_info("Input file: " .. input_file)

  -- Extract chapter name from path like "contents/core/introduction/introduction.qmd"
  chapter = string.match(input_file, "contents/core/([^/]+)/")
  if not chapter then
    -- Try pattern for when running from quarto directory
    chapter = string.match(input_file, "([^/]+)/[^/]+%.qmd$")
  end

  -- If still no match, try to extract from filename
  if not chapter then
    local filename = string.match(input_file, "([^/]+)%.qmd$")
    if filename and filename ~= "index" then
      chapter = filename
    end
  end

  return chapter
end

-- Load cross-references for the current chapter
local function load_chapter_xrefs(chapter_name)
  if not chapter_name then
    log_info("Could not determine chapter name - skipping cross-references")
    return nil
  end

  -- Construct path to chapter's xrefs file
  local xrefs_path = "contents/core/" .. chapter_name .. "/" .. chapter_name .. "_xrefs.json"

  log_info("Loading cross-references from: " .. xrefs_path)

  local json_content = read_file(xrefs_path)
  if not json_content then
    -- Try alternative path if running from different directory
    xrefs_path = "quarto/contents/core/" .. chapter_name .. "/" .. chapter_name .. "_xrefs.json"
    json_content = read_file(xrefs_path)
  end

  if not json_content then
    log_warning("Cross-references file not found for chapter: " .. chapter_name)
    return nil
  end

  local ok, data = pcall(quarto.json.decode, json_content)
  if not ok then
    log_error("Could not parse " .. xrefs_path .. " - invalid JSON format")
    return nil
  end

  -- Count total references
  local total_refs = 0
  if data and data.cross_references then
    for _, refs in pairs(data.cross_references) do
      if type(refs) == "table" then
        total_refs = total_refs + #refs
      end
    end
  end

  log_success("Loaded " .. total_refs .. " cross-references for " .. chapter_name)

  return data
end

-- Check if cross-references are enabled in config
local function is_xrefs_enabled(meta)
  -- Check if filter-metadata and cross-references are defined
  if not meta or not meta["filter-metadata"] or not meta["filter-metadata"]["cross-references"] then
    log_info("No filter-metadata.cross-references configuration - filter disabled")
    return false
  end

  local xref_config = meta["filter-metadata"]["cross-references"]

  -- Check if enabled
  local enabled = xref_config.enabled
  if enabled ~= nil then
    local enabled_str = pandoc.utils.stringify(enabled):lower()
    if enabled_str ~= "true" then
      log_info("Cross-references disabled in configuration")
      return false
    end
  end

  return true
end

-- Define chapter order from _quarto.yml (keep for potential future use)
local chapter_order = {
  introduction = 1,
  ml_systems = 2,
  dl_primer = 3,
  dnn_architectures = 4,
  workflow = 5,
  data_engineering = 6,
  frameworks = 7,
  training = 8,
  efficient_ai = 9,
  optimizations = 10,
  hw_acceleration = 11,
  benchmarking = 12,
  ops = 13,
  ondevice_learning = 14,
  robust_ai = 15,
  privacy_security = 16,
  responsible_ai = 17,
  sustainable_ai = 18,
  ai_for_good = 19,
  frontiers = 20,
  conclusion = 21
}

-- Chapter display names with proper capitalization
local chapter_names = {
  introduction = "Introduction",
  ml_systems = "ML Systems",
  dl_primer = "DL Primer",
  dnn_architectures = "DNN Architectures",
  workflow = "Workflow",
  data_engineering = "Data Engineering",
  frameworks = "Frameworks",
  training = "Training",
  efficient_ai = "Efficient AI",
  optimizations = "Optimizations",
  hw_acceleration = "HW Acceleration",
  benchmarking = "Benchmarking",
  ops = "Ops",
  ondevice_learning = "On-Device Learning",
  robust_ai = "Robust AI",
  privacy_security = "Privacy & Security",
  responsible_ai = "Responsible AI",
  sustainable_ai = "Sustainable AI",
  ai_for_good = "AI for Good",
  frontiers = "Frontiers",
  conclusion = "Conclusion"
}

-- Format a single cross-reference entry
local function format_xref_entry(ref)
  -- Build the reference text (no arrows, cleaner format)
  local ref_text = ""

  -- Add chapter name with proper capitalization
  if ref.target_chapter then
    local display_name = chapter_names[ref.target_chapter] or ref.target_chapter
    ref_text = ref_text .. "**" .. display_name .. "**: "
  end

  -- Add section reference with ?? for unbuilt chapters
  if ref.target_section then
    -- Check if this is PDF format
    if quarto.doc.isFormat("pdf") then
      ref_text = ref_text .. "(§\\ref{" .. ref.target_section .. "})"
    else
      -- For HTML, just use ?? for now
      ref_text = ref_text .. "(§??)"
    end
  end

  -- Clean up and format explanation
  if ref.explanation and ref.explanation ~= "" then
    -- Remove redundant prefixes
    local clean_explanation = ref.explanation
    clean_explanation = string.gsub(clean_explanation, "^Builds on foundational concepts: ", "")
    clean_explanation = string.gsub(clean_explanation, "^Essential prerequisite covering: ", "")
    clean_explanation = string.gsub(clean_explanation, "^Extends into: ", "")
    clean_explanation = string.gsub(clean_explanation, "^Foundation for: ", "")

    -- Extract just the key concepts if it's a list
    local concepts = string.match(clean_explanation, "^([^~]+~[^,]+)")
    if concepts and string.find(clean_explanation, "~") then
      -- Simplify concept pairs (remove tildes, limit to first few)
      local concept_list = {}
      for concept_pair in string.gmatch(concepts, "[^,]+") do
        local clean_concept = string.gsub(concept_pair, "~", "/")
        clean_concept = string.gsub(clean_concept, "^%s+", "")
        clean_concept = string.gsub(clean_concept, "%s+$", "")
        table.insert(concept_list, clean_concept)
        if #concept_list >= 3 then break end  -- Limit to 3 concepts
      end
      clean_explanation = table.concat(concept_list, ", ")
      if string.match(concepts, ",") and #concept_list >= 3 then
        clean_explanation = clean_explanation .. "..."
      end
    else
      -- For other explanations, just limit length
      if string.len(clean_explanation) > 80 then
        clean_explanation = string.sub(clean_explanation, 1, 77) .. "..."
      end
    end

    ref_text = ref_text .. " — " .. clean_explanation
  end

  return ref_text
end

-- Create a connection box for a section's cross-references
local function create_connection_box(refs)
  if not refs or #refs == 0 then
    return nil
  end

  -- Filter and sort references by priority
  local filtered_refs = {}
  for _, ref in ipairs(refs) do
    -- Include high-priority references or those with strong connections
    if (ref.priority and ref.priority <= 2) or
       (ref.strength and ref.strength > 0.2) or
       (ref.quality and ref.quality > 0.8) then
      table.insert(filtered_refs, ref)
    end
  end

  -- Limit to top 5 references
  if #filtered_refs > 5 then
    -- Sort by strength or quality
    table.sort(filtered_refs, function(a, b)
      local a_score = (a.strength or 0) * (a.quality or 1)
      local b_score = (b.strength or 0) * (b.quality or 1)
      return a_score > b_score
    end)
    -- Keep only top 5
    local top_refs = {}
    for i = 1, 5 do
      top_refs[i] = filtered_refs[i]
    end
    filtered_refs = top_refs
  end

  if #filtered_refs == 0 then
    return nil
  end

  -- Build content blocks
  local content_blocks = {}

  for _, ref in ipairs(filtered_refs) do
    local ref_text = format_xref_entry(ref)
    local ref_doc = pandoc.read(ref_text, "markdown")
    if ref_doc.blocks[1] then
      table.insert(content_blocks, ref_doc.blocks[1])
    end
  end

  -- Create a div with callout-chapter-connection class
  local callout_div = pandoc.Div(
    content_blocks,
    pandoc.Attr("", {"callout", "callout-chapter-connection"}, {})
  )

  return callout_div
end

-- Number of lines to ensure with \Needspace
local NEEDSPACE_LINES = 6

-- Helper to insert \Needspace{<n>\baselineskip}
local function needspace_block()
  return pandoc.RawBlock('latex',
    '\\Needspace{' .. NEEDSPACE_LINES .. '\\baselineskip}')
end

-- Main filter function
return {
  {
    Pandoc = function(doc)
      -- Check if enabled
      if not is_xrefs_enabled(doc.meta) then
        return doc
      end

      log_info("🚀 Initializing Cross-Reference Injection Filter")

      -- Get current chapter
      local chapter_name = get_chapter_name(doc)
      if not chapter_name then
        log_warning("Could not determine chapter - skipping")
        return doc
      end

      log_info("📖 Processing chapter: " .. chapter_name)

      -- Load cross-references for this chapter
      local xrefs_data = load_chapter_xrefs(chapter_name)
      if not xrefs_data or not xrefs_data.cross_references then
        log_info("No cross-references found for chapter: " .. chapter_name)
        return doc
      end

      -- Check if this is PDF output
      local is_pdf = quarto.doc.isFormat("pdf")

      -- Process document blocks
      local new_blocks = {}

      for _, block in ipairs(doc.blocks) do
        -- Check if this is a header with cross-references
        if block.t == "Header" and block.identifier and block.identifier ~= "" then
          local section_refs = xrefs_data.cross_references[block.identifier]

          if section_refs and #section_refs > 0 then
            -- For PDF: ensure minimum lines before section
            if is_pdf then
              table.insert(new_blocks, needspace_block())
            end

            -- Insert the header
            table.insert(new_blocks, block)
            stats.sections_found = stats.sections_found + 1

            -- Create and insert the connection box
            local connection_box = create_connection_box(section_refs)
            if connection_box then
              table.insert(new_blocks, connection_box)
              stats.injections_made = stats.injections_made + 1
              log_info("Injected connections for section: " .. block.identifier)
            end
          else
            -- No cross-references for this section
            table.insert(new_blocks, block)
          end
        else
          -- Not a header, just pass through
          table.insert(new_blocks, block)
        end
      end

      -- Summary
      if stats.injections_made > 0 then
        log_success("📊 SUMMARY: " .. stats.injections_made .. " connection boxes injected into " ..
                   stats.sections_found .. " sections")
      else
        log_info("No cross-references injected for this chapter")
      end

      return pandoc.Pandoc(new_blocks, doc.meta)
    end
  }
}
