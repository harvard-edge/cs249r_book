-- lua/inject-xrefs.lua
-- Production version with hybrid placement strategy

-- Configuration for hybrid mode (recommended default)
local PLACEMENT_MODE = "hybrid"  -- hybrid mode balances overview with specific guidance

-- Thresholds for filtering connections
local STRENGTH_THRESHOLD = 0.22  -- Show connections with >22% strength
local PRIORITY_THRESHOLD = 3     -- Show priority 1-3 (all important connections)
local MAX_CHAPTER_REFS = 6       -- Show up to 6 chapter-level refs
local MAX_SECTION_REFS = 3       -- Show up to 3 section refs (more variety)

-- Initialize logging counters
local stats = {
  files_processed = 0,
  sections_found = 0,
  injections_made = 0,
  chapter_boxes = 0,
  section_boxes = 0,
  filtered_refs = 0,
  total_refs = 0
}

-- Diversity tracking: track which chapters have been shown
local shown_chapters = {}  -- tracks {chapter_name = count}
local shown_in_section = {} -- tracks chapters shown in current section
local section_counter = 0   -- counts sections processed

-- Calculate diversity score for a reference
local function get_diversity_score(ref)
  local chapter = ref.target_chapter
  if not chapter then return 1.0 end

  -- Penalize if shown in current section
  if shown_in_section[chapter] then
    return 0.1  -- Very low score if already in this section
  end

  -- Calculate recency penalty based on how many times shown
  local times_shown = shown_chapters[chapter] or 0
  local diversity_multiplier = 1.0 / (1.0 + times_shown * 0.5)

  return diversity_multiplier
end

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

-- Mode can be overridden via environment variable for testing
if os.getenv("XREF_MODE") then
  PLACEMENT_MODE = os.getenv("XREF_MODE")
  log_info("Using placement mode from environment: " .. PLACEMENT_MODE)
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
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.identifier then
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

  -- Extract chapter name from path
  chapter = string.match(input_file, "contents/core/([^/]+)/")
  if not chapter then
    chapter = string.match(input_file, "([^/]+)/[^/]+%.qmd$")
  end

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
  stats.total_refs = total_refs

  return data
end

-- Check if cross-references are enabled in config
local function is_xrefs_enabled(meta)
  if not meta or not meta["filter-metadata"] or not meta["filter-metadata"]["cross-references"] then
    log_info("No filter-metadata.cross-references configuration - filter disabled")
    return false
  end

  local xref_config = meta["filter-metadata"]["cross-references"]

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
  local ref_text = ""

  -- Add visual indicator for connection type
  -- Using LaTeX-safe symbols that work in both HTML and PDF
  local connection_indicators = {}

  if quarto.doc.isFormat("pdf") then
    -- LaTeX symbols that render properly in PDF
    connection_indicators = {
      prerequisite = "$\\triangleright$",  -- Math triangle right
      foundation = "$\\blacklozenge$",      -- Black diamond (requires pifont or similar)
      extends = "$\\nearrow$",              -- Northeast arrow
      complements = "$\\leftrightarrow$",   -- Double arrow
      applies = "$\\rightarrow$",           -- Right arrow
      optimizes = "$\\star$",               -- Star for performance
      considers = "$\\triangle$",           -- Triangle for warning
      explores = "$\\circ$",                -- Circle (simpler, more available)
      anticipates = "$\\Rightarrow$",       -- Double right arrow
      specializes = "$\\Diamond$"           -- Diamond (from latexsym)
    }
  else
    -- Unicode for HTML
    connection_indicators = {
      prerequisite = "▸",  -- Triangle pointing right
      foundation = "◆",    -- Diamond
      extends = "↗",       -- Arrow up-right
      complements = "⇄",   -- Double arrow
      applies = "→",       -- Right arrow
      optimizes = "⚡",    -- Lightning
      considers = "⚠",     -- Warning
      explores = "🔍",     -- Magnifying glass
      anticipates = "➤",   -- Forward arrow
      specializes = "◈"    -- Special diamond
    }
  end

  -- Get indicator or default
  local indicator = connection_indicators[ref.connection_type] or "•"

  -- Add chapter name with section reference if available
  if ref.target_chapter then
    local display_name = chapter_names[ref.target_chapter] or ref.target_chapter

    -- Include section reference inline if available
    if ref.target_section then
      if quarto.doc.isFormat("pdf") then
        -- For PDF, use LaTeX \ref with indicator: ◆ Chapter (§num)
        ref_text = ref_text .. indicator .. " **" .. display_name .. "** (§\\ref{" .. ref.target_section .. "}): "
      else
        -- For HTML, with indicator
        ref_text = ref_text .. indicator .. " **" .. display_name .. "**: "
      end
    else
      -- No section reference, with indicator
      ref_text = ref_text .. indicator .. " **" .. display_name .. "**: "
    end
  end

  -- Add explanation without truncation
  if ref.explanation and ref.explanation ~= "" then
    local clean_explanation = ref.explanation

    -- Remove any markdown formatting
    clean_explanation = string.gsub(clean_explanation, "%*%*", "")
    clean_explanation = string.gsub(clean_explanation, "%*", "")

    ref_text = ref_text .. clean_explanation
  end

  return ref_text
end

-- Collect all chapter-level connections
local function collect_chapter_connections(xrefs_data)
  local chapter_refs = {}

  if not xrefs_data or not xrefs_data.cross_references then
    return chapter_refs
  end

  -- Collect all refs marked for chapter_start or high priority
  for section_id, refs in pairs(xrefs_data.cross_references) do
    for _, ref in ipairs(refs) do
      if ref.placement == "chapter_start" or
         (ref.priority and ref.priority == 1) or
         (ref.strength and ref.strength > 0.3) then
        table.insert(chapter_refs, ref)
      end
    end
  end

  -- Sort by diversity score, connection type, and strength
  local type_order = {prerequisite = 1, foundation = 2, complements = 3, extends = 4}
  table.sort(chapter_refs, function(a, b)
    -- First, consider diversity scores
    local a_diversity = get_diversity_score(a)
    local b_diversity = get_diversity_score(b)

    -- Calculate combined scores (diversity * quality * strength)
    local a_score = a_diversity * (a.quality or 1) * (a.strength or 0.5)
    local b_score = b_diversity * (b.quality or 1) * (b.strength or 0.5)

    -- Strong preference for diversity (if one is much more diverse)
    if math.abs(a_diversity - b_diversity) > 0.3 then
      return a_diversity > b_diversity
    end

    -- Then by connection type priority
    local a_order = type_order[a.connection_type] or 5
    local b_order = type_order[b.connection_type] or 5
    if a_order ~= b_order then
      return a_order < b_order
    end

    -- Finally by overall score
    return a_score > b_score
  end)

  -- Limit to MAX_CHAPTER_REFS
  if #chapter_refs > MAX_CHAPTER_REFS then
    local limited = {}
    for i = 1, MAX_CHAPTER_REFS do
      limited[i] = chapter_refs[i]
    end
    chapter_refs = limited
  end

  return chapter_refs
end

-- Create a chapter-level connection box
local function create_chapter_connection_box(chapter_refs)
  if not chapter_refs or #chapter_refs == 0 then
    return nil
  end

  -- Group by connection type
  local grouped = {
    prerequisite = {},
    foundation = {},
    extends = {},
    complements = {}
  }

  for _, ref in ipairs(chapter_refs) do
    local conn_type = ref.connection_type or "complements"
    if grouped[conn_type] then
      table.insert(grouped[conn_type], ref)
    else
      table.insert(grouped.complements, ref)
    end
  end

  -- Track shown chapters for diversity
  for _, ref in ipairs(chapter_refs) do
    if ref.target_chapter then
      shown_chapters[ref.target_chapter] = (shown_chapters[ref.target_chapter] or 0) + 1
      shown_in_section[ref.target_chapter] = true
    end
  end

  -- Build content blocks
  local content_blocks = {}

  -- Add a header
  table.insert(content_blocks, pandoc.Para({
    pandoc.Strong({pandoc.Str("Related Topics")})
  }))

  -- Add prerequisites
  if #grouped.prerequisite > 0 then
    -- Prerequisites section (visual indicators only)
    for _, ref in ipairs(grouped.prerequisite) do
      local ref_text = format_xref_entry(ref)
      local ref_doc = pandoc.read(ref_text, "markdown")
      if ref_doc.blocks[1] then
        table.insert(content_blocks, ref_doc.blocks[1])
      end
    end
  end

  -- Add foundations
  if #grouped.foundation > 0 then
    -- Foundations section (visual indicators only)
    for _, ref in ipairs(grouped.foundation) do
      local ref_text = format_xref_entry(ref)
      local ref_doc = pandoc.read(ref_text, "markdown")
      if ref_doc.blocks[1] then
        table.insert(content_blocks, ref_doc.blocks[1])
      end
    end
  end

  -- Add extensions
  if #grouped.extends > 0 then
    -- Extensions section (visual indicators only)
    for _, ref in ipairs(grouped.extends) do
      local ref_text = format_xref_entry(ref)
      local ref_doc = pandoc.read(ref_text, "markdown")
      if ref_doc.blocks[1] then
        table.insert(content_blocks, ref_doc.blocks[1])
      end
    end
  end

  -- Add complements
  if #grouped.complements > 0 then
    -- Complements section (visual indicators only)
    for _, ref in ipairs(grouped.complements) do
      local ref_text = format_xref_entry(ref)
      local ref_doc = pandoc.read(ref_text, "markdown")
      if ref_doc.blocks[1] then
        table.insert(content_blocks, ref_doc.blocks[1])
      end
    end
  end

  -- Create a div with callout-chapter-connection class and text-left style
  local callout_div = pandoc.Div(
    content_blocks,
    pandoc.Attr("", {"callout", "callout-chapter-connection"}, {["style"] = "text-align: left;"})
  )

  return callout_div
end

-- Filter section references based on placement and priority
local function should_show_section_refs(refs, experiment_mode)
  if experiment_mode == "chapter_only" then
    return false
  end

  if experiment_mode == "section_only" then
    return true
  end

  -- In hybrid or priority_based mode, check criteria
  for _, ref in ipairs(refs) do
    -- Show if explicitly marked for section placement
    if ref.placement == "section_start" or ref.placement == "section_end" then
      return true
    end
    -- Show if high priority and strong connection
    if ref.priority and ref.priority <= PRIORITY_THRESHOLD and
       ref.strength and ref.strength > STRENGTH_THRESHOLD then
      return true
    end
  end

  return false
end

-- Create a section connection box
local function create_section_connection_box(refs)
  if not refs or #refs == 0 then
    return nil
  end

  -- Filter and sort references
  local filtered_refs = {}
  for _, ref in ipairs(refs) do
    if (ref.priority and ref.priority <= PRIORITY_THRESHOLD) or
       (ref.strength and ref.strength > STRENGTH_THRESHOLD) then
      table.insert(filtered_refs, ref)
    end
  end

  -- Limit to MAX_SECTION_REFS with diversity consideration
  if #filtered_refs > MAX_SECTION_REFS then
    table.sort(filtered_refs, function(a, b)
      -- Include diversity in scoring
      local a_diversity = get_diversity_score(a)
      local b_diversity = get_diversity_score(b)
      local a_score = (a.strength or 0) * (a.quality or 1) * a_diversity
      local b_score = (b.strength or 0) * (b.quality or 1) * b_diversity
      return a_score > b_score
    end)
    local top_refs = {}
    for i = 1, MAX_SECTION_REFS do
      top_refs[i] = filtered_refs[i]
    end
    filtered_refs = top_refs
  end

  if #filtered_refs == 0 then
    return nil
  end

  -- Track shown chapters for diversity
  for _, ref in ipairs(filtered_refs) do
    if ref.target_chapter then
      shown_chapters[ref.target_chapter] = (shown_chapters[ref.target_chapter] or 0) + 1
      shown_in_section[ref.target_chapter] = true
    end
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

  -- Create a div with callout-chapter-connection class and text-left style
  local callout_div = pandoc.Div(
    content_blocks,
    pandoc.Attr("", {"callout", "callout-chapter-connection"}, {["style"] = "text-align: left;"})
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
        log_warning("Cross-references disabled or not configured")
        return doc
      end

      log_info("🚀 Initializing Cross-Reference Injection Filter")
      log_info("Placement mode: " .. PLACEMENT_MODE)

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

      -- Collect chapter-level connections
      local chapter_refs = collect_chapter_connections(xrefs_data)

      -- Process document blocks
      local new_blocks = {}
      local chapter_header_found = false

      for _, block in ipairs(doc.blocks) do
        -- Look for the main chapter header to insert chapter connections
        if not chapter_header_found and block.t == "Header" and block.level == 1 then
          chapter_header_found = true

          -- Insert the header first
          table.insert(new_blocks, block)

          -- Insert chapter-level connections if not in section_only mode
          if PLACEMENT_MODE ~= "section_only" and #chapter_refs > 0 then
            local chapter_box = create_chapter_connection_box(chapter_refs)
            if chapter_box then
              table.insert(new_blocks, chapter_box)
              stats.chapter_boxes = 1
              stats.total_refs = stats.total_refs + #chapter_refs
              log_info("Injected chapter-level connections: " .. #chapter_refs .. " references")
            end
          end
        -- Check for section headers with cross-references
        elseif block.t == "Header" and block.identifier and block.identifier ~= "" then
          -- Reset section tracking for new section
          shown_in_section = {}
          section_counter = section_counter + 1

          local section_refs = xrefs_data.cross_references[block.identifier]

          if section_refs and should_show_section_refs(section_refs, PLACEMENT_MODE) then
            -- For PDF: ensure minimum lines before section
            if is_pdf then
              table.insert(new_blocks, needspace_block())
            end

            -- Insert the header
            table.insert(new_blocks, block)

            -- Create and insert the connection box
            local connection_box = create_section_connection_box(section_refs)
            if connection_box then
              table.insert(new_blocks, connection_box)
              stats.section_boxes = stats.section_boxes + 1
              log_info("Injected section connections for: " .. block.identifier)
            end
          else
            -- No cross-references for this section or filtered out
            if section_refs then
              stats.filtered_refs = stats.filtered_refs + #section_refs
            end
            table.insert(new_blocks, block)
          end
        else
          -- Not a header, just pass through
          table.insert(new_blocks, block)
        end
      end

      -- Summary
      log_success("📊 SUMMARY:")
      log_success("  Mode: " .. PLACEMENT_MODE)
      log_success("  Chapter boxes: " .. stats.chapter_boxes)
      log_success("  Section boxes: " .. stats.section_boxes)
      log_success("  Filtered references: " .. stats.filtered_refs)
      log_success("  Total references: " .. stats.total_refs)

      return pandoc.Pandoc(new_blocks, doc.meta)
    end
  }
}
