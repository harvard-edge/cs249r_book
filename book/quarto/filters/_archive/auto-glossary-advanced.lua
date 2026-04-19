-- Advanced Auto-Glossary Filter for ML Systems Book
-- Automatically detects and marks glossary terms across multiple formats
-- Author: ML Systems Book Project

local glossary_data = {}
local marked_terms = {}
local glossary_loaded = false

-- Configuration
local config = {
  glossary_file = "data/master_glossary.yml",
  mark_first_occurrence = true,
  skip_in_headings = true,
  skip_in_captions = true,
  min_word_length = 3,  -- Don't mark very short terms

  -- Format-specific rendering
  render_mode = {
    html = "tooltip",      -- Options: tooltip, popup, link
    pdf = "sidenote",      -- Options: sidenote, footnote, link, none
    latex = "sidenote",    -- Options: sidenote, footnote, link, none
    epub = "link"          -- Options: link, inline, none
  }
}

-- Debug logging
local function log(message)
  if quarto and quarto.log then
    quarto.log.output("[Auto-Glossary] " .. message)
  end
end

-- Load and parse YAML glossary
local function load_glossary()
  if glossary_loaded then return true end

  local file = io.open(config.glossary_file, "r")
  if not file then
    log("Warning: Cannot open glossary file: " .. config.glossary_file)
    glossary_loaded = true
    return false
  end

  local yaml_content = file:read("*a")
  file:close()

  -- Parse YAML using Pandoc's built-in YAML parser
  local doc = pandoc.read("---\n" .. yaml_content .. "\n---\n", "markdown")

  if doc.meta and doc.meta.glossary then
    -- Parse the hierarchical glossary structure
    for category, terms in pairs(doc.meta.glossary) do
      if type(terms) == "table" then
        for term_key, term_data in pairs(terms) do
          if type(term_data) == "table" then
            local term = pandoc.utils.stringify(term_data.term or term_key)
            local definition = pandoc.utils.stringify(term_data.definition or "")

            -- Store both the term and its lowercase version for matching
            glossary_data[term:lower()] = {
              display = term,
              definition = definition,
              category = pandoc.utils.stringify(category),
              see_also = term_data.see_also or {}
            }

            -- Also store acronyms and variants
            if term_data.acronym then
              local acronym = pandoc.utils.stringify(term_data.acronym)
              glossary_data[acronym:lower()] = glossary_data[term:lower()]
            end
          end
        end
      end
    end
  end

  glossary_loaded = true
  log("Loaded " .. #glossary_data .. " glossary terms")
  return true
end

-- Check if we should process this element
local function should_process_element(elem)
  -- Skip if in code
  if elem.classes and (elem.classes:includes("code") or
                       elem.classes:includes("sourceCode")) then
    return false
  end

  -- Skip if in heading (optional)
  if config.skip_in_headings and elem.t == "Header" then
    return false
  end

  return true
end

-- Create format-specific output
local function render_glossary_term(term_text, term_data, format)
  local mode = config.render_mode[format] or "none"

  if mode == "none" then
    return pandoc.Str(term_text)
  end

  if format == "html" then
    if mode == "tooltip" then
      -- Bootstrap tooltip style
      local html = string.format(
        '<span class="glossary-term" data-bs-toggle="tooltip" ' ..
        'data-bs-placement="top" title="%s">%s</span>',
        term_data.definition:gsub('"', '&quot;'),
        term_text
      )
      return pandoc.RawInline("html", html)

    elseif mode == "popup" then
      -- Clickable popup style
      local html = string.format(
        '<button class="glossary-term glossary-popup" ' ..
        'data-glossary-def="%s">%s</button>',
        term_data.definition:gsub('"', '&quot;'),
        term_text
      )
      return pandoc.RawInline("html", html)

    elseif mode == "link" then
      -- Link to glossary
      return pandoc.Link(
        term_text,
        "#glossary-" .. term_data.display:lower():gsub("%s+", "-")
      )
    end

  elseif format == "latex" or format == "pdf" then
    if mode == "sidenote" then
      -- Use existing sidenote infrastructure
      local latex = string.format(
        '%s\\sidenote{\\textit{%s}: %s}',
        term_text,
        term_data.display,
        term_data.definition
      )
      return pandoc.RawInline("latex", latex)

    elseif mode == "footnote" then
      -- Footnote with definition
      local latex = string.format(
        '%s\\footnote{%s: %s}',
        term_text,
        term_data.display,
        term_data.definition
      )
      return pandoc.RawInline("latex", latex)

    elseif mode == "link" then
      -- Hyperlink to glossary section
      local latex = string.format(
        '\\hyperref[glossary:%s]{%s}',
        term_data.display:lower():gsub("%s+", "-"),
        term_text
      )
      return pandoc.RawInline("latex", latex)
    end

  elseif format == "epub" then
    if mode == "link" then
      return pandoc.Link(
        term_text,
        "glossary.xhtml#" .. term_data.display:lower():gsub("%s+", "-")
      )
    elseif mode == "inline" then
      -- Inline definition in parentheses
      return pandoc.Span(
        {pandoc.Str(term_text),
         pandoc.Str(" ("),
         pandoc.Emph(term_data.definition),
         pandoc.Str(")")}
      )
    end
  end

  -- Fallback
  return pandoc.Str(term_text)
end

-- Scan text for glossary terms
local function process_text(elem)
  if elem.t ~= "Str" then return nil end

  local text = elem.text
  local text_lower = text:lower()

  -- Check if any glossary term matches
  for term_key, term_data in pairs(glossary_data) do
    -- Simple word boundary check (can be improved)
    local pattern = "%f[%a]" .. term_key:gsub("%-", "%%-") .. "%f[%A]"

    if text_lower:match(pattern) then
      -- Check if we should mark this occurrence
      if config.mark_first_occurrence then
        if marked_terms[term_key] then
          return nil  -- Already marked
        end
        marked_terms[term_key] = true
      end

      -- Determine output format
      local format = FORMAT:match("html") and "html" or
                    FORMAT:match("latex") and "latex" or
                    FORMAT:match("pdf") and "pdf" or
                    FORMAT:match("epub") and "epub" or "other"

      -- Find the actual term in the original text (preserve case)
      local start_pos, end_pos = text_lower:find(pattern)
      if start_pos then
        local actual_term = text:sub(start_pos, end_pos)
        return render_glossary_term(actual_term, term_data, format)
      end
    end
  end

  return nil
end

-- Main filter return
return {
  {
    -- First pass: Initialize
    Pandoc = function(doc)
      if not load_glossary() then
        log("Failed to load glossary, filter disabled")
        return nil
      end
      marked_terms = {}  -- Reset for each document
      return nil
    end
  },
  {
    -- Second pass: Process text
    Str = process_text,

    -- Don't process these elements
    Code = function() return nil end,
    CodeBlock = function() return nil end,
    RawBlock = function() return nil end,
    RawInline = function() return nil end,
  }
}
