-- Auto-Glossary Filter for ML Systems Book
-- Automatically detects and marks glossary terms in text
-- Supports multiple output formats (HTML, PDF, EPUB)

-- Global glossary data
local glossary = {}
local glossary_loaded = false

-- Configuration
local config = {
  glossary_path = "data/master_glossary.yml",
  mark_first_only = true,  -- Only mark first occurrence per chapter
  case_sensitive = false,
  formats = {
    html = "tooltip",     -- tooltip, link, footnote
    pdf = "margin",       -- margin, footnote, link
    epub = "link"         -- link, inline, footnote
  }
}

-- Track which terms have been marked in this document
local marked_terms = {}

-- Helper function to load glossary from YAML
local function load_glossary()
  if glossary_loaded then return end

  local file = io.open(config.glossary_path, "r")
  if not file then
    io.stderr:write("Warning: Cannot open glossary file: " .. config.glossary_path .. "\n")
    glossary_loaded = true
    return
  end

  local content = file:read("*a")
  file:close()

  -- Parse YAML content (simplified - would need proper YAML parser)
  -- For now, using a simple pattern match approach
  -- In production, use a proper YAML parser

  -- This is a placeholder - implement actual YAML parsing
  glossary = {
    ["artificial intelligence"] = "The systematic pursuit of understanding and replicating intelligent behavior.",
    ["machine learning"] = "The methodological approach to implementing intelligent systems through computational techniques.",
    ["quantization"] = "The process of mapping continuous values to discrete, lower-precision representations.",
    ["pruning"] = "The removal of unnecessary parameters from neural networks to reduce model size.",
    -- Add more terms as needed
  }

  glossary_loaded = true
end

-- Helper function to check if term should be marked
local function should_mark_term(term)
  if not config.mark_first_only then
    return true
  end

  if marked_terms[term] then
    return false
  end

  marked_terms[term] = true
  return true
end

-- Helper function to create appropriate markup based on format
local function create_markup(term, definition, format)
  if format == "html" then
    -- HTML with tooltip
    return pandoc.RawInline("html",
      '<span class="glossary-term" data-bs-toggle="tooltip" title="' ..
      definition .. '">' .. term .. '</span>')
  elseif format == "latex" or format == "pdf" then
    -- LaTeX with margin note
    return pandoc.RawInline("latex",
      term .. '\\marginnote{\\footnotesize\\textit{' .. definition .. '}}')
  elseif format == "epub" then
    -- EPUB with link to glossary
    return pandoc.Link(term, "#glossary-" .. term:gsub(" ", "-"))
  else
    -- Fallback: just return the term
    return pandoc.Str(term)
  end
end

-- Function to scan text for glossary terms
local function scan_for_terms(elem)
  if elem.t ~= "Str" then return nil end

  load_glossary()

  local text = elem.text
  local format = FORMAT:match("html") and "html" or
                 FORMAT:match("latex") and "latex" or
                 FORMAT:match("pdf") and "pdf" or
                 FORMAT:match("epub") and "epub" or "other"

  -- Check each glossary term
  for term, definition in pairs(glossary) do
    local pattern = config.case_sensitive and term or term:lower()
    local search_text = config.case_sensitive and text or text:lower()

    if search_text:match(pattern) and should_mark_term(term) then
      return create_markup(term, definition, format)
    end
  end

  return nil
end

-- Main filter
return {
  {
    -- First pass: Reset marked terms for each chapter/document
    Meta = function(meta)
      marked_terms = {}
      return nil
    end
  },
  {
    -- Second pass: Scan and mark glossary terms
    Str = scan_for_terms,

    -- Skip code blocks and other elements where we don't want glossary marking
    Code = function(elem) return nil end,
    CodeBlock = function(elem) return nil end,
  }
}
