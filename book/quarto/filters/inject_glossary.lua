-- Auto-Glossary Filter for ML Systems Book
-- Automatically detects and marks glossary terms in text
-- Supports multiple output formats (HTML, PDF, EPUB)
-- Reads structured JSON glossary files directly

-- JSON parsing utility
local json = require("dkjson") or require("json")

-- Global glossary data
local glossary = {}
local glossary_loaded = false

-- Configuration
local config = {
  -- Try chapter-specific glossary first, then master glossary
  glossary_paths = {
    "../../../data/master_glossary.json",     -- Master glossary
    "./%s_glossary.json"                      -- Chapter-specific (filled in dynamically)
  },
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

-- Helper function to get current chapter name from file path
local function get_chapter_name()
  local info = pandoc.system.get_working_directory()
  if info then
    -- Extract chapter name from path like .../core/introduction/
    local chapter = info:match("/([^/]+)/?$")
    if chapter then
      return chapter
    end
  end
  return nil
end

-- Helper function to read file
local function read_file(path)
  local file = io.open(path, "r")
  if not file then
    return nil
  end
  local content = file:read("*a")
  file:close()
  return content
end

-- Load glossary from JSON file
local function load_json_glossary(path)
  local content = read_file(path)
  if not content then
    return nil
  end

  -- Parse JSON
  local data, pos, err = json.decode(content)
  if err then
    io.stderr:write("Error parsing JSON glossary: " .. err .. "\n")
    return nil
  end

  -- Extract terms from structured format
  local terms = {}
  if data and data.terms then
    for _, term_entry in ipairs(data.terms) do
      if term_entry.term and term_entry.definition then
        -- Store with lowercase key for case-insensitive matching
        local key = config.case_sensitive and term_entry.term or term_entry.term:lower()
        terms[key] = {
          term = term_entry.term,
          definition = term_entry.definition,
          chapter = term_entry.chapter_source,
          aliases = term_entry.aliases or {},
          see_also = term_entry.see_also or {}
        }
      end
    end
  end

  return terms, data.metadata
end

-- Load glossary from JSON files
local function load_glossary()
  if glossary_loaded then return end

  local chapter = get_chapter_name()
  local loaded_from = nil

  -- Try chapter-specific glossary first
  if chapter then
    local chapter_path = string.format("./%s_glossary.json", chapter)
    local terms, metadata = load_json_glossary(chapter_path)
    if terms then
      glossary = terms
      loaded_from = chapter_path
      io.stderr:write(string.format("Loaded chapter glossary: %s (%d terms)\n",
                                    chapter_path, metadata and metadata.total_terms or 0))
    end
  end

  -- If no chapter glossary, try master glossary
  if not loaded_from then
    local master_path = config.glossary_paths[1]
    local terms, metadata = load_json_glossary(master_path)
    if terms then
      glossary = terms
      loaded_from = master_path
      io.stderr:write(string.format("Loaded master glossary: %s (%d terms)\n",
                                    master_path, metadata and metadata.total_terms or 0))
    end
  end

  -- Silently continue if no glossary file is found
  -- This is expected behavior for selective chapter builds

  glossary_loaded = true
end

-- Create glossary markup based on output format
local function create_glossary_markup(term_data, text, format)
  local term = term_data.term
  local definition = term_data.definition

  if not format then
    format = quarto.doc.is_format("html") and config.formats.html or
             quarto.doc.is_format("pdf") and config.formats.pdf or
             config.formats.epub
  end

  if format == "tooltip" then
    -- HTML tooltip using data-definition attribute (no title to avoid browser tooltip)
    return pandoc.Span(
      text,
      {
        class = "glossary-term",
        ["data-definition"] = definition,
        ["data-term"] = term
      }
    )
  elseif format == "margin" then
    -- PDF margin note
    return {
      pandoc.Span(text, {class = "glossary-term"}),
      pandoc.RawInline("latex", string.format("\\marginnote{\\textbf{%s}: %s}",
                                              term, definition))
    }
  elseif format == "footnote" then
    -- Footnote for any format
    local note = pandoc.Note({pandoc.Para(pandoc.Str(definition))})
    return {
      pandoc.Span(text, {class = "glossary-term"}),
      note
    }
  elseif format == "link" then
    -- Link to glossary section
    return pandoc.Link(
      text,
      "#glossary-" .. term:gsub("%s", "-"):lower(),
      term,
      {class = "glossary-link"}
    )
  else
    -- Inline definition (fallback)
    return {
      pandoc.Span(text, {class = "glossary-term"}),
      pandoc.Str(" ("),
      pandoc.Emph(definition),
      pandoc.Str(")")
    }
  end
end

-- Process text to find and mark glossary terms
local function process_text(elem)
  if elem.t ~= "Str" then
    return elem
  end

  local text = elem.text
  local lower_text = config.case_sensitive and text or text:lower()

  -- Check if this text contains any glossary terms
  for key, term_data in pairs(glossary) do
    -- Check if term has already been marked (if marking first only)
    if not (config.mark_first_only and marked_terms[key]) then
      -- Simple word boundary check
      local pattern = "%f[%a]" .. key .. "%f[%A]"
      if lower_text:match(pattern) then
        -- Mark this term as used
        marked_terms[key] = true

        -- Create marked up version
        return create_glossary_markup(term_data, elem)
      end
    end
  end

  return elem
end

-- Main filter function
function Pandoc(doc)
  -- Load glossary on first run
  load_glossary()

  -- Process all blocks and inlines
  doc = doc:walk({
    Str = process_text
  })

  -- Add CSS for HTML output
  if quarto.doc.is_format("html") then
    local css = [[
<style>
.glossary-term {
  border-bottom: 1px dotted #666;
  cursor: help;
  position: relative;
}
.glossary-term {
  position: relative;
}
.glossary-term:hover::after {
  content: attr(data-definition);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: #333;
  color: white;
  padding: 6px 16px;
  border-radius: 3px;
  white-space: normal;
  width: 280px;
  max-width: 90vw;
  z-index: 1000;
  font-size: 0.5em;
  line-height: 1.3;
  box-shadow: 0 3px 10px rgba(0,0,0,0.3);
  border: 1px solid #555;
}
/* Adjust positioning for tooltips near screen edges */
.glossary-term:hover::after {
  left: clamp(140px, 50%, calc(100vw - 140px));
  transform: translateX(-50%);
}
</style>
]]
    table.insert(doc.blocks, 1, pandoc.RawBlock("html", css))
  end

  return doc
end

-- Return filter
return {
  {Pandoc = Pandoc}
}
