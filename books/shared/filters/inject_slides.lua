-- inject_slides.lua
--
-- Adds one quiet line under the chapter title linking to the chapter's
-- teaching materials:
--   "For this chapter: Lecture slides · Study sheet (PDF)"
--   "For this chapter: Lecture slides (Part 1, Part 2) · Study sheet (PDF)"
-- Either item is left out when the chapter does not have it.
--
-- The links come from two JSON manifests written at pre-render time (see
-- project.pre-render in _quarto-html-volN.yml):
--   _build/slides/lectures.json      by slides/tools/lectures.py
--     { "vol1": { "04_data_engineering": { "slides_url": ..., "lectures":
--         [ { "label": "Part 1", "url": ... }, ... ] } } }
--   _build/studysheets/manifest.json by books/shared/studysheet/build.py
--     { "vol1": { "04_data_engineering": { "url": ... } } }
-- The deck folders and the *_studysheet.tex files are the single sources of
-- truth; this filter never names a chapter or a URL itself.
--
-- HTML-only. shared/html/filters.yml lists it and pdf/ and epub/ do not;
-- as with inject_quizzes.lua, the format check runs inside the Pandoc
-- handler, not at load time. With no manifest, or a page that is not a
-- chapter with lectures, the filter returns the document unchanged.

local json = require("pandoc.json")

local SLIDES_MANIFEST = "_build/slides/lectures.json"
local SHEETS_MANIFEST = "_build/studysheets/manifest.json"

-- volN/<chapter>/<chapter>.qmd -> "volN", "<chapter>"
local function current_chapter()
  local input = (quarto and quarto.doc and quarto.doc.input_file)
    or (PANDOC_STATE and PANDOC_STATE.input_files and PANDOC_STATE.input_files[1])
    or ""
  input = input:gsub("\\", "/")
  local vol, chapter, stem = input:match("(vol%d+)/([^/]+)/([^/]+)%.qmd$")
  if vol and stem == chapter then
    return vol, chapter
  end
  return nil, nil
end

local function load_manifest(name)
  local dir = (quarto and quarto.project and quarto.project.directory) or "."
  local f = io.open(dir .. "/" .. name, "r")
  if not f then return nil end
  local content = f:read("*all")
  f:close()
  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    io.stderr:write("[Slides Filter] Could not parse " .. name .. "; its links are left out\n")
    return nil
  end
  return data
end

local function entry_for(name, vol, chapter)
  local manifest = load_manifest(name)
  return manifest and manifest[vol] and manifest[vol][chapter] or nil
end

-- Append a word (or punctuation) and the space after it.
local function sep(inlines, text)
  table.insert(inlines, pandoc.Str(text))
  table.insert(inlines, pandoc.Space())
end

local function materials_line(slides, sheet)
  local inlines = { pandoc.Str("For"), pandoc.Space(), pandoc.Str("this"), pandoc.Space(),
                    pandoc.Str("chapter:"), pandoc.Space() }
  if slides then
    local lectures = slides.lectures or {}
    if #lectures > 1 then
      -- "Lecture slides (Part 1, Part 2)"
      sep(inlines, "Lecture")
      sep(inlines, "slides")
      table.insert(inlines, pandoc.Str("("))
      for i, lec in ipairs(lectures) do
        if i > 1 then sep(inlines, ",") end
        table.insert(inlines, pandoc.Link(lec.label, lec.url))
      end
      table.insert(inlines, pandoc.Str(")"))
    else
      table.insert(inlines, pandoc.Link("Lecture slides", slides.slides_url))
    end
  end
  if sheet then
    if slides then
      table.insert(inlines, pandoc.Space())
      sep(inlines, "·")
    end
    table.insert(inlines, pandoc.Link("Study sheet (PDF)", sheet.url))
  end
  return pandoc.Div({ pandoc.Para(inlines) },
    pandoc.Attr("", { "chapter-slides", "text-muted", "small" }))
end

local function inject_slides(doc)
  if not quarto.doc.is_format("html:js") then return nil end
  local vol, chapter = current_chapter()
  if not vol then return nil end
  local slides = entry_for(SLIDES_MANIFEST, vol, chapter)
  local sheet = entry_for(SHEETS_MANIFEST, vol, chapter)
  if not slides and not sheet then return nil end

  -- Directly under the chapter's level-1 header. In the website build that
  -- header sits inside a wrapper Div, not at the top level of the body, so
  -- the search looks one level into Divs. With no header at all, the line
  -- goes first in the body, right below Quarto's title block. (Until
  -- 2026-11-20 the filter searched the top level only, never found the
  -- header, and the slide links never appeared on the website.)
  local function after_h1(blocks)
    for i, block in ipairs(blocks) do
      if block.t == "Header" and block.level == 1 then
        blocks:insert(i + 1, materials_line(slides, sheet))
        return true
      end
      if block.t == "Div" and after_h1(block.content) then
        return true
      end
    end
    return false
  end
  if not after_h1(doc.blocks) then
    doc.blocks:insert(1, materials_line(slides, sheet))
  end
  io.stderr:write("[Slides Filter] Linked " .. (slides and "slides" or "")
    .. (slides and sheet and " + " or "") .. (sheet and "study sheet" or "")
    .. " for " .. vol .. "/" .. chapter .. "\n")
  return doc
end

return {
  { Pandoc = inject_slides }
}
