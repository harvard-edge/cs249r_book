-- inject_slides.lua
--
-- Adds one quiet line under the chapter title linking to the chapter's
-- lecture slides: "Lecture slides for this chapter: Part 1 · Part 2", or a
-- single "Lecture slides" link when the chapter has one lecture.
--
-- The links come from a JSON manifest that slides/tools/lectures.py writes
-- at pre-render time (see project.pre-render in _quarto-html-volN.yml):
--   <project>/_build/slides/lectures.json
--   { "vol1": { "04_data_engineering": { "slides_url": ..., "lectures":
--       [ { "label": "Part 1", "url": ... }, ... ] } } }
-- The deck folders and their lecture_map.yml files are the single source
-- of truth; this filter never names a chapter or a URL itself.
--
-- HTML-only. shared/html/filters.yml lists it and pdf/ and epub/ do not;
-- as with inject_quizzes.lua, the format check runs inside the Pandoc
-- handler, not at load time. With no manifest, or a page that is not a
-- chapter with lectures, the filter returns the document unchanged.

local json = require("pandoc.json")

local MANIFEST = "_build/slides/lectures.json"

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

local function load_manifest()
  local dir = (quarto and quarto.project and quarto.project.directory) or "."
  local f = io.open(dir .. "/" .. MANIFEST, "r")
  if not f then return nil end
  local content = f:read("*all")
  f:close()
  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    io.stderr:write("[Slides Filter] Could not parse " .. MANIFEST .. "; no slide links added\n")
    return nil
  end
  return data
end

local function slides_line(entry)
  local inlines = {}
  local lectures = entry.lectures or {}
  if #lectures > 1 then
    inlines = { pandoc.Str("Lecture slides for this chapter:"), pandoc.Space() }
    for i, lec in ipairs(lectures) do
      if i > 1 then
        table.insert(inlines, pandoc.Space())
        table.insert(inlines, pandoc.Str("·"))
        table.insert(inlines, pandoc.Space())
      end
      table.insert(inlines, pandoc.Link(lec.label, lec.url))
    end
  else
    inlines = {
      pandoc.Link("Lecture slides", entry.slides_url),
      pandoc.Space(), pandoc.Str("for this chapter"),
    }
  end
  return pandoc.Div({ pandoc.Para(inlines) },
    pandoc.Attr("", { "chapter-slides", "text-muted", "small" }))
end

local function inject_slides(doc)
  if not quarto.doc.is_format("html:js") then return nil end
  local vol, chapter = current_chapter()
  if not vol then return nil end
  local manifest = load_manifest()
  local entry = manifest and manifest[vol] and manifest[vol][chapter]
  if not entry then return nil end

  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      doc.blocks:insert(i + 1, slides_line(entry))
      io.stderr:write("[Slides Filter] Linked lecture slides for " .. vol .. "/" .. chapter .. "\n")
      return doc
    end
  end
  return nil
end

return {
  { Pandoc = inject_slides }
}
