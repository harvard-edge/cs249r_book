-- studysheet_list.lua
--
-- Fills the volume's Study Sheets page (volN/backmatter/studysheets.qmd): a
-- download link for the whole volume, then every chapter's sheet in book order,
-- grouped under the book's parts. The page holds only an empty marker,
--
--   ::: {#studysheet-list}
--   :::
--
-- and this filter replaces its contents from the manifest that
-- books/shared/studysheet/build.py --manifest writes at pre-render time
-- (_build/studysheets/manifest.json, key "_volume"). The same manifest feeds
-- the line under each chapter title (inject_slides.lua) and the slides site,
-- so the three can never list different files.
--
-- HTML-only, like inject_slides.lua. Without a manifest the marker is left
-- with a one-line note instead of an empty page.

local json = require("pandoc.json")

local MANIFEST = "_build/studysheets/manifest.json"

local function current_volume()
  local input = (quarto and quarto.doc and quarto.doc.input_file)
    or (PANDOC_STATE and PANDOC_STATE.input_files and PANDOC_STATE.input_files[1])
    or ""
  return input:gsub("\\", "/"):match("(vol%d+)/backmatter/")
end

local function load_volume(vol)
  local dir = (quarto and quarto.project and quarto.project.directory) or "."
  local f = io.open(dir .. "/" .. MANIFEST, "r")
  if not f then return nil end
  local ok, data = pcall(json.decode, f:read("*all"))
  f:close()
  if not ok or type(data) ~= "table" or not data[vol] then return nil end
  return data[vol]["_volume"]
end

local function sheet_item(sheet)
  return { pandoc.Plain({
    pandoc.Link(sheet.title, sheet.url),
    pandoc.Space(),
    pandoc.Span({ pandoc.Str("(PDF)") }, pandoc.Attr("", { "text-muted" })),
  }) }
end

local function list_blocks(volume)
  local blocks = pandoc.List({
    pandoc.Para({
      pandoc.Strong({ pandoc.Link(string.format("Download all %d study sheets", math.floor(volume.count)),
        volume.url) }),
      pandoc.Space(),
      pandoc.Span({ pandoc.Str(string.format("(one PDF, %d pages, in book order)", math.floor(volume.pages))) },
        pandoc.Attr("", { "text-muted" })),
    }),
  })
  -- Consecutive sheets that share a part go under one heading; the
  -- introduction and conclusion stand on their own, as in the sidebar.
  local current, items = nil, nil
  local function flush()
    if items and #items > 0 then blocks:insert(pandoc.BulletList(items)) end
    items = pandoc.List()
  end
  for _, sheet in ipairs(volume.sheets) do
    local part = sheet.part or ""
    if part ~= current then
      flush()
      if part ~= "" then
        blocks:insert(pandoc.Header(3, part, pandoc.Attr("", { "unnumbered" })))
      end
      current = part
    end
    items:insert(sheet_item(sheet))
  end
  flush()
  return blocks
end

local function fill(div)
  if div.identifier ~= "studysheet-list" then return nil end
  if not quarto.doc.is_format("html:js") then return {} end
  local vol = current_volume()
  local volume = vol and load_volume(vol)
  if not volume then
    io.stderr:write("[Study Sheets] No manifest entry; page left with a note\n")
    div.content = { pandoc.Para({ pandoc.Str("The study sheets are not available in this build.") }) }
    return div
  end
  div.content = list_blocks(volume)
  io.stderr:write("[Study Sheets] Listed " .. math.floor(volume.count) .. " sheets for " .. vol .. "\n")
  return div
end

return {
  { Div = fill }
}
