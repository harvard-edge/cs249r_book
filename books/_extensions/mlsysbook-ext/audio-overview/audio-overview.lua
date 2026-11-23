-- Audio overview shortcode for MLSysBook.
-- Usage: {{< audio-overview vol1 >}}                         (a volume landing page)
--        {{< audio-overview vol1-ch01 local="../assets/media" >}} (a page one level down)
--
-- The audio files are not in git. They are GitHub Release assets (see
-- docs/media-hosting.md). This shortcode reads books/shared/media/audio-overviews.json
-- and renders a Listen block for the given id. An id that is not in the manifest
-- renders nothing, so a page can carry the shortcode before its audio is approved.

local MANIFEST = "shared/media/audio-overviews.json"

local function warn(msg)
  io.stderr:write("⚠️  [audio-overview] " .. msg .. "\n")
end

local function read_manifest()
  local root = quarto.project.directory or "."
  local path = pandoc.path.join({ root, MANIFEST })
  local f = io.open(path, "r")
  if not f then
    warn("manifest not found at " .. path)
    return nil
  end
  local text = f:read("*a")
  f:close()
  local ok, data = pcall(quarto.json.decode, text)
  if not ok or type(data) ~= "table" then
    warn("manifest is not valid JSON: " .. path)
    return nil
  end
  return data
end

local function esc(s)
  s = tostring(s or "")
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

return {
  ["audio-overview"] = function(args, kwargs, meta)
    if not quarto.doc.is_format("html") then
      return pandoc.Null()
    end
    local id = args[1] and pandoc.utils.stringify(args[1]) or ""
    if id == "" then
      warn("missing id; usage: {{< audio-overview vol1 >}}")
      return pandoc.Null()
    end
    local m = read_manifest()
    if not m or type(m.items) ~= "table" then
      return pandoc.Null()
    end
    local item = m.items[id]
    if not item then
      return pandoc.Null()
    end
    local base = item.base_url or m.base_url or ""
    local url = item.url or (base .. item.file)
    local mime = item.mime or "audio/mp4"
    local heading = item.heading or m.heading or "Listen"
    local caption = item.caption or m.caption or ""

    local parts = {}
    table.insert(parts, '<section class="podcast-section audio-overview" id="audio-overview-' .. esc(id) .. '">')
    table.insert(parts, "<h2>" .. esc(heading) .. "</h2>")
    if item.title then
      local dur = item.duration and (" <span class=\"audio-overview-duration\">(" .. esc(item.duration) .. ")</span>") or ""
      table.insert(parts, '<p class="audio-overview-title"><strong>' .. esc(item.title) .. "</strong>" .. dur .. "</p>")
    end
    if item.note then
      table.insert(parts, '<p class="audio-overview-note">' .. esc(item.note) .. "</p>")
    end
    -- The publish workflow copies each file into the volume's assets/media/ folder,
    -- so the first source is same-origin and served as audio/mp4. The release URL is
    -- the fallback (dev previews, or a deploy whose copy step was skipped).
    local local_dir = kwargs["local"] and pandoc.utils.stringify(kwargs["local"]) or ""
    if local_dir == "" then
      local_dir = "assets/media"
    elseif local_dir == "none" then
      local_dir = ""
    end
    table.insert(parts, '<audio controls preload="none">')
    if local_dir ~= "" and item.file then
      table.insert(parts, '<source src="' .. esc(local_dir .. "/" .. item.file) .. '" type="' .. esc(mime) .. '">')
    end
    table.insert(parts, '<source src="' .. esc(url) .. '" type="' .. esc(mime) .. '">')
    table.insert(parts, 'Your browser does not support the audio element. <a href="' .. esc(url) .. '">Download the audio</a>.')
    table.insert(parts, "</audio>")
    if caption ~= "" then
      table.insert(parts, '<p class="audio-overview-caption"><em>' .. esc(caption) .. "</em></p>")
    end
    table.insert(parts, "</section>")
    return pandoc.RawBlock("html", table.concat(parts, "\n"))
  end,
}
