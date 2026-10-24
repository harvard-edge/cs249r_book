-- Chapter learner graphs are authored in *_concepts.yml and compiled before render.
-- HTML only; inject after the existing learning-objectives block without editing prose.
local json = require("pandoc.json")

local function escape(value)
  return value:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;"):gsub(">", "&gt;")
end

local function inject(doc)
  if not quarto.doc.is_format("html:js") then return nil end
  local input = (quarto.doc.input_file or ""):gsub("\\", "/")
  local volume, chapter = input:match("(vol%d+)/([^/]+)/[^/]+%.qmd$")
  if not volume then return nil end
  local root = quarto.project.directory or "."
  local file = io.open(root .. "/_build/concepts/chapters.json", "r")
  if not file then return nil end
  local raw = file:read("*all")
  file:close()
  local ok, manifest = pcall(json.decode, raw)
  if not ok then error("Concept map manifest is invalid; rerun binder concepts build") end
  local entry = manifest[volume] and manifest[volume][chapter]
  if not entry then return nil end
  local viewer = escape(entry.viewer)
  local label = escape(entry.title .. " concept maps")
  -- Folded by default and visually quiet: one sentence, one text link out to
  -- the full map, and the map itself. The viewer hides its own page controls
  -- when embedded (explorer.css, .embedded-map) and follows the page theme.
  local markup = '<details class="chapter-concept-map">'
    .. '<summary>Explore the concepts in this chapter</summary>'
    .. '<p class="chapter-concept-map-lede">The volume’s concept map, opened on this chapter. Select a concept to read how it connects to ideas in other chapters. '
    .. '<a class="concept-map-expand" href="' .. viewer .. '" target="_blank" rel="noopener">Open the full map ↗</a></p>'
    .. '<iframe class="concept-map-frame" src="' .. viewer .. '" title="' .. label .. '" loading="lazy" sandbox="allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation"></iframe></details>'
    .. '<style>.chapter-concept-map{margin:1.4em 0;border:1px solid var(--bs-border-color,#dee2e6);border-left:3px solid var(--bs-border-color,#dee2e6);border-radius:.25rem;padding:.6em 1em}'
    .. '.chapter-concept-map>summary{cursor:pointer;font-weight:600;font-size:.95em}'
    .. '.chapter-concept-map-lede{font-size:.9em;margin:.7em 0 .6em;color:var(--bs-secondary-color,#6c757d)}'
    .. '.chapter-concept-map-lede a{white-space:nowrap}'
    .. '.chapter-concept-map iframe{display:block;width:100%;height:560px;border:0;background:transparent}</style>'
    .. '<script>(()=>{const box=document.querySelector(".chapter-concept-map");if(!box)return;const frame=box.querySelector("iframe"),link=box.querySelector(".concept-map-expand");window.addEventListener("message",event=>{if(event.source!==frame.contentWindow||event.origin!==location.origin||!event.data||event.data.type!=="mlsys-concept-map-state"||typeof event.data.map!=="string")return;const url=new URL(frame.src,location.href);url.searchParams.set("map",event.data.map);if(typeof event.data.chapter==="string"&&event.data.chapter)url.searchParams.set("chapter",event.data.chapter);else url.searchParams.delete("chapter");url.hash=event.data.concept?"concept="+encodeURIComponent(event.data.concept):"";link.href=url.href;});})();</script>'
  local added = false
  doc = doc:walk({Div = function(div)
    if not added and div.classes:includes("callout-learning-objectives") then
      added = true
      return {div, pandoc.RawBlock("html", markup)}
    end
  end})
  if not added then error("Chapter concept map needs a learning-objectives block: " .. input) end
  return doc
end

return {{Pandoc = inject}}
