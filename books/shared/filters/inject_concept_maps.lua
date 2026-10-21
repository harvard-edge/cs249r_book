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
  local markup = '<details class="chapter-concept-map" style="margin:1.4em 0;border:1px solid #d9e0e5;border-radius:8px;padding:1em">'
    .. '<summary style="cursor:pointer;font-weight:650">Explore the concepts in this chapter</summary>'
    .. '<p>Explore the volume’s connected concept map with this chapter highlighted. Follow labeled relationships into other chapters.</p>'
    .. '<p><a class="concept-map-expand" href="' .. viewer .. '" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;padding:.65em 1em;border-radius:6px;background:#a51c30;color:white;text-decoration:none;font-weight:650">Expand interactive map ↗</a></p>'
    .. '<iframe class="concept-map-frame" src="' .. viewer .. '" title="' .. label .. '" loading="lazy" style="width:100%;height:780px;border:0" sandbox="allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation"></iframe></details>'
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
