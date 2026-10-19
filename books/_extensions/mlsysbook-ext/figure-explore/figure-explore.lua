-- Opt-in web enhancement. Static figures and captions remain chapter-owned.
local extension_dir = pandoc.path.directory(PANDOC_SCRIPT_FILE)
local registry = {}
local registry_file = io.open(extension_dir .. '/figure-registry.json', 'r')
if registry_file then registry = pandoc.json.decode(registry_file:read('*a')); registry_file:close() end
local legacy = {
  ['tensor-memory-layout']={figure_id='fig-tensor-memory-layout', script='tensor-memory-layout.js', stylesheet='tensor-memory-layout.css'},
  ['matrix-tiling']={figure_id='fig-tiling-diagram', script='matrix-tiling.js', stylesheet='matrix-tiling.css'}
}
-- Read shared build metadata before walking figures. No browser preferences.
local playback = nil
local function configure(meta)
  playback = nil
  if not quarto.doc.is_format('html:js') then return end
  local config = meta['figure-explore']
  if not config then return end
  local function milliseconds(key, minimum)
    local value = tonumber(pandoc.utils.stringify(config[key] or ''))
    if not value or value ~= math.floor(value) or value < minimum or value > 60000 then
      io.stderr:write('[figure-explore] WARNING: figure-explore.'..key..' must be an integer between '..minimum..' and 60000 milliseconds; retaining static figures.\n')
      return nil
    end
    return tostring(value)
  end
  local movement, hold = milliseconds('movement-ms', 1), milliseconds('reading-pause-ms', 0)
  if movement and hold then playback = {movement=movement, hold=hold} end
end
local source_valid = {}
local function warn(message)
  io.stderr:write('[figure-explore] WARNING: '..message..'; retaining static figure.\n')
end
local function current_source(lesson)
  if source_valid[lesson.source] ~= nil then return source_valid[lesson.source] end
  local path = extension_dir .. '/../../../' .. lesson.source:gsub('^books/', '')
  local f = io.open(path, 'r')
  if not f then source_valid[lesson.source]=false;return false end
  local text=f:read('*a');f:close()
  text=text:gsub('%s+data%-explore%-key="[^"]*"',''):gsub('%s+data%-explore="[^"]*"','')
  source_valid[lesson.source] = pandoc.utils.sha1(text) == lesson.source_sha1
  return source_valid[lesson.source]
end
local function current_bundle(lesson)
  if not lesson.bundle or not lesson.bundle_sha1 or not lesson.lesson_sha1 then return false end
  local f = io.open(extension_dir .. '/' .. lesson.bundle, 'rb')
  if not f then return false end
  local bytes = f:read('*a'); f:close()
  return pandoc.utils.sha1(bytes) == lesson.bundle_sha1
end
local function enhance(el)
  local name=el.attributes['data-explore']
  if not name or not quarto.doc.is_format('html:js') then return nil end
  local key=el.attributes['data-explore-key']
  local lesson=key and registry[key] or legacy[name]
  if not lesson or lesson.figure_id~=el.identifier or (key and lesson.name~=name) then
    warn('unsupported lesson/figure pair '..tostring(name)..'/'..el.identifier);return nil
  end
  if key and not current_source(lesson) then
    warn('stale source mapping for '..key..'; review artwork and rerun compile_figure_explore.py --enroll');return nil
  end
  if not playback then
    warn('missing shared figure-explore timing metadata');return nil
  end
  if lesson.bundle and not current_bundle(lesson) then
    warn('missing or stale mechanism bundle for '..tostring(key));return nil
  end
  local view_assets = ''
  for _,filename in ipairs({'figure-views.js','figure-explore.js','figure-views.css'}) do
    local f = io.open(extension_dir .. '/' .. filename, 'rb')
    if not f then warn('missing figure-view runtime');return nil end
    view_assets = view_assets .. f:read('*a');f:close()
  end
  quarto.doc.add_html_dependency({name='mlsys-figure-explore',version='0.3.0-'..pandoc.utils.sha1(view_assets):sub(1,12),
    scripts={{path='figure-views.js',attribs={defer=true}},{path='figure-explore.js',attribs={defer=true}}},stylesheets={'figure-views.css'}})
  if lesson.shared_player then
    local assets = ''
    for _,filename in ipairs({'original-player.js','original-player.css'}) do
      local f = io.open(extension_dir .. '/' .. filename, 'rb')
      if not f then warn('missing shared original-diagram player');return nil end
      assets = assets .. f:read('*a');f:close()
    end
    quarto.doc.add_html_dependency({name='mlsys-original-player',version='0.1.0-'..pandoc.utils.sha1(assets):sub(1,12),
      scripts={{path='original-player.js',attribs={defer=true}}},stylesheets={'original-player.css'}})
  end
  local scripts={{path=lesson.script,attribs={defer=true}}}
  local volume = lesson.volume and string.format('%d',lesson.volume) or ''
  if lesson.bundle then table.insert(scripts,{path=lesson.bundle,attribs={defer=true}}) end
  if name=='guided-figure' then table.insert(scripts,{path='guided-lessons-vol'..volume..'.js',attribs={defer=true}}) end
  quarto.doc.add_html_dependency({name='mlsys-explore-'..name..((name=='guided-figure' or name=='mechanism-scene') and '-vol'..volume or ''),version=(lesson.bundle and '0.3.0-'..(lesson.runtime_sha1 or 'unversioned'):sub(1,12)..'-'..lesson.bundle_sha1:sub(1,12) or '0.2.0'),scripts=scripts,stylesheets={lesson.stylesheet}})
  if playback then
    el.attributes['data-mlsys-movement-ms']=playback.movement
    el.attributes['data-mlsys-reading-pause-ms']=playback.hold
  end
  el.attributes['data-mlsys-explore']=name
  if key then
    el.attributes['data-mlsys-explore-key']=key
    if lesson.bundle then el.attributes['data-mlsys-lesson-sha1']=lesson.lesson_sha1 end
    el.attributes['data-mlsys-art-sha1']=lesson.art_structural_sha1 or lesson.art_sha1
    el.attributes['data-mlsys-art-kind']=lesson.art_kind
  end
  return el
end
return {{Meta=configure},{Div=enhance,FloatRefTarget=enhance}}
