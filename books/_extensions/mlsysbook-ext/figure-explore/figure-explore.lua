-- Opt-in web enhancement. Static figures and captions remain chapter-owned.
local extension_dir = pandoc.path.directory(PANDOC_SCRIPT_FILE)
local registry = {}
local registry_file = io.open(extension_dir .. '/figure-registry.json', 'r')
if registry_file then registry = pandoc.json.decode(registry_file:read('*a')); registry_file:close() end
local legacy = {
  ['tensor-memory-layout']={figure_id='fig-tensor-memory-layout', script='tensor-memory-layout.js', stylesheet='tensor-memory-layout.css'},
  ['matrix-tiling']={figure_id='fig-tiling-diagram', script='matrix-tiling.js', stylesheet='matrix-tiling.css'}
}
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
  quarto.doc.add_html_dependency({name='mlsys-figure-explore',version='0.2.0',
    scripts={{path='figure-views.js',attribs={defer=true}},{path='figure-explore.js',attribs={defer=true}}},stylesheets={'figure-views.css'}})
  local scripts={{path=lesson.script,attribs={defer=true}}}
  local volume = lesson.volume and string.format('%d',lesson.volume) or ''
  if name=='guided-figure' then table.insert(scripts,{path='guided-lessons-vol'..volume..'.js',attribs={defer=true}}) end
  quarto.doc.add_html_dependency({name='mlsys-explore-'..name..(name=='guided-figure' and '-vol'..volume or ''),version='0.2.0',scripts=scripts,stylesheets={lesson.stylesheet}})
  el.attributes['data-mlsys-explore']=name
  if key then
    el.attributes['data-mlsys-explore-key']=key
    el.attributes['data-mlsys-art-sha1']=lesson.art_structural_sha1 or lesson.art_sha1
    el.attributes['data-mlsys-art-kind']=lesson.art_kind
  end
  return el
end
return {{Div=enhance,FloatRefTarget=enhance}}
