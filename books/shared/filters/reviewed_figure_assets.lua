-- Preserve reviewed HTML artwork independently of local fonts/PDF conversion.
-- Known bindings fail the build when source, configuration, or asset bytes drift.
local function check(ok, message)
  assert(ok, '[reviewed-figure-assets] ' .. message)
end
local function read(path)
  local file = io.open(path, 'rb')
  check(file ~= nil, 'missing declared file: ' .. path)
  local bytes = file:read('*a'); file:close()
  return bytes
end
local function normalized(path)
  path = pandoc.path.normalize(path):gsub('\\', '/')
  local parts = {}
  for part in path:gmatch('[^/]+') do
    if part == '..' then table.remove(parts)
    elseif part ~= '.' then parts[#parts + 1] = part end
  end
  return (path:sub(1, 1) == '/' and '/' or '') .. table.concat(parts, '/')
end
local script = PANDOC_SCRIPT_FILE
if not pandoc.path.is_absolute(script) then
  script = pandoc.path.join{pandoc.system.get_working_directory(), script}
end
local root = normalized(pandoc.path.join{pandoc.path.directory(script), '../../..'})
local art_dir = root .. '/books/_extensions/mlsysbook-ext/figure-explore/canonical-art'
local function project_file(path)
  check(type(path) == 'string' and path ~= '' and not pandoc.path.is_absolute(path)
    and not path:find('\\') and not ('/' .. path .. '/'):find('/%.%./'), 'invalid declared path')
  return root .. '/' .. path
end
-- Preserve actual parsed metadata, including RawInline TeX and its format.
-- stringify() drops raw TeX; flattening it would miss header changes.
local function canonical(value)
  if type(value) ~= 'table' then return pandoc.json.encode(value) end
  local parts = {}
  if #value > 0 then
    for _, child in ipairs(value) do parts[#parts + 1] = canonical(child) end
    return '[' .. table.concat(parts, ',') .. ']'
  end
  local keys = {}
  for key in pairs(value) do keys[#keys + 1] = key end
  table.sort(keys)
  for _, key in ipairs(keys) do
    parts[#parts + 1] = pandoc.json.encode(key) .. ':' .. canonical(value[key])
  end
  return '{' .. table.concat(parts, ',') .. '}'
end
local function block_options(block)
  local result = {identifier=block.identifier, classes={}, attributes={}}
  for _, name in ipairs(block.classes) do result.classes[#result.classes + 1] = name end
  for key, value in pairs(block.attributes) do result.attributes[key] = value end
  return result
end
local expected, known, seen, lesson_names
local function configure(meta)
  expected, known, seen, lesson_names = nil, nil, {}, {}
  if not quarto.doc.is_format('html:js') then return nil end
  local manifest = pandoc.json.decode(read(art_dir .. '/manifest.json'))
  check(manifest.schema_version == 1 and type(manifest.figures) == 'table', 'invalid manifest')
  local registry = pandoc.json.decode(read(art_dir .. '/../figure-registry.json'))
  local owner = normalized(project_file(manifest.owning_source))
  local bindings = 0
  for key, lesson in pairs(registry) do
    if lesson.source == manifest.owning_source then
      check(manifest.figures[key] ~= nil, 'missing canonical binding: ' .. key)
      bindings = bindings + 1
    end
  end
  check(bindings > 0, 'owning source has no reviewed registry entries')
  for key, entry in pairs(manifest.figures) do
    local lesson = registry[key]
    check(lesson and entry.source == manifest.owning_source and lesson.source == entry.source
      and lesson.figure_id == entry.figure_id and lesson.source_sha1 == entry.source_sha1
      and lesson.art_sha1 == entry.asset_sha1 and lesson.art_structural_sha1 == entry.reviewed_structural_sha1,
      'registry/provenance binding changed: ' .. key)
    lesson_names[key] = lesson.name
  end
  local input = quarto.doc.input_file or ''
  if not pandoc.path.is_absolute(input) then
    input = pandoc.path.join{pandoc.system.get_working_directory(), input}
  end
  input = normalized(input)
  known = manifest.figures
  if input ~= owner then return nil end
  expected = manifest.figures
  local effective = pandoc.json.decode(pandoc.json.encode(meta.diagram))
  check(canonical(effective) == canonical(manifest.effective_diagram),
    'effective diagram configuration changed; review the source and artwork')
  check(manifest.renderer.path == 'books/_extensions/pandoc-ext/diagram/diagram.lua',
    'missing renderer/template/seed binding')
  check(pandoc.utils.sha1(read(project_file(manifest.renderer.path))) == manifest.renderer.sha1,
    'diagram renderer/template/seed changed; review the source and artwork')
  check(type(manifest.declared_inputs) == 'table', 'missing declared input closure')
  local declared = {}
  for _, dependency in ipairs(manifest.declared_inputs) do
    check(not declared[dependency.path], 'duplicate declared input: ' .. dependency.path)
    declared[dependency.path] = true
    check(pandoc.utils.sha1(read(project_file(dependency.path))) == dependency.sha1,
      'declared input changed: ' .. dependency.path)
  end
  check(declared['books/config/shared/base/diagram.yml'], 'missing shared diagram input binding')
  local source_bytes = read(input)
  check(pandoc.utils.sha1(source_bytes) == manifest.source_bytes_sha1, 'source changed (exact bytes)')
  local source = source_bytes:gsub('%s+data%-explore%-key="[^"]*"', ''):gsub('%s+data%-explore="[^"]*"', '')
  for key, entry in pairs(expected) do
    local lesson = registry[key]
    check(lesson and lesson.source == entry.source and lesson.figure_id == entry.figure_id
      and lesson.source_sha1 == entry.source_sha1 and lesson.art_sha1 == entry.asset_sha1,
      'registry/provenance binding changed: ' .. key)
    check(pandoc.utils.sha1(source) == entry.source_sha1, 'source changed: ' .. key)
  end

end
local function replace(outer)
    if not known then return nil end
    if not expected then
      check(not known[outer.attributes['data-explore-key']],
        'canonical figure enrolled in the wrong owning source')
      return nil
    end
    local entry, key
    for candidate, bound in pairs(expected) do
      if outer.identifier == bound.figure_id then entry, key = bound, candidate; break end
    end
    if not entry then return nil end
    seen[key] = (seen[key] or 0) + 1
    check(seen[key] == 1, 'ambiguous figure wrapper: ' .. key)
    check(outer.attributes['data-explore-key'] == key, 'figure enrollment changed: ' .. key)
    check(outer.attributes['data-explore'] == lesson_names[key], 'figure lesson enrollment changed: ' .. key)
    check(outer.content ~= nil, 'expected exactly one TikZ block: ' .. key)
    local count = 0
    local content = pandoc.Div(outer.content):walk{CodeBlock=function(block)
      count = count + 1
      check(block.classes[1] == 'tikz' and pandoc.utils.sha1(block.text) == entry.tikz_sha1,
        'TikZ body changed: ' .. key)
      check(canonical(block_options(block)) == canonical(entry.block_options),
        'TikZ block options changed: ' .. key)
      check(type(entry.asset) == 'string' and entry.asset:match('^vol%d+/[%w_-]+%.svg$'),
        'invalid canonical artwork path: ' .. key)
      local bytes = read(art_dir .. '/' .. entry.asset)
      check(pandoc.utils.sha1(bytes) == entry.asset_sha1, 'canonical artwork bytes changed: ' .. key)
      local _, view_box = bytes:match('viewBox%s*=%s*(["\'])(.-)%1')
      check(view_box == entry.view_box, 'canonical artwork geometry changed: ' .. key)
      local filename = entry.asset_sha1 .. '.svg'
      pandoc.mediabag.insert(filename, 'image/svg+xml', bytes)
      return pandoc.Plain{pandoc.Image({}, filename)}
    end}
    check(count == 1, 'expected exactly one TikZ block: ' .. key)
    outer.content = content.content
    outer.attributes['data-mlsys-reviewed-art-sha1'] = entry.asset_sha1
    return outer
end
local function complete()
  if expected then
    for key in pairs(expected) do check(seen[key] == 1, 'missing figure wrapper: ' .. key) end
  end
end
return {{Meta=configure}, {Div=replace, FloatRefTarget=replace}, {Pandoc=complete}}
