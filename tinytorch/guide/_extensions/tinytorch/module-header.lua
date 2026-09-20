-- {{< module-header >}}
--
-- Emits the shared header every module page carries: the audio overview, the
-- Binder / source action cards, and the PDF slide viewer.
--
-- This used to be a 298-line raw-HTML block pasted by hand into all twenty
-- module pages. The copies were identical except for three values, and they
-- had already drifted: two pages named their slide deck differently from the
-- rest. One template, three parameters, read from the page's front matter:
--
--   module:   01_tensor      folder name under tinytorch/modules and src
--   notebook: tensor         notebook file stem inside that folder
--   slides:   01_tensor      slide deck file stem; defaults to `module`
--
-- HTML output only. Other formats get nothing, since the block is interactive.

local function read_template()
  local path = quarto.utils.resolve_path("module-header.html")
  local f = io.open(path, "r")
  if not f then
    error("module-header: template not found at " .. path)
  end
  local text = f:read("*a")
  f:close()
  return text
end

local TEMPLATE = nil

local function meta_string(meta, key)
  local v = meta[key]
  if v == nil then return nil end
  return pandoc.utils.stringify(v)
end

return {
  ["module-header"] = function(args, kwargs, meta)
    if not quarto.doc.is_format("html") then
      return pandoc.Null()
    end

    local module = meta_string(meta, "module")
    local notebook = meta_string(meta, "notebook")
    if module == nil or notebook == nil then
      error("module-header: the page must set `module:` and `notebook:` in its front matter")
    end
    local slides = meta_string(meta, "slides") or module

    TEMPLATE = TEMPLATE or read_template()
    -- Plain-text substitution; the values are filenames, never patterns.
    local html = TEMPLATE
    html = html:gsub("{{module}}", function() return module end)
    html = html:gsub("{{notebook}}", function() return notebook end)
    html = html:gsub("{{slides}}", function() return slides end)

    return pandoc.RawBlock("html", html)
  end
}
