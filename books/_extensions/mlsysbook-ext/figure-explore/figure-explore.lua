-- Opt-in web enhancement. Never alter the static diagram or caption AST.
-- is_format("html") includes EPUB; html:js deliberately does not.
local lessons = {
  ["tensor-memory-layout"] = {
    figure = "fig-tensor-memory-layout",
    script = "tensor-memory-layout.js",
    stylesheet = "tensor-memory-layout.css"
  }
}

local function enhance_figure(el)
  local name = el.attributes["data-explore"]
  if not name or not quarto.doc.is_format("html:js") then return nil end
  local lesson = lessons[name]
  if not lesson or el.identifier ~= lesson.figure then
    io.stderr:write("[figure-explore] WARNING: unsupported lesson/figure pair " ..
      tostring(name) .. "/" .. tostring(el.identifier) .. "; retaining static figure.\n")
    return nil
  end
  quarto.doc.add_html_dependency({
    name = "mlsys-figure-explore", version = "0.1.0",
    scripts = {
      {path = "figure-views.js", attribs = {defer = true}},
      {path = "figure-explore.js", attribs = {defer = true}}
    },
    stylesheets = {"figure-views.css"}
  })
  quarto.doc.add_html_dependency({
    name = "mlsys-explore-" .. name, version = "0.1.0",
    scripts = {{path = lesson.script, attribs = {defer = true}}},
    stylesheets = {lesson.stylesheet}
  })
  el.attributes["data-mlsys-explore"] = name
  return el
end

-- Quarto resolves numbered figure Divs into FloatRefTarget nodes before user filters.
return {{Div = enhance_figure, FloatRefTarget = enhance_figure}}
