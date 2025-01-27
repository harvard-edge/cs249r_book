local function ensureLatexDeps()
  quarto.doc.use_latex_package("fontawesome5")
end

local function ensureHtmlDeps()
  quarto.doc.add_html_dependency({
    name = 'fontawesome6',
    version = '1.2.0',
    stylesheets = {'assets/css/all.min.css', 'assets/css/latex-fontsize.css'}
  })
end

local function isEmpty(s)
  return s == nil or s == ''
end

local function isValidSize(size)
  local validSizes = {
    "tiny",
    "scriptsize",
    "footnotesize",
    "small",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge"
  }
  for _, v in ipairs(validSizes) do
    if v == size then
      return size
    end
  end
  return ""
end

return {
  ["fa"] = function(args, kwargs)

    local group = "solid"
    local icon = pandoc.utils.stringify(args[1])
    if #args > 1 then
      group = icon
      icon = pandoc.utils.stringify(args[2])
    end
    
    local title = pandoc.utils.stringify(kwargs["title"])
    if not isEmpty(title) then
      title = " title=\"" .. title  .. "\""
    end

    local label = pandoc.utils.stringify(kwargs["label"])
    if isEmpty(label) then
      label = " aria-label=\"" .. icon  .. "\""
    else
      label = " aria-label=\"" .. label  .. "\""
    end

    local size = pandoc.utils.stringify(kwargs["size"])
    
    -- detect html (excluding epub which won't handle fa)
    if quarto.doc.is_format("html:js") then
      ensureHtmlDeps()
      if not isEmpty(size) then
        size = " fa-" .. size
      end
      return pandoc.RawInline(
        'html',
        "<i class=\"fa-" .. group .. " fa-" .. icon .. size .. "\"" .. title .. label .. "></i>"
      )
    -- detect pdf / beamer / latex / etc
    elseif quarto.doc.is_format("pdf") then
      ensureLatexDeps()
      if isEmpty(isValidSize(size)) then
        return pandoc.RawInline('tex', "\\faIcon{" .. icon .. "}")
      else
        return pandoc.RawInline('tex', "{\\" .. size .. "\\faIcon{" .. icon .. "}}")
      end
    else
      return pandoc.Null()
    end
  end
}
