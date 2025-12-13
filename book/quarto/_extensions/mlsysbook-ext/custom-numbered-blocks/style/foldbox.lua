-- nice rename function learned from shafayetShafee :-)
local str = pandoc.utils.stringify
local pout = quarto.log.output


return {

defaultOptions={
  numbered = "true",
  boxstyle = "foldbox.default",
  collapse = "true",
  colors   = {"c0c0c0","808080"}
},

blockStart = function (tt, fmt)
  local Open =""
  local BoxStyle =" fbx-default closebutton"
  -- Special case for quiz answers
  if tt.type == "callout-quiz-answer" then
    BoxStyle = " fbx-answer closebutton"
  else
    -- All other callout types use default styling
    BoxStyle = " fbx-default closebutton"
  end

  local texEnv = "fbx"
  if #tt.title > 0 then tt.typlabelTag = tt.typlabelTag..": " end
  if fmt =="html" then
    -- For ePub (XHTML), use open="open" instead of just open (boolean attribute)
    -- ePub requires strict XHTML syntax
    local isEpub = quarto.doc.is_format("epub")
    if tt.collapse =="false" then
      if isEpub then
        Open=' open="open"'
      else
        Open=" open"
      end
    end
    if tt.boxstyle =="foldbox.simple"
      then
        BoxStyle=" fbx-simplebox fbx-default"
    --    Open=" open" do not force override. Chose this in yaml or individually.
    --    we would want e.g to have remarks closed by default
      end
    result = ('<details class=\"'..tt.type..BoxStyle ..'\"'..Open..'><summary>'..'<strong>'..tt.typlabelTag..'</strong>'..tt.title .. '</summary><div>')
    return result

  elseif fmt =="tex" then
    if tt.boxstyle=="foldbox.simple" then texEnv = "fbxSimple" end
    return('\\begin{'..texEnv..'}{'..tt.type..'}{'..tt.typlabelTag..'}{'..tt.title..'}\n'..
           '\\phantomsection\\label{'..tt.id..'}\n')
  else
    return("<details><summary>Hallihallo</summary>")
  end
end,

blockEnd = function (tt, fmt)
  local texEnv = "fbx"
  if fmt =="html" then
    return('</div></details>')
  elseif fmt =="tex" then
    if tt.boxstyle=="foldbox.simple" then texEnv = "fbxSimple" end
     return('\\end{'..texEnv..'}\n')
  else return ('ende mit format '..fmt..'=================')
  end
end,

insertPreamble = function(doc, classDefs, fmt)
  local ishtml = quarto.doc.is_format("html")
  local ispdf = quarto.doc.is_format("pdf")
  local isepub = quarto.doc.is_format("epub")
  -- Note: ePub format is treated as HTML (fmt="html") since ePub uses HTML internally
  local StyleCSSTeX = {}

  -- Set icon path from filter-metadata configuration if available
  local meta = doc.meta
  local filterMetadata = meta["filter-metadata"]
  local iconCSS = ""
  if filterMetadata and filterMetadata["mlsysbook-ext/custom-numbered-blocks"] then
    local config = filterMetadata["mlsysbook-ext/custom-numbered-blocks"]
    if config["icon-path"] then
      local iconPath = str(config["icon-path"])
      local iconFormat = str(config["icon-format"] or "png")

      if fmt == "html" then
        -- Generate dynamic CSS for icon paths - generic version using classDefs
        iconCSS = "<style>\n"
        if classDefs then
          for calloutType, _ in pairs(classDefs) do
            -- Convert hyphens to underscores for icon filename (e.g., callout-quiz-question -> callout_quiz_question)
            local iconFileName = calloutType:gsub("-", "_")
            iconCSS = iconCSS .. "details." .. calloutType .. " > summary::before {\n"
            iconCSS = iconCSS .. "  background-image: url(\"" .. iconPath .. "/icon_" .. iconFileName .. "." .. iconFormat .. "\");\n"
            iconCSS = iconCSS .. "}\n"
          end
        end
        iconCSS = iconCSS .. "</style>"
      elseif fmt == "pdf" then
        -- Define the commands before including foldbox.tex
        quarto.doc.include_text("in-header", "\\newcommand{\\fbxIconPath}{" .. iconPath .. "}")
        quarto.doc.include_text("in-header", "\\newcommand{\\fbxIconFormat}{" .. iconFormat .. "}")
      end
    end
  end

  -- if fmt==nil then pout("=== NIX ======= Format   ") else
  -- pout("============== Format :   "..str(fmt)) end
  -- make css or preamble tex for colors
  local extractStyleFromMeta = function (fmt)
    local result
    if classDefs ~= nil then
      for cls, options in pairs(classDefs) do
        --quarto.log.output(cls)
        if options.colors then
          -- quarto.log.output("  --> Farben!")
          if fmt == "html" then
            table.insert(StyleCSSTeX, "."..cls.." {\n")
            for i, col in ipairs(options.colors) do
              table.insert(StyleCSSTeX, "  --color"..i..": #"..col..";\n")
            end
            table.insert(StyleCSSTeX, "}\n")
          elseif fmt == "pdf" then
            for i, col in ipairs(options.colors) do
              table.insert(StyleCSSTeX, "\\definecolor{"..cls.."-color"..i.."}{HTML}{"..col.."}\n")
            end
          end
        end
      end
    end
    result = pandoc.utils.stringify(StyleCSSTeX)
    if fmt == "html" then result = "<style>\n"..result.."</style>" end
    if fmt == "pdf" then result="%%==== colors from yaml ===%\n"..result.."%=============%\n" end
    return(result)
  end

  local preamblestuff = extractStyleFromMeta(fmt)
  -- quarto.log.output(preamblestuff)

  if fmt == "html"
  then
    -- For ePub, we don't add the foldbox.css dependency as it's not needed
    -- The styles are defined in the main epub.css file
    if not isepub then
      quarto.doc.add_html_dependency({
        name = 'foldbox',
        -- version = '0.0.1',
        --stylesheets = {'style/'..style..'.css'}
        stylesheets = {'style/foldbox.css'}
      })
    end
   elseif fmt == "pdf"
    then
      quarto.doc.use_latex_package("tcolorbox","many")
      quarto.doc.include_file("in-header", 'style/foldbox.tex')
  end
  -- Skip injecting CSS styles for ePub - they're in epub.css already
  if not isepub then
    if preamblestuff then quarto.doc.include_text("in-header", preamblestuff) end
    if iconCSS ~= "" then quarto.doc.include_text("in-header", iconCSS) end
  end
  return(doc)
end
}
