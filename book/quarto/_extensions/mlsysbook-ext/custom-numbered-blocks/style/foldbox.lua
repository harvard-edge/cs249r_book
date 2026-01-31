-- nice rename function learned from shafayetShafee :-)
local str = pandoc.utils.stringify
local pout = quarto.log.output

-- Convert a 6-digit hex color string to R, G, B decimal values
-- e.g., "5B4B8A" -> 91, 75, 138
local function hexToRGB(hex)
  hex = hex:gsub("^#", "")
  local r = tonumber(hex:sub(1, 2), 16)
  local g = tonumber(hex:sub(3, 4), 16)
  local b = tonumber(hex:sub(5, 6), 16)
  return r, g, b
end


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
  if #tt.title > 0 then tt.typlabelTag = tt.typlabelTag..":" end
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
    local titleSep = ""
    if #tt.title > 0 then titleSep = "\u{2003}" end
    result = ('<details class=\"'..tt.type..BoxStyle ..'\"'..Open..'><summary>'..'<strong>'..tt.typlabelTag .. titleSep .. tt.title..'</strong></summary><div>')
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

  -- =========================================================================
  -- Generate CSS/LaTeX color styles from YAML configuration
  -- YAML is the single source of truth for all callout colors.
  -- colors[1] = background hex (e.g., "F0F0F8")
  -- colors[2] = border/accent hex (e.g., "5B4B8A")
  -- =========================================================================

  -- Light mode opacity for background/title (subtle tint)
  local LIGHT_BG_OPACITY = 0.04
  -- Dark mode opacity (stronger tint for visibility on dark backgrounds)
  local DARK_BG_OPACITY = 0.12

  local extractStyleFromMeta = function (fmt)
    local result
    if classDefs ~= nil then
      for cls, options in pairs(classDefs) do
        if options.colors then
          if fmt == "html" then
            -- Keep legacy --color1/--color2 for any CSS that still references them
            table.insert(StyleCSSTeX, "."..cls.." {\n")
            for i, col in ipairs(options.colors) do
              table.insert(StyleCSSTeX, "  --color"..i..": #"..col..";\n")
            end
            -- Generate semantic color variables from color2 (the accent/border color)
            -- Light mode values are used directly; dark mode values are stored
            -- as --dark-* for dark-mode.scss to reference via the manual toggle.
            if options.colors[2] then
              local borderHex = options.colors[2]
              local r, g, b = hexToRGB(borderHex)
              table.insert(StyleCSSTeX, "  --border-color: #"..borderHex..";\n")
              table.insert(StyleCSSTeX, "  --background-color: rgba("..r..", "..g..", "..b..", "..LIGHT_BG_OPACITY..");\n")
              table.insert(StyleCSSTeX, "  --title-background-color: rgba("..r..", "..g..", "..b..", "..LIGHT_BG_OPACITY..");\n")
              -- Pre-computed dark mode values for manual toggle (dark-mode.scss)
              table.insert(StyleCSSTeX, "  --dark-background-color: rgba("..r..", "..g..", "..b..", "..DARK_BG_OPACITY..");\n")
              table.insert(StyleCSSTeX, "  --dark-title-background-color: rgba("..r..", "..g..", "..b..", "..DARK_BG_OPACITY..");\n")
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

  -- Generate dark mode CSS overrides from YAML colors
  -- This covers BOTH @media (prefers-color-scheme: dark) AND manual toggle
  local generateDarkModeCSS = function ()
    local darkCSS = {}
    if classDefs == nil then return "" end

    -- 1. @media query block for system-level dark mode (detected by OS)
    table.insert(darkCSS, "<style>\n@media (prefers-color-scheme: dark) {\n")
    for cls, options in pairs(classDefs) do
      if options.colors and options.colors[2] then
        local borderHex = options.colors[2]
        local r, g, b = hexToRGB(borderHex)
        table.insert(darkCSS, "  details."..cls.." {\n")
        table.insert(darkCSS, "    --text-color: #e6e6e6;\n")
        table.insert(darkCSS, "    --background-color: rgba("..r..", "..g..", "..b..", "..DARK_BG_OPACITY..");\n")
        table.insert(darkCSS, "    --title-background-color: rgba("..r..", "..g..", "..b..", "..DARK_BG_OPACITY..");\n")
        table.insert(darkCSS, "    border-color: #"..borderHex..";\n")
        table.insert(darkCSS, "  }\n")
        -- Bright summary text
        table.insert(darkCSS, "  details."..cls.." summary,\n")
        table.insert(darkCSS, "  details."..cls.." summary strong,\n")
        table.insert(darkCSS, "  details."..cls.." > summary {\n")
        table.insert(darkCSS, "    color: #f0f0f0 !important;\n")
        table.insert(darkCSS, "  }\n")
        -- Code elements
        table.insert(darkCSS, "  details."..cls.." code {\n")
        table.insert(darkCSS, "    color: #e6e6e6 !important;\n")
        table.insert(darkCSS, "  }\n")
      end
    end
    table.insert(darkCSS, "}\n</style>\n")

    return pandoc.utils.stringify(darkCSS)
  end

  local preamblestuff = extractStyleFromMeta(fmt)
  local darkModeCSSBlock = ""
  if fmt == "html" and not isepub then
    darkModeCSSBlock = generateDarkModeCSS()
  end

  if fmt == "html"
  then
    -- For ePub, we don't add the foldbox.css dependency as it's not needed
    -- The styles are defined in the main epub.css file
    if not isepub then
      quarto.doc.add_html_dependency({
        name = 'foldbox',
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
    if darkModeCSSBlock ~= "" then quarto.doc.include_text("in-header", darkModeCSSBlock) end
    if iconCSS ~= "" then quarto.doc.include_text("in-header", iconCSS) end
  end
  return(doc)
end
}
