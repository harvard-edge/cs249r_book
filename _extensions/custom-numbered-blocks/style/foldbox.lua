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
  local texEnv = "fbx"
  if #tt.title > 0 then tt.typlabelTag = tt.typlabelTag..": " end
  if fmt =="html" then
    if tt.collapse =="false" then Open=" open" end
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
  local StyleCSSTeX = {}
  
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
    quarto.doc.add_html_dependency({
      name = 'foldbox',
      -- version = '0.0.1',
      --stylesheets = {'style/'..style..'.css'}
      stylesheets = {'style/foldbox.css'}
    })
   elseif fmt == "pdf"
    then
      quarto.doc.use_latex_package("tcolorbox","many")
      quarto.doc.include_file("in-header", 'style/foldbox.tex')
  end
  if preamblestuff then quarto.doc.include_text("in-header", preamblestuff) end
  return(doc)
end
}