local function isEmpty(s)
  return s == nil or s == ''
end

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local function getVal(s)
  return pandoc.utils.stringify(s)
end

local function is_equal (s, val)
  if isEmpty(s) then return false end
  if getVal(s) == val then return true end

  return false
end

local function has_value (tab, val)
    for index, value in ipairs(tab) do
        if value == val then
            return true
        end
    end

    return false
end

local function dump(o)
   if type(o) == 'table' then
      local s = '{ '
      for k,v in pairs(o) do
         if type(k) ~= 'number' then k = '"'..k..'"' end
         s = s .. '['..k..'] = ' .. dump(v) .. ','
      end
      return s .. '} '
   else
      return tostring(o)
   end
end

function Meta(m)
--[[
This function checks that the value the user set is ok and stops with an error message if no.
yamlelement: the yaml metadata. e.g. m["titlepage-theme"]["page-align"]
yamltext: page, how to print the yaml value in the error message. e.g. titlepage-theme: page-align
okvals: a text table of ok styles. e.g. {"right", "center"}
--]]
  local function check_yaml (yamlelement, yamltext, okvals)
    choice = pandoc.utils.stringify(yamlelement)
    if not has_value(okvals, choice) then
      print("\n\ntitlepage extension error: " .. yamltext .. " is set to " .. choice .. ". It can be " .. pandoc.utils.stringify(table.concat(okvals, ", ")) .. ".\n\n")
      return false
    else
      return true
    end

    return true
  end

--[[
This function gets the value of something like titlepage-theme.title-style and sets a value titlepage-theme.title-style.plain (for example). It also
does error checking against okvals. "plain" is always ok and if no value is set then the style is set to plain.
page: titlepage or coverpage
styleement: page, title, subtitle, header, footer, affiliation, etc
okvals: a text table of ok styles. e.g. {"plain", "two-column"}
--]]
  local function set_style (page, styleelement, okvals)
    yamltext = page .. "-theme" .. ": " .. styleelement .. "-style"
    yamlelement = m[page .. "-theme"][styleelement .. "-style"]
    if not isEmpty(yamlelement) then
      ok = check_yaml (yamlelement, yamltext, okvals)
      if ok then
        m[page .. "-style-code"][styleelement] = {}
        m[page .. "-style-code"][styleelement][getVal(yamlelement)] = true
      else
        error()
      end
    else
--      print("\n\ntitlepage extension error: " .. yamltext .. " needs a value. Should have been set in titlepage-theme lua filter.\n\n")
--      error()
        m[page .. "-style-code"][styleelement] = {}
        m[page .. "-style-code"][styleelement]["plain"] = true
    end
  end

--[[
This function assigns the themevals to the meta data
--]]
  local function assign_value (tab)
    for i, value in pairs(tab) do
      if isEmpty(m['titlepage-theme'][i]) then
        m['titlepage-theme'][i] = value
      end
    end

    return m
  end

  local titlepage_table = {
    ["academic"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\headerblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\dateblock")}
          },
        ["page-align"] = "center",
        ["title-style"] = "doublelinetight",
        ["title-fontstyle"] = {"huge", "bfseries"},
        ["title-space-after"] = "1.5cm",
        ["subtitle-fontstyle"] = {"Large"},
        ["author-style"] = "two-column",
        ["affiliation-style"] = "none",
        ["author-fontstyle"] = {"textsc"},
        ["affiliation-fontstyle"] = {"large"},
        ["logo-space-after"] = pandoc.MetaInlines{pandoc.RawInline("latex","2\\baselineskip")},
        ["header-fontstyle"] = {"textsc", "LARGE"},
        ["header-space-after"] = "1.5cm",
        ["date-fontstyle"] = {"large"}
        }
      assign_value(themevals)
        
      return m
    end,
    ["bg-image"] = function (m)
      if isEmpty(m['titlepage-bg-image']) then
        m['titlepage-bg-image'] = "corner-bg.png"
      end
      if isEmpty(m['titlepage-geometry']) then
        m['titlepage-geometry'] = pandoc.List({"top=3in", "bottom=1in", "right=1in", "left=1in"})
      end
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\affiliationblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "left",
        ["title-style"] = "plain",
        ["title-fontstyle"] = {"large", "bfseries"},
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","4\\baselineskip")},
        ["subtitle-fontstyle"] = {"large", "textit"},
        ["author-style"] = "superscript-with-and",
        ["author-fontstyle"] = {"large"},
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-fontstyle"] = {"large"},
        ["footer-space-after"] = "1pt",
        ["affiliation-space-after"] = "1pt",
        ["footer-style"] = "plain",
        ["footer-fontstyle"] = {"large"},
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.25\\textheight")},
        ["logo-space-after"] = pandoc.MetaInlines{pandoc.RawInline("latex","2\\baselineskip")},
        ["vrule-width"] = "1pt",
        ["bg-image-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.5\\paperwidth")},
        ["bg-image-location"] = "ULCorner",
        }
      assign_value(themevals)
        
      return m
    end,
    ["classic-lined"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "center",
        ["title-style"] = "doublelinewide",
        ["title-fontsize"] = 30,
        ["title-fontstyle"] = {"uppercase"},
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.1\\textheight")},
        ["subtitle-fontstyle"] = {"Large", "textit"},
        ["author-style"] = "plain",
        ["author-sep"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","\\hskip1em")},
        ["author-fontstyle"] = {"Large"},
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-fontstyle"] = {"large"},
        ["affiliation-space-after"] = "1pt",
        ["footer-style"] = "plain",
        ["footer-fontstyle"] = {"large", "textsc"},
        ["footer-space-after"] = "1pt",
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.25\\textheight")},
        ["logo-space-after"] = "1cm",
        }
      assign_value(themevals)
        
      return m
    end,
    ["colorbox"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")}
          },
        ["page-align"] = "left",
        ["title-style"] = "colorbox",
        ["title-fontsize"] = 40,
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["subtitle-fontsize"] = 25,
        ["subtitle-fontstyle"] = {"bfseries"},
        ["title-subtitle-space-between"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","5\\baselineskip")},
        ["author-style"] = "plain",
        ["author-sep"] = "newline",
        ["author-fontstyle"] = {"Large"},
        ["author-align"] = "right",
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["title-colorbox-borderwidth"] = "2mm",
        ["title-colorbox-bordercolor"] = "black",
        }
      assign_value(themevals)
        
      return m
    end,
    ["formal"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","A report presented at the annual\\\\meeting on 10 August 2025\\\\ \\vspace{0.8cm}")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "center",
        ["title-style"] = "plain",
        ["title-fontstyle"] = {"Huge", "textbf"},
        ["title-space-after"] = "1.5cm",
        ["subtitle-fontstyle"] = {"LARGE"},
        ["title-subtitle-space-between"] = "0.5cm",
        ["author-style"] = "plain",
        ["author-sep"] = "newline",
        ["author-fontstyle"] = {"textbf"},
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-fontstyle"] = {"large"},
        ["affiliation-space-after"] = "1pt",
        ["footer-style"] = "plain",
        ["footer-fontstyle"] = {"Large", "textsc"},
        ["footer-space-after"] = "1pt",
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.4\\textwidth")},
        ["logo-space-after"] = "1cm",
        }
      assign_value(themevals)
        
      return m
    end,
    ["vline"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\affiliationblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "left",
        ["title-style"] = "plain",
        ["title-fontstyle"] = {"large", "bfseries"},
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","4\\baselineskip")},
        ["subtitle-fontstyle"] = {"large", "textit"},
        ["author-style"] = "superscript-with-and",
        ["author-fontstyle"] = {"large"},
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-fontstyle"] = {"large"},
        ["affiliation-space-after"] = "1pt",
        ["footer-style"] = "plain",
        ["footer-fontstyle"] = {"large"},
        ["footer-space-after"] = "1pt",
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.15\\textheight")},
        ["logo-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.1\\textheight")},
        ["vrule-width"] = "2pt",
        ["vrule-align"] = "left",
        ["vrule-color"] = "black",
        }
      assign_value(themevals)
        
      return m
    end,
    ["vline-text"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\affiliationblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "left",
        ["title-style"] = "plain",
        ["title-fontstyle"] = {"large", "bfseries"},
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","4\\baselineskip")},
        ["subtitle-fontstyle"] = {"large", "textit"},
        ["author-style"] = "superscript-with-and",
        ["author-fontstyle"] = {"large"},
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-fontstyle"] = {"large"},
        ["affiliation-space-after"] = "1pt",
        ["footer-style"] = "plain",
        ["footer-fontstyle"] = {"large"},
        ["footer-space-after"] = "1pt",
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.15\\textheight")},
        ["logo-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.1\\textheight")},
        ["vrule-width"] = "0.5in",
        ["vrule-align"] = "left",
        ["vrule-color"] = "blue",
        ["vrule-text-color"] = "white",
        ["vrule-text-fontstyle"] = {"bfseries", "Large"},
        ["vrule-text"] = "Add your text in vrule-text"
        }
      assign_value(themevals)
        
      return m
    end,
    ["plain"] = function (m)
      themevals = {
        ["elements"] = {
          pandoc.MetaInlines{pandoc.RawInline("latex","\\headerblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\titleblock")}, 
          pandoc.MetaInlines{pandoc.RawInline("latex","\\authorblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\affiliationblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\vfill")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\logoblock")},
          pandoc.MetaInlines{pandoc.RawInline("latex","\\footerblock")}
          },
        ["page-align"] = "left",
        ["title-style"] = "plain",
        ["title-fontstyle"] = {"Large"},
        ["title-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","4\\baselineskip")},
        ["title-subtitle-space-between"] = "1pt",
        ["subtitle-fontstyle"] = {"textit"},
        ["author-style"] = "superscript-with-and",
        ["author-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["affiliation-style"] = "numbered-list-with-correspondence",
        ["affiliation-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","2\\baselineskip")},
        ["header-style"] = "plain",
        ["header-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.2\\textheight")},
        ["footer-style"] = "plain",
        ["footer-space-after"] = "1pt",
        ["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.1\\textheight")},
        ["logo-space-after"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","1\\baselineskip")},
        }
      assign_value(themevals)
        
      return m
    end,
    ["none"] = function (m) return m end
  }
  
  m['titlepage-file'] = false
  if isEmpty(m.titlepage) then m['titlepage'] = "plain" end
  if getVal(m.titlepage) == "false" then m['titlepage'] = "none" end
  if getVal(m.titlepage) == "true" then m['titlepage'] = "plain" end
  if getVal(m.titlepage) == "none" then 
    m['titlepage-true'] = false
  else
    m['titlepage-true'] = true 
  end
  choice = pandoc.utils.stringify(m.titlepage)
  okvals = {"plain", "vline", "vline-text", "bg-image", "colorbox", "academic", "formal", "classic-lined"}
  isatheme = has_value (okvals, choice)
  if not isatheme and choice ~= "none" then
    if not file_exists(choice) then
      error("titlepage extension error: titlepage can be a tex file or one of the themes: " .. pandoc.utils.stringify(table.concat(okvals, ", ")) .. ".")
    else
      m['titlepage-file'] = true
      m['titlepage-filename'] = choice
      m['titlepage'] = "file"
    end
  end
  if m['titlepage-file'] and not isEmpty(m['titlepage-theme']) then
    print("\n\ntitlepage extension message: since you passed in a static titlepage file, titlepage-theme is ignored.n\n")
  end
  if not m['titlepage-file'] and choice ~= "none" then
    if isEmpty(m['titlepage-theme']) then
      m['titlepage-theme'] = {}
    end
    titlepage_table[choice](m) -- add the theme defaults
  end

-- Only for themes
-- titlepage-theme will exist if using a theme
if not m['titlepage-file'] and m['titlepage-true'] then
--[[
Error checking and setting the style codes
--]]
  -- Style codes
  m["titlepage-style-code"] = {}
  okvals = {"none", "plain", "colorbox", "doublelinewide", "doublelinetight"}
  set_style("titlepage", "title", okvals)
  set_style("titlepage", "footer", okvals)
  set_style("titlepage", "header", okvals)
  set_style("titlepage", "date", okvals)
  okvals = {"none", "plain", "plain-with-and", "superscript", "superscript-with-and", "two-column", "author-address"}
  set_style("titlepage", "author", okvals)
  okvals = {"none", "numbered-list", "numbered-list-with-correspondence"}
  set_style("titlepage", "affiliation", okvals)
  if is_equal(m['titlepage-theme']["author-style"], "author-address") and is_equal(m['titlepage-theme']["author-align"], "spread") then
    error("\n\nquarto_titlepages error: If author-style is two-column, then author-align cannot be spread.\n\n")
  end

--[[
Set the fontsize defaults
if page-fontsize was passed in or if fontsize passed in but not spacing
--]]
  for key, val in pairs({"title", "author", "affiliation", "footer", "header", "date"}) do
    if isEmpty(m["titlepage-theme"][val .. "-fontsize"]) then
      if not isEmpty(m["titlepage-theme"]["page-fontsize"]) then
        m["titlepage-theme"][val .. "-fontsize"] = getVal(m["titlepage-theme"]["page-fontsize"])
      end
    end
  end
  for key, val in pairs({"page", "title", "subtitle", "author", "affiliation", "footer", "header", "date"}) do
    if not isEmpty(m['titlepage-theme'][val .. "-fontsize"]) then
      if isEmpty(m['titlepage-theme'][val .. "-spacing"]) then
        m['titlepage-theme'][val .. "-spacing"] = 1.2*getVal(m['titlepage-theme'][val .. "-fontsize"])
      end
    end
  end

--[[
Set author sep character
--]]
  if isEmpty(m['titlepage-theme']["author-sep"]) then
    m['titlepage-theme']["author-sep"] = pandoc.MetaInlines{
          pandoc.RawInline("latex",", ")}
  end
  if getVal(m['titlepage-theme']["author-sep"]) == "newline" then
    m['titlepage-theme']["author-sep"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","\\\\")}
  end

--[[
Set affiliation sep character
--]]
  if isEmpty(m['titlepage-theme']["affiliation-sep"]) then
    m['titlepage-theme']["affiliation-sep"] = pandoc.MetaInlines{
          pandoc.RawInline("latex",",~")}
  end
  if getVal(m['titlepage-theme']["affiliation-sep"]) == "newline" then
    m['titlepage-theme']["affiliation-sep"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","\\\\")}
  end
  
--[[
Set vrule defaults
--]]
  if not isEmpty(m['titlepage-theme']["vrule-width"]) then
    if isEmpty(m['titlepage-theme']["vrule-color"]) then
      m['titlepage-theme']["vrule-color"] = "black"
    end
    if isEmpty(m['titlepage-theme']["vrule-space"]) then
      m['titlepage-theme']["vrule-space"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.05\\textwidth")}
    end
    if isEmpty(m['titlepage-theme']["vrule-align"]) then
      m['titlepage-theme']["vrule-align"] = "left"
    end
  end
  if not isEmpty(m["titlepage-theme"]["vrule-align"]) then
    okvals = {"left", "right", "leftright"}
    ok = check_yaml (m["titlepage-theme"]["vrule-align"], "titlepage-theme: vrule-align", okvals)
    if not ok then error("") end
  end

--[[
Set the defaults for the titlepage alignments
default titlepage alignment is left
--]]    
  if isEmpty(m['titlepage-theme']["page-align"]) then
    m['titlepage-theme']["page-align"] = "left"
  end
  for key, val in pairs({"page", "title", "author", "affiliation", "footer", "header", "logo", "date"}) do
    if not isEmpty(m["titlepage-theme"][val .. "-align"]) then
      okvals = {"right", "left", "center"}
      if has_value({"title", "author", "footer", "header"}, val) then table.insert(okvals, "spread") end
      ok = check_yaml (m["titlepage-theme"][val .. "-align"], "titlepage-theme: " .. val .. "-align", okvals)
      if not ok then error("") end
    end
  end
  
--[[
Set bg-image defaults
--]]
  if not isEmpty(m['titlepage-bg-image']) then
    if isEmpty(m['titlepage-theme']["bg-image-size"]) then
      m['titlepage-theme']["bg-image-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","\\paperwidth")}
    end
    if not isEmpty(m["titlepage-theme"]["bg-image-location"]) then
      okvals = {"ULCorner", "URCorner", "LLCorner", "LRCorner", "TileSquare", "Center"}
      ok = check_yaml (m["titlepage-theme"]["bg-image-location"], "titlepage-theme: bg-image-location", okvals)
      if not ok then error("") end
    end  
  end

--[[
Set logo defaults
--]]
  if not isEmpty(m['titlepage-logo']) then
    if isEmpty(m['titlepage-theme']["logo-size"]) then
      m['titlepage-theme']["logo-size"] = pandoc.MetaInlines{
          pandoc.RawInline("latex","0.2\\paperwidth")}
    end
  end
  
end -- end the theme section

  return m
  
end


