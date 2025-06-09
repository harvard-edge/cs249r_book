---@meta

---@class CustomCalloutDefinition
---@field type string The type/identifier of the callout
---@field title string|pandoc.Inlines|nil The title of the callout
---@field icon boolean|nil Whether to show an icon
---@field appearance string|nil The appearance style ('default', 'minimal', 'simple')
---@field collapse boolean|string|nil Whether the callout is collapsible
---@field icon_symbol string|nil Custom icon symbol or font awesome icon
---@field color string|nil The color of the callout
---@field background_color string|nil The background color of the callout

---@class CustomCalloutsMap

-- Global variable to store custom callout definitions
---@type table<string, CustomCalloutDefinition>
local customCallouts = {}

local fa = require("fa")

---Converts a valid CSS color string or hexadecimal to RGBA format
---@param color string The color in hex (#RRGGBB) or named format
---@param alpha number The alpha value between 0 and 1
---@return string rgba The color in rgba() or rgb(from color) format
local function colorToRgba(color, alpha)
  if color:sub(1,1) == "#" then
    local r = tonumber(color:sub(2,3), 16)
    local g = tonumber(color:sub(4,5), 16)
    local b = tonumber(color:sub(6,7), 16)
    return string.format("rgba(%d, %d, %d, %.2f)", r, g, b, alpha)
  else
    -- For named colors, we use the functional notation of rgba()
    return string.format("rgb(from %s r g b / %.0f%%)", color, alpha * 100)
  end
end

---Checks if a string represents a Font Awesome icon
---@param icon string|nil The icon string to check
---@return boolean is_fa True if the string starts with "fa-"
local function isFontAwesomeIcon(icon)
  return icon ~= nil and icon:sub(1, 3) == "fa-"
end

---Generates CSS for all defined custom callouts
---@return string css The generated CSS rules
local function generateCustomCSS()
  local css = ""

  -- Translate YAML callout information for custom callouts
  for type, callout in pairs(customCallouts) do
    if callout.color then
      local color = pandoc.utils.stringify(callout.color)
      
      -- Base color
      css = css .. string.format("div.callout-%s.callout {\n", type)
      css = css .. string.format("  border-left-color: %s;\n", color)
      css = css .. "}\n"
      
      -- Header background
      css = css .. string.format("div.callout-%s.callout-style-default > .callout-header {\n", type)
      css = css .. string.format("  background-color: %s;\n", colorToRgba(color, 0.13))
      css = css .. "}\n"

      -- Collapse Icon
      css = css .. string.format("div.callout-%s .callout-toggle::before {", type)
      css = css .. "  background-image: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" fill=\"rgb(33, 37, 41)\" class=\"bi bi-chevron-down\" viewBox=\"0 0 16 16\"><path fill-rule=\"evenodd\" d=\"M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z\"/></svg>');"
      css = css .. "}\n"

      -- Icon Styling
      css = css .. string.format("div.callout-%s.callout-style-default .callout-icon::before, div.callout-%s.callout-titled .callout-icon::before {\n", type, type)

      if callout.icon_symbol then
        local icon_symbol_str = pandoc.utils.stringify(callout.icon_symbol)
        if isFontAwesomeIcon(icon_symbol_str) then
          -- Font Awesome icon
          css = css .. "  font-family: 'Font Awesome 6 Free', FontAwesome;\n"
          css = css .. "  font-style: normal;\n"
          css = css .. string.format("  content: '%s' !important;\n", fa.fa_unicode(icon_symbol_str))
        else 
          -- Custom icon symbol
          css = css .. string.format("  content: '%s';\n", icon_symbol_str)
        end
        css = css .. "  background-image: none;\n"
      else 
        -- The fallback case
        local escapedColor = color:gsub("#", "%%23")  -- Escape # in hex colors
        css = css .. string.format("  background-image: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" fill=\"%s\" class=\"bi bi-exclamation-triangle\" viewBox=\"0 0 16 16\"><path d=\"M7.938 2.016A.13.13 0 0 1 8.002 2a.13.13 0 0 1 .063.016.146.146 0 0 1 .054.057l6.857 11.667c.036.06.035.124.002.183a.163.163 0 0 1-.054.06.116.116 0 0 1-.066.017H1.146a.115.115 0 0 1-.066-.017.163.163 0 0 1-.054-.06.176.176 0 0 1 .002-.183L7.884 2.073a.147.147 0 0 1 .054-.057zm1.044-.45a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566z\"/></svg>');\n", escapedColor)
      end 

      css = css .. "}\n"

    end
  end
  return css
end


---Parses custom callout definitions from document metadata
---@param meta pandoc.Meta The document metadata
local function parseCustomCallouts(meta)
  if not meta['custom-callout'] then return end

  for k, v in pairs(meta['custom-callout']) do
    if type(v) == "table" then
      customCallouts[k] = {
        type = tostring(k),
        title = v.title or k:gsub("^%l", string.upper),
        icon = v.icon == 'true' or nil,
        appearance = v.appearance or nil,
        collapse = v.collapse or nil,
        icon_symbol = v['icon-symbol'] or nil,
        color = v.color or nil,
        background_color = v['background-color'] or nil
      }
    end
  end


  -- Generate and add custom CSS to the document
  local customCSS = generateCustomCSS()
  if customCSS ~= "" then
    quarto.doc.include_text('in-header', '<style>\n' .. customCSS .. '</style>')
  end

end


---Converts a div to a custom callout if it matches a defined custom callout
---@param div pandoc.Div The div to potentially convert
---@return pandoc.Div|quarto.Callout converted The converted callout or original div
local function convertToCustomCallout(div)
  -- Check if the div has classes
  for _, class in ipairs(div.classes) do
    
    -- Check if the class matches a custom callout
    local callout = customCallouts[class]

    if callout then 
      -- Use the default title if not provided
      local title = callout.title

      -- Check to see if the title is specified in the div content
      if div.content[1] ~= nil and div.content[1].t == "Header" then
        title = div.content[1]
        div.content:remove(1)
      end

      -- Create a new Callout with the custom callout parameters
      local calloutParams = {
        type = callout.type,
        content = div.content,
        title = div.attributes.title or title,
        icon = div.attributes.icon or callout.icon,
        appearance = div.attributes.appearance or callout.appearance,
        collapse = div.attributes.collapse or callout.collapse
      }
      
      return quarto.Callout(calloutParams)
    end
  end
  

  return div
end

---Walks the Pandoc document and processes divs to
---convert to custom callouts
---@class pandoc.Doc
---@field blocks pandoc.Blocks
---@param doc pandoc.Doc The Pandoc document
---@return pandoc.Doc doc The processed document
local function customCalloutFilter(doc)

  -- Walk the AST and process divs
  doc.blocks = doc.blocks:walk({
    Div = convertToCustomCallout
  })
  
  -- Return the modified document
  return doc
end

-- Return the Pandoc filter
return {
  ---@type fun(meta: pandoc.Meta)
  Meta = parseCustomCallouts,
  ---@type fun(doc: pandoc.Doc): pandoc.Doc
  Pandoc = customCalloutFilter
}
