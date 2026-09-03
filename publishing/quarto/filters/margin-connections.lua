-- margin-connections.lua

local pandoc = require "pandoc"

-- Helper: return TRUE if El.Classes contains the requested class
local function has_class(el, cls)
  for _, c in ipairs(el.classes) do
    if c == cls then
      return true
    end
  end
  return false
end

-- This is how Pandoc / Quarto calls the filter on every Div
function Div(el)
  -- 1) Check that this is your Callout
  if has_class(el, "callout-chapter-connection") then

    -- 2) If we generate PDF → marginpar
    if quarto.doc.is_format("pdf") then
      -- Cost all the text inside the block in the latex string
      local parts = {}
      for _, blk in ipairs(el.content) do
        if blk.t == "Para" or blk.t == "Plain"
        or blk.t == "BulletList" or blk.t == "OrderedList" then
          table.insert(parts,
            pandoc.write(pandoc.Pandoc({blk}), "latex")
          )
        end
      end
      local body = table.concat(parts, "\\\\[0.5ex]\n")  -- Add 0.5ex spacing between items
      -- Put together marginpar
      local m = string.format(
        "\\marginpar{\\footnotesize\\vspace*{1.8ex}\\par %s}",
        body
      )
      -- We return Rawblock and thus replace the entire Div
      return pandoc.RawBlock("latex", m)

    else
      -- 3) HTML → return the DIV unchanged (or let him custom-numbered-blocks to stylize
      if quarto.doc.is_format("html") then
        el.classes:insert("margin-chapter-connection")
        return pandoc.Div({el}, pandoc.Attr("", {"margin-container"}))
      end
    end
  end

  -- 4) Everything else let go without change
  return nil
end
