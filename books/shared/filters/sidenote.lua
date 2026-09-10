function Note(note)
  local offset = nil

  -- The offset marker is source metadata for print placement, not note text.
  -- Strip it for every output format so it cannot leak into HTML or EPUB.
  -- PDF/LaTeX uses the captured value below when constructing the sidenote.
  if #note.content > 0 then
    local first_block = note.content[1]
    if (first_block.t == "Para" or first_block.t == "Plain") and #first_block.content > 0 then
      local first_inline = first_block.content[1]
      if first_inline.t == "Str" then
        local m = first_inline.text:match("^%[offset=([^%]]+)%]")
        if m then
          offset = m

          first_inline.text = first_inline.text:gsub("^%[offset=[^%]]+%]", "")
          if first_inline.text == "" then
            table.remove(first_block.content, 1)
          end
          if #first_block.content > 0 and first_block.content[1].t == "Space" then
            table.remove(first_block.content, 1)
          end
        end
      end
    end
  end

  -- Only convert footnotes to sidenotes for PDF/LaTeX output.
  -- ePub is HTML-based: pandoc.RawInline('latex', ...) nodes are ignored by
  -- the EPUB renderer, so the surrounding \sidenote{} delimiters are stripped
  -- while the note body is emitted inline — causing the sidenote text to
  -- appear embedded in the running prose.
  local is_print = false
  if quarto and quarto.doc and quarto.doc.is_format then
    is_print = quarto.doc.is_format("latex") or quarto.doc.is_format("pdf")
  else
    is_print = FORMAT == "latex" or FORMAT == "beamer"
  end

  if is_print then
    local out = {}

    if offset then
      table.insert(out, pandoc.RawInline('latex', '\\styledsidenote[][' .. offset .. ']{'))
    else
      table.insert(out, pandoc.RawInline('latex', '\\sidenote{'))
    end

    -- Color the first Strong (bold headword) in accentcolor so readers can
    -- scan the margin for term names without reading every definition.
    local colored_first_strong = false
    for _, block in ipairs(note.content) do
      if block.t == "Para" or block.t == "Plain" then
        if not colored_first_strong then
          for i, inline in ipairs(block.content) do
            if inline.t == "Strong" and not colored_first_strong then
              -- Wrap the Strong's content in \textcolor{accentcolor}{...}
              local wrapped = pandoc.List({})
              wrapped:insert(pandoc.RawInline('latex', '\\textcolor{accentcolor}{'))
              wrapped:insert(inline)
              wrapped:insert(pandoc.RawInline('latex', '}'))
              -- Replace the Strong node with the wrapped inlines
              block.content[i] = pandoc.Span(wrapped)
              colored_first_strong = true
              break
            end
          end
        end
      end
    end

    -- Add the note content directly as inlines (not converted to latex yet)
    -- This allows citations to be processed by citeproc later
    for _, block in ipairs(note.content) do
      if block.t == "Para" or block.t == "Plain" then
        for _, inline in ipairs(block.content) do
          table.insert(out, inline)
        end
        -- Add space between paragraphs
        table.insert(out, pandoc.Space())
      end
    end

    table.insert(out, pandoc.RawInline('latex', '}'))
    return out
  end

  -- For EPUB and all other formats, render the ordinary note after removing
  -- the print-only placement metadata.
  return note
end
