function Note(note)
  -- Only convert footnotes to sidenotes for PDF/LaTeX output.
  -- ePub is HTML-based: pandoc.RawInline('latex', ...) nodes are ignored by
  -- the EPUB renderer, so the surrounding \sidenote{} delimiters are stripped
  -- while the note body is emitted inline — causing the sidenote text to
  -- appear embedded in the running prose.
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") then
    local offset = nil

    -- Detect optional [offset=...] marker at the start of the first Para/Plain block
    if #note.content > 0 then
      local first_block = note.content[1]
      if (first_block.t == "Para" or first_block.t == "Plain") and #first_block.content > 0 then
        local first_inline = first_block.content[1]
        if first_inline.t == "Str" then
          local m = first_inline.text:match("^%[offset=([^%]]+)%]")
          if m then
            offset = m

            -- Remove only the [offset=...] part, keep the rest of the text
            first_inline.text = first_inline.text:gsub("^%[offset=[^%]]+%]", "")

            -- If the inline becomes empty, remove it
            if first_inline.text == "" then
              table.remove(first_block.content, 1)
            end

            -- Remove following space if present
            if #first_block.content > 0 and first_block.content[1].t == "Space" then
              table.remove(first_block.content, 1)
            end
          end
        end
      end
    end

    local out = {}

    if offset then
      table.insert(out, pandoc.RawInline('latex', '\\styledsidenote[][' .. offset .. ']{'))
    else
      table.insert(out, pandoc.RawInline('latex', '\\sidenote{'))
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

  -- For ePub and all other formats, let Pandoc render footnotes normally.
  return nil
end