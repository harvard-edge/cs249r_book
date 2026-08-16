-- Pull a leading [offset=...] marker off a note, returning the value (or nil).
--
-- The marker is a PDF-only layout directive: it feeds \styledsidenote so the
-- copyeditor can nudge a sidenote clear of a figure or another note. It must be
-- removed in EVERY format, not only the one that consumes it. HTML and EPUB have
-- no margin to place the note in, and if the marker survives it is rendered to
-- the reader as literal text at the head of the footnote ("[offset=65mm] Total
-- Cost of Ownership (TCO): ..."). That was leaking 19 times across 8 chapters of
-- the Volume I web build before this was hoisted out of the PDF branch
-- (2026-08-16).
local function take_offset(note)
  if #note.content == 0 then
    return nil
  end
  local first_block = note.content[1]
  if not ((first_block.t == "Para" or first_block.t == "Plain") and #first_block.content > 0) then
    return nil
  end
  local first_inline = first_block.content[1]
  if first_inline.t ~= "Str" then
    return nil
  end
  local offset = first_inline.text:match("^%[offset=([^%]]+)%]")
  if not offset then
    return nil
  end

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

  return offset
end

function Note(note)
  -- Strip the layout directive first, for all formats (see take_offset).
  local offset = take_offset(note)

  -- Only convert footnotes to sidenotes for PDF/LaTeX output.
  -- ePub is HTML-based: pandoc.RawInline('latex', ...) nodes are ignored by
  -- the EPUB renderer, so the surrounding \sidenote{} delimiters are stripped
  -- while the note body is emitted inline — causing the sidenote text to
  -- appear embedded in the running prose.
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") then
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

  -- For ePub and all other formats, let Pandoc render the footnote normally --
  -- but return the note rather than nil, because take_offset above edited its
  -- content. Returning nil tells Pandoc "unchanged" and the stripped [offset=]
  -- marker comes back, which is exactly how it kept leaking into HTML.
  return note
end