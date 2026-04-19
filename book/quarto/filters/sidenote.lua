function Note (note)
  -- Only convert footnotes to sidenotes for PDF/LaTeX output.
  -- ePub is HTML-based: pandoc.RawInline('latex', ...) nodes are ignored by
  -- the EPUB renderer, so the surrounding \sidenote{} delimiters are stripped
  -- while the note body is emitted inline — causing the sidenote text to
  -- appear embedded in the running prose (see issue #1333).
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") then
    -- For PDF/LaTeX, convert footnote to sidenote for margin placement
    -- Requires sidenotes package (loaded in header-includes.tex)
    -- Fallback to \footnote is defined in header-includes.tex if package fails
    local sidenote_content = {}
    table.insert(sidenote_content, pandoc.RawInline('latex', '\\sidenote{'))

    -- Add the note content directly as inlines (not converted to latex yet)
    -- This allows citations to be processed by citeproc later
    for _, block in ipairs(note.content) do
      if block.t == "Para" or block.t == "Plain" then
        for _, inline in ipairs(block.content) do
          table.insert(sidenote_content, inline)
        end
        -- Add space between paragraphs
        table.insert(sidenote_content, pandoc.Space())
      end
    end

    table.insert(sidenote_content, pandoc.RawInline('latex', '}'))
    return sidenote_content
  end
  -- For ePub and all other formats, let Pandoc render footnotes normally.
  return nil
end
