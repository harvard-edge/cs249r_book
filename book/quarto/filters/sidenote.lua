function Note (note)
  -- Only process for latex, pdf, or epub formats
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") or quarto.doc.is_format("epub") then
    -- For PDF/LaTeX, convert footnote to sidenote with content inline
    -- We need to construct this carefully to preserve citation processing
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
  -- For other formats, return the content unchanged
  return nil
end
