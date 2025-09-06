function Note (note)
  -- Only process for latex, pdf, or epub formats
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") or quarto.doc.is_format("epub") then
    -- Convert content to LaTeX while preserving formatting
    local content = pandoc.write(pandoc.Pandoc(note.content), 'latex')
    return pandoc.RawInline('latex', '\\sidenote{' .. content .. '}')
  end
  -- For other formats, return the content unchanged
  return nil
end
