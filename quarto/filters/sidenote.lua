function Note (note)
  -- Only process for latex, pdf, or epub formats
  if quarto.doc.is_format("latex") or quarto.doc.is_format("pdf") or quarto.doc.is_format("epub") then
    return pandoc.RawInline('latex', '\\sidenote{' .. pandoc.utils.stringify(note.content) .. '}')
  end
  -- For other formats, return the content unchanged
  return nil
end
