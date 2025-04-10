function Note (note)
  return pandoc.RawInline('latex', '\\sidenote{' .. pandoc.utils.stringify(note.content) .. '}')
end
