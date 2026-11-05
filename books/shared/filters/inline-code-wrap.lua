-- Preserve literal inline code while allowing PDF line breaks at its existing
-- spaces and punctuation. Pandoc's default \texttt uses nonbreaking escaped spaces,
-- which can push a tool call or dotted API name beyond a narrow book column.
function Code(code)
  if not FORMAT:match("latex") and not FORMAT:match("pdf") then
    return nil
  end
  -- Explicitly styled or anchored code keeps its writer-specific treatment.
  if code.identifier ~= "" or #code.classes > 0 then
    return nil
  end
  local document = pandoc.Pandoc({pandoc.Plain({code})})
  local latex = pandoc.write(document, "latex"):gsub("\n$", "")
  latex = latex:gsub("\\ ", " "):gsub("%.", ".\\allowbreak{}")
  latex = latex:gsub("\\_", "\\_\\allowbreak{}")
    :gsub("%(", "(\\allowbreak{}"):gsub("=", "=\\allowbreak{}")
  return pandoc.RawInline("latex", latex)
end
