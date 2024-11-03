function Header(el)
  -- Check if the output format is LaTeX and if the header is a chapter (level 1)
  if FORMAT == "latex" and el.level == 1 then
    -- Insert the \chaptertoc macro after the chapter heading
    local chapter_toc = pandoc.RawBlock("latex", "\\chaptertoc")
    -- Return the header followed by the \chaptertoc block
    return {el, chapter_toc}
  end
  return el
end