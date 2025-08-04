-- Margin Video Filter for MLSysBook
-- Converts {.margin-video} divs to appropriate HTML/PDF output

function Div(el)
  if el.classes:includes("margin-video") then
    local url = el.attr.attributes["url"] or ""
    local title = el.attr.attributes["title"] or "Video"
    local author = el.attr.attributes["author"] or ""
    
    -- Extract YouTube video ID for HTML
    local video_id = string.match(url, "youtube%.com/watch%?v=([%w_-]+)")
    if not video_id then
      video_id = string.match(url, "youtu%.be/([%w_-]+)")
    end
    
    if FORMAT:match("html") then
      -- HTML: Margin video with auto-numbering
      local caption = title
      if author ~= "" then
        caption = caption .. " - " .. author
      end
      
      local html_output = [[
<div class="column-margin">
  <div class="margin-video">
    <iframe src="https://www.youtube.com/embed/]] .. video_id .. [[" 
            style="width: 100%; height: auto; aspect-ratio: 16/9; border: 0; border-radius: 6px;" 
            allowfullscreen>
    </iframe>
  </div>
  <p><em>]] .. caption .. [[</em></p>
</div>
]]
      
      return pandoc.RawBlock("html", html_output)
      
    elseif FORMAT:match("latex") or FORMAT:match("pdf") then
      -- PDF: QR code and margin note
      local latex_output = [[
\marginnote{\centering\\\vspace*{5mm}%
  \parbox{30mm}{\centering\footnotesize%
    \textbf{Watch on YouTube}\\
    ]] .. title .. [[\\[1mm]
  }
  \begingroup
    \hypersetup{urlcolor=black}
    \qrcode[height=15mm]{]] .. url .. [[}
  \endgroup
\\[1mm]
  \parbox{25mm}{\centering\footnotesize%
    Scan with your phone\\
    to watch the video
  }
}

\faTv{} \href{]] .. url .. [[}{Watch on YouTube}
]]
      
      return pandoc.RawBlock("latex", latex_output)
    end
  end
  
  return el
end