-- Video insertion shortcode for MLSysBook
-- Usage: {{< margin-video "URL" "Title" "Author" >}}

-- Helper function for formatted logging
local function log_info(message)
  io.stderr:write("📹 [Margin Video Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_success(message)
  io.stderr:write("✅ [Margin Video Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_warning(message)
  io.stderr:write("⚠️  [Margin Video Filter] " .. message .. "\n")
  io.stderr:flush()
end

local function log_error(message)
  io.stderr:write("❌ [Margin Video Filter] " .. message .. "\n")
  io.stderr:flush()
end

return {
  ['margin-video'] = function(args, kwargs, meta)
    -- Shortcode is disabled - returns nothing
    return pandoc.Null()
  end,

  ['margin-video-DISABLED'] = function(args, kwargs, meta)
    -- Original implementation (disabled)
    -- Validate arguments
    if not args[1] then
      log_error("No URL argument provided")
      error("ERROR: margin-video requires at least a URL argument.\nUsage: {{< margin-video \"URL\" \"Title\" \"Author\" >}}")
    end

    local url = pandoc.utils.stringify(args[1]) or ""
    local title = pandoc.utils.stringify(args[2]) or "Video"
    local author = pandoc.utils.stringify(args[3]) or ""

    log_info("URL: " .. url)
    log_info("Title: " .. title)
    log_info("Author: " .. author)

    -- Optional configuration via kwargs
    local aspect_ratio = pandoc.utils.stringify(kwargs["aspect-ratio"]) or "16/9"
    local autoplay = pandoc.utils.stringify(kwargs["autoplay"]) == "true"
    local start_time = pandoc.utils.stringify(kwargs["start"]) or nil

    log_info("Aspect ratio: " .. aspect_ratio)
    log_info("Autoplay: " .. tostring(autoplay))
    log_info("Start time: " .. (start_time or "none"))

    -- Validate URL is not empty
    if url == "" then
      log_error("URL is empty")
      error("ERROR: margin-video URL cannot be empty.\nUsage: {{< margin-video \"URL\" \"Title\" \"Author\" >}}")
    end

    -- Check if it's a YouTube URL with better validation
    if not (string.match(url, "youtube%.com") or string.match(url, "youtu%.be")) then
      log_error("Non-YouTube URL provided: " .. url)
      error("ERROR: margin-video currently only supports YouTube URLs.\nGot: " .. url .. "\nSupported formats:\n  - https://www.youtube.com/watch?v=VIDEO_ID\n  - https://youtu.be/VIDEO_ID")
    end

    -- Extract YouTube video ID (handles various URL formats and parameters)
    local video_id = nil

    log_info("Extracting video ID from URL...")

    -- Handle youtube.com/watch?v=ID format (with optional additional parameters)
    video_id = string.match(url, "youtube%.com/watch%?.*v=([%w_-]+)")
    if video_id then
      log_success("Extracted video ID from youtube.com/watch format: " .. video_id)
    end

    -- Handle youtu.be/ID format (with optional parameters)
    if not video_id then
      video_id = string.match(url, "youtu%.be/([%w_-]+)")
      if video_id then
        log_success("Extracted video ID from youtu.be format: " .. video_id)
      end
    end

    -- Handle youtube.com/embed/ID format
    if not video_id then
      video_id = string.match(url, "youtube%.com/embed/([%w_-]+)")
      if video_id then
        log_success("Extracted video ID from youtube.com/embed format: " .. video_id)
      end
    end

    if not video_id then
      log_error("Could not extract video ID from URL: " .. url)
      error("ERROR: Could not extract YouTube video ID from URL: " .. url .. "\nPlease check the URL format is correct.")
    end

    if FORMAT:match("html") then
      log_info("Generating HTML output...")
      -- HTML: Margin video with auto-numbering
      local caption = title
      if author ~= "" then
        caption = caption .. " - " .. author
      end
      log_info("Caption: " .. caption)

      -- Build iframe URL with optional parameters
      local iframe_url = "https://www.youtube.com/embed/" .. video_id
      local url_params = {}

      if autoplay then
        table.insert(url_params, "autoplay=1")
      end

      if start_time then
        table.insert(url_params, "start=" .. start_time)
      end

      if #url_params > 0 then
        iframe_url = iframe_url .. "?" .. table.concat(url_params, "&")
      end

      local html_output = [[
<div class="column-margin">
  <div class="margin-video">
    <iframe src="]] .. iframe_url .. [["
            style="width:100%; aspect-ratio: ]] .. aspect_ratio .. [[; border:0;"
            allowfullscreen>
    </iframe>
  </div>
  <p><em>]] .. caption .. [[</em></p>
</div>
]]
      log_success("HTML output generated successfully")
      return pandoc.RawBlock("html", html_output)
    elseif FORMAT:match("pdf") or FORMAT:match("latex") then
      log_info("Generating PDF output...")
      -- PDF: QR code and margin note with brand-aligned styling
      local pdf_output = [[
\marginnote{\centering\\\vspace*{5mm}%
  \begin{tcolorbox}[
    enhanced,
    colback=white,
    colframe=callout-resource-videos-color2,
    boxrule=0.5pt,
    arc=1.5pt,
    width=32mm,
    left=1mm,
    right=1mm,
    top=0mm,
    bottom=1mm,
    before skip=0pt,
    after skip=0pt,
    attach boxed title to top*={xshift=0pt},
    boxed title style={
      colback=callout-resource-videos-color1,
      colframe=callout-resource-videos-color2,
      arc=1.5pt,
      rounded corners=north,
      sharp corners=south,
      boxrule=0.5pt,
      top=1mm,
      bottom=1mm,
      left=1mm,
      right=1mm,
    },
    title=\centering\footnotesize\color{callout-resource-videos-color2}\textbf{\raisebox{-0.3mm}{\includegraphics[width=2mm]{assets/images/icons/callouts/icon_callout-resource-videos.pdf}}\hspace{1mm}Video Resource}
  ]
    \vspace{1.5mm}
    \centering\footnotesize%
    \textbf{]] .. title .. [[}\\
    ]] .. (author ~= "" and "\\textcolor{black!60}{" .. author .. "}\\\\[2mm]" or "[2mm]") .. [[
    \begingroup
      \hypersetup{urlcolor=black}
      \qrcode[height=14mm]{]] .. url .. [[}
    \endgroup
    \\[2mm]
    \textcolor{black!60}{\scriptsize Scan with your phone\\to watch the video}
  \end{tcolorbox}
}
]]
      log_success("PDF output generated successfully")
      return pandoc.RawBlock("latex", pdf_output)
    else
      log_warning("Unknown format: " .. FORMAT .. " - using fallback link")
      -- Fallback for other formats (e.g., just a link)
      return pandoc.Link(pandoc.Str(title), url)
    end
  end
}
