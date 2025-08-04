-- Enhanced Video Shortcode for MLSysBook
-- Provides better video formatting with metadata and styling options

function Div(el)
  if el.classes:includes("video-enhanced") then
    local video_url = el.attr.attributes["url"] or ""
    local title = el.attr.attributes["title"] or "Video"
    local duration = el.attr.attributes["duration"] or ""
    local author = el.attr.attributes["author"] or ""
    local description = el.attr.attributes["description"] or ""
    
    -- Extract YouTube video ID
    local video_id = string.match(video_url, "youtube%.com/watch%?v=([%w_-]+)")
    if not video_id then
      video_id = string.match(video_url, "youtu%.be/([%w_-]+)")
    end
    
    if video_id then
      local embed_url = "https://www.youtube.com/embed/" .. video_id
      
      local html_output = [[
<div class="video-enhanced-container" style="margin: 2rem 0;">
  <div class="video-header" style="margin-bottom: 1rem;">
    <h4 style="margin: 0; color: #333; font-weight: 600;">]] .. title .. [[</h4>
    ]] .. (duration ~= "" and '<span style="font-size: 0.9em; color: #666;"><i class="fas fa-clock"></i> ' .. duration .. '</span>' or "") .. [[
    ]] .. (author ~= "" and '<span style="font-size: 0.9em; color: #666; margin-left: 1rem;"><i class="fas fa-user"></i> ' .. author .. '</span>' or "") .. [[
  </div>
  <div class="video-wrapper" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 12px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
    <iframe src="]] .. embed_url .. [[" 
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
            allowfullscreen>
    </iframe>
  </div>
  ]] .. (description ~= "" and '<div class="video-description" style="margin-top: 0.5rem; font-size: 0.9em; color: #666; font-style: italic;">' .. description .. '</div>' or "") .. [[
</div>
]]
      
      return pandoc.RawBlock("html", html_output)
    end
  end
  return el
end 