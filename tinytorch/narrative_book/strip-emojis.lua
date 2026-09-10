-- strip-emojis.lua
-- Pandoc filter: drop emoji codepoints from Str text when rendering to LaTeX/PDF.
-- XeLaTeX's default Latin Modern fonts don't cover the Emoji ranges; absent any
-- cross-platform emoji font (Apple Color Emoji on macOS vs Noto Color Emoji on
-- the CI Ubuntu image), the safest path is to strip them. The web rendering
-- (HTML target) is unaffected.
--
-- Ranges removed:
--   U+1F300-U+1FAFF  Miscellaneous Symbols & Pictographs, Emoticons, Transport,
--                    Supplemental Symbols & Pictographs, Symbols and Pictographs
--                    Extended-A (the bulk of decorative emojis)
--   U+2600-U+27BF    Miscellaneous Symbols (☆, ★, ✓, ✗, ➜ ...) and Dingbats
--                    that don't exist in Latin Modern; drop for safety.
--   U+200D           Zero-Width Joiner, used in compound emojis (🧑‍🚀).
--   U+FE0F           Variation Selector-16, makes preceding char render as emoji.

local emoji_bytes = {
  -- 4-byte UTF-8: U+10000 and above. Covers U+1F300-U+1FAFF via first byte 0xF0.
  "[\xF0-\xF4][\x80-\xBF][\x80-\xBF][\x80-\xBF]",
  -- 3-byte UTF-8 starting E2 98..9F: covers U+2600-U+27FF (symbols + dingbats).
  "\xE2[\x98-\x9F][\x80-\xBF]",
  -- Zero-Width Joiner U+200D = E2 80 8D
  "\xE2\x80\x8D",
  -- Variation Selector-16 U+FE0F = EF B8 8F
  "\xEF\xB8\x8F",
}

function Str(el)
  if not (FORMAT:match("latex") or FORMAT:match("pdf")) then
    return nil
  end
  local t = el.text
  for _, pat in ipairs(emoji_bytes) do
    t = t:gsub(pat, "")
  end
  if t ~= el.text then
    el.text = t
    return el
  end
  return nil
end
