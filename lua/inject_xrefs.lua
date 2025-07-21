-- lua/inject_xrefs.lua

-- This script is a Pandoc Lua filter that injects cross-references
-- into a Quarto document based on a JSON file.

-- It reads a JSON file where cross-references are grouped by source file,
-- then injects a "Chapter Connection" callout box into the appropriate sections.

local json = require("json")
local anpan = require("anpan")

-- Helper function to read file content
local function read_file(path)
  local file = io.open(path, "r")
  if not file then return nil end
  local content = file:read("*a")
  file:close()
  return content
end

-- Load and parse the cross-references JSON file
local function load_cross_references()
  local json_path = "scripts/cross_referencing/cross_references.json"
  local json_content = read_file(json_path)
  if not json_content then
    -- Don't raise a warning if the file simply doesn't exist
    return nil
  end
  
  local ok, data = pcall(json.decode, json_content)
  if not ok then
    -- Raise a warning if the file is invalid
    quarto.log.warning("Could not parse " .. json_path)
    return nil
  end
  return data
end

-- Store the loaded cross-references
local xrefs_data = load_cross_references()
if not xrefs_data then
  -- If there's no data, there's nothing to do
  return {}
end

-- Organize references by source section ID for quick lookup
local refs_by_source_id = {}
if xrefs_data and xrefs_data.cross_references then
  for source_file, refs in pairs(xrefs_data.cross_references) do
    for _, ref in ipairs(refs) do
      if not refs_by_source_id[ref.source_section_id] then
        refs_by_source_id[ref.source_section_id] = {}
      end
      table.insert(refs_by_source_id[ref.source_section_id], ref)
    end
  end
end

-- Function to create the markdown for the connection box
local function create_connection_box(refs)
  local previews = {}
  local foundations = {}
  
  -- Sort references by type
  for _, ref in ipairs(refs) do
    if ref.connection_type == "Preview" then
      table.insert(previews, ref)
    elseif ref.connection_type == "Foundation" then
      table.insert(foundations, ref)
    end
  end

  -- Don't create a box if there are no valid references
  if #previews == 0 and #foundations == 0 then
    return nil
  end

  local lines = {"::: {.callout-chapter-connection}", "##### Connections"}
  
  if #previews > 0 then
    table.insert(lines, "")
    table.insert(lines, "**Preview**")
    for _, ref in ipairs(previews) do
      -- Format: * **{Chapter}:** For more on *{Section}*, see @{id}.
      local chapter_name = ref.target_chapter_name:gsub("_", " "):gsub("%b()", "")
      chapter_name = chapter_name:gsub("^%s*%d*%.?%d*\\s*", "") -- remove leading numbers
      chapter_name = string.upper(string.sub(chapter_name, 1, 1)) .. string.sub(chapter_name, 2)

      table.insert(lines, 
        string.format("* **%s:** For more on *%s*, see `@%s`.", 
          chapter_name, 
          ref.target_section_title, 
          ref.target_section_id))
    end
  end
  
  if #foundations > 0 then
    table.insert(lines, "")
    table.insert(lines, "**Foundation**")
    for _, ref in ipairs(foundations) do
      -- Format: * **{Chapter}:** This builds on *{Section}* (`@{id}`).
      local chapter_name = ref.target_chapter_name:gsub("_", " "):gsub("%b()", "")
      chapter_name = chapter_name:gsub("^%s*%d*%.?%d*\\s*", "") -- remove leading numbers
      chapter_name = string.upper(string.sub(chapter_name, 1, 1)) .. string.sub(chapter_name, 2)
      
      table.insert(lines, 
        string.format("* **%s:** This builds on the principles of *%s* (`@%s`).", 
          chapter_name,
          ref.target_section_title, 
          ref.target_section_id))
    end
  end
  
  table.insert(lines, ":::")
  
  return table.concat(lines, "\n")
end

-- This is the main filter function called by Pandoc
return {
  ['Div'] = function(div)
    -- We are looking for sections, which are Divs with an ID
    if div.id and div.id ~= "" then
      local section_id = div.id
      
      -- Check if we have references for this section
      if refs_by_source_id[section_id] then
        local connection_box_md = create_connection_box(refs_by_source_id[section_id])
        
        if connection_box_md then
          -- Parse the markdown into Pandoc AST
          local new_content = anpan.parse(connection_box_md)
          
          -- Inject the connection box at the beginning of the section's content
          for i = #new_content.blocks, 1, -1 do
            table.insert(div.content, 1, new_content.blocks[i])
          end
          
          return div
        end
      end
    end
  end
} 