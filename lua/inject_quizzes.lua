-- inject_quizzes.lua

local json  = require("pandoc.json")
local utils = pandoc.utils

-- global state: all quiz sections are collected here, organized by file source
local quiz_sections = {}
local quiz_sections_by_file = {} -- track which sections came from which file
local current_document_file = "unknown" -- track which document is being processed

-- helper: checks if output is PDF/LaTeX
local function is_pdf()
  return FORMAT
    and (FORMAT:lower():match("pdf") or FORMAT:lower():match("latex"))
end

-- 1) Load a single JSON file
local function load_quiz_data(path)
  local f, err = io.open(path, "r")
  if not f then
    io.stderr:write("❌ [QUIZ] Failed to open file: " .. path .. " - " .. (err or "unknown error") .. "\n")
    return nil
  end
  local content = f:read("*all")
  f:close()

  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    io.stderr:write("❌ [QUIZ] Failed to parse JSON from file: " .. path .. "\n")
    return nil
  end
  return data
end

-- 2) Extract sections from JSON and map them to section_id
local function register_sections(data, file_path)
  local secs = {}
  local sections_found = 0
  
  if data.sections then
    for _, s in ipairs(data.sections) do
      if s.quiz_data
         and s.quiz_data.quiz_needed
         and s.quiz_data.questions then
        secs[s.section_id] = s.quiz_data.questions
        sections_found = sections_found + 1
      end
    end
  else
    for sid, qs in pairs(data) do
      if sid ~= "metadata" then
        secs[sid] = qs
        sections_found = sections_found + 1
      end
    end
  end
  
  return secs, sections_found
end

-- 3) Render content_md into a callout Div for all formats
local function create_quiz_div(div_id, div_class, content_md)
  -- parse the Markdown content
  local doc = pandoc.read(content_md, "markdown")

  -- return a callout Div with proper attributes
  return pandoc.Div(
    doc.blocks,
    pandoc.Attr(div_id, {"callout", div_class}, {})
  )
end

-- 4) From a given set of questions, create question + answer divs
local function process_quiz_questions(questions, section_id)
  local ql, al = {}, {}
  local clean = section_id:gsub("^#", "")
  local qid   = "quiz-question-" .. clean
  local aid   = "quiz-answer-" .. clean

  for i, q in ipairs(questions) do
    table.insert(ql, i..". "..q.question)
    if q.question_type == "MCQ" and q.choices then
      local choices_line = {}
      for j, c in ipairs(q.choices) do
        table.insert(choices_line, "   "..string.char(96+j)..") "..c)
      end
      table.insert(ql, table.concat(choices_line, "\n"))
    end
    -- build answers
    table.insert(al, i..". **"..q.question.."**")
    if q.question_type == "MCQ" and q.choices then
      local choices_line = {}
      for j, c in ipairs(q.choices) do
        table.insert(choices_line, "   "..string.char(96+j)..") "..c)
      end
      table.insert(al, table.concat(choices_line, "\n"))
      table.insert(al, "")
    end
    table.insert(al, "   *Answer*: "..q.answer)
    table.insert(al, "")
    table.insert(al, "   *Learning Objective*: "..q.learning_objective)
    table.insert(al, "")
  end

  table.insert(ql, "\n[See Answers →](#" .. aid .. ")")
  table.insert(al, "\n[← Back to Questions](#" .. qid .. ")")

  return create_quiz_div(qid, "callout-quiz-question", table.concat(ql, "\n\n")),
         create_quiz_div(aid, "callout-quiz-answer",   table.concat(al, "\n\n"))
end

-- 5) Meta phase: read one or more paths from meta.quiz
local function handle_meta(meta)
  local raw = meta.quiz
  
  -- Try to get the current document filename
  if PANDOC_DOCUMENT and PANDOC_DOCUMENT.meta and PANDOC_DOCUMENT.meta.filename then
    current_document_file = utils.stringify(PANDOC_DOCUMENT.meta.filename)
  elseif PANDOC_STATE and PANDOC_STATE.input_files and PANDOC_STATE.input_files[1] then
    current_document_file = PANDOC_STATE.input_files[1]
  end

  -- Check if this is a PDF build (combined document) or HTML build (individual files)
  local is_pdf_build = false
  if PANDOC_STATE and PANDOC_STATE.input_files and #PANDOC_STATE.input_files > 1 then
    is_pdf_build = true
  end
  
  -- Alternative PDF detection: check if we're in a PDF format
  if FORMAT and (FORMAT:lower():match("pdf") or FORMAT:lower():match("latex")) then
    is_pdf_build = true
  end

  -- Get quiz configuration from global metadata
  local quiz_config = meta["quiz-config"] or {}
  local file_pattern = quiz_config["file-pattern"] or "*_quizzes.json"
  local scan_directory = quiz_config["scan-directory"] or "contents/core"
  local auto_discover_pdf = quiz_config["auto-discover-pdf"] ~= false -- default to true
  
  -- Convert Meta objects to strings if needed
  if type(file_pattern) == "table" and file_pattern.t == "MetaString" then
    file_pattern = file_pattern.text
  elseif type(file_pattern) == "table" then
    file_pattern = utils.stringify(file_pattern)
  end
  
  if type(scan_directory) == "table" and scan_directory.t == "MetaString" then
    scan_directory = scan_directory.text
  elseif type(scan_directory) == "table" then
    scan_directory = utils.stringify(scan_directory)
  end
  
  if type(auto_discover_pdf) == "table" and auto_discover_pdf.t == "MetaBool" then
    auto_discover_pdf = auto_discover_pdf.bool
  elseif type(auto_discover_pdf) == "table" then
    auto_discover_pdf = utils.stringify(auto_discover_pdf) ~= "false"
  end

  io.stderr:write("🔍 Number of input files: " .. (PANDOC_STATE and PANDOC_STATE.input_files and #PANDOC_STATE.input_files or "unknown") .. "\n")
  io.stderr:write("🔍 Is PDF build: " .. tostring(is_pdf_build) .. "\n")
  io.stderr:write("🔍 Quiz metadata: " .. tostring(raw) .. "\n")
  io.stderr:write("🔍 FORMAT: " .. tostring(FORMAT) .. "\n")
  io.stderr:write("🔍 Current document file: " .. current_document_file .. "\n")
  io.stderr:write("🔍 Quiz config - file pattern: " .. file_pattern .. "\n")
  io.stderr:write("🔍 Quiz config - scan directory: " .. scan_directory .. "\n")
  io.stderr:write("🔍 Quiz config - auto discover PDF: " .. tostring(auto_discover_pdf) .. "\n")

  if not raw then
    if is_pdf_build and auto_discover_pdf then
      -- For PDF builds, auto-discover quiz files from the input files
      io.stderr:write("\n" .. string.rep("=", 80) .. "\n")
      io.stderr:write("📄 [QUIZ] Processing PDF Book Document - Auto-discovering quiz files\n")
      io.stderr:write(string.rep("=", 80) .. "\n")
      
      local files = {}
      local total_sections_loaded = 0
      local successful_files = 0
      
      -- Since Quarto combines files into a temporary document, we need to scan the directory directly
      local function scan_for_quiz_files()
        local quiz_files = {}
        
        -- Use the configured directory and pattern
        local find_command = string.format("find %s -name '%s' 2>/dev/null", scan_directory, file_pattern)
        io.stderr:write("🔍 Running command: " .. find_command .. "\n")
        
        local dir = io.popen(find_command)
        if dir then
          for file in dir:lines() do
            table.insert(quiz_files, file)
          end
          dir:close()
        end
        
        return quiz_files
      end
      
      files = scan_for_quiz_files()
      
      if #files > 0 then
        io.stderr:write("📁 Found " .. #files .. " quiz file(s) to process for this document\n\n")
        
        for i, path in ipairs(files) do
          io.stderr:write("📄 [" .. i .. "/" .. #files .. "] Loading quiz file: " .. path .. "\n")
          
          local data = load_quiz_data(path)
          if data then
            local secs, sections_found = register_sections(data, path)
            if sections_found > 0 then
              -- Track which sections came from this file
              quiz_sections_by_file[path] = {}
              for k, v in pairs(secs) do
                quiz_sections[k] = v
                quiz_sections_by_file[path][k] = v
              end
              total_sections_loaded = total_sections_loaded + sections_found
              successful_files = successful_files + 1
              io.stderr:write("   ✅ Loaded " .. sections_found .. " quiz section(s)\n")
            else
              io.stderr:write("   ⚠️  No quiz sections found in file\n")
            end
          else
            io.stderr:write("   ❌ Failed to load file\n")
          end
          io.stderr:write("\n")
        end
        
        io.stderr:write("📊 Quiz File Loading Summary for PDF Book:\n")
        io.stderr:write("   • Files processed: " .. successful_files .. "/" .. #files .. " ✅\n")
        io.stderr:write("   • Total quiz sections loaded: " .. total_sections_loaded .. " 📝\n")
        io.stderr:write(string.rep("-", 80) .. "\n")
      else
        io.stderr:write("📁 No quiz files found in " .. scan_directory .. " directory\n")
        io.stderr:write(string.rep("-", 80) .. "\n")
      end
    else
      -- For HTML builds, no quiz metadata means no quizzes needed
      io.stderr:write("ℹ️  [QUIZ] No quiz metadata found for document: " .. current_document_file .. "\n")
    end
    return meta
  end

  io.stderr:write("\n" .. string.rep("=", 80) .. "\n")
  if is_pdf_build then
    io.stderr:write("📄 [QUIZ] Processing PDF Book Document\n")
  else
    io.stderr:write("📄 [QUIZ] Processing HTML Document: " .. current_document_file .. "\n")
  end
  io.stderr:write(string.rep("=", 80) .. "\n")

  -- collect all files into this list
  local files = {}

  -- if raw is a table with elements, treat it as a list
  if type(raw) == "table" and raw[1] ~= nil then
    for _, item in ipairs(raw) do
      local p = utils.stringify(item)
      table.insert(files, p)
    end
  else
    -- not a list, so it's a single file
    local p = utils.stringify(raw)
    table.insert(files, p)
  end

  io.stderr:write("📁 Found " .. #files .. " quiz file(s) to process for this document\n\n")

  -- load each file individually with detailed reporting
  local total_sections_loaded = 0
  local successful_files = 0
  
  for i, path in ipairs(files) do
    io.stderr:write("📄 [" .. i .. "/" .. #files .. "] Loading quiz file: " .. path .. "\n")
    
    local data = load_quiz_data(path)
    if data then
      local secs, sections_found = register_sections(data, path)
      if sections_found > 0 then
        -- Track which sections came from this file
        quiz_sections_by_file[path] = {}
        for k, v in pairs(secs) do
          quiz_sections[k] = v
          quiz_sections_by_file[path][k] = v
        end
        total_sections_loaded = total_sections_loaded + sections_found
        successful_files = successful_files + 1
        io.stderr:write("   ✅ Loaded " .. sections_found .. " quiz section(s)\n")
      else
        io.stderr:write("   ⚠️  No quiz sections found in file\n")
      end
    else
      io.stderr:write("   ❌ Failed to load file\n")
    end
    io.stderr:write("\n")
  end

  if is_pdf_build then
    io.stderr:write("📊 Quiz File Loading Summary for PDF Book:\n")
  else
    io.stderr:write("📊 Quiz File Loading Summary for " .. current_document_file .. ":\n")
  end
  io.stderr:write("   • Files processed: " .. successful_files .. "/" .. #files .. " ✅\n")
  io.stderr:write("   • Total quiz sections loaded: " .. total_sections_loaded .. " 📝\n")
  io.stderr:write(string.rep("-", 80) .. "\n")

  return meta
end

local function is_part_header(block)
  if block.t == "Header" and block.level == 1 then
    -- Check for class 'part'
    if block.attr and block.attr.classes then
      for _, cls in ipairs(block.attr.classes) do
        if cls == "part" then return true end
      end
    end
    -- Check for id starting with 'part-'
    if block.identifier and block.identifier:match("^part%-.+") then
      return true
    end
    -- Check for content like '(PART)' (common in Quarto)
    if block.content and #block.content > 0 then
      local txt = pandoc.utils.stringify(block.content)
      if txt:match("%(%s*PART%s*%)") then return true end
    end
  end
  return false
end

-- 6) Pandoc phase: iterate over blocks and insert quizzes into chapters
local function insert_quizzes(doc)
  if not next(quiz_sections) then
    io.stderr:write("ℹ️  [QUIZ] No quiz sections found for document: " .. current_document_file .. "\n")
    return doc
  end

  local is_pdf_build = false
  if FORMAT and (FORMAT:lower():match("pdf") or FORMAT:lower():match("latex")) then
    is_pdf_build = true
  end

  local function is_part_header(block)
    if block.t == "Header" and block.level == 1 then
      if block.attr and block.attr.classes then
        for _, cls in ipairs(block.attr.classes) do
          if cls == "part" then return true end
        end
      end
      if block.identifier and block.identifier:match("^part%-.+") then
        return true
      end
      if block.content and #block.content > 0 then
        local txt = pandoc.utils.stringify(block.content)
        if txt:match("%(%s*PART%s*%)") then return true end
      end
    end
    return false
  end

  local function flush_chapter(chapter_blocks, chapter_answers, chapter_title)
    local out = {}
    for _, b in ipairs(chapter_blocks) do table.insert(out, b) end
    if #chapter_answers > 0 then
      table.insert(out, pandoc.Header(2, "Self-Check Answers", { id="self-check-answers" }))
      for _, adiv in ipairs(chapter_answers) do table.insert(out, adiv) end
      io.stderr:write("\n📊 [QUIZ] Chapter: " .. (chapter_title or "(unknown)") .. "\n")
      io.stderr:write("   • Self-Check Answers inserted at end of chapter\n")
    end
    return out
  end

  local new_blocks = {}
  local chapter_blocks = {}
  local chapter_answers = {}
  local chapter_title = nil
  local sections_by_file = {}
  local sections_processed = 0

  local section_blocks = {}
  local current_section_id = nil
  local current_section_level = nil
  local current_section_has_quiz = false
  local current_section_quizdiv = nil
  local current_section_answerdiv = nil

  local function flush_section()
    if current_section_has_quiz and current_section_quizdiv then
      local source_file = "unknown"
      for file_path, sections in pairs(quiz_sections_by_file) do
        if sections[current_section_id] then
          source_file = file_path
          if not sections_by_file[source_file] then sections_by_file[source_file] = {} end
          table.insert(sections_by_file[source_file], current_section_id)
          break
        end
      end
      io.stderr:write("   - Section: " .. tostring(current_section_id) .. " (from: " .. source_file .. ")\n")
      table.insert(section_blocks, current_section_quizdiv)
      table.insert(chapter_answers, current_section_answerdiv)
      sections_processed = sections_processed + 1
    end
    for _, b in ipairs(section_blocks) do table.insert(chapter_blocks, b) end
    section_blocks = {}
    current_section_id = nil
    current_section_level = nil
    current_section_has_quiz = false
    current_section_quizdiv = nil
    current_section_answerdiv = nil
  end

  local i = 1
  while i <= #doc.blocks do
    local block = doc.blocks[i]
    local is_chapter_header = block.t == "Header" and block.level == 1
    local is_part = is_part_header(block)
    local is_section_header = block.t == "Header" and block.identifier and block.level == 2
    local sid = is_section_header and ("#" .. block.identifier) or nil
    local level = is_section_header and block.level or nil

    if (is_part or is_chapter_header) and i == #doc.blocks then
      flush_section()
      for _, b in ipairs(flush_chapter(chapter_blocks, chapter_answers, chapter_title)) do
        table.insert(new_blocks, b)
      end
      chapter_blocks = {}
      chapter_answers = {}
      table.insert(new_blocks, block)
      break
    end

    if is_chapter_header or is_part then
      if (#chapter_blocks > 0 or #chapter_answers > 0) or (i == 1) then
        if i ~= 1 then
          flush_section()
          for _, b in ipairs(flush_chapter(chapter_blocks, chapter_answers, chapter_title)) do
            table.insert(new_blocks, b)
          end
          chapter_blocks = {}
          chapter_answers = {}
        end
      end
      chapter_title = pandoc.utils.stringify(block.content)
      io.stderr:write("\n==============================\n[QUIZ] Starting Chapter: " .. chapter_title .. "\n==============================\n")
      section_blocks = { block }
      current_section_id = nil
      current_section_level = nil
      current_section_has_quiz = false
      current_section_quizdiv = nil
      current_section_answerdiv = nil
      i = i + 1
      goto continue
    end

    if is_section_header then
      if current_section_id then
        flush_section()
      else
        for _, b in ipairs(section_blocks) do table.insert(chapter_blocks, b) end
        section_blocks = {}
      end
      section_blocks = { block }
      current_section_id = sid
      current_section_level = level
      current_section_has_quiz = false
      current_section_quizdiv = nil
      current_section_answerdiv = nil
      if quiz_sections[sid] then
        current_section_has_quiz = true
        local qdiv, adiv = process_quiz_questions(quiz_sections[sid], sid)
        current_section_quizdiv = qdiv
        current_section_answerdiv = adiv
      end
      i = i + 1
      goto continue
    end

    table.insert(section_blocks, block)
    i = i + 1
    ::continue::
  end
  flush_section()
  if #chapter_blocks > 0 or #chapter_answers > 0 then
    for _, b in ipairs(flush_chapter(chapter_blocks, chapter_answers, chapter_title)) do
      table.insert(new_blocks, b)
    end
  end
  return pandoc.Pandoc(new_blocks, doc.meta)
end

-- 7) Register the filter
return {
  { Meta   = handle_meta   },
  { Pandoc = insert_quizzes }
}

