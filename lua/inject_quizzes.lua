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

  io.stderr:write("🔍 [DEBUG] Number of input files: " .. (PANDOC_STATE and PANDOC_STATE.input_files and #PANDOC_STATE.input_files or "unknown") .. "\n")
  io.stderr:write("🔍 [DEBUG] Is PDF build: " .. tostring(is_pdf_build) .. "\n")
  io.stderr:write("🔍 [DEBUG] Quiz metadata: " .. tostring(raw) .. "\n")
  io.stderr:write("🔍 [DEBUG] FORMAT: " .. tostring(FORMAT) .. "\n")
  io.stderr:write("🔍 [DEBUG] Current document file: " .. current_document_file .. "\n")

  if not raw then
    if is_pdf_build then
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
        local core_dir = "contents/core"
        
        -- Try to open the core directory
        local dir = io.popen("find " .. core_dir .. " -name '*_quizzes.json' 2>/dev/null")
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
        io.stderr:write("📁 No quiz files found in contents/core/ directory\n")
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

-- 6) Pandoc phase: iterate over blocks and insert quizzes into chapters
local function insert_quizzes(doc)
  if not next(quiz_sections) then
    io.stderr:write("ℹ️  [QUIZ] No quiz sections found for document: " .. current_document_file .. "\n")
    return doc
  end

  -- Try to extract chapter title
  local chapter_title = "Unknown Chapter"
  
  -- Method 1: Try to get title from document metadata
  if doc.meta and doc.meta.title then
    chapter_title = utils.stringify(doc.meta.title)
  end
  
  -- Method 2: If no metadata title, try to get from first heading
  if chapter_title == "Unknown Chapter" and doc.blocks and #doc.blocks > 0 then
    for _, block in ipairs(doc.blocks) do
      if block.t == "Header" and block.level == 1 then
        chapter_title = utils.stringify(block.content)
        break
      end
    end
  end
  
  -- Method 3: If still no title, try to get from first level 2 heading
  if chapter_title == "Unknown Chapter" and doc.blocks and #doc.blocks > 0 then
    for _, block in ipairs(doc.blocks) do
      if block.t == "Header" and block.level == 2 then
        chapter_title = utils.stringify(block.content)
        break
      end
    end
  end

  io.stderr:write("🔄 [QUIZ] Inserting quizzes into document: " .. current_document_file .. "\n")
  io.stderr:write("📖 Chapter: " .. chapter_title .. "\n")

  local new_blocks      = {}
  local chapter_answers = {}
  local sections_processed = 0
  local sections_by_file = {}

  local section_blocks = {}
  local current_section_id = nil
  local current_section_level = nil
  local current_section_has_quiz = false
  local current_section_quizdiv = nil
  local current_section_answerdiv = nil

  local function flush_section()
    for _, b in ipairs(section_blocks) do
      table.insert(new_blocks, b)
    end
    if current_section_has_quiz and current_section_quizdiv then
      -- Find which file this section came from
      local source_file = "unknown"
      for file_path, sections in pairs(quiz_sections_by_file) do
        if sections[current_section_id] then
          source_file = file_path
          if not sections_by_file[source_file] then
            sections_by_file[source_file] = {}
          end
          table.insert(sections_by_file[source_file], current_section_id)
          break
        end
      end
      
      io.stderr:write("✅ Inserted quiz for section: " .. tostring(current_section_id) .. " (from: " .. source_file .. ")\n")
      table.insert(new_blocks, current_section_quizdiv)
      table.insert(chapter_answers, current_section_answerdiv)
      sections_processed = sections_processed + 1
    end
    section_blocks = {}
    current_section_id = nil
    current_section_level = nil
    current_section_has_quiz = false
    current_section_quizdiv = nil
    current_section_answerdiv = nil
  end

  for i, block in ipairs(doc.blocks) do
    local is_section_header = block.t == "Header" and block.identifier and block.level == 2
    local sid = is_section_header and ("#" .. block.identifier) or nil
    local level = is_section_header and block.level or nil

    if is_section_header then
      if current_section_id and current_section_level and level and level <= current_section_level then
        flush_section()
      end
      current_section_id = sid
      current_section_level = level
      if quiz_sections[sid] then
        current_section_has_quiz = true
        local qdiv, adiv = process_quiz_questions(quiz_sections[sid], sid)
        current_section_quizdiv = qdiv
        current_section_answerdiv = adiv
      end
    end
    table.insert(section_blocks, block)
  end
  flush_section()

  -- Insert all answers at the end of the chapter
  if #chapter_answers > 0 then
    -- Add a section header for Self-Check Answers
    table.insert(new_blocks, pandoc.Header(2, "Self-Check Answers", { id="self-check-answers" }))
    
    for _, adiv in ipairs(chapter_answers) do
      table.insert(new_blocks, adiv)
    end
    
    io.stderr:write("\n📊 Quiz Insertion Summary for " .. current_document_file .. ":\n")
    io.stderr:write("📖 Chapter: " .. chapter_title .. "\n")
    io.stderr:write("   • Total sections processed: " .. sections_processed .. " ✅\n")
    io.stderr:write("   • Total answers added: " .. #chapter_answers .. " 📝\n")
    
    -- Show breakdown by file
    for file_path, sections in pairs(sections_by_file) do
      io.stderr:write("   • From " .. file_path .. ": " .. #sections .. " section(s) 📄\n")
    end
    
    io.stderr:write(string.rep("=", 80) .. "\n")
    io.stderr:write("✅ [QUIZ] Document processing complete: " .. current_document_file .. "\n")
    io.stderr:write("📖 Chapter: " .. chapter_title .. "\n")
    io.stderr:write(string.rep("=", 80) .. "\n\n")
  end

  return pandoc.Pandoc(new_blocks, doc.meta)
end

-- 7) Register the filter
return {
  { Meta   = handle_meta   },
  { Pandoc = insert_quizzes }
}

