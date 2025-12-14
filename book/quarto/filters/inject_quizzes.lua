-- inject_quizzes.lua
---@diagnostic disable-next-line: undefined-global
PANDOC_DOCUMENT = PANDOC_DOCUMENT


local json  = require("pandoc.json")
local utils = pandoc.utils

-- global state: all quiz sections are collected here, organized by file source
local quiz_sections = {}
local quiz_sections_by_file = {} -- track which sections came from which file
local current_document_file = "unknown" -- track which document is being processed
local has_quizzes = false -- track if current document has any quiz sections

-- Helper function to check if document has any sections with quizzes
local function document_has_quizzes(doc, quiz_lookup)
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.identifier and block.identifier ~= "" then
      -- Check both with and without "#" prefix since quiz system uses "#" prefix
      local section_id = block.identifier
      local section_id_with_hash = "#" .. section_id
      if quiz_lookup[section_id] or quiz_lookup[section_id_with_hash] then
        return true
      end
    end
  end
  return false
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

  local question_num = 0
  for i, q in ipairs(questions) do
    -- Skip hidden questions
    if not q.hidden then
      question_num = question_num + 1
      table.insert(ql, question_num..". "..q.question)
      if q.question_type == "MCQ" and q.choices then
        local choices_line = {}
        for j, c in ipairs(q.choices) do
          table.insert(choices_line, "   "..string.char(96+j)..") "..c)
        end
        table.insert(ql, table.concat(choices_line, "\n"))
      end
      -- build answers
      table.insert(al, question_num..". **"..q.question.."**")
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
  end

  -- Only create quiz divs if there are visible questions
  if question_num == 0 then
    return nil, nil
  end

  if FORMAT == "latex" then
    table.insert(ql, string.format("\\noindent\\hspace*{1.25em}\\hyperref[%s]{\\textbf{See Answer~$\\rightarrow$}}", aid))
    table.insert(al, string.format("\\noindent\\hspace*{1.25em}\\hyperref[%s]{\\textbf{$\\leftarrow$~Back to Question}}", qid))
  else
    table.insert(ql, '<a href="#' .. aid .. '" class="question-label">See Answers →</a>')
    table.insert(al, '<a href="#' .. qid .. '" class="answer-label">← Back to Questions</a>')
  end

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

  -- Configuration loaded silently

  if not raw then
    if is_pdf_build and auto_discover_pdf then
      -- For PDF builds, auto-discover quiz files from the input files (silently)
      local files = {}
      local total_sections_loaded = 0

      -- Since Quarto combines files into a temporary document, we need to scan the directory directly

local function scan_for_quiz_files()
  local quiz_files = {}

  local command
  if package.config:sub(1,1) == "\\" then
    -- Windows
    command = string.format('dir "%s\\%s" /b /s 2>nul', scan_directory, "*_quizzes.json")
  else
    -- Unix
    command = string.format("find %s -name '%s' 2>/dev/null", scan_directory, "*_quizzes.json")
  end

  local pipe = io.popen(command)
  if pipe then
    for file in pipe:lines() do
      table.insert(quiz_files, file)
    end
    pipe:close()
  end

  return quiz_files
end

      files = scan_for_quiz_files()

      if #files > 0 then
        for i, path in ipairs(files) do
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
            end
          end
        end
      end
    end
    return meta
  end

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

  -- load each file silently
  local total_sections_loaded = 0

  for i, path in ipairs(files) do
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
      end
    end
  end

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

local function insert_quizzes(doc)
  if not next(quiz_sections) then return doc end

  -- Check if this document has any quiz sections
  has_quizzes = document_has_quizzes(doc, quiz_sections)

  if not has_quizzes then
    -- No quizzes for this document, process silently
    return doc
  end

  -- Document has quizzes, show clean processing info
  io.stderr:write("📝 [Quiz Filter] 📚 Document has quizzes - processing...\n")

  local quizzes_injected = 0

  local function has_class(classes, cls)
    for _, c in ipairs(classes) do
      if c == cls then return true end
    end
    return false
  end

  -- priznajemo dva boundary-a: marker ili H1
  local function is_marker(b)
    return b.t == "Div"
       and b.attr
       and has_class(b.attr.classes, "quiz-end")
  end

  local function is_h1(b)
    return b.t == "Header" and b.level == 1
  end

  -- akumulators
  local new_blocks      = {}
  local chapter_blocks  = {}
  local chapter_answers = {}
  local chapter_title   = nil

  local section_blocks       = {}
  local current_section_id   = nil
  local current_quiz_block   = nil
  local current_answer_block = nil

  local function flush_section()
    if current_quiz_block and current_answer_block then
      table.insert(section_blocks, current_quiz_block)
      table.insert(chapter_answers, current_answer_block)
    end
    for _, blk in ipairs(section_blocks) do
      table.insert(chapter_blocks, blk)
    end
    section_blocks       = {}
    current_section_id   = nil
    current_quiz_block   = nil
    current_answer_block = nil
  end

  local function flush_chapter()
    local out = {}
    for _, blk in ipairs(chapter_blocks) do
      table.insert(out, blk)
    end
    if #chapter_answers > 0 then
      table.insert(out,
        pandoc.Header(2,
          { pandoc.Str("Self-Check Answers") },
          pandoc.Attr("self-check-answers")
        )
      )
      for _, ans in ipairs(chapter_answers) do
        table.insert(out, ans)
      end
    end
    return out
  end

  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]

    -- 1) MARKER flush
    if is_marker(b) then
      flush_section()
      if #chapter_blocks>0 or #chapter_answers>0 then
        for _, x in ipairs(flush_chapter()) do
          table.insert(new_blocks, x)
        end
        chapter_blocks, chapter_answers = {}, {}
      end
      -- ne ubacuj marker u novi chapter (ili ga po želji prebaciš)
      -- table.insert(new_blocks, b)
      i = i + 1
      goto continue
    end

    -- 2) H1 boundary flush
    if is_h1(b) then
      flush_section()
      if #chapter_blocks>0 or #chapter_answers>0 then
        for _, x in ipairs(flush_chapter()) do
          table.insert(new_blocks, x)
        end
        chapter_blocks, chapter_answers = {}, {}
      end
      -- sada umetni novi H1
      table.insert(new_blocks, b)
      chapter_title  = pandoc.utils.stringify(b.content or {})
      section_blocks = {}
      i = i + 1
      goto continue
    end

    -- 3) H2 → sekcija sa kvizom
    if b.t=="Header" and b.level==2 and b.identifier then
      if current_section_id then
        flush_section()
      else
        for _, x in ipairs(section_blocks) do
          table.insert(chapter_blocks, x)
        end
        section_blocks = {}
      end
      section_blocks      = { b }
      current_section_id  = "#" .. b.identifier
      local quiz = quiz_sections[current_section_id]
      if quiz then
        local q, a = process_quiz_questions(quiz, current_section_id)
        -- Only set quiz blocks if questions were not all hidden
        if q and a then
          current_quiz_block, current_answer_block = q, a
          quizzes_injected = quizzes_injected + 1
        end
      end
      i = i + 1
      goto continue
    end

    -- 4) ostali blokovi → u sekciju
    table.insert(section_blocks, b)
    i = i + 1

    ::continue::
  end

  -- završni flush na kraju dokumenta
  flush_section()
  if #chapter_blocks>0 or #chapter_answers>0 then
    for _, x in ipairs(flush_chapter()) do
      table.insert(new_blocks, x)
    end
  end

  -- Show summary if quizzes were injected
  if quizzes_injected > 0 then
    io.stderr:write("✅ [Quiz Filter] 📊 SUMMARY: " .. quizzes_injected .. " quiz sections injected\n")
  end

  return pandoc.Pandoc(new_blocks, doc.meta)
end



-- 7) Register the filter
return {
  { Meta   = handle_meta   },
  { Pandoc = insert_quizzes }
}
