-- inject_quizzes.lua

local json  = require("pandoc.json")
local utils = pandoc.utils

-- global state: all quiz sections are collected here
local quiz_sections = {}

-- helper: checks if output is PDF/LaTeX
local function is_pdf()
  return FORMAT
    and (FORMAT:lower():match("pdf") or FORMAT:lower():match("latex"))
end

-- 1) Load a single JSON file
local function load_quiz_data(path)
  local f, err = io.open(path, "r")
  if not f then
    return nil
  end
  local content = f:read("*all")
  f:close()

  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    return nil
  end
  return data
end

-- 2) Extract sections from JSON and map them to section_id
local function register_sections(data)
  local secs = {}
  if data.sections then
    for _, s in ipairs(data.sections) do
      if s.quiz_data
         and s.quiz_data.quiz_needed
         and s.quiz_data.questions then
        secs[s.section_id] = s.quiz_data.questions
      end
    end
  else
    for sid, qs in pairs(data) do
      if sid ~= "metadata" then
        secs[sid] = qs
      end
    end
  end
  return secs
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
  if not raw then
    -- Only log if we're in verbose mode or if this is a file that should have quizzes
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

  -- load each file individually
  for _, path in ipairs(files) do
    local data = load_quiz_data(path)
    if data then
      local secs = register_sections(data)
      for k, v in pairs(secs) do
        quiz_sections[k] = v
      end
    end
  end

  return meta
end

-- 6) Pandoc phase: iterate over blocks and insert quizzes into chapters
local function insert_quizzes(doc)
  if not next(quiz_sections) then
    io.stderr:write("No quiz found for this file.\n")
    return doc
  end
  io.stderr:write("\n========== [QUIZ] Inserting Quizzes ==========" .. "\n")

  local new_blocks      = {}
  local chapter_answers = {}

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
      io.stderr:write("✅ [QUIZ] Inserted quiz for section: " .. tostring(current_section_id) .. "\n")
      table.insert(new_blocks, current_section_quizdiv)
      table.insert(chapter_answers, current_section_answerdiv)
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
    for _, adiv in ipairs(chapter_answers) do
      table.insert(new_blocks, adiv)
    end
    io.stderr:write("\n========== [QUIZ] Done ==========" .. "\n")
  end

  return pandoc.Pandoc(new_blocks, doc.meta)
end

-- 7) Register the filter
return {
  { Meta   = handle_meta   },
  { Pandoc = insert_quizzes }
}

