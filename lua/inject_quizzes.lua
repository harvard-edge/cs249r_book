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
  io.stderr:write("[QUIZ] load_quiz_data: opening → " .. path .. "\n")
  local f, err = io.open(path, "r")
  if not f then
    io.stderr:write("[QUIZ] ❌ cannot open quiz file: " .. tostring(err) .. "\n")
    return nil
  end
  local content = f:read("*all")
  f:close()

  local ok, data = pcall(json.decode, content)
  if not ok or type(data) ~= "table" then
    io.stderr:write("[QUIZ] ❌ JSON parse error\n")
    return nil
  end
  io.stderr:write("[QUIZ] JSON parsed successfully\n")
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
        io.stderr:write("[QUIZ] register: " .. s.section_id .. "\n")
        secs[s.section_id] = s.quiz_data.questions
      end
    end
  else
    for sid, qs in pairs(data) do
      if sid ~= "metadata" then
        io.stderr:write("[QUIZ] register (fallback): " .. sid .. "\n")
        secs[sid] = qs
      end
    end
  end
  local cnt = 0
  for _ in pairs(secs) do cnt = cnt + 1 end
  io.stderr:write("[QUIZ] total sections: " .. cnt .. "\n")
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
  io.stderr:write("[QUIZ] processing questions for " .. section_id .. "\n")
  local ql, al = {}, {}
  local clean = section_id:gsub("^#", "")
  local qid   = "sec-" .. clean
  local aid   = qid .. "-answer"

  for i, q in ipairs(questions) do
    table.insert(ql, i..". "..q.question)
    if q.question_type == "MCQ" and q.choices then
      for j, c in ipairs(q.choices) do
        table.insert(ql, "   "..string.char(96+j)..") "..c)
      end
    end
    -- build answers
    table.insert(al, i..". **"..q.question.."**")
    if q.question_type == "MCQ" and q.choices then
      for j, c in ipairs(q.choices) do
        table.insert(al, "   "..string.char(96+j)..") "..c)
      end
      table.insert(al, "")
    end
    table.insert(al, "   *Answer*: "..q.answer)
    table.insert(al, "")
    table.insert(al, "   *Learning Objective*: "..q.learning_objective)
    table.insert(al, "")
  end

  table.insert(ql, "[See Answer →](#" .. aid .. ")")
  table.insert(al, "[↩ Back to Question](#" .. qid .. ")")

  return create_quiz_div(qid, "callout-quiz-question", table.concat(ql, "\n\n")),
         create_quiz_div(aid, "callout-quiz-answer",   table.concat(al, "\n\n"))
end

-- 5) Meta phase: read one or more paths from meta.quiz
local function handle_meta(meta)
  local raw = meta.quiz
  if not raw then
    io.stderr:write("[QUIZ] Meta: no quiz metadata\n")
    return meta
  end

  -- collect all files into this list
  local files = {}

  -- if raw is a table with elements, treat it as a list
  if type(raw) == "table" and raw[1] ~= nil then
    for _, item in ipairs(raw) do
      local p = utils.stringify(item)
      io.stderr:write("[QUIZ] Meta: queueing JSON → " .. p .. "\n")
      table.insert(files, p)
    end
  else
    -- not a list, so it's a single file
    local p = utils.stringify(raw)
    io.stderr:write("[QUIZ] Meta: queueing single JSON → " .. p .. "\n")
    table.insert(files, p)
  end

  -- load each file individually
  for _, path in ipairs(files) do
    io.stderr:write("[QUIZ] Meta: loading JSON → " .. path .. "\n")
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
  io.stderr:write("[QUIZ] insert_quizzes: start\n")

  local new_blocks      = {}
  local chapter_answers = {}

  local function flush_answers()
    if #chapter_answers > 0 then
      table.insert(new_blocks,
        pandoc.Header(2, "Self-Check Answers", { id="self-check-answers" })
      )
      for _, ans in ipairs(chapter_answers) do
        table.insert(new_blocks, ans)
      end
      chapter_answers = {}
    end
  end

  for _, block in ipairs(doc.blocks) do
    local is_chapter_start = block.t == "Header" and block.level == 1
    local is_references    = block.t == "Header"
      and pandoc.utils.stringify(block.content)
         :upper():find("REFERENCES")

    if is_chapter_start or is_references then
      flush_answers()
    end

    table.insert(new_blocks, block)

    if block.t == "Header" and block.identifier then
      local sid = "#" .. block.identifier
      if quiz_sections[sid] then
        io.stderr:write("[QUIZ] MATCHED " .. sid .. "\n")
        local qdiv, adiv = process_quiz_questions(quiz_sections[sid], sid)
        table.insert(new_blocks, qdiv)
        table.insert(chapter_answers, adiv)
      end
    end
  end

  -- answers for the last chapter
  flush_answers()

  doc.blocks = new_blocks
  return doc
end

-- 7) Register the filter
return {
  { Meta   = handle_meta   },
  { Pandoc = insert_quizzes }
}
