-- Lua filter to inject quizzes and answers from a JSON file listed in the YAML metadata

local json = require("dkjson")
local utils = require("pandoc.utils")
local system = require("pandoc.system")

local section_map = {}
local answer_blocks = {}
local loaded = false

-- Helper: convert section title to ID (if not provided explicitly)
local function slugify(text)
  return "#" .. text:gsub("[^%w]+", "-"):gsub("-$", ""):lower()
end

-- Load quiz data only once
local function load_quiz_data(meta)
  if loaded then return end
  loaded = true
  if not meta.quiz then return end
  local quiz_file = pandoc.utils.stringify(meta.quiz)
  
  -- Try multiple path resolution strategies
  local possible_paths = {
    quiz_file,  -- Try as-is first
    "contents/core/introduction/" .. quiz_file,  -- Try relative to common structure
    "contents/core/" .. quiz_file,  -- Try broader relative path
  }
  
  local found_file = nil
  for _, path in ipairs(possible_paths) do
    io.stderr:write("[QUIZ] Trying path: " .. path .. "\n")
    local fh = io.open(path, "r")
    if fh then
      fh:close()
      found_file = path
      break
    end
  end
  
  if not found_file then
    io.stderr:write("[ERROR] Quiz file not found in any of the attempted paths\n")
    return
  end
  
  io.stderr:write("[QUIZ] Loading quiz file: " .. found_file .. "\n")
  local fh = io.open(found_file, "r")
  local content = fh:read("*all")
  fh:close()
  local data = json.decode(content)
  for _, sec in ipairs(data.sections) do
    if sec.quiz_data and sec.quiz_data.quiz_needed then
      io.stderr:write("[QUIZ] Registered section: " .. sec.section_id .. "\n")
      section_map[sec.section_id] = sec.quiz_data.questions
    end
  end
end

-- Format questions for inline quiz block
local function format_quiz_block(questions, section_id)
  local answer_id = section_id:gsub("^#", "quiz-answer-sec-")
  
  -- Build the entire quiz block as raw markdown
  local quiz_text = "::: {.callout-quiz-question}\n\n"
  
  for i, q in ipairs(questions) do
    quiz_text = quiz_text .. i .. ". " .. q.question .. "\n\n"
    
    if q.question_type == "MCQ" and q.choices then
      for _, choice in ipairs(q.choices) do
        quiz_text = quiz_text .. "   " .. choice .. "\n"
      end
      quiz_text = quiz_text .. "\n"
    elseif q.question_type == "TF" then
      quiz_text = quiz_text .. "Answer True or False.\n\n"
    elseif q.question_type == "FILL" then
      quiz_text = quiz_text .. "Fill in the blank.\n\n"
    elseif q.question_type == "SHORT" then
      quiz_text = quiz_text .. "Short answer expected.\n\n"
    elseif q.question_type == "ORDER" then
      quiz_text = quiz_text .. "Reorder the items appropriately.\n\n"
    elseif q.question_type == "CALC" then
      quiz_text = quiz_text .. "Show your calculation.\n\n"
    end
  end
  
  quiz_text = quiz_text .. "See Answer \\ref{" .. answer_id .. "}.\n\n:::\n"
  
  return pandoc.RawBlock("markdown", quiz_text)
end

-- Format answer block for end-of-file insertion
local function format_answer_block(section_id, questions)
  local id = section_id:gsub("^#", "quiz-answer-sec-")
  local blocks = {}
  for i, q in ipairs(questions) do
    table.insert(blocks, pandoc.Para{pandoc.Strong{pandoc.Str(i .. ". "), pandoc.Str(q.question)}})
    if q.question_type == "MCQ" and q.choices then
      for _, choice in ipairs(q.choices) do
        table.insert(blocks, pandoc.Plain{pandoc.Str("   "), pandoc.Str(choice:sub(1,1)), pandoc.Str(") "), pandoc.Str(choice:sub(3))})
      end
    end
    table.insert(blocks, pandoc.Para{pandoc.Str("   *Answer*: "), pandoc.Str(q.answer)})
    table.insert(blocks, pandoc.Para{pandoc.Str("   *Learning Objective*: "), pandoc.Str(q.learning_objective)})
  end
  return pandoc.Div(blocks, pandoc.Attr(id, {"callout-quiz-answer"}))
end

-- Insert quizzes after each ## section, collect answers for later
function Pandoc(doc)
  load_quiz_data(doc.meta)
  local new_blocks = {}
  local current_section_id = nil

  for i = 1, #doc.blocks do
    local blk = doc.blocks[i]
    table.insert(new_blocks, blk)

    if blk.t == "Header" and blk.level == 2 then
      current_section_id = "#" .. utils.stringify(blk.content):gsub("[^%w]+", "-"):gsub("-$", ""):lower()
    end

    local next_is_section = (i == #doc.blocks) or (doc.blocks[i+1].t == "Header" and doc.blocks[i+1].level == 2)
    if current_section_id and section_map[current_section_id] and next_is_section then
      io.stderr:write("[QUIZ] Inserting quiz for section: " .. current_section_id .. "\n")
      local quiz_block = format_quiz_block(section_map[current_section_id], current_section_id)
      local answer_block = format_answer_block(current_section_id, section_map[current_section_id])
      table.insert(new_blocks, quiz_block)
      table.insert(answer_blocks, answer_block)
      section_map[current_section_id] = nil
    end
  end

  -- Add final answer section
  if #answer_blocks > 0 then
    io.stderr:write("[QUIZ] Appending Self-Check Answers section with " .. #answer_blocks .. " blocks.\n")
    table.insert(new_blocks, pandoc.Header(2, pandoc.Str("Self-Check Answers")))
    for _, b in ipairs(answer_blocks) do
      table.insert(new_blocks, b)
    end
  end

  return pandoc.Pandoc(new_blocks, doc.meta)
end
