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

  -- For Quarto, we need to handle temporary file paths
  -- Try multiple strategies to find the quiz file
  local possible_paths = {}
  
  -- Strategy 1: Try relative to current working directory (for direct pandoc calls)
  table.insert(possible_paths, quiz_file)
 
  -- Strategy 2: If it's an absolute path, use it directly
  if quiz_file:match("^/") then
    possible_paths = {quiz_file}
  end
  
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
    io.stderr:write("[ERROR] Quiz file not found in any of the tried paths. Tried:\n")
    for _, path in ipairs(possible_paths) do
      io.stderr:write("  - " .. path .. "\n")
    end
    return
  end

  io.stderr:write("[QUIZ] Loading quiz file: " .. found_file .. "\n")
  local fh = io.open(found_file, "r")
  if not fh then
    io.stderr:write("[ERROR] Quiz file not found: " .. found_file .. "\n")
    return
  end
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
  local question_id = section_id:gsub("^#sec%-", "quiz-question-sec-")
  local blocks = {}
  
  -- Create ordered list items for all questions
  local question_items = {}
  
  for i, q in ipairs(questions) do
    local qtype = q.question_type or "SHORT"
    local question_content = {}
    
    -- Add question text
    table.insert(question_content, pandoc.Para{pandoc.Str(q.question)})
    
    if qtype == "MCQ" and q.choices then
      -- Create nested ordered list for choices with letter styling
      local choice_items = {}
      for j, choice in ipairs(q.choices) do
        table.insert(choice_items, pandoc.Plain{pandoc.Str(choice)})
      end
      -- Create ordered list with letter styling (a, b, c...)
      local choice_list = pandoc.OrderedList(choice_items, {1, "LowerAlpha", "OneParen"})
      table.insert(question_content, choice_list)
    end
    
    -- Create list item with question content
    table.insert(question_items, question_content)
  end
  
  -- Create the main ordered list for questions
  local questions_list = pandoc.OrderedList(question_items, {1, "Decimal", "Period"})
  table.insert(blocks, questions_list)
  
  -- Add blank line and "See Answer" reference
  table.insert(blocks, pandoc.Para{})
  table.insert(blocks, pandoc.Para{
    pandoc.Str("\u{00A0}\u{00A0}\u{00A0}\u{00A0}"),
    pandoc.Link({pandoc.Str("See Answers →")}, "#" .. answer_id)
  })
  
  return pandoc.Div(blocks, pandoc.Attr(question_id, {"callout-quiz-question"}))
end

-- Format answer block for end-of-file insertion
local function format_answer_block(section_id, questions)
  local id = section_id:gsub("^#", "quiz-answer-sec-")
  local question_id = section_id:gsub("^#sec%-", "quiz-question-sec-")
  local blocks = {}
  
  -- Create ordered list items for all answer questions
  local answer_items = {}
  
  for i, q in ipairs(questions) do
    local qtype = q.question_type or "SHORT"
    local answer_content = {}
    
    -- Add bold question text
    table.insert(answer_content, pandoc.Para{pandoc.Strong{pandoc.Str(q.question)}})
    
    -- Add MCQ choices if present
    if qtype == "MCQ" and q.choices then
      -- Create nested ordered list for choices in answers
      local choice_items = {}
      for j, choice in ipairs(q.choices) do
        table.insert(choice_items, pandoc.Plain{pandoc.Str(choice)})
      end
      -- Create ordered list with letter styling (a, b, c...)
      local choice_list = pandoc.OrderedList(choice_items, {1, "LowerAlpha", "OneParen"})
      table.insert(answer_content, choice_list)
    end
    
    -- Format answer
    local answer_str = q.answer or ""
    if qtype == "FILL" and answer_str ~= "" then
      local first_period = answer_str:find("%.")
      local fill_word, rest = answer_str, ""
      if first_period then
        fill_word = answer_str:sub(1, first_period - 1):gsub("\n", " "):gsub("%s+$", "")
        rest = answer_str:sub(first_period + 1):gsub("^%s+", "")
      end
      answer_str = string.format('The answer is "%s".', fill_word)
      if rest ~= "" then
        answer_str = answer_str .. " " .. rest
      end
    elseif qtype == "ORDER" and answer_str ~= "" then
      answer_str = "The order is as follows: " .. answer_str
    end
    
    -- Add Answer and Learning Objective as regular paragraphs (no list structure)
    
    -- Add Answer paragraph
    table.insert(answer_content, pandoc.Para{
      pandoc.Emph{pandoc.Str("Answer")},
      pandoc.Str(": " .. answer_str)
    })
    
    -- Add Learning Objective paragraph (if present)
    if q.learning_objective and q.learning_objective ~= "" then
      table.insert(answer_content, pandoc.Para{
        pandoc.Emph{pandoc.Str("Learning Objective")},
        pandoc.Str(": " .. q.learning_objective)
      })
    end
    
    -- Create list item with answer content
    table.insert(answer_items, answer_content)
  end
  
  -- Create the main ordered list for answers
  local answers_list = pandoc.OrderedList(answer_items, {1, "Decimal", "Period"})
  table.insert(blocks, answers_list)
  
  -- Add blank line and "Back to Question" link at the end of the callout
  table.insert(blocks, pandoc.Para{})
  table.insert(blocks, pandoc.Para{
    pandoc.Str("\u{00A0}\u{00A0}\u{00A0}\u{00A0}"),
    pandoc.Link({pandoc.Str("↩ Back to Self-Check Questions")}, "#" .. question_id)
  })
  
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
      -- Also check if there's an explicit ID in the header attributes
      if blk.identifier and blk.identifier ~= "" then
        current_section_id = "#" .. blk.identifier
      end
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
    table.insert(new_blocks, pandoc.Header(2, {pandoc.Str("Self-Check Answers")}, "self-check-answers"))
    for _, b in ipairs(answer_blocks) do
      table.insert(new_blocks, b)
    end
  end

  return pandoc.Pandoc(new_blocks, doc.meta)
end
