local json = require("dkjson")
local quizzes = {}
local answer_blocks = {}

-- Utility: read quiz file path relative to current doc
function resolve_relative_path(doc_path, quiz_file)
  -- Assume both doc_path and quiz_file are relative
  local dir = string.match(doc_path, "^(.*)/[^/]+%.qmd$")
  return dir and (dir .. "/" .. quiz_file) or quiz_file
end

-- Load quiz data from per-doc YAML
function Reader(meta)
  if not meta.quiz then return end

  local doc_path = quarto.doc.input_file
  local quiz_file = pandoc.utils.stringify(meta.quiz)
  local resolved_path = resolve_relative_path(doc_path, quiz_file)

  local f = io.open(resolved_path, "r")
  if not f then
    io.stderr:write("⚠️  Could not open quiz file: " .. resolved_path .. "\n")
    return
  end

  local content = f:read("*all")
  f:close()

  local data, _, err = json.decode(content, 1, nil)
  if err then
    io.stderr:write("⚠️  Failed to parse JSON in " .. resolved_path .. ": " .. err .. "\n")
    return
  end

  for _, s in ipairs(data.sections or {}) do
    if s.quiz_data and s.quiz_data.quiz_needed then
      local id = string.gsub(s.section_id, "^#", "")
      quizzes[id] = s.quiz_data.questions
    end
  end
end

-- Generate callout blocks
function create_callout_blocks(section_id, questions)
  local blocks = {}
  for i, q in ipairs(questions) do
    local qid = "quiz-question-" .. section_id .. "-" .. i
    local aid = "quiz-answer-" .. section_id .. "-" .. i

    table.insert(blocks, pandoc.RawBlock("markdown", string.format([[
:::{.callout-quiz-question #%s}
%s  
[See Answer](#%s)
:::
]], qid, q.question, aid)))

    table.insert(answer_blocks, pandoc.RawBlock("markdown", string.format([[
:::{.callout-quiz-answer #%s}
%s
:::
]], aid, q.answer)))
  end
  return blocks
end

-- Match headers and inject questions
function Header(el)
  if quizzes[el.identifier] then
    local blocks = create_callout_blocks(el.identifier, quizzes[el.identifier])
    return {el, table.unpack(blocks)}
  end
end

-- Insert answer section at end of doc
function Doc(doc)
  if #answer_blocks > 0 then
    table.insert(doc.blocks, pandoc.Header(2, "Quiz Answers"))
    for _, b in ipairs(answer_blocks) do
      table.insert(doc.blocks, b)
    end
  end
  return doc
end
