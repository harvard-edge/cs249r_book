-- Quiz injection filter for Quarto
local dkjson = require("dkjson")

-- Load quiz data and register sections
function load_quiz_data(quiz_file)
    print("[QUIZ] Trying path: " .. quiz_file)
    
    local file = io.open(quiz_file, "r")
    if not file then
        print("[QUIZ] Could not open quiz file: " .. quiz_file)
        return nil
    end
    
    print("[QUIZ] Loading quiz file: " .. quiz_file)
    local content = file:read("*all")
    file:close()
    
    local quiz_data, pos, err = dkjson.decode(content)
    if err then
        print("[QUIZ] Error parsing JSON: " .. err)
        return nil
    end
    
    return quiz_data
end

function register_sections(quiz_data)
    local sections = {}
    
    -- Handle the actual JSON structure: {sections: [...]}
    if quiz_data.sections then
        for _, section in ipairs(quiz_data.sections) do
            if section.quiz_data and section.quiz_data.quiz_needed and section.quiz_data.questions then
                print("[QUIZ] Registered section: " .. section.section_id)
                sections[section.section_id] = section.quiz_data.questions
            end
        end
    else
        -- Fallback: handle direct section_id -> questions mapping
        for section_id, questions in pairs(quiz_data) do
            if section_id ~= "metadata" then
                print("[QUIZ] Registered section: " .. section_id)
                sections[section_id] = questions
            end
        end
    end
    
    return sections
end

function create_quiz_div(div_id, div_class, content_markdown)
    print("[QUIZ] Creating " .. string.gsub(div_class, "callout%-", "") .. " div with parsed content for ID: " .. div_id)
    print("[QUIZ] Div classes: " .. div_class)
    print("[QUIZ] Div identifier: " .. div_id)
    
    -- Parse the markdown content into Pandoc AST
    local parsed_content = pandoc.read(content_markdown, "markdown")
    
    -- Create the Div with class and ID
    local div = pandoc.Div(parsed_content.blocks, {class = div_class, id = div_id})
    
    return div
end

function process_quiz_questions(questions, section_id)
    local quiz_blocks = {}
    local answer_blocks = {}
    
    -- Create question block
    local question_content = {}
    local answer_content = {}
    
    -- Clean section id for use in anchors
    local clean_section_id = string.gsub(section_id, "^#", "")
    local question_id = "sec-" .. clean_section_id
    local answer_id = question_id .. "-answer"
    
    for i, question in ipairs(questions) do
        table.insert(question_content, tostring(i) .. ". " .. question.question)
        
        if question.question_type == "MCQ" then
            for j, choice in ipairs(question.choices) do
                local choice_letter = string.char(96 + j) -- a, b, c, d
                table.insert(question_content, "   " .. choice_letter .. ") " .. choice)
            end
        elseif question.question_type == "TF" then
            -- True/False questions don't need choices displayed
        elseif question.question_type == "FILL" then
            -- Fill-in-the-blank questions don't need choices
        elseif question.question_type == "SHORT" then
            -- Short answer questions don't need choices
        elseif question.question_type == "ORDER" then
            if question.choices then
                table.insert(question_content, "   Order the following: [" .. table.concat(question.choices, ", ") .. "]")
            end
        elseif question.question_type == "CALC" then
            -- Calculation questions don't need choices displayed
        end
        
        -- Add answer content
        table.insert(answer_content, tostring(i) .. ". **" .. question.question .. "**")
        if question.question_type == "MCQ" then
            for j, choice in ipairs(question.choices) do
                local choice_letter = string.char(96 + j) -- a, b, c, d
                table.insert(answer_content, "   " .. choice_letter .. ") " .. choice)
            end
            table.insert(answer_content, "")
        end
        table.insert(answer_content, "   *Answer*: " .. question.answer)
        table.insert(answer_content, "")
        table.insert(answer_content, "   *Learning Objective*: " .. question.learning_objective)
        table.insert(answer_content, "")
    end
    
    -- Add see-answer link to question block, on its own line, no leading spaces
    table.insert(question_content, "[See Answer →](#" .. answer_id .. ")")
    
    -- Add back-link to question, on its own line, no leading spaces
    table.insert(answer_content, "[↩ Back to Question](#" .. question_id .. ")")
    
    local question_markdown = table.concat(question_content, "\n\n")
    local answer_markdown = table.concat(answer_content, "\n\n")
    
    local question_div = create_quiz_div(question_id, "callout-quiz-question", question_markdown)
    local answer_div = create_quiz_div(answer_id, "callout-quiz-answer", answer_markdown)
    
    return question_div, answer_div
end

function insert_quizzes(doc)
    -- Get quiz file from metadata
    local quiz_file = doc.meta.quiz
    if not quiz_file then
        return doc
    end
    
    local quiz_data = load_quiz_data(pandoc.utils.stringify(quiz_file))
    if not quiz_data then
        return doc
    end
    
    local sections = register_sections(quiz_data)
    local answer_blocks = {}
    local new_blocks = {}
    
    local current_section_id = nil
    local current_section_level = nil
    local section_blocks = {}
    local section_quiz = nil
    
    local function flush_section()
        for _, b in ipairs(section_blocks) do
            table.insert(new_blocks, b)
        end
        if section_quiz then
            table.insert(new_blocks, section_quiz.question_div)
            table.insert(answer_blocks, section_quiz.answer_div)
        end
        section_blocks = {}
        section_quiz = nil
    end
    
    for i, block in ipairs(doc.blocks) do
        if block.t == "Header" and block.identifier then
            -- If we were in a section, flush it
            if current_section_id then
                flush_section()
            end
            -- Start new section
            current_section_id = "#" .. block.identifier
            current_section_level = block.level
            table.insert(section_blocks, block)
            -- Prepare quiz for this section if needed
            if sections[current_section_id] then
                local question_div, answer_div = process_quiz_questions(sections[current_section_id], current_section_id)
                section_quiz = {question_div = question_div, answer_div = answer_div}
            else
                section_quiz = nil
            end
        elseif block.t == "Header" and current_section_level and block.level <= current_section_level then
            -- New header of same or higher level: flush previous section
            flush_section()
            current_section_id = nil
            current_section_level = nil
            table.insert(section_blocks, block)
        else
            table.insert(section_blocks, block)
        end
    end
    -- Flush any remaining section at end of document
    if current_section_id then
        flush_section()
    else
        for _, b in ipairs(section_blocks) do
            table.insert(new_blocks, b)
        end
    end
    
    -- Add "Self-Check Answers" section at the end if we have answer blocks
    if #answer_blocks > 0 then
        local answers_header = pandoc.Header(2, "Self-Check Answers", {id = "self-check-answers"})
        table.insert(new_blocks, answers_header)
        for _, answer_block in ipairs(answer_blocks) do
            table.insert(new_blocks, answer_block)
        end
    end
    
    doc.blocks = new_blocks
    return doc
end

-- Return the filter
return {
    {Pandoc = insert_quizzes}
} 