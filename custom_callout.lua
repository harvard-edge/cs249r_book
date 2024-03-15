-- Helper function to extract options from div attributes or content
local function extractOptions(div, defaultType)
    local options = {
        type = defaultType,
        collapse = div.attributes["collapse"] == "true", -- Check if collapse attribute is true
        appearance = "minimal", -- Default appearance
        -- Default icons for each callout type, could be overridden
        icon = defaultType == "exercise" and "📚" or
               defaultType == "question" and "❓" or
               defaultType == "hint" and "🤔" or
               defaultType == "answer" and "✅" or
               defaultType == "slide" and "🎫" or
               defaultType == "video" and "📺" or
               defaultType == "hint" and "🤔" or
               defaultType == "lab" and "👩‍💻"
    }
    
    -- Example of extracting options from div attributes or structured content
    -- This is where you'd implement logic based on how options are specified in your documents

    -- Use the first element as the title if it's a Header
    if div.content[1] ~= nil and div.content[1].t == "Header" then
        options.title = pandoc.utils.stringify(div.content[1])
        table.remove(div.content, 1) -- Remove the header from content
    end

    -- Content is always the remaining div
    options.content = pandoc.Blocks(div.content)

    return options
end

-- Function to create a callout with specified options
local function createCallout(div, calloutType)
    local options = extractOptions(div, calloutType)
    return quarto.Callout(options)
end

-- Main Div function
function Div(div)
    if quarto.doc.isFormat("html") then
        -- Determine callout type based on div classes and process accordingly
        if div.classes:includes("callout-exercise") then
            local callout = createCallout(div, "exercise")
            return callout
        elseif div.classes:includes("callout-answer") then
            local callout = createCallout(div, "answer")
            return callout
        elseif div.classes:includes("callout-hint") then
            local callout = createCallout(div, "hint")
            return callout
        elseif div.classes:includes("callout-lab") then
            local callout = createCallout(div, "lab")
            return callout
        elseif div.classes:includes("callout-slide") then
            local callout = createCallout(div, "slide")
            return callout
        elseif div.classes:includes("callout-question") then
            local callout = createCallout(div, "question")
            return callout
        elseif div.classes:includes("callout-video") then
            local callout = createCallout(div, "video")
            return callout

        end
    end
end
