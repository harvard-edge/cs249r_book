local callouts = {}

-- Helper function to extract options from div attributes or content
local function extractOptions(div, defaultType)
    local options = {
        type = defaultType,
        collapse = div.attributes["collapse"] == "true",
        appearance = "minimal",
        icon = defaultType == "colab" and "👩‍💻" or
               defaultType == "question" and "❓" or
               defaultType == "hint" and "🤔" or
               defaultType == "answer" and "✅" or
               defaultType == "slide" and "🎫" or
               defaultType == "video" and "📺" or
               defaultType == "lab" and "🧪"
    }

    if div.attr and div.attr.identifier ~= "" then
        options.id = div.attr.identifier
    end

    if div.content[1] ~= nil and div.content[1].t == "Header" then
        options.title = pandoc.utils.stringify(div.content[1])
        table.remove(div.content, 1)
    end

    options.content = pandoc.Blocks(div.content)

    return options
end

-- Function to create a callout with specified options
local function createCallout(div, calloutType)
    local options = extractOptions(div, calloutType)
    if options.id then
        callouts[options.id] = options
        print("Collected callout with ID:", options.id)
    end
    return quarto.Callout(options)
end

-- Main Div function
function Div(div)
    if quarto.doc.isFormat("html") then
        if div.classes:includes("callout-warning") then
            return createCallout(div, "colab")
        elseif div.classes:includes("callout-answer") then
            return createCallout(div, "answer")
        elseif div.classes:includes("callout-hint") then
            return createCallout(div, "hint")
        elseif div.classes:includes("callout-lab") then
            return createCallout(div, "lab")
        elseif div.classes:includes("callout-slide") then
            return createCallout(div, "slide")
        elseif div.classes:includes("callout-question") then
            return createCallout(div, "question")
        elseif div.classes:includes("callout-video") then
            return createCallout(div, "video")
        end
    end
end