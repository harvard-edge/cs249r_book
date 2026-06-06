local function ensure_html_deps()
  quarto.doc.add_html_dependency({
    name = "pseudocode",
    version = "2.4.1",
    scripts = { "pseudocode.min.js" },
    stylesheets = { "pseudocode.min.css" },
  })
  quarto.doc.include_text(
    "in-header",
    [[
    <style type="text/css">
    .ps-root .ps-algorithm {
      border-top: 2px solid;
      border-bottom: 2px solid;
    }
    .pseudocode-container {
      text-align: left;
    }
    </style>
  ]]
  )
  quarto.doc.include_text(
    "after-body",
    [[
    <script type="text/javascript">
    (function(d) {
      d.querySelectorAll(".pseudocode-container").forEach(function(el) {
        let pseudocodeOptions = {
          indentSize: el.dataset.indentSize,
          commentDelimiter: " " + el.dataset.commentDelimiter + " ",
          lineNumber: el.dataset.lineNumber.toLowerCase() === "true",
          lineNumberPunc: el.dataset.lineNumberPunc,
          noEnd: el.dataset.noEnd.toLowerCase() === "true",
          scopeLines: el.dataset.indentLines.toLowerCase() === "true",
          titlePrefix: el.dataset.captionPrefix,
        };
        pseudocode.renderElement(el.querySelector(".pseudocode"), pseudocodeOptions);
      });
    })(document);
    (function(d) {
      d.querySelectorAll(".pseudocode-container").forEach(function(el) {
        let captionSpan = el.querySelector(".ps-root > .ps-algorithm > .ps-line > .ps-keyword")
        if (captionSpan !== null) {
          let captionPrefix = el.dataset.captionPrefix + " ";
          let captionNumber = "";
          if (el.dataset.pseudocodeNumber) {
            captionNumber = el.dataset.pseudocodeNumber + " ";
            if (el.dataset.chapterLevel) {
              captionNumber = el.dataset.chapterLevel + "." + captionNumber;
            }
          }
          captionSpan.innerHTML = captionPrefix + captionNumber;
        }
      });
    })(document);
    </script>
  ]]
  )
end

local function nil_to_default(value, default)
  if value == nil then
    return default
  else
    return value
  end
end

local function ensure_latex_deps()
  quarto.doc.use_latex_package("algorithm")
  quarto.doc.use_latex_package(
    "algpseudocodex",
    "noEnd=false, indLines=false, italicComments=false, rightComments=false, commentColor=black, beginComment=//~"
  )
  quarto.doc.use_latex_package("caption")
  quarto.doc.include_text(
    "in-header",
    [[
    \makeatletter
    \newcommand\fs@nocaption{
      \def\@fs@cfont{\bfseries}
      \let\@fs@capt\floatc@plain
      \def\@fs@pre{}%
      \def\@fs@post{\kern2pt\hrule}%
      \def\@fs@mid{\hrule\kern2pt}%
      \let\@fs@iftopcapt\iftrue}
    \newcommand{\algpxEndIndentHelper}{\algpx@endIndent}
    \newcommand{\algpxSetCommentColor}[1]{\renewcommand{\algpx@commentColor}{#1}}
    \newcommand{\algpxSetBeginComment}[1]{\renewcommand{\algpx@beginComment}{#1}}
    \newcommand{\algpxSetEndComment}[1]{\renewcommand{\algpx@endComment}{#1}}
    \makeatother
  ]]
  )
end

local function extract_source_code_options(source_code, render_type)
  local options = {}
  local source_codes = {}
  local found_source_code = false

  for str in string.gmatch(source_code, "([^\n]*)\n?") do
    if (string.match(str, "^%s*#|.*") or string.gsub(str, "%s", "") == "") and not found_source_code then
      if string.match(str, "^%s*#|%s+[" .. render_type .. "|label].*") then
        str = string.gsub(str, "^%s*#|%s+", "")
        local idx_start, idx_end = string.find(str, ":%s*")

        if idx_start and idx_end and idx_end + 1 < #str then
          k = string.sub(str, 1, idx_start - 1)
          v = string.sub(str, idx_end + 1)
          v = string.gsub(v, '^%s*"', "")
          v = string.gsub(v, '"%s*$', "")

          options[k] = v
        else
          quarto.log.warning("Invalid pseducode option: " .. str)
        end
      end
    else
      found_source_code = true
      table.insert(source_codes, str)
    end
  end

  return options, table.concat(source_codes, "\n")
end

local function render_pseudocode_block_html(global_options)
  ensure_html_deps()

  if global_options.caption_align then
    quarto.doc.include_text("in-header", [[
      <style type="text/css">
      .ps-algorithm > .ps-line {
        text-align: ]] .. global_options.caption_align .. [[;
      }
      </style>
    ]])
  end

  local filter = {
    CodeBlock = function(el)
      if not el.attr.classes:includes("pseudocode") then
        return el
      end

      local options, source_code = extract_source_code_options(el.text, "html")

      source_code = string.gsub(source_code, "%s*\\begin{algorithm}[^\n]+", "\\begin{algorithm}")
      source_code = string.gsub(source_code, "%s*\\begin{algorithmic}[^\n]+", "\\begin{algorithmic}")

      local algorithm_id = options["label"]
      options["label"] = nil
      options["html-caption-prefix"] = global_options.caption_prefix

      if global_options.number_with_in_chapter and global_options.html_chapter_level then
        options["html-chapter-level"] = global_options.html_chapter_level
      end

      if global_options.caption_number then
        options["html-pseudocode-number"] = global_options.html_current_number
      end

      options["html-indent-size"] = nil_to_default(options["html-indent-size"], "1.2em")
      options["html-comment-delimiter"] = nil_to_default(options["html-comment-delimiter"], "//")
      options["html-line-number"] = string.lower(nil_to_default(options["html-line-number"], "true"))
      options["html-line-number-punc"] = nil_to_default(options["html-line-number-punc"], ":")
      options["html-no-end"] = string.lower(nil_to_default(options["html-no-end"], "false"))
      options["html-indent-lines"] = string.lower(nil_to_default(options["html-indent-lines"], "false"))

      local data_options = {}
      for k, v in pairs(options) do
        if string.match(k, "^html-") then
          data_k = string.gsub(k, "^html", "data")
          data_options[data_k] = v
        end
      end

      local inner_el = pandoc.Div(source_code)
      inner_el.attr.classes = pandoc.List()
      inner_el.attr.classes:insert("pseudocode")

      local outer_el = pandoc.Div(inner_el)
      outer_el.attr.classes = pandoc.List()
      outer_el.attr.classes:insert("pseudocode-container")
      outer_el.attr.classes:insert("quarto-float")
      outer_el.attr.attributes = data_options

      if algorithm_id then
        outer_el.attr.identifier = algorithm_id
        global_options.html_identifier_number_mapping[algorithm_id] = global_options.html_current_number
        global_options.html_current_number = global_options.html_current_number + 1
      end

      return outer_el
    end,
  }

  return filter
end

local function render_pseudocode_block_latex(global_options)
  ensure_latex_deps()

  if global_options.caption_number then
    quarto.doc.include_text("before-body", "\\floatname{algorithm}{" .. global_options.caption_prefix .. "}")
  else
    quarto.doc.include_text(
      "in-header",
      "\\DeclareCaptionLabelFormat{algnonumber}{" .. global_options.caption_prefix .. "}"
    )
    quarto.doc.include_text("before-body", "\\captionsetup[algorithm]{labelformat=algnonumber}")
  end

  if global_options.caption_align then
    if global_options.caption_align == "center" then
      quarto.doc.include_text("in-header", "\\captionsetup[algorithm]{justification=centering}")
    elseif global_options.caption_align == "right" then
      quarto.doc.include_text("in-header", "\\captionsetup[algorithm]{justification=raggedleft}")
    else
      quarto.doc.include_text("in-header", "\\captionsetup[algorithm]{justification=raggedright}")
    end
  end

  if global_options.number_with_in_chapter then
    quarto.doc.include_text("before-body", "\\numberwithin{algorithm}{chapter}")
  end

  local filter = {
    CodeBlock = function(el)
      if not el.attr.classes:includes("pseudocode") then
        return el
      end

      local options, source_code = extract_source_code_options(el.text, "pdf")
      local algpseudocodex_options = ""

      options["pdf-no-end"] = nil_to_default(options["pdf-no-end"], "false")
      options["pdf-indent-lines"] = nil_to_default(options["pdf-indent-lines"], "false")
      options["pdf-italic-comment"] = nil_to_default(options["pdf-italic-comment"], "true")
      options["pdf-right-comment"] = nil_to_default(options["pdf-right-comment"], "true")
      options["pdf-comment-color"] = nil_to_default(options["pdf-comment-color"], "black")
      options["pdf-comment-delimiter"] = nil_to_default(options["pdf-comment-delimiter"], "$\\triangleright$"):gsub("%%", "%%%%")
      
      if string.lower(options["pdf-no-end"]) == "true" then
        algpseudocodex_options = algpseudocodex_options
          .. [[
          \setbool{algpx@noEnd}{true}%%
          \algtext*{EndWhile}%%
          \algtext*{EndFor}%%
          \algtext*{EndLoop}%%
          \algtext*{EndIf}%%
          \algtext*{EndProcedure}%%
          \algtext*{EndFunction}%%
          \algtext*{EndStructure}%%
          \algtext*{EndClass}%%
          \algtext*{EndProperties}%%
          \algtext*{EndMethods}%%
          \pretocmd{\EndWhile}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndFor}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndLoop}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndIf}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndProcedure}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndFunction}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndStructure}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndClass}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndProperties}{\algpxEndIndentHelper}{}{}%%
          \pretocmd{\EndMethods}{\algpxEndIndentHelper}{}{}%%
        ]]
      end
      if string.lower(options["pdf-indent-lines"]) == "true" then
        algpseudocodex_options = algpseudocodex_options .. "\\setbool{algpx@indLines}{true}%%\n"
      end
      if string.lower(options["pdf-italic-comment"]) == "true" then
        algpseudocodex_options = algpseudocodex_options .. "\\setbool{algpx@italicComments}{true}%%\n"
      end
      if string.lower(options["pdf-right-comment"]) == "true" then
        algpseudocodex_options = algpseudocodex_options .. "\\setbool{algpx@rightComments}{true}%%\n"
      end
      if string.lower(options["pdf-comment-color"]) ~= "black" then
        algpseudocodex_options = algpseudocodex_options .. "\\algpxSetCommentColor{" .. options["pdf-comment-color"] .. "}%%\n"
      end
      if string.lower(options["pdf-comment-delimiter"]) ~= "//" then
        algpseudocodex_options = algpseudocodex_options
          .. "\\algpxSetBeginComment{"
          .. options["pdf-comment-delimiter"]
          .. "~}%%\n"
      end

      options["pdf-placement"] = nil_to_default(options["pdf-placement"], "H")
      source_code = string.gsub(
        source_code,
        "\\begin{algorithm}%s*\n",
        "\\begin{algorithm}[" .. options["pdf-placement"] .. "]\n"
      )

      if string.lower(nil_to_default(options["pdf-line-number"], "true")) == "true" then
        source_code =
          string.gsub(source_code, "\\begin{algorithmic}%s*\n", "\\begin{algorithmic}[1]\n" .. algpseudocodex_options)
      else
        source_code =
          string.gsub(source_code, "\\begin{algorithmic}%s*\n", "\\begin{algorithmic}[0]\n" .. algpseudocodex_options)
      end

      if options["label"] then
        source_code = string.gsub(source_code, "\\caption{", "\\caption{\\label{" .. options["label"] .. "}")
      end

      if string.find(source_code, "\\caption{") then
        source_code = "\\floatstyle{ruled}\n\\restylefloat{algorithm}\n"
          .. source_code
          .. "\n\\floatstyle{plain}\n"
      else
        source_code = "\\floatstyle{nocaption}\n\\restylefloat{algorithm}\n"
          .. source_code
          .. "\n\\floatstyle{plain}\n"
      end

      return pandoc.RawInline("latex", source_code)
    end,
  }

  return filter
end

-- ── EPUB: static rendering (pseudocode.js does not run in most e-readers) ─────
local function take_braced(s)
  local depth = 1
  for i = 1, #s do
    local c = s:sub(i, i)
    if c == "{" then
      depth = depth + 1
    elseif c == "}" then
      depth = depth - 1
      if depth == 0 then
        return s:sub(1, i - 1), s:sub(i + 1)
      end
    end
  end
  return s, ""
end

local function replace_braced(s, cmd, pre, post)
  local result = ""
  local pat = "\\" .. cmd .. "{"
  while true do
    local a, b = s:find(pat, 1, true)
    if not a then
      return result .. s
    end
    result = result .. s:sub(1, a - 1) .. pre
    local content, rest = take_braced(s:sub(b + 1))
    result = result .. content .. post
    s = rest
  end
end

local function remove_braced(s, cmd)
  local result = ""
  local pat = "\\" .. cmd .. "{"
  while true do
    local a, b = s:find(pat, 1, true)
    if not a then
      return result .. s
    end
    result = result .. s:sub(1, a - 1)
    local _, rest = take_braced(s:sub(b + 1))
    s = rest
  end
end

local function epub_step_md(line)
  local t = line:gsub("^%s+", ""):gsub("%s+$", "")
  if t == "" then
    return nil
  end
  local comment = nil
  local ca, cb = t:find("\\Comment{", 1, true)
  if ca then
    comment = take_braced(t:sub(cb + 1))
  end
  t = remove_braced(t, "Comment")
  t = t:gsub("\\quad", " "):gsub("\\;", " "):gsub("\\,", " "):gsub("\\!", "")
  t = replace_braced(t, "ElsIf", "**else if** ", " **then**")
  t = replace_braced(t, "If", "**if** ", " **then**")
  t = replace_braced(t, "For", "**for** ", " **do**")
  t = replace_braced(t, "While", "**while** ", " **do**")
  t = replace_braced(t, "Until", "**until** ", "")
  t = replace_braced(t, "textbf", "**", "**")
  t = replace_braced(t, "textit", "*", "*")
  t = t:gsub("\\Statex%s*", ""):gsub("\\State%s*", "")
  t = t:gsub("\\Require%s*", ""):gsub("\\Ensure%s*", "")
  t = t:gsub("\\Return%s*", "**return** ")
  t = t:gsub("\\EndFor", "**end for**"):gsub("\\EndWhile", "**end while**"):gsub("\\EndIf", "**end if**")
  t = t:gsub("\\EndProcedure", "**end procedure**"):gsub("\\EndFunction", "**end function**")
  t = t:gsub("\\Else", "**else**"):gsub("\\Repeat", "**repeat**")
  t = t:gsub("%s+$", "")
  if comment then
    comment = comment:gsub("\\quad", " "):gsub("\\;", " "):gsub("\\,", " ")
    t = t .. "  — *" .. comment .. "*"
  end
  if t == "" then
    return nil
  end
  return t
end

local function render_pseudocode_block_epub(global_options)
  local filter = {
    CodeBlock = function(el)
      if not el.attr.classes:includes("pseudocode") then
        return el
      end

      local _, source_code = extract_source_code_options(el.text, "html")
      local options = extract_source_code_options(el.text, "html")
      local algorithm_id = options["label"]

      local caption = global_options.caption_prefix
      local ca, cb = source_code:find("\\caption{", 1, true)
      if ca then
        caption = take_braced(source_code:sub(cb + 1))
      end

      local numtxt = ""
      if algorithm_id and global_options.caption_number then
        local n = tostring(global_options.html_current_number)
        if global_options.html_chapter_level then
          n = global_options.html_chapter_level .. "." .. n
        end
        numtxt = " " .. n
      end

      local body = source_code:match("\\begin{algorithmic}.-\n(.-)\\end{algorithmic}") or ""

      local md = { "**" .. global_options.caption_prefix .. numtxt .. "** " .. caption, "" }
      local reqlines = {}
      local steps = {}
      for line in (body .. "\n"):gmatch("([^\n]*)\n") do
        local trimmed = line:gsub("^%s+", "")
        if trimmed:find("^\\Require") then
          local c = epub_step_md((trimmed:gsub("^\\Require%s*", "")))
          if c then table.insert(reqlines, "**Require:** " .. c) end
        elseif trimmed:find("^\\Ensure") then
          local c = epub_step_md((trimmed:gsub("^\\Ensure%s*", "")))
          if c then table.insert(reqlines, "**Ensure:** " .. c) end
        else
          local s = epub_step_md(trimmed)
          if s then table.insert(steps, s) end
        end
      end

      if #reqlines > 0 then
        table.insert(md, table.concat(reqlines, "  \n"))
        table.insert(md, "")
      end
      for i, s in ipairs(steps) do
        table.insert(md, i .. ". " .. s)
      end

      local parsed = pandoc.read(table.concat(md, "\n"), "markdown")
      local div = pandoc.Div(parsed.blocks)
      div.attr.classes = pandoc.List()
      div.attr.classes:insert("pseudocode-epub")
      if algorithm_id then
        div.attr.identifier = algorithm_id
        global_options.html_identifier_number_mapping[algorithm_id] = global_options.html_current_number
        global_options.html_current_number = global_options.html_current_number + 1
      end
      return div
    end,
  }

  return filter
end

local function render_pseudocode_block(global_options)
  local filter = {
    CodeBlock = function(el)
      return el
    end,
  }

  if quarto.doc.is_format("epub") then
    filter = render_pseudocode_block_epub(global_options)
  elseif quarto.doc.is_format("html") then
    filter = render_pseudocode_block_html(global_options)
  elseif quarto.doc.is_format("latex") then
    filter = render_pseudocode_block_latex(global_options)
  end

  return filter
end

local function render_pseudocode_ref_html(global_options)
  local filter = {
    Cite = function(el)
      local cite_text = pandoc.utils.stringify(el.content)

      -- MIT Press casing: @algo-x -> lowercase prefix; @Algo-x -> capitalized prefix.
      local case, rest = string.match(cite_text, "^@([Aa]lgo)%-(.+)$")
      if not case then
        return nil
      end
      local key = "algo-" .. rest
      local v = global_options.html_identifier_number_mapping[key]
      if not v then
        return nil
      end

      local link_src = "#" .. key
      local algorithm_id = v
      if global_options.html_chapter_level then
        algorithm_id = global_options.html_chapter_level .. "." .. algorithm_id
      end

      local prefix = global_options.reference_prefix
      if case == "algo" then
        prefix = string.lower(prefix)
      end

      local link = pandoc.Link(prefix .. " " .. algorithm_id, link_src)
      link.attr.classes = pandoc.List()
      link.attr.classes:insert("quarto-xref")

      return link
    end,
  }

  return filter
end

local function render_pseudocode_ref_latex(global_options)
  local filter = {
    Cite = function(el)
      local cite_text = pandoc.utils.stringify(el.content)

      -- MIT Press casing: @algo-x -> lowercase prefix; @Algo-x -> capitalized prefix.
      local case, rest = string.match(cite_text, "^@([Aa]lgo)%-(.+)$")
      if case then
        local prefix = global_options.reference_prefix
        if case == "algo" then
          prefix = string.lower(prefix)
        end
        return pandoc.RawInline("latex", prefix .. "~\\ref{algo-" .. rest .. "}")
      end
    end,
  }

  return filter
end

local function render_pseudocode_ref(global_options)
  local filter = {
    Cite = function(el)
      return el
    end,
  }

  if quarto.doc.is_format("html") then
    filter = render_pseudocode_ref_html(global_options)
  elseif quarto.doc.is_format("latex") then
    filter = render_pseudocode_ref_latex(global_options)
  end

  return filter
end

function Pandoc(doc)
  local global_options = {
    caption_prefix = "Algorithm",
    reference_prefix = "Algorithm",
    caption_number = true,
    caption_align = "left",
    number_with_in_chapter = false,
    html_chapter_level = nil,
    html_current_number = 1,
    html_identifier_number_mapping = {},
  }

  if doc.meta["pseudocode"] then
    global_options.caption_prefix = pandoc.utils.stringify(
      nil_to_default(doc.meta["pseudocode"]["caption-prefix"], global_options.caption_prefix)
    )
    global_options.reference_prefix = pandoc.utils.stringify(
      nil_to_default(doc.meta["pseudocode"]["reference-prefix"], global_options.reference_prefix)
    )
    global_options.caption_number =
      nil_to_default(doc.meta["pseudocode"]["caption-number"], global_options.caption_number)
    global_options.caption_align = pandoc.utils.stringify(
      nil_to_default(doc.meta["pseudocode"]["caption-align"], global_options.caption_align)
    )
  end

  if doc.meta["book"] then
    global_options.number_with_in_chapter = true

    if quarto.doc.is_format("html") then
      local _, input_qmd_filename = string.match(quarto.doc["input_file"], "^(.-)([^\\/]-%.([^\\/%.]-))$")
      local renders = doc.meta["book"]["render"]

      for _, render in pairs(renders) do
        if
          render["file"]
          and render["number"]
          and pandoc.utils.stringify(render["file"]) == input_qmd_filename
        then
          global_options.html_chapter_level = pandoc.utils.stringify(render["number"])
        end
      end
    end
  end

  doc = doc:walk(render_pseudocode_block(global_options))

  return doc:walk(render_pseudocode_ref(global_options))
end
