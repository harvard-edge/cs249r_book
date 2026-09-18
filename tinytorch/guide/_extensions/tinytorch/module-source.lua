-- Shortcodes that read a module's source file (tinytorch/src/NN_name/NN_name.py)
-- at render time, so the page cannot drift from the code students receive.
--
--   {{< what-you-write >}}   table of the functions the student implements
--   {{< module-workflow >}}  start / resume / complete commands for this module
--   {{< module-checks >}}    the tests `complete` runs, and the milestone it unlocks
--   {{< book-chapter >}}     pointer to the companion book chapter, by its real title
--
-- Both read `module` (e.g. 01_tensor) and `notebook` (e.g. tensor) from the
-- page front matter, the same keys module-header uses.

local function meta_str(meta, key)
  local v = meta[key]
  if v == nil then
    error("tinytorch shortcode: page front matter is missing '" .. key .. "'")
  end
  return pandoc.utils.stringify(v)
end

local function read_source(module)
  local root = quarto.project.directory
  local path = pandoc.path.join({ root, "..", "src", module, module .. ".py" })
  local f = io.open(path, "r")
  if not f then
    error("tinytorch shortcode: cannot read " .. path)
  end
  local lines = {}
  for line in f:lines() do lines[#lines + 1] = line end
  f:close()
  return lines
end

local function indent_of(s) return #(s:match("^(%s*)")) end

-- The class a function belongs to, by the three patterns the modules use:
-- defined inside the class, decorated with @method_of(Class), or assigned
-- afterwards with `Class.name = fn`.
local function owner(lines, def_i, fn, fn_indent)
  local dec = def_i > 1 and lines[def_i - 1]:match("^%s*@method_of%((%w+)%)")
  if dec then return dec, fn end
  if fn_indent > 0 then
    for j = def_i - 1, 1, -1 do
      local ci, cname = lines[j]:match("^(%s*)class (%w+)")
      if cname and #ci < fn_indent then return cname, fn end
    end
    return nil, fn
  end
  for _, l in ipairs(lines) do
    local cls, name = l:match("^(%w+)%.([%w_]+) = " .. fn:gsub("%p", "%%%0") .. "$")
    if cls then return cls, name end
  end
  return nil, fn
end

local function docstring_summary(lines, def_i)
  local k = def_i
  while k <= #lines and not lines[k]:match(":%s*$") do k = k + 1 end
  local first = (lines[k + 1] or ""):gsub('^%s*"""', ""):gsub('"""%s*$', ""):match("^%s*(.-)%s*$")
  if first == "" then first = (lines[k + 2] or ""):match("^%s*(.-)%s*$") end
  return first
end

-- Every solution region the student tier strips: plain markers and
-- role="core". role="scaffold" regions ship pre-solved and are skipped.
local function student_functions(lines)
  local found = {}
  for i, l in ipairs(lines) do
    local ws, rest = l:match("^(%s*)### BEGIN SOLUTION(.*)$")
    if ws and not rest:find("scaffold", 1, true) then
      for j = i - 1, 1, -1 do
        local di, fn = lines[j]:match("^(%s*)def ([%w_]+)%(")
        if fn and #di < #ws then
          local cls, name = owner(lines, j, fn, #di)
          found[#found + 1] = {
            name = cls and (cls .. "." .. name) or name,
            doc = docstring_summary(lines, j),
          }
          break
        end
      end
    end
  end
  return found
end

local function export_target(lines)
  for _, l in ipairs(lines) do
    local t = l:match("^#| default_exp ([%w_.]+)")
    if t then return "tinytorch." .. t end
  end
  error("tinytorch shortcode: no '#| default_exp' line in module source")
end


local function read_lines(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local lines = {}
  for line in f:lines() do lines[#lines + 1] = line end
  f:close()
  return lines
end

local function tinytorch_root()
  return pandoc.path.join({ quarto.project.directory, ".." })
end

-- The files `tito module complete` hands to pytest, chosen by the same rule as
-- tito/commands/module/workflow.py (_run_integration_tests).
local function integration_tests(module)
  local dir = pandoc.path.join({ tinytorch_root(), "tests", module })
  local ok, entries = pcall(pandoc.system.list_directory, dir)
  if not ok then return {} end
  table.sort(entries)
  local primary = "test_" .. module .. "_progressive.py"
  for _, e in ipairs(entries) do
    if e == primary then return { e } end
  end
  local picked = {}
  for _, e in ipairs(entries) do
    if e:match("^test_.*_progressive%.py$") then picked[#picked + 1] = e end
  end
  if #picked > 0 then return picked end
  for _, e in ipairs(entries) do
    if e:match("^test_.*%.py$") then picked[#picked + 1] = e end
  end
  return picked
end

local function count_tests(module, files)
  local n = 0
  for _, f in ipairs(files) do
    for _, l in ipairs(read_lines(pandoc.path.join({ tinytorch_root(), "tests", module, f })) or {}) do
      if l:match("^%s*def test_") then n = n + 1 end
    end
  end
  return n
end

-- Milestones whose highest required module is this one, parsed from the
-- top-level "required_modules" of each entry in tito/commands/milestone.py.
local function unlocked_milestones(module_number)
  local lines = read_lines(pandoc.path.join({ tinytorch_root(), "tito", "commands", "milestone.py" }))
  if not lines then error("tinytorch shortcode: cannot read tito/commands/milestone.py") end
  local found, current, name = {}, nil, nil
  for _, l in ipairs(lines) do
    local id = l:match('^    "(%d%d)": {')
    if id then current, name = id, nil end
    if current then
      local nm = l:match('^        "name": "(.-)"')
      if nm then name = nm end
      local req = l:match('^        "required_modules": %[(.-)%]')
      if req then
        local maxm = 0
        for d in req:gmatch("%d+") do maxm = math.max(maxm, tonumber(d)) end
        if maxm == module_number then found[#found + 1] = { id = current, name = name } end
        current = nil
      end
    end
  end
  return found
end


local function milestone_page(id)
  local dir = pandoc.path.join({ quarto.project.directory, "milestones" })
  local ok, entries = pcall(pandoc.system.list_directory, dir)
  if ok then
    table.sort(entries)
    for _, e in ipairs(entries) do
      if e:match("^" .. id .. "_.*%.qmd$") then return e end
    end
  end
  error("tinytorch shortcode: no site page for milestone " .. id)
end

local function module_checks(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local module = meta_str(meta, "module")
  local lines = read_source(module)
  local units = {}
  for _, l in ipairs(lines) do
    local name = l:match("^### 🧪 Unit Test: (.+)$")
    if name then units[#units + 1] = name:gsub("%s+$", "") end
  end
  local files = integration_tests(module)
  local out = {
    "`tito module complete` stops at the first step that fails:",
    "",
    "1. the unit tests inside your notebook run;",
    "2. your code is exported into `" .. export_target(lines) .. "`;",
    "3. the integration tests run against that exported package, together with the modules before it;",
    "4. the module is recorded as done, and `tito module status` shows it.",
    "",
    "**Unit tests in your notebook (" .. #units .. ").** Each prints a ✅ line when it passes.",
    "",
  }
  out[#out + 1] = "::: {.unit-test-list}"
  for _, u in ipairs(units) do
    -- test names are labels, not code; escape Markdown so __init__ stays literal
    out[#out + 1] = "- " .. u:gsub("([_*`])", "\\%1")
  end
  out[#out + 1] = ":::"
  if #files > 0 then
    out[#out + 1] = ""
    out[#out + 1] = "**Integration tests after export (" .. count_tests(module, files) .. ").**"
    out[#out + 1] = ""
    for _, f in ipairs(files) do out[#out + 1] = "- `tests/" .. module .. "/" .. f .. "`" end
  end
  local nn = tonumber(module:match("^(%d%d)"))
  local ms = unlocked_milestones(nn)
  if #ms > 0 then
    local items = {}
    for _, m in ipairs(ms) do
      items[#items + 1] = "[Milestone " .. m.id .. ", " .. m.name .. "](../milestones/" .. milestone_page(m.id) .. ") (`tito milestone run " .. m.id .. "`)"
    end
    out[#out + 1] = ""
    out[#out + 1] = "Completing this module unlocks " .. table.concat(items, " and ") .. "."
  end
  return quarto.utils.string_to_blocks(table.concat(out, "\n") .. "\n")
end

local BOOK_PDF = "https://mlsysbook.ai/tinytorch/assets/downloads/TinyTorch-Book.pdf"

local function book_chapter(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local module = meta_str(meta, "module")
  local nn = module:match("^(%d%d)")
  local dir = pandoc.path.join({ tinytorch_root(), "book" })
  local ok, entries = pcall(pandoc.system.list_directory, dir)
  if not ok then error("tinytorch shortcode: cannot list " .. dir) end
  table.sort(entries)
  local title
  for _, e in ipairs(entries) do
    if e:match("^" .. nn .. "_.*%.qmd$") then
      local first = (read_lines(pandoc.path.join({ dir, e })) or {})[1] or ""
      title = first:match("^#%s+(.-)%s*{") or first:match("^#%s+(.-)%s*$")
      break
    end
  end
  if not title then error("tinytorch shortcode: no book chapter for module " .. nn) end
  local md = "The reasoning behind this module (why it is built this way, what it costs, "
    .. "and how production frameworks differ) is the chapter *" .. title
    .. "* in the companion book, [*TinyTorch: From Tensors to Transformers*](" .. BOOK_PDF .. ") (PDF). "
    .. "The book prints complete reference implementations, so read it after you finish the module, not while you are working on it.\n"
  return quarto.utils.string_to_blocks(md)
end

local function what_you_write(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local lines = read_source(meta_str(meta, "module"))
  local fns = student_functions(lines)
  local items = {}
  for _, f in ipairs(fns) do
    items[#items + 1] = "`" .. f.name .. "`\n:   " .. f.doc
  end
  local count = #fns == 1 and "one function" or (#fns .. " functions")
  local md = "The notebook arrives with the surrounding code already written and explained. You write "
    .. count .. ", each marked `# YOUR CODE HERE` and followed by a test cell:\n\n"
    .. table.concat(items, "\n\n") .. "\n"
  local block = pandoc.Div(quarto.utils.string_to_blocks(md), pandoc.Attr("", {"what-you-write"}))
  return block
end

local function module_workflow(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local module = meta_str(meta, "module")
  local notebook = meta_str(meta, "notebook")
  local nn = module:match("^(%d%d)")
  local md = table.concat({
    "```bash",
    "# first time",
    "tito module start " .. nn,
    "",
    "# later sessions",
    "tito module resume " .. nn,
    "",
    "# when your tests pass",
    "tito module complete " .. nn,
    "```",
    "",
    "Your notebook is `modules/" .. module .. "/" .. notebook .. ".ipynb`.",
    "",
  }, "\n")
  return quarto.utils.string_to_blocks(md)
end

local function stub_failure(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  return quarto.utils.string_to_blocks(
    "A bare `NotImplementedError` with no message means a cell reached a function you have not written yet: "
    .. "the notebook ships each one as `# YOUR CODE HERE` followed by `raise NotImplementedError()`. "
    .. "The messages below are ones this module actually prints when an implementation is present but wrong.\n")
end

return {
  ["what-you-write"] = what_you_write,
  ["module-workflow"] = module_workflow,
  ["module-checks"] = module_checks,
  ["book-chapter"] = book_chapter,
  ["stub-failure"] = stub_failure,
}
