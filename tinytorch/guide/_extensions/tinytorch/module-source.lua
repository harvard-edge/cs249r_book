-- Shortcodes that read a module's source file (tinytorch/src/NN_name/NN_name.py)
-- at render time, so the page cannot drift from the code students receive.
--
--   {{< what-you-write >}}   table of the functions the student implements
--   {{< module-workflow >}}  start / resume / complete commands for this module
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

local function what_you_write(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local lines = read_source(meta_str(meta, "module"))
  local fns = student_functions(lines)
  local rows = { "| You implement | What it does |", "|---|---|" }
  for _, f in ipairs(fns) do
    rows[#rows + 1] = "| `" .. f.name .. "` | " .. f.doc:gsub("|", "\\|") .. " |"
  end
  local count = #fns == 1 and "one function" or (#fns .. " functions")
  local md = "The notebook arrives with the surrounding code already written. You write "
    .. count .. ", marked `YOUR CODE HERE`:\n\n" .. table.concat(rows, "\n") .. "\n"
  return quarto.utils.string_to_blocks(md)
end

local function module_workflow(args, kwargs, meta)
  if not quarto.doc.is_format("html") then return pandoc.Null() end
  local module = meta_str(meta, "module")
  local notebook = meta_str(meta, "notebook")
  local lines = read_source(module)
  local nn = module:match("^(%d%d)")
  local md = table.concat({
    "```bash",
    "tito module start " .. nn .. "      # create the notebook, open Jupyter Lab",
    "tito module resume " .. nn .. "     # reopen it in a later session",
    "tito module complete " .. nn .. "   # test, export, and record the module",
    "```",
    "",
    "Your notebook is `modules/" .. module .. "/" .. notebook .. ".ipynb`. "
      .. "`complete` stops at the first step that fails:",
    "",
    "1. the unit tests inside your notebook run;",
    "2. your code is exported into `" .. export_target(lines) .. "`;",
    "3. the integration tests in `tests/" .. module .. "/` run against that exported package;",
    "4. the module is recorded as done, and `tito module status` shows it.",
    "",
  }, "\n")
  return quarto.utils.string_to_blocks(md)
end

return {
  ["what-you-write"] = what_you_write,
  ["module-workflow"] = module_workflow,
}
