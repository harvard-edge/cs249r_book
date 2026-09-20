# Development Tools

This directory contains tools for TinyTorch maintainers and contributors.

## Structure

- **`release_check.py`** - The pre-release gate suite. `--fast` runs the 34 quick
  gates; the full run adds the top-to-bottom notebook journey and the pytest
  suite. `--list` prints every gate. See `RELEASE_TESTING.md`.
- **`check_reference.py`** - Builds the instructor solutions into a temporary
  package and runs the numerical and training regressions against it, so a stale
  local export cannot make the check pass.
- **`validate_nbgrader_config.py`** - Checks the nbgrader cell-header shapes and
  `grade_id`s across all 20 modules. Expect `Passed: 20, Failed: 0`.
- **`check_prose_solution_leaks.py`** - Fails if a graded solution is reprinted in
  prose outside its `BEGIN/END SOLUTION` region.
- **`check_grading_holes.py`** - Reports graded cells that pass on an empty student
  notebook, which happens when a cell's assertions only reach regions tagged
  `role="scaffold"` and therefore shipped pre-solved.
- **`dev/`** - Development environment setup and utilities.

## For Students

Students don't need anything in this directory. Use the main setup scripts in the
project root.

## For Developers

- `RELEASE_TESTING.md` in this directory describes what each gate covers and what
  it deliberately does not.
- `../MODULE_ANATOMY.md` is the specification for how a module is built, and wins
  whenever a module, a checklist, or an agent prompt disagrees with it.
- `../CONTRIBUTING.md` covers the branch and release workflow.
