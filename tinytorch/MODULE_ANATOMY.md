# TinyTorch Module Anatomy

Every one of the 20 TinyTorch modules is built to the same shape. A student who
has finished Module 03 should open Module 12 and already know where the
prerequisites are, where the code goes, what a test cell looks like, and what the
last cell will do. This document is the specification for that shape. It is the
authority when a module, a review checklist, or an agent prompt disagree.

`tools/release_check.py --fast` enforces everything below that a script can
check. When you change a convention, change the gate and this document in the
same commit.

## 1. The pipeline: one source, three products

```
src/NN_name/NN_name.py          the source of truth (jupytext percent format)
        │
        │  jupytext --to ipynb            tito dev export
        ▼
modules/NN_name/name.ipynb      what the student opens in Jupyter (gitignored)
        │
        │  nbdev_export (#| export)       tito module complete NN
        ▼
tinytorch/<target>.py           the package the milestones and tests import (gitignored)
        │
        │  nbgrader assign (release tiers)
        ▼
modules/release/...             the graded student notebook, solutions stripped
```

Contributors edit only the `src/` file. The notebook and the package are
generated, ignored by git, and regenerated on every fresh checkout. Three
directives in the source drive the export:

| Directive | Where | Effect |
|---|---|---|
| `#| default_exp core.tensor` | first line of the imports cell | names the package target, one per module |
| `#| export` | first line of a code cell | the cell's code lands in the package and in `__all__` |
| `#| exporti` | first line of a code cell | the cell's code lands in the package but not in `__all__` (module-private helpers, Module 06's `backward` patches) |

A cell without a directive stays in the notebook only. That is the right home
for demos, analysis cells that print measurements, and tests. Everything a later
module or a milestone imports must carry `#| export` or `#| exporti`, and every
`solution: true` cell must carry one on its first line (gate:
`package: every solution cell opens with an export directive`).

The chapter listings in `narrative_book/` quote the source by symbol name, so a
renamed class or function must be followed by
`python3 narrative_book/tools/listings.py` to regenerate the quotes and
`--check` to confirm they match.

## 2. The spine: thirteen sections, one order

Every module has exactly these `##` sections, in this order. Nothing else is
allowed at `##` level; anything more is a `###` subsection of one of these.

| # | Heading | Cell type | Job |
|---|---|---|---|
| 1 | `## 🔗 Prerequisites & Progress` | markdown, in the title cell | **You've Built / You'll Build / You'll Enable** and a Connection Map |
| 2 | `## 🎯 Learning Objectives` | markdown, same cell | four to six numbered outcomes |
| 3 | `## 📦 Where This Code Lives in the Final Package` | markdown | learning side vs building side, the import line, why the target is where it is |
| 4 | `## 📋 Module Dependencies` | markdown | four bold labels, in order: **Prerequisites**, **External Dependencies**, **TinyTorch Dependencies**, **Dependency Flow** |
| | imports cell | code, `grade_id` `imports` | `#| default_exp` + `#| export`, then the imports from earlier modules |
| 5 | `## 💡 Introduction: <subtitle>` | markdown | why this module exists, in the student's terms |
| 6 | `## 📐 Foundations: <subtitle>` | markdown | the math and the mental model, before any code |
| 7 | `## 🏗️ Implementation: <subtitle>` and further `## 🏗️ <Component>: <subtitle>` | markdown + code | one section per component, each with its exercise and unit test |
| 8 | `## 🔧 Integration: <subtitle>` | markdown + code | the components working together; export wrappers and pitfalls live here as `###` |
| 9 | `## 📊 Systems Analysis: <subtitle>` | markdown + code | memory, time, and scaling measurements on what was just built |
| 10 | `## 🧪 Module Integration Test` | markdown + one locked test cell | `test_module()` runs every unit test then the integration scenarios |
| 11 | `## 🤔 ML Systems Reflection Questions` | markdown | `### Question N: <title>` numbered from 1, optional `### Bonus Question: <title>` |
| 12 | `## ⭐ Aha Moment: <subtitle>` | markdown + `demo_<module>()` + tail cell | one short demonstration of the payoff |
| 13 | `## 🚀 MODULE SUMMARY: <Module Name>` | markdown | `### Key Accomplishments`, `### Systems Insights Discovered`, `### Ready for Next Steps`, then `Export with:` and the closing `**Next**:` line |

Only the 🏗️ marker may repeat. A module with five components has five 🏗️
sections, the first titled `Implementation: ...` and the rest titled by the
component they build. The order 🔧 before 📊 is deliberate: the analysis
measures the assembled system, so the assembly has to exist first.

Module 20 (the capstone) has no 📊 section. It builds a benchmarking and
submission pipeline rather than a framework component, and its measurements are
the pipeline's output. That is the only exemption and the gate encodes it.

## 3. Heading grammar

- `##` headings are `<emoji> <Section>: <Subtitle>`. The subtitle separator is
  `: `, never ` - `. The four front-matter headings and the two test headings
  have no subtitle.
- `###` headings carry no emoji, except `### 🧪 Unit Test: <Name>`. The emoji
  belongs to the section, not to its parts, so a student scanning the notebook
  outline sees thirteen markers and knows where they are.
- `###` subtitles also use `: `. Reflection questions are
  `### Question N: <Title>`, numbered from 1 in order; the extra prompt, if any,
  is `### Bonus Question: <Title>`.
- `####` is for parts of a `###`, for example the sub-steps of a decomposed
  helper. Nothing goes deeper.

## 4. The exercise triplet

Each component the student writes is three cells, in this order:

**1. The lead-in (markdown).** A `###` heading named for the component, then
what the student is about to build and why, with the math or the algorithm if
the 📐 section did not already cover it. Every `solution: true` cell must be
preceded by a markdown cell (gate: `pedagogy: every exercise has a markdown cell
explaining it first`).

**2. The solution cell (code).**

```python
# %% nbgrader={"grade": false, "grade_id": "sigmoid-impl", "solution": true}
#| export
class SigmoidFunction(Function):
    def forward(self, x):
        """
        Apply sigmoid activation element-wise.

        TODO: Implement sigmoid function

        APPROACH:
        1. Apply sigmoid formula: 1 / (1 + exp(-x))
        2. ...

        EXAMPLE:
        >>> sigmoid = Sigmoid()
        >>> ...

        HINT: np.exp(-x) overflows for large negative x ...
        """
        ### BEGIN SOLUTION
        ...
        ### END SOLUTION
```

The docstring scaffold is `TODO`, `APPROACH`, `EXAMPLE`, `HINT`/`HINTS`, in
that order. `TODO` and `APPROACH` are required (gate: `pedagogy: docstring
scaffold present on exercises`); `EXAMPLE` and `HINT` are used wherever they
help. The scaffold is what remains when nbgrader strips the solution, so it has
to be enough for a student to start. Code between `### BEGIN SOLUTION` and
`### END SOLUTION` is the reference implementation and is removed from the
student release.

**3. The unit test (markdown + locked code).**

```markdown
### 🧪 Unit Test: Sigmoid

This test validates sigmoid activation behavior.

**What we're testing**: Sigmoid maps inputs to (0, 1) range
**Why it matters**: Ensures proper probability-like outputs
**Expected**: All outputs between 0 and 1, sigmoid(0) = 0.5
```

```python
# %% nbgrader={"grade": true, "grade_id": "test-sigmoid", "locked": true, "points": 10}
def test_unit_sigmoid():
    """🧪 Test Sigmoid implementation."""
    print("🧪 Unit Test: Sigmoid...")
    ...
    print("✅ Sigmoid works correctly!")

if __name__ == "__main__":
    test_unit_sigmoid()
```

The three bold labels are mandatory and are the same words in every module. The
test function is `test_unit_<name>`, prints `🧪 Unit Test: <Name>...` on entry
and a `✅` line on success, and runs itself under `__main__` so a student who
executes the cell sees the result immediately. A test that fails raises; it
never signals failure with a bare `return` or a swallowed exception.

Where a component is a class with several methods, each method may get its own
test (Module 09's Conv2d has four). The heading and print grammar do not change.

A family of small classes may share one test, but the run has a limit: no more
than five exercises may follow one another without a graded cell between them
(gate: `pedagogy: an exercise is followed by its test, not by more exercises`).
Module 14 once asked for nine profiling helpers in a row and tested them 1,100
lines later; Module 06 ran to ten. A student should not write five components
before learning whether the first one works.

## 5. The three nbgrader cell shapes

Every code cell that nbgrader sees uses exactly one of three headers:

```python
# %% nbgrader={"grade": false, "grade_id": "<id>", "solution": true}                  # exercise
# %% nbgrader={"grade": true,  "grade_id": "<id>", "locked": true, "points": <n>}      # test
# %% nbgrader={"grade": false, "grade_id": "<id>", "solution": false}                 # given code
```

Cells with no `nbgrader` header (`# %%`) are notebook-only: demos, analysis,
the tail runner. `grade_id`s are kebab-case, unique within a module, and named
for the thing they hold (`conv2d-class`, `test-conv2d-forward`). They are
stable identifiers that the grading tier files reference, so renaming one is a
change to the grading configuration, not a style edit.

`tests/validate_nbgrader_config.py` checks the header shapes and ids across all
20 modules; `NBGRADER_RELEASE_TIERS.md` describes which cells reach which
student tier.

## 6. The module test, the demo, and the tail

**`test_module()`** lives in the one locked cell under `## 🧪 Module Integration
Test`. It prints a banner, calls every `test_unit_*` function in the module in
order, then runs the integration scenarios, each announced with
`print("🧪 Integration Test: ...")`. It does not run itself; the tail cell does.

**`demo_<module>()`** lives in a plain `# %%` cell under `## ⭐ Aha Moment`. It
is a zero-argument function that prints one memorable result in under a second.
It is not exported. The `package: no unreferenced zero-arg demo functions` gate
makes sure every demo is called from the tail.

**The tail cell** is identical in all 20 modules and is the last code cell:

```python
# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_activations()
```

Running the notebook top to bottom therefore ends with the full test suite and
the demo. Importing the exported package runs neither.

## 7. Naming

| Thing | Convention | Example |
|---|---|---|
| Source file | `src/NN_name/NN_name.py` | `src/09_convolutions/09_convolutions.py` |
| Notebook | `modules/NN_name/name.ipynb` | `modules/09_convolutions/convolutions.ipynb` |
| Export target | `core.<name>` for 01 to 13 (Module 09 is `core.spatial`), `perf.<name>` for 14 to 19, `olympics` for 20 | `#| default_exp perf.quantization` |
| Public class | CapWords, PyTorch name where one exists | `Conv2d`, `CrossEntropyLoss`, `KVCache` |
| Operation class | `<Name>Function(Function)` with `forward`/`backward` | `SigmoidFunction` |
| Private helper | leading underscore, `#| exporti` if the package needs it | `_stable_softmax` |
| Unit test | `test_unit_<snake_name>` | `test_unit_conv2d_forward` |
| Module test | `test_module` | |
| Demo | `demo_<module_name>` | `demo_convolutions` |
| Summary heading | `## 🚀 MODULE SUMMARY: <Module Name>` | `MODULE SUMMARY: Convolutions` |

Progressive disclosure is a naming rule too: a module imports only from
lower-numbered modules (gate: `disclosure: no module imports from a
later-numbered module`), and any mention of a later module in prose is framed as
a preview ("Module 06 will add...").

## 8. What is deliberately not standardized

- **Analysis and demo cells** are plain `# %%` cells with whatever prints make
  the measurement clear. They are not exported and not graded.
- **The number of 🏗️ sections** follows the number of components. Module 02 has
  one; Module 15 has twelve.
- **Section subtitles** are written for the module, not from a template. The
  grammar is fixed; the words are the author's.
- **`EXAMPLE` and `HINT`** appear where they help a student start, and are
  omitted where the `APPROACH` steps already say everything.
- **`grade_id`s** were left as they were during the 2026-09 anatomy pass even
  where a newer naming would read better, because the release tiers reference
  them.

## 9. Decisions and their reasons

These are the calls made during the 2026-09-08 anatomy pass, recorded so they
are not relitigated one module at a time.

- **🔧 before 📊.** Five modules had the analysis before the integration. The
  analysis measures the assembled thing, so assembly comes first; this also
  matches the order the spine gate had claimed but never checked.
- **Extras fold down, never up.** Eight modules had grown top-level sections
  outside the spine (a second Foundations, an "Optimization Insights", a
  "Consolidated Classes for Export", a pitfalls warning). Each became a `###`
  under the spine section it belongs to. The content is unchanged; the outline a
  student sees is the same thirteen markers in every module.
- **Tests sit beside their code.** Module 20 had gathered its seven unit tests
  into a trailing "Unit Tests" section. They now follow the component they test,
  as in the other nineteen. Module 13's causal-mask helper moved from the front
  matter into Implementation, just above the TransformerBlock test that
  first calls it (a first placement below that test broke the top-to-bottom
  notebook run, which is why the runner-order gate now exists).
- **One test emoji.** 🧪 marks every test heading and print. 🔬 had crept into
  two modules and the review checklist; it is gone from both.
- **`###` headings carry no emoji** beyond the unit-test marker, so the notebook
  outline stays readable at a glance.
- **`Question N:`** replaced four competing reflection-heading styles.
- **The summary ends on `**Next**:`.** Four modules had trailing prose after
  it; the prose moved up into "Ready for Next Steps." The last line a student
  reads points at the next module or milestone.
- **Adam and AdamW stay as two classes** with the moment code repeated
  (Module 07). A subclass would be shorter and worse to teach; the book records
  the same choice.

- **Dependencies before imports.** The 2026-09-11 source audit found that heading
  checks could pass while setup code still ran before its dependency explanation.
  A separate release gate now enforces the imports cell position. Stable grading
  identifiers are retained, including historical setup-cell names.
- **Missing code is a failure.** Progressive tests must fail when a required
  implementation is missing or has the wrong API; an exception handler cannot
  replace that failure with `assert True`. A release gate checks this pattern.

## 10. Checking a module

```bash
python3 -m tito.main dev export --all         # regenerate reference notebooks and exports
python3 tools/release_check.py --fast          # 33 gates (two slow gates omitted)
python3 tests/validate_nbgrader_config.py      # expect Passed: 20, Failed: 0
python3 narrative_book/tools/listings.py --check
python3 tools/release_check.py                 # adds the notebook run and full pytest
```

The source-built regression gate checks the instructor solutions in a temporary
package. The notebook journey uses a fresh interpreter for each module and
exports only earlier modules, so later implementations cannot hide a missing
prerequisite.

The two slow gates (the top-to-bottom notebook run and the full pytest)
are part of the release, not optional: run the full command before tagging.
The gate names say what they check. A new convention gets a new gate in the
same file, with a dated comment saying which defect it was added to catch; every
gate there corresponds to a real one.
