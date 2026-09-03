# TinyTorch Release Testing

What must be true before a release ships, how to check it, and why each check
exists. Every gate below corresponds to a defect actually found in the
pre-release cleanup of 2026-09 — none are hypothetical.

## Run it

```bash
python3 tools/release_check.py          # all gates (~5 min)
python3 tools/release_check.py --fast   # skip the two slow gates (~10 s)
python3 tools/release_check.py --list   # show gates without running
python3 tools/release_check.py -k spine # run one gate
```

Exit code is 0 only when every non-advisory gate passes.

`--fast` is the pre-commit loop. The full run is the pre-release gate; it adds
the student-journey simulation and the pytest suite, which together take most of
the wall time.

---

## The gates

### Structure — every module looks the same

A student who learns the shape of one module should navigate any of the other
nineteen without relearning it.

| Gate | What it enforces |
|---|---|
| 20 modules, numbered 1–20, each with an export target | No gaps, and every module declares `#\| default_exp` |
| The 13-section spine, in order | 🔗 Prerequisites → 🎯 Learning Objectives → 📦 Where This Code Lives → 📋 Module Dependencies → 💡 Introduction → 📐 Foundations → 🏗️ Implementation → 🔧 Integration → 📊 Systems Analysis → 🧪 Module Integration Test → 🤔 Reflection Questions → ⭐ Aha Moment → 🚀 MODULE SUMMARY |
| No duplicate `##` headings | Two identical headings are indistinguishable in the Colab table of contents |
| Subtitled headings use `: ` | 154 used a colon, 26 used a dash |
| MODULE SUMMARY subsections, in order | Key Accomplishments → Systems Insights Discovered → Ready for Next Steps → Export with → **Next** |

**The emoji contract.** The emoji is what students internalize, so it must mean
the same thing everywhere: 💡 motivation, 📐 theory, 🏗️ things you build, 🔧
wiring them together, 📊 measurement, 🧪 tests, ⚠️ hazards. Module 15 once marked
four sections 🔧 that were each followed by an exercise; module 17 used 🔧 for a
section module 14 marks 📊. The spine gate catches missing sections; emoji
misuse inside a section is still a human review item.

### nbgrader — the metadata must be honest

| Gate | What it enforces |
|---|---|
| Exactly three canonical header shapes | exercise / graded test / given code, nothing else |
| `solution: true` implies `BEGIN/END SOLUTION` | 56 cells were marked as exercises with nothing to fill in — imports, setup, demos. They survived only because `ClearSolutions.enforce_metadata = False` |
| `grade_id` unique within a module | Duplicates make nbgrader's mapping ambiguous |

Current counts: **147 exercises, 185 graded tests, 67 given cells.**

### Pedagogy — the notebook must read

| Gate | What it enforces |
|---|---|
| Every exercise has a markdown cell before it | 15 exercises once had none. Module 06 had **seven consecutive** under a heading about a different function |
| Every unit test has a What/Why/Expected header | 19 of 20 modules did this; module 17 had none |
| Exercise docstrings carry TODO and APPROACH | The scaffold students follow |
| Reflection Questions render as markdown | Module 15's was a code cell showing a raw Python string |

### Progressive disclosure — nothing arrives early

| Gate | What it enforces |
|---|---|
| No module imports from a later-numbered module | Hard gate. Currently clean |
| Forward references are framed as previews | **Advisory** — a regex cannot judge a roadmap diagram or a mid-sentence pointer. Read the warnings; do not treat them as failures |

### Package surface — what students import must exist

| Gate | What it enforces |
|---|---|
| Every export target imports | All 20 `tinytorch.*` modules load |
| Every documented import resolves | Each "Where This Code Lives" block is executable. This is the gate that caught `OlympicEvent` never being exported because `#\| export` sat below an import line instead of at the top of its cell |
| No unreferenced zero-arg demo functions | 16 of 20 modules call their analysis functions under `if __name__ == "__main__"` so students see output. Seven were defined and never called. Functions that take arguments are utilities, not demos, and are exempt |

### Tests — a green suite must mean something

| Gate | What it enforces |
|---|---|
| No test signals failure with a bare `return` | pytest discards the value. **Thirteen tests could not fail; six were actually failing** |
| No bare `except:` | Six swallowed real failures. The worst skipped the gradient update entirely, turning "the network never trained" into a pass |
| Every test file imports and collects | Catches a stale import before the suite runs |

### Slow gates

| Gate | What it enforces |
|---|---|
| Student journey | All 20 notebooks execute end-to-end as `__main__` — every demo, analysis and test, exactly what "Run all" does in Colab |
| pytest | The full suite, green |

---

## Determinism

Nineteen `tinytorch` modules carry a module-level `rng = np.random.default_rng(7)`
that layer initialization draws from. It is shared and mutable, so how a layer
initializes depends on how many draws happened earlier in the session.

Two tests passed alone and failed inside the suite because of this. Both created
a local `rng = default_rng(7)` believing it seeded `Linear`; it did not, because
`Linear` reads its own module global. One even carried a comment saying the seed
was there to prevent exactly that flake.

`tests/conftest.py` now reseeds all nineteen generators before each test via an
autouse fixture. **If you add a module with a global RNG, the fixture finds it
automatically** — it walks the package rather than using a hardcoded list.

To confirm order-independence after a change:

```bash
python3 -m pytest tests/integration/test_training_capabilities.py -q -k xor_learning
python3 -m pytest tests/integration -q -k xor_learning
python3 -m pytest tests/ -q --ignore=tests/environment
```

All three must agree.

---

## Manual review — what the gates cannot check

1. **Is the explanation any good?** The gates verify a markdown cell exists
   before each exercise, not that it teaches.
2. **Is the emoji semantically right?** Enforced for missing sections, not for a
   section filed under the wrong emoji.
3. **Grading weights.** Points per exercise currently spans 4.3 (module 06, 21
   exercises for 91 points) to 90 (module 01, one exercise). Module totals span
   54 to 220. Deliberate authorial decision, not yet normalized.
4. **The 251 hand-copied Python blocks in the published chapters**, which nothing
   keeps in sync with `src/`.

---

## Adding a gate

When a defect is found, add the gate that would have caught it, in
`tools/release_check.py`:

```python
@gate("category: what must be true")
def g_short_name():
    errs = []
    ...                       # append one string per offender
    return errs               # empty list == pass
```

Use `slow=True` for anything over a few seconds, and `advisory=True` when the
check is a judgment call that should warn rather than block. The gate's message
should name the offender specifically enough to fix without re-deriving it.
