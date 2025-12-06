# TinyTorch Formatting Standards

This document defines the consistent formatting and style standards for all TinyTorch modules.

## Overview

All 20 TinyTorch modules follow consistent patterns to provide students with a uniform learning experience. This guide documents the standards discovered through comprehensive review of the codebase.

## ✅ Current Status

**Modules Reviewed**: 20/20
**Overall Grade**: 9/10 (Excellent)
**Last Updated**: 2025-11-24

---

## 1. Test Function Naming

### ✅ Current Standard (ALL 20 MODULES COMPLIANT)

```python
# Unit tests - test individual functions/features
def test_unit_feature_name():
    """🔬 Unit Test: Feature Name"""
    # Test code here

# Module integration test - ALWAYS named test_module()
def test_module():
    """🧪 Module Test: Complete Integration"""  # ⚠️ Currently missing emoji in all modules
    # Integration test code
```

### Rules

1. **Unit tests**: Always prefix with `test_unit_`
2. **Integration test**: Always named exactly `test_module()` (never `test_unit_all()` or `test_integration()`)
3. **Docstrings**:
   - Unit tests: Start with `🔬 Unit Test:`
   - Module test: Start with `🧪 Module Test:` (currently needs fixing)

### Status
- ✅ All 20 modules use correct `test_module()` naming
- ⚠️ All 20 modules missing 🧪 emoji in `test_module()` docstrings
- ✅ Most unit test functions have 🔬 emoji

---

## 2. `if __name__ == "__main__"` Guards

### ✅ Current Standard (18/20 MODULES COMPLIANT)

```python
def test_unit_something():
    """🔬 Unit Test: Something"""
    print("🔬 Unit Test: Something...")
    # test code
    print("✅ test_unit_something passed!")

# IMMEDIATELY after function definition
if __name__ == "__main__":
    test_unit_something()

# ... more functions ...

def test_module():
    """🧪 Module Test: Complete Integration"""
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    # Run all unit tests
    test_unit_something()
    # ... more tests ...
    print("🎉 ALL TESTS PASSED!")

# Final integration guard
if __name__ == "__main__":
    test_module()
```

### Rules

1. **Every test function** gets an `if __name__` guard immediately after
2. **Analysis functions** also get guards to prevent execution on import
3. **Final module test** has guard at end of file
4. **More guards than test functions** is OK (protects analysis functions too)

### Status
- ✅ 18/20 modules have adequate guards
- ⚠️ Module 08 (dataloader): 6 test functions, 5 guards (1 missing)
- ⚠️ Module 16 (compression): 7 test functions, 1 guard (6 missing - needs immediate attention)

---

## 3. Emoji Protocol

### Standard Emoji Usage

```python
# Implementation sections
🏗️ Implementation      # For new components being built

# Testing
🔬 Unit Test          # ALWAYS for test_unit_*() functions
🧪 Module Test        # ALWAYS for test_module() (currently missing in ALL modules)

# Analysis & Performance
📊 Analysis           # ALWAYS for analyze_*() functions
⏱️ Performance        # Timing/benchmarking analysis
🧠 Memory            # Memory profiling

# Educational markers
💡 Key Insight        # Important "aha!" moments
🤔 Assessment         # Reflection questions
📚 Background         # Theory/context

# System markers
⚠️ Warning            # Common mistakes/pitfalls
🚀 Production         # Real-world patterns
🔗 Connection         # Module relationships
✅ Success            # Test passed
❌ Failure            # Test failed
```

### Rules

1. **Test docstrings**: MUST start with emoji
2. **Print statements**: Use emojis for visual clarity
3. **Section headers**: Use emojis sparingly in markdown cells

### Current Issues (⚠️ NEEDS FIXING)

All 20 modules are missing the 🧪 emoji in `test_module()` docstrings.

**Before**:
```python
def test_module():
    """
    Comprehensive test of entire module functionality.
    """
```

**After**:
```python
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.
    """
```

---

## 4. Markdown Cell Formatting

### ✅ Current Standard (ALL MODULES COMPLIANT)

```python
# %% [markdown]
"""
## Section Title

Clear explanation with **formatting**.

### Subsection

More content...

### Visual Diagrams

```
ASCII art here
```

Key points:
- Point 1
- Point 2
"""
```

### Rules

1. **Use Jupytext format**: `# %% [markdown]` with triple-quote strings
2. **NEVER use Jupyter JSON**: No `<cell id="...">` format in .py files
3. **Hierarchical headers**: Use `##` for main sections, `###` for subsections
4. **Code formatting**: Use triple backticks for code examples

### Status
- ✅ All modules use proper Jupytext format
- ✅ No Jupyter JSON leakage found

---

## 5. ASCII Diagram Standards

### Excellent Examples Found

**Module 01 - Tensor Dimensions**:
```python
"""
Tensor Dimensions:
┌─────────────┐
│ 0D: Scalar  │  5.0          (just a number)
│ 1D: Vector  │  [1, 2, 3]    (list of numbers)
│ 2D: Matrix  │  [[1, 2]      (grid of numbers)
│             │   [3, 4]]
│ 3D: Cube    │  [[[...       (stack of matrices)
└─────────────┘
```

**Module 01 - Matrix Multiplication**:
```python
"""
Matrix Multiplication Process:
    A (2×3)      B (3×2)         C (2×2)
   ┌       ┐    ┌     ┐       ┌         ┐
   │ 1 2 3 │    │ 7 8 │       │ 1×7+2×9+3×1 │   ┌      ┐
   │       │ ×  │ 9 1 │  =    │             │ = │ 28 13│
   │ 4 5 6 │    │ 1 2 │       │ 4×7+5×9+6×1 │   │ 79 37│
   └       ┘    └     ┘       └             ┘   └      ┘
```

**Module 12 - Attention Matrix**:
```python
"""
Attention Matrix (after softmax):
        The   cat   sat  down
The   [0.30  0.20  0.15  0.35]  ← "The" attends mostly to "down"
cat   [0.10  0.60  0.25  0.05]  ← "cat" focuses on itself and "sat"
sat   [0.05  0.40  0.50  0.05]  ← "sat" attends to "cat" and itself
down  [0.25  0.15  0.10  0.50]  ← "down" focuses on itself and "The"
```

### Rules

1. **Use box-drawing characters**: `┌─┐│└─┘` for consistency
2. **Align multi-step processes** vertically
3. **Add arrows** (`→`, `↓`, `↑`, `←`) to show data flow
4. **Label dimensions** clearly in every diagram
5. **Include semantic explanation** (like attention example above)

### Status
- ✅ Most modules have excellent diagrams
- 🟡 Module 09 (spatial): Minor alignment inconsistencies
- 💡 Opportunity: Add more diagrams to complex operations

---

## 6. Module Structure Template

### Standard Module Layout

```python
# --- HEADER ---
# jupytext metadata
# #| default_exp directive
# #| export marker

# --- SECTION 1: INTRODUCTION ---
# %% [markdown]
"""
# Module XX: Title - Tagline

Introduction and context...

## 🔗 Prerequisites & Progress
...

## Learning Objectives
...
"""

# --- SECTION 2: IMPORTS ---
# %%
#| export
import numpy as np
# ... other imports

# --- SECTION 3: PEDAGOGICAL CONTENT ---
# %% [markdown]
"""
## Part 1: Foundation - Topic
...
"""

# --- SECTION 4: IMPLEMENTATION ---
# %%
#| export
def function_or_class():
    """Docstring with TODO, APPROACH, HINTS"""
    ### BEGIN SOLUTION
    # implementation
    ### END SOLUTION

# --- SECTION 5: TESTING ---
# %%
def test_unit_feature():
    """🔬 Unit Test: Feature"""
    print("🔬 Unit Test: Feature...")
    # test code
    print("✅ test_unit_feature passed!")

if __name__ == "__main__":
    test_unit_feature()

# --- SECTION 6: SYSTEMS ANALYSIS ---
# %%
def analyze_performance():
    """📊 Analysis: Performance Characteristics"""
    print("📊 Analyzing performance...")
    # analysis code

if __name__ == "__main__":
    analyze_performance()

# --- SECTION 7: MODULE INTEGRATION ---
# %%
def test_module():
    """🧪 Module Test: Complete Integration"""  # ⚠️ ADD EMOJI
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    test_unit_feature()
    # ... more tests
    print("🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    test_module()

# --- SECTION 8: REFLECTION ---
# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions
...
"""

# --- SECTION 9: SUMMARY ---
# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Module Title
...
"""
```

---

## Priority Fixes Needed

### 🔴 HIGH PRIORITY (Quick Wins)

1. **Add 🧪 emoji to all `test_module()` docstrings** (~5 minutes)
   - Affects: All 20 modules
   - Pattern: Add "🧪 Module Test:" to first line of docstring

2. **Fix Module 16 (compression) `if __name__` guards** (~15 minutes)
   - Missing guards for 6 out of 7 test functions

### 🟡 MEDIUM PRIORITY

3. **Align ASCII diagrams in Module 09** (~30 minutes)
   - Minor visual consistency improvements

4. **Review Module 08 for missing guard** (~5 minutes)
   - Identify which test function needs guard

### 🟢 LOW PRIORITY (Enhancements)

5. **Add more ASCII diagrams** (~2-3 hours)
   - Target complex operations without visual aids
   - Modules: 05, 06, 07, 13, 14, 15

6. **Create diagram style guide** (~1 hour)
   - Document best practices with examples
   - Add to CONTRIBUTING.md

---

## Validation Checklist

When creating or modifying a module, verify:

- [ ] Test functions follow naming convention (`test_unit_*`, `test_module`)
- [ ] Test docstrings have correct emojis (🔬 for unit, 🧪 for module)
- [ ] Every test function has `if __name__` guard immediately after
- [ ] Markdown cells use Jupytext format (`# %% [markdown]`)
- [ ] ASCII diagrams are aligned and use proper box-drawing characters
- [ ] Systems analysis functions have `if __name__` protection
- [ ] Module structure follows standard template
- [ ] `#| export` markers are placed correctly
- [ ] NBGrader cell markers (`### BEGIN SOLUTION`, `### END SOLUTION`) are present

---

## Implementation Status

| Priority | Fix | Time | Modules Affected | Status |
|----------|-----|------|------------------|--------|
| 🔴 HIGH | Add 🧪 to test_module() | 5 min | All 20 | ⏳ Pending |
| 🔴 HIGH | Fix Module 16 guards | 15 min | 1 (Module 16) | ⏳ Pending |
| 🟡 MEDIUM | Fix Module 08 guard | 5 min | 1 (Module 08) | ⏳ Pending |
| 🟡 MEDIUM | Align Module 09 diagrams | 30 min | 1 (Module 09) | ⏳ Pending |
| 🟢 LOW | Add more diagrams | 2-3 hrs | Multiple | 💡 Enhancement |

**Total Quick Fixes**: 25 minutes
**Total Enhancements**: 3-4 hours

---

## Conclusion

The TinyTorch codebase is in **excellent shape** with strong consistency across all 20 modules. The formatting standards are well-established and largely followed. The few remaining issues are minor and can be resolved with minimal effort.

**Current Grade**: 9/10
**With Quick Fixes**: 10/10

---

*Generated by comprehensive module review - 2025-11-24*
*Review conducted by: module-developer agent*
*Coordinated by: technical-program-manager agent*
