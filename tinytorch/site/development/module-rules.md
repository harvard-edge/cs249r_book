# TinyTorch Module Development Rules

**Version**: 2.0
**Date**: January 2025
**Status**: Complete Reference Guide
**Reference Implementation**: `src/08_optimizers/08_optimizers.py`

This document defines the complete set of rules, patterns, and conventions for developing TinyTorch modules. Instead of maintaining separate documentation, **use `08_optimizers` as your reference implementation** - it follows all current patterns perfectly.

## Development Structure

**Key Paths:**
- **Source**: `src/NN_modulename/NN_modulename.py` ← Developers edit here
- **Generated**: `modules/NN_modulename/NN_modulename.ipynb` ← Auto-generated for students
- **Package**: `tinytorch/` ← Auto-generated package code

## 📚 Educational Philosophy

### Core Principles
1. **Educational First**: Every module is designed for learning, not just functionality
2. **Progressive Complexity**: Start simple, build complexity step by step
3. **Real-World Connection**: Connect concepts to practical ML applications
4. **Standalone Learning**: Each module should be self-contained
5. **Professional Standards**: Use industry-standard patterns and practices

### "Build → Use → [Understand/Reflect/Analyze/Optimize]" Framework
Each module follows this pedagogical pattern:
- **Build**: Implement the component from scratch
- **Use**: Apply it to real data and problems
- **Third Stage**: Varies by module (Understand/Reflect/Analyze/Optimize)

## 📁 File Structure and Organization

### 1. **File Naming ConventionDeveloper Workflow (Source):**
```
src/NN_modulename/
├── NN_modulename.py           # SOURCE OF TRUTH (developers edit)
├── module.yaml                # Module configuration
├── ABOUT.md                   # Module documentation
└── README.md                  # Optional additional docs
```

**Student Workflow (Generated):**
```
modules/NN_modulename/
└── NN_modulename.ipynb        # AUTO-GENERATED notebook (students use)
```

### 2. **File Format: Jupytext Percent Format**
All source Python files in `src/` MUST use Jupytext percent format:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---
```

**Build Process:**
1. Developer edits `src/NN_name/NN_name.py`
2. Run `tito src export NN_name`
3. Generates `modules/NN_name/NN_name.ipynb` (notebook)
4. Exports to `tinytorch/` (package code)

##  Module Template Structure

**Follow this exact structure** (see `08_optimizers` for reference):

### A. **Header Section**
```python
# %% [markdown]
"""
# Module N: Title - Brief Description

## Learning Goals
- Goal 1: Specific outcome
- Goal 2: Another objective
- Goal 3: Connection to ML concepts

## Build → Use → [Understand/Reflect/Analyze/Optimize]
1. **Build**: What students implement
2. **Use**: How they apply it
3. **[Third Stage]**: Deeper engagement
"""
```

### B. **Setup and Imports**
```python
# %% nbgrader={"grade": false, "grade_id": "modulename-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.modulename

#| export
import numpy as np
import sys
from typing import Union, List, Tuple, Optional, Any

# %% nbgrader={"grade": false, "grade_id": "modulename-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch [Module] Module")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to [action]!")
```

### C. **Package Location**
```python
# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/NN_modulename/modulename_dev.py`
**Building Side:** Code exports to `tinytorch.core.modulename`

```python
# Final package structure:
from tinytorch.core.modulename import ComponentName  # Main functionality!
from tinytorch.core.tensor import Tensor  # Foundation
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's structure
- **Consistency:** All [module] operations live together
- **Foundation:** Connection to broader ML systems
"""
```

### D. **Educational Content Structure**
```python
# %% [markdown]
"""
## What Are [Components]?

### The Problem/Motivation
Explain why this module exists and what problem it solves.

### The Solution
Describe the approach and key insights.

### Real-World Impact
Show concrete applications and industry relevance.

### What We'll Build
1. **Component 1**: Brief description
2. **Component 2**: Brief description
3. **Integration**: How components work together
"""
```

### E. **ASCII Diagrams and Tables**

ASCII diagrams are powerful educational tools that visualize concepts, architectures, and data flows. Follow these guidelines to ensure they render correctly across all environments.

#### Box Alignment Rules

**1. Border Width Consistency**: The top border (`┌───┐`) defines the box width. All content lines must match this width exactly.

```
✅ CORRECT:
┌───────────────────────────────┐
│ Content fits within borders  │
│ All lines same width         │
└───────────────────────────────┘

❌ WRONG:
┌─────────────────┐
│ Content extends beyond│
│ Border mismatch      │
└─────────────────┘
```

**2. Nested Boxes**: Inner boxes must align with the outer box's right border. Use consistent spacing.

```
✅ CORRECT:
┌───────────────────────────────────────────────────────┐
│ OUTER BOX TITLE                                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│ Section Header                                        │
│ ┌───────────────────────────────────────────────────┐ │
│ │ Inner content properly aligned                    │ │
│ │ All borders match outer box width                 │ │
│ └───────────────────────────────────────────────────┘ │
│                                                       │
└───────────────────────────────────────────────────────┘

❌ WRONG (misaligned nested boxes):
┌──────────────────────────────────┐
│ OUTER BOX                        │
│ ┌──────────────────────────────┐          │
│ │ Inner box extends beyond     │          │
│ └──────────────────────────────┘          │
└──────────────────────────────────────────────────┘
```

**3. Side-by-Side Boxes**: Both boxes should have equal or proportional widths with consistent spacing.

```
✅ CORRECT:
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ Left Box                      │    │ Right Box                     │
│ Content here                  │ →  │ Content here                  │
│ Same width as right           │    │ Same width as left            │
└───────────────────────────────┘    └───────────────────────────────┘

❌ WRONG (unequal widths, misaligned):
┌─────────────────┐         ┌─────────────────────────┐
│ Left Box          │         │ Right Box            │
│ Width mismatch   │    →    │ Different size      │
└─────────────────┘         └─────────────────────────┘
```

#### Table Formatting Rules

**1. Column Alignment**: All column separators (`│`) must align vertically. Use consistent column widths.

```
✅ CORRECT:
┌─────────────────┬─────────────────┬─────────────────┐
│ Column 1        │ Column 2        │ Column 3        │
├─────────────────┼─────────────────┼─────────────────┤
│ Data aligned    │ Same width      │ Consistent      │
│ Short           │ Medium content  │ Longer content  │
└─────────────────┴─────────────────┴─────────────────┘

❌ WRONG (separators don't align):
┌─────────────────┬─────────────────┬─────────────────┐
│ Column 1        │ Column 2        │ Column 3        │
├─────────────────┼─────────────────┼─────────────────┤
│ Misaligned│ Content      │ Wrong widths     │
└─────────────────┴─────────────────┴─────────────────┘
```

**2. Multi-Column Tables**: Header separators (`┬`, `┼`, `┴`) must align with content separators.

#### Special Characters

**Emoji Width**: Emojis display as ~2 characters wide but count as 1 in string length. Add extra padding after emojis.

```
✅ CORRECT (emoji padding):
│ ✅ Success message here      │
│ ❌ Error message here        │

❌ WRONG (no padding compensation):
│ ✅ Success message here│
│ ❌ Error message here │
```

**Box-Drawing Characters Reference**:
- Corners: `┌` `┐` `└` `┘`
- Lines: `─` (horizontal), `│` (vertical)
- Intersections: `┬` `┴` `├` `┤` `┼`
- Arrows: `→` `←` `↑` `↓` `▼` `▲`

#### Quality Checklist for Diagrams

Before committing, verify each diagram:
- [ ] Top border width matches bottom border width
- [ ] All content lines have consistent right-edge `│` alignment
- [ ] Nested boxes align with outer box boundaries
- [ ] Table columns have uniform widths
- [ ] Side-by-side boxes are visually balanced
- [ ] Emojis have appropriate padding

### F. **Implementation Sections**
```python
# %% [markdown]
"""
## Step N: [Component Name]

### Mathematical Foundation
Mathematical explanation with formulas and intuition.

### Implementation Strategy
Step-by-step approach to building the component.
"""

# %% nbgrader={"grade": false, "grade_id": "component-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class ComponentName:
    """
    Brief description of the component.

    TODO: Student implementation guidance

    APPROACH:
    1. [First step with specific guidance]
    2. [Second step with specific guidance]
    3. [Third step with specific guidance]

    EXAMPLE:
    Input: [concrete example]
    Expected: [concrete expected output]
    """
    def __init__(self, parameter1, parameter2):
        ### BEGIN SOLUTION
        # Complete implementation (hidden from students)
        ### END SOLUTION
        raise NotImplementedError("Student implementation required")
```

### G. **Test Functions**
```python
# %% [markdown]
"""
### 🧪 Unit Test: Component Name

**Description**: Brief explanation of what is tested

**This is a unit test** - it tests [specific functionality] in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-component", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_component_function():
    """Test the component functionality."""
    print("🔬 Unit Test: Component Function...")

    # Test implementation
    try:
        # Test logic
        assert condition, "Error message"
        print("✅ Component test works")
        print("📈 Progress: Component ✓")
        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

# Test execution
if __name__ == "__main__":
    test_component_function()
```

### H. **Module Summary** (CRITICAL)
```python
# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented [module description]:

### ✅ What You've Built
- **Component 1**: Description of accomplishment
- **Component 2**: Description of accomplishment
- **Integration**: How components work together
- **Complete System**: End-to-end functionality

### ✅ Key Learning Outcomes
- **Understanding**: Core concepts mastered
- **Implementation**: Technical skills developed
- **Mathematical mastery**: Formulas and algorithms implemented
- **Real-world application**: Practical applications understood

### ✅ Mathematical Foundations Mastered
- **Formula 1**: Mathematical concept with notation
- **Formula 2**: Another key mathematical insight
- **Algorithm**: Implementation of key algorithm

### ✅ Professional Skills Developed
- **Skill 1**: Technical capability gained
- **Skill 2**: Another professional competency
- **Integration**: Systems thinking and design

### ✅ Ready for Advanced Applications
Your implementations now enable:
- **Application 1**: What students can build next
- **Application 2**: Another capability unlocked
- **Real Systems**: Connection to production applications

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch**: `torch.component` provides identical functionality
- **TensorFlow**: Similar concepts in TensorFlow
- **Industry Standard**: Used in major ML frameworks

### 🎯 The Power of [Technology]
You've unlocked the key technology that [impact description]:
- **Capability 1**: What this enables
- **Capability 2**: Another important capability
- **Scale**: How this technology scales

### 🧠 Deep Learning Revolution/Impact
You now understand the technology that [revolutionary impact]:
- **Historical context**: Before/after this technology
- **Modern applications**: Current uses
- **Future implications**: What this enables

### 🚀 What's Next
Your implementations are the foundation for:
- **Next Module**: Natural progression
- **Advanced Topics**: Related advanced concepts
- **Research**: Opportunities for exploration

**Next Module**: [Description of next module and its connection]

[Motivational closing emphasizing what students have accomplished]
"""
```

## ▶ Testing Standards

### 1. **Test Function Naming**
All test functions MUST follow this pattern:
```python
def test_component_name():
    """Test the component functionality."""
```

### 2. **Test Function Structure**
```python
def test_component_function():
    """Test description."""
    print("🔬 Unit Test: Component Function...")

    ### 🧪 Unit Test: Component Function

    **Description**: Brief explanation of what is tested

    **This is a unit test** - it tests [specific functionality] in isolation.

    try:
        # Test logic
        print("✅ [check] works")
        print("📈 Progress: Component ✓")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
```

### 3. **Test Execution**
```python
if __name__ == "__main__":
    test_function_1()
    test_function_2()
    test_function_3()
```

## 📦 NBDev Integration

### 1. **Export Directives**
```python
#| export
def function_to_export():
    """Function that becomes part of tinytorch package."""
    pass
```

### 2. **Default Export Target**
```python
#| default_exp core.modulename
```

### 3. **NBGrader Integration**
```python
# %% nbgrader={"grade": false, "grade_id": "unique-id", "locked": false, "schema_version": 3, "solution": true, "task": false}
```

### 4. **Solution Hiding (NBGrader)**
```python
def student_function():
    """
    Student implementation function.

    TODO: Implementation guidance for students.
    """
    ### BEGIN SOLUTION
    # Complete implementation hidden from students
    ### END SOLUTION
    raise NotImplementedError("Student implementation required")
```

## 🔧 Development Workflow

### 1. **Python-First Development**
- Work in `.py` files (source of truth)
- Generate `.ipynb` with `tito nbgrader generate`
- Never commit `.ipynb` files to version control

### 2. **Testing Integration**
- Use inline tests for immediate feedback
- All tests must pass before module completion
- Use `pytest` for any external testing

## 📋 Module Metadata (module.yaml)

```yaml
name: "modulename"
title: "Module Title"
description: "Brief description of module functionality"
version: "1.0.0"
author: "TinyTorch Team"

learning_objectives:
  - "Objective 1"
  - "Objective 2"

prerequisites:
  - "prerequisite_module"

metadata:
  difficulty: "intermediate"
  time_estimate: "4-6 hours"
  pedagogical_framework: "Build → Use → Understand"

concepts:
  - "concept1"
  - "concept2"

exports:
  - "ComponentName"
  - "helper_function"

files:
  main: "modulename_dev.py"
  readme: "README.md"

assessment:
  total_points: 50
  breakdown:
    component1: 20
    component2: 20
    integration: 10

next_modules:
  - "next_module"
```

## ✓ Quality Checklist

Before completing a module:

### Content Requirements
- [ ] Jupytext percent format header
- [ ] Educational content with clear explanations
- [ ] Step-by-step implementation guidance
- [ ] Mathematical foundations explained
- [ ] Real-world applications discussed
- [ ] Complete module summary (following 08_optimizers pattern)

### Technical Requirements
- [ ] All functions have docstrings
- [ ] NBGrader cells properly configured
- [ ] NBDev export directives in place
- [ ] Solution blocks use `### BEGIN SOLUTION` / `### END SOLUTION`
- [ ] Error handling implemented
- [ ] Type hints where appropriate

### Testing Requirements
- [ ] All inline tests pass
- [ ] Test functions use standard naming (`test_*`)
- [ ] Test output follows emoji standards
- [ ] `if __name__ == "__main__":` block present
- [ ] Tests provide educational feedback

### Documentation Requirements
- [ ] module.yaml properly configured
- [ ] README.md updated
- [ ] Learning objectives clear
- [ ] Prerequisites documented
- [ ] Export list accurate

## 📚 Additional Resources

- **Reference Implementation**: `modules/08_optimizers/optimizers_dev.py`
- **NBGrader Documentation**: [NBGrader docs](https://nbgrader.readthedocs.io/)
- **NBDev Documentation**: [NBDev docs](https://nbdev.fast.ai/)
- **TinyTorch CLI**: Use `tito --help` for development commands


**Remember**: When in doubt, reference `08_optimizers` - it follows all these patterns perfectly and serves as the living example of proper module structure.
