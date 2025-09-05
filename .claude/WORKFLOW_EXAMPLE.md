# Two-Agent Workflow Example

This document demonstrates how the Reviewer and Editor agents work together to improve a chapter while maintaining progressive knowledge boundaries.

## Scenario: Improving Chapter 3 (Deep Learning Primer)

User runs: `/improve dl_primer.qmd`

### Phase 1: Reviewer Agent Analysis

**Input:** Chapter 3 file path
**First Action:** Reads KNOWLEDGE_MAP.md to understand boundaries

#### Knowledge Boundaries Identified:
- **Available (Ch 1-2)**: ML systems, deployment tiers, resource constraints
- **Forbidden (Ch 4+)**: CNNs, RNNs, Transformers, quantization, GPUs, etc.

#### Issues Found During Line-by-Line Scan:

**Line 145:**
```
Found: "Neural networks can be optimized through quantization and pruning"
Issue: "quantization" and "pruning" introduced in Chapter 10
```

**Line 267:**
```
Found: "Training on GPUs significantly speeds up the process"
Issue: "GPUs" introduced in Chapter 11
```

**Line 234:**
```
Found: "The matrix operations are straightforward for anyone familiar with linear algebra"
Issue: Assumes mathematical background not established
```

#### Reviewer Output (YAML + Summary):

```yaml
---
chapter: 3
title: "Deep Learning Primer"
file: "quarto/contents/core/dl_primer/dl_primer.qmd"
total_lines: 650
review_timestamp: "2024-01-05T14:30:00Z"
---

forward_references:
  - location:
      line: 145
      exact_match: "optimized through quantization and pruning"
    found_text: "quantization and pruning"
    context: "Neural networks can be optimized through quantization and pruning to reduce their computational requirements."
    violation: "Terms introduced in Chapter 10"
    severity: "critical"
    suggested_fix:
      type: "footnote"
      new_text: "optimized through various techniques[^ch3-opt]"
      footnote_text: "[^ch3-opt]: Specific optimization methods including quantization and pruning are covered in detail in Chapter 10."
      
  - location:
      line: 267
      exact_match: "Training on GPUs significantly speeds up"
    found_text: "GPUs"
    context: "Training on GPUs significantly speeds up the process compared to traditional processors."
    violation: "Hardware accelerators introduced in Chapter 11"
    severity: "critical"
    suggested_fix:
      type: "replacement"
      new_text: "Training on specialized hardware significantly speeds up"

clarity_issues:
  - location:
      line: 234
      exact_match: "The matrix operations are straightforward for anyone familiar with linear algebra"
    issue: "Assumes mathematical background"
    severity: "high"
    consensus: 3
    suggested_fix:
      type: "replacement"
      new_text: "The matrix operations use basic mathematical concepts to process the numerical data"

technical_corrections:
  - location:
      line: 456
      exact_match: "All neural networks require exactly 1000 training examples"
    issue: "Factually incorrect absolute statement"
    severity: "medium"
    consensus: 4
    suggested_fix:
      type: "replacement"
      new_text: "Neural networks typically require many training examples, with the exact number varying by problem complexity"
```

**Summary:**
- 🔴 2 Critical forward references found
- 🟡 1 High priority clarity issue
- 🟡 1 Technical correction needed
- All issues have specific location data and suggested fixes

### Phase 2: Editor Agent Implementation

**Input:** Review report (YAML + summary)
**First Action:** Parse YAML to extract edit instructions

#### Edit Processing Order:
1. Critical forward references (lines 145, 267)
2. High priority clarity (line 234)
3. Technical corrections (line 456)

#### Edits Implemented:

**Edit 1: Forward Reference Fix (Line 145)**
```
OLD: "optimized through quantization and pruning"
NEW: "optimized through various techniques[^ch3-opt]"
FOOTNOTE ADDED: "[^ch3-opt]: Specific optimization methods including quantization and pruning are covered in detail in Chapter 10."
```

**Edit 2: Forward Reference Fix (Line 267)**
```
OLD: "Training on GPUs significantly speeds up"
NEW: "Training on specialized hardware significantly speeds up"
```

**Edit 3: Clarity Improvement (Line 234)**
```
OLD: "The matrix operations are straightforward for anyone familiar with linear algebra"
NEW: "The matrix operations use basic mathematical concepts to process the numerical data"
```

**Edit 4: Technical Correction (Line 456)**
```
OLD: "All neural networks require exactly 1000 training examples"
NEW: "Neural networks typically require many training examples, with the exact number varying by problem complexity"
```

#### MultiEdit Tool Usage:
```yaml
file_path: "quarto/contents/core/dl_primer/dl_primer.qmd"
edits:
  - old_string: "optimized through quantization and pruning"
    new_string: "optimized through various techniques[^ch3-opt]"
  - old_string: "Training on GPUs significantly speeds up"
    new_string: "Training on specialized hardware significantly speeds up"  
  - old_string: "The matrix operations are straightforward for anyone familiar with linear algebra"
    new_string: "The matrix operations use basic mathematical concepts to process the numerical data"
  - old_string: "All neural networks require exactly 1000 training examples"
    new_string: "Neural networks typically require many training examples, with the exact number varying by problem complexity"
  - old_string: ""  # Append footnote at end of section
    new_string: "\n[^ch3-opt]: Specific optimization methods including quantization and pruning are covered in detail in Chapter 10."
```

### Phase 3: Final Output

**Editor Summary:**
- ✅ 2 critical forward references eliminated
- ✅ 1 high priority clarity issue resolved
- ✅ 1 technical accuracy correction made
- ✅ 1 footnote added for future reference
- ✅ Protected content (TikZ, tables, equations) preserved
- ✅ Academic tone maintained throughout

**File Changes:**
- 4 text replacements
- 1 footnote addition
- 0 protected content modifications
- Total: 5 precise edits

## Key Benefits Demonstrated

1. **Precision**: Exact line numbers and text matching ensures surgical edits
2. **Traceability**: YAML format provides complete audit trail
3. **Safety**: Knowledge map consultation prevents new violations
4. **Flexibility**: Footnotes allow brief mentions of future concepts
5. **Quality**: Multi-perspective review ensures comprehensive analysis
6. **Efficiency**: Structured handoff enables clean automation

## Error Prevention

The structured approach prevents:
- **Missing issues**: Line-by-line scanning catches all violations
- **Incorrect edits**: Exact text matching ensures right location
- **New violations**: Knowledge map re-consultation during editing
- **Protected content damage**: Explicit constraints prevent changes
- **Loss of context**: Full sentence context preserves meaning

This workflow ensures that Chapter 3 now builds properly on Chapters 1-2 knowledge while preparing students for Chapter 4 concepts, all without introducing undefined terms or breaking the pedagogical progression.