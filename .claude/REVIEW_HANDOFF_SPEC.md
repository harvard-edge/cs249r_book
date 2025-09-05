# Review-to-Editor Handoff Specification

This document defines the precise format for passing information from the Reviewer agent to the Editor agent, ensuring accurate and efficient edits.

## Structured Review Report Format

The reviewer must produce a machine-parseable report with exact location information:

```yaml
---
chapter: 3
title: "Deep Learning Primer"
file: "quarto/contents/core/dl_primer/dl_primer.qmd"
total_lines: 850
review_timestamp: "2024-01-05T10:30:00Z"
---

forward_references:
  - location:
      line: 145
      column_start: 23
      column_end: 35
    found_text: "quantization"
    context: "Neural networks can be optimized through quantization and pruning"
    violation: "Term introduced in Chapter 10"
    severity: "critical"
    suggested_fix:
      type: "replacement"
      new_text: "optimization techniques"
      
  - location:
      line: 267
      exact_match: "GPUs provide significant acceleration"
    found_text: "GPUs"
    violation: "Hardware accelerators introduced in Chapter 11"
    severity: "critical"
    suggested_fix:
      type: "replacement"
      new_text: "Specialized hardware provides significant acceleration"
      
  - location:
      line: 389
      paragraph_id: "sec-training-process"
    found_text: "federated learning approaches"
    violation: "Federated learning introduced in Chapter 14"
    severity: "critical"
    suggested_fix:
      type: "footnote"
      new_text: "distributed training approaches[^distributed-note]"
      footnote_text: "[^distributed-note]: Specific techniques for distributed and privacy-preserving training will be covered in Chapter 14."

clarity_issues:
  - location:
      section: "3.2"
      subsection: "Forward Propagation"
      line_range: [234, 245]
    issue: "Assumes understanding of matrix operations"
    severity: "high"
    consensus: 3
    suggested_fix:
      type: "insertion"
      after_line: 233
      new_text: |
        Before we dive into forward propagation, recall that we represent
        data as numerical arrays and perform mathematical operations on them.

technical_corrections:
  - location:
      line: 456
      exact_match: "The learning rate should always be 0.01"
    issue: "Incorrect absolute statement"
    severity: "medium"
    consensus: 4
    suggested_fix:
      type: "replacement"
      new_text: "The learning rate is often initialized to small values like 0.01"

enhancement_suggestions:
  - location:
      section: "3.4"
      after_line: 567
    enhancement: "Add practical example"
    severity: "low"
    consensus: 2
    suggested_fix:
      type: "example_box"
      content: |
        ::: {.callout-note title="Example: Simple Pattern Recognition"}
        Consider a system that needs to distinguish between photos of cats and dogs...
        :::
```

## Location Specification Methods

### Method 1: Line + Exact Text (Most Precise)
```yaml
location:
  line: 145
  exact_match: "exact phrase to find"
```

### Method 2: Line Range
```yaml
location:
  line_range: [234, 245]
  containing: "partial text to identify"
```

### Method 3: Section/Subsection Reference
```yaml
location:
  section: "3.2"
  subsection: "Forward Propagation"
  paragraph: 2
```

### Method 4: Quarto Label Reference
```yaml
location:
  label_id: "sec-training-process"
  offset_lines: 3  # lines after the label
```

## Fix Types

### 1. Simple Replacement
```yaml
suggested_fix:
  type: "replacement"
  new_text: "specialized hardware"
```

### 2. Footnote Addition
```yaml
suggested_fix:
  type: "footnote"
  new_text: "concept[^note-id]"
  footnote_text: "[^note-id]: This will be explained in detail in Chapter X."
```

### 3. Clarification Insertion
```yaml
suggested_fix:
  type: "insertion"
  position: "before" | "after"
  reference_line: 234
  new_text: "Additional clarifying text..."
```

### 4. Definition Box
```yaml
suggested_fix:
  type: "definition"
  term: "Neural Network"
  definition: "A computational model inspired by..."
  position: "first_use"  # or specific line
```

### 5. Cross-Reference
```yaml
suggested_fix:
  type: "cross_reference"
  new_text: "this technique (introduced in Chapter 2)"
```

## Editor Processing Instructions

The Editor agent should process fixes in this priority order:

1. **Critical Forward References** - Must be fixed
2. **High-Priority Clarity** - 3+ reviewer consensus
3. **Technical Corrections** - Accuracy issues
4. **Medium Priority** - 2+ reviewer consensus
5. **Enhancements** - Optional improvements

## Footnote Guidelines

Use footnotes when:
- Need to mention a future concept briefly
- Want to provide optional clarification
- Reference to later chapters is helpful but not essential

Footnote format:
```markdown
The model uses optimization techniques[^opt-note] to improve performance.

[^opt-note]: Specific optimization methods including quantization and pruning are covered in Chapter 10.
```

## Validation Checklist

Before handoff, the Reviewer must ensure:
- [ ] Every issue has precise location information
- [ ] All forward references include line numbers
- [ ] Suggested fixes maintain sentence flow
- [ ] Footnotes are used appropriately
- [ ] Critical issues are clearly marked
- [ ] Context is sufficient for accurate edits

## Example Complete Handoff

```yaml
---
chapter: 2
title: "ML Systems"
---

forward_references:
  - location:
      line: 89
      exact_match: "neural network architectures"
    found_text: "neural network architectures"
    violation: "Neural networks not introduced until Chapter 3"
    suggested_fix:
      type: "replacement"
      new_text: "machine learning model architectures"
      
  - location:
      line: 145
      exact_match: "GPU acceleration enables"
    found_text: "GPU acceleration"
    violation: "GPUs introduced in Chapter 11"
    suggested_fix:
      type: "footnote"
      new_text: "hardware acceleration[^hw-accel] enables"
      footnote_text: "[^hw-accel]: Specialized hardware accelerators, including GPUs and TPUs, are discussed in detail in Chapter 11."

clarity_issues:
  - location:
      line: 234
      exact_match: "The deployment pipeline consists of"
    issue: "Needs clearer introduction"
    suggested_fix:
      type: "replacement"
      new_text: "The deployment pipeline - the series of steps to move models from development to production - consists of"
```

## Benefits of This Structure

1. **Precision**: Editor knows exactly where to make changes
2. **Flexibility**: Multiple location methods for different scenarios
3. **Footnotes**: Can reference future content without confusion
4. **Traceability**: Clear record of what was changed and why
5. **Automation**: Structure enables automated editing tools

This structured handoff ensures the Editor agent can make surgical, accurate edits based on the Reviewer's analysis.