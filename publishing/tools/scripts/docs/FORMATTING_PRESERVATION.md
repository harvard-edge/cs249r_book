# Formatting Preservation Guidelines

## Critical Formatting Rules for AI Agents

When performing textbook polish, editing, or content improvements, you MUST preserve these specific formatting patterns. These are not arbitrary style choices but deliberate educational formatting that enhances readability and pedagogical effectiveness.

## 1. Grid Tables: Bold First Column

**Rule**: The first column of grid tables MUST have bold formatting.

**Why**: This emphasizes the primary concept being compared or categorized, making tables scannable and improving comprehension.

### ✅ CORRECT:
```markdown
+---------------------------+----------------------+
| Strategy                  | Search Efficiency    |
+:==========================+:=====================+
| **Reinforcement Learning** | 400-1000 GPU-days   |
+---------------------------+----------------------+
| **Evolutionary Algorithms** | 200-500 GPU-days    |
+---------------------------+----------------------+
```

### ❌ INCORRECT:
```markdown
+---------------------------+----------------------+
| Strategy                  | Search Efficiency    |
+:==========================+:=====================+
| Reinforcement Learning    | 400-1000 GPU-days    |
+---------------------------+----------------------+
| Evolutionary Algorithms   | 200-500 GPU-days     |
+---------------------------+----------------------+
```

## 2. Callout Section Headers: Bold and Title Case

**Rule**: Section headers inside callout blocks MUST be bold and title case, ending with a colon.

**Why**: These headers organize complex information within callouts, creating visual hierarchy that helps students navigate checkpoint questions and learning objectives.

### ✅ CORRECT:
```markdown
::: {.callout-note title="Checkpoint"}

Before proceeding, verify your understanding:

**Integration Across Phases:**

- [ ] Can you trace how architectural decisions impact performance?
- [ ] Do you understand memory requirements?

**Training To Deployment:**

- [ ] Can you explain the complete lifecycle?

:::
```

### ❌ INCORRECT:
```markdown
::: {.callout-note title="Checkpoint"}

Before proceeding, verify your understanding:

Integration across phases:

- [ ] Can you trace how architectural decisions impact performance?
- [ ] Do you understand memory requirements?

Training to deployment:

- [ ] Can you explain the complete lifecycle?

:::
```

## 3. Figure and Table Captions: Bold Title Format

**Rule**: All figure and table captions MUST follow the `**Bold Title**: Explanation` format.

**Why**: This creates consistent visual anchoring across hundreds of figures and tables, making it easy for students to identify what they're looking at and why it matters.

### ✅ CORRECT:
```markdown
: **NAS Search Strategy Comparison**: Trade-offs between search efficiency, use cases, and limitations for different NAS approaches. {#tbl-nas-strategies}

: **System Resource Evolution**: Programming paradigms shift system demands from sequential computation to massive matrix operations. {#tbl-evolution}

: **Attention Mechanism**: Transformer models compute attention through query-key-value interactions, enabling dynamic focus across input sequences. {#fig-attention}
```

### ❌ INCORRECT:
```markdown
: NAS Search Strategy Comparison {#tbl-nas-strategies}

: System Resource Evolution: Programming paradigms shift... {#tbl-evolution}

: Attention Mechanism - Transformer models compute... {#fig-attention}
```

## 4. Implementation Instructions

### For AI Polish/Editing Agents

When editing content:
1. **Do NOT remove bold formatting** from table first columns
2. **Do NOT convert** `**Section Header:**` to lowercase or remove bold
3. **Do NOT change caption format** from `**Title**: Explanation` to any other format
4. **Do NOT assume** that bold text in these contexts is "pseudo-headers" that should be converted to prose

### Restoration Script

If formatting is accidentally removed, use:
```bash
python tools/scripts/restore_formatting.py <file_paths>
```

This script automatically:
- Restores bold formatting in grid table first columns
- Restores bold + title case for callout section headers
- Validates caption formatting
- Reports any missing bold captions

## 5. Context-Specific Rules

### What "Don't Start Paragraphs with Bold" Means

The rule about not starting paragraphs with bold text **does NOT apply** to:
- Grid table first columns
- Callout section headers
- Figure/table caption titles
- Definition list terms

It **DOES apply** to:
- Regular prose paragraphs in chapter text
- List items in regular (non-callout) contexts
- Sidebar content outside callouts

### Example of When to Remove Bold:

```markdown
## Section Title

**Key Concept**: Neural networks process data through layers.
```

This ↑ should be rewritten as:
```markdown
## Section Title

Neural networks process data through layers, transforming inputs...
```

But this ↓ should be kept as-is:
```markdown
::: {.callout-note}

**Key Concept:**

- Point 1
- Point 2

:::
```

## 6. Why These Rules Matter

### Educational Design Principles

1. **Cognitive Load Management**: Bold formatting creates visual hierarchy that helps students scan and navigate complex technical content without overwhelming working memory.

2. **Pattern Recognition**: Consistent formatting across 270+ figures and 90+ tables trains students to quickly identify and understand visual elements, reducing cognitive overhead for content comprehension.

3. **Checkpoint Effectiveness**: Bold section headers in callouts improve self-assessment by making it easy to identify which skill category each checkpoint question targets.

4. **Table Scannability**: Bold first columns enable rapid table lookup during problem-solving, when students need to quickly find relevant information.

### Production Considerations

This textbook renders in multiple formats (HTML, PDF, EPUB). The formatting patterns ensure:
- Consistent visual hierarchy across all formats
- Proper semantic structure for accessibility
- Clean PDF table rendering
- Proper EPUB navigation

## 7. Testing and Validation

After any content editing, run:

```bash
# Check for formatting issues
python tools/scripts/restore_formatting.py --dry-run quarto/contents/core/*/*.qmd

# Fix any issues found
python tools/scripts/restore_formatting.py quarto/contents/core/*/*.qmd
```

The script will report:
- Files with formatting violations
- Captions missing bold titles
- Tables with non-bold first columns
- Callout headers needing title case

## 8. Future Development

When creating new AI agents or editing workflows:
1. Include this document in the agent's context
2. Add formatting validation to pre-commit hooks
3. Test on sample files before production runs
4. Always run `restore_formatting.py` after AI editing passes

---

**Last Updated**: October 7, 2025
**Maintainer**: MLSysBook Team
**Related Scripts**: `tools/scripts/restore_formatting.py`
