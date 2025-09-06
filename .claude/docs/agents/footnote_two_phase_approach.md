# Footnote Agent: Two-Phase Approach

## Overview
To ensure correct placement of footnotes (definitions immediately after paragraphs), use a two-phase approach:
1. **Planning Phase**: Identify and map all footnotes
2. **Execution Phase**: Add footnotes paragraph by paragraph

## Phase 1: Planning

### Steps
1. Read entire chapter
2. Identify terms needing footnotes
3. Create a detailed plan

### Plan Format
```markdown
## Footnote Plan for [Chapter Name]

### Paragraph 1 (lines 20-25)
Terms to footnote:
- Dartmouth Conference → [^fn-dartmouth-conference]
- Claude Shannon → [^fn-claude-shannon]

### Paragraph 2 (lines 27-30)
Terms to footnote:
- Perceptron → [^fn-perceptron]
- Frank Rosenblatt → [^fn-rosenblatt]

### Paragraph 3 (lines 32-38)
No footnotes needed (basic concepts already known to CS students)

[Continue for entire chapter...]
```

## Phase 2: Execution

### Steps
1. Work through chapter paragraph by paragraph
2. For each paragraph with footnotes:
   - Add inline references in the paragraph
   - Add ALL definitions for that paragraph immediately after it
3. Never accumulate definitions to add later

### Example Execution

#### Original Text:
```markdown
The field of AI was born at the Dartmouth Conference in 1956, where John McCarthy
and Claude Shannon gathered to explore machine intelligence.

The next major breakthrough came with Frank Rosenblatt's perceptron in 1957,
which could learn to classify simple patterns.

Modern deep learning builds on these foundations with massive neural networks.
```

#### After Phase 2:
```markdown
The field of AI was born at the Dartmouth Conference[^fn-dartmouth-conference] in 1956, where John McCarthy
and Claude Shannon[^fn-claude-shannon] gathered to explore machine intelligence.

[^fn-dartmouth-conference]: **Dartmouth Conference**: The legendary 8-week summer workshop in 1956 where the term "artificial intelligence" was coined and the field was formally born, attended by just 10 researchers who would shape the next 70 years of AI.
[^fn-claude-shannon]: **Claude Shannon**: The MIT professor who founded information theory in 1948, establishing the mathematical basis for all digital communications and later contributing to early AI research including the first chess-playing programs.

The next major breakthrough came with Frank Rosenblatt's perceptron[^fn-perceptron] in 1957,
which could learn to classify simple patterns.

[^fn-perceptron]: **Perceptron**: The first artificial neural network capable of learning, created by Frank Rosenblatt at Cornell. Despite its limitations to linearly separable problems, it laid the foundation for modern deep learning 60 years later.

Modern deep learning builds on these foundations with massive neural networks.
```

## Benefits of Two-Phase Approach

1. **Comprehensive**: Planning ensures no important terms are missed
2. **Organized**: Systematic paragraph-by-paragraph execution
3. **Correct placement**: Definitions always immediately follow usage
4. **Reviewable**: Plan can be reviewed before execution
5. **Consistent**: Same approach for every chapter

## Agent Instructions Template

When invoking the footnote agent, include:

```markdown
Use the two-phase approach:

PHASE 1 - PLANNING:
1. Read the entire chapter
2. Create a paragraph-by-paragraph plan listing:
   - Line numbers for each paragraph
   - Terms that need footnotes in that paragraph
   - Skip paragraphs that need no footnotes
3. Present the plan for review

PHASE 2 - EXECUTION:
1. Work through the chapter paragraph by paragraph
2. For each paragraph in your plan:
   - Add inline references [^fn-term] where terms appear
   - IMMEDIATELY after that paragraph, add ALL footnote definitions
3. Never accumulate definitions to add in bulk later
4. Each footnote definition must appear right after its paragraph
```

## Common Mistakes to Avoid

### ❌ DON'T: Accumulate definitions
```markdown
[Add all inline references throughout chapter first]
[Then add all definitions at the end]
```

### ✅ DO: Add as you go
```markdown
[Process paragraph 1]
[Add paragraph 1's definitions]
[Process paragraph 2]
[Add paragraph 2's definitions]
[Continue...]
```

### ❌ DON'T: Split related footnotes
```markdown
Paragraph mentions GPU and TPU.
[^fn-gpu]: definition after paragraph
[^fn-tpu]: definition 10 paragraphs later
```

### ✅ DO: Keep related footnotes together
```markdown
Paragraph mentions GPU and TPU.
[^fn-gpu]: definition
[^fn-tpu]: definition
[Both right after the paragraph]
```

## Verification

After execution, verify:
```bash
# Check that no footnote definition is more than a few lines from its reference
# Each [^fn-X] reference should have its definition within ~10 lines in the file
```

This approach ensures footnotes are maintainable, reviewable, and properly placed!