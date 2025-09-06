# Footnote Agent Workflow

## Overview
This document describes the workflow for adding footnotes to MLSysBook chapters. The footnote agent has exclusive authority over all footnotes in the book.

## Pre-Work: Catalog Existing Footnotes
Before adding footnotes to any chapter, ALWAYS run the catalog script to understand what footnotes already exist:

```bash
python /Users/VJ/GitHub/MLSysBook/scripts/catalog_footnotes.py
```

This generates:
- `/Users/VJ/GitHub/MLSysBook/footnote_catalog.json` - Complete footnote data
- `/Users/VJ/GitHub/MLSysBook/.claude/footnote_context.md` - Agent-ready context

## Workflow Steps

### 1. Initialize Context
```bash
# Always start by cataloging existing footnotes
python scripts/catalog_footnotes.py

# Read the generated context
cat .claude/footnote_context.md
```

### 2. Chapter-Specific Analysis
When working on a specific chapter:
- Check if terms have been defined in other chapters (avoid duplication)
- Review existing footnote patterns and style
- Identify technical terms needing clarification

### 3. Footnote Addition Guidelines

#### ID Format
- Use: `[^fn-term-name]` (lowercase, hyphens)
- Examples: `[^fn-api]`, `[^fn-neural-network]`, `[^fn-backprop]`

#### Definition Format
```markdown
[^fn-term-name]: **Bold Term**: Clear, concise definition. Optional helpful analogy or context.
```

#### Placement Rules
1. Add inline reference where term first appears: `API[^fn-api]`
2. Place definition at end of section or document
3. Keep definitions together, not scattered

#### Content Guidelines
- **Concise**: Aim for 1-2 sentences (avg ~200 characters)
- **Educational**: Add historical context or analogies when helpful
- **No Duplication**: Never redefine terms from other chapters
- **Technical Focus**: Prioritize terms that need clarification for CS/Engineering students

### 4. Quality Checks

Before committing:
1. Run catalog script to verify no duplicates
2. Check all inline references have definitions
3. Verify no term is defined multiple times
4. Ensure consistent formatting

### 5. Example Workflow

```bash
# 1. Start on new branch
git checkout -b footnote/chapter-name

# 2. Catalog existing footnotes
python scripts/catalog_footnotes.py

# 3. Read context
cat .claude/footnote_context.md

# 4. Edit chapter file
# Add footnotes following guidelines

# 5. Verify changes
python scripts/catalog_footnotes.py
# Check the updated catalog for any issues

# 6. DO NOT stage or commit
# Leave changes unstaged for user review
# User will decide what to stage and commit
```

## Common Patterns

### Technical Terms
```markdown
Machine learning[^fn-ml] is a subset of AI...

[^fn-ml]: **Machine Learning**: Systems that improve performance through experience, using algorithms to find patterns in data rather than following explicit programmed rules.
```

### Historical Context
```markdown
The perceptron[^fn-perceptron] was invented in 1957...

[^fn-perceptron]: **Perceptron**: One of the first computational learning algorithms created by Frank Rosenblatt, inspired by biological neurons. It could learn to classify patterns through adjustable weights.
```

### Helpful Analogies
```markdown
API[^fn-api] design is crucial...

[^fn-api]: **Application Programming Interface (API)**: A set of protocols for software communication, similar to how electrical plugs provide standard interfaces for connecting devices.
```

## Avoiding Duplication

The catalog script identifies:
- **Duplicate IDs**: Same footnote ID used in multiple places
- **Duplicate Terms**: Same term defined with different IDs
- **Undefined References**: Inline refs without definitions
- **Unused Definitions**: Definitions never referenced

Always check these before adding new footnotes!

## Integration with Other Agents

- **Reviewer Agent**: Identifies where footnotes are needed (doesn't add them)
- **Editor Agent**: Works on text improvements (doesn't touch footnotes)
- **Stylist Agent**: Ensures consistency (doesn't modify footnotes)
- **Footnote Agent**: Has exclusive authority over all footnotes

## Removal Script

If you need to start fresh or clean up:
```bash
# Remove ALL footnotes from all files
python scripts/remove_footnotes.py

# This removes both inline references and definitions
```

## Best Practices

1. **Run catalog before and after changes** to ensure consistency
2. **One term, one definition** across the entire book
3. **Follow established patterns** from existing footnotes
4. **Keep definitions concise** but informative
5. **Add value** - don't footnote obvious terms
6. **Think pedagogically** - what would help a CS student understand ML concepts?