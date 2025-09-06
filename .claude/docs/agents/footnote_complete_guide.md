# Complete Footnote Agent Guide

## Overview
The footnote agent has exclusive authority over all footnotes in MLSysBook. This guide consolidates all footnote-related documentation.

## 1. Quality Guidelines

### What Makes a Good Footnote

#### ✅ KEEP These Types:
1. **Historical Context with Details**
   - Dartmouth Conference (1956) - birthplace of AI
   - Claude Shannon - father of information theory
   - MYCIN - Stanford expert system with performance stats

2. **Technical ML Terms with Explanations**
   - Moore's Law - computing power doubling
   - SVMs - with kernel trick explanation
   - Precision/Recall - fundamental metrics with tradeoffs

3. **Modern ML Concepts with Numbers**
   - Foundation models - large-scale general-purpose
   - GPT-3 - 175B parameters
   - ImageNet - 14M images

#### ❌ AVOID These:
1. **Basic CS Terms Without Enrichment**
   - Plain "computer engineering"
   - Generic "IoT" without stats
   - Simple "latency" definition

2. **Terms Already Covered**
   - Check KNOWLEDGE_MAP for chapter coverage
   - Don't redefine from earlier chapters

### Quality Criteria
Every footnote must:
- Add value beyond basic definition
- Include numbers, dates, or fascinating facts
- Target CS/Engineering students learning ML
- Be 1-2 sentences (200-400 characters ideal)

### Examples
**Bad:** "IoT: Internet of Things, connected devices"
**Good:** "IoT: Expected to reach 75 billion devices by 2025, generating more data annually than all of human history before 2020"

## 2. Placement Rules (CRITICAL)

### The Golden Rule
**Footnote definitions MUST appear immediately after the paragraph containing their references.**

### Correct Example
```markdown
The field began at the Dartmouth Conference[^fn-dartmouth] in 1956.

[^fn-dartmouth]: **Dartmouth Conference**: The legendary 8-week workshop where AI was born...

The next breakthrough was the perceptron[^fn-perceptron] in 1957.

[^fn-perceptron]: **Perceptron**: First neural network capable of learning...
```

### Never Do This
```markdown
Text with reference[^fn-term]...
[Many paragraphs later]
[^fn-term]: Definition...
```

## 3. Two-Phase Approach

### Phase 1: Planning
1. Read entire chapter
2. Create paragraph-by-paragraph plan:
   - Line numbers
   - Terms needing footnotes
   - Skip paragraphs with no terms

### Phase 2: Execution
1. Process paragraph by paragraph
2. Add inline references
3. Add definitions IMMEDIATELY after paragraph
4. Never accumulate definitions for later

## 4. Workflow

### Before Starting
```bash
# Catalog existing footnotes
python /Users/VJ/GitHub/MLSysBook/scripts/catalog_footnotes.py

# Read context
cat .claude/footnote_context.md

# Check KNOWLEDGE_MAP for chapter coverage
cat .claude/docs/shared/KNOWLEDGE_MAP.md
```

### During Work
1. Create branch: `footnote/chapter-name`
2. Add footnotes following two-phase approach
3. Verify with catalog script
4. DO NOT stage or commit (per agent policy)

### Format Standards
- ID: `[^fn-descriptive-name]` (lowercase, hyphens)
- Definition: `[^fn-name]: **Bold Term**: Explanation. Interesting fact.`

## 5. Chapter-Specific Guidelines

Consult KNOWLEDGE_MAP for what each chapter teaches. Key principle:
- **Historical mentions** → OK anywhere
- **Technical explanations** → Only in designated chapter

Example:
- Mentioning "AlexNet breakthrough in 2012" → OK anywhere
- Explaining "how convolutions work" → Only in DNN Architectures chapter

## 6. Target Quantities
- 20-30 high-value footnotes per chapter
- Balance across sections
- Quality over quantity

## 7. Verification
After adding footnotes:
1. Run catalog script
2. Check for undefined references
3. Verify no duplicate terms
4. Ensure proper placement (definitions near references)