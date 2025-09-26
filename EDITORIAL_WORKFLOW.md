# 📚 ML Systems Textbook - Editorial Workflow

## Overview
Expert editorial workflow for creating the definitive ML Systems textbook - the first comprehensive treatment of this field, designed to become the foundational reference for the discipline.

## Editorial Mission
This textbook represents a landmark contribution to the field of ML Systems Engineering. Each agent in this workflow is an expert specialist working to ensure the highest quality scholarly work that will serve as the primary reference for years to come.

## The Editorial Process

**When to use**: After making content edits to ensure everything remains clean, consistent, and builds knowledge progressively.

### Step 1: Fix Technical Claims
**Goal**: Ensure technical accuracy
- **fact-checker**: Directly fixes wrong technical claims, specs, benchmarks, model parameters

### Step 2: Check Conceptual Progression
**Goal**: Ensure knowledge builds progressively without repetition
- **concept-progression**:
  - Identifies conceptual repetitions within chapters
  - Ensures each concept builds on previously established knowledge
  - Flags instances where concepts are re-explained instead of referenced
  - Verifies logical progression from basic to advanced concepts
  - Maintains awareness that multiple writing sessions may cause unintended repetition

### Step 3: Clean Structure & Flow  
**Goal**: Expert editorial refinement for landmark textbook quality
- **editor** (Expert Textbook Editor): 
  - Ensures smooth transitions between all header levels (H2→H3, H3→H4, etc.)
  - Adds introductory paragraphs where headers are back-to-back
  - Converts bold pseudo-headers to proper structure
  - Improves paragraph flow and transitions
  - Maintains awareness this is creating the definitive reference work
  - PRESERVES figure caption bold text

### Step 4: Add Academic Elements
**Goal**: Maintain scholarly apparatus befitting the field's foundational text
- **citation-validator**: Adds missing citations, fixes format
- **footnote**: Adds clarifying footnotes where needed

### Step 5: Ensure Consistent Tone
**Goal**: Unified book voice matching introduction chapter style
- **stylist**:
  - Maintains scholarly, professional tone (like introduction chapter)
  - Removes AI writing patterns ("delving into", "furthermore", "comprehensive", "harnessing")
  - Ensures natural paragraph transitions and logical flow
  - Standardizes terminology book-wide
  - Fixes cross-reference format to simple @sec- (no brackets)
  - Preserves technical precision with specific examples and metrics
  - Makes the whole book sound like one unified academic voice

## How to Execute

### For Any Chapter After Edits:

1. **Fix Technical Claims**:
```
Task fact-checker "Fix wrong technical claims in [chapter]"
```

2. **Check Conceptual Progression**:
```
Task concept-progression "Ensure progressive knowledge building in [chapter]"
```

3. **Clean Structure & Flow**:
```
Task editor "Expert editorial refinement of [chapter] for landmark textbook"
```

4. **Add Academic Elements**:
```
Task citation-validator "Add missing citations in [chapter]"
Task footnote "Add clarifying footnotes in [chapter]"
```

5. **Ensure Consistent Tone**:
```
Task stylist "Maintain unified academic tone in [chapter]"
```

**That's it!** Five expert-level steps to maintain the highest quality for this landmark textbook.

## The 21 Chapters (in order)
```
introduction, dl_primer, ml_systems, data_engineering, frameworks, 
training, efficient_ai, optimizations, hw_acceleration, dnn_architectures, 
benchmarking, ops, workflow, ondevice_learning, robust_ai, privacy_security, 
responsible_ai, sustainable_ai, ai_for_good, frontiers, conclusion
```

## Key Principles & Conflict Prevention

1. **Run after any content edits** - maintains consistency
2. **Follow the sequence** - fact-checker → editor → academic → stylist  
3. **Figure caption bold is ALWAYS preserved by ALL agents**
4. **Each agent is idempotent** - safe to re-run

### What Each Agent MUST Preserve:
- **ALL agents**: Figure caption bold text (never remove)
- **editor**: Technical corrections made by fact-checker
- **citation-validator**: Structure changes made by editor  
- **footnote**: Citations added by citation-validator
- **stylist**: All content changes made by previous agents (only changes tone/style)

### Agent Boundaries & Style Requirements:
- **fact-checker**: ONLY fixes technical claims, dates, numbers, specs (maintains academic precision)
- **concept-progression**: ONLY analyzes conceptual flow, identifies repetition and knowledge building issues
- **editor**: Expert structural refinement - transitions between ALL headers, introductory paragraphs, flow
- **citation-validator**: ONLY adds/fixes citations, maintains scholarly format
- **footnote**: Adds clarifying footnotes - connects to ML Systems perspective where natural (not forced)
- **stylist**: ONLY adjusts tone/style to match introduction chapter voice, preserves all content changes

### Chapter-Aware Considerations:
- **Technical chapters** (training, optimizations, hw_acceleration): May include equations, algorithms, performance metrics
- **Conceptual chapters** (ai_for_good, responsible_ai, sustainable_ai): Focus on applications and implications
- **All chapters**: Should maintain connection to ML Systems engineering perspective where appropriate
- **Footnotes**: Balance between technical details, historical context, and ML Systems connections

## Quick Reference

| Step | Agent | What They Do | Style Goal |
|------|-------|--------------|------------|
| 1 | fact-checker | Fix technical claims ONLY | Academic precision with specific metrics |
| 2 | **concept-progression** | Check conceptual flow | Progressive knowledge building without repetition |
| 3 | **editor** | Expert structural refinement | Smooth transitions between ALL header levels |
| 4 | citation-validator | Add/fix citations ONLY | Scholarly citation format |
| 4 | footnote | Add clarifying footnotes | Connect to ML Systems where appropriate |
| 5 | **stylist** | Unify academic voice | Match introduction's scholarly tone |

## Success Criteria
- Technical facts verified and corrected
- Clean structure with proper headers
- Consistent style across chapters
- Cross-references use @sec- format
- Proper citations and footnotes

---

**Simple, repeatable process**: After any content edits, run these 4 agents in sequence to maintain quality and consistency.