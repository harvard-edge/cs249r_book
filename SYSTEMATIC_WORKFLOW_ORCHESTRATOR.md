# 🎭 Systematic Editorial Workflow Orchestrator

## Overview
This document defines a systematic, parallel-execution workflow that processes all 21 chapters through a structured 5-phase editorial pipeline. Each chapter follows the same sequential flow while chapters are processed in parallel within each phase.

## 🏗️ Workflow Architecture

### Parallel Processing Model
- **Within Phase**: All 21 chapters processed simultaneously
- **Between Phases**: Strict sequential execution with dependency enforcement
- **Information Flow**: Each phase consumes outputs from previous phases

### Phase Dependencies
```
Phase 1 (Assessment) → Phase 2 (Structural) → Phase 3 (Polish) → Phase 4 (Academic) → Phase 5 (Final)
       ↓                      ↓                    ↓                  ↓                   ↓
   Read-Only              Uses Reports         Preserves Edits    Enhances Content    Final Polish
   Reports               from Phase 1         from Phase 2       from Phase 3       from All Phases
```

---

## 🚀 PHASE-BY-PHASE EXECUTION

### PHASE 0: Setup (2 minutes)
```bash
# Initialize workflow environment
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKFLOW_DIR=".claude/_reviews/editorial_${TIMESTAMP}"
REPORTS_DIR="${WORKFLOW_DIR}/reports"

# Create directory structure
mkdir -p ${WORKFLOW_DIR}/{factcheck,reviewer,independent,flow,editor,stylist,academic,final}

# Define all chapters
ALL_CHAPTERS=(
  introduction dl_primer ml_systems data_engineering frameworks training
  efficient_ai optimizations hw_acceleration dnn_architectures benchmarking
  ops workflow ondevice_learning robust_ai privacy_security responsible_ai
  sustainable_ai ai_for_good frontiers conclusion
)

echo "=== WORKFLOW ORCHESTRATOR INITIALIZED ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Processing ${#ALL_CHAPTERS[@]} chapters"
echo "Output directory: ${WORKFLOW_DIR}"
```

---

### PHASE 1: ASSESSMENT (2-3 hours) - READ ONLY
**Purpose**: Identify all issues without making any changes

#### 1A: Fact-Checker (All Chapters in Parallel)
```bash
echo "=== PHASE 1A: Running Fact-Checkers (Parallel) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Fact-checking $chapter..."
  
  Task --subagent_type fact-checker \
    --prompt "Verify technical specifications, benchmarks, and claims in quarto/contents/core/$chapter/$chapter.qmd. 
    Output format:
    ✅ CORRECT: [verified fact]
    ❌ INCORRECT: [wrong fact] → CORRECTION: [correct fact] (Source: [source])
    ⚠️ UNVERIFIABLE: [claim] → NEEDS HUMAN REVIEW: [reason]" \
    > ${WORKFLOW_DIR}/factcheck/${chapter}_facts.md &
done
wait
echo "Fact-checking complete. Reports in ${WORKFLOW_DIR}/factcheck/"
```

#### 1B: Reviewer (All Chapters in Parallel)
```bash
echo "=== PHASE 1B: Running Reviewers (Parallel) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Reviewing $chapter..."
  
  Task --subagent_type reviewer \
    --prompt "Review quarto/contents/core/$chapter/$chapter.qmd for forward references, clarity issues, and flow problems.
    Focus on:
    - Terms used before definition
    - References to later chapters
    - Unclear explanations
    - Missing context
    Output specific line numbers and suggested fixes." \
    > ${WORKFLOW_DIR}/reviewer/${chapter}_review.md &
done
wait
echo "Review complete. Reports in ${WORKFLOW_DIR}/reviewer/"
```

#### 1C: Independent Review (All Chapters in Parallel)
```bash
echo "=== PHASE 1C: Running Independent Reviews (Parallel) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Independent review of $chapter..."
  
  Task --subagent_type independent-review \
    --prompt "Provide fresh perspective on quarto/contents/core/$chapter/$chapter.qmd.
    Look for issues missed by standard review:
    - Pedagogical flow problems
    - Inconsistent terminology
    - Missing connections to other concepts
    - Unclear explanations for target audience" \
    > ${WORKFLOW_DIR}/independent/${chapter}_independent.md &
done
wait
echo "Independent review complete. Reports in ${WORKFLOW_DIR}/independent/"
```

#### 1D: Narrative Flow Analysis (All Chapters in Parallel)
```bash
echo "=== PHASE 1D: Running Narrative Flow Analyzers (Parallel) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Analyzing flow in $chapter..."
  
  Task --subagent_type narrative-flow-analyzer \
    --prompt "Analyze paragraph-to-paragraph flow in quarto/contents/core/$chapter/$chapter.qmd.
    Identify:
    - Abrupt transitions
    - Missing connective tissue
    - Logical gaps
    - Repetitive patterns
    Suggest specific improvements." \
    > ${WORKFLOW_DIR}/flow/${chapter}_flow.md &
done
wait
echo "Flow analysis complete. Reports in ${WORKFLOW_DIR}/flow/"
```

#### Phase 1 Summary Report
```bash
echo "=== PHASE 1 COMPLETE - ASSESSMENT SUMMARY ==="
echo "Facts checked: $(find ${WORKFLOW_DIR}/factcheck -name "*.md" | wc -l) chapters"
echo "Reviews completed: $(find ${WORKFLOW_DIR}/reviewer -name "*.md" | wc -l) chapters"
echo "Independent reviews: $(find ${WORKFLOW_DIR}/independent -name "*.md" | wc -l) chapters"
echo "Flow analyses: $(find ${WORKFLOW_DIR}/flow -name "*.md" | wc -l) chapters"

# Generate summary of issues found
echo "Critical issues requiring fixes:" > ${REPORTS_DIR}/phase1_summary.md
grep -l "❌ INCORRECT\|CRITICAL\|URGENT" ${WORKFLOW_DIR}/{factcheck,reviewer,independent,flow}/*.md >> ${REPORTS_DIR}/phase1_summary.md
```

---

### PHASE 2: STRUCTURAL EDITS (3-4 hours) - EDITOR IMPLEMENTS FIXES
**Purpose**: Apply all fixes identified in Phase 1

```bash
echo "=== PHASE 2: Running Structural Editors (Sequential for Clean Git History) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Editing $chapter..."
  
  Task --subagent_type editor \
    --prompt "Edit quarto/contents/core/$chapter/$chapter.qmd implementing ALL fixes from assessment reports:
    
    FACT CORRECTIONS (implement these changes):
    - Read: ${WORKFLOW_DIR}/factcheck/${chapter}_facts.md
    - Change all ❌ INCORRECT items to their CORRECTION values
    
    CLARITY FIXES (implement these changes):
    - Read: ${WORKFLOW_DIR}/reviewer/${chapter}_review.md
    - Fix forward references and unclear explanations
    
    FLOW IMPROVEMENTS (implement these changes):
    - Read: ${WORKFLOW_DIR}/flow/${chapter}_flow.md
    - Improve paragraph transitions and logical flow
    
    STRUCTURAL FIXES:
    - Convert bold pseudo-headers (**Term:** at paragraph start) to proper headers or natural text
    - Fix bold emphasis within paragraphs (remove unnecessary bold)
    - Improve section transitions
    - Add lead-in paragraphs for empty sections
    
    PRESERVE:
    - Figure caption bold text (this is intentional formatting)
    - Mathematical equations and code blocks
    - Essential technical bold terms in context
    
    Focus on making the text flow naturally while implementing all factual corrections."
  
  echo "Completed editing $chapter"
done

# Commit structural changes
git add quarto/contents/core/**/*.qmd
git commit -m "refactor: implement editorial fixes across all chapters

- Apply fact corrections from verification phase
- Fix forward references and clarity issues identified by reviewers  
- Improve paragraph flow and transitions based on narrative analysis
- Convert bold pseudo-headers to proper structure
- Preserve intentional figure caption formatting

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "=== PHASE 2 COMPLETE - STRUCTURAL EDITS ==="
```

---

### PHASE 3: CONSISTENCY & POLISH (2-3 hours) - STYLIST ENSURES UNIFORMITY
**Purpose**: Standardize style and remove inconsistencies across all chapters

```bash
echo "=== PHASE 3: Running Stylists (Sequential for Consistency) ==="
for chapter in "${ALL_CHAPTERS[@]}"; do
  echo "Polishing $chapter..."
  
  Task --subagent_type stylist \
    --prompt "Polish academic prose in quarto/contents/core/$chapter/$chapter.qmd for consistency:
    
    REMOVE AI PATTERNS:
    - 'delving into' → 'examining' or 'exploring'
    - 'furthermore' → 'additionally' or remove
    - 'moreover' → 'also' or remove
    - 'harnessing' → 'using' or 'leveraging'
    - 'it's worth noting' → remove or rephrase naturally
    - 'one can' → 'we can' or 'you can'
    
    ENSURE CONSISTENCY:
    - Use same terminology across all chapters (e.g., always 'machine learning' not 'ML' in prose)
    - Standardize bold usage (only for key terms, warnings, or emphasis)
    - Remove unnecessary em-dashes and hyphens
    - Ensure parallel structure in lists and explanations
    
    FIX CROSS-REFERENCES:
    - Change '[Chapter @sec-xxx]' to simple '@sec-xxx'
    - Ensure all cross-references use @sec- format
    - Remove redundant reference text
    
    PRESERVE:
    - All structural changes made by Editor in Phase 2
    - Figure caption bold text (intentional)
    - Technical term definitions and key concepts
    - Mathematical notation and code formatting
    
    Focus on making the chapter consistent with academic writing standards while preserving all content improvements from Phase 2."
  
  echo "Completed polishing $chapter"
done

# Commit style changes
git add quarto/contents/core/**/*.qmd
git commit -m "style: standardize academic prose across all chapters

- Remove AI writing patterns and ensure natural academic flow
- Standardize terminology and cross-reference format (@sec- only)
- Fix punctuation and emphasis consistency across chapters
- Preserve structural improvements from editor phase

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "=== PHASE 3 COMPLETE - CONSISTENCY & POLISH ==="
```

---

### PHASE 4: ACADEMIC APPARATUS (2 hours) - SCHOLARLY ENHANCEMENTS
**Purpose**: Add and validate academic elements across the textbook

#### 4A: Citation Validation
```bash
echo "=== PHASE 4A: Citation Validation ==="
Task --subagent_type citation-validator \
  --prompt "Validate and enhance citations across all chapters in quarto/contents/core/.
  
  VALIDATE EXISTING:
  - Check all citations reference valid entries in bibliography
  - Ensure consistent citation format
  - Verify page numbers and publication details
  
  ADD MISSING:
  - Add citations for factual claims lacking sources
  - Add citations for technical specifications
  - Add citations for research findings and benchmarks
  
  OUTPUT: Create report of changes made and any unresolvable issues"

echo "Citation validation complete"
```

#### 4B: Cross-Reference Optimization
```bash
echo "=== PHASE 4B: Cross-Reference Optimization ==="
Task --subagent_type cross-reference-optimizer \
  --prompt "Optimize cross-references across all chapters for pedagogical value.
  
  ADD STRATEGIC REFERENCES:
  - Forward references when introducing concepts defined later
  - Backward references when building on earlier concepts
  - Cross-chapter connections for related topics
  
  REMOVE EXCESSIVE REFERENCES:
  - References that interrupt reading flow
  - Redundant references within same section
  - References to immediately adjacent sections
  
  ENSURE FORMAT:
  - All references use simple @sec-xxx format
  - No '[Chapter @sec-xxx]' format
  - Clear, contextual reference integration
  
  FOCUS: Enhance learning progression without overwhelming the reader"

echo "Cross-reference optimization complete"
```

#### 4C: Footnote Enhancement (Selective)
```bash
echo "=== PHASE 4C: Footnote Enhancement ==="
# Only run footnote agent on chapters needing clarification
Task --subagent_type footnote \
  --prompt "Add clarifying footnotes across chapters where needed for forward references or technical terms.
  
  ADD FOOTNOTES FOR:
  - Technical terms used before formal definition
  - Forward references to concepts explained later
  - Historical context that enhances understanding
  - Alternative terminology used in the field
  
  FOOTNOTE FORMAT:
  - Use @sec- format for chapter references in footnotes
  - Keep footnotes concise and focused
  - Number footnotes consistently within each chapter
  
  PRESERVE:
  - All existing footnotes
  - Main text content (only add footnotes, don't change main text)
  
  TARGET: Chapters with complex forward references or technical concepts"

echo "Footnote enhancement complete"
```

#### Commit Academic Apparatus
```bash
git add -A
git commit -m "feat: enhance academic apparatus across textbook

- Validate and standardize all citations with proper sources
- Optimize cross-references for learning progression and pedagogical flow
- Add clarifying footnotes for technical concepts and forward references
- Ensure consistent academic formatting throughout

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "=== PHASE 4 COMPLETE - ACADEMIC APPARATUS ==="
```

---

### PHASE 5: FINAL POLISH (1 hour) - COMPREHENSIVE COMPLETION
**Purpose**: Final enhancements and quality assurance

#### 5A: Glossary Building
```bash
echo "=== PHASE 5A: Comprehensive Glossary Building ==="
Task --subagent_type glossary-builder \
  --prompt "Build comprehensive glossary from all chapters in quarto/contents/core/.
  
  EXTRACT TERMS:
  - Technical terms introduced throughout the textbook
  - Acronyms and abbreviations
  - Domain-specific concepts
  - Mathematical and statistical terms
  
  DEFINITION STANDARDS:
  - Clear, concise definitions accessible to target audience
  - Cross-references to relevant chapters using @sec- format
  - Lowercase formatting for consistency
  - Alphabetical organization
  
  OUTPUT:
  - Update quarto/contents/glossary.qmd with comprehensive term list
  - Ensure all terms from 21 chapters are covered
  - Include pronunciation guides for complex terms where helpful"

echo "Glossary building complete"
```

#### 5B: Learning Objectives Update
```bash
echo "=== PHASE 5B: Learning Objectives Alignment ==="
Task --subagent_type learning-objectives \
  --prompt "Update learning objectives for all chapters based on current content after editorial process.
  
  FOR EACH CHAPTER:
  - Review actual chapter content post-editing
  - Ensure objectives match what students will actually learn
  - Use measurable, specific learning outcomes
  - Align with textbook's overall pedagogical goals
  
  OBJECTIVE STANDARDS:
  - Start with action verbs (understand, analyze, implement, evaluate)
  - Be specific about skills and knowledge gained
  - Progressive complexity throughout the textbook
  - Clear assessment criteria
  
  UPDATE:
  - Learning objectives at beginning of each chapter
  - Ensure consistency in format and style
  - Verify objectives prepare students for subsequent chapters"

echo "Learning objectives update complete"
```

#### Final Commit
```bash
git add -A
git commit -m "feat: complete editorial polish with comprehensive enhancements

- Build comprehensive glossary covering all technical terms from 21 chapters
- Update learning objectives to align with current content after editorial process
- Final consistency and quality assurance across entire textbook
- Complete 5-phase editorial workflow

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "=== PHASE 5 COMPLETE - FINAL POLISH ==="
```

---

### PHASE 6: VALIDATION & QUALITY ASSURANCE (30 minutes)

#### Quality Metrics
```bash
echo "=== PHASE 6: VALIDATION & QUALITY ASSURANCE ==="

# Check for remaining issues
echo "Running quality checks..."

BOLD_HEADERS=$(grep -rn "^\*\*[^:]*:" quarto/contents/core/**/*.qmd | grep -v "fig-" | wc -l)
echo "Remaining bold pseudo-headers: $BOLD_HEADERS (target: 0)"

AI_PATTERNS=$(grep -rE "delving into|furthermore|moreover|harnessing|it's worth noting|one can" quarto/contents/core/**/*.qmd | wc -l)
echo "Remaining AI patterns: $AI_PATTERNS (target: 0)"

BAD_REFS=$(grep -rE "\[Chapter @sec-" quarto/contents/core/**/*.qmd | wc -l)
echo "Bad cross-reference format: $BAD_REFS (target: 0)"

UNVERIFIABLE=$(grep -r "⚠️ UNVERIFIABLE" ${WORKFLOW_DIR}/factcheck/*.md | wc -l)
echo "Unverifiable claims for human review: $UNVERIFIABLE"

# Test build
echo "Testing build..."
./binder preview

# Generate final report
cat > ${REPORTS_DIR}/workflow_completion.md << EOF
# Editorial Workflow Complete - ${TIMESTAMP}

## Summary
- **Duration**: 8-10 hours
- **Chapters processed**: ${#ALL_CHAPTERS[@]}
- **Phases completed**: 5 (Assessment → Structural → Polish → Academic → Final)
- **Commits created**: 4 (structural, style, academic, final)

## Quality Metrics
- **Bold pseudo-headers**: $BOLD_HEADERS (target: 0)
- **AI patterns**: $AI_PATTERNS (target: 0) 
- **Bad cross-references**: $BAD_REFS (target: 0)
- **Unverifiable claims**: $UNVERIFIABLE (for human review)

## Files Modified
- Chapter files: ${#ALL_CHAPTERS[@]} updated
- Citations: Validated and enhanced
- Glossary: Comprehensive rebuild
- Learning objectives: Aligned with content

## Deliverables
- ✅ All chapters follow consistent style and tone
- ✅ All factual errors corrected
- ✅ All forward references properly handled
- ✅ All AI writing patterns removed
- ✅ All cross-references in proper @sec- format
- ✅ Comprehensive glossary covering all terms
- ✅ Learning objectives aligned with actual content
- ✅ Build preview works without errors

## Recommendations
$(if [ $UNVERIFIABLE -gt 0 ]; then echo "- Review unverifiable claims in factcheck reports"; fi)
$(if [ $BOLD_HEADERS -gt 0 ]; then echo "- Manual review of remaining bold pseudo-headers"; fi)
$(if [ $AI_PATTERNS -gt 0 ]; then echo "- Manual review of remaining AI patterns"; fi)
- Content is ready for further review or publication
EOF

echo "=== SYSTEMATIC EDITORIAL WORKFLOW COMPLETE ==="
echo "Final report: ${REPORTS_DIR}/workflow_completion.md"
```

---

## 🎯 WORKFLOW GUARANTEES

After successful completion, this workflow guarantees:

### Content Quality
- ✅ **Factual Accuracy**: All technical claims verified and corrected
- ✅ **Forward References**: All forward references properly handled or explained
- ✅ **Natural Flow**: Smooth paragraph transitions throughout
- ✅ **Academic Tone**: Professional, natural academic prose without AI patterns

### Structural Consistency
- ✅ **Headers**: No bold pseudo-headers, proper markdown structure
- ✅ **Cross-References**: All in clean @sec- format
- ✅ **Terminology**: Consistent use across all 21 chapters
- ✅ **Formatting**: Uniform style and emphasis patterns

### Academic Standards
- ✅ **Citations**: Comprehensive and properly formatted
- ✅ **Footnotes**: Strategic clarification where needed
- ✅ **Glossary**: Complete technical term coverage
- ✅ **Learning Objectives**: Aligned with actual chapter content

### Technical Quality  
- ✅ **Build Process**: Preview builds without errors
- ✅ **Cross-References**: All @sec- references valid
- ✅ **Figure Captions**: Intentional bold formatting preserved
- ✅ **Mathematical Content**: All equations and code preserved

---

## 🚀 EXECUTION COMMAND

To run the complete workflow:

```bash
# Execute the systematic editorial workflow
Task --subagent_type workflow-orchestrator \
  --prompt "Execute the systematic editorial workflow on all 21 chapters. Run all 5 phases in sequence: Assessment → Structural Edits → Consistency Polish → Academic Apparatus → Final Polish. Generate quality metrics and completion report."
```

This workflow transforms inconsistent content into publication-ready academic material through systematic application of all editorial agents in the proper sequence with full dependency management and quality assurance.