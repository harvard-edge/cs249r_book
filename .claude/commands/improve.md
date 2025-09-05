Progressive textbook improvement using specialized subagents for review and editing.

This command ensures improvements never reference concepts that haven't been introduced yet. Each chapter can only use terminology and concepts from chapters that come before it.

Usage: `/improve chapter.qmd`

## Subagent Pipeline

This command orchestrates two specialized subagents from `.claude/agents/`:

### **Reviewer Subagent**
- **Expertise**: Multi-perspective analysis with 7 different viewpoints
- **Task**: Analyze chapter for forward references, clarity issues, and pedagogical problems
- **Output**: Structured YAML report with exact line locations and suggested fixes
- **Perspectives Covered**:
  - CS Junior (Systems Track) - OS/architecture background, new to ML
  - CS Junior (AI Track) - ML theory knowledge, needs systems context  
  - Industry New Grad - Practical coding experience, mixed theory
  - Career Transition - Smart professional, minimal tech background
  - Graduate Student - Deep theory, needs practical applications
  - Industry Practitioner - Real-world experience, current best practices
  - Professor/Educator - Pedagogical effectiveness and teaching quality

### **Editor Subagent**
- **Expertise**: Precise implementation of improvements using MultiEdit
- **Task**: Parse YAML review report and implement surgical edits
- **Output**: Clean chapter edits with preserved academic tone
- **Capabilities**: Forward reference fixes, footnote additions, clarity improvements

## Critical Constraint: Progressive Knowledge Only

### ⚠️ FORBIDDEN: Forward References
- **Never use terms from future chapters**
- **Never assume knowledge not yet taught** 
- **Never reference concepts that come later**

### ✅ ALLOWED: Previous Knowledge
- **Only use terms already defined**
- **Only reference previous chapters**
- **Build on established foundations**

## Knowledge Foundation

The system uses `.claude/KNOWLEDGE_MAP.md` which contains:
- Complete concept progression for all 20 chapters
- What's introduced in each chapter
- Common violations and safe alternatives
- Progressive building examples

## Subagent Workflow

### Phase 1: Multi-Perspective Review
The reviewer subagent:
1. Reads KNOWLEDGE_MAP.md to understand chapter boundaries
2. Identifies available concepts (previous chapters) vs forbidden (future chapters)
3. Reviews chapter from all 7 perspectives simultaneously
4. Scans every line for forward references
5. Generates structured YAML report with exact fixes

### Phase 2: Precise Implementation
The editor subagent:
1. Parses the YAML review report
2. Prioritizes critical issues (forward references first)
3. Implements fixes using exact line numbers and text matching
4. Uses MultiEdit for surgical precision
5. Adds footnotes for brief future concept references
6. Preserves protected content (TikZ, tables, equations, Purpose sections)

## Example Forward Reference Fixes

### ❌ BAD (Forward Reference):
Chapter 3: "Neural networks can be optimized through quantization and pruning"
→ Quantization/pruning not introduced until Chapter 10!

### ✅ GOOD (Progressive with Footnote):
Chapter 3: "Neural networks can be optimized through various techniques[^ch3-opt]"
[^ch3-opt]: Specific methods like quantization and pruning are covered in Chapter 10.

### ❌ BAD (Undefined Term):
Chapter 2: "Edge devices often use GPUs for acceleration"
→ GPUs haven't been defined yet!

### ✅ GOOD (Using Known Terms):
Chapter 2: "Edge devices often use specialized hardware for acceleration"

## Common Replacements

| Forbidden Term | First Introduced | Safe Alternative |
|----------------|------------------|------------------|
| Neural networks | Chapter 3 | "machine learning models" |
| CNNs, RNNs | Chapter 4 | "specialized architectures" |
| Quantization | Chapter 10 | "optimization techniques" |
| GPUs, TPUs | Chapter 11 | "specialized hardware" |
| MLOps | Chapter 13 | "operational practices" |
| Federated Learning | Chapter 14 | "distributed approaches" |
| Differential Privacy | Chapter 16 | "privacy techniques" |

## Multi-Perspective Benefits

The reviewer subagent ensures the chapter works for:
- **Systems students** learning ML for the first time
- **AI students** gaining practical systems knowledge
- **Industry professionals** transitioning between domains
- **Career changers** building foundational understanding
- **Educators** teaching these concepts effectively

## Subagent Capabilities

### Reviewer Features:
- Loads complete knowledge progression automatically
- Flags every forward reference with exact line numbers
- Provides consensus scoring across 7 perspectives
- Suggests specific fixes (replacement vs footnote vs insertion)
- Preserves protected content during analysis

### Editor Features:  
- Parses structured YAML reports automatically
- Implements exact text matching for surgical edits
- Batches changes using MultiEdit for efficiency
- Manages footnote IDs and placement
- Validates all fixes were applied correctly

## Quality Assurance

- **Forward Reference Elimination**: 100% detection and fixing
- **Multi-Background Accessibility**: Content works for diverse readers
- **Academic Quality**: Maintains professional tone and structure
- **Progressive Learning**: Each chapter builds on previous knowledge only
- **Protected Content**: TikZ, tables, equations never modified

## Output

The command produces:
1. **Detailed Review Report**: Issues found by all 7 perspectives with consensus scoring
2. **Clean Chapter Edits**: Precise improvements with surgical accuracy
3. **Validation Summary**: Confirmation all critical issues resolved

This ensures the textbook builds knowledge progressively, serving readers from diverse backgrounds while never assuming knowledge that hasn't been taught yet.