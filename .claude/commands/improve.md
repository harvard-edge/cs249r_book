Progressive textbook improvement using Task subagents for dynamic, specialized review and editing.

This command ensures improvements never reference concepts that haven't been introduced yet. Each chapter can only use terminology and concepts from chapters that come before it.

Usage: `/improve chapter.qmd`

## Task Subagent Pipeline

This command orchestrates multiple Task subagents, each with specialized expertise:

### **Subagent 1: Knowledge Reviewer**
- **Expertise**: Progressive knowledge boundary enforcement
- **Task**: Scan for forward references, validate concept sequences
- **Output**: YAML report with exact violations and fixes

### **Subagent 2: Multi-Perspective Reviewer** 
- **Expertise**: Seven different student/professional backgrounds
- **Task**: Review chapter from diverse reader perspectives
- **Output**: Structured feedback highlighting clarity issues

### **Subagent 3: Content Editor**
- **Expertise**: Precise implementation of improvements
- **Task**: Apply fixes from review reports using MultiEdit
- **Output**: Surgically edited chapter with clean diffs

### **Subagent 4: Quality Validator**
- **Expertise**: Post-edit verification and regression detection
- **Task**: Verify all issues resolved, no new problems introduced
- **Output**: Validation report and quality metrics

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

## Dynamic Subagent Process

### Phase 1: Knowledge Analysis
```
Task("knowledge-reviewer"):
- Load KNOWLEDGE_MAP.md for chapter boundaries
- Identify available vs forbidden concepts
- Scan every line for forward references
- Generate YAML report with exact fixes
```

### Phase 2: Multi-Perspective Review
```
Task("multi-perspective-reviewer"):
- CS Undergrad (Systems Track): OS/arch background, new to ML
- CS Undergrad (AI Track): ML theory, needs systems context
- Industry New Grad: Practical coding, mixed theory
- Career Transition: Smart but minimal tech background
- Graduate Student: Deep theory, needs practical application
- Industry Practitioner: Real experience, needs cutting-edge updates
- Educator/Professor: Pedagogical effectiveness focus
```

### Phase 3: Content Implementation
```
Task("content-editor"):
- Parse YAML review reports
- Implement forward reference fixes first
- Apply clarity improvements using MultiEdit
- Add footnotes for brief future concept mentions
- Preserve TikZ, tables, equations, Purpose sections
```

### Phase 4: Quality Validation
```
Task("quality-validator"):
- Verify all reported issues fixed
- Check for new forward references introduced
- Validate protected content unchanged
- Confirm academic tone maintained
```

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

## Subagent Benefits

- **Fresh Context**: Each subagent starts with clean context
- **Specialized Expertise**: Tailored prompts for specific tasks
- **Dynamic Configuration**: No files to maintain
- **Isolated Execution**: Independent operation prevents cross-contamination
- **Scalable Design**: Easy to add new perspectives or capabilities

## Output

The command produces:
1. **Knowledge Review Report**: All forward references and violations found
2. **Multi-Perspective Feedback**: Issues identified by different backgrounds
3. **Clean Chapter Edits**: Precise fixes with surgical accuracy
4. **Quality Validation**: Confirmation all issues resolved

This ensures the textbook builds knowledge progressively, with each chapter serving readers from diverse backgrounds while never assuming knowledge that hasn't been taught yet.