Progressive textbook improvement using a two-agent pipeline: reviewer analyzes, editor implements.

This command ensures improvements never reference concepts that haven't been introduced yet. Each chapter can only use terminology and concepts from chapters that come before it.

Usage: `/improve chapter.qmd`

## Two-Agent Pipeline

### Agent 1: Reviewer
- Analyzes chapter for forward references
- Provides multi-perspective feedback
- Creates detailed review report
- Suggests specific improvements

### Agent 2: Editor
- Receives review report
- Implements approved changes
- Ensures clean, minimal edits
- Maintains academic quality

## Critical Constraint: Progressive Knowledge Only

### ⚠️ FORBIDDEN: Forward References
- **Never use terms from future chapters**
- **Never assume knowledge not yet taught**
- **Never reference concepts that come later**

### ✅ ALLOWED: Previous Knowledge
- **Only use terms already defined**
- **Only reference previous chapters**
- **Build on established foundations**

## Chapter Knowledge Map

The system uses `.claude/KNOWLEDGE_MAP.md` which contains:
- Complete concept progression for all 20 chapters
- What's introduced in each chapter
- Common violations and safe alternatives
- Progressive building examples

## Review Process

### Phase 1: Analysis (Reviewer Agent)
1. Loads KNOWLEDGE_MAP.md to understand boundaries
2. Identifies chapter position and available concepts
3. Scans for forward references line-by-line
4. Gathers multi-perspective feedback:
   - CS Junior (new to ML)
   - CS Senior (some ML exposure)
   - Early Career Engineer
   - Platform Architect
   - MLOps Engineer
   - Data Engineer
   - Professor/Educator
5. Produces detailed review report with consensus scoring

### Phase 2: Implementation (Editor Agent)
1. Parses review report for critical issues
2. Implements forward reference fixes first
3. Addresses high-priority clarity issues
4. Makes minimal, surgical edits
5. Preserves TikZ, tables, equations, Purpose sections
6. Maintains clean diffs without comments

## Example Progressive Improvements

### ❌ BAD (Forward Reference):
Chapter 3: "Neural networks can be optimized through quantization and pruning"
→ Quantization/pruning not introduced until Chapter 10!

### ✅ GOOD (Progressive):
Chapter 3: "Neural networks can be made more efficient through techniques we'll explore in later chapters"

### ❌ BAD (Undefined Term):
Chapter 2: "Edge devices often use GPUs for acceleration"
→ GPUs haven't been defined yet!

### ✅ GOOD (Using Known Terms):
Chapter 2: "Edge devices often use specialized hardware for faster processing"

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

## Implementation

When running `/improve dl_primer.qmd`:

1. **Reviewer agent** activated first:
   - Reads `.claude/KNOWLEDGE_MAP.md`
   - Identifies this is Chapter 3
   - Lists allowed concepts from Chapters 1-2
   - Lists forbidden concepts from Chapters 4-20
   - Produces comprehensive review report

2. **Editor agent** activated second:
   - Receives review report
   - Implements all critical fixes (4+ reviewer consensus)
   - Addresses high-priority issues (3+ reviewers)
   - Makes clean, minimal edits
   - Preserves protected content

## Benefits

- **True progressive learning** - Students never encounter undefined terms
- **Clear knowledge building** - Each chapter adds specific concepts
- **No confusion** - Everything is defined before use
- **Proper pedagogy** - Concepts introduced in optimal order
- **Clean implementation** - Separation of analysis and editing
- **Quality assurance** - Multi-perspective validation

## Output

The command produces:
1. Detailed review report showing all issues found
2. Clean edits to the chapter file
3. Summary of changes made

This ensures the textbook truly builds knowledge progressively, never assuming students know something that hasn't been taught yet.