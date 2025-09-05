Progressive textbook improvement that ONLY uses knowledge from previous chapters.

This command ensures improvements never reference concepts that haven't been introduced yet. Each chapter can only use terminology and concepts from chapters that come before it.

Usage: `/improve chapter.qmd`

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

```
Chapter 1 (Introduction) - Can use:
- Basic CS concepts (systems, deployment, infrastructure)
- No ML-specific terms yet

Chapter 2 (ML Systems) - Can use:
- Everything from Ch 1
- Deployment tiers (Cloud, Edge, Mobile, TinyML)
- Resource constraints (memory, compute, power)

Chapter 3 (DL Primer) - Can use:
- Everything from Ch 1-2
- Neural networks, neurons, weights, connections
- Forward/backward propagation
- Training vs inference

Chapter 4 (DNN Architectures) - Can use:
- Everything from Ch 1-3
- MLPs, CNNs, RNNs, Transformers (introduced here)
- Architectural patterns

Chapter 10 (Optimizations) - First mentions:
- Quantization
- Pruning
- Model compression
- Knowledge distillation
```

## Review Process

### Phase 1: Knowledge Inventory
Before reviewing any chapter, Claude:
1. Lists all concepts introduced in PREVIOUS chapters
2. Identifies NEW concepts in CURRENT chapter
3. Flags any forward references to remove

### Phase 2: Progressive Review
Reviewers are given strict knowledge boundaries:
```
"You can ONLY use these concepts: [list from previous chapters]
You CANNOT use these terms: [list from future chapters]"
```

### Phase 3: Validation
Every improvement is checked:
- Does it use undefined terms? → Remove
- Does it reference future concepts? → Rewrite
- Does it build on established knowledge? → Keep

## Example Progressive Improvements

### ❌ BAD (Forward Reference):
Chapter 3: "Neural networks can be optimized through quantization and pruning"
→ Quantization/pruning not introduced until Chapter 10!

### ✅ GOOD (Progressive):
Chapter 3: "Neural networks can be made smaller and faster through techniques we'll explore in later chapters"

### ❌ BAD (Undefined Term):
Chapter 2: "Edge devices often use NPUs for acceleration"
→ NPUs haven't been defined yet!

### ✅ GOOD (Using Known Terms):
Chapter 2: "Edge devices often use specialized hardware for faster processing"

## Reviewer Constraints

All reviewers receive this critical instruction:
```
CRITICAL CONSTRAINT: You are reviewing Chapter N.
You can ONLY use terminology from Chapters 1 through N-1.
You CANNOT use any terms that will be introduced in Chapter N+1 or later.

Known concepts so far: [explicit list]
Forbidden future concepts: [explicit list]
```

## Implementation

When running `/improve dl_primer.qmd`:

1. The textbook-editor agent FIRST reads `.claude/KNOWLEDGE_MAP.md`
2. Agent identifies chapter position (Chapter 3)
3. Agent extracts EXACT allowed concepts (Chapters 1-2)
4. Agent extracts EXACT forbidden terms (from Chapters 4+)
5. Reviewers operate strictly within these boundaries
6. Every edit is checked against the knowledge map
7. Improvements use only established concepts

## Benefits

- **True progressive learning** - Students never encounter undefined terms
- **Clear knowledge building** - Each chapter adds specific concepts
- **No confusion** - Everything is defined before use
- **Proper pedagogy** - Concepts introduced in optimal order

This ensures the textbook truly builds knowledge progressively, never assuming students know something that hasn't been taught yet.