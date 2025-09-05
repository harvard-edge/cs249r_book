Advanced textbook review system using Claude's Task subagent infrastructure.

This command orchestrates multiple review passes using ONLY Claude's native capabilities - no external code needed. Everything runs through Task subagents with intelligent coordination.

Usage: `/review introduction.qmd`

## Review Architecture (Pure Claude)

### Phase 1: Progressive Student Validation
Run students SEQUENTIALLY, passing learned knowledge forward:
```
CS Junior (OS/Architecture background) → validates ML concepts introduced clearly
     ↓ (pass ML knowledge learned)
CS Senior (+ Some ML exposure) → validates systems integration makes sense
     ↓ (pass cumulative knowledge)  
Masters Student → checks production readiness
```

### Phase 2: Parallel Expert Review
Run all experts SIMULTANEOUSLY for efficiency:
```
[Systems Expert] [Data Expert] [Security Expert] [Platform Expert]
        ↓              ↓              ↓                ↓
    All reviewing the same content in parallel
        ↓              ↓              ↓                ↓
    [Consolidated Expert Feedback]
```

### Phase 3: Pedagogical Review
Single comprehensive teaching effectiveness review

### Phase 4: Consensus Analysis & Application
Main Claude orchestrator finds patterns and applies changes

## How It Works (Claude Native)

When you run `/review chapter.qmd`, Claude will:

1. **Read and chunk** the chapter using built-in file operations
2. **Launch progressive student reviews** using Task subagents:
   - Each student Task gets previous student's "learned knowledge"
   - Validates content builds appropriately
3. **Launch parallel expert reviews** using Task subagents:
   - All experts review simultaneously
   - Each focuses on their domain
4. **Analyze consensus** using Claude's reasoning
5. **Apply improvements** using Edit operations

## Student Progression Tracking

Each student validates from their perspective:
- **CS Junior**: Has OS, architecture, compilers background. NEW to ML.
- **CS Senior**: Has CS fundamentals + basic ML. Learning deployment.
- **Masters**: Has both CS and ML. Focusing on production systems.

Next student receives previous knowledge and validates building.

## Expert Domain Focus

Each expert Task returns:
```json
{
  "domain": "systems|data|security|platform",
  "critical_issues": ["must fix"],
  "recommendations": ["should consider"],
  "best_practices": ["industry standards missing"]
}
```

## Consensus Rules

Claude analyzes all feedback and applies:
- **Critical**: Any expert flags as critical OR 3+ students confused
- **High**: 2+ experts recommend OR 2+ students struggle
- **Medium**: Single expert OR student suggestion
- **Low**: Nice to have improvements

## Pure Claude Implementation

NO Python files needed! Everything happens through:
- **Task subagents**: All review perspectives
- **Edit tool**: Apply improvements
- **Bash tool**: Git operations
- **Claude's reasoning**: Consensus analysis

## Advantages of Claude-Native Approach

1. **No code maintenance**: System evolves with Claude
2. **Natural language config**: Just tell Claude what you want
3. **Adaptive reasoning**: Claude adjusts approach based on content
4. **Integrated workflow**: Everything in Claude Code interface
5. **Self-documenting**: Claude explains what it's doing

## Student Prerequisites (Important!)

Our students HAVE:
- Operating systems knowledge
- Computer architecture understanding  
- Basic compilers/systems programming
- Data structures & algorithms

They DON'T necessarily have:
- Deep learning theory (we teach this)
- ML deployment experience
- Production systems knowledge
- Cloud infrastructure expertise

Claude will adjust the review accordingly!

## Output

1. **Direct improvements** applied to file
2. **Review report** with full analysis
3. **Git branch** for experimentation

All orchestrated by Claude, no external code needed!