# ML Systems Book - Agent Documentation

This directory contains documentation for all agents working on the ML Systems textbook.

## Directory Structure

```
docs/
├── README.md           # This file
├── shared/            # Documents ALL agents must read
│   ├── CONTEXT.md     # Book context and philosophy
│   ├── KNOWLEDGE_MAP.md # What each chapter teaches
│   └── GIT_WORKFLOW.md # Git branching and merge strategy
└── agents/            # Agent-specific documentation
    └── FOOTNOTE_GUIDELINES.md # Specific to footnote agent
```

## Documentation Rules

### All Agents MUST Read:
1. `shared/CONTEXT.md` - Understand the book's philosophy and target audience
2. `shared/KNOWLEDGE_MAP.md` - Know what concepts are taught in each chapter
3. `shared/GIT_WORKFLOW.md` - Follow proper Git workflow

### Agent-Specific Documents:
- Agents should ONLY read their specific documentation if it exists
- The footnote agent reads `agents/FOOTNOTE_GUIDELINES.md`
- Reviewer and editor agents reference the knowledge map heavily

## Agent Documentation References

Each agent's prompt should include:

```markdown
## Required Reading
1. .claude/docs/shared/CONTEXT.md
2. .claude/docs/shared/KNOWLEDGE_MAP.md
3. .claude/docs/shared/GIT_WORKFLOW.md
[4. Any agent-specific docs if applicable]
```

## Adding New Documentation

When adding new documentation:
- **Shared docs** go in `shared/` if ALL agents need them
- **Agent-specific docs** go in `agents/` named after the agent
- Update this README with the new document's purpose