# .claude Directory - ML Systems Textbook Editorial System

## Overview
This directory contains the complete editorial workflow system for the ML Systems textbook, including AI agents, documentation, templates, and review outputs.

## Directory Structure

```
.claude/
├── agents/                          # AI Agent Definitions
│   ├── editorial/                   # Editorial workflow agents
│   │   ├── editor.md               # Master editor
│   │   ├── stylist.md              # Style consistency
│   │   ├── reviewer.md             # Chapter reviewer
│   │   ├── footnote.md             # Footnote specialist
│   │   ├── citation-validator.md   # Citation management
│   │   ├── cross-reference-optimizer.md
│   │   ├── glossary-builder.md
│   │   ├── learning-objectives.md
│   │   ├── narrative-flow-analyzer.md
│   │   ├── fact-checker.md
│   │   └── independent-review.md
│   │
│   ├── workflow/                    # Orchestration agents
│   │   ├── workflow-orchestrator.md # Main orchestrator
│   │   ├── consensus-builder.md     # Expert consensus
│   │   └── git-commit.md           # Git management
│   │
│   ├── experts/                     # Expert review agents
│   ├── classroom/                   # Student perspective agents
│   └── optional/                    # Specialized/experimental agents
│
├── docs/                           # Documentation & Guides
│   ├── shared/                     # Shared resources & standards
│   │   ├── CONTEXT.md              # Book context
│   │   ├── KNOWLEDGE_MAP.md        # Chapter knowledge map
│   │   ├── FORMATTING_GUIDELINES.md # Style standards
│   │   ├── GIT_WORKFLOW.md         # Git procedures
│   │   └── agent_commit_policy.md  # Commit standards
│   │
│   ├── agents/                     # Agent-specific documentation
│   └── experiments/                # Experimental workflows
│
├── _reviews/                       # Review Outputs (timestamped)
│   ├── editorial_YYYYMMDD_HHMM/    # Editorial workflow runs
│   ├── expert_feedback/            # Expert review runs
│   └── archived/                   # Historical/completed reviews
│
├── _progress/                      # Workflow Status Tracking
└── templates/                      # Templates for reports/workflows
```

## Quick Start

### Run Complete Editorial Workflow
```bash
Task --subagent_type workflow-orchestrator \
  --prompt "Run the polish workflow on all chapters"
```

### Individual Agent Usage
```bash
# Editorial agents
Task --subagent_type editor --prompt "Fix writing style in introduction"
Task --subagent_type stylist --prompt "Polish academic prose in dl_primer"

# Review agents  
Task --subagent_type reviewer --prompt "Review introduction for forward references"
Task --subagent_type fact-checker --prompt "Verify technical specs in training chapter"
```

## Key Files

### Primary Workflows
- [`SYSTEMATIC_WORKFLOW_ORCHESTRATOR.md`](../SYSTEMATIC_WORKFLOW_ORCHESTRATOR.md) - Integrated 5-phase editorial workflow
- [`WHO_DOES_WHAT.md`](../WHO_DOES_WHAT.md) - Complete agent responsibilities and workflow
- `docs/shared/FORMATTING_GUIDELINES.md` - Style standards
- `docs/shared/KNOWLEDGE_MAP.md` - Chapter content mapping

### Agent Categories
- **Editorial**: Core editing, style, structure, content quality (11 agents)
- **Workflow**: Orchestration, consensus building, git management (3 agents)
- **Experts**: Domain expert review perspectives (9 agents)
- **Classroom**: Student learning perspectives (8 agents)

## Workflow Types

### 1. Polish Workflow (Primary)
Complete editorial pass for content cleanup:
- **Assessment** (fact-check, review, flow analysis)
- **Implementation** (editor applies all fixes)  
- **Polish** (stylist ensures consistency)
- **Academic apparatus** (citations, cross-refs, footnotes)
- **Final polish** (glossary, learning objectives)

### 2. Expert Review Workflow
Multi-expert review and consensus building

### 3. Individual Agent Workflows
Targeted improvements with specific agents

## Quality Standards

All agents follow these principles:
- **Comprehensive passes** - thorough work every time
- **Self-correcting** - fix previous errors if found
- **Convergent** - fewer changes needed over time as book stabilizes
- **Preserve quality** - maintain good existing work
- **Academic tone** - scholarly, professional writing

## Important Rules

1. **No Auto-Commits**: Agents leave changes staged for user review
2. **No Auto-Branching**: Agents work on current branch unless explicitly requested
3. **Knowledge Map**: Always consult to avoid content duplication
4. **Reviews**: Timestamped outputs in `_reviews/`, never overwritten
5. **Figure Caption Bold**: Always preserved (intentional formatting)

## Recent Updates

- Reorganized agent structure for clarity (editorial/workflow/experts/classroom)
- Added workflow orchestration system
- Implemented convergence tracking
- Established comprehensive formatting standards
- Created clean documentation structure