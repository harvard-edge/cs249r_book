Pure Claude-native textbook review using Task subagents - NO external code needed!

Reviews chapters from multiple perspectives using Claude's built-in Task subagent infrastructure. Everything runs through Claude - no Python files, no external scripts.

Usage: `/improve introduction.qmd`

## How It Works (100% Claude Native)

When you run this command, Claude will:
1. Launch student Task subagents to validate learning progression
2. Launch expert Task subagents to verify technical accuracy
3. Analyze consensus patterns using Claude's reasoning
4. Apply improvements directly to files using Edit tools

No Python code, no external files - just Claude!

## Default Reviewers (Balanced Mix)

**Students (3):**
- Senior Undergrad: Has ML basics, needs deployment guidance
- Masters Student: Cross-disciplinary, working on projects
- PhD Student: Strong theory, learning systems aspects

**Experts (3):**
- Systems Engineer: Production deployment, scaling, operations
- ML Practitioner: Notebook-to-production specialist
- Data Engineer: Pipelines, data quality, feature stores

## Consensus Application

Based on `config/settings.json`:
- **4+ reviewers agree**: Applied automatically
- **3 reviewers agree**: Applied (high priority)
- **2 reviewers agree**: Noted in report
- **1 reviewer**: Documented only

## Customizing Reviewers

### Customize Review (Natural Language)

Just tell Claude what you want:
- "Focus on student confusion points"
- "Add a mobile developer perspective"
- "Check for security issues"
- "Use only expert reviewers"

Claude will adjust the review approach dynamically - no config files to edit!

## Output

1. **Direct file improvements** (no markdown clutter)
2. **Report** (`.improve-report-[chapter].md`) with full context
3. **Git branch** for safe experimentation

Review changes in GitKraken - stage what you like, discard what you don't.

The system focuses on the textbook's mission: teaching students to build real ML systems, not just understand algorithms.