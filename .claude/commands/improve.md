Review + automatic improvement application for ML Systems textbook chapters.

Runs full multi-perspective review FIRST, then applies high-consensus improvements. Creates clean diffs for GitKraken review with important constraints respected.

Usage: `/improve introduction.qmd`

## How It Works

1. **Full Review First** - Same as `/review` command
2. **Consensus Analysis** - Identifies high-agreement issues
3. **Smart Application** - Applies improvements with constraints
4. **Git Branch** - Safe experimentation environment

## Important Constraints

### ⚠️ AUTOMATIC CONSTRAINTS (Always Applied)
- **TikZ code blocks** - All `.tikz` environments preserved exactly (NEVER touch)
- **Tables** - All markdown and LaTeX tables preserved exactly
- **Mathematical equations** - LaTeX math left untouched
- **Figure references** - @fig- references maintained
- **Purpose sections** - MUST remain as SINGLE paragraph only
- **No markdown comments** - Clean diffs for GitKraken
- **Preserve formatting** - Maintain existing style

### 🎯 Auto-Applied by All Agents
Every review agent automatically:
1. Skips all TikZ code blocks completely
2. Skips all tables (markdown and LaTeX)
3. Ensures Purpose stays single paragraph
4. Preserves mathematical notation
5. Creates clean diffs without comment clutter
6. Maintains existing formatting patterns

## Consensus Thresholds

- **5+ reviewers agree**: Auto-apply (critical)
- **4 reviewers agree**: Auto-apply (high priority)
- **3 reviewers agree**: Apply if safe
- **2 or fewer**: Document only

## Review Perspectives (Optimized for ML Systems Book)

### Learning Path Validators (Sequential)
Claude launches these in order, passing insights forward:

1. **Systems CS Student**: Strong systems, NO ML → "Can I use my systems knowledge?"
2. **ML Algorithm Student**: Theory-rich, deployment-poor → "How do I productionize?"
3. **Early Career Engineer**: 1-2 years industry → "Is this real-world accurate?"

### Domain Experts (Parallel Review)
Claude launches these simultaneously for comprehensive coverage:

1. **Platform Architect**: Cloud/K8s/GPU clusters → Infrastructure design
2. **MLOps Engineer**: CI/CD, monitoring, versioning → Operational excellence
3. **Edge Systems Engineer**: Embedded, quantization → Resource-constrained deployment
4. **Data Platform Engineer**: Feature stores, pipelines → Data foundation
5. **Professor/Educator**: Curriculum design → Teachability & exercises

### How Claude Creates the Prompts

At runtime, Claude constructs Task subagent prompts like:
```
Task(
  prompt="You are a CS Junior with strong OS/architecture background 
          but have NEVER seen ML before. Review this chapter and 
          identify what's confusing for someone with your background.
          CONSTRAINTS: Skip TikZ, skip tables, keep Purpose single paragraph..."
)
```

**All agents automatically follow constraints:**
- Never touch TikZ code or tables
- Keep Purpose sections as single paragraphs
- Add inline definitions
- Create clean, mergeable improvements

## Output

1. **Scorecard** - Same as `/review` output
2. **Applied changes** - Direct file improvements
3. **Git branch** - `improve-[chapter]-[date]`
4. **Summary report** - What was changed and why

## CRITICAL: Commit Separation Rules

**NEVER mix system and content changes in one commit!**

When committing:
1. **First commit**: .claude/ changes only (system/commands)
2. **Second commit**: .qmd changes only (content improvements)
3. **See**: .claude/COMMIT_RULES.md for detailed requirements

This separation is MANDATORY for clean review and rollback.

## Example Workflow

```bash
# First, just review without changes
/review introduction.qmd  

# If scorecard looks good, apply improvements
/improve introduction.qmd

# Review changes in GitKraken
# Stage what you like, discard what you don't
```

The system ensures your textbook teaches AI engineering effectively while maintaining technical accuracy and pedagogical quality.