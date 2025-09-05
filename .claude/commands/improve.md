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

## Review Perspectives (With Built-in Constraints)

**Students (Progressive Validation):**
- CS Junior: Has systems background, learning ML
- CS Senior: Some ML knowledge, learning deployment
- Masters: Strong foundation, production focus

**Experts (Technical Accuracy):**
- Systems Engineer: Scalability and operations
- ML Practitioner: Algorithm to production
- Data Engineer: Pipeline and quality

**All agents are programmed to:**
- Never suggest changes to TikZ code
- Never modify tables (markdown or LaTeX)
- Keep Purpose sections as single paragraphs
- Add inline definitions rather than separate glossaries
- Focus on clean, mergeable improvements

## Output

1. **Scorecard** - Same as `/review` output
2. **Applied changes** - Direct file improvements
3. **Git branch** - `improve-[chapter]-[date]`
4. **Summary report** - What was changed and why

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