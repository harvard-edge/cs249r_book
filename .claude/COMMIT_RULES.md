# Claude Commit Rules - MUST FOLLOW

## CRITICAL: Separation of Concerns

### Rule 1: NEVER Mix System and Content Changes

**ALWAYS commit separately:**
1. **System/Command commits** - Changes to `.claude/` directory
2. **Content commits** - Changes to `.qmd` files
3. **Documentation commits** - Changes to README, docs
4. **Configuration commits** - Changes to settings, configs

### Rule 2: Commit Types and Patterns

#### System/Command Commits
```
feat: Add [command/system feature]
- Files: .claude/commands/*.md, .claude/*.md
- NEVER include: *.qmd files
```

#### Content Improvement Commits  
```
fix: [Chapter name] - [specific improvement]
- Files: quarto/contents/**/*.qmd
- NEVER include: .claude/* files
```

#### Mixed Changes Workflow
When you have both system and content changes:
```bash
# Step 1: Commit system changes
git add .claude/
git commit -m "feat: Add review system feature"

# Step 2: Commit content changes
git add quarto/
git commit -m "fix: Chapter improvements based on review"

# NEVER: git add -A with mixed changes!
```

### Rule 3: Commit Message Structure

#### For System Changes:
```
feat|fix|docs: [Component] - [What changed]

[Why it was needed]

[What it enables]

🤖 Generated with [Claude Code]
Co-Authored-By: Claude
```

#### For Content Changes:
```
fix|improve: [Chapter Name] - [What was fixed]

[Specific issues addressed]
- Removed: [what was removed]
- Added: [what was added]
- Fixed: [what was corrected]

🤖 Generated with [Claude Code]
Co-Authored-By: Claude
```

### Rule 4: Pre-Commit Checklist

Before EVERY commit, verify:
- [ ] Am I mixing system and content changes?
- [ ] Are .claude/ files separate from .qmd files?
- [ ] Is my commit message clear about what's included?
- [ ] Have I used git status to verify what's staged?

### Rule 5: Staging Commands

**GOOD Patterns:**
```bash
# System only
git add .claude/
git add .claude/commands/

# Content only
git add quarto/contents/
git add *.qmd

# Specific files
git add .claude/commands/improve.md
git add quarto/contents/core/introduction/introduction.qmd
```

**BAD Patterns:**
```bash
# NEVER use these with mixed changes
git add -A
git add .
git commit -a
```

### Rule 6: Review Branch Strategy

When doing chapter reviews:
```
1. Create feature branch for system: feature/review-system-update
2. Create fix branch for content: fix/chapter-improvements
3. Merge system first, then content
```

### Rule 7: Validation Before Push

```bash
# Check your commits are properly separated
git log --oneline -5

# Should see clear separation:
# 23e6103 fix: Remove forward references from Introduction
# 125b039 feat: Add progressive review system commands
# NOT: "Updated review system and fixed chapters" (mixed!)
```

## Examples of Proper Separation

### Example 1: Adding New Review Feature
```bash
# First commit - system
git add .claude/commands/new-reviewer.md
git commit -m "feat: Add professor reviewer perspective"

# Second commit - applying it
git add quarto/contents/core/*/
git commit -m "fix: Apply professor review feedback to chapters"
```

### Example 2: Fixing Review Bug + Chapter Issues
```bash
# First commit - fix the bug
git add .claude/commands/improve.md
git commit -m "fix: Correct reviewer knowledge boundaries"

# Second commit - fix content
git add quarto/contents/core/introduction/introduction.qmd
git commit -m "fix: Introduction - Remove undefined ML terms"
```

## Enforcement

Claude MUST:
1. Always check this file before committing
2. Refuse to mix system and content in one commit
3. Split commits if user hasn't specified
4. Warn user if attempting mixed commit

## Rationale

Separating commits enables:
- Clean review in GitKraken
- Easy rollback of either system OR content
- Clear commit history
- Better collaboration
- Simpler debugging