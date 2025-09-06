# Claude Code Working Rules

This document defines how Claude Code should work with this repository to maintain clean, organized development.

## Branch Management Rules

### 1. Branch Creation
- **One task, one branch**: Create a new branch for each distinct task
- **Descriptive naming**: Use clear, descriptive branch names following conventions
- **Start from correct base**: Always branch from `dev` unless fixing a specific branch

### 2. Branch Naming Conventions

#### Task-Based Branches
- `feat/` - New features or capabilities
- `fix/` - Bug fixes or corrections
- `refactor/` - Code reorganization without changing functionality
- `docs/` - Documentation updates
- `test/` - Test additions or modifications
- `cleanup/` - Removing unused files or organizing structure

#### Agent-Created Branches
When agents create branches, prefix with agent name:
- `reviewer/` - Created by reviewer agent
- `editor/` - Created by editor agent  
- `footnote/` - Created by footnote agent
- `stylist/` - Created by stylist agent

### 3. Commit Organization
- **Atomic commits**: Each commit should represent one logical change
- **Related changes only**: Only commit files related to the current task
- **Review before committing**: Always check `git diff` before staging
- **Selective staging**: Use `git add <specific-files>` not `git add -A` when mixing changes

### 4. Merging Rules
- **Always use --no-ff**: Preserve merge history with `git merge --no-ff branch-name`
- **Merge to appropriate target**: Usually `dev`, never directly to `main`
- **Clean up after merge**: Delete feature branches after successful merge
- **One feature at a time**: Don't mix unrelated changes in a single merge

### 5. Working Practices

#### When Starting Work
1. Check current branch: `git branch`
2. Ensure working directory is clean: `git status`
3. Create appropriate branch for the task
4. Make changes related ONLY to that task

#### When Changes Span Multiple Concerns
If you realize changes involve multiple unrelated tasks:
1. STOP and assess what you've changed
2. Create separate branches for each concern
3. Use `git stash` to temporarily save changes
4. Apply relevant changes to each branch separately
5. Commit and merge each branch independently

#### Example Workflow
```bash
# Working on agent updates
git checkout -b refactor/agent-documentation
# ... make agent-related changes ...
git add .claude/agents/*.md
git commit -m "refactor: update agent documentation structure"

# Realize you also need to fix footnotes
git checkout dev
git checkout -b fix/footnote-references  
# ... make footnote fixes ...
git add quarto/contents/core/introduction/introduction.qmd
git commit -m "fix: update footnote references to use @sec- format"

# Merge each separately
git checkout dev
git merge --no-ff refactor/agent-documentation
git merge --no-ff fix/footnote-references
```

## File Organization Rules

### Temporary Files
- Use `.claude/_reviews/` for temporary review files
- Never commit temporary working files
- Clean up after task completion

### Documentation Structure
- Shared docs in `.claude/docs/shared/`
- Agent-specific docs in `.claude/docs/agents/`
- Keep README files updated when adding new documentation

## Agent-Specific Rules

### Reviewer Agent
- Creates branches with `reviewer/` prefix
- Does NOT add footnotes (only identifies needs)
- Outputs to `.claude/_reviews/`

### Editor Agent  
- Creates branches with `editor/` prefix
- Does NOT add footnotes (only text improvements)
- Works from reviewer reports

### Footnote Agent
- Creates branches with `footnote/` prefix
- Has full authority over footnotes (add/modify/remove)
- Uses `@sec-` format for chapter references

### Stylist Agent
- Creates branches with `stylist/` prefix
- Focuses on tone and consistency
- Removes AI/LLM writing patterns

## Quality Checks

Before committing:
1. Run relevant linters/formatters if available
2. Ensure no debug code or TODOs are left
3. Verify changes match commit message
4. Check that tests pass if applicable

Before merging:
1. Ensure branch is up to date with target
2. Resolve any conflicts properly
3. Verify feature is complete
4. Clean up any temporary files

## Important Reminders

- **Never force push** to shared branches
- **Always pull before starting** new work
- **Communicate through commits** with clear messages
- **Keep main and dev stable** - test before merging
- **Document significant changes** in appropriate places